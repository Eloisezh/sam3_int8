"""Evaluate a directly-loaded multi-file SAM3 ONNX model on COCO JSON."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from .inference import SAM3ONNXInference
except ImportError:
    from inference import SAM3ONNXInference


METRICS = ["AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
           "AR_maxDets@1", "AR_maxDets@10", "AR_maxDets@100", "AR_small", "AR_medium", "AR_large"]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def score_predictions(gt_path: Path, paths: dict[str, Path]) -> dict:
    metrics = {}
    for kind, path in paths.items():
        predictions = json.loads(path.read_text(encoding="utf-8"))
        gt = COCO(str(gt_path))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]),
                          "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType=kind)
        evaluator.params.useCats = 1
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics.update({f"coco_eval_{kind}_{key}": float(value)
                        for key, value in zip(METRICS, evaluator.stats)})
        print("4. METRICS UPDATED!")
        # COCOeval retains large IoU/mask tables. Drop one evaluation completely
        # before loading the prediction JSON for the next kind.
        del evaluator, dt, gt, predictions
        gc.collect()
    return metrics


def evaluate_coco(model_dir="model/sam3_onnx",
                  annotations="D:/APP/datasets/coco/annotations/instances_val2017.json",
                  image_dir="D:/APP/datasets/coco/images/val2017",
                  output_dir="outputs/sam3_val", *, device="auto", score_threshold=0.25,
                  mask_threshold=0.5, max_images=None, query_batch_size=1) -> dict:
    """在 COCO 验证集上评估 SAM3 ONNX 模型，并用 pycocotools 计算检测与分割指标。

    逐图执行文本查询推理，将边界框与 RLE 分割结果流式写入 JSON 文件，
    最后调用 pycocotools 评分并保存指标；全程状态写入 run.json，
    异常时记录错误信息后原样抛出。

    Args:
        model_dir: SAM3 ONNX 模型目录路径。
        annotations: COCO 标注文件（instances_val2017.json）路径。
        image_dir: 图像文件所在目录路径。
        output_dir: 输出目录，用于保存 ground_truth.json、run.json、
            predictions_bbox.json、predictions_segm.json 和 metrics.json。
        device: 推理设备，"auto" 表示自动选择。
        score_threshold: 预测分数阈值，低于该值的预测被过滤。
        mask_threshold: 掩码二值化阈值。
        max_images: 最多评估的图像数量；None 表示评估全部图像。
        query_batch_size: 推理引擎的查询批大小。

    Returns:
        dict: pycocotools 对 bbox 与 segm 两类预测计算的评估指标。
    """
    annotations, image_dir, output_dir = Path(annotations), Path(image_dir), Path(output_dir)
    raw_bytes = annotations.read_bytes()
    annotations_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes)
    all_images = sorted(data["images"], key=lambda row: row["id"])
    categories = sorted(data["categories"], key=lambda row: row["id"])
    images = all_images if max_images is None else all_images[:max_images]
    selected_ids = {row["id"] for row in images}
    output_dir.mkdir(parents=True, exist_ok=True)
    gt = {**data, "images": images,
          "annotations": [ann for ann in data["annotations"] if ann["image_id"] in selected_ids]}
    gt.setdefault("info", {})
    gt_path = output_dir / "ground_truth.json"
    _write_json(gt_path, gt)
    print("1. Variable All Set!")

    engine = SAM3ONNXInference(model_dir, device=device, query_batch_size=query_batch_size)
    print("2. SAM3ONNXInference Engine Initialized!")
    versions = {}
    for package in ("onnxruntime-gpu", "onnxruntime", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = {
        "status": "running", "model_dir": str(Path(model_dir).resolve()), "device": engine.device,
        "inference_backend": "onnxruntime", "evaluation_backend": "pycocotools",
        "annotations": str(annotations.resolve()),
        "annotations_sha256": annotations_sha256,
        "image_dir": str(image_dir.resolve()), "image_ids": sorted(selected_ids),
        "num_images": len(images), "num_categories": len(categories), "max_images": max_images,
        "score_threshold": score_threshold, "mask_threshold": mask_threshold,
        "query_batch_size": query_batch_size, "versions": versions,
    }
    _write_json(output_dir / "run.json", manifest)
    # The full COCO annotations are already written to ground_truth.json. Keeping
    # these duplicate Python objects alive wastes hundreds of MB on a 5000-image run.
    del raw_bytes, data, all_images, gt
    gc.collect()
    paths = {kind: output_dir / f"predictions_{kind}.json" for kind in ("bbox", "segm")}
    count, started = 0, time.perf_counter()
    try:
        print("2. Start predicting!")
        with paths["bbox"].open("w", encoding="utf-8") as boxes_file, \
             paths["segm"].open("w", encoding="utf-8") as masks_file:
            boxes_file.write("[")
            masks_file.write("[")
            queries = [category["name"] for category in categories]
            for index, info in enumerate(images):
                with Image.open(image_dir / info["file_name"]) as source:
                    image = source.convert("RGB")
                for prediction in engine.iter_predict(image, queries, score_threshold=score_threshold,
                                                      mask_threshold=mask_threshold):
                    category = categories[prediction.query_index]
                    x1, y1, x2, y2 = prediction.box_xyxy
                    rle = mask_utils.encode(np.asfortranarray(prediction.mask))
                    rle["counts"] = rle["counts"].decode("ascii")
                    common = {"image_id": info["id"], "category_id": category["id"],
                              "score": prediction.score}
                    separator = ",\n" if count else ""
                    boxes_file.write(separator + json.dumps(
                        {**common, "bbox": [x1, y1, x2 - x1, y2 - y1]}, allow_nan=False))
                    masks_file.write(separator + json.dumps(
                        {**common, "segmentation": rle}, allow_nan=False))
                    count += 1
                boxes_file.flush()
                masks_file.flush()
                print(f"COCO {index + 1}/{len(images)} images; {count} predictions", flush=True)
            boxes_file.write("]")
            masks_file.write("]")
        # Evaluation is CPU-only; release model sessions before COCOeval's memory peak.
        engine.close()
        metrics = score_predictions(gt_path, paths)
        _write_json(output_dir / "metrics.json", metrics)
        manifest.update(status="complete", num_predictions=count,
                        elapsed_seconds=time.perf_counter() - started)
        return metrics
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        engine.close()
        _write_json(output_dir / "run.json", manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate direct ONNX Runtime SAM3 on COCO JSON.")
    parser.add_argument("--model", default="model/sam3_onnx")
    parser.add_argument("--annotations", type=Path,
                        default=Path("D:/APP/datasets/coco/annotations/instances_val2017.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("D:/APP/datasets/coco/images/val2017"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_val"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--query-batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    metrics = evaluate_coco(args.model, args.annotations, args.image_dir, args.output_dir,
                            device=args.device, score_threshold=args.score_threshold,
                            mask_threshold=args.mask_threshold, max_images=args.max_images,
                            query_batch_size=args.query_batch_size)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
