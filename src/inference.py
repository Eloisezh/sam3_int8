"""Direct ONNX Runtime inference for exported SAM3 text models."""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@dataclass
class Prediction:
    query_index: int
    score: float
    box_xyxy: list[float]
    mask: np.ndarray


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def _nms(boxes, scores, threshold=0.7, limit=100):
    if not len(boxes):
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order, keep = scores.argsort()[::-1], []
    while order.size and len(keep) < limit:
        i, rest = int(order[0]), order[1:]
        keep.append(i)
        if not rest.size:
            break
        inter = (np.maximum(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0) *
                 np.maximum(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0))
        union = area[i] + area[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


class SAM3ONNXInference:
    """SAM3 vision -> text -> decoder pipeline, without Ultralytics model loading."""

    def __init__(self, model_dir="model/sam3_onnx", *, device="auto", query_batch_size=1):
        self.model_dir = Path(model_dir).resolve()
        required = ["sam3_vision_encoder.onnx", "sam3_text_encoder.onnx", "sam3_decoder_text.onnx"]
        missing = [x for x in required if not (self.model_dir / x).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ONNX modules: {', '.join(missing)}")
        if query_batch_size != 1:
            raise ValueError("sam3_decoder_text.onnx has static batch 1; use --query-batch-size 1")
        # ORT needs CUDA/cuDNN DLLs. This only exposes DLLs; no torch weights/model are loaded.
        self._dll_handles = []
        if os.name == "nt":
            try:
                import torch
                self._dll_handles.append(os.add_dll_directory(str(Path(torch.__file__).parent / "lib")))
            except (ImportError, OSError):
                pass
        import onnxruntime as ort
        available = ort.get_available_providers()
        cuda = device == "auto" or device.startswith("cuda")
        if cuda and "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
        if cuda:
            index = int(device.split(":")[1]) if ":" in device else 0
            self.providers = [("CUDAExecutionProvider", {"device_id": index,
                               "arena_extend_strategy": "kSameAsRequested"})]
            self.device = f"cuda:{index}"
        else:
            self.providers, self.device = ["CPUExecutionProvider"], "cpu"
        self.text_cache = {}
        self._sessions = {}

    def _session(self, name):
        import onnxruntime as ort
        if name not in self._sessions:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self._sessions[name] = ort.InferenceSession(
                str(self.model_dir / name), options, providers=self.providers)
        return self._sessions[name]

    def _run(self, session, feed, *, shrink_cuda_arena=False):
        run_options = None
        if shrink_cuda_arena and "CUDAExecutionProvider" in session.get_providers():
            import onnxruntime as ort
            run_options = ort.RunOptions()
            device_index = self.device.split(":", 1)[1]
            run_options.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage", f"gpu:{device_index}")
        values = session.run(None, feed, run_options)
        return {item.name: value for item, value in zip(session.get_outputs(), values)}

    def _image_features(self, image):
        session = self._session("sam3_vision_encoder.onnx")
        shape = session.get_inputs()[0].shape
        original = image.height, image.width
        resized = image.convert("RGB").resize((int(shape[3]), int(shape[2])), Image.Resampling.BILINEAR)
        array = ((np.asarray(resized, np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        output = self._run(
            session, {"images": np.ascontiguousarray(array)}, shrink_cuda_arena=True)
        # Even after shrinking its arena, the vision weights occupy about 2 GB.
        # Release them before loading/running the decoder on a 6 GB GPU.
        self._sessions.pop("sam3_vision_encoder.onnx", None)
        del session
        gc.collect()
        return output, original

    def _cache_text(self, queries):
        missing = [q for q in dict.fromkeys(queries) if q not in self.text_cache]
        if not missing:
            return
        import clip
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        session = self._session("sam3_text_encoder.onnx")
        for query in missing:
            tokens = tokenizer([query], context_length=32).numpy().astype(np.int64)
            output = self._run(session, {"tokens": tokens})
            feature = output["text_features"]
            if feature.shape == (1, 32, 256):
                feature = feature.transpose(1, 0, 2)
            self.text_cache[query] = feature, output["text_mask"].astype(bool, copy=False)
        # Text features are now cached as NumPy arrays; its 1.4 GB CUDA Session
        # is no longer needed for the rest of this evaluation.
        self._sessions.pop("sam3_text_encoder.onnx", None)
        del session, output
        gc.collect()

    def iter_predict(self, image, queries, *, score_threshold=0.25, mask_threshold=0.5):
        features, (height, width) = self._image_features(image)
        self._cache_text(queries)
        decoder = self._session("sam3_decoder_text.onnx")
        mask_cutoff = float(np.log(np.clip(mask_threshold, 1e-6, 1 - 1e-6) /
                                   np.clip(1 - mask_threshold, 1e-6, 1 - 1e-6)))
        try:
            out = masks = resized = None
            for query_index, query in enumerate(queries):
                # Release the previous query's large mask tensors before ORT
                # allocates the next output. Assignment alone releases them only
                # after session.run has already completed.
                out = masks = resized = None
                prompt, prompt_mask = self.text_cache[query]
                out = self._run(decoder, {
                    "fpn_feat_0": features["fpn_feat_0"], "fpn_feat_1": features["fpn_feat_1"],
                    "fpn_feat_2": features["fpn_feat_2"], "fpn_pos_2": features["fpn_pos_2"],
                    "prompt_features": prompt, "prompt_mask": prompt_mask})
                scores = (_sigmoid(out["pred_logits"]) *
                          _sigmoid(out["presence_logit_dec"])[:, None, :]).reshape(-1)
                selected = np.flatnonzero(scores > score_threshold)
                if not selected.size:
                    continue
                box = out["pred_boxes"].reshape(-1, 4)[selected]
                cx, cy, bw, bh = box.T
                xyxy = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)
                keep = _nms(xyxy, scores[selected])
                selected, xyxy = selected[keep], xyxy[keep]
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, 1) * width
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, 1) * height
                masks = out["pred_masks"].reshape(-1, *out["pred_masks"].shape[-2:])
                for model_index, box in zip(selected, xyxy):
                    resized = Image.fromarray(masks[model_index].astype(np.float32), mode="F").resize(
                        (width, height), Image.Resampling.BILINEAR)
                    mask = (np.asarray(resized) > mask_cutoff).astype(np.uint8)
                    yield Prediction(query_index, float(scores[model_index]), box.tolist(), mask)
        finally:
            # Decoder must not coexist with the next vision peak on a 6 GB GPU.
            self._sessions.pop("sam3_decoder_text.onnx", None)
            del decoder, features
            gc.collect()

    def close(self):
        """Release cached ONNX sessions before CPU-side evaluation."""
        self._sessions.clear()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def predict(self, image, queries, **kwargs):
        return list(self.iter_predict(image, queries, **kwargs))
