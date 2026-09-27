"""Convert a YOLO segmentation dataset directory to COCO instance JSON."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from PIL import Image


COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Official COCO category IDs are non-contiguous; YOLO class indices are contiguous.
COCO_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def polygon_area(points: list[float]) -> float:
    pairs = list(zip(points[0::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pairs, pairs[1:] + pairs[:1]))) / 2


def convert(dataset: Path, output: Path, splits: list[str]) -> dict:
    images, annotations = [], []
    annotation_id = 1
    for split in splits:
        image_root = dataset / "images" / split
        label_root = dataset / "labels" / split
        if not image_root.is_dir():
            raise FileNotFoundError(f"Image split not found: {image_root}")
        for image_path in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            with Image.open(image_path) as image:
                width, height = image.size
            try:
                image_id = int(image_path.stem)
            except ValueError:
                image_id = len(images) + 1
            images.append({"id": image_id, "file_name": image_path.relative_to(dataset / "images").as_posix(),
                           "width": width, "height": height})
            label_path = label_root / image_path.relative_to(image_root).with_suffix(".txt")
            if not label_path.exists():
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if not fields:
                    continue
                class_index = int(fields[0])
                if not 0 <= class_index < len(COCO_IDS):
                    raise ValueError(f"Invalid class {class_index} in {label_path}:{line_number}")
                coordinates = [float(value) for value in fields[1:]]
                if len(coordinates) < 6 or len(coordinates) % 2:
                    raise ValueError(f"Invalid polygon in {label_path}:{line_number}")
                polygon = [value * (width if index % 2 == 0 else height)
                           for index, value in enumerate(coordinates)]
                xs, ys = polygon[0::2], polygon[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                annotations.append({
                    "id": annotation_id, "image_id": image_id, "category_id": COCO_IDS[class_index],
                    "segmentation": [polygon], "area": polygon_area(polygon),
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0,
                })
                annotation_id += 1
    result = {
        "info": {"description": f"Converted from {dataset}", "date_created": date.today().isoformat()},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": category_id, "name": name, "supercategory": "none"}
                       for category_id, name in zip(COCO_IDS, COCO_NAMES)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    result = convert(args.dataset.resolve(), args.output.resolve(), args.splits)
    print(f"Wrote {args.output}: {len(result['images'])} images, {len(result['annotations'])} annotations")


if __name__ == "__main__":
    main()
