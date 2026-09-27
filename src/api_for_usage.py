"""Small user-facing facade; numerical operations remain in experiment.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from . import evaluate, quantize


def coco_paths(data):
    """Resolve dataset-relative image and original instances-JSON paths."""
    path = Path(data).resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / spec.get("path", ".")).resolve()
    return root / spec["annotations"], root / spec["val"]


class SAM3:
    """SAM3(path).val(data=...). This is our facade, not an Ultralytics class."""

    def __init__(self, model_path="sam3.onnx", *, config_dir="model_source",
                 device="cuda", dtype=torch.float16, image_size=1008, providers=None):
        self.path = Path(model_path)
        self.asset = load_model(config_dir, self.path, device=device, dtype=dtype,
                                image_size=image_size, providers=providers)

    @property
    def model(self):
        return self.asset.model

    def val(self, data, *, output_dir=None, **kwargs):
        annotations, images = coco_paths(data)
        return evaluate(self.asset, annotations, images, output_dir or f"outputs/{self.path.stem}_val", **kwargs)


class SAM3Quantizer:
    """Selective ONNX Runtime dynamic QUInt8. Call val() separately to evaluate."""

    def __init__(self, model_path="sam3.onnx", data=None, *,
                 nodes_to_exclude=None, nodes_to_quantize=None, save_suffix="int8",
                 print_original=False, print_quantized=True, output_dir="outputs", **model_options):
        self.output_dir = Path(output_dir) / f"{Path(model_path).stem}-{save_suffix}"
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(self.output_dir)
        self.data = data
        self.sam3 = SAM3(model_path, **model_options)
        self.quantized_path = self.output_dir / Path(model_path).name
        if print_original:
            print(f"Original ONNX: {model_path}")
        self.sam3.asset = quantize(self.sam3.asset, nodes_to_quantize,
                                          output_path=self.quantized_path,
                                          nodes_to_exclude=nodes_to_exclude)
        self.sam3.path = self.quantized_path
        self.recipe_path = self.quantized_path.with_suffix(".quantization.json")
        result = self.sam3.asset.metadata["quantization"][0]
        self.nodes = result["nodes_to_quantize"]
        if print_quantized:
            print(json.dumps(result, indent=2))

    @property
    def model(self):
        return self.sam3.model

    def val(self, data=None, **kwargs):
        kwargs.setdefault("output_dir", self.output_dir / "coco")
        return self.sam3.val(data or self.data, **kwargs)
