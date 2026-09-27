import json
from importlib.metadata import version
from pathlib import Path

import onnx


def partial_quantize_onnx(
    model_input,
    model_output,
    *,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
):
    """Dynamically quantize constant MatMul weights in an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_input = Path(model_input).resolve()
    model_output = Path(model_output).resolve()

    if not model_input.is_file():
        raise FileNotFoundError(model_input)
    if model_input == model_output:
        raise ValueError("model_output must be different from model_input")

    model_output.parent.mkdir(parents=True, exist_ok=True)

    # Both None and [] mean no node-name restriction in ORT:
    # all eligible MatMul nodes are considered for quantization.
    selected = None if nodes_to_quantize is None else list(nodes_to_quantize)
    excluded = list(nodes_to_exclude or [])

    quantize_dynamic(
        model_input=str(model_input),
        model_output=str(model_output),
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=selected,
        nodes_to_exclude=excluded,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
        extra_options={
            "MatMulConstBOnly": True,
        },
    )

    onnx.checker.check_model(str(model_output))
    quantized = onnx.load(str(model_output), load_external_data=False)
    matmul_integer_count = sum(
        node.op_type == "MatMulInteger" for node in quantized.graph.node
    )
    if selected is None and matmul_integer_count == 0:
        raise RuntimeError("Quantization completed but produced no MatMulInteger nodes")

    manifest = {
        "backend": "onnxruntime.quantize_dynamic",
        "onnxruntime_version": version("onnxruntime"),
        "model_input": str(model_input),
        "model_output": str(model_output),
        "weight_type": "QUInt8",
        "matmul_integer_count": matmul_integer_count,
        "nodes_to_quantize": selected,
        "nodes_to_exclude": excluded,
    }
    manifest_path = model_output.with_suffix(".quantization.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply ONNX Runtime dynamic QUInt8 quantization to MatMul nodes."
    )
    parser.add_argument(
        "model_input",
        nargs="?",
        default="model/sam3_onnx/sam3_prompt_encoder.onnx",
    )
    parser.add_argument(
        "model_output",
        nargs="?",
        default="model/sam3_int8_onnx/sam3_prompt_encoder_int8.onnx",
    )
    parser.add_argument("--nodes-to-quantize", nargs="+")
    parser.add_argument("--nodes-to-exclude", nargs="*", default=[])
    args = parser.parse_args()

    result = partial_quantize_onnx(
        args.model_input,
        args.model_output,
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
    )
    print(json.dumps(result, indent=2))
