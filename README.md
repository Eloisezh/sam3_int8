# SAM3 量化

本目录用于 SAM3 模型的 ONNX 导出、文本提示推理、COCO 检测与实例分割评估，以及部分节点的动态量化实验。建议先使用已有 ONNX 模型跑通少量图片评估，再进行量化实验。

本文根据当前源码与已有运行记录整理。命令以 Windows PowerShell 为例，均从项目根目录执行；数据集路径需要按实际位置调整。

## 1. 目录与入口

```text
SAM3/
├── README.md                 入门说明
├── export.py                 使用 Ultralytics 导出 ONNX
├── export.ipynb              对应的 Notebook 实验
├── model/
│   ├── sam3.pt               原始模型权重
│   ├── sam3_onnx/            已导出的 ONNX 模块
│   └── sam3_int8_onnx/       部分模块的量化结果及外部权重
├── data/
│   └── coco8-seg.json        小规模 COCO 格式标注，不含图片
├── src/
│   ├── inference.py          文本提示 ONNX 推理引擎
│   ├── evaluate.py           COCO 评估命令行入口
│   ├── quantize.py           MatMul 动态 QUInt8 量化入口
│   ├── yolo_seg_to_coco.py   YOLO 分割标注转换工具
│   ├── api_for_usage.py      尚未接通的高层接口草稿
│   └── ewq.py                权重评分与节点选择实验
└── outputs/                  各次评估结果
```

当前文本推理依次使用以下三个文件：

1. `sam3_vision_encoder.onnx`：提取图像特征。
2. `sam3_text_encoder.onnx`：将文本查询编码为特征。
3. `sam3_decoder_text.onnx`：结合图像与文本生成框、掩码和分数。

其他 ONNX 模块虽存在于目录中，但没有被 `SAM3ONNXInference` 的文本推理流程使用。

## 2. 准备 Python 环境

当前目录没有 `requirements.txt` 或环境锁定文件。Notebook 记录使用过 Python 3.10.21；已有评估记录使用过 `onnxruntime-gpu==1.23.2` 和 `pycocotools==2.0.8` / `2.0.11`，这些记录不是完整的可复现环境清单。

优先使用已有 SAM3 环境。本机 Notebook 记录的解释器路径为：

```powershell
Set-Location E:\AAAphd\SAM3
$Sam3Python = 'D:\APP\Coding\conda\envs\sam3\python.exe'
& $Sam3Python -c "import sys; print(sys.executable)"
```

下文用 `python` 表示你选定的解释器。如果尚未激活环境，可将命令开头的 `python` 换成 `& $Sam3Python`。先确认普通 `python` 没有指向其他环境：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import numpy, PIL, torch, onnxruntime, pycocotools; print('基础依赖可导入')"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
python -c "import clip; t = clip.simple_tokenizer.SimpleTokenizer(); print(t(['person'], context_length=32).shape)"
```

依赖按用途分为：

| 用途 | 源码使用的包 |
| --- | --- |
| 文本推理 | `numpy`、`Pillow`、`torch`、`onnxruntime`、`clip` |
| COCO 评估 | 上述依赖，加 `pycocotools` |
| 动态量化 | `onnx`、ONNX Runtime 的量化模块 |
| 模型导出 | `ultralytics` 及其导出依赖 |

特别注意：源码要求 `clip.simple_tokenizer.SimpleTokenizer()` 能以 `context_length=32` 调用并返回 Tensor；仅有同名 `clip` 包不代表接口兼容。当前目录没有记录这个包的安装来源，应先通过上面的检查，避免直接安装同名包后假定可用。

GPU 环境还需要匹配的 CUDA/cuDNN 运行库。`--device auto` 在当前实现中会要求 CUDA provider，**不会自动回退 CPU**；没有可用 GPU 时显式传入 `--device cpu`。

## 3. 第一次运行：评估 1 张图片

已有 `data/coco8-seg.json` 的图片路径包含 `train/`、`val/` 前缀，因此 `--image-dir` 应指向数据集的 `images` 目录，而不是 `images/val`。本机历史记录使用 `D:/APP/datasets/coco8-seg/images`。

先检查关键输入：

```powershell
Test-Path model/sam3_onnx/sam3_vision_encoder.onnx
Test-Path model/sam3_onnx/sam3_text_encoder.onnx
Test-Path model/sam3_onnx/sam3_decoder_text.onnx
Test-Path D:/APP/datasets/coco8-seg/images/train/000000000009.jpg
python -m src.evaluate --help
```

然后执行小规模验证。PowerShell 中续行符是反引号，反引号后不要留空格：

```powershell
python -m src.evaluate `
  --model model/sam3_onnx `
  --annotations data/coco8-seg.json `
  --image-dir D:/APP/datasets/coco8-seg/images `
  --output-dir outputs/getting_started_1image `
  --device cuda:0 `
  --max-images 1
```

若使用 CPU，将 `cuda:0` 改为 `cpu`。即使只评估一张图，也会对标注中所有类别名称逐一执行文本查询；这个小数据集含 80 个类别，因此运行不一定很快。

跑通后把 `--max-images 1` 改为 `--max-images 8`，并换一个输出目录。这里的小样本用于检查流程，不适合代表完整 COCO 验证集的精度。

### 如何判断运行成功

输出目录中会生成：

| 文件 | 含义 |
| --- | --- |
| `run.json` | 设备、阈值、样本 ID、版本及运行状态 |
| `ground_truth.json` | 本次选中图片对应的真实标注 |
| `predictions_bbox.json` | 检测框预测，COCO 的 `[x, y, width, height]` 格式 |
| `predictions_segm.json` | 分割预测，COCO RLE 掩码格式 |
| `metrics.json` | 检测与分割的 AP / AR 指标 |

`run.json` 中的 `status: "complete"` 表示评估已完成。`bbox` 表示框指标，`segm` 表示掩码指标；`AP_50` 和 `AP_75` 分别使用 0.50 和 0.75 的 IoU 阈值。小样本上部分指标可能为 `-1`，表示对应条件下没有可计算的有效指标。

同一输出目录再次运行会覆盖同名结果，请为每次实验指定新的 `--output-dir`。运行中断时，预测 JSON 可能尚未写入结束括号；部分初始化错误也发生在 `run.json` 写入之前，应结合终端报错排查。

## 4. 在 Python 中对单张图片推理

将下面代码保存为根目录下的 `demo.py`，修改图片路径后执行 `python demo.py`：

```python
from PIL import Image
from src.inference import SAM3ONNXInference

queries = ["person", "car"]
with Image.open("D:/APP/datasets/coco8-seg/images/train/000000000009.jpg") as source:
    image = source.convert("RGB")

with SAM3ONNXInference("model/sam3_onnx", device="cuda:0") as engine:
    for prediction in engine.iter_predict(image, queries, score_threshold=0.25):
        print(queries[prediction.query_index], prediction.score)
        print("像素坐标框:", prediction.box_xyxy)
        print("掩码尺寸:", prediction.mask.shape)
```

每个结果包含查询序号、分数、原图像素坐标下的 `[x1, y1, x2, y2]` 框，以及形状为 `(原图高度, 原图宽度)`、值为 0/1 的掩码。`predict()` 会将所有结果收集为列表；`iter_predict()` 适合逐个处理结果，减少掩码同时驻留内存。

## 5. 扩大评估与准备数据

评估完整 COCO 数据时，明确传入标注和图片目录：

```powershell
python -m src.evaluate `
  --model model/sam3_onnx `
  --annotations D:/APP/datasets/coco/annotations/instances_val2017.json `
  --image-dir D:/APP/datasets/coco/images/val2017 `
  --output-dir outputs/getting_started_coco100 `
  --device cuda:0 `
  --max-images 100
```

去掉 `--max-images` 会评估标注中的全部图片。样本按图片 ID 排序后截取，并非随机抽样。评估会把 `categories[].name` 当作文本提示，并把查询结果映射回该类别 ID。

常用参数：`--score-threshold` 默认 `0.25`，`--mask-threshold` 默认 `0.5`；当前解码器固定 batch 为 1，`--query-batch-size` 必须为 `1`。比较不同模型时保持数据、样本和阈值一致。

若数据是 YOLO 分割格式，可运行：

```powershell
python -m src.yolo_seg_to_coco `
  D:/APP/datasets/coco8-seg `
  data/coco8-seg-converted.json `
  --splits train val
```

输入目录需包含 `images/<split>/` 与 `labels/<split>/`。标签每行是 `类别索引 x1 y1 x2 y2 ...`，坐标为归一化多边形坐标。工具使用固定的 COCO 80 类顺序和官方类别 ID，**不能直接用于任意自定义类别数据集**。转换后评估的图片根目录仍为 `images/`；还应检查图片 ID 唯一，尤其是跨 split 同名或非数字文件名的情况。缺少标签文件的图片会作为无标注图片写入结果。

## 6. 可选：导出与量化实验

### 导出

已有 ONNX 文件时可跳过导出。`export.py` 加载 `model/sam3.pt`，调用 `model.export(format="onnx", imgsz=1036)`：

```powershell
python export.py
```

导出前确认当前 Ultralytics 环境支持所需 SAM3 导出接口。Notebook 保存过内核崩溃记录，不能据此认为导出已验证成功；导出后应检查实际产物路径、模块名及输入输出是否与 `inference.py` 匹配。

### 动态量化

下面示例将文本编码器输出到单独的实验目录：

```powershell
python -m src.quantize `
  model/sam3_onnx/sam3_text_encoder.onnx `
  model/quantization_trial/sam3_text_encoder_int8.onnx
```

当前量化仅针对常量权重的 `MatMul`，使用动态 `QUInt8`，不需要校准数据。可通过 `--nodes-to-quantize` 或 `--nodes-to-exclude` 指定 ONNX 节点名。输出可能包括 `.onnx`、外部权重文件和 `.quantization.json`，移动模型时需要保留外部权重及其相对路径。

量化单个模块不会自动构成完整的可评估模型目录。评估引擎要求目录内存在前述三个固定名称的模块；需要在单独目录中准备完整组合，并确认量化模块与所选 provider 兼容。默认量化目标 `sam3_prompt_encoder.onnx` 不在当前文本评估链路中，对它量化不会改变这条链路。

源码目前用 `importlib.metadata.version("onnxruntime")` 写量化版本记录；若环境只安装 `onnxruntime-gpu`，可能在产物已生成后因找不到该发行包元数据而失败。这需要修正版本查询逻辑，不能仅凭文件存在认定整个量化步骤成功。

## 7. 常见问题与阅读顺序

| 现象 | 优先检查 |
| --- | --- |
| `Missing ONNX modules` | `--model` 是否指向包含三个必需模块的目录 |
| `CUDAExecutionProvider unavailable` | 当前解释器和 ORT provider；无 GPU 时显式用 `--device cpu` |
| `clip` 不存在或 tokenizer 不可调用 | 环境中的 `clip` 来源与调用接口是否匹配 |
| 图片找不到 | `--image-dir` 与标注的 `file_name` 拼接后是否存在 |
| 显存或内存不足 | 先用少量图片验证；大模型仍有单张图的内存峰值，减少图片总数不会降低该峰值 |
| `SAM3(...).val(...)` 报错 | `api_for_usage.py` 尚未接通，应使用 `src.evaluate` |

`api_for_usage.py` 引用了未定义的 `load_model`，且将导入的 `evaluate`、`quantize` 模块当作函数调用，目前不应作为使用入口。`ewq.py` 仍以 YOLO 卷积权重命名为筛选条件，默认模型也是 YOLO，尚未形成 SAM3 量化与评估的自动流程。

建议按 `evaluate.py` → `inference.py` → `quantize.py` 的顺序阅读源码：先理解输入输出，再看文本推理过程，最后研究量化。本文编写时核对了源码和历史记录，未重新执行模型导出、推理或完整评估。
