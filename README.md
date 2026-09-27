# LHPCMamba

LHPCMamba: Lightweight Hybrid Parallel CNN-Mamba Network for Driver State Detection 

This repository contains the implementation used for LHPCMamba experiments. The model definition is available at:

```text
ultralytics/cfg/models/LHPCMamba/LHPCMamba.yaml
```


## Main Files

```text
LHPCMamba/
|-- lhpcmamba_train.py
|-- selective_scan/
`-- ultralytics/
    |-- cfg/models/LHPCMamba/LHPCMamba.yaml
    `-- nn/modules/
        |-- lhpcmamba.py
        |-- common_utils_lhpcmamba.py
        `-- block.py
```


## Installation

The code was developed and tested in a CUDA-enabled Linux environment. We recommend using a fresh conda environment.

```bash
conda create -n lhpcmamba python=3.11 -y
conda activate lhpcmamba
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch==2.3.0 torchvision torchaudio
```

Install the remaining dependencies:

```bash
pip install seaborn thop timm einops
cd selective_scan
pip install .
cd ..
pip install -e .
```

If your CUDA, PyTorch, or compiler versions differ, rebuild `selective_scan` after activating the target environment.

## Dataset

LHPCMamba follows the standard Ultralytics dataset YAML format.

Example COCO-style dataset YAML:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
```

For COCO, the default config is:

```text
ultralytics/cfg/datasets/coco.yaml
```

## Training

Run training from the repository root:

```bash
python lhpcmamba_train.py \
  --task train \
  --data ultralytics/cfg/datasets/coco.yaml \
  --config ultralytics/cfg/models/LHPCMamba/LHPCMamba.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch_size 32 \
  --device 0 \
  --workers 8 \
  --amp \
  --project output_dir/coco \
  --name lhpcmamba
```


## Validation

```bash
python lhpcmamba_train.py \
  --task val \
  --data ultralytics/cfg/datasets/coco.yaml \
  --config output_dir/coco/lhpcmamba/weights/best.pt \
  --imgsz 640 \
  --device 0 \
  --project output_dir/coco \
  --name lhpcmamba_val
```

To validate a trained checkpoint with the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/lhpcmamba/weights/best.pt")
metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", imgsz=640, device=0)
print(metrics)
```

## Export

Export to ONNX:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/lhpcmamba/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

Export to TensorRT engine:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/lhpcmamba/weights/best.pt")
model.export(format="engine", imgsz=640, half=True, device=0)
```


## Acknowledgement

This project is built on top of:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [VMamba selective scan](https://github.com/MzeroMiko/VMamba)

We thank the authors of these open-source projects for their excellent work.

## License

This repository follows the license terms of the upstream Ultralytics YOLO codebase. Please check `LICENSE` for details.
