#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(pwd)}"
VENV="${VENV:-.venv-rocm}"
MODEL_PATH="${MODEL_PATH:-yolo26s.pt}"
export MODEL_PATH

cd "$PROJECT"
source "$VENV/bin/activate"
export YOLO_CONFIG_DIR="$PROJECT/.ultralytics"

python - <<'PY'
from ultralytics import YOLO
import os
import torch

print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

if not torch.cuda.is_available():
    raise SystemExit("GPU is not available through PyTorch ROCm.")

model = YOLO(os.environ["MODEL_PATH"])
results = model.train(
    data="coco8.yaml",
    epochs=1,
    imgsz=320,
    batch=1,
    device=0,
    workers=0,
    project="runs/train",
    name="ultragui_rocm_smoke",
    exist_ok=True,
    plots=False,
    verbose=False,
)
print("save_dir:", results.save_dir)
PY
