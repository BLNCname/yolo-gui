# UltraGUI

UltraGUI is a PyQt6 desktop interface for running a YOLO training, inference,
tracking, export, and dataset-validation workflow from one local application.
The project is tuned for Windows developers who want to use AMD Strix Halo
machines through WSLg and ROCm while keeping the UI on the desktop.

## Why This Project Exists

AMD Strix Halo laptops and mini PCs have strong integrated Radeon graphics, but
the Python ML setup can be awkward when the Windows package stack falls back to
CPU. UltraGUI keeps the application workflow simple while pushing the heavy
YOLO work into a ROCm-enabled WSL environment.

The app currently includes:

- YOLO training controls with epoch progress, logs, and metric charts.
- Image/video/camera inference preview.
- Video object tracking with BoT-SORT or ByteTrack.
- Model export controls for ONNX and other Ultralytics-supported formats.
- YOLO dataset YAML validation and basic dataset statistics.
- A Windows PowerShell launcher for WSLg + ROCm environments.

## Target Setup: AMD Strix Halo + ROCm + WSLg

The recommended development target is:

- Windows 11
- WSL2 with Ubuntu 24.04
- WSLg enabled for desktop GUI windows
- ROCm-capable PyTorch build for AMD graphics
- Python 3.11 or 3.12

The local ROCm virtual environments used during development are intentionally
not part of the repository. They can be very large because ROCm and driver
components pull in native libraries.

## Quick Start

Create a Python environment, install dependencies, and run the app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

For AMD Strix Halo/ROCm work, use the WSL launcher after preparing the WSL
environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_wsl_gui.ps1 -Distro Ubuntu-24.04
```

The launcher checks the WSL Python environment, verifies PyTorch GPU visibility,
and starts the PyQt6 app through WSLg.

## ROCm Notes

Install the ROCm PyTorch packages that match your runtime. The exact index URL
changes with PyTorch and ROCm releases, so treat this as a starting point rather
than a pinned production command:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r requirements.txt
```

If GPU detection fails, first verify PyTorch from inside WSL:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", None))
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0:", torch.cuda.get_device_name(0))
PY
```

## Tests

Run the test suite with:

```powershell
pytest -q
```

The current tests are intentionally lightweight. They cover core object
creation, helper behavior, basic widget wiring, and repository readiness checks.

