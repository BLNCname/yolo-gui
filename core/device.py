"""Device autodetection for GPU/CPU."""

import torch


def get_device() -> str:
    """
    Autodetect available device: CUDA (ROCm) > MPS > CPU.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🟢 GPU обнаружен: {torch.cuda.get_device_name(0)}")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("🟡 Apple Silicon MPS обнаружен")
        return device
    
    device = "cpu"
    print("⚪ GPU не обнаружен — обучение на CPU (будет медленно)")
    return device


def get_device_name() -> str:
    """Get human-readable device name."""
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"
