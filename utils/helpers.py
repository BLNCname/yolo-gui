"""Utility functions for UltraGUI."""

import os
from pathlib import Path


def validate_path(path: str, must_exist: bool = True) -> tuple[bool, str]:
    """
    Validate a file or directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Path is empty"
    
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    if must_exist and not os.path.exists(path):
        return False, f"Path does not exist: {path}"
    
    return True, ""


def validate_yaml_path(path: str) -> tuple[bool, str]:
    """
    Validate a YAML file path.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    is_valid, error = validate_path(path)
    if not is_valid:
        return False, error
    
    if not path.lower().endswith(('.yaml', '.yml')):
        return False, "File must have .yaml or .yml extension"
    
    return True, ""


def get_image_files(directory: str) -> list[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of image file paths
    """
    if not os.path.isdir(directory):
        return []
    
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    
    for f in Path(directory).iterdir():
        if f.suffix.lower() in extensions:
            images.append(str(f))
    
    return sorted(images)


def get_label_files(directory: str) -> list[str]:
    """
    Get all label files from a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of label file paths
    """
    if not os.path.isdir(directory):
        return []
    
    labels = [str(f) for f in Path(directory).glob("*.txt")]
    return sorted(labels)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def parse_yolo_label(line: str) -> dict | None:
    """
    Parse a YOLO format label line.
    
    Args:
        line: Line in format "class x_center y_center width height"
        
    Returns:
        Dict with class, x, y, w, h or None if invalid
    """
    try:
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        
        return {
            'class': int(parts[0]),
            'x_center': float(parts[1]),
            'y_center': float(parts[2]),
            'width': float(parts[3]),
            'height': float(parts[4]),
        }
    except (ValueError, IndexError):
        return None


def validate_yolo_coordinates(xc: float, yc: float, w: float, h: float) -> bool:
    """
    Validate that YOLO coordinates are in valid range [0, 1].
    
    Args:
        xc: x_center
        yc: y_center  
        w: width
        h: height
        
    Returns:
        True if all coordinates are valid
    """
    return all(0.0 <= v <= 1.0 for v in [xc, yc, w, h])
