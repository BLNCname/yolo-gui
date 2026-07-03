"""Tests for utility functions."""

import tempfile
import os

import pytest


class TestHelpers:
    """Test utility helper functions."""
    
    def test_validate_path_empty(self):
        """validate_path should reject empty path."""
        from utils.helpers import validate_path
        
        valid, error = validate_path("")
        
        assert valid is False
        assert "empty" in error.lower()
    
    def test_validate_path_nonexistent(self):
        """validate_path should reject nonexistent path when must_exist=True."""
        from utils.helpers import validate_path
        
        valid, error = validate_path("/nonexistent/path", must_exist=True)
        
        assert valid is False
    
    def test_validate_path_exists(self):
        """validate_path should accept existing path."""
        from utils.helpers import validate_path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, error = validate_path(tmpdir, must_exist=True)
            
            assert valid is True
            assert error == ""
    
    def test_validate_yaml_path_valid(self):
        """validate_yaml_path should accept valid .yaml file."""
        from utils.helpers import validate_yaml_path
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            f.write(b"# Test\n")
            yaml_path = f.name
        
        try:
            valid, error = validate_yaml_path(yaml_path)
            
            assert valid is True
            assert error == ""
        finally:
            os.unlink(yaml_path)
    
    def test_validate_yaml_path_invalid_extension(self):
        """validate_yaml_path should reject non-YAML files."""
        from utils.helpers import validate_yaml_path
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            yaml_path = f.name
        
        try:
            valid, error = validate_yaml_path(yaml_path)
            
            assert valid is False
            assert "yaml" in error.lower() or "extension" in error.lower()
        finally:
            os.unlink(yaml_path)
    
    def test_get_image_files(self):
        """get_image_files should return image files from directory."""
        from utils.helpers import get_image_files
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some image files
            for ext in ['.jpg', '.png', '.txt']:
                open(os.path.join(tmpdir, f"test{ext}"), 'w').close()
            
            images = get_image_files(tmpdir)
            
            assert len(images) == 2
            assert all(img.endswith(('.jpg', '.png')) for img in images)
    
    def test_format_bytes(self):
        """format_bytes should format bytes correctly."""
        from utils.helpers import format_bytes
        
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
    
    def test_parse_yolo_label_valid(self):
        """parse_yolo_label should parse valid YOLO label."""
        from utils.helpers import parse_yolo_label
        
        line = "0 0.5 0.5 0.3 0.4"
        result = parse_yolo_label(line)
        
        assert result is not None
        assert result['class'] == 0
        assert result['x_center'] == 0.5
    
    def test_parse_yolo_label_invalid(self):
        """parse_yolo_label should return None for invalid format."""
        from utils.helpers import parse_yolo_label
        
        assert parse_yolo_label("") is None
        assert parse_yolo_label("invalid data") is None
        assert parse_yolo_label("0 0.5 0.5") is None  # Missing fields
    
    def test_validate_yolo_coordinates_valid(self):
        """validate_yolo_coordinates should accept valid coordinates."""
        from utils.helpers import validate_yolo_coordinates
        
        assert validate_yolo_coordinates(0.5, 0.5, 0.3, 0.4) is True
        assert validate_yolo_coordinates(0.0, 0.0, 1.0, 1.0) is True
    
    def test_validate_yolo_coordinates_invalid(self):
        """validate_yolo_coordinates should reject out-of-range coordinates."""
        from utils.helpers import validate_yolo_coordinates
        
        assert validate_yolo_coordinates(-0.1, 0.5, 0.3, 0.4) is False
        assert validate_yolo_coordinates(0.5, 1.1, 0.3, 0.4) is False
