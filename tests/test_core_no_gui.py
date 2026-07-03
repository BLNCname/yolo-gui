"""Tests for core modules (no GUI required)."""

import pytest


class TestDevice:
    """Test device autodetection functions."""
    
    def test_get_device_returns_valid_string(self):
        """get_device should return one of 'cuda', 'mps', or 'cpu'."""
        from core.device import get_device
        
        device = get_device()
        
        assert device in ('cuda', 'mps', 'cpu')
    
    def test_get_device_name_returns_non_empty_string(self):
        """get_device_name should return a non-empty string."""
        from core.device import get_device_name
        
        name = get_device_name()
        
        assert isinstance(name, str)
        assert len(name) > 0


class TestTrainer:
    """Test Trainer class (without GUI dependencies)."""
    
    def test_trainer_creation(self):
        """Trainer should be created without errors."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        assert trainer is not None
    
    def test_trainer_default_values(self):
        """Trainer should have default values set correctly."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        assert trainer.model_path == "yolo26s.pt"
        assert trainer.epochs == 100
        assert trainer.imgsz == 640
        assert trainer.batch_size == -1
        assert trainer.lr0 == 0.01
    
    def test_trainer_properties_setters(self):
        """Trainer property setters should work correctly."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        trainer.model_path = "custom_model.pt"
        trainer.epochs = 50
        trainer.imgsz = 320
        trainer.batch_size = 8
        trainer.lr0 = 0.001
        
        assert trainer.model_path == "custom_model.pt"
        assert trainer.epochs == 50
        assert trainer.imgsz == 320
        assert trainer.batch_size == 8
        assert trainer.lr0 == 0.001


class TestTracker:
    """Test Tracker class."""
    
    def test_tracker_creation(self):
        """Tracker should be created without errors."""
        from core.tracker import create_tracker
        
        tracker = create_tracker()
        
        assert tracker is not None
    
    def test_tracker_default_values(self):
        """Tracker should have default values."""
        from core.tracker import create_tracker
        
        tracker = create_tracker()
        
        assert tracker.model_path == ""
        assert tracker.source == ""
        assert tracker.conf == 0.5
        assert tracker.iou == 0.3


class TestPredictor:
    """Test Predictor class."""
    
    def test_predictor_creation(self):
        """Predictor should be created without errors."""
        from core.predictor import create_predictor
        
        predictor = create_predictor()
        
        assert predictor is not None
    
    def test_predictor_default_values(self):
        """Predictor should have default values."""
        from core.predictor import create_predictor
        
        predictor = create_predictor()
        
        assert predictor.model_path == ""
        assert predictor.source == ""
        assert predictor.conf == 0.25


class TestExporter:
    """Test Exporter class."""
    
    def test_exporter_creation(self):
        """Exporter should be created without errors."""
        from core.exporter import create_exporter
        
        exporter = create_exporter()
        
        assert exporter is not None
    
    def test_exporter_default_values(self):
        """Exporter should have default values."""
        from core.exporter import create_exporter
        
        exporter = create_exporter()
        
        assert exporter.model_path == ""
        assert exporter.format == "onnx"
        assert exporter.half is True
        assert exporter.dynamic is False


class TestDatasetManager:
    """Test DatasetManager class."""
    
    def test_dataset_manager_creation(self):
        """DatasetManager should be created without errors."""
        from core.dataset_manager import create_dataset_manager
        
        manager = create_dataset_manager()
        
        assert manager is not None
    
    def test_validate_yaml_missing_file(self):
        """validate_yaml should raise FileNotFoundError for missing file."""
        from core.dataset_manager import DatasetManager
        
        manager = DatasetManager()
        manager.yaml_path = "/nonexistent/path.yaml"
        
        valid, errors = manager.validate_yaml()
        
        assert valid is False
        assert len(errors) > 0
