"""Tests for trainer module."""

import pytest


class TestTrainer:
    """Test Trainer class."""
    
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
    
    def test_trainer_batch_size_auto(self):
        """Batch size should support auto (-1)."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        trainer.batch_size = -1
        assert trainer.batch_size == -1
        
        trainer.batch_size = 0
        assert trainer.batch_size == -1
    
    def test_trainer_lr0_bounds(self):
        """Learning rate should be bounded to reasonable values."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        trainer.lr0 = 0.0000001  # Very small
        assert trainer.lr0 >= 1e-7
        
        trainer.lr0 = -1.0  # Negative should be clamped
        assert trainer.lr0 == 0.0000001
    
    def test_trainer_cache_validation(self):
        """Cache setting should validate against allowed values."""
        from core.trainer import create_trainer
        
        trainer = create_trainer()
        
        trainer.cache = "ram"
        assert trainer.cache == "ram"
        
        trainer.cache = "disk"
        assert trainer.cache == "disk"
        
        trainer.cache = "invalid"
        assert trainer.cache == "off"
