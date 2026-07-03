"""Tests for device autodetection."""

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
