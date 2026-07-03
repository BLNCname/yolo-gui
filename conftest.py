"""Pytest configuration for UltraGUI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure pytest-qt to use PyQt6
import pytest


def pytest_configure(config):
    """Configure pytest settings."""
    # Markers for different test types
    config.addinivalue_line("markers", "gui: mark test as GUI-related (requires QApplication)")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def qapp():
    """Provide QApplication fixture for GUI tests."""
    from PyQt6.QtWidgets import QApplication
    
    # Check if app already exists
    if QApplication.instance() is None:
        app = QApplication([])
        app.setApplicationName("UltraGUI-Tests")
    else:
        app = QApplication.instance()
    
    yield app
    
    # Cleanup (only if we created it)
    if QApplication.instance() is app:
        app.quit()
