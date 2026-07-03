"""Tests for log viewer widget."""

import pytest


class TestLogViewer:
    """Test LogViewerWidget class."""
    
    def test_log_viewer_creation(self, qapp):
        """LogViewer should be created without errors."""
        from ui.widgets.log_viewer import LogViewer, LogViewerWidget
        
        viewer = LogViewer()
        assert viewer is not None
        
        widget = LogViewerWidget()
        assert widget is not None
    
    def test_append_log(self, qapp):
        """LogViewer should append log messages."""
        from ui.widgets.log_viewer import LogViewer
        
        viewer = LogViewer()
        
        viewer.append_log("Test message 1")
        viewer.append_log("Test message 2")
        
        text = viewer.toPlainText()
        assert "Test message 1" in text
        assert "Test message 2" in text
    
    def test_clear_logs(self, qapp):
        """LogViewer should clear logs."""
        from ui.widgets.log_viewer import LogViewer
        
        viewer = LogViewer()
        
        viewer.append_log("Message to clear")
        viewer.clear_logs()
        
        assert viewer.toPlainText() == ""
    
    def test_widget_append_log(self, qapp):
        """LogViewerWidget should forward append calls."""
        from ui.widgets.log_viewer import LogViewerWidget
        
        widget = LogViewerWidget()
        widget.append_log("Test message")
        
        text = widget.log_viewer.toPlainText()
        assert "Test message" in text
    
    def test_widget_clear_logs(self, qapp):
        """LogViewerWidget should forward clear calls."""
        from ui.widgets.log_viewer import LogViewerWidget
        
        widget = LogViewerWidget()
        widget.append_log("Message to clear")
        widget.clear_logs()
        
        assert widget.log_viewer.toPlainText() == ""