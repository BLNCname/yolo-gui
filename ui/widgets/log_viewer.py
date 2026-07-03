"""Log viewer for training output."""

from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt


class LogViewer(QTextEdit):
    """Widget displaying training logs in real-time."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self.setReadOnly(True)
        self.setFontFamily("Courier New")
        self.setFontPointSize(9)
        
        # Dark theme styling
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        # Keep last 500 lines by limiting block count
        doc = self.document()
        doc.setMaximumBlockCount(500)
    
    def append_log(self, message: str):
        """Append a log message."""
        self.append(message)
    
    def clear_logs(self):
        """Clear all logs."""
        self.clear()


class LogViewerWidget(QWidget):
    """Container widget for log viewer with controls."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        clear_btn = QPushButton("Очистить")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        clear_btn.clicked.connect(self.log_viewer.clear_logs)
        
        toolbar.addWidget(clear_btn)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        layout.addWidget(self.log_viewer)
    
    @property
    def log_viewer(self) -> QTextEdit:
        """Get the underlying log viewer."""
        if not hasattr(self, '_log_viewer'):
            self._log_viewer = LogViewer()
        return self._log_viewer
    
    def append_log(self, message: str):
        """Append a log message to the viewer."""
        self.log_viewer.append_log(message)
    
    def clear_logs(self):
        """Clear all logs."""
        self.log_viewer.clear_logs()
