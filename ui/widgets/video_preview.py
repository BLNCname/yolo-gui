"""Video preview widget with tracking visualization."""

import cv2
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QHBoxLayout, QPushButton


class VideoPreviewLabel(QLabel):
    """Widget displaying video with tracking bounding boxes."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        
        # Dark background
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        
        self._pixmap = None
    
    def set_image(self, image):
        """Set image to display (BGR from OpenCV)."""
        if image is None:
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.width(), 
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)


class VideoPreviewWidget(QWidget):
    """Container widget for video preview with controls."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video preview label
        self.preview_label = QLabel("Нет видео")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                color: #666;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        self.preview_label.setMinimumSize(640, 480)
        
        main_layout.addWidget(self.preview_label)
    
    def set_image(self, image):
        """Set image to display."""
        if image is None:
            self.preview_label.setText("Нет видео")
            return
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.preview_label.setPixmap(scaled_pixmap)
    
    def clear(self):
        """Clear the preview."""
        self.preview_label.setText("Нет видео")


def create_video_preview(parent: QWidget = None) -> VideoPreviewWidget:
    """Factory function to create a video preview widget."""
    return VideoPreviewWidget(parent)
