"""Tracking module for video object tracking with YOLO."""

import sys
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class TrackerSignals(QObject):
    """PyQt signals for tracking events."""
    
    # Emitted when tracking starts
    started = pyqtSignal()
    
    # Emitted on each frame processed
    frame_processed = pyqtSignal(int)
    
    # Emitted with progress updates (0-100%)
    progress = pyqtSignal(int, str)
    
    # Emitted with log messages
    log_message = pyqtSignal(str)
    
    # Emitted when tracking completes successfully
    completed = pyqtSignal(dict)
    
    # Emitted on error
    error = pyqtSignal(str)


class Tracker(QThread):
    """Tracking thread that runs model.track() in background."""
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.signals = TrackerSignals()
        
        self._model_path: str = ""
        self._source: str = ""  # File path or camera index
        self._tracker_type: str = "botsort"
        self._conf: float = 0.5
        self._iou: float = 0.3
        
        self._running: bool = False
    
    @property
    def model_path(self) -> str:
        return self._model_path
    
    @model_path.setter
    def model_path(self, value: str):
        self._model_path = value
    
    @property
    def source(self) -> str:
        return self._source
    
    @source.setter
    def source(self, value: str):
        self._source = value
    
    @property
    def tracker_type(self) -> str:
        return self._tracker_type
    
    @tracker_type.setter
    def tracker_type(self, value: str):
        valid_trackers = ("botsort", "bytetrack")
        self._tracker_type = value if value in valid_trackers else "botsort"
    
    @property
    def conf(self) -> float:
        return self._conf
    
    @conf.setter
    def conf(self, value: float):
        self._conf = max(0.0, min(1.0, float(value)))
    
    @property
    def iou(self) -> float:
        return self._iou
    
    @iou.setter
    def iou(self, value: float):
        self._iou = max(0.0, min(1.0, float(value)))
    
    @pyqtSlot()
    def start_tracking(self):
        """Start tracking (slot for QThread)."""
        if not self._model_path:
            self.signals.error.emit("Model path is required")
            return
        
        if not self._source:
            self.signals.error.emit("Source (video file or camera index) is required")
            return
        
        self._running = True
        self.signals.started.emit()
        
        try:
            # Import ultralytics
            from ultralytics import YOLO
            
            # Load model
            self.signals.log_message.emit(f"Loading model: {self._model_path}")
            model = YOLO(self._model_path)
            
            # Track
            results = model.track(
                source=self._source,
                tracker=f"{self._tracker_type}.yaml",
                conf=self._conf,
                iou=self._iou,
                stream=True,
            )
            
            frame_count = 0
            total_frames = self._estimate_total_frames()
            
            for result in results:
                if not self._running:
                    break
                
                frame_count += 1
                progress = int((frame_count / total_frames) * 100) if total_frames else 0
                
                self.signals.frame_processed.emit(frame_count)
                self.signals.progress.emit(progress, f"Frame {frame_count}")
                
                if frame_count % 30 == 0:
                    self.signals.log_message.emit(f"Processed {frame_count} frames")
            
            self.signals.completed.emit({
                "frames": frame_count,
            })
        
        except Exception as e:
            self.signals.error.emit(str(e))
    
    def _estimate_total_frames(self) -> int:
        """Estimate total frames for progress bar."""
        # For video files, try to get actual count
        import cv2
        
        if self._source.isdigit():
            return 1000  # Camera - arbitrary large number
        
        cap = cv2.VideoCapture(self._source)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return max(total, 1)
        
        return 1000  # Default estimate
    
    @pyqtSlot()
    def stop_tracking(self):
        """Stop tracking."""
        self._running = False
        self.signals.log_message.emit("Tracking stopped by user")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def run(self):
        """Override QThread.run() to start tracking."""
        self.start_tracking()


def create_tracker():
    """Factory function to create a tracker instance."""
    return Tracker()
