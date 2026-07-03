"""Prediction/inference module for YOLO model."""

import sys
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class PredictorSignals(QObject):
    """PyQt signals for prediction events."""
    
    # Emitted when prediction starts
    started = pyqtSignal()
    
    # Emitted on each image processed
    image_processed = pyqtSignal(str)
    
    # Emitted with progress updates (0-100%)
    progress = pyqtSignal(int, str)
    
    # Emitted with log messages
    log_message = pyqtSignal(str)
    
    # Emitted when prediction completes successfully
    completed = pyqtSignal(dict)
    
    # Emitted on error
    error = pyqtSignal(str)


class Predictor(QThread):
    """Prediction thread that runs model.predict() in background."""
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.signals = PredictorSignals()
        
        self._model_path: str = ""
        self._source: str = ""  # Image/video file path or camera index
        self._conf: float = 0.25
        
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
    def conf(self) -> float:
        return self._conf
    
    @conf.setter
    def conf(self, value: float):
        self._conf = max(0.0, min(1.0, float(value)))
    
    @pyqtSlot()
    def start_predict(self):
        """Start prediction (slot for QThread)."""
        if not self._model_path:
            self.signals.error.emit("Model path is required")
            return
        
        if not self._source:
            self.signals.error.emit("Source is required")
            return
        
        self._running = True
        self.signals.started.emit()
        
        try:
            # Import ultralytics
            from ultralytics import YOLO
            
            # Load model
            self.signals.log_message.emit(f"Loading model: {self._model_path}")
            model = YOLO(self._model_path)
            
            # Predict
            results = model.predict(
                source=self._source,
                conf=self._conf,
                stream=True,
            )
            
            processed_count = 0
            
            for result in results:
                if not self._running:
                    break
                
                processed_count += 1
                progress = int((processed_count / 1) * 100)
                
                self.signals.image_processed.emit(str(result.path))
                self.signals.progress.emit(progress, f"Processed {processed_count} image(s)")
            
            self.signals.completed.emit({
                "images": processed_count,
            })
        
        except Exception as e:
            self.signals.error.emit(str(e))
    
    @pyqtSlot()
    def stop_predict(self):
        """Stop prediction."""
        self._running = False
        self.signals.log_message.emit("Prediction stopped by user")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def run(self):
        """Override QThread.run() to start prediction."""
        self.start_predict()


def create_predictor():
    """Factory function to create a predictor instance."""
    return Predictor()
