"""Model export module for ONNX, TensorRT, and other formats."""

import sys
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class ExporterSignals(QObject):
    """PyQt signals for export events."""

    # Emitted when export starts
    started = pyqtSignal()

    # Emitted with progress updates (0-100%)
    progress = pyqtSignal(int, str)

    # Emitted with log messages
    log_message = pyqtSignal(str)

    # Emitted when export completes successfully
    completed = pyqtSignal(dict)

    # Emitted on error
    error = pyqtSignal(str)


class Exporter(QThread):
    """Export thread that runs model.export() in background."""

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.signals = ExporterSignals()

        self._model_path: str = ""
        self._format: str = "onnx"
        self._half: bool = True
        self._dynamic: bool = False

        self._running: bool = False

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, value: str):
        self._model_path = value

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, value: str):
        valid_formats = ("onnx", "engine", "coreml", "tflite", "pb", "tfjs", "paddle", "ncnn")
        self._format = value if value in valid_formats else "onnx"

    @property
    def half(self) -> bool:
        return self._half

    @half.setter
    def half(self, value: bool):
        self._half = value

    @property
    def dynamic(self) -> bool:
        return self._dynamic

    @dynamic.setter
    def dynamic(self, value: bool):
        self._dynamic = value

    @pyqtSlot()
    def start_export(self):
        """Start export (slot for QThread)."""
        if not self._model_path:
            self.signals.error.emit("Model path is required")
            return

        self._running = True
        self.signals.started.emit()

        try:
            # Import ultralytics
            from ultralytics import YOLO

            # Load model
            self.signals.log_message.emit(f"Loading model: {self._model_path}")
            self.signals.progress.emit(10, "Loading model...")
            model = YOLO(self._model_path)

            # Export with appropriate parameters
            args = {
                "format": self._format,
            }

            if self._half:
                args["half"] = True

            if self._dynamic and self._format == "onnx":
                args["dynamic"] = True

            self.signals.log_message.emit(f"Exporting to {self._format}...")
            self.signals.progress.emit(30, f"Exporting to {self._format}...")

            # Export model
            result = model.export(**args)

            self.signals.progress.emit(100, "Export complete!")
            self.signals.completed.emit({
                "path": str(result),
                "format": self._format,
            })

        except Exception as e:
            self.signals.error.emit(str(e))

    @pyqtSlot()
    def stop_export(self):
        """Stop export."""
        self._running = False
        self.signals.log_message.emit("Export stopped by user")

    @property
    def is_running(self) -> bool:
        return self._running

    def run(self):
        """Override QThread.run() to start export."""
        self.start_export()


def create_exporter():
    """Factory function to create an exporter instance."""
    return Exporter()
