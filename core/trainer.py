"""Training wrapper with PyQt signals for real-time updates."""

import time
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️  Ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


class TrainerSignals(QObject):
    """PyQt signals for training events."""
    
    # Emitted when training starts
    started = pyqtSignal()
    
    # Emitted on each epoch completion with metrics
    epoch_complete = pyqtSignal(dict)
    
    # Emitted with progress updates (0-100%)
    progress = pyqtSignal(int, str)
    
    # Emitted with log messages
    log_message = pyqtSignal(str)
    
    # Emitted when training completes successfully
    completed = pyqtSignal(dict)
    
    # Emitted when training is paused
    paused = pyqtSignal()
    
    # Emitted when training stops
    stopped = pyqtSignal()
    
    # Emitted on error
    error = pyqtSignal(str)


class Trainer(QThread):
    """Training thread that runs YOLO.train() in background."""
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.signals = TrainerSignals()
        
        self._model_path: str = "yolo26s.pt"
        self._data_path: str = ""
        self._epochs: int = 100
        self._imgsz: int = 640
        self._batch_size: int = -1  # Auto
        self._device: Optional[str] = None
        self._lr0: float = 0.01
        self._optimizer: str = "auto"
        self._amp: bool = True
        self._patience: int = 100
        self._cache: str = "off"
        
        self._running: bool = False
        self._paused: bool = False
        self._results = None
    
    @property
    def model_path(self) -> str:
        return self._model_path
    
    @model_path.setter
    def model_path(self, value: str):
        self._model_path = value
    
    @property
    def data_path(self) -> str:
        return self._data_path
    
    @data_path.setter
    def data_path(self, value: str):
        self._data_path = value
    
    @property
    def epochs(self) -> int:
        return self._epochs
    
    @epochs.setter
    def epochs(self, value: int):
        self._epochs = max(1, int(value))
    
    @property
    def imgsz(self) -> int:
        return self._imgsz
    
    @imgsz.setter
    def imgsz(self, value: int):
        self._imgsz = value
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value if value > 0 else -1
    
    @property
    def device(self) -> Optional[str]:
        return self._device
    
    @device.setter
    def device(self, value: Optional[str]):
        self._device = value
    
    @property
    def lr0(self) -> float:
        return self._lr0
    
    @lr0.setter
    def lr0(self, value: float):
        self._lr0 = max(1e-7, float(value))
    
    @property
    def optimizer(self) -> str:
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, value: str):
        self._optimizer = value
    
    @property
    def amp(self) -> bool:
        return self._amp
    
    @amp.setter
    def amp(self, value: bool):
        self._amp = value
    
    @property
    def patience(self) -> int:
        return self._patience
    
    @patience.setter
    def patience(self, value: int):
        self._patience = max(0, int(value))
    
    @property
    def cache(self) -> str:
        return self._cache
    
    @cache.setter
    def cache(self, value: str):
        valid_caches = ("off", "ram", "disk")
        self._cache = value if value in valid_caches else "off"
    
    @pyqtSlot()
    def start_training(self):
        """Start training (slot for QThread)."""
        if YOLO is None:
            self.signals.error.emit("Ultralytics package not installed.")
            return
        
        self._running = True
        self._paused = False
        self.signals.started.emit()
        
        try:
            # Load model
            self.signals.log_message.emit(f"Loading model: {self._model_path}")
            model = YOLO(self._model_path)
            
            # Prepare arguments
            args = {
                "data": self._data_path,
                "epochs": self._epochs,
                "imgsz": self._imgsz,
                "batch": self._batch_size,
                "device": self._device or "",
                "lr0": self._lr0,
                "optimizer": self._optimizer,
                "amp": self._amp,
                "patience": self._patience,
            }
            
            # Cache configuration: ultralytics supports True (RAM) or 'disk'
            if self._cache == "ram":
                args["cache"] = True
            elif self._cache == "disk":
                args["cache"] = "disk"
            
            def on_train_epoch_end(trainer):
                """Callback called by Ultralytics after training/fit epochs."""
                if not self._running:
                    trainer.stop = True
                    return

                while self._paused and self._running:
                    time.sleep(0.2)

                if not self._running:
                    trainer.stop = True
                    return

                rd = getattr(trainer, "results_dict", {}) or {}
                metrics_dict = getattr(trainer, "metrics", {}) or {}
                loss_items = getattr(trainer, "loss_items", None)

                def metric_value(*names, default=0.0):
                    for source in (rd, metrics_dict):
                        for name in names:
                            if name in source:
                                try:
                                    return float(source[name])
                                except (TypeError, ValueError):
                                    return default
                    return default

                losses = []
                if loss_items is not None:
                    try:
                        losses = [float(x) for x in loss_items]
                    except TypeError:
                        losses = [float(loss_items)]

                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total_epochs = int(getattr(trainer, "epochs", self._epochs))
                metrics = {
                    "epoch": epoch,
                    "epochs": total_epochs,
                    "loss_box": metric_value("train/box_loss", "box_loss", default=losses[0] if len(losses) > 0 else 0.0),
                    "loss_cls": metric_value("train/cls_loss", "cls_loss", default=losses[1] if len(losses) > 1 else 0.0),
                    "loss_dfl": metric_value("train/dfl_loss", "dfl_loss", default=losses[2] if len(losses) > 2 else 0.0),
                    "metrics/mAP50(B)": metric_value("metrics/mAP50(B)", "metrics/mAP50"),
                    "metrics/mAP50-95(B)": metric_value("metrics/mAP50-95(B)", "metrics/mAP50-95"),
                    "metrics/precision(B)": metric_value("metrics/precision(B)", "metrics/precision"),
                    "metrics/recall(B)": metric_value("metrics/recall(B)", "metrics/recall"),
                }

                progress = int((epoch / total_epochs) * 100) if total_epochs else 0
                eta = (total_epochs - epoch) * 30  # Rough estimate
                self.signals.progress.emit(progress, f"ETA: {eta}s")
                self.signals.epoch_complete.emit(metrics)

                loss_str = f"L={metrics['loss_box']:.4f}"
                map50 = f"mAP50={metrics['metrics/mAP50(B)']:.4f}" if metrics["metrics/mAP50(B)"] > 0 else ""
                self.signals.log_message.emit(
                    f"Epoch {epoch}/{total_epochs} | {loss_str} | {map50}"
                )

            model.add_callback("on_fit_epoch_end", on_train_epoch_end)
            self._results = model.train(**args)
            
            if self._running:
                result_metrics = getattr(self._results, "metrics", {})
                save_dir = getattr(self._results, "save_dir", "")
                self.signals.completed.emit({
                    "metrics": result_metrics,
                    "save_dir": str(save_dir),
                })
                self._running = False
        
        except Exception as e:
            self._running = False
            self._paused = False
            self.signals.error.emit(str(e))
    
    @pyqtSlot()
    def pause_training(self):
        """Pause/resume training between Ultralytics callbacks."""
        self._paused = not self._paused
        if self._paused:
            self.signals.log_message.emit(
                "Training pause requested. It will pause after the current epoch step."
            )
            self.signals.paused.emit()
        else:
            self.signals.log_message.emit("Training resumed.")
            self.signals.started.emit()  # Resume
    
    @pyqtSlot()
    def stop_training(self):
        """Stop training."""
        self._running = False
        self._paused = False
        self.signals.stopped.emit()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_paused(self) -> bool:
        return self._paused
    
    def run(self):
        """Override QThread.run() to start training."""
        self.start_training()


def create_trainer():
    """Factory function to create a trainer instance."""
    return Trainer()
