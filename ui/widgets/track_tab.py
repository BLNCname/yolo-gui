"""Tracking tab widget with video preview and controls."""

import os
import time
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class TrackingWorker(QThread):
    """Background thread for video tracking."""

    frame_processed = pyqtSignal(int)
    fps_updated = pyqtSignal(float)
    tracks_count = pyqtSignal(int)
    log_message = pyqtSignal(str)
    completed = pyqtSignal()
    error = pyqtSignal(str)
    frame_image = pyqtSignal(QImage)

    def __init__(
        self,
        model_path: str,
        source: str,
        tracker_type: str,
        conf: float,
        iou: float,
        output_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.model_path = model_path
        self.source = source
        self.tracker_type = tracker_type
        self.conf = conf
        self.iou = iou
        self.output_path = output_path
        self._running = True

    def run(self):
        """Run tracking in a background thread."""
        writer = None
        try:
            if YOLO is None:
                self.error.emit("Ultralytics is not installed")
                return

            self.log_message.emit(f"Loading model: {self.model_path}")
            model = YOLO(self.model_path)

            frame_count = 0
            start_time = time.time()
            results = model.track(
                source=self.source,
                tracker=f"{self.tracker_type}.yaml",
                conf=self.conf,
                iou=self.iou,
                stream=True,
            )

            for result in results:
                if not self._running:
                    break

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                self.fps_updated.emit(fps)

                try:
                    track_ids = result.boxes.id
                    count = len(set(track_ids.cpu().numpy().tolist())) if track_ids is not None else 0
                    self.tracks_count.emit(count)
                except Exception:
                    self.tracks_count.emit(0)

                annotated_frame = result.plot()
                if annotated_frame is not None:
                    if self.output_path:
                        writer = self._ensure_writer(writer, annotated_frame, fps)
                        writer.write(annotated_frame)

                    self.frame_image.emit(self._frame_to_qimage(annotated_frame))

                self.frame_processed.emit(frame_count)
                if frame_count % 30 == 0:
                    self.log_message.emit(f"Processed frames: {frame_count}")

            if writer is not None:
                writer.release()
                writer = None
                self.log_message.emit(f"Saved video: {self.output_path}")

            self.completed.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if writer is not None:
                writer.release()

    def _ensure_writer(self, writer, frame, fps):
        """Create a video writer lazily from the first annotated frame."""
        if writer is not None:
            return writer

        height, width = frame.shape[:2]
        fps_out = max(float(fps), 1.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps_out, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open output video: {self.output_path}")
        return writer

    @staticmethod
    def _frame_to_qimage(frame):
        """Convert a BGR numpy frame to a detached QImage."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    def stop(self):
        """Request tracking stop."""
        self._running = False


class TrackTab(QWidget):
    """Tracking tab with configuration and video preview."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tracking_worker = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        settings_group = QGroupBox("Tracking settings")
        settings_layout = QVBoxLayout(settings_group)
        form_layout = QFormLayout()

        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(lambda: self._browse_file(self.model_edit, "PT files (*.pt)"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(browse_model_btn)
        form_layout.addRow("Model (best.pt):", model_layout)

        source_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Video file", "Camera 0", "Stream URL"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_edit = QLineEdit()
        browse_video_btn = QPushButton("Browse...")
        browse_video_btn.clicked.connect(
            lambda: self._browse_file(self.source_edit, "Video files (*.mp4 *.avi *.mov *.mkv)")
        )
        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(browse_video_btn)
        form_layout.addRow("Source:", source_layout)

        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["BoT-SORT", "ByteTrack"])
        form_layout.addRow("Tracker:", self.tracker_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setDecimals(2)
        form_layout.addRow("Confidence threshold (conf):", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setValue(0.3)
        self.iou_spin.setDecimals(2)
        form_layout.addRow("Tracking IoU threshold:", self.iou_spin)

        settings_layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.save_btn = QPushButton("Save video")
        self.save_btn.setCheckable(True)
        self.stop_btn.setEnabled(False)

        for btn in [self.start_btn, self.stop_btn, self.save_btn]:
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #444;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #3d3d3d;
                }
                """
            )

        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.save_btn)
        settings_layout.addLayout(buttons_layout)
        main_layout.addWidget(settings_group)

        preview_group = QGroupBox("Video preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No video")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet(
            """
            QLabel {
                background-color: #1a1a1a;
                color: #666;
                border: 1px solid #333;
                border-radius: 4px;
            }
            """
        )
        preview_layout.addWidget(self.preview_label)
        main_layout.addWidget(preview_group)

        stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout(stats_group)
        self.fps_label = QLabel("FPS: -")
        self.tracks_label = QLabel("Tracks: -")
        for label in [self.fps_label, self.tracks_label]:
            label.setStyleSheet(
                """
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    padding: 4px 8px;
                    background-color: #1a1a1a;
                    border-radius: 3px;
                }
                """
            )
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.tracks_label)
        main_layout.addWidget(stats_group)

    def _connect_signals(self):
        """Connect UI signals."""
        self.start_btn.clicked.connect(self._start_tracking)
        self.stop_btn.clicked.connect(self._stop_tracking)

    def _browse_file(self, line_edit: QLineEdit, filter_str: str = "All files (*)"):
        """Open a file dialog and put the selected path into the line edit."""
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter_str)
        if path:
            line_edit.setText(path)

    def _on_source_changed(self, index: int):
        """Handle source type changes."""
        if index == 1:
            self.source_edit.setText("0")
        else:
            self.source_edit.clear()

    def _start_tracking(self):
        """Start tracking."""
        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Error", "Please select a model (.pt)")
            return

        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Error", "Please specify a video source")
            return

        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model not found:\n{model_path}")
            return

        is_stream = source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))
        if not source.isdigit() and not is_stream and not os.path.exists(source):
            QMessageBox.critical(self, "Error", f"Video source not found:\n{source}")
            return

        if YOLO is None:
            QMessageBox.critical(self, "Error", "Ultralytics is not installed. Install it with: pip install ultralytics")
            return

        output_path = ""
        if self.save_btn.isChecked():
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save tracked video",
                "tracked_output.mp4",
                "MP4 video (*.mp4);;AVI video (*.avi)",
            )
            if not output_path:
                return

        tracker_text = self.tracker_combo.currentText().lower()
        tracker_type = "bytetrack" if "byte" in tracker_text else "botsort"

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.preview_label.setText("Starting tracking...")
        self.fps_label.setText("FPS: -")
        self.tracks_label.setText("Tracks: -")

        self.tracking_worker = TrackingWorker(
            model_path=model_path,
            source=source,
            tracker_type=tracker_type,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            output_path=output_path,
        )
        self.tracking_worker.frame_processed.connect(self._on_frame_processed)
        self.tracking_worker.fps_updated.connect(lambda fps: self.fps_label.setText(f"FPS: {fps:.1f}"))
        self.tracking_worker.tracks_count.connect(lambda n: self.tracks_label.setText(f"Tracks: {n}"))
        self.tracking_worker.frame_image.connect(self._on_frame_image)
        self.tracking_worker.log_message.connect(lambda msg: self.preview_label.setToolTip(msg))
        self.tracking_worker.completed.connect(self._on_tracking_completed)
        self.tracking_worker.error.connect(self._on_tracking_error)
        self.tracking_worker.start()

    def _stop_tracking(self):
        """Stop tracking."""
        if self.tracking_worker is not None:
            self.tracking_worker.stop()
            self.tracking_worker.wait(3000)

        self.preview_label.setText("Tracking stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(int)
    def _on_frame_processed(self, frame_count: int):
        """Handle frame processed signal."""
        self.preview_label.setToolTip(f"Frames processed: {frame_count}")

    @pyqtSlot(QImage)
    def _on_frame_image(self, qimage: QImage):
        """Display an annotated frame."""
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled_pixmap)

    @pyqtSlot()
    def _on_tracking_completed(self):
        """Handle tracking completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_label.setToolTip("Tracking completed")

    @pyqtSlot(str)
    def _on_tracking_error(self, error: str):
        """Handle tracking error."""
        QMessageBox.critical(self, "Tracking error", f"An error occurred:\n{error}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def create_track_tab(parent: Optional[QWidget] = None) -> TrackTab:
    """Factory function to create a track tab."""
    return TrackTab(parent)
