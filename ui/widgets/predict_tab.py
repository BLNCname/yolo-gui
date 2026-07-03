"""Prediction/inference tab widget with image display."""

import os
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QDoubleSpinBox,
    QTextEdit, QGroupBox, QMessageBox, QFileDialog
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PredictionWorker(QThread):
    """Background thread for prediction/inference."""

    # Signals
    frame_image = pyqtSignal(QImage)  # Emitted with annotated frame
    log_message = pyqtSignal(str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path: str, source: str, conf: float, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source = source
        self.conf = conf
        self._running = True

    def run(self):
        """Run prediction in background thread."""
        try:
            if YOLO is None:
                self.error.emit("Ultralytics не установлен")
                return

            # Load model
            self.log_message.emit(f"Загрузка модели: {self.model_path}")
            model = YOLO(self.model_path)

            source_idx = self._get_source_type()

            if source_idx == 0:  # Image
                results = model.predict(source=self.source, conf=self.conf)
                for result in results:
                    annotated_frame = result.plot()
                    qimg = self._frame_to_qimage(annotated_frame)
                    self.frame_image.emit(qimg)
                    break

            elif source_idx == 1:  # Video
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    raise ValueError(f"Не удалось открыть видео: {self.source}")

                frame_count = 0
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(source=frame, conf=self.conf)
                    for result in results:
                        annotated_frame = result.plot()
                        qimg = self._frame_to_qimage(annotated_frame)
                        self.frame_image.emit(qimg)
                        break

                    frame_count += 1
                    if frame_count % 30 == 0:
                        self.log_message.emit(f"Обработано кадров: {frame_count}")

                cap.release()

            elif source_idx == 2:  # Camera
                cam_index = int(self.source) if self.source.isdigit() else 0
                cap = cv2.VideoCapture(cam_index)
                if not cap.isOpened():
                    raise ValueError(f"Не удалось открыть камеру {cam_index}")

                frame_count = 0
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(source=frame, conf=self.conf)
                    for result in results:
                        annotated_frame = result.plot()
                        qimg = self._frame_to_qimage(annotated_frame)
                        self.frame_image.emit(qimg)
                        break

                    frame_count += 1
                    if frame_count % 30 == 0:
                        self.log_message.emit(f"Камера: кадр {frame_count}")

                cap.release()

            self.completed.emit()

        except Exception as e:
            self.error.emit(str(e))

    def _get_source_type(self) -> int:
        """Determine source type from the source string."""
        # Check if it's a camera index
        if self.source.isdigit():
            return 2
        # Check file extensions
        ext = os.path.splitext(self.source)[1].lower()
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        if ext in video_exts:
            return 1
        elif ext in image_exts:
            return 0
        # Default to image
        return 0

    @staticmethod
    def _frame_to_qimage(frame):
        """Convert BGR numpy frame to QImage."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

    def stop(self):
        """Stop prediction."""
        self._running = False


class PredictTab(QWidget):
    """Prediction tab with configuration and image display."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self.prediction_worker = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Top section: Settings form
        settings_group = QGroupBox("Настройки инференса")
        settings_layout = QVBoxLayout(settings_group)

        form_layout = QFormLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_model_btn = QPushButton("Обзор...")
        browse_model_btn.clicked.connect(lambda: self._browse_file(self.model_edit, "PT files (*.pt)"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(browse_model_btn)
        form_layout.addRow("Модель:", model_layout)

        # Source selector
        source_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Изображение", "Видео", "Камера 0"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_edit = QLineEdit()
        browse_src_btn = QPushButton("Обзор...")
        browse_src_btn.clicked.connect(lambda: self._browse_file(
            self.source_edit, "Image files (*.png *.jpg *.jpeg);;Video files (*.mp4 *.avi *.mov)"
        ))

        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(browse_src_btn)
        form_layout.addRow("Источник:", source_layout)

        # Confidence threshold
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setDecimals(2)
        form_layout.addRow("Порог уверенности (conf):", self.conf_spin)

        settings_layout.addLayout(form_layout)

        # Buttons row
        buttons_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶️ Запустить")
        self.stop_btn = QPushButton("⏹ Стоп")

        for btn in [self.run_btn, self.stop_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #388E3C);
                    color: white;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #66BB6A, stop:1 #4CAF50);
                }
            """)

        self.stop_btn.setEnabled(False)

        buttons_layout.addWidget(self.run_btn)
        buttons_layout.addWidget(self.stop_btn)
        settings_layout.addLayout(buttons_layout)

        main_layout.addWidget(settings_group)

        # Image preview
        preview_group = QGroupBox("Результат инференса")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("Нет изображения")
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

        preview_layout.addWidget(self.preview_label)
        main_layout.addWidget(preview_group)

    def _connect_signals(self):
        """Connect signals to slots."""
        self.run_btn.clicked.connect(self._run_inference)
        self.stop_btn.clicked.connect(self._stop_inference)

    def _browse_file(self, line_edit: QLineEdit, filter_str: str = "All files (*)"):
        """Open file dialog to select a file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            filter_str
        )
        if path:
            line_edit.setText(path)

    def _on_source_changed(self, index: int):
        """Handle source type change."""
        if index == 2:  # Camera
            self.source_edit.setText("0")
        else:
            self.source_edit.clear()

    def _run_inference(self):
        """Run inference in background thread."""
        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите модель (.pt)"
            )
            return

        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, укажите источник"
            )
            return

        if not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Модель не найдена:\n{model_path}"
            )
            return

        # Validate source exists (skip for camera)
        if not source.isdigit():
            if not os.path.exists(source):
                QMessageBox.critical(
                    self,
                    "Ошибка",
                    f"Источник не найден:\n{source}"
                )
                return

        if YOLO is None:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Ultralytics не установлен. Установите: pip install ultralytics"
            )
            return

        # Stop any previous worker
        if self.prediction_worker and self.prediction_worker.isRunning():
            self.prediction_worker.stop()
            self.prediction_worker.wait(1000)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.preview_label.setText("Загрузка модели...")

        # Create and start worker thread
        self.prediction_worker = PredictionWorker(
            model_path=model_path,
            source=source,
            conf=self.conf_spin.value()
        )

        self.prediction_worker.frame_image.connect(self._on_frame_image)
        self.prediction_worker.log_message.connect(lambda msg: self.preview_label.setToolTip(msg))
        self.prediction_worker.completed.connect(self._on_prediction_completed)
        self.prediction_worker.error.connect(self._on_prediction_error)

        self.prediction_worker.start()

    def _stop_inference(self):
        """Stop inference."""
        if self.prediction_worker and self.prediction_worker.isRunning():
            self.prediction_worker.stop()
            self.prediction_worker.quit()
            self.prediction_worker.wait(1000)

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_label.setText("Инференс остановлен")

    @pyqtSlot(QImage)
    def _on_frame_image(self, qimage: QImage):
        """Display annotated frame in preview label."""
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)

    @pyqtSlot()
    def _on_prediction_completed(self):
        """Handle prediction completion."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @pyqtSlot(str)
    def _on_prediction_error(self, error: str):
        """Handle prediction error."""
        QMessageBox.critical(self, "Ошибка", f"Не удалось запустить инференс:\n{error}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def create_predict_tab(parent: QWidget = None) -> PredictTab:
    """Factory function to create a predict tab."""
    return PredictTab(parent)
