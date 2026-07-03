"""Export tab widget with model export controls."""

import os
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox,
    QTextEdit, QGroupBox, QProgressBar, QMessageBox, QFileDialog
)

from core.exporter import create_exporter
from ui.widgets.log_viewer import LogViewer


class ExportTab(QWidget):
    """Export tab with configuration and progress monitoring."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self._setup_ui()
        self._initialize_exporter()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Top section: Settings form
        settings_group = QGroupBox("Настройки экспорта")
        settings_layout = QVBoxLayout(settings_group)

        form_layout = QFormLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        browse_model_btn = QPushButton("Обзор...")
        browse_model_btn.clicked.connect(lambda: self._browse_file(self.model_edit, "PT files (*.pt)"))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(browse_model_btn)
        form_layout.addRow("Модель (.pt):", model_layout)

        # Export format
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "onnx",
            "engine",      # TensorRT
            "coreml",      # Apple CoreML
            "tflite",      # TensorFlow Lite
            "pb",          # TensorFlow protobuf
            "tfjs",        # TensorFlow.js
            "paddle",      # PaddlePaddle
            "ncnn",        # NCNN
        ])
        form_layout.addRow("Формат экспорта:", self.format_combo)

        # Half precision (FP16)
        self.half_check = QCheckBox()
        self.half_check.setChecked(True)
        form_layout.addRow("Half precision (FP16):", self.half_check)

        # Dynamic shapes (ONNX only)
        self.dynamic_check = QCheckBox()
        self.dynamic_check.setChecked(False)
        form_layout.addRow("Dynamic shapes (ONNX):", self.dynamic_check)

        settings_layout.addLayout(form_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF9800, stop:1 #FFC107);
                border-radius: 3px;
            }
        """)
        settings_layout.addWidget(self.progress_bar)

        # Buttons row
        buttons_layout = QHBoxLayout()
        self.export_btn = QPushButton("📤 Экспортировать")
        self.stop_btn = QPushButton("⏹ Стоп")

        for btn in [self.export_btn, self.stop_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FF9800, stop:1 #F57C00);
                    color: white;
                    padding: 8px 20px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FFB74D, stop:1 #FF9800);
                }
            """)

        self.stop_btn.setEnabled(False)

        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addWidget(self.stop_btn)
        settings_layout.addLayout(buttons_layout)

        main_layout.addWidget(settings_group)

        # Log viewer — use consistent LogViewer widget
        log_group = QGroupBox("Лог экспорта")
        log_layout = QVBoxLayout(log_group)

        self.log_viewer = LogViewer()
        self.log_viewer.setMaximumHeight(150)
        self.log_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #FFC107;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
            }
        """)

        log_layout.addWidget(self.log_viewer)
        main_layout.addWidget(log_group)

    def _initialize_exporter(self):
        """Initialize the exporter instance."""
        self.exporter = create_exporter()

    def _connect_signals(self):
        """Connect signals to slots."""
        # Exporter signals
        self.exporter.signals.started.connect(self._on_export_started)
        self.exporter.signals.progress.connect(self._on_progress_update)
        self.exporter.signals.log_message.connect(self._on_log_message)
        self.exporter.signals.completed.connect(self._on_export_completed)
        self.exporter.signals.error.connect(self._on_error)

        # Button connections
        self.export_btn.clicked.connect(self._start_export)
        self.stop_btn.clicked.connect(self._stop_export)

    def _browse_file(self, line_edit: QLineEdit, filter_str: str = "PT files (*.pt)"):
        """Open file dialog to select a model file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите модель",
            "",
            filter_str
        )
        if path:
            line_edit.setText(path)

    def _start_export(self):
        """Start export."""
        model_path = self.model_edit.text().strip()
        if not model_path:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите модель (.pt)"
            )
            return

        # Validate model exists
        if not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Модель не найдена:\n{model_path}"
            )
            return

        # Update exporter settings from UI
        self.exporter.model_path = model_path
        self.exporter.format = self.format_combo.currentText()
        self.exporter.half = self.half_check.isChecked()
        self.exporter.dynamic = self.dynamic_check.isChecked()

        # Reset UI state
        self.progress_bar.setValue(0)
        self.log_viewer.clear_logs()

        # Start exporter thread
        self.exporter.start()

    def _stop_export(self):
        """Stop export."""
        self.exporter.stop_export()

    @pyqtSlot()
    def _on_export_started(self):
        """Handle export start."""
        self.export_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_viewer.append_log("Export started...")

    @pyqtSlot(int, str)
    def _on_progress_update(self, progress: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(progress)

    @pyqtSlot(str)
    def _on_log_message(self, message: str):
        """Handle log message."""
        self.log_viewer.append_log(message)

    @pyqtSlot(dict)
    def _on_export_completed(self, results: dict):
        """Handle export completion."""
        self.export_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.log_viewer.append_log("Export completed successfully!")
        self.log_viewer.append_log(f"Exported model saved to:\n{results.get('path', 'N/A')}")
        self.log_viewer.append_log(f"Format: {results.get('format', 'N/A')}")

    @pyqtSlot(str)
    def _on_error(self, error: str):
        """Handle error."""
        QMessageBox.critical(self, "Ошибка", f"Ошибка экспорта:\n{error}")

        self.export_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def create_export_tab(parent: QWidget = None) -> ExportTab:
    """Factory function to create an export tab."""
    return ExportTab(parent)
