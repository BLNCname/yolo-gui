"""Training tab widget with all controls and monitoring."""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QGroupBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSlot

import torch
from core.trainer import create_trainer
from .metrics_chart import MetricsChartWidget
from .log_viewer import LogViewer
from utils.helpers import validate_path


class TrainTab(QWidget):
    """Training tab with configuration and monitoring."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self._setup_ui()
        self._initialize_trainer()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top section: Settings form
        settings_group = QGroupBox("Настройки обучения")
        settings_layout = QVBoxLayout(settings_group)
        
        form_layout = QFormLayout()
        
        # Model selector
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", 
            "yolo26l.pt", "yolo26x.pt"
        ])
        form_layout.addRow("Модель:", self.model_combo)
        
        # Dataset path
        dataset_layout = QHBoxLayout()
        self.dataset_edit = QLineEdit()
        browse_btn = QPushButton("Обзор")
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_layout.addWidget(self.dataset_edit)
        dataset_layout.addWidget(browse_btn)
        form_layout.addRow("Датасет (YAML):", dataset_layout)
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(100)
        form_layout.addRow("Эпохи:", self.epochs_spin)
        
        # Image size
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "480", "640", "960"])
        self.imgsz_combo.setCurrentText("640")
        form_layout.addRow("Размер изображения:", self.imgsz_combo)
        
        # Batch size
        batch_layout = QHBoxLayout()
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(256)
        self.batch_spin.setValue(16)
        self.auto_batch_check = QCheckBox("Авто")
        self.auto_batch_check.stateChanged.connect(self._toggle_auto_batch)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addWidget(self.auto_batch_check)
        form_layout.addRow("Batch size:", batch_layout)
        
        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto-detect", None)
        self.device_combo.addItem("CPU", "cpu")
        
        # Add available CUDA devices if available
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    self.device_combo.addItem(f"CUDA:{i} ({device_name})", f"cuda:{i}")
        except Exception:
            pass  # GPU not available or error occurred
        form_layout.addRow("Устройство:", self.device_combo)
        
        # Learning rate
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-7, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(6)
        form_layout.addRow("Learning rate (lr0):", self.lr_spin)
        
        # Optimizer
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "MuSGD", "AdamW", "SGD"])
        form_layout.addRow("Оптимизатор:", self.optimizer_combo)
        
        # AMP
        self.amp_check = QCheckBox()
        self.amp_check.setChecked(True)
        form_layout.addRow("AMP (смешанная точность):", self.amp_check)
        
        # Early stopping
        self.patience_spin = QSpinBox()
        self.patience_spin.setMinimum(0)
        self.patience_spin.setMaximum(500)
        self.patience_spin.setValue(100)
        form_layout.addRow("Early stopping (patience):", self.patience_spin)
        
        # Cache
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["Off", "RAM", "Disk"])
        form_layout.addRow("Кэширование данных:", self.cache_combo)
        
        settings_layout.addLayout(form_layout)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶️ Start")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        
        for btn in [self.start_btn, self.pause_btn, self.stop_btn]:
            btn.setStyleSheet("""
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
                QPushButton:pressed {
                    background-color: #555;
                }
            """)
        
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.pause_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addStretch()
        
        settings_layout.addLayout(buttons_layout)
        
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
                    stop:0 #4CAF50, stop:1 #8BC34A);
                border-radius: 3px;
            }
        """)
        settings_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(settings_group)
        
        # Middle section: Monitoring
        monitoring_group = QGroupBox("Мониторинг")
        monitoring_layout = QVBoxLayout(monitoring_group)
        
        # Metrics chart
        self.metrics_chart = MetricsChartWidget()
        monitoring_layout.addWidget(self.metrics_chart)
        
        main_layout.addWidget(monitoring_group)
        
        # Bottom section: Log viewer
        log_group = QGroupBox("Лог обучения")
        log_layout = QVBoxLayout(log_group)
        
        self.log_viewer = LogViewer()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFontFamily("Courier New")
        self.log_viewer.setFontPointSize(9)
        self.log_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                min-height: 150px;
            }
        """)
        
        log_layout.addWidget(self.log_viewer)
        main_layout.addWidget(log_group)
    
    def _initialize_trainer(self):
        """Initialize the trainer instance."""
        self.trainer = create_trainer()
    
    def _connect_signals(self):
        """Connect signals to slots."""
        # Trainer signals
        self.trainer.signals.started.connect(self._on_training_started)
        self.trainer.signals.epoch_complete.connect(self._on_epoch_complete)
        self.trainer.signals.progress.connect(self._on_progress_update)
        self.trainer.signals.log_message.connect(self._on_log_message)
        self.trainer.signals.completed.connect(self._on_training_completed)
        self.trainer.signals.paused.connect(self._on_training_paused)
        self.trainer.signals.stopped.connect(self._on_training_stopped)
        self.trainer.signals.error.connect(self._on_error)
        
        # Button connections
        self.start_btn.clicked.connect(self._start_training)
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop_training)
    
    def _browse_dataset(self):
        """Open file dialog to select dataset YAML."""
        from PyQt6.QtWidgets import QFileDialog
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл датасета (YAML)",
            "",
            "YAML files (*.yaml *.yml)"
        )
        
        if path:
            self.dataset_edit.setText(path)
    
    def _toggle_auto_batch(self, state: int):
        """Toggle auto batch size."""
        self.batch_spin.setEnabled(state == 0)  # Qt.Unchecked
    
    def _start_training(self):
        """Start training."""
        # Update trainer settings from UI
        self.trainer.model_path = self.model_combo.currentText()
        self.trainer.data_path = self.dataset_edit.text()
        self.trainer.epochs = self.epochs_spin.value()
        self.trainer.imgsz = int(self.imgsz_combo.currentText())
        
        if self.auto_batch_check.isChecked():
            self.trainer.batch_size = -1
        else:
            self.trainer.batch_size = self.batch_spin.value()
        
        device_idx = self.device_combo.currentIndex()
        self.trainer.device = self.device_combo.itemData(device_idx)
        
        self.trainer.lr0 = self.lr_spin.value()
        self.trainer.optimizer = self.optimizer_combo.currentText()
        self.trainer.amp = self.amp_check.isChecked()
        self.trainer.patience = self.patience_spin.value()
        
        cache_text = self.cache_combo.currentText().lower()
        self.trainer.cache = "ram" if cache_text == "ram" else ("disk" if cache_text == "disk" else "off")
        
        # Validate dataset path
        if not self.trainer.data_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите файл датасета (YAML)"
            )
            return
        
        # Validate YAML file exists and has correct extension
        is_valid, error = validate_path(self.trainer.data_path)
        if not is_valid:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Ошибка",
                f"Путь к датасету некорректен:\n{error}"
            )
            return
        
        # Check YAML extension
        if not self.trainer.data_path.lower().endswith(('.yaml', '.yml')):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Ошибка",
                "Файл датасета должен иметь расширение .yaml или .yml"
            )
            return
        
        # Reset UI state
        self.metrics_chart.reset()
        self.log_viewer.clear_logs()
        self.progress_bar.setValue(0)
        
        # Start trainer thread
        self.trainer.start()
    
    def _toggle_pause(self):
        """Toggle pause/resume training."""
        if self.trainer.is_paused:
            # Resume training
            self.trainer.pause_training()
            self.log_viewer.append_log("Training resumed...")
        else:
            # Pause training
            self.trainer.pause_training()
            self.log_viewer.append_log("Training paused...")
    
    def _stop_training(self):
        """Stop training."""
        self.trainer.stop_training()
    
    @pyqtSlot()
    def _on_training_started(self):
        """Handle training start."""
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.log_viewer.append_log("Training started...")
    
    @pyqtSlot(dict)
    def _on_epoch_complete(self, metrics: dict):
        """Handle epoch completion with metrics."""
        self.metrics_chart.update_metrics(metrics)
    
    @pyqtSlot(int, str)
    def _on_progress_update(self, progress: int, eta: str):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def _on_log_message(self, message: str):
        """Handle log message."""
        self.log_viewer.append_log(message)
    
    @pyqtSlot(dict)
    def _on_training_completed(self, results: dict):
        """Handle training completion."""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        self.log_viewer.append_log("Training completed successfully!")
        self.log_viewer.append_log(f"Results saved to: {results.get('save_dir', 'N/A')}")
    
    @pyqtSlot()
    def _on_training_paused(self):
        """Handle training pause."""
        if self.trainer.is_paused:
            self.pause_btn.setText("▶️ Resume")
            self.log_viewer.append_log("Training paused...")
        else:
            self.pause_btn.setText("⏸ Pause")
            self.log_viewer.append_log("Training resumed...")
    
    @pyqtSlot()
    def _on_training_stopped(self):
        """Handle training stop."""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.log_viewer.append_log("Training stopped.")
    
    @pyqtSlot(str)
    def _on_error(self, error: str):
        """Handle error."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Ошибка", f"Ошибка обучения:\n{error}")
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
