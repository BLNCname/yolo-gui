"""Main window with tabbed interface."""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt

import torch
from core.device import get_device_name


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UltraGUI - YOLO26s Training & Inference")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Layout
        main_layout = QVBoxLayout(central)

        # Device status bar (top of window)
        device_status = QHBoxLayout()
        device_label = QLabel("Устройство:")
        device_value = QLabel(get_device_name())
        device_value.setStyleSheet(
            "QLabel { color: #4CAF50; font-weight: bold; padding: 4px 8px; "
            "background-color: #1a1a1a; border-radius: 3px; }"
        )

        # Add GPU warning if not available
        if not torch.cuda.is_available():
            device_value.setStyleSheet(
                "QLabel { color: #FF9800; font-weight: bold; padding: 4px 8px; "
                "background-color: #1a1a1a; border-radius: 3px; }"
            )

        device_status.addWidget(device_label)
        device_status.addWidget(device_value)
        device_status.addStretch()

        main_layout.addLayout(device_status)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Add actual tabs
        self._add_tabs()

        # Show GPU warning if needed
        self._check_gpu_warning()

    def _add_tabs(self):
        """Add actual tab widgets."""
        from ui.widgets.train_tab import TrainTab
        from ui.widgets.export_tab import ExportTab
        from ui.widgets.track_tab import TrackTab
        from ui.widgets.predict_tab import PredictTab
        from ui.widgets.dataset_tab import DatasetTab

        # Tab 1: Train (actual implementation)
        train_tab = TrainTab()
        self.tabs.addTab(train_tab, "🎯 Обучение")

        # Tab 2: Track (actual implementation with video preview)
        track_tab = TrackTab()
        self.tabs.addTab(track_tab, "🎬 Трекинг")

        # Tab 3: Predict (actual implementation with image display)
        predict_tab = PredictTab()
        self.tabs.addTab(predict_tab, "🖼️ Инференс")

        # Tab 4: Export (actual implementation with progress bar)
        export_tab = ExportTab()
        self.tabs.addTab(export_tab, "📦 Экспорт")

        # Tab 5: Dataset Manager (actual implementation with wizard)
        dataset_tab = DatasetTab()
        self.tabs.addTab(dataset_tab, "⚙️ Датасет")

    def _check_gpu_warning(self):
        """Show warning if GPU is not available."""
        if not torch.cuda.is_available():
            QMessageBox.warning(
                self,
                "GPU не обнаружен",
                "Обучение будет выполняться на CPU (медленно).\n\n"
                "Для ускорения обучения с AMD GPU через ROCm:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1\n\n"
                "Для NVIDIA GPU:\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            )


def main():
    """Entry point for testing the main window."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Check if running in WSL with X11 display
    if not QApplication.instance().primaryScreen():
        print("⚠️  Предупреждение: Нет отображения. Убедитесь, что WSLg запущен.")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
