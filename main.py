"""UltraGUI — YOLO26s Training & Inference Desktop Application."""

import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont

from ui.main_window import MainWindow


def main():
    """Entry point for the UltraGUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("UltraGUI")
    app.setOrganizationName("NousResearch")

    # Check if running in WSL with X11 display
    if not app.primaryScreen():
        QMessageBox.warning(
            None,
            "Предупреждение",
            "Нет отображения. Убедитесь, что WSLg запущен.\n\n"
            "Для запуска в Windows:\n"
            "  wsl --update\n"
            "  wsl ~\n"
        )

    # Load dark theme stylesheet
    from pathlib import Path
    style_path = Path(__file__).parent / "resources" / "style.qss"
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
