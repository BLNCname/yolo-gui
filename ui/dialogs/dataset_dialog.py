"""Dataset configuration dialogs for UltraGUI."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QTextEdit, QSlider, QLabel,
    QFileDialog, QMessageBox, QWidget, QTabWidget, QGroupBox
)


class DatasetCreationDialog(QDialog):
    """Dialog for creating new YOLO dataset configuration."""

    # Signal emitted when dataset is successfully created
    dataset_created = pyqtSignal(str)  # path to generated YAML

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Мастер создания датасета")
        self.setMinimumWidth(600)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        main_layout = QVBoxLayout(self)

        # Tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Tab 1: Dataset Creation
        creation_tab = QWidget()
        creation_layout = QVBoxLayout(creation_tab)
        creation_layout.setContentsMargins(10, 10, 10, 10)

        # Form layout for settings
        form_layout = QFormLayout()

        # Classes input (one per line)
        classes_label = QLabel("Классы (по одному на строку):")
        self.classes_textedit = QTextEdit()
        self.classes_textedit.setPlaceholderText("cat\ndog\ncar\nperson")
        self.classes_textedit.setMaximumHeight(100)
        form_layout.addRow(classes_label, self.classes_textedit)

        # Dataset folder
        dataset_folder_layout = QHBoxLayout()
        self.dataset_folder_edit = QLineEdit()
        browse_btn = QPushButton("Обзор...")
        browse_btn.clicked.connect(self._browse_dataset_folder)
        dataset_folder_layout.addWidget(self.dataset_folder_edit)
        dataset_folder_layout.addWidget(browse_btn)
        form_layout.addRow("Папка с датасетом:", dataset_folder_layout)

        # Train/val split slider
        split_layout = QHBoxLayout()
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setMinimum(10)
        self.split_slider.setMaximum(90)
        self.split_slider.setValue(80)  # Default 80/20
        self.split_slider.valueChanged.connect(self._update_split_labels)

        split_value_label = QLabel("80%")
        self.val_percent_label = QLabel("20%")

        self.split_slider.valueChanged.connect(
            lambda v: [split_value_label.setText(f"{v}%"), self.val_percent_label.setText(f"{100 - v}%")]
        )

        split_layout.addWidget(QLabel("Train/Val split:"))
        split_layout.addWidget(self.split_slider)
        split_layout.addWidget(split_value_label)
        split_layout.addWidget(self.val_percent_label)

        form_layout.addRow("", split_layout)

        # Output folder
        output_folder_layout = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        output_browse_btn = QPushButton("Обзор...")
        output_browse_btn.clicked.connect(self._browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(output_browse_btn)
        form_layout.addRow("Папка для YAML:", output_folder_layout)

        # Dataset name
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setText("dataset")
        form_layout.addRow("Имя датасета:", self.dataset_name_edit)

        creation_layout.addLayout(form_layout)

        # Create button
        create_btn = QPushButton("🗑️ Создать структуру и YAML")
        create_btn.clicked.connect(self._create_dataset)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #388E3C);
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #66BB6A, stop:1 #4CAF50);
            }
        """)
        creation_layout.addWidget(create_btn)

        tabs.addTab(creation_tab, "📝 Создание")

        # Tab 2: Validation
        validation_tab = QWidget()
        validation_layout = QVBoxLayout(validation_tab)
        validation_layout.setContentsMargins(10, 10, 10, 10)

        yaml_path_layout = QHBoxLayout()
        self.yaml_path_edit = QLineEdit()
        yaml_browse_btn = QPushButton("Обзор...")
        yaml_browse_btn.clicked.connect(self._browse_yaml_file)
        yaml_path_layout.addWidget(self.yaml_path_edit)
        yaml_path_layout.addWidget(yaml_browse_btn)
        validation_form = QFormLayout()
        validation_form.addRow("YAML файл:", yaml_path_layout)
        validation_layout.addLayout(validation_form)

        validate_btn = QPushButton("✅ Проверить датасет")
        validate_btn.clicked.connect(self._validate_dataset)
        validate_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
        """)
        validation_layout.addWidget(validate_btn)

        self.validation_result = QTextEdit()
        self.validation_result.setReadOnly(True)
        self.validation_result.setMaximumHeight(200)
        self.validation_result.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333;
                padding: 8px;
            }
        """)
        validation_layout.addWidget(QLabel("Результат проверки:"))
        validation_layout.addWidget(self.validation_result)

        tabs.addTab(validation_tab, "🔍 Проверка")

        # Tab 3: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        stats_layout.setContentsMargins(10, 10, 10, 10)

        yaml_stats_layout = QHBoxLayout()
        self.yaml_stats_edit = QLineEdit()
        yaml_stats_browse_btn = QPushButton("Обзор...")
        yaml_stats_browse_btn.clicked.connect(self._browse_yaml_for_stats)
        yaml_stats_layout.addWidget(self.yaml_stats_edit)
        yaml_stats_layout.addWidget(yaml_stats_browse_btn)
        stats_form = QFormLayout()
        stats_form.addRow("YAML файл:", yaml_stats_layout)
        stats_layout.addLayout(stats_form)

        stats_btn = QPushButton("📊 Показать статистику")
        stats_btn.clicked.connect(self._show_statistics)
        stats_layout.addWidget(stats_btn)

        self.stats_result = QTextEdit()
        self.stats_result.setReadOnly(True)
        self.stats_result.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #333;
                padding: 8px;
            }
        """)
        stats_layout.addWidget(QLabel("Статистика:"))
        stats_layout.addWidget(self.stats_result)

        tabs.addTab(stats_tab, "📈 Статистика")

    def _browse_dataset_folder(self) -> None:
        """Browse for dataset folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с датасетом",
            ""
        )
        if folder:
            self.dataset_folder_edit.setText(folder)

    def _browse_output_folder(self) -> None:
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения YAML",
            ""
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def _browse_yaml_file(self) -> None:
        """Browse for existing YAML file to validate."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите YAML файл",
            "",
            "YAML files (*.yaml *.yml)"
        )
        if path:
            self.yaml_path_edit.setText(path)

    def _browse_yaml_for_stats(self) -> None:
        """Browse for YAML file to show statistics."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите YAML файл",
            "",
            "YAML files (*.yaml *.yml)"
        )
        if path:
            self.yaml_stats_edit.setText(path)

    def _update_split_labels(self, value: int) -> None:
        """Update split percentage labels."""
        pass  # Connected in _setup_ui

    def _create_dataset(self) -> None:
        """Create dataset structure and YAML file."""
        classes_text = self.classes_textedit.toPlainText().strip()
        if not classes_text:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, укажите классы (по одному на строку)"
            )
            return

        classes = [c.strip() for c in classes_text.split('\n') if c.strip()]
        if len(classes) == 0:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Не найдены классы"
            )
            return

        dataset_folder = self.dataset_folder_edit.text().strip()
        if not dataset_folder:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите папку с датасетом"
            )
            return

        output_folder = self.output_folder_edit.text().strip()
        if not output_folder:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите папку для сохранения YAML"
            )
            return

        dataset_name = self.dataset_name_edit.text().strip()
        if not dataset_name:
            dataset_name = "dataset"

        train_split = self.split_slider.value()
        val_split = 100 - train_split

        # Create directory structure
        try:
            base_path = Path(dataset_folder)
            output_path = Path(output_folder)

            # Create standard YOLO structure
            for split in ['train', 'val']:
                (base_path / f'images/{split}').mkdir(parents=True, exist_ok=True)
                (base_path / f'labels/{split}').mkdir(parents=True, exist_ok=True)

            # Generate YAML file
            yaml_content = self._generate_yaml(
                classes=classes,
                base_path=str(base_path),
                train_split=train_split,
                val_split=val_split
            )

            yaml_file = output_path / f"{dataset_name}.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            QMessageBox.information(
                self,
                "Успех",
                f"Датасет создан успешно!\n\nYAML файл: {yaml_file}"
            )

            self.dataset_created.emit(str(yaml_file))
            self.accept()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось создать датасет:\n{str(e)}"
            )

    def _generate_yaml(
        self, classes: List[str], base_path: str, train_split: int, val_split: int
    ) -> str:
        """Generate YAML content for dataset configuration."""
        lines = [
            "# UltraGUI Dataset Configuration",
            f"# Generated on: 2026-06-15",
            "",
            f"path: {base_path}",
            f"train: images/train",
            f"val: images/val",
            "",
            f"nc: {len(classes)}",
            "names:",
        ]

        for class_name in classes:
            lines.append(f"  - {class_name}")

        return '\n'.join(lines) + '\n'

    def _validate_dataset(self) -> None:
        """Validate existing dataset."""
        yaml_path = self.yaml_path_edit.text().strip()
        if not yaml_path:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите YAML файл для проверки"
            )
            return

        try:
            from core.dataset_manager import DatasetManager

            manager = DatasetManager()
            manager.yaml_path = yaml_path

            # Run all validations
            results = []

            # YAML validity
            valid, errors = manager.validate_yaml(yaml_path)
            if valid:
                results.append("✅ YAML файл валиден")
            else:
                results.append(f"❌ YAML файл некорректен:")
                for err in errors:
                    results.append(f"   - {err}")

            # Images/labels matching
            valid, errors = manager.validate_images_labels(yaml_path)
            if valid:
                results.append("✅ Изображения и аннотации совпадают")
            else:
                results.append(f"⚠️  Расхождения в изображениях/аннотациях:")
                for err in errors[:5]:  # Show first 5
                    results.append(f"   - {err}")

            # Class validation
            valid, errors = manager.validate_classes(yaml_path)
            if valid:
                results.append("✅ Классы валидны")
            else:
                results.append(f"⚠️  Проблемы с классами:")
                for err in errors[:5]:
                    results.append(f"   - {err}")

            # Coordinate validation
            valid, errors = manager.validate_coordinates(yaml_path)
            if valid:
                results.append("✅ Координаты валидны")
            else:
                results.append(f"⚠️  Проблемы с координатами:")
                for err in errors[:5]:
                    results.append(f"   - {err}")

            self.validation_result.setPlainText('\n'.join(results))

        except Exception as e:
            self.validation_result.setPlainText(f"Ошибка проверки: {str(e)}")

    def _show_statistics(self) -> None:
        """Show dataset statistics."""
        yaml_path = self.yaml_stats_edit.text().strip()
        if not yaml_path:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Пожалуйста, выберите YAML файл"
            )
            return

        try:
            from core.dataset_manager import DatasetManager

            manager = DatasetManager()
            stats = manager.get_statistics(yaml_path)

            if "error" in stats:
                self.stats_result.setPlainText(f"Ошибка: {stats['error']}")
                return

            lines = [
                f"📊 Статистика датасета",
                "",
                f"Классов: {len(stats.get('classes', []))}",
                "",
                "Классы:",
            ]

            for i, cls in enumerate(stats.get('classes', [])):
                lines.append(f"  {i}. {cls}")

            lines.append("")
            lines.append("Количество изображений:")

            for split, count in stats.get('image_counts', {}).items():
                lines.append(f"  {split}: {count}")

            lines.append("")
            lines.append("Распределение классов:")

            for split, distribution in stats.get('class_distribution', {}).items():
                lines.append(f"  {split}:")
                if distribution:
                    for cls, cnt in sorted(distribution.items()):
                        lines.append(f"    - {cls}: {cnt}")
                else:
                    lines.append("    (нет данных)")

            self.stats_result.setPlainText('\n'.join(lines))

        except Exception as e:
            self.stats_result.setPlainText(f"Ошибка получения статистики: {str(e)}")


def show_dataset_dialog(parent: Optional[QWidget] = None) -> Optional[str]:
    """Helper function to show dataset dialog and return created YAML path."""
    dialog = DatasetCreationDialog(parent)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        # Check if dataset was created
        if hasattr(dialog, 'dataset_created'):
            for yaml_path in dialog.dataset_created:
                return yaml_path

    return None


# Export function for UI integration
def create_dataset_dialog(parent: Optional[QWidget] = None) -> DatasetCreationDialog:
    """Factory function to create a dataset dialog."""
    return DatasetCreationDialog(parent)
