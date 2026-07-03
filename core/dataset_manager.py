"""Dataset management module for YOLO format datasets."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DatasetManager:
    """Manager for YOLO dataset validation and configuration."""
    
    def __init__(self):
        self._yaml_path: str = ""
    
    @property
    def yaml_path(self) -> str:
        return self._yaml_path
    
    @yaml_path.setter
    def yaml_path(self, value: str):
        self._yaml_path = value
    
    def load_yaml(self, path: Optional[str] = None) -> Dict:
        """Load and parse YAML dataset configuration."""
        import yaml
        
        if path is None:
            path = self._yaml_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def validate_yaml(self, path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate YAML dataset configuration."""
        errors = []
        
        try:
            config = self.load_yaml(path)
        except Exception as e:
            return False, [f"Failed to parse YAML: {str(e)}"]

        if not isinstance(config, dict):
            return False, ["YAML root must be a mapping/object"]
        
        # Check required fields
        required_fields = ["path", "names"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate names in common YOLO forms: list or index->name mapping.
        if "names" in config:
            names = self._normalize_names(config["names"])
            if len(names) == 0:
                errors.append("'names' must be a non-empty list or mapping")

            if "nc" in config:
                try:
                    if int(config["nc"]) != len(names):
                        errors.append(f"'nc' ({config['nc']}) does not match names count ({len(names)})")
                except (TypeError, ValueError):
                    errors.append("'nc' must be an integer")

            if len(names) != len(set(names)):
                errors.append("Duplicate class names detected")
        
        # Validate path structure
        if "path" in config:
            base_path = config["path"]
            for subdir in ["images/train", "images/val"]:
                full_path = os.path.join(base_path, subdir)
                if not os.path.exists(full_path):
                    errors.append(f"Directory not found: {full_path}")
        
        return len(errors) == 0, errors

    @staticmethod
    def _normalize_names(names) -> List[str]:
        """Return class names from YOLO list or numeric-key mapping."""
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, dict):
            try:
                return [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
            except (TypeError, ValueError):
                return [str(value) for _, value in sorted(names.items())]
        return []
    
    def validate_images_labels(self, path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate that images and labels match 1-to-1."""
        import yaml
        
        if path is None:
            path = self._yaml_path
        
        try:
            config = self.load_yaml(path)
        except Exception as e:
            return False, [f"Failed to load YAML: {str(e)}"]
        
        errors = []
        base_path = config.get("path", "")
        
        for split in ["train", "val"]:
            images_dir = os.path.join(base_path, f"images/{split}")
            labels_dir = os.path.join(base_path, f"labels/{split}")
            
            if not os.path.exists(images_dir):
                errors.append(f"Images directory not found: {images_dir}")
                continue
            
            if not os.path.exists(labels_dir):
                errors.append(f"Labels directory not found: {labels_dir}")
                continue
            
            # Get image files
            image_files = set()
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.update(
                    f.stem for f in Path(images_dir).glob(f"*{ext}")
                )
            
            # Get label files
            label_files = set(f.stem for f in Path(labels_dir).glob("*.txt"))
            
            # Check mismatch
            only_in_images = image_files - label_files
            only_in_labels = label_files - image_files
            
            if only_in_images:
                errors.append(
                    f"Images without labels ({split}): {len(only_in_images)}"
                )
            if only_in_labels:
                errors.append(
                    f"Labels without images ({split}): {len(only_in_labels)}"
                )
        
        return len(errors) == 0, errors
    
    def validate_classes(self, path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate class indices in label files."""
        import yaml
        
        if path is None:
            path = self._yaml_path
        
        try:
            config = self.load_yaml(path)
        except Exception as e:
            return False, [f"Failed to load YAML: {str(e)}"]
        
        errors = []
        names = self._normalize_names(config.get("names", []))
        nc = len(names)
        
        for split in ["train", "val"]:
            labels_dir = os.path.join(config.get("path", ""), f"labels/{split}")
            
            if not os.path.exists(labels_dir):
                continue
            
            for label_file in Path(labels_dir).glob("*.txt"):
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            errors.append(
                                f"Invalid format in {label_file}:{line_num}"
                            )
                            continue
                        
                        try:
                            class_idx = int(parts[0])
                            if class_idx < 0 or class_idx >= nc:
                                errors.append(
                                    f"Class index out of range ({class_idx} >= {nc}) "
                                    f"in {label_file}:{line_num}"
                                )
                        except ValueError:
                            errors.append(
                                f"Invalid class index in {label_file}:{line_num}"
                            )
        
        return len(errors) == 0, errors
    
    def validate_coordinates(self, path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate that coordinates are normalized [0, 1]."""
        import yaml
        
        if path is None:
            path = self._yaml_path
        
        try:
            config = self.load_yaml(path)
        except Exception as e:
            return False, [f"Failed to load YAML: {str(e)}"]
        
        errors = []
        
        for split in ["train", "val"]:
            labels_dir = os.path.join(config.get("path", ""), f"labels/{split}")
            
            if not os.path.exists(labels_dir):
                continue
            
            for label_file in Path(labels_dir).glob("*.txt"):
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        try:
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            for val, name in [
                                (x_center, "x_center"),
                                (y_center, "y_center"),
                                (width, "width"),
                                (height, "height"),
                            ]:
                                if not (0.0 <= val <= 1.0):
                                    errors.append(
                                        f"{name} out of range ({val}) in "
                                        f"{label_file}:{line_num}"
                                    )
                        except ValueError:
                            errors.append(
                                f"Invalid coordinate value in {label_file}:{line_num}"
                            )
        
        return len(errors) == 0, errors
    
    def get_statistics(self, path: Optional[str] = None) -> Dict:
        """Get dataset statistics."""
        import yaml
        from collections import Counter
        
        if path is None:
            path = self._yaml_path
        
        stats = {
            "classes": [],
            "image_counts": {},
            "label_counts": {},
            "class_distribution": {},
        }
        
        try:
            config = self.load_yaml(path)
        except Exception as e:
            stats["error"] = str(e)
            return stats
        
        names = self._normalize_names(config.get("names", []))
        base_path = config.get("path", "")
        
        # Get class names
        stats["classes"] = names
        
        for split in ["train", "val"]:
            images_dir = os.path.join(base_path, f"images/{split}")
            labels_dir = os.path.join(base_path, f"labels/{split}")
            
            if os.path.exists(images_dir):
                image_count = sum(
                    1 for _ in Path(images_dir).glob("*")
                    if _.suffix.lower() in [".jpg", ".jpeg", ".png"]
                )
                stats["image_counts"][split] = image_count
            
            if os.path.exists(labels_dir):
                # Count labels
                total_labels = 0
                class_counter = Counter()
                
                for label_file in Path(labels_dir).glob("*.txt"):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                try:
                                    class_idx = int(parts[0])
                                    if class_idx < len(names):
                                        class_counter[names[class_idx]] += 1
                                    total_labels += 1
                                except ValueError:
                                    pass
                
                stats["label_counts"][split] = total_labels
                stats["class_distribution"][split] = dict(class_counter)
        
        return stats


def create_dataset_manager():
    """Factory function to create a dataset manager instance."""
    return DatasetManager()
