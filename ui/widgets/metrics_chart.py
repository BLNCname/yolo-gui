"""Real-time metrics chart using matplotlib in PyQt6."""

from typing import List, Dict

from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class MetricsChartWidget(QWidget):
    """Widget displaying training metrics in real-time."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self._setup_ui()
        self._initialize_metrics()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure — 3 rows x 2 cols for Loss, mAP50, mAP50-95, Precision, Recall
        self.figure, self.axes = plt.subplots(3, 2, figsize=(8, 7))
        self.figure.patch.set_facecolor('#1e1e1e')

        # Hide the unused bottom-right subplot
        self.axes[2, 1].set_visible(False)

        for ax in self.axes.flat:
            if ax.get_visible():
                ax.set_facecolor('#2d2d2d')
                ax.grid(True, alpha=0.3)

        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def _initialize_metrics(self):
        """Initialize metric storage."""
        self._epochs: List[int] = []
        self._losses: List[float] = []
        self._mAP50s: List[float] = []
        self._mAP50_95s: List[float] = []
        self._precisions: List[float] = []
        self._recalls: List[float] = []

    def update_metrics(self, metrics: Dict):
        """Update chart with new metrics."""
        epoch = metrics.get("epoch", len(self._epochs) + 1)

        # Extract values
        loss = metrics.get("loss_box", 0.0)
        mAP50 = metrics.get("metrics/mAP50(B)", 0.0)
        mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)

        # Store data
        self._epochs.append(epoch)
        self._losses.append(loss)
        self._mAP50s.append(mAP50)
        self._mAP50_95s.append(mAP50_95)
        self._precisions.append(precision)
        self._recalls.append(recall)

        # Update charts
        self._update_loss_chart()
        self._update_mAP50_chart()
        self._update_mAP50_95_chart()
        self._update_precision_chart()
        self._update_recall_chart()

        self.canvas.draw()

    def _update_loss_chart(self):
        """Update loss chart."""
        ax = self.axes[0, 0]
        ax.clear()
        ax.set_facecolor('#2d2d2d')

        if self._epochs:
            ax.plot(self._epochs, self._losses, 'r-', linewidth=1.5, label='Loss')
            ax.fill_between(self._epochs, self._losses, alpha=0.3)

        ax.set_xlabel('Epoch', color='#cccccc')
        ax.set_ylabel('Loss', color='#cccccc')
        ax.set_title('Training Loss', color='#ffffff')
        ax.legend(facecolor='#2d2d2d', labelcolor='white')
        ax.grid(True, alpha=0.3)

    def _update_mAP50_chart(self):
        """Update mAP50 chart."""
        ax = self.axes[0, 1]
        ax.clear()
        ax.set_facecolor('#2d2d2d')

        if self._epochs:
            ax.plot(self._epochs, self._mAP50s, 'g-', linewidth=1.5, label='mAP50')
            ax.fill_between(self._epochs, self._mAP50s, alpha=0.3)

        ax.set_xlabel('Epoch', color='#cccccc')
        ax.set_ylabel('mAP50', color='#cccccc')
        ax.set_title('Mean Average Precision (IoU=0.5)', color='#ffffff')
        ax.legend(facecolor='#2d2d2d', labelcolor='white')
        ax.grid(True, alpha=0.3)

    def _update_mAP50_95_chart(self):
        """Update mAP50-95 chart."""
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_facecolor('#2d2d2d')

        if self._epochs:
            ax.plot(self._epochs, self._mAP50_95s, 'b-', linewidth=1.5, label='mAP50-95')
            ax.fill_between(self._epochs, self._mAP50_95s, alpha=0.3)

        ax.set_xlabel('Epoch', color='#cccccc')
        ax.set_ylabel('mAP50-95', color='#cccccc')
        ax.set_title('Mean Average Precision (IoU=0.5:0.95)', color='#ffffff')
        ax.legend(facecolor='#2d2d2d', labelcolor='white')
        ax.grid(True, alpha=0.3)

    def _update_precision_chart(self):
        """Update Precision chart."""
        ax = self.axes[1, 1]
        ax.clear()
        ax.set_facecolor('#2d2d2d')

        if self._epochs:
            ax.plot(self._epochs, self._precisions, 'm-', linewidth=1.5, label='Precision')
            ax.fill_between(self._epochs, self._precisions, alpha=0.3)

        ax.set_xlabel('Epoch', color='#cccccc')
        ax.set_ylabel('Precision', color='#cccccc')
        ax.set_title('Precision', color='#ffffff')
        ax.legend(facecolor='#2d2d2d', labelcolor='white')
        ax.grid(True, alpha=0.3)

    def _update_recall_chart(self):
        """Update Recall chart."""
        ax = self.axes[2, 0]
        ax.clear()
        ax.set_facecolor('#2d2d2d')

        if self._epochs:
            ax.plot(self._epochs, self._recalls, 'c-', linewidth=1.5, label='Recall')
            ax.fill_between(self._epochs, self._recalls, alpha=0.3)

        ax.set_xlabel('Epoch', color='#cccccc')
        ax.set_ylabel('Recall', color='#cccccc')
        ax.set_title('Recall', color='#ffffff')
        ax.legend(facecolor='#2d2d2d', labelcolor='white')
        ax.grid(True, alpha=0.3)

    def reset(self):
        """Reset chart data."""
        self._initialize_metrics()
        for ax in self.axes.flat:
            if ax.get_visible():
                ax.clear()
        self.canvas.draw()
