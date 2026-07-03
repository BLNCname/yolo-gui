"""Tests for metrics chart widget."""

import pytest


class TestMetricsChart:
    """Test MetricsChartWidget class."""
    
    def test_metrics_chart_creation(self, qapp):
        """MetricsChart should be created without errors."""
        from ui.widgets.metrics_chart import MetricsChartWidget
        
        chart = MetricsChartWidget()
        
        assert chart is not None
    
    def test_metrics_chart_update(self, qapp):
        """MetricsChart should update with new metrics."""
        from ui.widgets.metrics_chart import MetricsChartWidget
        
        chart = MetricsChartWidget()
        
        # Update with sample metrics
        metrics = {
            "epoch": 1,
            "loss_box": 0.5,
            "metrics/mAP50(B)": 0.7,
            "metrics/mAP50-95(B)": 0.55,
        }
        
        chart.update_metrics(metrics)
        
        # Check that data was stored
        assert len(chart._epochs) == 1
        assert chart._epochs[0] == 1
        assert chart._losses[0] == 0.5
    
    def test_metrics_chart_multiple_updates(self, qapp):
        """MetricsChart should accumulate data across multiple updates."""
        from ui.widgets.metrics_chart import MetricsChartWidget
        
        chart = MetricsChartWidget()
        
        for i in range(5):
            metrics = {
                "epoch": i + 1,
                "loss_box": 0.5 / (i + 1),
                "metrics/mAP50(B)": 0.7 + i * 0.05,
                "metrics/mAP50-95(B)": 0.55 + i * 0.03,
            }
            chart.update_metrics(metrics)
        
        assert len(chart._epochs) == 5
        assert chart._losses[-1] < chart._losses[0]
    
    def test_metrics_chart_reset(self, qapp):
        """MetricsChart should reset data."""
        from ui.widgets.metrics_chart import MetricsChartWidget
        
        chart = MetricsChartWidget()
        
        # Add some data
        metrics = {
            "epoch": 1,
            "loss_box": 0.5,
            "metrics/mAP50(B)": 0.7,
            "metrics/mAP50-95(B)": 0.55,
        }
        chart.update_metrics(metrics)
        
        # Reset
        chart.reset()
        
        assert len(chart._epochs) == 0
        assert len(chart._losses) == 0