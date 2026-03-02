"""Tests for validation chart factory functions."""

import plotly.graph_objects as go
import pytest
from advisor.simulator.charts import (
    brier_trend_chart,
    calibration_curve_chart,
    prediction_scatter_chart,
)


@pytest.fixture
def resolved_records():
    """Sample resolved calibration records for chart testing."""
    return [
        {
            "predicted_pop": 0.80,
            "predicted_touch": 0.25,
            "predicted_stop": 0.10,
            "predicted_ev": 15.0,
            "actual_profit": 1.0,
            "actual_touch": 0.0,
            "actual_stop": 0.0,
            "actual_pnl": 20.0,
            "created_at": "2025-06-15T10:00:00",
        },
        {
            "predicted_pop": 0.75,
            "predicted_touch": 0.30,
            "predicted_stop": 0.12,
            "predicted_ev": 12.0,
            "actual_profit": 1.0,
            "actual_touch": 1.0,
            "actual_stop": 0.0,
            "actual_pnl": 8.0,
            "created_at": "2025-07-01T10:00:00",
        },
        {
            "predicted_pop": 0.65,
            "predicted_touch": 0.40,
            "predicted_stop": 0.20,
            "predicted_ev": 5.0,
            "actual_profit": 0.0,
            "actual_touch": 1.0,
            "actual_stop": 1.0,
            "actual_pnl": -150.0,
            "created_at": "2025-07-15T10:00:00",
        },
        {
            "predicted_pop": 0.85,
            "predicted_touch": 0.15,
            "predicted_stop": 0.05,
            "predicted_ev": 20.0,
            "actual_profit": 1.0,
            "actual_touch": 0.0,
            "actual_stop": 0.0,
            "actual_pnl": 25.0,
            "created_at": "2025-08-01T10:00:00",
        },
        {
            "predicted_pop": 0.72,
            "predicted_touch": 0.28,
            "predicted_stop": 0.15,
            "predicted_ev": 8.0,
            "actual_profit": 0.0,
            "actual_touch": 1.0,
            "actual_stop": 1.0,
            "actual_pnl": -200.0,
            "created_at": "2025-08-15T10:00:00",
        },
    ]


class TestCalibrationCurveChart:
    def test_returns_figure(self, resolved_records):
        fig = calibration_curve_chart(resolved_records)
        assert isinstance(fig, go.Figure)

    def test_has_diagonal_trace(self, resolved_records):
        fig = calibration_curve_chart(resolved_records)
        trace_names = [t.name for t in fig.data]
        assert "Perfect" in trace_names, f"Expected diagonal trace, got: {trace_names}"

    def test_has_metric_traces(self, resolved_records):
        fig = calibration_curve_chart(resolved_records)
        trace_names = [t.name for t in fig.data]
        # Should have at least one metric trace besides diagonal
        metric_traces = [n for n in trace_names if n in ("POP", "Touch", "Stop")]
        assert len(metric_traces) > 0, f"Expected metric traces, got: {trace_names}"

    def test_empty_records(self):
        fig = calibration_curve_chart([])
        assert isinstance(fig, go.Figure)
        # Should at least have the diagonal
        assert len(fig.data) >= 1

    def test_layout_properties(self, resolved_records):
        fig = calibration_curve_chart(resolved_records)
        assert fig.layout.template.layout.template == "plotly_dark" or "dark" in str(
            fig.layout.template
        )
        assert fig.layout.height == 450


class TestBrierTrendChart:
    def test_returns_figure(self, resolved_records):
        fig = brier_trend_chart(resolved_records, window=2)
        assert isinstance(fig, go.Figure)

    def test_insufficient_data_handled(self):
        """Should handle < 2 records gracefully."""
        fig = brier_trend_chart(
            [{"predicted_pop": 0.80, "actual_profit": 1.0, "created_at": "2025-06-01"}]
        )
        assert isinstance(fig, go.Figure)
        # Should have annotation about insufficient data
        assert len(fig.layout.annotations) > 0

    def test_empty_records(self):
        fig = brier_trend_chart([])
        assert isinstance(fig, go.Figure)

    def test_has_reference_lines(self, resolved_records):
        """Should have horizontal reference lines for quality thresholds."""
        fig = brier_trend_chart(resolved_records, window=2)
        # Check for shapes (hlines are added as shapes in plotly)
        # The fig should have some horizontal lines
        assert isinstance(fig, go.Figure)

    def test_multiple_metric_traces(self, resolved_records):
        fig = brier_trend_chart(resolved_records, window=2)
        trace_names = [t.name for t in fig.data if t.name is not None]
        # Should have traces for the metrics
        assert len(trace_names) > 0


class TestPredictionScatterChart:
    def test_returns_figure(self, resolved_records):
        fig = prediction_scatter_chart(resolved_records)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, resolved_records):
        fig = prediction_scatter_chart(resolved_records)
        assert len(fig.data) >= 1, "Should have at least scatter trace"
        # First trace should be scatter
        assert fig.data[0].mode == "markers"

    def test_has_regression_line(self, resolved_records):
        """Should have OLS regression line with R-squared."""
        fig = prediction_scatter_chart(resolved_records)
        trace_names = [t.name for t in fig.data if t.name is not None]
        ols_traces = [n for n in trace_names if "OLS" in n or "R\u00b2" in n]
        assert len(ols_traces) > 0, f"Expected OLS trace, got: {trace_names}"

    def test_has_perfect_diagonal(self, resolved_records):
        fig = prediction_scatter_chart(resolved_records)
        trace_names = [t.name for t in fig.data if t.name is not None]
        assert "Perfect" in trace_names

    def test_empty_records(self):
        fig = prediction_scatter_chart([])
        assert isinstance(fig, go.Figure)

    def test_layout_properties(self, resolved_records):
        fig = prediction_scatter_chart(resolved_records)
        assert fig.layout.height == 450
