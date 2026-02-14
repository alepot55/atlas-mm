"""Tests for Phase 6: Evaluation Metrics."""

import numpy as np

from atlas_mm.evaluation.metrics import compute_metrics


class TestMetrics:
    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        pnl = np.cumsum(np.random.default_rng(42).normal(0.01, 0.1, 1000))
        pnl = np.concatenate([[0.0], pnl])
        inventory = np.random.default_rng(42).integers(-10, 10, 1001)
        spread = np.random.default_rng(42).uniform(0.01, 0.05, 1000)

        metrics = compute_metrics(pnl, inventory, spread, n_quotes=900, n_fills=500)

        assert isinstance(metrics.total_pnl, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown >= 0
        assert metrics.fill_rate == 500 / 900
        assert metrics.total_fills == 500
        assert np.isfinite(metrics.sharpe_ratio)

    def test_metrics_zero_fills(self):
        """Test metrics when there are no fills."""
        pnl = np.zeros(100)
        inventory = np.zeros(100, dtype=int)
        spread = np.ones(99) * 0.02

        metrics = compute_metrics(pnl, inventory, spread, n_quotes=0, n_fills=0)
        assert metrics.fill_rate == 0.0
        assert metrics.pnl_per_trade == 0.0

    def test_metrics_drawdown(self):
        """Test max drawdown calculation."""
        # PnL that goes up to 10 then drops to 5: drawdown = 5
        pnl = np.array([0, 2, 5, 8, 10, 9, 7, 5, 6, 7])
        inventory = np.zeros(10, dtype=int)
        spread = np.ones(9) * 0.02

        metrics = compute_metrics(pnl, inventory, spread, n_quotes=10, n_fills=5)
        assert metrics.max_drawdown == 5.0
