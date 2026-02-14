"""Performance metrics for market making evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """Complete evaluation of a market making simulation run."""

    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    max_inventory: int
    mean_inventory: float
    inventory_std: float
    total_fills: int
    fill_rate: float  # fills / total_quotes
    mean_spread_quoted: float  # average spread the agent quoted
    total_volume: int
    pnl_per_trade: float


def compute_metrics(
    pnl_series: np.ndarray,
    inventory_series: np.ndarray,
    spread_series: np.ndarray,
    n_quotes: int,
    n_fills: int,
) -> EvaluationMetrics:
    """Compute evaluation metrics from simulation data."""

    # PnL metrics
    returns = np.diff(pnl_series)
    total_pnl = float(pnl_series[-1] - pnl_series[0])
    sharpe = float(
        np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 6.5 * 3600)
    )

    # Drawdown
    cummax = np.maximum.accumulate(pnl_series)
    drawdowns = cummax - pnl_series
    max_dd = float(np.max(drawdowns))

    # Inventory
    max_inv = int(np.max(np.abs(inventory_series)))
    mean_inv = float(np.mean(inventory_series))
    inv_std = float(np.std(inventory_series))

    # Fill rate
    fill_rate = n_fills / max(n_quotes, 1)

    # Spread
    mean_spread = float(np.mean(spread_series)) if len(spread_series) > 0 else 0.0

    return EvaluationMetrics(
        total_pnl=total_pnl,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_inventory=max_inv,
        mean_inventory=mean_inv,
        inventory_std=inv_std,
        total_fills=n_fills,
        fill_rate=fill_rate,
        mean_spread_quoted=mean_spread,
        total_volume=n_fills * 5,  # approximate
        pnl_per_trade=total_pnl / max(n_fills, 1),
    )
