"""Visualization for market making evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import EvaluationMetrics


def set_style():
    """Set publication-quality plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


AS_COLOR = "#2196F3"
RL_COLOR = "#FF5722"


def plot_pnl_comparison(
    as_pnl: np.ndarray,
    rl_pnl: np.ndarray,
    output_path: str | Path = "results/pnl_comparison.png",
) -> None:
    """Plot cumulative PnL for both agents."""
    set_style()
    fig, ax = plt.subplots()

    steps = np.arange(len(as_pnl))
    ax.plot(steps, as_pnl, color=AS_COLOR, label="Avellaneda-Stoikov", linewidth=1.5)
    ax.plot(steps, rl_pnl, color=RL_COLOR, label="RL Agent (PPO)", linewidth=1.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Market Making PnL: Analytical vs RL")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.savefig(output_path)
    plt.close()


def plot_inventory_distribution(
    as_inventory: np.ndarray,
    rl_inventory: np.ndarray,
    output_path: str | Path = "results/inventory_distribution.png",
) -> None:
    """Plot inventory histograms for both agents."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(as_inventory, bins=50, color=AS_COLOR, alpha=0.7, edgecolor="white")
    ax1.set_title("A-S Inventory Distribution")
    ax1.set_xlabel("Inventory")
    ax1.set_ylabel("Frequency")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    ax2.hist(rl_inventory, bins=50, color=RL_COLOR, alpha=0.7, edgecolor="white")
    ax2.set_title("RL Inventory Distribution")
    ax2.set_xlabel("Inventory")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.suptitle("Inventory Distribution Comparison")
    plt.savefig(output_path)
    plt.close()


def plot_spread_dynamics(
    as_spread: np.ndarray,
    rl_spread: np.ndarray,
    volatility: np.ndarray,
    output_path: str | Path = "results/spread_dynamics.png",
) -> None:
    """Plot spread over time with volatility overlay."""
    set_style()
    fig, ax1 = plt.subplots()

    steps = np.arange(len(as_spread))
    ax1.plot(
        steps, as_spread, color=AS_COLOR, label="A-S Spread", alpha=0.7, linewidth=0.8
    )
    ax1.plot(
        steps, rl_spread, color=RL_COLOR, label="RL Spread", alpha=0.7, linewidth=0.8
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Quoted Spread")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        steps[: len(volatility)],
        volatility,
        color="gray",
        alpha=0.3,
        linewidth=0.8,
        label="Volatility",
    )
    ax2.set_ylabel("Volatility", color="gray")
    ax2.legend(loc="upper right")

    ax1.set_title("Spread Dynamics vs Volatility")
    plt.savefig(output_path)
    plt.close()


def plot_metrics_comparison(
    as_metrics: EvaluationMetrics,
    rl_metrics: EvaluationMetrics,
    output_path: str | Path = "results/metrics_comparison.png",
) -> None:
    """Panel of bar charts comparing key metrics (one subplot per metric)."""
    set_style()

    metrics = [
        ("Total PnL", as_metrics.total_pnl, rl_metrics.total_pnl),
        ("Sharpe Ratio", as_metrics.sharpe_ratio, rl_metrics.sharpe_ratio),
        ("Max Drawdown", as_metrics.max_drawdown, rl_metrics.max_drawdown),
        ("Inventory Std", as_metrics.inventory_std, rl_metrics.inventory_std),
        ("Fill Rate", as_metrics.fill_rate, rl_metrics.fill_rate),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    labels = ["A-S", "RL"]
    colors = [AS_COLOR, RL_COLOR]

    for ax, (name, as_val, rl_val) in zip(axes, metrics):
        bars = ax.bar(labels, [as_val, rl_val], color=colors, width=0.6, edgecolor="white")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)
        for bar, val in zip(bars, [as_val, rl_val]):
            fmt = f"{val:.2f}" if abs(val) >= 1 else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt, ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

    fig.suptitle("Performance Metrics Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
