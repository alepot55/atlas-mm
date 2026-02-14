"""Mid-price dynamics for simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PriceConfig:
    """Configuration for price process."""

    initial_price: float = 100.0
    dt: float = 1.0 / 252 / 6.5 / 3600  # ~1 second in trading-year units
    mu: float = 0.0  # zero drift for market making (no directional bias)
    sigma: float = 0.02  # annualized volatility (~2% for liquid equity)
    seed: int | None = None


@dataclass
class GarchConfig(PriceConfig):
    """Additional GARCH(1,1) parameters."""

    omega: float = 1e-5
    alpha: float = 0.1
    beta: float = 0.85


class PriceProcess(ABC):
    """Abstract base class for price processes."""

    def __init__(self, config: PriceConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._price = config.initial_price
        self._t = 0.0
        self._history: list[float] = [config.initial_price]

    @property
    def price(self) -> float:
        return self._price

    @abstractmethod
    def step(self) -> float:
        """Advance one time step. Returns new price."""
        ...

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate n_steps of prices. Returns array of shape (n_steps+1,)."""
        prices = np.empty(n_steps + 1)
        prices[0] = self._price
        for i in range(1, n_steps + 1):
            prices[i] = self.step()
        return prices


class GBMProcess(PriceProcess):
    """Geometric Brownian Motion."""

    def __init__(self, config: PriceConfig) -> None:
        super().__init__(config)

    def step(self) -> float:
        dt = self.config.dt
        z = self.rng.standard_normal()
        self._price *= np.exp(
            (self.config.mu - 0.5 * self.config.sigma**2) * dt
            + self.config.sigma * np.sqrt(dt) * z
        )
        self._t += dt
        self._history.append(self._price)
        return self._price


class GarchProcess(PriceProcess):
    """GBM with GARCH(1,1) stochastic volatility."""

    def __init__(self, config: GarchConfig) -> None:
        super().__init__(config)
        self._garch_config = config
        self._sigma_sq = config.sigma**2  # initial variance
        self._last_epsilon = 0.0

    @property
    def current_volatility(self) -> float:
        """Current instantaneous volatility (annualized)."""
        return float(np.sqrt(self._sigma_sq))

    def step(self) -> float:
        cfg = self._garch_config
        dt = cfg.dt

        # Update GARCH variance
        self._sigma_sq = cfg.omega + cfg.alpha * self._last_epsilon**2 + cfg.beta * self._sigma_sq

        sigma = np.sqrt(self._sigma_sq)
        z = self.rng.standard_normal()
        self._last_epsilon = sigma * z

        self._price *= np.exp((cfg.mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        self._t += dt
        self._history.append(self._price)
        return self._price
