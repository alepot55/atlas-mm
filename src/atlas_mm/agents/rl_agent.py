"""Reinforcement Learning market making agent using PPO.

The agent observes market state and outputs:
- spread_level: how wide to quote (discretized)
- skew_level: how much to skew toward reducing inventory (discretized)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AgentState, MarketMakingAgent, Quote


@dataclass
class RLConfig:
    """RL agent configuration."""

    n_spread_levels: int = 5  # discrete spread choices (1 tick to 5 ticks)
    n_skew_levels: int = 5  # discrete skew choices (-2 ticks to +2 ticks)
    base_quantity: int = 5
    model_path: str | None = None  # path to trained model


class RLMarketMaker(MarketMakingAgent):
    """PPO-based market maker.

    Action space: MultiDiscrete([n_spread_levels, n_skew_levels])
    - spread_level: index into [1, 2, 3, 4, 5] ticks half-spread
    - skew_level: index into [-2, -1, 0, 1, 2] ticks skew

    Observation space: Box with features:
    - inventory_normalized: inventory / max_inventory, in [-1, 1]
    - volatility_normalized: sigma / sigma_baseline
    - spread_normalized: current_spread / tick_size
    - time_remaining: [0, 1]
    - order_imbalance: [-1, 1]
    - recent_pnl_normalized: rolling PnL / initial_price
    """

    def __init__(
        self,
        config: RLConfig,
        agent_id: str = "rl_agent",
        max_inventory: int = 100,
        tick_size: float = 0.01,
    ) -> None:
        super().__init__(agent_id=agent_id, max_inventory=max_inventory, tick_size=tick_size)
        self.config = config
        self._model = None
        self._spread_choices = list(range(1, config.n_spread_levels + 1))  # in ticks
        self._skew_choices = list(
            range(
                -(config.n_skew_levels // 2),
                config.n_skew_levels // 2 + 1,
            )
        )
        self._rng = np.random.default_rng(42)

    def load_model(self, path: str) -> None:
        """Load a trained SB3 model."""
        from stable_baselines3 import PPO

        self._model = PPO.load(path)

    def state_to_obs(self, state: AgentState) -> np.ndarray:
        """Convert AgentState to observation vector for the RL model."""
        return np.array(
            [
                self.inventory / self.max_inventory,  # [-1, 1]
                state.volatility_estimate / 0.02,  # normalized to baseline
                (state.spread or 0.02) / self.tick_size,  # spread in ticks
                state.time_remaining,  # [0, 1]
                state.order_imbalance,  # [-1, 1]
                (self.cash + self.inventory * state.mid_price)
                / state.mid_price,  # normalized PnL
            ],
            dtype=np.float32,
        )

    def quote(self, state: AgentState) -> Quote | None:
        if self._model is None:
            # Random policy for untrained agent
            spread_idx = int(self._rng.integers(0, len(self._spread_choices)))
            skew_idx = int(self._rng.integers(0, len(self._skew_choices)))
        else:
            obs = self.state_to_obs(state)
            action, _ = self._model.predict(obs, deterministic=True)
            spread_idx, skew_idx = int(action[0]), int(action[1])

        half_spread_ticks = self._spread_choices[spread_idx]
        skew_ticks = self._skew_choices[skew_idx]

        mid = state.mid_price
        bid = self.snap_to_tick(
            mid - half_spread_ticks * self.tick_size + skew_ticks * self.tick_size
        )
        ask = self.snap_to_tick(
            mid + half_spread_ticks * self.tick_size + skew_ticks * self.tick_size
        )

        # Ensure ask > bid
        if ask <= bid:
            ask = bid + self.tick_size

        bid_qty = self.config.base_quantity if self.inventory < self.max_inventory else 0
        ask_qty = self.config.base_quantity if self.inventory > -self.max_inventory else 0

        if bid_qty == 0 and ask_qty == 0:
            return None

        return Quote(
            bid_price=bid,
            ask_price=ask,
            bid_quantity=bid_qty,
            ask_quantity=ask_qty,
        )
