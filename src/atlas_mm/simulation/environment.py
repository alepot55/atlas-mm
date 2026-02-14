"""Gymnasium environment for market making RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..engine.matching import MatchingEngine
from ..engine.orders import OrderType, Side
from ..simulation.agents_zoo import BackgroundAgent, MomentumTrader, NoiseTrader
from ..simulation.flow_generator import FlowConfig, FlowGenerator, OrderEvent
from ..simulation.price_process import GarchConfig, GarchProcess


@dataclass
class EnvConfig:
    """Environment configuration."""

    n_steps: int = 5000  # steps per episode
    tick_size: float = 0.01
    initial_price: float = 100.0
    max_inventory: int = 50
    inventory_penalty: float = 0.01  # lambda for inventory penalty in reward
    terminal_penalty: float = 0.1  # penalty for terminal inventory
    seed: int | None = None


class MarketMakingEnv(gym.Env):
    """Market making environment for RL training.

    Observation space (6 features):
        0: inventory / max_inventory          [-1, 1]
        1: volatility / baseline_vol          [0, inf)
        2: spread / tick_size                 [0, inf)
        3: time_remaining                     [0, 1]
        4: order_imbalance                    [-1, 1]
        5: normalized_pnl                     (-inf, inf)

    Action space: MultiDiscrete([5, 5])
        0: spread_level (1-5 ticks half-spread)
        1: skew_level (-2 to +2 ticks)

    Reward: realized_pnl_this_step - lambda * inventory^2
        This incentivizes capturing spread while penalizing inventory accumulation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, -1.0, -10.0]),
            high=np.array([1.0, 10.0, 100.0, 1.0, 1.0, 10.0]),
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete([5, 5])

        # Will be initialized in reset()
        self.engine: MatchingEngine = MatchingEngine(tick_size=self.config.tick_size)
        self.price_process: GarchProcess = GarchProcess(GarchConfig(
            initial_price=self.config.initial_price, seed=self.config.seed,
        ))
        self.flow_gen: FlowGenerator = FlowGenerator(FlowConfig(seed=self.config.seed))
        self._bg_agents: list[BackgroundAgent] = []
        self._step_count = 0
        self._inventory = 0
        self._cash = 0.0
        self._prev_pnl = 0.0
        self._bid_order_id: int | None = None
        self._ask_order_id: int | None = None

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        cfg = self.config
        actual_seed = seed if seed is not None else cfg.seed

        self.engine = MatchingEngine(tick_size=cfg.tick_size)
        self.price_process = GarchProcess(
            GarchConfig(
                initial_price=cfg.initial_price,
                seed=actual_seed,
            )
        )
        self.flow_gen = FlowGenerator(FlowConfig(seed=actual_seed))

        # Background agents
        self._bg_agents = [
            NoiseTrader(seed=actual_seed),
            MomentumTrader(seed=(actual_seed + 1) if actual_seed is not None else None),
        ]

        self._step_count = 0
        self._inventory = 0
        self._cash = 0.0
        self._prev_pnl = 0.0
        self._bid_order_id = None
        self._ask_order_id = None

        # Warm up: generate some initial book state
        mid = self.price_process.price
        for _ in range(50):
            events = self.flow_gen.generate_orders(mid, cfg.tick_size, dt=0.1)
            for ev in events:
                self.engine.process_order(
                    side=ev.side,
                    price=ev.price,
                    quantity=ev.quantity,
                    agent_id=ev.agent_id,
                    order_type=ev.order_type,
                )

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.config

        # 1. Cancel previous quotes
        if self._bid_order_id is not None:
            self.engine.cancel(self._bid_order_id)
            self._bid_order_id = None
        if self._ask_order_id is not None:
            self.engine.cancel(self._ask_order_id)
            self._ask_order_id = None

        # 2. Decode action
        spread_ticks = int(action[0]) + 1  # 1 to 5
        skew_ticks = int(action[1]) - 2  # -2 to +2

        mid = self.price_process.price
        bid_price = self.engine.book.snap_to_tick(
            mid - spread_ticks * cfg.tick_size + skew_ticks * cfg.tick_size
        )
        ask_price = self.engine.book.snap_to_tick(
            mid + spread_ticks * cfg.tick_size + skew_ticks * cfg.tick_size
        )
        if ask_price <= bid_price:
            ask_price = bid_price + cfg.tick_size

        # 3. Place new quotes (if within inventory limits)
        qty = 5
        if self._inventory < cfg.max_inventory:
            oid, fills = self.engine.process_order(
                side=Side.BID,
                price=bid_price,
                quantity=qty,
                agent_id="rl_agent",
                order_type=OrderType.LIMIT,
            )
            self._bid_order_id = oid
            for f in fills:
                self._inventory += f.quantity
                self._cash -= f.price * f.quantity

        if self._inventory > -cfg.max_inventory:
            oid, fills = self.engine.process_order(
                side=Side.ASK,
                price=ask_price,
                quantity=qty,
                agent_id="rl_agent",
                order_type=OrderType.LIMIT,
            )
            self._ask_order_id = oid
            for f in fills:
                self._inventory -= f.quantity
                self._cash += f.price * f.quantity

        # 4. Advance price and generate background flow
        new_price = self.price_process.step()

        bg_events: list[OrderEvent] = self.flow_gen.generate_orders(
            new_price, cfg.tick_size, dt=1.0
        )
        for agent in self._bg_agents:
            for order in agent.act(new_price, cfg.tick_size, dt=1.0):
                bg_events.append(
                    OrderEvent(
                        side=order.side,
                        price=order.price,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        agent_id=agent.agent_id,
                    )
                )

        for ev in bg_events:
            _, fills = self.engine.process_order(
                side=ev.side,
                price=ev.price,
                quantity=ev.quantity,
                agent_id=ev.agent_id,
                order_type=ev.order_type,
            )
            # Check if any fills hit our resting orders
            for f in fills:
                if f.bid_order_id == self._bid_order_id:
                    self._inventory += f.quantity
                    self._cash -= f.price * f.quantity
                    self._bid_order_id = None
                if f.ask_order_id == self._ask_order_id:
                    self._inventory -= f.quantity
                    self._cash += f.price * f.quantity
                    self._ask_order_id = None

        self.engine.advance_time(1.0)
        self._step_count += 1

        # 5. Compute reward
        current_pnl = self._cash + self._inventory * self.price_process.price
        step_pnl = current_pnl - self._prev_pnl
        inventory_penalty = cfg.inventory_penalty * self._inventory**2
        reward = step_pnl - inventory_penalty
        self._prev_pnl = current_pnl

        # 6. Check termination
        terminated = self._step_count >= cfg.n_steps
        truncated = False

        if terminated:
            # Terminal inventory penalty
            reward -= (
                cfg.terminal_penalty
                * abs(self._inventory)
                * self.price_process.price
                * cfg.tick_size
            )

        obs = self._get_obs()
        info = {
            "inventory": self._inventory,
            "cash": self._cash,
            "pnl": current_pnl,
            "mid_price": self.price_process.price,
        }

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        mid = self.price_process.price
        book = self.engine.book

        # Volatility estimate from price history
        history = self.price_process._history
        if len(history) > 20:
            returns = np.diff(np.log(history[-21:]))
            vol = float(np.std(returns))
        else:
            vol = 0.02

        # Order imbalance
        book_state = book.get_book_state(depth=5)
        bid_vol = sum(qty for _, qty in book_state.get("bids", []))
        ask_vol = sum(qty for _, qty in book_state.get("asks", []))
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        spread = book.spread or (2 * cfg.tick_size)
        pnl = self._cash + self._inventory * mid

        return np.array(
            [
                np.clip(self._inventory / cfg.max_inventory, -1.0, 1.0),
                np.clip(vol / 0.02, 0.0, 10.0),
                np.clip(spread / cfg.tick_size, 0.0, 100.0),
                1.0 - self._step_count / cfg.n_steps,
                np.clip(imbalance, -1.0, 1.0),
                np.clip(pnl / mid, -10.0, 10.0),
            ],
            dtype=np.float32,
        )
