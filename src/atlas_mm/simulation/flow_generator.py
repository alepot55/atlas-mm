"""Synthetic order flow generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.orders import OrderType, Side


@dataclass
class FlowConfig:
    """Configuration for order flow generation."""

    base_arrival_rate: float = 10.0  # orders per second (each side)
    price_decay: float = 2.0  # exponential decay of intensity with distance from mid
    cancel_rate: float = 0.3  # fraction of resting orders cancelled per second
    market_order_fraction: float = 0.1  # fraction of orders that are market orders
    min_quantity: int = 1
    max_quantity: int = 10
    max_levels: int = 10  # max distance in ticks from mid for limit orders
    seed: int | None = None


@dataclass
class OrderEvent:
    """A generated order event to be processed by the engine."""

    side: Side
    price: float | None  # None for market orders
    quantity: int
    order_type: OrderType
    agent_id: str = "background"


class FlowGenerator:
    """Generates synthetic order flow for simulation.

    Uses Poisson process with intensity that decays exponentially
    with distance from mid-price. This creates a realistic
    order book shape (more liquidity near the mid).
    """

    def __init__(self, config: FlowConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate_orders(
        self, mid_price: float, tick_size: float, dt: float
    ) -> list[OrderEvent]:
        """Generate orders for one time step.

        Args:
            mid_price: Current mid-price.
            tick_size: Minimum price increment.
            dt: Time step duration in seconds.

        Returns:
            List of OrderEvent to submit to the engine.
        """
        events: list[OrderEvent] = []
        cfg = self.config

        for side in [Side.BID, Side.ASK]:
            # Number of orders this step (Poisson)
            n_orders = self.rng.poisson(cfg.base_arrival_rate * dt)

            for _ in range(n_orders):
                # Market or limit?
                if self.rng.random() < cfg.market_order_fraction:
                    qty = self.rng.integers(cfg.min_quantity, cfg.max_quantity + 1)
                    events.append(
                        OrderEvent(
                            side=side,
                            price=None,
                            quantity=int(qty),
                            order_type=OrderType.MARKET,
                        )
                    )
                else:
                    # Distance from mid in ticks (exponential distribution)
                    distance_ticks = int(self.rng.exponential(1.0 / cfg.price_decay)) + 1
                    distance_ticks = min(distance_ticks, cfg.max_levels)

                    if side == Side.BID:
                        price = mid_price - distance_ticks * tick_size
                    else:
                        price = mid_price + distance_ticks * tick_size

                    # Snap to tick grid
                    price = round(round(price / tick_size) * tick_size, 10)

                    qty = self.rng.integers(cfg.min_quantity, cfg.max_quantity + 1)
                    events.append(
                        OrderEvent(
                            side=side,
                            price=price,
                            quantity=int(qty),
                            order_type=OrderType.LIMIT,
                        )
                    )

        return events
