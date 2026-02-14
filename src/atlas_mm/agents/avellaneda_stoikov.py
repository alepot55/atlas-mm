"""Avellaneda-Stoikov optimal market making model (2008).

Reference: Avellaneda, M., & Stoikov, S. (2008). High-frequency trading
in a limit order book. Quantitative Finance, 8(3), 217-224.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import AgentState, MarketMakingAgent, Quote


@dataclass
class ASConfig:
    """Avellaneda-Stoikov parameters.

    Attributes:
        gamma: Risk aversion. Higher = wider spreads, more inventory aversion.
               Typical range: [0.01, 1.0]. Start with 0.1.
        kappa: Order arrival intensity parameter. Higher = tighter spreads.
               Represents how likely the market maker's orders are to get filled.
               Typical range: [1.0, 10.0]. Start with 1.5.
        order_quantity: Fixed quantity per side.
    """

    gamma: float = 0.1
    kappa: float = 1.5
    order_quantity: int = 5


class AvellanedaStoikovAgent(MarketMakingAgent):
    """Avellaneda-Stoikov optimal market maker.

    Computes reservation price and optimal spread analytically,
    then quotes bid/ask around the reservation price.

    The reservation price shifts away from the mid-price
    proportionally to inventory, risk aversion, and volatility.
    This creates a natural inventory mean-reversion mechanism.
    """

    def __init__(
        self,
        config: ASConfig,
        agent_id: str = "avellaneda_stoikov",
        max_inventory: int = 100,
        tick_size: float = 0.01,
    ) -> None:
        super().__init__(agent_id=agent_id, max_inventory=max_inventory, tick_size=tick_size)
        self.config = config

    def quote(self, state: AgentState) -> Quote | None:
        cfg = self.config
        s = state.mid_price
        q = self.inventory
        sigma = state.volatility_estimate
        tau = max(state.time_remaining, 1e-6)  # avoid division by zero

        # Reservation price: shifts away from mid when inventory != 0
        reservation_price = s - q * cfg.gamma * sigma**2 * tau

        # Optimal spread
        spread = cfg.gamma * sigma**2 * tau + (2 / cfg.gamma) * math.log(
            1 + cfg.gamma / cfg.kappa
        )
        spread = max(spread, self.tick_size)  # minimum spread is 1 tick

        # Compute bid/ask
        bid = self.snap_to_tick(reservation_price - spread / 2)
        ask = self.snap_to_tick(reservation_price + spread / 2)

        # Ensure ask > bid (at least 1 tick spread)
        if ask <= bid:
            ask = bid + self.tick_size

        # Inventory limits: don't quote the side that would exceed limits
        bid_qty = cfg.order_quantity if self.inventory < self.max_inventory else 0
        ask_qty = cfg.order_quantity if self.inventory > -self.max_inventory else 0

        if bid_qty == 0 and ask_qty == 0:
            return None

        return Quote(
            bid_price=bid,
            ask_price=ask,
            bid_quantity=bid_qty,
            ask_quantity=ask_qty,
        )
