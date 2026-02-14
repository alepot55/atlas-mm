"""Abstract interface for market making agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..engine.orders import Side


@dataclass
class Quote:
    """A two-sided quote from a market maker."""

    bid_price: float
    ask_price: float
    bid_quantity: int
    ask_quantity: int


@dataclass
class AgentState:
    """Observable state for the market making agent."""

    mid_price: float
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    inventory: int  # current inventory (positive = long)
    cash: float  # accumulated cash from trades
    unrealized_pnl: float  # inventory * (mid_price - avg_entry_price)
    realized_pnl: float  # total realized PnL
    volatility_estimate: float  # recent volatility
    order_imbalance: float  # (bid_volume - ask_volume) / total_volume at top N levels
    time_remaining: float  # fraction of session remaining [0, 1]
    step: int  # current simulation step


class MarketMakingAgent(ABC):
    """Abstract base class for market making agents.

    All agents must implement the `quote` method, which returns
    a two-sided quote given the current state.
    """

    def __init__(
        self, agent_id: str, max_inventory: int = 100, tick_size: float = 0.01
    ) -> None:
        self.agent_id = agent_id
        self.max_inventory = max_inventory
        self.tick_size = tick_size

        # Portfolio tracking
        self.inventory: int = 0
        self.cash: float = 0.0
        self._realized_pnl: float = 0.0

    @abstractmethod
    def quote(self, state: AgentState) -> Quote | None:
        """Generate a two-sided quote given current state.

        Returns None if the agent chooses not to quote (e.g., inventory limit reached).
        """
        ...

    def on_fill(self, side: Side, price: float, quantity: int) -> None:
        """Called when one of the agent's orders is filled.

        Updates inventory and cash tracking.
        """
        if side == Side.BID:
            # We bought
            self.inventory += quantity
            self.cash -= price * quantity
        else:
            # We sold
            self.inventory -= quantity
            self.cash += price * quantity

    def snap_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    @property
    def pnl(self) -> float:
        """Total PnL = cash + inventory value. Requires current mid for unrealized."""
        return self.cash  # unrealized added at evaluation time

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.inventory = 0
        self.cash = 0.0
        self._realized_pnl = 0.0
