"""Matching engine that processes order streams and maintains state."""

from __future__ import annotations

from dataclasses import dataclass

from .orderbook import OrderBook
from .orders import CancelRequest, Fill, Order, OrderType, Side


@dataclass
class EngineStats:
    """Running statistics of the matching engine."""

    total_orders_processed: int = 0
    total_fills: int = 0
    total_volume: int = 0
    total_cancellations: int = 0


class MatchingEngine:
    """Orchestrates order processing and maintains audit trail.

    This is the main interface for the simulation layer.
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.book = OrderBook(tick_size=tick_size)
        self.stats = EngineStats()
        self._next_order_id: int = 0
        self._fill_log: list[Fill] = []
        self._time: float = 0.0

    def process_order(
        self,
        side: Side,
        price: float | None,
        quantity: int,
        agent_id: str,
        order_type: OrderType = OrderType.LIMIT,
    ) -> tuple[int, list[Fill]]:
        """Submit an order and return (order_id, fills).

        For limit orders, price must be provided.
        For market orders, price should be None.
        """
        order_id = self._next_order_id
        self._next_order_id += 1

        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self._time,
            agent_id=agent_id,
            order_type=order_type,
        )

        if order_type == OrderType.MARKET:
            fills = self.book.add_market_order(order)
        else:
            fills = self.book.add_limit_order(order)

        self._fill_log.extend(fills)
        self.stats.total_orders_processed += 1
        self.stats.total_fills += len(fills)
        self.stats.total_volume += sum(f.quantity for f in fills)

        return order_id, fills

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order."""
        success = self.book.cancel_order(
            CancelRequest(order_id=order_id, timestamp=self._time)
        )
        if success:
            self.stats.total_cancellations += 1
        return success

    def advance_time(self, dt: float) -> None:
        """Advance simulation clock."""
        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    @property
    def fills(self) -> list[Fill]:
        return self._fill_log
