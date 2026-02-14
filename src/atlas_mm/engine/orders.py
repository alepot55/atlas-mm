"""Order type definitions for the matching engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Side(Enum):
    BID = auto()  # Buy
    ASK = auto()  # Sell


class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()
    CANCEL = auto()


@dataclass(frozen=True, slots=True)
class Order:
    """Immutable order representation.

    Attributes:
        order_id: Unique identifier (monotonically increasing).
        side: BID (buy) or ASK (sell).
        price: Limit price. None for market orders.
        quantity: Number of units. Must be > 0.
        timestamp: Simulation time when order was placed.
        agent_id: Identifier of the agent that placed the order.
    """

    order_id: int
    side: Side
    price: float | None
    quantity: int
    timestamp: float
    agent_id: str
    order_type: OrderType = OrderType.LIMIT


@dataclass(frozen=True, slots=True)
class Fill:
    """Represents a completed trade.

    Attributes:
        bid_order_id: ID of the buying order.
        ask_order_id: ID of the selling order.
        price: Execution price.
        quantity: Number of units traded.
        timestamp: Simulation time of the fill.
        aggressor_side: Which side initiated the trade (crossed the spread).
    """

    bid_order_id: int
    ask_order_id: int
    price: float
    quantity: int
    timestamp: float
    aggressor_side: Side


@dataclass
class CancelRequest:
    """Request to cancel a resting order."""

    order_id: int
    timestamp: float
