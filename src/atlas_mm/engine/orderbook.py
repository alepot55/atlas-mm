"""Limit Order Book with price-time priority matching."""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass, field

from .orders import CancelRequest, Fill, Order, Side


@dataclass
class PriceLevel:
    """A single price level in the book."""

    price: float
    orders: deque[Order] = field(default_factory=deque)
    total_quantity: int = 0


class OrderBook:
    """L2 Order Book with price-time priority.

    Invariants (enforced and verified):
    1. best_bid < best_ask (no crossed book) -- unless book is empty on one side
    2. Orders at the same price level are filled FIFO
    3. All prices are multiples of tick_size
    4. All quantities are positive integers
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size

        # Price level storage: price -> PriceLevel
        self._bid_levels: dict[float, PriceLevel] = {}
        self._ask_levels: dict[float, PriceLevel] = {}

        # Sorted price tracking (maintained manually)
        # _bid_prices sorted descending (highest first)
        # _ask_prices sorted ascending (lowest first)
        self._bid_prices: list[float] = []
        self._ask_prices: list[float] = []

        # Order lookup for cancellations: order_id -> (side, price)
        self._order_index: dict[int, tuple[Side, float]] = {}

        # Trade log
        self._fills: list[Fill] = []

        # Statistics
        self._total_orders: int = 0
        self._total_fills: int = 0

    @property
    def best_bid(self) -> float | None:
        """Highest bid price, or None if no bids."""
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price, or None if no asks."""
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> float | None:
        """Mid-price = (best_bid + best_ask) / 2, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        """Bid-ask spread, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def snap_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def add_limit_order(self, order: Order) -> list[Fill]:
        """Add a limit order. May generate fills if the order crosses the spread.

        Returns list of fills generated (empty if order rests in book).
        """
        assert order.price is not None, "Limit order must have a price"
        assert order.quantity > 0, "Order quantity must be positive"
        assert order.price > 0, "Order price must be positive"

        price = self.snap_to_tick(order.price)
        self._total_orders += 1

        fills, remaining_qty = self._match_incoming_order(order, price)

        # If there's remaining quantity, add to the book
        if remaining_qty > 0:
            resting_order = Order(
                order_id=order.order_id,
                side=order.side,
                price=price,
                quantity=remaining_qty,
                timestamp=order.timestamp,
                agent_id=order.agent_id,
                order_type=order.order_type,
            )
            self._add_to_book(resting_order)

        return fills

    def add_market_order(self, order: Order) -> list[Fill]:
        """Execute a market order against resting liquidity.

        Returns list of fills. Unfilled portion is discarded (no partial resting).
        """
        assert order.quantity > 0, "Order quantity must be positive"
        self._total_orders += 1

        fills, _ = self._match_incoming_order(order, price=None)
        return fills

    def cancel_order(self, cancel: CancelRequest) -> bool:
        """Cancel a resting order. Returns True if order was found and cancelled."""
        if cancel.order_id not in self._order_index:
            return False

        side, price = self._order_index[cancel.order_id]

        if side == Side.BID:
            levels = self._bid_levels
            prices = self._bid_prices
        else:
            levels = self._ask_levels
            prices = self._ask_prices

        if price not in levels:
            del self._order_index[cancel.order_id]
            return False

        level = levels[price]
        # Find and remove the order from the deque
        for i, o in enumerate(level.orders):
            if o.order_id == cancel.order_id:
                level.orders.remove(o)
                level.total_quantity -= o.quantity
                del self._order_index[cancel.order_id]

                # If level is empty, remove it
                if level.total_quantity <= 0:
                    del levels[price]
                    self._remove_price(prices, price)

                return True

        # Order not found in level (shouldn't happen)
        del self._order_index[cancel.order_id]
        return False

    def get_book_state(self, depth: int = 5) -> dict:
        """Return top N levels on each side for observation.

        Returns dict with keys: 'bids', 'asks', 'mid_price', 'spread'.
        Each side is a list of (price, total_quantity) tuples.
        """
        bids = []
        for p in self._bid_prices[:depth]:
            level = self._bid_levels[p]
            bids.append((p, level.total_quantity))

        asks = []
        for p in self._ask_prices[:depth]:
            level = self._ask_levels[p]
            asks.append((p, level.total_quantity))

        return {
            "bids": bids,
            "asks": asks,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }

    def _add_to_book(self, order: Order) -> None:
        """Add a resting order to the appropriate side of the book."""
        price = order.price
        assert price is not None

        if order.side == Side.BID:
            levels = self._bid_levels
            prices = self._bid_prices
            ascending = False
        else:
            levels = self._ask_levels
            prices = self._ask_prices
            ascending = True

        if price not in levels:
            levels[price] = PriceLevel(price=price)
            self._insert_price_sorted(prices, price, ascending)

        levels[price].orders.append(order)
        levels[price].total_quantity += order.quantity
        self._order_index[order.order_id] = (order.side, price)

    def _insert_price_sorted(self, prices: list[float], price: float, ascending: bool) -> None:
        """Insert price into sorted list using bisect. Maintain sort order."""
        if ascending:
            bisect.insort(prices, price)
        else:
            # For descending: use negated values with bisect
            # Find the insertion point in descending order
            pos = bisect.bisect_left([-p for p in prices], -price)
            prices.insert(pos, price)

    def _remove_price(self, prices: list[float], price: float) -> None:
        """Remove price from sorted list."""
        try:
            prices.remove(price)
        except ValueError:
            pass

    def _match_incoming_order(
        self, order: Order, price: float | None
    ) -> tuple[list[Fill], int]:
        """Match an incoming order against the opposite side of the book.

        For limit orders, `price` is the snapped limit price.
        For market orders, `price` is None (matches at any price).

        Returns (fills, remaining_quantity).
        """
        fills: list[Fill] = []
        remaining_qty = order.quantity

        if order.side == Side.BID:
            # Match against asks (lowest first)
            while remaining_qty > 0 and self._ask_prices:
                best_ask_price = self._ask_prices[0]

                # For limit orders, stop if the ask price is above our limit
                if price is not None and price < best_ask_price:
                    break

                level = self._ask_levels[best_ask_price]

                while remaining_qty > 0 and level.orders:
                    resting = level.orders[0]
                    fill_qty = min(remaining_qty, resting.quantity)

                    fill = Fill(
                        bid_order_id=order.order_id,
                        ask_order_id=resting.order_id,
                        price=best_ask_price,  # fill at resting price
                        quantity=fill_qty,
                        timestamp=order.timestamp,
                        aggressor_side=Side.BID,
                    )
                    fills.append(fill)
                    self._fills.append(fill)
                    self._total_fills += 1

                    remaining_qty -= fill_qty

                    if fill_qty >= resting.quantity:
                        # Resting order fully filled
                        level.orders.popleft()
                        level.total_quantity -= resting.quantity
                        if resting.order_id in self._order_index:
                            del self._order_index[resting.order_id]
                    else:
                        # Resting order partially filled - replace with reduced quantity
                        level.orders[0] = Order(
                            order_id=resting.order_id,
                            side=resting.side,
                            price=resting.price,
                            quantity=resting.quantity - fill_qty,
                            timestamp=resting.timestamp,
                            agent_id=resting.agent_id,
                            order_type=resting.order_type,
                        )
                        level.total_quantity -= fill_qty

                # Remove empty level
                if not level.orders:
                    del self._ask_levels[best_ask_price]
                    self._ask_prices.pop(0)

        else:
            # ASK side: match against bids (highest first)
            while remaining_qty > 0 and self._bid_prices:
                best_bid_price = self._bid_prices[0]

                # For limit orders, stop if the bid price is below our limit
                if price is not None and price > best_bid_price:
                    break

                level = self._bid_levels[best_bid_price]

                while remaining_qty > 0 and level.orders:
                    resting = level.orders[0]
                    fill_qty = min(remaining_qty, resting.quantity)

                    fill = Fill(
                        bid_order_id=resting.order_id,
                        ask_order_id=order.order_id,
                        price=best_bid_price,  # fill at resting price
                        quantity=fill_qty,
                        timestamp=order.timestamp,
                        aggressor_side=Side.ASK,
                    )
                    fills.append(fill)
                    self._fills.append(fill)
                    self._total_fills += 1

                    remaining_qty -= fill_qty

                    if fill_qty >= resting.quantity:
                        level.orders.popleft()
                        level.total_quantity -= resting.quantity
                        if resting.order_id in self._order_index:
                            del self._order_index[resting.order_id]
                    else:
                        level.orders[0] = Order(
                            order_id=resting.order_id,
                            side=resting.side,
                            price=resting.price,
                            quantity=resting.quantity - fill_qty,
                            timestamp=resting.timestamp,
                            agent_id=resting.agent_id,
                            order_type=resting.order_type,
                        )
                        level.total_quantity -= fill_qty

                if not level.orders:
                    del self._bid_levels[best_bid_price]
                    self._bid_prices.pop(0)

        return fills, remaining_qty
