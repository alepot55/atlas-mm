"""Tests for the order book engine -- Phase 1 validation (17 test cases)."""

from atlas_mm.engine.orderbook import OrderBook
from atlas_mm.engine.orders import CancelRequest, Order, OrderType, Side


def _make_order(
    order_id: int,
    side: Side,
    price: float | None,
    quantity: int,
    order_type: OrderType = OrderType.LIMIT,
    agent_id: str = "test",
    timestamp: float = 0.0,
) -> Order:
    return Order(
        order_id=order_id,
        side=side,
        price=price,
        quantity=quantity,
        timestamp=timestamp,
        agent_id=agent_id,
        order_type=order_type,
    )


class TestOrderBookBasic:
    """Tests 1-4: Basic book operations."""

    def test_empty_book(self):
        """Test 1: new book has None for best_bid, best_ask, mid_price, spread."""
        book = OrderBook()
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None

    def test_single_bid(self):
        """Test 2: add one bid, best_bid == bid price, best_ask == None."""
        book = OrderBook()
        order = _make_order(1, Side.BID, 99.98, 10)
        fills = book.add_limit_order(order)
        assert fills == []
        assert book.best_bid == 99.98
        assert book.best_ask is None

    def test_single_ask(self):
        """Test 3: add one ask, best_ask == ask price, best_bid == None."""
        book = OrderBook()
        order = _make_order(1, Side.ASK, 100.02, 10)
        fills = book.add_limit_order(order)
        assert fills == []
        assert book.best_ask == 100.02
        assert book.best_bid is None

    def test_bid_ask_no_cross(self):
        """Test 4: add bid at 99.98, ask at 100.02. spread == 0.04, mid == 100.00."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.BID, 99.98, 10))
        book.add_limit_order(_make_order(2, Side.ASK, 100.02, 10))
        assert book.best_bid == 99.98
        assert book.best_ask == 100.02
        assert abs(book.spread - 0.04) < 1e-10
        assert abs(book.mid_price - 100.00) < 1e-10


class TestOrderBookCrossing:
    """Tests 5-6: Crossing orders and partial fills."""

    def test_crossing_order_fills(self):
        """Test 5: add ask at 100.00, then bid at 100.00. Should produce 1 fill. Book empty."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.00, 10))
        fills = book.add_limit_order(_make_order(2, Side.BID, 100.00, 10))
        assert len(fills) == 1
        assert fills[0].price == 100.00
        assert fills[0].quantity == 10
        assert fills[0].bid_order_id == 2
        assert fills[0].ask_order_id == 1
        assert fills[0].aggressor_side == Side.BID
        assert book.best_bid is None
        assert book.best_ask is None

    def test_crossing_order_partial(self):
        """Test 6: add ask(qty=10) at 100.00, then bid(qty=5) at 100.00.
        Should produce fill(qty=5). Ask side should have 5 remaining."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.00, 10))
        fills = book.add_limit_order(_make_order(2, Side.BID, 100.00, 5))
        assert len(fills) == 1
        assert fills[0].quantity == 5
        assert book.best_ask == 100.00
        state = book.get_book_state()
        assert state["asks"][0] == (100.00, 5)
        assert book.best_bid is None


class TestOrderBookPriority:
    """Tests 7-8: Price-time priority."""

    def test_price_time_priority(self):
        """Test 7: add ask(qty=5, id=1) at 100.00, add ask(qty=5, id=2) at 100.00.
        Then bid(qty=5) at 100.00. Fill should be against order id=1 (first in)."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.00, 5))
        book.add_limit_order(_make_order(2, Side.ASK, 100.00, 5))
        fills = book.add_limit_order(_make_order(3, Side.BID, 100.00, 5))
        assert len(fills) == 1
        assert fills[0].ask_order_id == 1  # first order fills first (FIFO)
        assert fills[0].quantity == 5
        # Order 2 should still be resting
        assert book.best_ask == 100.00
        state = book.get_book_state()
        assert state["asks"][0] == (100.00, 5)

    def test_price_priority(self):
        """Test 8: add ask(qty=5) at 100.02, add ask(qty=5) at 100.01.
        Then bid(qty=5) at 100.02. Fill should be at 100.01 (better price)."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.02, 5))
        book.add_limit_order(_make_order(2, Side.ASK, 100.01, 5))
        fills = book.add_limit_order(_make_order(3, Side.BID, 100.02, 5))
        assert len(fills) == 1
        assert fills[0].price == 100.01  # fills at better resting price
        assert fills[0].ask_order_id == 2


class TestMarketOrders:
    """Tests 9-10: Market order execution."""

    def test_market_order_buy(self):
        """Test 9: add asks at 100.01(qty=5) and 100.02(qty=5). Market buy(qty=8).
        Should get 2 fills: 5@100.01 + 3@100.02. Ask side should have 2 remaining at 100.02."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.01, 5))
        book.add_limit_order(_make_order(2, Side.ASK, 100.02, 5))
        market = _make_order(3, Side.BID, None, 8, order_type=OrderType.MARKET)
        fills = book.add_market_order(market)
        assert len(fills) == 2
        assert fills[0].price == 100.01
        assert fills[0].quantity == 5
        assert fills[1].price == 100.02
        assert fills[1].quantity == 3
        # 2 remaining at 100.02
        assert book.best_ask == 100.02
        state = book.get_book_state()
        assert state["asks"][0] == (100.02, 2)

    def test_market_order_no_liquidity(self):
        """Test 10: empty book, market buy. Should return 0 fills."""
        book = OrderBook()
        market = _make_order(1, Side.BID, None, 10, order_type=OrderType.MARKET)
        fills = book.add_market_order(market)
        assert len(fills) == 0


class TestCancellation:
    """Tests 11-12: Order cancellation."""

    def test_cancel_order(self):
        """Test 11: add bid, cancel it, best_bid should be None."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.BID, 99.98, 10))
        assert book.best_bid == 99.98
        result = book.cancel_order(CancelRequest(order_id=1, timestamp=1.0))
        assert result is True
        assert book.best_bid is None

    def test_cancel_nonexistent(self):
        """Test 12: cancel order_id that doesn't exist, returns False."""
        book = OrderBook()
        result = book.cancel_order(CancelRequest(order_id=999, timestamp=1.0))
        assert result is False


class TestTickRounding:
    """Test 13: Tick alignment."""

    def test_tick_rounding(self):
        """Test 13: order at price 100.005 with tick=0.01 should snap to 100.00 or 100.01."""
        book = OrderBook(tick_size=0.01)
        snapped = book.snap_to_tick(100.005)
        # Python round uses banker's rounding (round-half-even)
        # 100.005 / 0.01 = 10000.5 -> round to 10000 (even) -> 100.00
        assert snapped == 100.00 or snapped == 100.01  # depends on rounding mode
        # Verify it's on the tick grid
        assert abs(snapped / 0.01 - round(snapped / 0.01)) < 1e-10


class TestBookState:
    """Test 14: Book state depth."""

    def test_book_state_depth(self):
        """Test 14: add 10 ask levels, get_book_state(depth=5) returns only top 5."""
        book = OrderBook()
        for i in range(10):
            price = 100.01 + i * 0.01
            book.add_limit_order(_make_order(i + 1, Side.ASK, price, 5))
        state = book.get_book_state(depth=5)
        assert len(state["asks"]) == 5
        # Top 5 should be the lowest 5 prices
        assert state["asks"][0][0] == 100.01
        assert state["asks"][4][0] == 100.05


class TestAggressiveLimitOrders:
    """Tests 15-16: Aggressive limit orders that sweep the book."""

    def test_aggressive_limit_order(self):
        """Test 15: bid at 100.05 when best ask is 100.02.
        Should fill at 100.02 (resting price)."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.02, 5))
        fills = book.add_limit_order(_make_order(2, Side.BID, 100.05, 5))
        assert len(fills) == 1
        assert fills[0].price == 100.02  # fill at resting price, not aggressive price
        assert fills[0].quantity == 5

    def test_multiple_fills_single_order(self):
        """Test 16: bid at 100.05 qty=15, when asks are 100.01(5), 100.02(5), 100.03(5).
        Should get 3 fills sweeping the book."""
        book = OrderBook()
        book.add_limit_order(_make_order(1, Side.ASK, 100.01, 5))
        book.add_limit_order(_make_order(2, Side.ASK, 100.02, 5))
        book.add_limit_order(_make_order(3, Side.ASK, 100.03, 5))
        fills = book.add_limit_order(_make_order(4, Side.BID, 100.05, 15))
        assert len(fills) == 3
        assert fills[0].price == 100.01
        assert fills[0].quantity == 5
        assert fills[1].price == 100.02
        assert fills[1].quantity == 5
        assert fills[2].price == 100.03
        assert fills[2].quantity == 5
        assert book.best_ask is None  # all ask liquidity consumed


class TestInvariant:
    """Test 17: Book invariant."""

    def test_invariant_no_crossed_book(self):
        """Test 17: after any sequence of operations, assert best_bid < best_ask (when both exist)."""
        book = OrderBook()

        # Build up some book state
        book.add_limit_order(_make_order(1, Side.BID, 99.95, 10))
        book.add_limit_order(_make_order(2, Side.BID, 99.96, 5))
        book.add_limit_order(_make_order(3, Side.BID, 99.97, 8))
        book.add_limit_order(_make_order(4, Side.ASK, 100.01, 10))
        book.add_limit_order(_make_order(5, Side.ASK, 100.02, 5))
        book.add_limit_order(_make_order(6, Side.ASK, 100.03, 8))

        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask

        # Now submit a crossing order
        book.add_limit_order(_make_order(7, Side.BID, 100.02, 12))

        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask

        # Submit another crossing from the ask side
        book.add_limit_order(_make_order(8, Side.ASK, 99.90, 20))

        if book.best_bid is not None and book.best_ask is not None:
            assert book.best_bid < book.best_ask
