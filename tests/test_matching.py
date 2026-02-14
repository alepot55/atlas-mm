"""Tests for the matching engine orchestrator."""

from atlas_mm.engine.matching import MatchingEngine
from atlas_mm.engine.orders import OrderType, Side


class TestMatchingEngine:
    def test_process_limit_order(self):
        engine = MatchingEngine()
        oid, fills = engine.process_order(Side.BID, 99.98, 10, "agent1")
        assert oid == 0
        assert fills == []
        assert engine.stats.total_orders_processed == 1
        assert engine.book.best_bid == 99.98

    def test_process_market_order(self):
        engine = MatchingEngine()
        engine.process_order(Side.ASK, 100.02, 10, "agent1")
        oid, fills = engine.process_order(
            Side.BID, None, 5, "agent2", order_type=OrderType.MARKET
        )
        assert len(fills) == 1
        assert fills[0].quantity == 5

    def test_cancel(self):
        engine = MatchingEngine()
        oid, _ = engine.process_order(Side.BID, 99.98, 10, "agent1")
        assert engine.cancel(oid) is True
        assert engine.stats.total_cancellations == 1
        assert engine.book.best_bid is None

    def test_advance_time(self):
        engine = MatchingEngine()
        assert engine.time == 0.0
        engine.advance_time(1.5)
        assert engine.time == 1.5
        engine.advance_time(0.5)
        assert engine.time == 2.0

    def test_fill_log(self):
        engine = MatchingEngine()
        engine.process_order(Side.ASK, 100.00, 10, "seller")
        engine.process_order(Side.BID, 100.00, 5, "buyer")
        assert len(engine.fills) == 1
        assert engine.fills[0].quantity == 5
        assert engine.stats.total_volume == 5

    def test_order_id_monotonic(self):
        engine = MatchingEngine()
        oid1, _ = engine.process_order(Side.BID, 99.98, 10, "a1")
        oid2, _ = engine.process_order(Side.ASK, 100.02, 10, "a2")
        oid3, _ = engine.process_order(Side.BID, 99.99, 5, "a3")
        assert oid1 == 0
        assert oid2 == 1
        assert oid3 == 2
