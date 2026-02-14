"""Tests for Phase 3: Market Making Agents."""

import math

from atlas_mm.agents.avellaneda_stoikov import ASConfig, AvellanedaStoikovAgent
from atlas_mm.agents.base import AgentState
from atlas_mm.agents.rl_agent import RLConfig, RLMarketMaker
from atlas_mm.engine.orders import Side


def _make_state(
    mid_price: float = 100.0,
    inventory: int = 0,
    volatility: float = 0.02,
    time_remaining: float = 1.0,
    spread: float | None = 0.02,
    order_imbalance: float = 0.0,
) -> AgentState:
    return AgentState(
        mid_price=mid_price,
        best_bid=mid_price - 0.01 if spread else None,
        best_ask=mid_price + 0.01 if spread else None,
        spread=spread,
        inventory=inventory,
        cash=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        volatility_estimate=volatility,
        order_imbalance=order_imbalance,
        time_remaining=time_remaining,
        step=0,
    )


class TestAvellanedaStoikov:
    def test_as_symmetric_quote(self):
        """Test 1: with zero inventory and time_remaining=1.0, bid and ask should
        be symmetric around mid."""
        agent = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))
        state = _make_state(mid_price=100.0, inventory=0, time_remaining=1.0)
        quote = agent.quote(state)
        assert quote is not None
        mid = 100.0
        # Should be symmetric: bid and ask equidistant from mid
        bid_dist = mid - quote.bid_price
        ask_dist = quote.ask_price - mid
        assert abs(bid_dist - ask_dist) <= agent.tick_size

    def test_as_inventory_skew(self):
        """Test 2: with positive inventory, bid should be further from mid than ask
        (agent wants to sell)."""
        agent = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))
        agent.inventory = 20
        state = _make_state(mid_price=100.0, volatility=0.5, time_remaining=1.0)
        quote = agent.quote(state)
        assert quote is not None
        mid = 100.0
        # Reservation price should be BELOW mid when long
        # So ask is closer to mid than bid
        bid_dist = mid - quote.bid_price
        ask_dist = quote.ask_price - mid
        assert bid_dist > ask_dist

    def test_as_high_volatility_wider_spread(self):
        """Test 3: higher sigma should produce wider spread."""
        agent_low = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))
        agent_high = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))

        state_low = _make_state(volatility=0.01)
        state_high = _make_state(volatility=0.5)

        quote_low = agent_low.quote(state_low)
        quote_high = agent_high.quote(state_high)

        spread_low = quote_low.ask_price - quote_low.bid_price
        spread_high = quote_high.ask_price - quote_high.bid_price
        assert spread_high > spread_low

    def test_as_time_decay(self):
        """Test 4: as time_remaining -> 0, spread should decrease."""
        agent_early = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))
        agent_late = AvellanedaStoikovAgent(ASConfig(gamma=0.1, kappa=1.5))

        state_early = _make_state(time_remaining=1.0, volatility=0.5)
        state_late = _make_state(time_remaining=0.01, volatility=0.5)

        quote_early = agent_early.quote(state_early)
        quote_late = agent_late.quote(state_late)

        spread_early = quote_early.ask_price - quote_early.bid_price
        spread_late = quote_late.ask_price - quote_late.bid_price
        assert spread_late < spread_early

    def test_as_inventory_limits(self):
        """Test 5: at max_inventory, should not quote the accumulating side."""
        agent = AvellanedaStoikovAgent(ASConfig(), max_inventory=50)
        agent.inventory = 50  # at max long
        state = _make_state()
        quote = agent.quote(state)
        assert quote is not None
        assert quote.bid_quantity == 0  # should not buy more
        assert quote.ask_quantity > 0  # should still sell

    def test_as_minimum_spread(self):
        """Test 6: spread should never be less than 1 tick."""
        agent = AvellanedaStoikovAgent(ASConfig(gamma=0.001, kappa=10.0))
        state = _make_state(volatility=0.0001, time_remaining=0.001)
        quote = agent.quote(state)
        assert quote is not None
        assert quote.ask_price - quote.bid_price >= agent.tick_size

    def test_as_tick_alignment(self):
        """Test 7: bid and ask should always be on tick grid."""
        agent = AvellanedaStoikovAgent(ASConfig(gamma=0.3, kappa=2.0))
        state = _make_state(mid_price=100.033, volatility=0.15)
        quote = agent.quote(state)
        assert quote is not None
        tick = agent.tick_size
        assert abs(quote.bid_price / tick - round(quote.bid_price / tick)) < 1e-10
        assert abs(quote.ask_price / tick - round(quote.ask_price / tick)) < 1e-10


class TestRLAgent:
    def test_rl_random_policy(self):
        """Test 8: untrained agent should produce valid quotes (on tick grid, ask > bid)."""
        agent = RLMarketMaker(RLConfig())
        state = _make_state()
        quote = agent.quote(state)
        assert quote is not None
        assert quote.ask_price > quote.bid_price
        tick = agent.tick_size
        assert abs(quote.bid_price / tick - round(quote.bid_price / tick)) < 1e-10
        assert abs(quote.ask_price / tick - round(quote.ask_price / tick)) < 1e-10

    def test_rl_obs_shape(self):
        """Test 9: state_to_obs should return array of correct shape."""
        agent = RLMarketMaker(RLConfig())
        state = _make_state()
        obs = agent.state_to_obs(state)
        assert obs.shape == (6,)
        assert obs.dtype.name == "float32"

    def test_rl_inventory_limits(self):
        """Test 10: at max_inventory, should not quote the accumulating side."""
        agent = RLMarketMaker(RLConfig(), max_inventory=50)
        agent.inventory = 50
        state = _make_state()
        quote = agent.quote(state)
        assert quote is not None
        assert quote.bid_quantity == 0
        assert quote.ask_quantity > 0
