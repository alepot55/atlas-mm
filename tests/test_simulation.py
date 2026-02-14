"""Tests for Phase 2: Market Simulation Layer."""

import numpy as np

from atlas_mm.simulation.price_process import (
    GBMProcess,
    GarchConfig,
    GarchProcess,
    PriceConfig,
)
from atlas_mm.simulation.flow_generator import FlowConfig, FlowGenerator
from atlas_mm.engine.orders import OrderType
from atlas_mm.simulation.agents_zoo import NoiseTrader, MomentumTrader


class TestGBM:
    def test_gbm_positive_prices(self):
        """Test 1: generate 10000 steps, all prices > 0."""
        proc = GBMProcess(PriceConfig(seed=42))
        prices = proc.generate(10000)
        assert np.all(prices > 0)

    def test_gbm_mean(self):
        """Test 2: with mu=0, mean of log returns should be approximately -sigma^2/2 * dt."""
        cfg = PriceConfig(mu=0.0, sigma=0.2, seed=42)
        proc = GBMProcess(cfg)
        prices = proc.generate(50000)
        log_returns = np.diff(np.log(prices))
        expected_mean = -0.5 * cfg.sigma**2 * cfg.dt
        assert abs(np.mean(log_returns) - expected_mean) < 0.001

    def test_gbm_volatility(self):
        """Test 3: std of log returns should be approximately sigma * sqrt(dt)."""
        cfg = PriceConfig(mu=0.0, sigma=0.2, seed=42)
        proc = GBMProcess(cfg)
        prices = proc.generate(50000)
        log_returns = np.diff(np.log(prices))
        expected_std = cfg.sigma * np.sqrt(cfg.dt)
        assert abs(np.std(log_returns) - expected_std) / expected_std < 0.05  # within 5%


class TestGARCH:
    def test_garch_volatility_clustering(self):
        """Test 4: large shocks should increase subsequent volatility."""
        cfg = GarchConfig(seed=42, sigma=0.02, omega=1e-5, alpha=0.15, beta=0.80)
        proc = GarchProcess(cfg)
        # Run for a while to get baseline
        for _ in range(100):
            proc.step()
        baseline_vol = proc.current_volatility

        # Inject a large shock by manipulating the last epsilon
        proc._last_epsilon = 0.5  # very large shock
        proc.step()
        after_shock_vol = proc.current_volatility
        assert after_shock_vol > baseline_vol

    def test_garch_stationarity(self):
        """Test 5: with alpha + beta < 1, variance should not explode."""
        cfg = GarchConfig(seed=42, omega=1e-5, alpha=0.1, beta=0.85)
        assert cfg.alpha + cfg.beta < 1.0
        proc = GarchProcess(cfg)
        prices = proc.generate(10000)
        # Variance should stay bounded
        assert np.all(np.isfinite(prices))
        assert np.all(prices > 0)


class TestFlowGenerator:
    def test_flow_generator_poisson(self):
        """Test 6: generate many steps, count orders. Mean should be close to rate * dt * n_steps."""
        cfg = FlowConfig(base_arrival_rate=10.0, seed=42)
        gen = FlowGenerator(cfg)
        total_orders = 0
        n_steps = 1000
        dt = 1.0
        for _ in range(n_steps):
            events = gen.generate_orders(100.0, 0.01, dt)
            total_orders += len(events)
        # Expected: 10 * 1.0 * 2 sides * 1000 steps = 20000
        expected = cfg.base_arrival_rate * dt * 2 * n_steps
        assert abs(total_orders - expected) / expected < 0.1  # within 10%

    def test_flow_generator_price_distribution(self):
        """Test 7: most limit orders should be near the mid (within 3 ticks)."""
        cfg = FlowConfig(base_arrival_rate=50.0, market_order_fraction=0.0, seed=42)
        gen = FlowGenerator(cfg)
        distances = []
        mid = 100.0
        tick = 0.01
        for _ in range(100):
            events = gen.generate_orders(mid, tick, 1.0)
            for ev in events:
                if ev.price is not None:
                    dist = abs(ev.price - mid) / tick
                    distances.append(dist)
        distances = np.array(distances)
        near_mid = np.sum(distances <= 3) / len(distances)
        assert near_mid > 0.5  # majority within 3 ticks

    def test_flow_generator_market_fraction(self):
        """Test 8: fraction of market orders should be approximately market_order_fraction."""
        cfg = FlowConfig(
            base_arrival_rate=50.0, market_order_fraction=0.2, seed=42
        )
        gen = FlowGenerator(cfg)
        total = 0
        market = 0
        for _ in range(200):
            events = gen.generate_orders(100.0, 0.01, 1.0)
            for ev in events:
                total += 1
                if ev.order_type == OrderType.MARKET:
                    market += 1
        frac = market / total
        assert abs(frac - 0.2) < 0.05  # within 5 percentage points


class TestBackgroundAgents:
    def test_noise_trader_symmetric(self):
        """Test 9: over many steps, roughly equal buy/sell."""
        agent = NoiseTrader(seed=42, arrival_rate=10.0)
        bids = 0
        asks = 0
        from atlas_mm.engine.orders import Side

        for _ in range(1000):
            orders = agent.act(100.0, 0.01, 1.0)
            for o in orders:
                if o.side == Side.BID:
                    bids += 1
                else:
                    asks += 1
        total = bids + asks
        assert total > 0
        ratio = bids / total
        assert 0.4 < ratio < 0.6  # roughly symmetric

    def test_deterministic_seed(self):
        """Test 10: same seed produces identical sequences."""
        cfg = PriceConfig(seed=123)
        proc1 = GBMProcess(cfg)
        prices1 = proc1.generate(100)

        proc2 = GBMProcess(PriceConfig(seed=123))
        prices2 = proc2.generate(100)

        np.testing.assert_array_equal(prices1, prices2)
