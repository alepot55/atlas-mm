"""Tests for Phase 4: Gymnasium Environment."""

import numpy as np

from atlas_mm.simulation.environment import EnvConfig, MarketMakingEnv


class TestMarketMakingEnv:
    def test_env_reset(self):
        """Test 1: env.reset() returns obs of correct shape, info dict."""
        env = MarketMakingEnv(EnvConfig(n_steps=100, seed=42))
        obs, info = env.reset(seed=42)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_env_step(self):
        """Test 2: env.step(action) returns correct types."""
        env = MarketMakingEnv(EnvConfig(n_steps=100, seed=42))
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "inventory" in info
        assert "pnl" in info

    def test_env_episode(self):
        """Test 3: run full episode (n_steps), verify terminated=True at end."""
        n_steps = 50
        env = MarketMakingEnv(EnvConfig(n_steps=n_steps, seed=42))
        env.reset(seed=42)
        terminated = False
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
        assert step_count == n_steps
        assert terminated is True

    def test_env_inventory_tracking(self):
        """Test 4: inventory is tracked in info."""
        env = MarketMakingEnv(EnvConfig(n_steps=100, seed=42))
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            _, _, _, _, info = env.step(action)
        assert "inventory" in info
        assert isinstance(info["inventory"], int)

    def test_env_reward_shape(self):
        """Test 5: reward should be finite for all steps."""
        env = MarketMakingEnv(EnvConfig(n_steps=50, seed=42))
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(reward)

    def test_env_obs_bounds(self):
        """Test 6: observations should be within reasonable bounds."""
        env = MarketMakingEnv(EnvConfig(n_steps=50, seed=42))
        obs, _ = env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert np.all(np.isfinite(obs))

    def test_env_action_space(self):
        """Test 7: action space should be MultiDiscrete([5, 5])."""
        env = MarketMakingEnv(EnvConfig(seed=42))
        assert env.action_space.shape == (2,)
        assert np.all(env.action_space.nvec == [5, 5])
