"""Train RL market making agent with PPO."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from atlas_mm.simulation.environment import MarketMakingEnv, EnvConfig


def make_env(seed: int):
    def _init():
        return MarketMakingEnv(EnvConfig(seed=seed, n_steps=3000))

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RL market maker")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output", type=str, default="models/ppo_mm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create vectorized environments
    envs = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args.seed + 100)])

    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="runs/",
        seed=args.seed,
    )

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path="runs/eval/",
        eval_freq=10_000,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
