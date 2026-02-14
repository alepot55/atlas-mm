"""Main simulation entry point.

Runs Avellaneda-Stoikov and RL (random policy) agents side-by-side,
computes metrics, and generates plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from atlas_mm.agents.avellaneda_stoikov import ASConfig, AvellanedaStoikovAgent
from atlas_mm.agents.base import AgentState, MarketMakingAgent
from atlas_mm.agents.rl_agent import RLConfig, RLMarketMaker
from atlas_mm.engine.matching import MatchingEngine
from atlas_mm.engine.orders import OrderType, Side
from atlas_mm.evaluation.metrics import compute_metrics
from atlas_mm.evaluation.visualization import (
    plot_inventory_distribution,
    plot_metrics_comparison,
    plot_pnl_comparison,
    plot_spread_dynamics,
)
from atlas_mm.simulation.agents_zoo import MomentumTrader, NoiseTrader
from atlas_mm.simulation.flow_generator import FlowConfig, FlowGenerator, OrderEvent
from atlas_mm.simulation.price_process import GarchConfig, GarchProcess


def run_agent_simulation(
    agent: MarketMakingAgent,
    n_steps: int = 5000,
    seed: int = 42,
    tick_size: float = 0.01,
    initial_price: float = 100.0,
) -> dict:
    """Run a full simulation with the given agent.

    Returns a dict with time series data for evaluation.
    """
    engine = MatchingEngine(tick_size=tick_size)
    price_proc = GarchProcess(
        GarchConfig(initial_price=initial_price, seed=seed)
    )
    flow_gen = FlowGenerator(FlowConfig(seed=seed))

    bg_agents = [
        NoiseTrader(seed=seed),
        MomentumTrader(seed=seed + 1),
    ]

    # Arrays to collect data
    pnl_series = np.zeros(n_steps + 1)
    inventory_series = np.zeros(n_steps + 1, dtype=int)
    spread_series = np.zeros(n_steps)
    volatility_series = np.zeros(n_steps)

    agent.reset()
    n_quotes = 0
    n_fills = 0
    bid_order_id = None
    ask_order_id = None

    # Warm up the book
    mid = price_proc.price
    for _ in range(100):
        events = flow_gen.generate_orders(mid, tick_size, dt=0.1)
        for ev in events:
            engine.process_order(
                side=ev.side,
                price=ev.price,
                quantity=ev.quantity,
                agent_id=ev.agent_id,
                order_type=ev.order_type,
            )

    for step in range(n_steps):
        # Cancel previous quotes
        if bid_order_id is not None:
            engine.cancel(bid_order_id)
            bid_order_id = None
        if ask_order_id is not None:
            engine.cancel(ask_order_id)
            ask_order_id = None

        # Compute volatility estimate (annualized)
        history = price_proc._history
        if len(history) > 20:
            returns = np.diff(np.log(history[-21:]))
            per_step_vol = float(np.std(returns))
            # Annualize: multiply by sqrt(steps_per_year)
            # dt ~ 1/(252*6.5*3600), so steps_per_year ~ 252*6.5*3600
            vol = per_step_vol * np.sqrt(252 * 6.5 * 3600)
        else:
            vol = 0.02

        # Order imbalance
        book_state = engine.book.get_book_state(depth=5)
        bid_vol = sum(qty for _, qty in book_state.get("bids", []))
        ask_vol = sum(qty for _, qty in book_state.get("asks", []))
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        mid = price_proc.price
        state = AgentState(
            mid_price=mid,
            best_bid=engine.book.best_bid,
            best_ask=engine.book.best_ask,
            spread=engine.book.spread,
            inventory=agent.inventory,
            cash=agent.cash,
            unrealized_pnl=agent.inventory * mid,
            realized_pnl=agent._realized_pnl,
            volatility_estimate=vol,
            order_imbalance=imbalance,
            time_remaining=1.0 - step / n_steps,
            step=step,
        )

        quote = agent.quote(state)

        if quote is not None:
            n_quotes += 1
            spread_series[step] = quote.ask_price - quote.bid_price
            volatility_series[step] = vol

            # Place bid
            if quote.bid_quantity > 0:
                oid, fills = engine.process_order(
                    side=Side.BID,
                    price=quote.bid_price,
                    quantity=quote.bid_quantity,
                    agent_id=agent.agent_id,
                    order_type=OrderType.LIMIT,
                )
                bid_order_id = oid
                for f in fills:
                    agent.on_fill(Side.BID, f.price, f.quantity)
                    n_fills += 1

            # Place ask
            if quote.ask_quantity > 0:
                oid, fills = engine.process_order(
                    side=Side.ASK,
                    price=quote.ask_price,
                    quantity=quote.ask_quantity,
                    agent_id=agent.agent_id,
                    order_type=OrderType.LIMIT,
                )
                ask_order_id = oid
                for f in fills:
                    agent.on_fill(Side.ASK, f.price, f.quantity)
                    n_fills += 1
        else:
            spread_series[step] = engine.book.spread or tick_size
            volatility_series[step] = vol

        # Advance price
        new_price = price_proc.step()

        # Generate background flow
        bg_events: list[OrderEvent] = flow_gen.generate_orders(
            new_price, tick_size, dt=1.0
        )
        for bg_agent in bg_agents:
            for order in bg_agent.act(new_price, tick_size, dt=1.0):
                bg_events.append(
                    OrderEvent(
                        side=order.side,
                        price=order.price,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        agent_id=bg_agent.agent_id,
                    )
                )

        for ev in bg_events:
            _, fills = engine.process_order(
                side=ev.side,
                price=ev.price,
                quantity=ev.quantity,
                agent_id=ev.agent_id,
                order_type=ev.order_type,
            )
            for f in fills:
                if f.bid_order_id == bid_order_id:
                    agent.on_fill(Side.BID, f.price, f.quantity)
                    n_fills += 1
                    bid_order_id = None
                if f.ask_order_id == ask_order_id:
                    agent.on_fill(Side.ASK, f.price, f.quantity)
                    n_fills += 1
                    ask_order_id = None

        engine.advance_time(1.0)

        # Record state
        current_pnl = agent.cash + agent.inventory * price_proc.price
        pnl_series[step + 1] = current_pnl
        inventory_series[step + 1] = agent.inventory

    return {
        "pnl_series": pnl_series,
        "inventory_series": inventory_series,
        "spread_series": spread_series,
        "volatility_series": volatility_series,
        "n_quotes": n_quotes,
        "n_fills": n_fills,
    }


def main():
    parser = argparse.ArgumentParser(description="Run market making simulation")
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="assets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ATLAS-MM: Market Making Simulation")
    print("=" * 60)

    # Run A-S agent
    # With annualized vol ~0.007, gamma=50.0 and kappa=50.0 give spreads
    # of ~0.03 (3 ticks), which is competitive with the simulated order book
    print(f"\nRunning Avellaneda-Stoikov agent ({args.n_steps} steps)...")
    as_agent = AvellanedaStoikovAgent(
        ASConfig(gamma=50.0, kappa=50.0, order_quantity=5),
        max_inventory=50,
    )
    as_data = run_agent_simulation(
        as_agent, n_steps=args.n_steps, seed=args.seed
    )
    print(f"  Fills: {as_data['n_fills']}, Final PnL: {as_data['pnl_series'][-1]:.2f}")

    # Run RL agent
    rl_agent = RLMarketMaker(
        RLConfig(n_spread_levels=5, n_skew_levels=5, base_quantity=5),
        max_inventory=50,
    )
    model_path = Path("models/ppo_mm.zip")
    if model_path.exists():
        print(f"\nLoading trained RL model from {model_path}...")
        rl_agent.load_model(str(model_path))
        rl_label = "RL (PPO)"
    else:
        rl_label = "RL (random)"
    print(f"\nRunning {rl_label} agent ({args.n_steps} steps)...")
    rl_data = run_agent_simulation(
        rl_agent, n_steps=args.n_steps, seed=args.seed
    )
    print(f"  Fills: {rl_data['n_fills']}, Final PnL: {rl_data['pnl_series'][-1]:.2f}")

    # Compute metrics
    print("\nComputing metrics...")
    # Use rl_label for display (defined above)
    as_metrics = compute_metrics(
        as_data["pnl_series"],
        as_data["inventory_series"],
        as_data["spread_series"],
        as_data["n_quotes"],
        as_data["n_fills"],
    )
    rl_metrics = compute_metrics(
        rl_data["pnl_series"],
        rl_data["inventory_series"],
        rl_data["spread_series"],
        rl_data["n_quotes"],
        rl_data["n_fills"],
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'A-S':>15} {rl_label:>15}")
    print("-" * 55)
    print(f"{'Total PnL':<25} {as_metrics.total_pnl:>15.2f} {rl_metrics.total_pnl:>15.2f}")
    print(f"{'Sharpe Ratio':<25} {as_metrics.sharpe_ratio:>15.2f} {rl_metrics.sharpe_ratio:>15.2f}")
    print(f"{'Max Drawdown':<25} {as_metrics.max_drawdown:>15.2f} {rl_metrics.max_drawdown:>15.2f}")
    print(f"{'Max Inventory':<25} {as_metrics.max_inventory:>15d} {rl_metrics.max_inventory:>15d}")
    print(f"{'Inventory Std':<25} {as_metrics.inventory_std:>15.2f} {rl_metrics.inventory_std:>15.2f}")
    print(f"{'Total Fills':<25} {as_metrics.total_fills:>15d} {rl_metrics.total_fills:>15d}")
    print(f"{'Fill Rate':<25} {as_metrics.fill_rate:>15.2%} {rl_metrics.fill_rate:>15.2%}")
    print(f"{'Mean Spread':<25} {as_metrics.mean_spread_quoted:>15.4f} {rl_metrics.mean_spread_quoted:>15.4f}")
    print(f"{'PnL/Trade':<25} {as_metrics.pnl_per_trade:>15.4f} {rl_metrics.pnl_per_trade:>15.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_pnl_comparison(
        as_data["pnl_series"],
        rl_data["pnl_series"],
        output_dir / "pnl_comparison.png",
    )
    plot_inventory_distribution(
        as_data["inventory_series"],
        rl_data["inventory_series"],
        output_dir / "inventory_distribution.png",
    )
    plot_spread_dynamics(
        as_data["spread_series"],
        rl_data["spread_series"],
        as_data["volatility_series"],
        output_dir / "spread_dynamics.png",
    )
    plot_metrics_comparison(
        as_metrics,
        rl_metrics,
        output_dir / "metrics_comparison.png",
    )

    print(f"\nPlots saved to {output_dir}/")
    print("  - pnl_comparison.png")
    print("  - inventory_distribution.png")
    print("  - spread_dynamics.png")
    print("  - metrics_comparison.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
