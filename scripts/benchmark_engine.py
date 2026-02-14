"""Benchmark the matching engine throughput."""

import time

from atlas_mm.engine.matching import MatchingEngine
from atlas_mm.engine.orders import OrderType, Side


def main():
    engine = MatchingEngine(tick_size=0.01)

    n_orders = 100_000
    print(f"Benchmarking matching engine with {n_orders:,} orders...")

    # Pre-populate some liquidity
    for i in range(100):
        engine.process_order(Side.BID, 99.0 + i * 0.01, 10, "init")
        engine.process_order(Side.ASK, 100.0 + i * 0.01, 10, "init")

    start = time.perf_counter()

    for i in range(n_orders):
        side = Side.BID if i % 2 == 0 else Side.ASK
        if i % 10 == 0:
            # Market order
            engine.process_order(side, None, 1, "bench", OrderType.MARKET)
        else:
            # Limit order
            price = 99.5 + (i % 100) * 0.01
            engine.process_order(side, price, 1, "bench", OrderType.LIMIT)

    elapsed = time.perf_counter() - start
    throughput = n_orders / elapsed

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} orders/sec")
    print(f"  Total fills: {engine.stats.total_fills:,}")
    print(f"  Total volume: {engine.stats.total_volume:,}")


if __name__ == "__main__":
    main()
