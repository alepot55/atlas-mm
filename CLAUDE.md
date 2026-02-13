# atlas-mm — Complete Implementation Plan for Claude Code

> **INSTRUCTIONS FOR CLAUDE CODE:** Follow this document sequentially from Phase 0 to Phase 6. Each phase has explicit deliverables and validation criteria. Do NOT proceed to the next phase until all validation checks pass. When in doubt, prefer simplicity and correctness over cleverness.

---

## PROJECT IDENTITY

**Name:** atlas-mm
**Tagline:** GPU-Accelerated Limit Order Book Simulator with Formally Verified Market Making
**Purpose:** Portfolio project for ML Engineer Intern interview at IMC Trading (Amsterdam). Must demonstrate: (1) understanding of market microstructure, (2) GPU engineering skills, (3) formal verification rigor, (4) ML applied to a real financial problem.
**Author:** Alessandro Potenza (ap.alessandro.potenza@gmail.com)
**License:** MIT
**Python:** 3.11+
**Repository:** Will be public on GitHub at github.com/alepot55/atlas-mm

---

## CRITICAL DOMAIN CONTEXT — READ BEFORE WRITING ANY CODE

### What is Market Making?

A market maker continuously quotes bid (buy) and ask (sell) prices on a financial instrument. They profit from the bid-ask spread while managing inventory risk. They do NOT predict price direction.

**Example:** Mid-price is 100.00. Market maker quotes bid=99.98, ask=100.02 (spread=0.04). If both sides fill, profit = 0.04 per unit regardless of where the price goes next.

**Key risks:**
- **Inventory risk:** If the market maker accumulates a large long/short position, adverse price moves cause losses. A market maker holding +100 shares when price drops by 1.00 loses 100.00, wiping out many spread captures.
- **Adverse selection:** Informed traders (who know price is about to move) pick off stale quotes. The market maker loses on these trades systematically.

### Avellaneda-Stoikov Model (2008)

The analytical optimal market making model. Core equations:

```
reservation_price = s - q * gamma * sigma^2 * tau
optimal_spread = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)

bid = reservation_price - optimal_spread / 2
ask = reservation_price + optimal_spread / 2
```

Where:
- `s` = current mid-price
- `q` = current inventory (positive=long, negative=short)
- `gamma` (γ) = risk aversion parameter (higher = more conservative, wider spreads)
- `sigma` (σ) = volatility estimate (annualized or per-period, be consistent)
- `tau` (T-t) = time remaining in trading session, normalized to [0, 1]
- `kappa` (κ) = order arrival intensity (higher = more aggressive quoting)

**Key insight:** The reservation price shifts away from mid when inventory != 0. If long (q>0), reservation drops below mid, making the ask more attractive → encourages selling to reduce inventory. Vice versa for short.

### Tick Size

All prices MUST be multiples of tick_size (default: 0.01). When computing bid/ask from A-S formulas, ALWAYS round to nearest tick: `round(price / tick_size) * tick_size`. The minimum spread is 1 tick.

### Order Book Mechanics

- **Limit order:** "I want to buy 10 units at price 99.95 or better." Sits in the book until filled or cancelled.
- **Market order:** "I want to buy 10 units at the best available price right now." Executes immediately against resting limit orders.
- **Price-time priority:** Orders at the same price are filled in the order they arrived (FIFO).
- **Book crossing:** MUST NEVER happen. Best bid must always be strictly less than best ask. If a new limit order would cross (buy at 100.02 when ask is 100.01), it executes as a marketable limit order against resting orders.

---

## ARCHITECTURE OVERVIEW

```
atlas-mm/
├── src/
│   └── atlas_mm/
│       ├── __init__.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── orderbook.py         # L2 order book with price-time priority
│       │   ├── orders.py            # Order dataclasses (Limit, Market, Cancel)
│       │   └── matching.py          # Matching engine orchestrator
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── price_process.py     # Mid-price dynamics (GBM + GARCH)
│       │   ├── flow_generator.py    # Synthetic order flow (Poisson arrivals)
│       │   ├── environment.py       # Gymnasium environment for RL
│       │   └── agents_zoo.py        # Background agents (noise, momentum, mean-rev)
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract MarketMakingAgent interface
│       │   ├── avellaneda_stoikov.py # Analytical baseline
│       │   └── rl_agent.py          # PPO/DQN RL agent
│       ├── verification/
│       │   ├── __init__.py
│       │   └── properties.py        # Z3 formal verification of book invariants
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py           # PnL, Sharpe, drawdown, fill rate
│           └── visualization.py     # Matplotlib plots
├── scripts/
│   ├── run_simulation.py            # Main entry point
│   ├── train_rl.py                  # RL training script
│   ├── run_verification.py          # Z3 verification runner
│   └── benchmark_engine.py          # Engine throughput benchmark
├── tests/
│   ├── test_orderbook.py
│   ├── test_matching.py
│   ├── test_avellaneda_stoikov.py
│   ├── test_environment.py
│   ├── test_verification.py
│   └── test_metrics.py
├── notebooks/
│   └── demo.ipynb                   # Showcase notebook with results
├── results/                         # Generated plots and data (gitignored except examples)
│   └── .gitkeep
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## PHASE 0: Project Scaffolding

### 0.1 Create pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "atlas-mm"
version = "0.1.0"
description = "GPU-Accelerated LOB Simulator with Formally Verified Market Making"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Alessandro Potenza", email = "ap.alessandro.potenza@gmail.com"}]

dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "gymnasium>=0.29",
    "z3-solver>=4.12",
    "tqdm>=4.66",
    "scipy>=1.11",
]

[project.optional-dependencies]
rl = [
    "stable-baselines3>=2.2",
    "torch>=2.1",
    "tensorboard>=2.15",
]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.1",
    "mypy>=1.7",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 0.2 Create .gitignore

```
__pycache__/
*.pyc
.mypy_cache/
.ruff_cache/
*.egg-info/
dist/
build/
.venv/
results/*.png
results/*.csv
results/*.json
!results/.gitkeep
*.pt
*.pth
runs/
wandb/
.DS_Store
```

### 0.3 Create all `__init__.py` files

Every `__init__.py` should have a module docstring and explicit exports. Example for `src/atlas_mm/__init__.py`:

```python
"""atlas-mm: GPU-Accelerated LOB Simulator with Formally Verified Market Making."""

__version__ = "0.1.0"
```

### 0.4 Install the project

```bash
pip install -e ".[dev]" --break-system-packages
```

### PHASE 0 VALIDATION
- [ ] `python -c "import atlas_mm; print(atlas_mm.__version__)"` prints `0.1.0`
- [ ] `pytest` runs (0 tests collected, but no import errors)
- [ ] All directories exist with `__init__.py` files

---

## PHASE 1: Order Book Engine (Core)

This is the foundation. It must be 100% correct before anything else is built.

### 1.1 orders.py — Order Types

```python
"""Order type definitions for the matching engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


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
    price: Optional[float]
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
```

### 1.2 orderbook.py — L2 Order Book

Implement a price-level aggregated order book. Each price level is a FIFO queue of orders.

**Data structures:**
- `bids`: `SortedDict` (from `sortedcontainers`) mapping price → deque of Orders, sorted descending (highest bid first). OR use a regular dict + maintain a sorted list of prices. Use `sortedcontainers` if available, else manual approach.
- `asks`: Same, sorted ascending (lowest ask first).

**IMPORTANT:** Do NOT use `sortedcontainers` — it's an extra dependency. Instead, use `heapq` for best bid/ask tracking and a `defaultdict(deque)` for the price levels. Maintain `_best_bid` and `_best_ask` manually.

Actually, for simplicity and correctness, use a **dict of deques** with **manual best price tracking**. Here's the approach:

```python
"""Limit Order Book with price-time priority matching."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from .orders import Order, Fill, Side, OrderType, CancelRequest


@dataclass
class PriceLevel:
    """A single price level in the book."""
    price: float
    orders: deque[Order] = field(default_factory=deque)
    total_quantity: int = 0


class OrderBook:
    """L2 Order Book with price-time priority.
    
    Invariants (enforced and verified):
    1. best_bid < best_ask (no crossed book) — unless book is empty on one side
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
        self._bid_prices: list[float] = []  # sorted descending
        self._ask_prices: list[float] = []  # sorted ascending
        
        # Order lookup for cancellations: order_id -> (side, price)
        self._order_index: dict[int, tuple[Side, float]] = {}
        
        # Trade log
        self._fills: list[Fill] = []
        
        # Statistics
        self._total_orders: int = 0
        self._total_fills: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price, or None if no bids."""
        return self._bid_prices[0] if self._bid_prices else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price, or None if no asks."""
        return self._ask_prices[0] if self._ask_prices else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid-price = (best_bid + best_ask) / 2, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
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
        # Implementation: see detailed logic below
        ...

    def add_market_order(self, order: Order) -> list[Fill]:
        """Execute a market order against resting liquidity.
        
        Returns list of fills. Unfilled portion is discarded (no partial resting).
        """
        ...

    def cancel_order(self, cancel: CancelRequest) -> bool:
        """Cancel a resting order. Returns True if order was found and cancelled."""
        ...

    def get_book_state(self, depth: int = 5) -> dict:
        """Return top N levels on each side for observation.
        
        Returns dict with keys: 'bids', 'asks', 'mid_price', 'spread', 'timestamp'.
        Each side is a list of (price, total_quantity) tuples.
        """
        ...
    
    def _insert_price_sorted(self, prices: list[float], price: float, ascending: bool) -> None:
        """Insert price into sorted list using bisect. Maintain sort order."""
        ...
    
    def _remove_price(self, prices: list[float], price: float) -> None:
        """Remove price from sorted list."""
        ...

    def _match_incoming_order(self, order: Order) -> tuple[list[Fill], int]:
        """Match an incoming order against the opposite side of the book.
        
        Returns (fills, remaining_quantity).
        """
        ...
```

**CRITICAL MATCHING LOGIC for `add_limit_order`:**

```
1. Validate: price > 0, quantity > 0, price is on tick grid
2. If order is a BID:
   a. While remaining_qty > 0 AND best_ask is not None AND order.price >= best_ask:
      - Take from the best ask level (FIFO)
      - Create Fill at the RESTING order's price (price improvement for aggressor)
      - If ask level is depleted, remove it
   b. If remaining_qty > 0, insert remainder into bid side of book
3. If order is an ASK:
   a. While remaining_qty > 0 AND best_bid is not None AND order.price <= best_bid:
      - Take from the best bid level (FIFO)
      - Create Fill at the RESTING order's price
      - If bid level is depleted, remove it
   b. If remaining_qty > 0, insert remainder into ask side of book
4. Return all fills generated
```

**CRITICAL:** Fills execute at the RESTING order's price, not the incoming order's price. This is price-time priority.

### 1.3 matching.py — Matching Engine Orchestrator

```python
"""Matching engine that processes order streams and maintains state."""

from __future__ import annotations

from dataclasses import dataclass, field

from .orderbook import OrderBook
from .orders import Order, Fill, CancelRequest, Side, OrderType


@dataclass
class EngineStats:
    """Running statistics of the matching engine."""
    total_orders_processed: int = 0
    total_fills: int = 0
    total_volume: int = 0
    total_cancellations: int = 0


class MatchingEngine:
    """Orchestrates order processing and maintains audit trail.
    
    This is the main interface for the simulation layer.
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.book = OrderBook(tick_size=tick_size)
        self.stats = EngineStats()
        self._next_order_id: int = 0
        self._fill_log: list[Fill] = []
        self._time: float = 0.0

    def process_order(self, side: Side, price: float | None, quantity: int,
                      agent_id: str, order_type: OrderType = OrderType.LIMIT) -> tuple[int, list[Fill]]:
        """Submit an order and return (order_id, fills).
        
        For limit orders, price must be provided.
        For market orders, price should be None.
        """
        order_id = self._next_order_id
        self._next_order_id += 1

        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self._time,
            agent_id=agent_id,
            order_type=order_type,
        )

        if order_type == OrderType.MARKET:
            fills = self.book.add_market_order(order)
        else:
            fills = self.book.add_limit_order(order)

        self._fill_log.extend(fills)
        self.stats.total_orders_processed += 1
        self.stats.total_fills += len(fills)
        self.stats.total_volume += sum(f.quantity for f in fills)

        return order_id, fills

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order."""
        success = self.book.cancel_order(CancelRequest(order_id=order_id, timestamp=self._time))
        if success:
            self.stats.total_cancellations += 1
        return success

    def advance_time(self, dt: float) -> None:
        """Advance simulation clock."""
        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    @property
    def fills(self) -> list[Fill]:
        return self._fill_log
```

### PHASE 1 VALIDATION

Write `tests/test_orderbook.py` and `tests/test_matching.py` with these test cases:

```
TEST CASES FOR ORDER BOOK (all must pass):

1. test_empty_book: new book has None for best_bid, best_ask, mid_price, spread
2. test_single_bid: add one bid, best_bid == bid price, best_ask == None
3. test_single_ask: add one ask, best_ask == ask price, best_bid == None
4. test_bid_ask_no_cross: add bid at 99.98, ask at 100.02. spread == 0.04, mid == 100.00
5. test_crossing_order_fills: add ask at 100.00, then bid at 100.00. Should produce 1 fill at 100.00. Book should be empty after.
6. test_crossing_order_partial: add ask(qty=10) at 100.00, then bid(qty=5) at 100.00. Should produce fill(qty=5). Ask side should have 5 remaining.
7. test_price_time_priority: add ask(qty=5, id=1) at 100.00, add ask(qty=5, id=2) at 100.00. Then bid(qty=5) at 100.00. Fill should be against order id=1 (first in).
8. test_price_priority: add ask(qty=5) at 100.02, add ask(qty=5) at 100.01. Then bid(qty=5) at 100.02. Fill should be at 100.01 (better price).
9. test_market_order_buy: add asks at 100.01(qty=5) and 100.02(qty=5). Market buy(qty=8). Should get 2 fills: 5@100.01 + 3@100.02. Ask side should have 2 remaining at 100.02.
10. test_market_order_no_liquidity: empty book, market buy. Should return 0 fills.
11. test_cancel_order: add bid, cancel it, best_bid should be None.
12. test_cancel_nonexistent: cancel order_id that doesn't exist, returns False.
13. test_tick_rounding: order at price 100.005 with tick=0.01 should snap to 100.01 (or 100.00 depending on rounding — use round-half-even).
14. test_book_state_depth: add 10 ask levels, get_book_state(depth=5) returns only top 5.
15. test_aggressive_limit_order: bid at 100.05 when best ask is 100.02. Should fill at 100.02 (resting price).
16. test_multiple_fills_single_order: bid at 100.05 qty=15, when asks are 100.01(5), 100.02(5), 100.03(5). Should get 3 fills sweeping the book.
17. test_invariant_no_crossed_book: after any sequence of operations, assert best_bid < best_ask (when both exist).
```

Run: `pytest tests/test_orderbook.py tests/test_matching.py -v`

ALL 17 tests must pass before proceeding to Phase 2.

---

## PHASE 2: Market Simulation Layer

### 2.1 price_process.py — Mid-Price Dynamics

Implement two price processes:

**Geometric Brownian Motion (GBM) — simple baseline:**
```
ds = mu * s * dt + sigma * s * dW
```
Where `dW ~ N(0, dt)`. Discrete: `s_{t+1} = s_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)` where `Z ~ N(0,1)`.

**GBM + GARCH(1,1) volatility — realistic:**
```
sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
```
Where `epsilon_t = sigma_t * Z_t`, with typical params: `omega=0.00001, alpha=0.1, beta=0.85` (alpha + beta < 1 for stationarity).

```python
"""Mid-price dynamics for simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PriceConfig:
    """Configuration for price process."""
    initial_price: float = 100.0
    dt: float = 1.0 / 252 / 6.5 / 3600  # ~1 second in trading-year units
    mu: float = 0.0  # zero drift for market making (no directional bias)
    sigma: float = 0.02  # annualized volatility (~2% for liquid equity)
    seed: int | None = None


@dataclass
class GarchConfig(PriceConfig):
    """Additional GARCH(1,1) parameters."""
    omega: float = 1e-5
    alpha: float = 0.1
    beta: float = 0.85


class PriceProcess(ABC):
    """Abstract base class for price processes."""
    
    def __init__(self, config: PriceConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._price = config.initial_price
        self._t = 0.0
        self._history: list[float] = [config.initial_price]

    @property
    def price(self) -> float:
        return self._price

    @abstractmethod
    def step(self) -> float:
        """Advance one time step. Returns new price."""
        ...

    def generate(self, n_steps: int) -> np.ndarray:
        """Generate n_steps of prices. Returns array of shape (n_steps+1,)."""
        prices = np.empty(n_steps + 1)
        prices[0] = self._price
        for i in range(1, n_steps + 1):
            prices[i] = self.step()
        return prices


class GBMProcess(PriceProcess):
    """Geometric Brownian Motion."""
    
    def __init__(self, config: PriceConfig) -> None:
        super().__init__(config)

    def step(self) -> float:
        dt = self.config.dt
        z = self.rng.standard_normal()
        self._price *= np.exp(
            (self.config.mu - 0.5 * self.config.sigma**2) * dt
            + self.config.sigma * np.sqrt(dt) * z
        )
        self._t += dt
        self._history.append(self._price)
        return self._price


class GarchProcess(PriceProcess):
    """GBM with GARCH(1,1) stochastic volatility."""
    
    def __init__(self, config: GarchConfig) -> None:
        super().__init__(config)
        self._garch_config = config
        self._sigma_sq = config.sigma**2  # initial variance
        self._last_epsilon = 0.0

    @property
    def current_volatility(self) -> float:
        """Current instantaneous volatility (annualized)."""
        return np.sqrt(self._sigma_sq)

    def step(self) -> float:
        cfg = self._garch_config
        dt = cfg.dt
        
        # Update GARCH variance
        self._sigma_sq = (
            cfg.omega
            + cfg.alpha * self._last_epsilon**2
            + cfg.beta * self._sigma_sq
        )
        
        sigma = np.sqrt(self._sigma_sq)
        z = self.rng.standard_normal()
        self._last_epsilon = sigma * z
        
        self._price *= np.exp(
            (cfg.mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )
        self._t += dt
        self._history.append(self._price)
        return self._price
```

### 2.2 flow_generator.py — Synthetic Order Flow

Generate realistic order flow with Poisson arrivals. Order intensity depends on distance from mid-price (more orders near the mid, fewer far away).

```python
"""Synthetic order flow generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.orders import Side, OrderType


@dataclass
class FlowConfig:
    """Configuration for order flow generation."""
    base_arrival_rate: float = 10.0      # orders per second (each side)
    price_decay: float = 2.0             # exponential decay of intensity with distance from mid
    cancel_rate: float = 0.3             # fraction of resting orders cancelled per second
    market_order_fraction: float = 0.1   # fraction of orders that are market orders
    min_quantity: int = 1
    max_quantity: int = 10
    max_levels: int = 10                 # max distance in ticks from mid for limit orders
    seed: int | None = None


@dataclass
class OrderEvent:
    """A generated order event to be processed by the engine."""
    side: Side
    price: float | None          # None for market orders
    quantity: int
    order_type: OrderType
    agent_id: str = "background"


class FlowGenerator:
    """Generates synthetic order flow for simulation.
    
    Uses Poisson process with intensity that decays exponentially
    with distance from mid-price. This creates a realistic
    order book shape (more liquidity near the mid).
    """

    def __init__(self, config: FlowConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate_orders(
        self, mid_price: float, tick_size: float, dt: float
    ) -> list[OrderEvent]:
        """Generate orders for one time step.
        
        Args:
            mid_price: Current mid-price.
            tick_size: Minimum price increment.
            dt: Time step duration in seconds.
            
        Returns:
            List of OrderEvent to submit to the engine.
        """
        events: list[OrderEvent] = []
        cfg = self.config

        for side in [Side.BID, Side.ASK]:
            # Number of orders this step (Poisson)
            n_orders = self.rng.poisson(cfg.base_arrival_rate * dt)

            for _ in range(n_orders):
                # Market or limit?
                if self.rng.random() < cfg.market_order_fraction:
                    qty = self.rng.integers(cfg.min_quantity, cfg.max_quantity + 1)
                    events.append(OrderEvent(
                        side=side, price=None, quantity=int(qty),
                        order_type=OrderType.MARKET,
                    ))
                else:
                    # Distance from mid in ticks (exponential distribution)
                    distance_ticks = int(self.rng.exponential(1.0 / cfg.price_decay)) + 1
                    distance_ticks = min(distance_ticks, cfg.max_levels)
                    
                    if side == Side.BID:
                        price = mid_price - distance_ticks * tick_size
                    else:
                        price = mid_price + distance_ticks * tick_size
                    
                    # Snap to tick grid
                    price = round(round(price / tick_size) * tick_size, 10)
                    
                    qty = self.rng.integers(cfg.min_quantity, cfg.max_quantity + 1)
                    events.append(OrderEvent(
                        side=side, price=price, quantity=int(qty),
                        order_type=OrderType.LIMIT,
                    ))

        return events
```

### 2.3 agents_zoo.py — Background Agents

```python
"""Background trading agents that populate the simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..engine.orders import Side, OrderType


@dataclass
class AgentOrder:
    """Order generated by a background agent."""
    side: Side
    price: float | None
    quantity: int
    order_type: OrderType


class BackgroundAgent(ABC):
    """Base class for background agents."""
    
    def __init__(self, agent_id: str, seed: int | None = None) -> None:
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def act(self, mid_price: float, tick_size: float, dt: float) -> list[AgentOrder]:
        """Generate orders for this time step."""
        ...


class NoiseTrader(BackgroundAgent):
    """Trades randomly. Represents uninformed liquidity."""
    
    def __init__(self, agent_id: str = "noise", arrival_rate: float = 2.0,
                 seed: int | None = None) -> None:
        super().__init__(agent_id, seed)
        self.arrival_rate = arrival_rate

    def act(self, mid_price: float, tick_size: float, dt: float) -> list[AgentOrder]:
        n = self.rng.poisson(self.arrival_rate * dt)
        orders = []
        for _ in range(n):
            side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            # Market orders with small qty
            orders.append(AgentOrder(
                side=side, price=None, quantity=self.rng.integers(1, 5),
                order_type=OrderType.MARKET,
            ))
        return orders


class MomentumTrader(BackgroundAgent):
    """Buys on up moves, sells on down moves. Represents informed flow."""
    
    def __init__(self, agent_id: str = "momentum", lookback: int = 20,
                 threshold: float = 0.001, arrival_rate: float = 1.0,
                 seed: int | None = None) -> None:
        super().__init__(agent_id, seed)
        self.lookback = lookback
        self.threshold = threshold
        self.arrival_rate = arrival_rate
        self._price_history: list[float] = []

    def act(self, mid_price: float, tick_size: float, dt: float) -> list[AgentOrder]:
        self._price_history.append(mid_price)
        if len(self._price_history) < self.lookback:
            return []
        
        ret = (mid_price - self._price_history[-self.lookback]) / self._price_history[-self.lookback]
        orders = []
        
        n = self.rng.poisson(self.arrival_rate * dt)
        for _ in range(n):
            if ret > self.threshold:
                orders.append(AgentOrder(
                    side=Side.BID, price=None, quantity=self.rng.integers(1, 8),
                    order_type=OrderType.MARKET,
                ))
            elif ret < -self.threshold:
                orders.append(AgentOrder(
                    side=Side.ASK, price=None, quantity=self.rng.integers(1, 8),
                    order_type=OrderType.MARKET,
                ))
        
        # Keep history bounded
        if len(self._price_history) > self.lookback * 2:
            self._price_history = self._price_history[-self.lookback:]
        
        return orders


class MeanReversionTrader(BackgroundAgent):
    """Buys on dips, sells on rallies. Provides liquidity."""
    
    def __init__(self, agent_id: str = "mean_reversion", lookback: int = 50,
                 threshold: float = 0.002, seed: int | None = None) -> None:
        super().__init__(agent_id, seed)
        self.lookback = lookback
        self.threshold = threshold
        self._price_history: list[float] = []

    def act(self, mid_price: float, tick_size: float, dt: float) -> list[AgentOrder]:
        self._price_history.append(mid_price)
        if len(self._price_history) < self.lookback:
            return []
        
        mean = np.mean(self._price_history[-self.lookback:])
        deviation = (mid_price - mean) / mean
        orders = []
        
        if deviation < -self.threshold:
            # Price below mean — buy
            price = mid_price - tick_size  # limit order just below mid
            price = round(round(price / tick_size) * tick_size, 10)
            orders.append(AgentOrder(
                side=Side.BID, price=price, quantity=self.rng.integers(3, 10),
                order_type=OrderType.LIMIT,
            ))
        elif deviation > self.threshold:
            # Price above mean — sell
            price = mid_price + tick_size
            price = round(round(price / tick_size) * tick_size, 10)
            orders.append(AgentOrder(
                side=Side.ASK, price=price, quantity=self.rng.integers(3, 10),
                order_type=OrderType.LIMIT,
            ))
        
        if len(self._price_history) > self.lookback * 2:
            self._price_history = self._price_history[-self.lookback:]
        
        return orders
```

### PHASE 2 VALIDATION

```
TEST CASES:

1. test_gbm_positive_prices: generate 10000 steps, all prices > 0
2. test_gbm_mean: with mu=0, mean of log returns should be approximately -sigma^2/2 * dt
3. test_gbm_volatility: std of log returns should be approximately sigma * sqrt(dt)
4. test_garch_volatility_clustering: large shocks should increase subsequent volatility
5. test_garch_stationarity: with alpha + beta < 1, variance should not explode
6. test_flow_generator_poisson: generate many steps, count orders. Mean should be close to rate * dt * n_steps
7. test_flow_generator_price_distribution: most limit orders should be near the mid (within 3 ticks)
8. test_flow_generator_market_fraction: fraction of market orders should be approximately market_order_fraction
9. test_noise_trader_symmetric: over many steps, roughly equal buy/sell
10. test_deterministic_seed: same seed produces identical sequences
```

Run: `pytest tests/test_simulation.py -v` (or split into multiple test files)

ALL tests must pass before proceeding.

---

## PHASE 3: Market Making Agents

### 3.1 base.py — Agent Interface

```python
"""Abstract interface for market making agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..engine.orders import Side


@dataclass
class Quote:
    """A two-sided quote from a market maker."""
    bid_price: float
    ask_price: float
    bid_quantity: int
    ask_quantity: int


@dataclass
class AgentState:
    """Observable state for the market making agent."""
    mid_price: float
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    inventory: int                        # current inventory (positive = long)
    cash: float                           # accumulated cash from trades
    unrealized_pnl: float                 # inventory * (mid_price - avg_entry_price)
    realized_pnl: float                   # total realized PnL
    volatility_estimate: float            # recent volatility
    order_imbalance: float                # (bid_volume - ask_volume) / total_volume at top N levels
    time_remaining: float                 # fraction of session remaining [0, 1]
    step: int                             # current simulation step


class MarketMakingAgent(ABC):
    """Abstract base class for market making agents.
    
    All agents must implement the `quote` method, which returns
    a two-sided quote given the current state.
    """

    def __init__(self, agent_id: str, max_inventory: int = 100,
                 tick_size: float = 0.01) -> None:
        self.agent_id = agent_id
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        
        # Portfolio tracking
        self.inventory: int = 0
        self.cash: float = 0.0
        self._entry_prices: list[float] = []  # for avg entry price calc
        self._realized_pnl: float = 0.0

    @abstractmethod
    def quote(self, state: AgentState) -> Quote | None:
        """Generate a two-sided quote given current state.
        
        Returns None if the agent chooses not to quote (e.g., inventory limit reached).
        """
        ...

    def on_fill(self, side: Side, price: float, quantity: int) -> None:
        """Called when one of the agent's orders is filled.
        
        Updates inventory and cash tracking.
        """
        if side == Side.BID:
            # We bought
            self.inventory += quantity
            self.cash -= price * quantity
        else:
            # We sold
            self.inventory -= quantity
            self.cash += price * quantity

    def snap_to_tick(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    @property
    def pnl(self) -> float:
        """Total PnL = cash + inventory value. Requires current mid for unrealized."""
        return self.cash  # unrealized added at evaluation time

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.inventory = 0
        self.cash = 0.0
        self._entry_prices = []
        self._realized_pnl = 0.0
```

### 3.2 avellaneda_stoikov.py — Analytical Baseline

```python
"""Avellaneda-Stoikov optimal market making model (2008).

Reference: Avellaneda, M., & Stoikov, S. (2008). High-frequency trading
in a limit order book. Quantitative Finance, 8(3), 217-224.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import MarketMakingAgent, AgentState, Quote


@dataclass
class ASConfig:
    """Avellaneda-Stoikov parameters.
    
    Attributes:
        gamma: Risk aversion. Higher = wider spreads, more inventory aversion.
               Typical range: [0.01, 1.0]. Start with 0.1.
        kappa: Order arrival intensity parameter. Higher = tighter spreads.
               Represents how likely the market maker's orders are to get filled.
               Typical range: [1.0, 10.0]. Start with 1.5.
        order_quantity: Fixed quantity per side.
    """
    gamma: float = 0.1
    kappa: float = 1.5
    order_quantity: int = 5


class AvellanedaStoikovAgent(MarketMakingAgent):
    """Avellaneda-Stoikov optimal market maker.
    
    Computes reservation price and optimal spread analytically,
    then quotes bid/ask around the reservation price.
    
    The reservation price shifts away from the mid-price
    proportionally to inventory, risk aversion, and volatility.
    This creates a natural inventory mean-reversion mechanism.
    """

    def __init__(self, config: ASConfig, agent_id: str = "avellaneda_stoikov",
                 max_inventory: int = 100, tick_size: float = 0.01) -> None:
        super().__init__(agent_id=agent_id, max_inventory=max_inventory, tick_size=tick_size)
        self.config = config

    def quote(self, state: AgentState) -> Quote | None:
        cfg = self.config
        s = state.mid_price
        q = self.inventory
        sigma = state.volatility_estimate
        tau = max(state.time_remaining, 1e-6)  # avoid division by zero

        # Reservation price: shifts away from mid when inventory != 0
        reservation_price = s - q * cfg.gamma * sigma**2 * tau

        # Optimal spread
        spread = cfg.gamma * sigma**2 * tau + (2 / cfg.gamma) * math.log(1 + cfg.gamma / cfg.kappa)
        spread = max(spread, self.tick_size)  # minimum spread is 1 tick

        # Compute bid/ask
        bid = self.snap_to_tick(reservation_price - spread / 2)
        ask = self.snap_to_tick(reservation_price + spread / 2)

        # Ensure ask > bid (at least 1 tick spread)
        if ask <= bid:
            ask = bid + self.tick_size

        # Inventory limits: don't quote the side that would exceed limits
        bid_qty = cfg.order_quantity if self.inventory < self.max_inventory else 0
        ask_qty = cfg.order_quantity if self.inventory > -self.max_inventory else 0

        if bid_qty == 0 and ask_qty == 0:
            return None

        return Quote(
            bid_price=bid,
            ask_price=ask,
            bid_quantity=bid_qty,
            ask_quantity=ask_qty,
        )
```

### 3.3 rl_agent.py — Reinforcement Learning Agent

Use Stable Baselines 3 with PPO. The agent learns to set spread and inventory skew.

```python
"""Reinforcement Learning market making agent using PPO.

The agent observes market state and outputs:
- spread_level: how wide to quote (discretized)
- skew_level: how much to skew toward reducing inventory (discretized)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import MarketMakingAgent, AgentState, Quote


@dataclass
class RLConfig:
    """RL agent configuration."""
    n_spread_levels: int = 5         # discrete spread choices (1 tick to 5 ticks)
    n_skew_levels: int = 5           # discrete skew choices (-2 ticks to +2 ticks)
    base_quantity: int = 5
    model_path: Optional[str] = None  # path to trained model


class RLMarketMaker(MarketMakingAgent):
    """PPO-based market maker.
    
    Action space: MultiDiscrete([n_spread_levels, n_skew_levels])
    - spread_level: index into [1, 2, 3, 4, 5] ticks half-spread
    - skew_level: index into [-2, -1, 0, 1, 2] ticks skew
    
    Observation space: Box with features:
    - inventory_normalized: inventory / max_inventory, in [-1, 1]
    - volatility_normalized: sigma / sigma_baseline
    - spread_normalized: current_spread / tick_size
    - time_remaining: [0, 1]
    - order_imbalance: [-1, 1]
    - recent_pnl_normalized: rolling PnL / initial_price
    """

    def __init__(self, config: RLConfig, agent_id: str = "rl_agent",
                 max_inventory: int = 100, tick_size: float = 0.01) -> None:
        super().__init__(agent_id=agent_id, max_inventory=max_inventory, tick_size=tick_size)
        self.config = config
        self._model = None
        self._spread_choices = list(range(1, config.n_spread_levels + 1))  # in ticks
        self._skew_choices = list(range(
            -(config.n_skew_levels // 2),
            config.n_skew_levels // 2 + 1,
        ))

    def load_model(self, path: str) -> None:
        """Load a trained SB3 model."""
        from stable_baselines3 import PPO
        self._model = PPO.load(path)

    def state_to_obs(self, state: AgentState) -> np.ndarray:
        """Convert AgentState to observation vector for the RL model."""
        return np.array([
            self.inventory / self.max_inventory,                   # [-1, 1]
            state.volatility_estimate / 0.02,                      # normalized to baseline
            (state.spread or 0.02) / self.tick_size,               # spread in ticks
            state.time_remaining,                                  # [0, 1]
            state.order_imbalance,                                 # [-1, 1]
            (self.cash + self.inventory * state.mid_price) / state.mid_price,  # normalized PnL
        ], dtype=np.float32)

    def quote(self, state: AgentState) -> Quote | None:
        if self._model is None:
            # Random policy for untrained agent
            spread_idx = np.random.randint(0, len(self._spread_choices))
            skew_idx = np.random.randint(0, len(self._skew_choices))
        else:
            obs = self.state_to_obs(state)
            action, _ = self._model.predict(obs, deterministic=True)
            spread_idx, skew_idx = int(action[0]), int(action[1])

        half_spread_ticks = self._spread_choices[spread_idx]
        skew_ticks = self._skew_choices[skew_idx]

        mid = state.mid_price
        bid = self.snap_to_tick(mid - half_spread_ticks * self.tick_size + skew_ticks * self.tick_size)
        ask = self.snap_to_tick(mid + half_spread_ticks * self.tick_size + skew_ticks * self.tick_size)

        # Ensure ask > bid
        if ask <= bid:
            ask = bid + self.tick_size

        bid_qty = self.config.base_quantity if self.inventory < self.max_inventory else 0
        ask_qty = self.config.base_quantity if self.inventory > -self.max_inventory else 0

        if bid_qty == 0 and ask_qty == 0:
            return None

        return Quote(
            bid_price=bid, ask_price=ask,
            bid_quantity=bid_qty, ask_quantity=ask_qty,
        )
```

### PHASE 3 VALIDATION

```
TEST CASES:

1. test_as_symmetric_quote: with zero inventory and time_remaining=1.0, bid and ask should be symmetric around mid
2. test_as_inventory_skew: with positive inventory, bid should be further from mid than ask (agent wants to sell)
3. test_as_high_volatility_wider_spread: higher sigma should produce wider spread
4. test_as_time_decay: as time_remaining -> 0, spread should decrease
5. test_as_inventory_limits: at max_inventory, should not quote the accumulating side
6. test_as_minimum_spread: spread should never be less than 1 tick
7. test_as_tick_alignment: bid and ask should always be on tick grid
8. test_rl_random_policy: untrained agent should produce valid quotes (on tick grid, ask > bid)
9. test_rl_obs_shape: state_to_obs should return array of correct shape
10. test_rl_inventory_limits: same as A-S test
```

ALL tests must pass.

---

## PHASE 4: Gymnasium Environment + RL Training

### 4.1 environment.py — Gym Environment

```python
"""Gymnasium environment for market making RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..engine.matching import MatchingEngine
from ..engine.orders import Side, OrderType
from ..simulation.price_process import GarchProcess, GarchConfig
from ..simulation.flow_generator import FlowGenerator, FlowConfig
from ..simulation.agents_zoo import NoiseTrader, MomentumTrader
from ..agents.base import AgentState


@dataclass
class EnvConfig:
    """Environment configuration."""
    n_steps: int = 5000                   # steps per episode
    tick_size: float = 0.01
    initial_price: float = 100.0
    max_inventory: int = 50
    inventory_penalty: float = 0.01       # lambda for inventory penalty in reward
    terminal_penalty: float = 0.1         # penalty for terminal inventory
    seed: int | None = None


class MarketMakingEnv(gym.Env):
    """Market making environment for RL training.
    
    Observation space (6 features):
        0: inventory / max_inventory          [-1, 1]
        1: volatility / baseline_vol          [0, inf)
        2: spread / tick_size                 [0, inf)
        3: time_remaining                     [0, 1]
        4: order_imbalance                    [-1, 1]
        5: normalized_pnl                     (-inf, inf)
    
    Action space: MultiDiscrete([5, 5])
        0: spread_level (1-5 ticks half-spread)
        1: skew_level (-2 to +2 ticks)
    
    Reward: realized_pnl_this_step - lambda * inventory^2
        This incentivizes capturing spread while penalizing inventory accumulation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, -1.0, -10.0]),
            high=np.array([1.0, 10.0, 100.0, 1.0, 1.0, 10.0]),
            dtype=np.float32,
        )
        
        self.action_space = spaces.MultiDiscrete([5, 5])
        
        # Will be initialized in reset()
        self.engine: MatchingEngine | None = None
        self.price_process = None
        self.flow_gen = None
        self._step_count = 0
        self._inventory = 0
        self._cash = 0.0
        self._prev_pnl = 0.0
        self._bid_order_id: int | None = None
        self._ask_order_id: int | None = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        cfg = self.config
        actual_seed = seed if seed is not None else cfg.seed
        
        self.engine = MatchingEngine(tick_size=cfg.tick_size)
        self.price_process = GarchProcess(GarchConfig(
            initial_price=cfg.initial_price,
            seed=actual_seed,
        ))
        self.flow_gen = FlowGenerator(FlowConfig(seed=actual_seed))
        
        # Background agents
        self._bg_agents = [
            NoiseTrader(seed=actual_seed),
            MomentumTrader(seed=(actual_seed + 1) if actual_seed else None),
        ]
        
        self._step_count = 0
        self._inventory = 0
        self._cash = 0.0
        self._prev_pnl = 0.0
        self._bid_order_id = None
        self._ask_order_id = None
        self._volatility_history: list[float] = []
        
        # Warm up: generate some initial book state
        mid = self.price_process.price
        for _ in range(50):
            events = self.flow_gen.generate_orders(mid, cfg.tick_size, dt=0.1)
            for ev in events:
                self.engine.process_order(
                    side=ev.side, price=ev.price, quantity=ev.quantity,
                    agent_id=ev.agent_id, order_type=ev.order_type,
                )
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.config
        
        # 1. Cancel previous quotes
        if self._bid_order_id is not None:
            self.engine.cancel(self._bid_order_id)
            self._bid_order_id = None
        if self._ask_order_id is not None:
            self.engine.cancel(self._ask_order_id)
            self._ask_order_id = None
        
        # 2. Decode action
        spread_ticks = int(action[0]) + 1       # 1 to 5
        skew_ticks = int(action[1]) - 2         # -2 to +2
        
        mid = self.price_process.price
        bid_price = self.engine.book.snap_to_tick(
            mid - spread_ticks * cfg.tick_size + skew_ticks * cfg.tick_size
        )
        ask_price = self.engine.book.snap_to_tick(
            mid + spread_ticks * cfg.tick_size + skew_ticks * cfg.tick_size
        )
        if ask_price <= bid_price:
            ask_price = bid_price + cfg.tick_size
        
        # 3. Place new quotes (if within inventory limits)
        qty = 5
        if self._inventory < cfg.max_inventory:
            oid, fills = self.engine.process_order(
                side=Side.BID, price=bid_price, quantity=qty,
                agent_id="rl_agent", order_type=OrderType.LIMIT,
            )
            self._bid_order_id = oid
            for f in fills:
                self._inventory += f.quantity
                self._cash -= f.price * f.quantity
        
        if self._inventory > -cfg.max_inventory:
            oid, fills = self.engine.process_order(
                side=Side.ASK, price=ask_price, quantity=qty,
                agent_id="rl_agent", order_type=OrderType.LIMIT,
            )
            self._ask_order_id = oid
            for f in fills:
                self._inventory -= f.quantity
                self._cash += f.price * f.quantity
        
        # 4. Advance price and generate background flow
        new_price = self.price_process.step()
        
        bg_events = self.flow_gen.generate_orders(new_price, cfg.tick_size, dt=1.0)
        for agent in self._bg_agents:
            for order in agent.act(new_price, cfg.tick_size, dt=1.0):
                bg_events.append(type('OrderEvent', (), {
                    'side': order.side, 'price': order.price,
                    'quantity': order.quantity, 'order_type': order.order_type,
                    'agent_id': agent.agent_id,
                })())
        
        for ev in bg_events:
            _, fills = self.engine.process_order(
                side=ev.side, price=ev.price, quantity=ev.quantity,
                agent_id=ev.agent_id, order_type=ev.order_type,
            )
            # Check if any fills hit our resting orders
            for f in fills:
                if f.bid_order_id == self._bid_order_id:
                    self._inventory += f.quantity
                    self._cash -= f.price * f.quantity
                    self._bid_order_id = None
                if f.ask_order_id == self._ask_order_id:
                    self._inventory -= f.quantity
                    self._cash += f.price * f.quantity
                    self._ask_order_id = None
        
        self.engine.advance_time(1.0)
        self._step_count += 1
        
        # 5. Compute reward
        current_pnl = self._cash + self._inventory * self.price_process.price
        step_pnl = current_pnl - self._prev_pnl
        inventory_penalty = cfg.inventory_penalty * self._inventory**2
        reward = step_pnl - inventory_penalty
        self._prev_pnl = current_pnl
        
        # 6. Check termination
        terminated = self._step_count >= cfg.n_steps
        truncated = False
        
        if terminated:
            # Terminal inventory penalty
            reward -= cfg.terminal_penalty * abs(self._inventory) * self.price_process.price * cfg.tick_size
        
        obs = self._get_obs()
        info = {
            "inventory": self._inventory,
            "cash": self._cash,
            "pnl": current_pnl,
            "mid_price": self.price_process.price,
        }
        
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        mid = self.price_process.price
        book = self.engine.book
        
        # Volatility estimate from price history
        history = self.price_process._history
        if len(history) > 20:
            returns = np.diff(np.log(history[-21:]))
            vol = float(np.std(returns))
        else:
            vol = 0.02
        
        # Order imbalance
        book_state = book.get_book_state(depth=5)
        bid_vol = sum(qty for _, qty in book_state.get("bids", []))
        ask_vol = sum(qty for _, qty in book_state.get("asks", []))
        total_vol = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
        
        spread = book.spread or (2 * cfg.tick_size)
        pnl = self._cash + self._inventory * mid
        
        return np.array([
            self._inventory / cfg.max_inventory,
            vol / 0.02,
            spread / cfg.tick_size,
            1.0 - self._step_count / cfg.n_steps,
            imbalance,
            pnl / mid,
        ], dtype=np.float32)
```

### 4.2 scripts/train_rl.py — Training Script

```python
"""Train RL market making agent with PPO."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(args.output).parent / "best"),
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
```

### PHASE 4 VALIDATION

```
1. test_env_reset: env.reset() returns obs of correct shape, info dict
2. test_env_step: env.step(action) returns (obs, reward, terminated, truncated, info) with correct types
3. test_env_episode: run full episode (n_steps), verify terminated=True at end
4. test_env_inventory_tracking: place orders, verify inventory in info matches expected
5. test_env_reward_shape: reward should be finite for all steps
6. test_sb3_compatibility: PPO("MlpPolicy", env) should instantiate without errors
7. test_training_smoke: train for 1000 timesteps without errors
```

---

## PHASE 5: Z3 Formal Verification

### 5.1 verification/properties.py

```python
"""Formal verification of order book and market making invariants using Z3.

Proves mathematical properties that must hold for ALL possible inputs,
not just tested cases. This is the key differentiator of this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from z3 import (
    Real, Int, Bool, And, Or, Not, Implies, ForAll, Exists,
    Solver, sat, unsat, unknown, RealVal, IntVal,
    If, simplify,
)


@dataclass
class VerificationResult:
    """Result of a formal verification check."""
    property_name: str
    holds: bool
    counterexample: Optional[dict] = None
    solver_time_ms: float = 0.0
    description: str = ""


class OrderBookVerifier:
    """Formally verify properties of the order book and market making logic.
    
    Each method attempts to prove a property holds for ALL possible
    inputs. If the property fails, a concrete counterexample is returned.
    """

    def verify_no_crossed_book(self) -> VerificationResult:
        """Prove: After matching, best_bid < best_ask (when both exist).
        
        Formalization: For any incoming limit order with price p on side s,
        after matching, if the book has both bids and asks,
        then max(bids) < min(asks).
        """
        solver = Solver()
        
        # Variables
        best_bid = Real("best_bid")
        best_ask = Real("best_ask")
        incoming_price = Real("incoming_price")
        incoming_is_bid = Bool("incoming_is_bid")
        
        # Pre-conditions: valid book state before order
        solver.add(best_bid > 0)
        solver.add(best_ask > 0)
        solver.add(best_bid < best_ask)  # book is not crossed before
        solver.add(incoming_price > 0)
        
        # Post-condition after matching logic:
        # If incoming is a bid at price p >= best_ask, it matches (fills).
        # The remaining (if any) rests at p, but p < new_best_ask after fills.
        # If incoming is a bid at price p < best_ask, it rests. new_best_bid = max(best_bid, p).
        # In both cases, new_best_bid < new_best_ask.
        
        new_best_bid = Real("new_best_bid")
        new_best_ask = Real("new_best_ask")
        
        # Case: incoming bid
        bid_case = And(
            incoming_is_bid,
            # If crosses: fills at best_ask, remaining rests below new best_ask
            If(incoming_price >= best_ask,
               And(new_best_bid == incoming_price, new_best_ask > incoming_price),
               # If doesn't cross: rests in book
               And(new_best_bid == If(incoming_price > best_bid, incoming_price, best_bid),
                   new_best_ask == best_ask)),
        )
        
        # Case: incoming ask
        ask_case = And(
            Not(incoming_is_bid),
            If(incoming_price <= best_bid,
               And(new_best_ask == incoming_price, new_best_bid < incoming_price),
               And(new_best_ask == If(incoming_price < best_ask, incoming_price, best_ask),
                   new_best_bid == best_bid)),
        )
        
        solver.add(Or(bid_case, ask_case))
        
        # Try to find a counterexample where the book IS crossed after
        solver.add(new_best_bid >= new_best_ask)
        
        import time
        start = time.time()
        result = solver.check()
        elapsed = (time.time() - start) * 1000
        
        if result == unsat:
            return VerificationResult(
                property_name="no_crossed_book",
                holds=True,
                solver_time_ms=elapsed,
                description="Proved: matching engine never produces a crossed book.",
            )
        elif result == sat:
            model = solver.model()
            return VerificationResult(
                property_name="no_crossed_book",
                holds=False,
                counterexample={str(d): str(model[d]) for d in model.decls()},
                solver_time_ms=elapsed,
                description="COUNTEREXAMPLE FOUND: book can become crossed.",
            )
        else:
            return VerificationResult(
                property_name="no_crossed_book",
                holds=False,
                solver_time_ms=elapsed,
                description="Solver returned unknown.",
            )

    def verify_as_spread_positive(self) -> VerificationResult:
        """Prove: Avellaneda-Stoikov spread is always positive.
        
        spread = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
        
        For gamma > 0, sigma > 0, tau > 0, kappa > 0:
        Both terms are positive, so spread > 0.
        """
        solver = Solver()
        
        gamma = Real("gamma")
        sigma = Real("sigma")
        tau = Real("tau")
        kappa = Real("kappa")
        
        # Pre-conditions (all parameters positive)
        solver.add(gamma > 0)
        solver.add(sigma > 0)
        solver.add(tau > 0)
        solver.add(kappa > 0)
        
        # Term 1: gamma * sigma^2 * tau
        term1 = gamma * sigma * sigma * tau
        
        # Term 2: (2/gamma) * ln(1 + gamma/kappa)
        # Since gamma > 0 and kappa > 0: gamma/kappa > 0
        # So 1 + gamma/kappa > 1, and ln(1 + gamma/kappa) > 0
        # And 2/gamma > 0
        # We model ln(x) > 0 for x > 1 as an axiom
        ln_arg = 1 + gamma / kappa
        ln_value = Real("ln_value")
        solver.add(ln_arg > 1)
        solver.add(ln_value > 0)  # axiom: ln(x) > 0 for x > 1
        
        term2 = (2 / gamma) * ln_value
        
        spread = term1 + term2
        
        # Try to find counterexample where spread <= 0
        solver.add(spread <= 0)
        
        import time
        start = time.time()
        result = solver.check()
        elapsed = (time.time() - start) * 1000
        
        if result == unsat:
            return VerificationResult(
                property_name="as_spread_positive",
                holds=True,
                solver_time_ms=elapsed,
                description="Proved: A-S optimal spread is always positive for valid parameters.",
            )
        else:
            model = solver.model() if result == sat else None
            return VerificationResult(
                property_name="as_spread_positive",
                holds=False,
                counterexample={str(d): str(model[d]) for d in model.decls()} if model else None,
                solver_time_ms=elapsed,
            )

    def verify_as_inventory_mean_reversion(self) -> VerificationResult:
        """Prove: A-S reservation price creates inventory mean reversion.
        
        When inventory q > 0 (long), reservation_price < mid_price,
        so the ask becomes more attractive → encourages selling → reduces inventory.
        When q < 0 (short), reservation_price > mid_price → encourages buying.
        """
        solver = Solver()
        
        s = Real("mid_price")
        q = Real("inventory")
        gamma = Real("gamma")
        sigma = Real("sigma")
        tau = Real("tau")
        
        solver.add(s > 0)
        solver.add(gamma > 0)
        solver.add(sigma > 0)
        solver.add(tau > 0)
        
        reservation = s - q * gamma * sigma * sigma * tau
        
        # Property: when q > 0, reservation < s (and vice versa)
        # Try to find counterexample
        solver.add(Or(
            And(q > 0, reservation >= s),
            And(q < 0, reservation <= s),
        ))
        
        import time
        start = time.time()
        result = solver.check()
        elapsed = (time.time() - start) * 1000
        
        if result == unsat:
            return VerificationResult(
                property_name="as_inventory_mean_reversion",
                holds=True,
                solver_time_ms=elapsed,
                description="Proved: A-S reservation price always pushes inventory toward zero.",
            )
        else:
            model = solver.model() if result == sat else None
            return VerificationResult(
                property_name="as_inventory_mean_reversion",
                holds=False,
                counterexample={str(d): str(model[d]) for d in model.decls()} if model else None,
                solver_time_ms=elapsed,
            )

    def verify_price_time_priority(self) -> VerificationResult:
        """Prove: Orders at better prices always fill before orders at worse prices.
        
        For two resting ask orders at prices p1 < p2 (p1 is better for the seller?
        No — p1 is better for the BUYER, so p1 fills first when a buy comes in).
        
        Formalization: If ask_order_1.price < ask_order_2.price, and a bid comes in
        that can fill at least one, ask_order_1 fills first.
        """
        solver = Solver()
        
        p1 = Real("ask_price_1")
        p2 = Real("ask_price_2")
        t1 = Int("arrival_time_1")
        t2 = Int("arrival_time_2")
        bid_price = Real("bid_price")
        fill_order = Int("fill_order")  # 1 if order1 fills first, 2 if order2
        
        # Pre-conditions
        solver.add(p1 > 0)
        solver.add(p2 > 0)
        solver.add(p1 < p2)  # order 1 has better (lower) ask price
        solver.add(bid_price >= p1)  # bid can fill at least order 1
        solver.add(t1 >= 0)
        solver.add(t2 >= 0)
        
        # Price-time priority: lower price fills first, regardless of arrival time
        # Try to find case where order 2 fills first
        solver.add(fill_order == 2)
        
        # The matching rule: fill_order should be 1 when p1 < p2
        solver.add(Implies(p1 < p2, fill_order == 1))
        
        import time
        start = time.time()
        result = solver.check()
        elapsed = (time.time() - start) * 1000
        
        if result == unsat:
            return VerificationResult(
                property_name="price_time_priority",
                holds=True,
                solver_time_ms=elapsed,
                description="Proved: Better-priced orders always fill before worse-priced orders.",
            )
        else:
            model = solver.model() if result == sat else None
            return VerificationResult(
                property_name="price_time_priority",
                holds=False,
                counterexample={str(d): str(model[d]) for d in model.decls()} if model else None,
                solver_time_ms=elapsed,
            )

    def verify_all(self) -> list[VerificationResult]:
        """Run all verification checks."""
        results = [
            self.verify_no_crossed_book(),
            self.verify_as_spread_positive(),
            self.verify_as_inventory_mean_reversion(),
            self.verify_price_time_priority(),
        ]
        return results
```

### 5.2 scripts/run_verification.py

```python
"""Run formal verification checks and print results."""

from atlas_mm.verification.properties import OrderBookVerifier


def main():
    verifier = OrderBookVerifier()
    results = verifier.verify_all()
    
    print("\n" + "="*70)
    print("FORMAL VERIFICATION RESULTS")
    print("="*70)
    
    all_pass = True
    for r in results:
        status = "PROVED" if r.holds else "FAILED"
        icon = "✓" if r.holds else "✗"
        print(f"\n{icon} [{status}] {r.property_name} ({r.solver_time_ms:.1f}ms)")
        print(f"  {r.description}")
        if r.counterexample:
            print(f"  Counterexample: {r.counterexample}")
        if not r.holds:
            all_pass = False
    
    print("\n" + "="*70)
    print(f"{'ALL PROPERTIES VERIFIED' if all_pass else 'SOME PROPERTIES FAILED'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
```

### PHASE 5 VALIDATION

```
1. test_no_crossed_book_holds: verify_no_crossed_book().holds == True
2. test_as_spread_positive_holds: verify_as_spread_positive().holds == True
3. test_as_inventory_mean_reversion_holds: verify_as_inventory_mean_reversion().holds == True
4. test_price_time_priority_holds: verify_price_time_priority().holds == True
5. test_verify_all: all results in verify_all() have holds == True
6. Run: python scripts/run_verification.py — all 4 properties should show PROVED
```

---

## PHASE 6: Evaluation, Visualization, and README

### 6.1 evaluation/metrics.py

```python
"""Performance metrics for market making evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    """Complete evaluation of a market making simulation run."""
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    max_inventory: int
    mean_inventory: float
    inventory_std: float
    total_fills: int
    fill_rate: float              # fills / total_quotes
    mean_spread_quoted: float     # average spread the agent quoted
    total_volume: int
    pnl_per_trade: float


def compute_metrics(
    pnl_series: np.ndarray,
    inventory_series: np.ndarray,
    spread_series: np.ndarray,
    n_quotes: int,
    n_fills: int,
) -> EvaluationMetrics:
    """Compute evaluation metrics from simulation data."""
    
    # PnL metrics
    returns = np.diff(pnl_series)
    total_pnl = float(pnl_series[-1] - pnl_series[0])
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 6.5 * 3600))
    
    # Drawdown
    cummax = np.maximum.accumulate(pnl_series)
    drawdowns = cummax - pnl_series
    max_dd = float(np.max(drawdowns))
    
    # Inventory
    max_inv = int(np.max(np.abs(inventory_series)))
    mean_inv = float(np.mean(inventory_series))
    inv_std = float(np.std(inventory_series))
    
    # Fill rate
    fill_rate = n_fills / max(n_quotes, 1)
    
    # Spread
    mean_spread = float(np.mean(spread_series)) if len(spread_series) > 0 else 0.0
    
    return EvaluationMetrics(
        total_pnl=total_pnl,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        max_inventory=max_inv,
        mean_inventory=mean_inv,
        inventory_std=inv_std,
        total_fills=n_fills,
        fill_rate=fill_rate,
        mean_spread_quoted=mean_spread,
        total_volume=n_fills * 5,  # approximate
        pnl_per_trade=total_pnl / max(n_fills, 1),
    )
```

### 6.2 evaluation/visualization.py

Create 4 publication-quality figures:

1. **PnL comparison:** A-S vs RL cumulative PnL over time (line plot, two lines)
2. **Inventory distribution:** Histogram of inventory over time for both agents
3. **Spread dynamics:** Time series of quoted spread for both agents, with volatility overlay
4. **Summary table:** Bar chart or table comparing all metrics side-by-side

Use `matplotlib` with `seaborn` style. Colors: `#2196F3` (blue) for A-S, `#FF5722` (orange) for RL.

```python
"""Visualization for market making evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import EvaluationMetrics


def set_style():
    """Set publication-quality plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


AS_COLOR = "#2196F3"
RL_COLOR = "#FF5722"


def plot_pnl_comparison(
    as_pnl: np.ndarray, rl_pnl: np.ndarray,
    output_path: str | Path = "results/pnl_comparison.png",
) -> None:
    """Plot cumulative PnL for both agents."""
    set_style()
    fig, ax = plt.subplots()
    
    steps = np.arange(len(as_pnl))
    ax.plot(steps, as_pnl, color=AS_COLOR, label="Avellaneda-Stoikov", linewidth=1.5)
    ax.plot(steps, rl_pnl, color=RL_COLOR, label="RL Agent (PPO)", linewidth=1.5)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Market Making PnL: Analytical vs RL")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    plt.savefig(output_path)
    plt.close()


def plot_inventory_distribution(
    as_inventory: np.ndarray, rl_inventory: np.ndarray,
    output_path: str | Path = "results/inventory_distribution.png",
) -> None:
    """Plot inventory histograms for both agents."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(as_inventory, bins=50, color=AS_COLOR, alpha=0.7, edgecolor="white")
    ax1.set_title("A-S Inventory Distribution")
    ax1.set_xlabel("Inventory")
    ax1.set_ylabel("Frequency")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    
    ax2.hist(rl_inventory, bins=50, color=RL_COLOR, alpha=0.7, edgecolor="white")
    ax2.set_title("RL Inventory Distribution")
    ax2.set_xlabel("Inventory")
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    
    plt.suptitle("Inventory Distribution Comparison")
    plt.savefig(output_path)
    plt.close()


def plot_spread_dynamics(
    as_spread: np.ndarray, rl_spread: np.ndarray,
    volatility: np.ndarray,
    output_path: str | Path = "results/spread_dynamics.png",
) -> None:
    """Plot spread over time with volatility overlay."""
    set_style()
    fig, ax1 = plt.subplots()
    
    steps = np.arange(len(as_spread))
    ax1.plot(steps, as_spread, color=AS_COLOR, label="A-S Spread", alpha=0.7, linewidth=0.8)
    ax1.plot(steps, rl_spread, color=RL_COLOR, label="RL Spread", alpha=0.7, linewidth=0.8)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Quoted Spread")
    ax1.legend(loc="upper left")
    
    ax2 = ax1.twinx()
    ax2.plot(steps[:len(volatility)], volatility, color="gray", alpha=0.3, linewidth=0.8, label="Volatility")
    ax2.set_ylabel("Volatility", color="gray")
    ax2.legend(loc="upper right")
    
    ax1.set_title("Spread Dynamics vs Volatility")
    plt.savefig(output_path)
    plt.close()


def plot_metrics_comparison(
    as_metrics: EvaluationMetrics, rl_metrics: EvaluationMetrics,
    output_path: str | Path = "results/metrics_comparison.png",
) -> None:
    """Bar chart comparing key metrics."""
    set_style()
    
    metric_names = ["Total PnL", "Sharpe", "Max DD", "Mean |Inv|", "Fill Rate"]
    as_values = [as_metrics.total_pnl, as_metrics.sharpe_ratio, as_metrics.max_drawdown,
                 as_metrics.inventory_std, as_metrics.fill_rate]
    rl_values = [rl_metrics.total_pnl, rl_metrics.sharpe_ratio, rl_metrics.max_drawdown,
                 rl_metrics.inventory_std, rl_metrics.fill_rate]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, as_values, width, label="Avellaneda-Stoikov", color=AS_COLOR)
    ax.bar(x + width/2, rl_values, width, label="RL Agent (PPO)", color=RL_COLOR)
    
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    
    plt.savefig(output_path)
    plt.close()
```

### 6.3 scripts/run_simulation.py — Main Entry Point

Create a script that:
1. Initializes the matching engine, price process, and flow generator
2. Runs the A-S agent for N steps, logging PnL, inventory, spread
3. Resets and runs the RL agent (or random RL if untrained) for N steps
4. Computes metrics for both
5. Generates all 4 plots
6. Prints a summary table to stdout

### 6.4 README.md

Write a professional README with:

1. **Header:** Project name, one-line description, badges (Python, PyTorch, Z3, MIT)
2. **The Problem:** Why market making is hard (2-3 sentences)
3. **The Approach:** Three components — LOB engine, A-S vs RL comparison, Z3 verification
4. **Results:** Table of metrics (PnL, Sharpe, drawdown, inventory) + key figures
5. **Formal Verification:** Table showing 4 proved properties with solver times
6. **Architecture:** File tree diagram
7. **Quick Start:** Installation + `python scripts/run_simulation.py`
8. **How It Works:** Brief explanation of A-S model, RL formulation, and Z3 proofs
9. **References:** Avellaneda-Stoikov (2008), Guéant et al. (2013), Falces Marin et al. (2022)
10. **Citation:** BibTeX entry

Style: Technical but accessible. Use the same README quality as Flash-Reasoning and Flash-SAE (already in the project files as reference).

### PHASE 6 VALIDATION

```
1. python scripts/run_simulation.py completes without errors
2. 4 PNG files generated in results/
3. python scripts/run_verification.py shows 4/4 PROVED
4. README.md renders correctly on GitHub (no broken links/images)
5. Full test suite: pytest -v passes all tests (target: 30+ tests total)
6. ruff check src/ -- no errors
7. mypy src/atlas_mm/ -- no errors (or minimal)
```

---

## FINAL CHECKLIST BEFORE PUBLISHING

- [ ] All tests pass (`pytest -v`)
- [ ] Linting clean (`ruff check src/`)
- [ ] Type checking clean (`mypy src/atlas_mm/`)
- [ ] `run_simulation.py` generates all plots
- [ ] `run_verification.py` shows 4/4 PROVED
- [ ] README has results table, figures, verification table
- [ ] `.gitignore` includes model files, generated results
- [ ] No hardcoded absolute paths
- [ ] All random operations use configurable seeds for reproducibility
- [ ] License file (MIT) exists
- [ ] Clean git history (no huge binary files)

---

## INTERVIEW TALKING POINTS (for Alessandro, not for Claude Code)

When presenting this project at the IMC interview:

1. **Lead with business context:** "Market makers need to quote optimally under inventory risk. I built a simulator to compare the analytical solution (Avellaneda-Stoikov) against a learned policy."

2. **Highlight the Z3 angle:** "Unlike testing, formal verification proves properties hold for ALL possible inputs. I proved the matching engine never crosses the book and the A-S model always produces mean-reverting quotes."

3. **Connect to IMC's work:** "Your ML engineering team builds GPU-accelerated training pipelines and low-latency inference — my background in CUDA kernel optimization for LLM inference translates directly."

4. **Show depth:** Be ready to explain: why inventory penalty is quadratic (risk scales with variance, variance scales with position squared), what adverse selection is and how the momentum trader simulates it, why the RL agent uses PPO over DQN (continuous-like action space, stable training).
