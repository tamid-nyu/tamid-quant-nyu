# BTC Kalshi Tail Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A walk-forward backtest that trades only Kalshi BTC tail contracts (mid < 10% or > 90%, 3h/6h before expiry) and sizes each expiry day's book so its worst settlement loses at most 3% of bankroll.

**Architecture:** `tail_sizer.py` holds pure functions: the settlement payoff matrix, the worst-case check, and the LP sizer (plus the per-trade baseline). `tail_strategy.py` holds the tail filter, the walk-forward edge estimator, the day-by-day simulator (portfolio / one-contract / per-trade sizing modes), reporting and the CLI. It reads the existing `btc_observations.csv` and changes no existing code.

**Tech Stack:** Python 3 (`/opt/miniconda3/bin/python3`), numpy, pandas, scipy (`optimize.linprog`, HiGHS), matplotlib; pytest (dev only).

**Spec:** `docs/superpowers/specs/2026-11-11-btc-tail-strategy-design.md`

## Global Constraints

- All paths below are relative to `strategies/btc-kalshi-digital/` unless they start with the repo root `docs/` or `requirements`.
- Interpreter: `/opt/miniconda3/bin/python3` (the only one on this machine with pandas/scipy). Run tests from `strategies/btc-kalshi-digital/` as `/opt/miniconda3/bin/python3 -m pytest tests -q`.
- Runtime deps limited to numpy, pandas, scipy, matplotlib (already in `requirements.txt`). `pytest` goes in a new repo-root `requirements-dev.txt`.
- Do not modify `btc_digital_model_backtest.py`, `btc_pm_data_pull.py` or anything under `output/btc_pm_results/`.
- Constants (exact values from the spec): `BANKROLL = 1000.0`, `CAP_FRAC = 0.03`, `Q_MAX = 50`, `MIN_EDGE = 0.005`, `SLIP = 0.02`, `N0 = 50`, `MIN_CELL_N = 100`, `WARMUP_DAYS = 30`, `HOURS = (6, 3)`, `MAX_SPREAD = 0.06`, `FEE_RATE = 0.07`, `TAIL_LO = 0.10`, `TAIL_HI = 0.90`, `BINS = [0, .005, .015, .025, .035, .05, .075, .10, .90, .925, .95, .965, .975, .985, .995, 1.0001]`, rounding scale-down step 5%.
- Fee: `FEE_RATE * p * (1 - p)` per contract at the price used.
- Side convention everywhere: `+1` = buy YES, `-1` = sell YES. A contract is YES iff BTC at expiry > `floor`.
- Input: `output/btc_pm_results/btc_observations.csv`. Outputs: `output/btc_tail_results/`.
- Git: work on the current branch `btc-digital-model-backtest`. Stage only the files each task names (the repo has many unrelated untracked files). Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Fill quote unusable at the fill hour (`ask_f >= 1.0` for a buy, `bid_f <= 0.0` for a sell): the trade is treated as unfilled (no position, no P&L), not filled at 0 or 1. Test in Task 4.
2. An expiry day that has 6h observations but no 3h observations: 6h positions still settle and the day appears in the daily table. Test in Task 4.
3. A fill worse than the slippage buffer: the day's `realized_worst` must come out worse than `assumed_worst`, and `breach` must be set whenever `realized_worst` is past the cap, never silently clipped. Test in Task 4.
4. A mid exactly at 0.10 or 0.90 is not a tail candidate (strict inequalities). Test in Task 3.
5. No trades at all (e.g. fewer than 30 prior days): the run returns empty trades and a daily table of zeros, and the summary prints "no trades" instead of crashing. Tests in Tasks 4 and 5.

---

### Task 1: Payoff matrix and worst case (`tail_sizer.py`), test scaffolding

**Files:**
- Create: `tail_sizer.py`
- Create: `tests/conftest.py` (inserts the parent directory into `sys.path`)
- Create: `tests/test_tail_sizer.py`
- Create: repo-root `requirements-dev.txt` containing `pytest>=7`

**Interfaces:**
- Produces:
  - `fee(p: float | np.ndarray, rate: float = 0.07) -> float | np.ndarray`
  - `payoff_matrix(grid: np.ndarray, strikes: np.ndarray, sides: np.ndarray, prices: np.ndarray, fee_rate: float = 0.07) -> np.ndarray` with shape `(len(grid) + 1, len(strikes))`. `grid` is the sorted unique strikes of the day. Row 0 = settlement at or below `grid[0]` (every contract NO); row j >= 1 = settlement just above `grid[j-1]` (contract i is YES iff `strikes[i] <= grid[j-1]`). Entry = `side * (yes - price) - fee(price)`.
  - `worst_case(P: np.ndarray, q: np.ndarray, base: np.ndarray | None = None) -> float` = `min(base + P @ q)`; `base` defaults to zeros.

- [ ] **Step 1: Install pytest and write failing tests**

Run: `/opt/miniconda3/bin/python3 -m pip install "pytest>=7"`

```python
from tail_sizer import fee, payoff_matrix, worst_case
import numpy as np

def test_fee_values():
    assert fee(0.5) == pytest.approx(0.0175)
    assert fee(0.05) == pytest.approx(0.07 * 0.05 * 0.95)

def test_payoff_matrix_short_and_long():
    grid = np.array([100.0, 200.0])
    P = payoff_matrix(grid, np.array([200.0, 100.0]), np.array([-1, 1]), np.array([0.05, 0.95]))
    assert P.shape == (3, 2)
    # short YES @0.05 on K=200: wins 0.05 unless settlement > 200
    assert P[:, 0] == pytest.approx([0.05 - fee(0.05), 0.05 - fee(0.05), 0.05 - 1 - fee(0.05)])
    # long YES @0.95 on K=100: loses 0.95 only when settlement <= 100
    assert P[:, 1] == pytest.approx([-0.95 - fee(0.95), 0.05 - fee(0.95), 0.05 - fee(0.95)])

def test_worst_case_with_base():
    P = np.array([[1.0], [-2.0]])
    assert worst_case(P, np.array([3.0])) == pytest.approx(-6.0)
    assert worst_case(P, np.array([3.0]), base=np.array([0.0, 4.0])) == pytest.approx(-2.0)
```

- [ ] **Step 2: Run tests, expect FAIL** (`ModuleNotFoundError: tail_sizer`)

Run: `/opt/miniconda3/bin/python3 -m pytest tests -q`

- [ ] **Step 3: Implement `fee`, `payoff_matrix`, `worst_case` in `tail_sizer.py`** (vectorized numpy; YES matrix via `strikes[None, :] <= grid[:, None]` with a leading all-False row).

- [ ] **Step 4: Run tests, expect PASS**

Run: `/opt/miniconda3/bin/python3 -m pytest tests -q`

- [ ] **Step 5: Commit**

```bash
git add strategies/btc-kalshi-digital/tail_sizer.py strategies/btc-kalshi-digital/tests/conftest.py strategies/btc-kalshi-digital/tests/test_tail_sizer.py requirements-dev.txt
git commit -m "Add tail sizer payoff matrix and worst-case check"
```

### Task 2: LP sizer and per-trade baseline (`tail_sizer.py`)

**Files:**
- Modify: `tail_sizer.py`
- Test: `tests/test_tail_sizer.py`

**Interfaces:**
- Consumes: `fee`, `payoff_matrix`, `worst_case` (Task 1).
- Produces:
  - `size_portfolio(edges: np.ndarray, P: np.ndarray, base: np.ndarray, cap: float, q_max: int) -> np.ndarray` (int contracts, same length as `edges`). Maximizes `edges @ q` subject to `base + P @ q >= -cap` in every row and `0 <= q <= q_max`. Returns all zeros if `len(edges) == 0`, the LP is infeasible, or `worst_case(P, 0, base) < -cap`.
  - `size_per_trade(P: np.ndarray, cap: float, q_max: int) -> np.ndarray`: `q_i = min(q_max, floor(cap / max(-min(P[:, i]), 1e-12)))`, ignoring other positions.

- [ ] **Step 1: Write failing tests**

```python
from tail_sizer import size_portfolio, size_per_trade
CAP = 30.0

def _book(strikes, sides, prices):
    grid = np.unique(strikes)
    return payoff_matrix(grid, np.asarray(strikes, float), np.asarray(sides), np.asarray(prices, float))

def test_random_books_never_exceed_cap():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 8)
        strikes = rng.choice(np.arange(60_000, 70_000, 250), n)
        sides = rng.choice([-1, 1], n)
        prices = np.where(sides == -1, rng.uniform(0.01, 0.10, n), rng.uniform(0.90, 0.99, n))
        P = _book(strikes, sides, prices)
        q = size_portfolio(rng.uniform(0.005, 0.03, n), P, np.zeros(P.shape[0]), CAP, 50)
        assert q.dtype.kind == "i" and (q >= 0).all() and (q <= 50).all()
        assert worst_case(P, q) >= -CAP - 1e-9

def test_nested_shorts_share_one_budget():
    P1 = _book([66_000], [-1], [0.05])
    alone = size_portfolio(np.array([0.01]), P1, np.zeros(2), CAP, 100)
    P2 = _book([65_000, 66_000], [-1, -1], [0.05, 0.05])
    pair = size_portfolio(np.array([0.01, 0.01]), P2, np.zeros(3), CAP, 100)
    assert alone[0] == 31 and pair.sum() <= alone[0]

def test_two_sided_book_gets_more_room():
    P = _book([66_000, 60_000], [-1, 1], [0.05, 0.95])
    q = size_portfolio(np.array([0.01, 0.01]), P, np.zeros(3), CAP, 100)
    assert q.sum() > 31 and worst_case(P, q) >= -CAP - 1e-9

def test_base_already_breaching_returns_zeros():
    P = _book([66_000], [-1], [0.05])
    assert size_portfolio(np.array([0.01]), P, np.array([0.0, -31.0]), CAP, 50).sum() == 0

def test_per_trade_ignores_nesting():
    P = _book([65_000, 66_000], [-1, -1], [0.05, 0.05])
    q = size_per_trade(P, CAP, 100)
    assert list(q) == [31, 31] and worst_case(P, q) < -CAP
```

- [ ] **Step 2: Run tests, expect FAIL** (`ImportError: size_portfolio`)

- [ ] **Step 3: Implement both functions**

LP: `linprog(c=-edges, A_ub=-P, b_ub=base + cap, bounds=[(0, q_max)] * n, method="highs")`. Rounding: `scale = 1.0`; loop `q = floor(x * scale).astype(int)`; stop when `worst_case(P, q, base) >= -cap - 1e-9`; else `scale *= 0.95` (stop at all zeros).

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add strategies/btc-kalshi-digital/tail_sizer.py strategies/btc-kalshi-digital/tests/test_tail_sizer.py
git commit -m "Add LP worst-case sizer and per-trade baseline"
```

### Task 3: Tail filter and walk-forward edge (`tail_strategy.py`)

**Files:**
- Create: `tail_strategy.py`
- Create: `tests/test_tail_strategy.py`

**Interfaces:**
- Consumes: `fee` (Task 1).
- Produces:
  - Module constants from Global Constraints, plus `@dataclass Config` with fields `bankroll, cap_frac, q_max, min_edge, slip, n0, min_cell_n, warmup_days, fee_rate` defaulting to those constants and a property `cap = bankroll * cap_frac`.
  - `load_obs(path: str | Path) -> pd.DataFrame`: reads the CSV, keeps `H in HOURS` and `strike_type == "greater"`, parses `day` to `datetime.date`, adds `mid = (bid + ask) / 2`.
  - `tail_candidates(obs: pd.DataFrame) -> pd.DataFrame`: rows with `(mid < TAIL_LO) | (mid > TAIL_HI)`, `bid >= 0.01`, `ask <= 0.99`, `ask - bid <= MAX_SPREAD`.
  - `class CellStats(n0: float, bins: list[float])`: `update(df: pd.DataFrame) -> None` adds rows (needs `H, mid, y`) to per-`(H, bin)` counts `n, sum_y, sum_mid`; `estimate(H: int, mid: float) -> tuple[float, int]` returns `(f, n)`, `f = (sum_y + n0 * m) / (n + n0)` with `m = sum_mid / n`, and `(nan, 0)` if `n == 0`. Bin index = `np.searchsorted(bins, mid, side="right") - 1`.
  - `best_side(bid: float, ask: float, f: float, cfg: Config) -> tuple[int, float, float]` returns `(side, size_price, edge)`: `buy_price = min(ask + slip, 0.99)`, `sell_price = max(bid - slip, 0.01)`, `buy edge = f - buy_price - fee(buy_price)`, `sell edge = sell_price - f - fee(sell_price)`; returns the larger edge's side.

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd, numpy as np, pytest
from tail_strategy import tail_candidates, CellStats, best_side, Config, BINS

def _obs(mid, bid=None, ask=None, H=6):
    bid = mid - 0.005 if bid is None else bid
    ask = mid + 0.005 if ask is None else ask
    return pd.DataFrame(dict(H=[H], mid=[mid], bid=[bid], ask=[ask]))

def test_tail_bounds_are_strict():
    assert len(tail_candidates(_obs(0.10, 0.095, 0.105))) == 0
    assert len(tail_candidates(_obs(0.90, 0.895, 0.905))) == 0
    assert len(tail_candidates(_obs(0.03))) == 1 and len(tail_candidates(_obs(0.97))) == 1

def test_tail_filters_spread_and_extremes():
    assert len(tail_candidates(_obs(0.05, 0.01, 0.08))) == 0      # spread 7c
    assert len(tail_candidates(_obs(0.995, 0.99, 1.00))) == 0     # ask > 0.99

def test_cellstats_empty_and_shrinkage():
    cs = CellStats(50, BINS)
    f, n = cs.estimate(6, 0.02)
    assert np.isnan(f) and n == 0
    cs.update(pd.DataFrame(dict(H=[6] * 100, mid=[0.02] * 100, y=[0.0] * 100)))
    f, n = cs.estimate(6, 0.02)
    assert n == 100 and f == pytest.approx(50 * 0.02 / 150)
    assert cs.estimate(3, 0.02)[1] == 0                               # cells are per horizon

def test_best_side_sells_overpriced_low_tail():
    side, price, edge = best_side(0.05, 0.06, 0.005, Config())
    assert side == -1 and price == pytest.approx(0.03)
    assert edge == pytest.approx(0.03 - 0.005 - 0.07 * 0.03 * 0.97)
```

- [ ] **Step 2: Run tests, expect FAIL** (`ModuleNotFoundError: tail_strategy`)

- [ ] **Step 3: Implement the constants, `Config`, `load_obs`, `tail_candidates`, `CellStats`, `best_side`**

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add strategies/btc-kalshi-digital/tail_strategy.py strategies/btc-kalshi-digital/tests/test_tail_strategy.py
git commit -m "Add tail filter and walk-forward edge estimator"
```

### Task 4: Day-by-day simulator (`tail_strategy.py`)

**Files:**
- Modify: `tail_strategy.py`
- Test: `tests/test_tail_strategy.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `run(obs: pd.DataFrame, cfg: Config, sizing: str = "portfolio") -> tuple[pd.DataFrame, pd.DataFrame]`, `sizing in {"portfolio", "one", "per_trade"}`.
  - `trades` columns: `day, ticker, H, side, q, size_price, fill, fee, edge, f, y, pnl` (`pnl = q * (side * (y - fill) - fee(fill))`).
  - `daily` columns: `day, pnl, ret, assumed_worst, realized_worst, breach, deployed, n_trades`; one row per expiry day in `obs` (zeros when nothing traded). `ret = pnl / bankroll`; `assumed_worst` = worst case at the last sizing call (held positions at actual fills, new ones at `size_price`); `realized_worst` = worst case of all positions at actual fills; `breach = realized_worst < -cap - 1e-9`; `deployed = sum(q * (fill if side == 1 else 1 - fill))`.

Algorithm per expiry day D (in date order):
1. If fewer than `warmup_days` prior days have been added to `CellStats`, trade nothing that day.
2. For each round `H in HOURS` (6 then 3): take that day's tail candidates with this H; for each, `estimate(H, mid)`; skip if `n < min_cell_n`; `best_side`; keep rows with `edge >= min_edge`. Build the payoff grid from the strikes of held positions plus candidates; held payoffs at actual fills give `base`; candidate payoffs at `size_price` give `P`. Size with `size_portfolio` / `q = 1` / `size_per_trade`. Fill each sized row at `ask_f` (buy) or `bid_f` (sell); drop it as unfilled if `ask_f >= 1.0` or `bid_f <= 0.0`. Filled rows become held positions.
3. Settle with `y`, write the daily row, then `CellStats.update(tail_candidates(day's obs))`.

- [ ] **Step 1: Write failing tests** using a synthetic-obs builder `_synth(days=40)` in the test file. It returns a DataFrame with the `load_obs` columns (`ticker, expiry, day, H, y, S, bid, ask, bid_f, ask_f, strike_type, floor, mid`). Use consecutive dates from 2026-06-01; for each day and each `H in (6, 3)`, 21 strikes `60_000 + 500 * i`, `i = 0..20`, with this fixed ladder:
  - `i = 0..3`: mid 0.94 (bid 0.935, ask 0.945), `y = 1`
  - `i = 4..16`: mid 0.50 (bid 0.49, ask 0.51), `y = 1` for `i < 10` else 0
  - `i = 17..20`: mid 0.06 (bid 0.055, ask 0.065), `y = 0`
  - `bid_f = bid`, `ask_f = ask`, `strike_type = "greater"`, `S = 65_000`, ticker `f"D{d}-T{i}"`.

  Each tail cell gets 4 observations per H per day, so after 30 warm-up days `n = 120 >= 100`. The estimator then gives `f ≈ 0.018` for the 0.06 cell and `f ≈ 0.982` for the 0.94 cell, so both sides clear `MIN_EDGE` at the 2c buffer and trading starts on day 31.

```python
from tail_strategy import run

def test_portfolio_never_breaches_without_slippage():
    tr, dy = run(_synth(), Config())
    assert len(tr) > 0
    assert (dy.assumed_worst >= -30 - 1e-9).all() and not dy.breach.any()

def test_no_trades_during_warmup():
    tr, dy = run(_synth(days=25), Config())
    assert tr.empty and (dy.pnl == 0).all() and len(dy) == 25

def test_no_lookahead():
    obs = _synth(days=40)
    tr1, _ = run(obs, Config())
    cut = sorted(obs.day.unique())[34]
    obs2 = obs.copy()
    later = obs2.day > cut
    obs2.loc[later, "y"] = np.random.default_rng(1).permutation(obs2.loc[later, "y"].values)
    tr2, _ = run(obs2, Config())
    a = tr1[tr1.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    b = tr2[tr2.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)

def test_unusable_fill_is_unfilled():
    obs = _synth()
    obs.loc[obs.mid < 0.10, "bid_f"] = 0.0
    obs.loc[obs.mid > 0.90, "ask_f"] = 1.0
    tr, _ = run(obs, Config())
    assert tr.empty

def test_day_without_3h_round_still_settles():
    obs = _synth()
    last = obs.day.max()
    obs = obs[~((obs.day == last) & (obs.H == 3))]
    tr, dy = run(obs, Config())
    assert last in set(dy.day) and dy.loc[dy.day == last, "pnl"].iloc[0] == pytest.approx(tr[tr.day == last].pnl.sum())

def test_bad_fill_shows_in_realized_worst():
    obs = _synth()
    obs.loc[obs.mid < 0.10, "bid_f"] = 0.01   # sells fill 2.5c below the sized price
    obs.loc[obs.mid > 0.90, "ask_f"] = 0.99   # buys fill 2.5c above the sized price
    _, dy = run(obs, Config())
    traded = dy[dy.n_trades > 0]
    assert len(traded) > 0 and (traded.realized_worst < traded.assumed_worst).all()
    assert (dy.breach == (dy.realized_worst < -30 - 1e-9)).all()

def test_per_trade_baseline_can_exceed_cap():
    _, dy = run(_synth(), Config(), sizing="per_trade")
    assert (dy.realized_worst < -30).any()
```

- [ ] **Step 2: Run tests, expect FAIL** (`ImportError: run`)

- [ ] **Step 3: Implement `run`** following the algorithm above.

- [ ] **Step 4: Run tests, expect PASS**

- [ ] **Step 5: Commit**

```bash
git add strategies/btc-kalshi-digital/tail_strategy.py strategies/btc-kalshi-digital/tests/test_tail_strategy.py
git commit -m "Add walk-forward tail simulator with portfolio and baseline sizing"
```

### Task 5: Reporting, CLI, real-data run, README

**Files:**
- Modify: `tail_strategy.py`
- Modify: `README.md` (add a "Tail strategy" section: what it does, the run commands, the output files)
- Test: `tests/test_tail_strategy.py`

**Interfaces:**
- Consumes: `run`, `Config`, `load_obs`.
- Produces:
  - `day_ci(ret: pd.Series, n: int = 4000, seed: int = 11) -> tuple[float, float]`: bootstrap over days of the mean daily return, 2.5/97.5 percentiles.
  - `summarize(trades: pd.DataFrame, daily: pd.DataFrame, cfg: Config, label: str) -> str`: lines for trades by side and H, mean daily return with `day_ci`, cumulative return, max drawdown (on cumulative pnl / bankroll), worst day, max `assumed_worst`, mean `deployed`, breach count, and first-half / second-half mean daily return (split on the sorted traded days). Returns `"<label>: no trades"` when `trades` is empty.
  - `main()` with CLI flags `--data`, `--bankroll`, `--cap`, `--q-max`, `--min-edge`, `--slip`, `--n0`, `--warmup-days`, `--out-dir`, `--no-plot`. Runs the three sizing modes, prints the three summaries plus, for each baseline, the worst day as % of bankroll and the number of days with `realized_worst < -cap`. Writes `tail_trades.csv`, `tail_daily.csv` (portfolio mode), `tail_summary.txt` (everything printed) and, unless `--no-plot`, `tail_backtest.png` (cumulative return of the three modes; histogram of daily return for portfolio mode).

- [ ] **Step 1: Write failing tests**

```python
from tail_strategy import summarize, main

def test_summarize_empty():
    tr, dy = run(_synth(days=25), Config())
    assert summarize(tr, dy, Config(), "portfolio") == "portfolio: no trades"

def test_main_writes_outputs(tmp_path):
    p = tmp_path / "obs.csv"
    _synth().to_csv(p, index=False)
    main(["--data", str(p), "--out-dir", str(tmp_path / "out"), "--no-plot"])
    for f in ("tail_trades.csv", "tail_daily.csv", "tail_summary.txt"):
        assert (tmp_path / "out" / f).exists()
```

(`main` takes an optional `argv: list[str] | None = None` passed to `argparse`.)

- [ ] **Step 2: Run tests, expect FAIL** (`ImportError: summarize`)

- [ ] **Step 3: Implement `day_ci`, `summarize`, `main`; add the README section**

- [ ] **Step 4: Run the full test suite, expect PASS**

Run: `/opt/miniconda3/bin/python3 -m pytest tests -q`

- [ ] **Step 5: Run on the real data**

Run: `/opt/miniconda3/bin/python3 -I tail_strategy.py`
Expected: three summaries print; `output/btc_tail_results/` contains the four files; the portfolio summary shows max assumed worst case <= 3.00% of bankroll; breach count is reported (any non-zero count is reported, not hidden).

- [ ] **Step 6: Commit**

```bash
git add strategies/btc-kalshi-digital/tail_strategy.py strategies/btc-kalshi-digital/tests/test_tail_strategy.py strategies/btc-kalshi-digital/README.md strategies/btc-kalshi-digital/output/btc_tail_results
git commit -m "Add tail strategy reporting, CLI and first results"
```
