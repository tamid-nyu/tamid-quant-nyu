# Multi-market Kalshi tail strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest the BTC tail strategy across BTC, S&P 500, gold, WTI and natural gas with a per-market cap plus a
k-worst-markets cap, using two edge estimators (S separate, P partial pooling), and compare with BTC-only Model 0.

**Architecture:** New folder `strategies/kalshi-multi-tail/` that imports `tail_sizer` and `tail_strategy` from
`strategies/btc-kalshi-digital/` (read-only). Pull → observations → walk-forward simulator with one LP per decision
timestamp across all markets → report.

**Tech Stack:** Python 3.13 (`/opt/miniconda3/bin/python3`), pandas, numpy, scipy (HiGHS linprog), pyyaml, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-multi-market-tail-design.md`

## Global Constraints

- `strategies/btc-kalshi-digital/` is never modified.
- Defaults: `B = 1000`, `c = 0.03`, `k = 2`, `C = 0.04`, `N0 = 50`, `N1 = 100`, min cell n 100, warmup 30 prior
  expiry days per market, `SLIP = 0.02`, `MIN_EDGE = 0.005`, `Q_MAX = 50`, horizons (6, 3), fee rate 0.07.
- Tail universe, bins, fee rounding and fill rules are imported from `tail_strategy` / `tail_sizer`, not copied.
- Holdout 2026-07-01 → 2026-11-11 is the headline; check period 2026-01-01 → 2026-06-30.
- No commits (user instruction, 2026-11-12). Commit steps are replaced by "leave changes unstaged".
- Tests: `cd strategies/kalshi-multi-tail && /opt/miniconda3/bin/python3 -m pytest tests -q`.

## Review Focus

1. A market with history much shorter than BTC: it must trade under P after 30 of its own days, never under S
   before its cells have n ≥ 100, and must never crash the run. → Task 5 test `test_short_history_market`.
2. A day where only some markets have expiries (weekends, US holidays): book contains only those markets. → Task 5
   test `test_weekend_only_btc`.
3. Held positions in one market already past `c` (slippage): that market gets no new size, others still trade. →
   Task 4 test `test_breached_market_blocks_only_itself`.
4. Two markets with the same decision timestamp are sized in one LP; different timestamps sequentially. → Task 5
   test `test_rounds_in_clock_order`.
5. Missing quotes at the fill hour drop the candidate before sizing (no phantom hedge). → covered by the imported
   base rule; Task 5 test `test_missing_fill_quote_dropped`.

---

### Task 1: Pull and markets config (done during brainstorming)

**Files:** Create `strategies/kalshi-multi-tail/markets.yaml`, `strategies/kalshi-multi-tail/multi_pull.py`.

**Interfaces:** Produces `load_markets() -> dict[key, dict]` and `output/raw/<KEY>/{markets,hourly}.csv` with columns
`market,ticker,event_ticker,series,strike_type,strike_rule,floor,expiry,result,expiration_value,ref_value` and
`ticker,ts,bid,ask,close,volume,open_interest`.

- [x] Smoke run `python multi_pull.py --market GOLD --days 6` writes both files.
- [x] Full pulls for SPX, GOLD, WTI, NATGAS, NDX, SILVER at `--days 860` running in background.

### Task 2: Observations and history gate (`multi_obs.py`)

**Files:** Create `strategies/kalshi-multi-tail/multi_obs.py`; Test `tests/test_multi_obs.py`.

**Interfaces:**
- Produces `build_market_obs(markets: DataFrame, hourly: DataFrame, key: str, group: str, hours=(6, 3)) -> DataFrame`
  with columns `market, group, ticker, expiry, day, H, y, bid, ask, bid_f, ask_f, floor, strike_rule, strike_type, decision_ts, mid`.
- Produces `load_btc_obs(path) -> DataFrame` (same columns, from `btc_observations.csv`, via `tail_strategy.load_obs`).
- Produces `load_all(keys: list[str]) -> DataFrame` reading `output/obs/<KEY>.csv` and BTC.
- CLI `python multi_obs.py` writes `output/obs/<KEY>.csv` and prints the history gate table
  (expiry days, first/last day, tail observations per market; flags < 150 days).

- [ ] **Step 1: tests**
  - `test_decision_floor_wti`: expiry `18:30Z`, H=6 → `decision_ts == 12:00Z`, uses candle at 12:00 and fill at 13:00.
  - `test_on_hour_expiry`: expiry `21:00Z`, H=3 → candle at 18:00, fill at 19:00.
  - `test_missing_quote_dropped`: NaN bid at decision hour → no row.
  - `test_day_is_new_york_date`: expiry `2026-11-10T01:00Z` → `day == 2026-11-09`.
  - `test_y_from_result`: `result == "yes"` → `y == 1.0`.
- [ ] **Step 2: implement**, run tests, PASS.

### Task 3: Estimators (`estimators.py`)

**Files:** Create `strategies/kalshi-multi-tail/estimators.py`; Test `tests/test_estimators.py`.

**Interfaces:**
- `class SeparateEstimator(n0, bins)`: `update(df)` (df has `market,H,mid,y`), `estimate(market, H, mid) -> (f, n)`
  where n is the market's raw cell count; eligibility `n >= min_cell_n`. Wraps one `tail_strategy.CellStats` per market.
- `class PooledEstimator(n0, n1, bins)`: `update(df)`, `estimate(market, H, mid) -> (f, n_pool)`;
  `f_pool = pooled CellStats.estimate`, `f = (sy_m + n1 * f_pool) / (n_m + n1)`; eligibility on `n_pool`.
- `make_estimator(kind: "S"|"P", cfg) -> estimator`.

- [ ] **Step 1: tests**
  - `test_S_matches_cellstats`: on random BTC-only rows, `SeparateEstimator.estimate` equals `CellStats.estimate`.
  - `test_S_markets_independent`: updating market A does not change market B's estimate or count.
  - `test_P_no_market_data_equals_pool`: market B never updated → `f == f_pool`, n = pooled count.
  - `test_P_large_n1_tends_to_pool`: `n1 = 1e9` → `|f - f_pool| < 1e-6`.
  - `test_P_formula`: hand-computed example (pool 200 obs, market 50 obs with 5 YES).
- [ ] **Step 2: implement**, run tests, PASS.

### Task 4: Multi-market sizer (`multi_sizer.py`)

**Files:** Create `strategies/kalshi-multi-tail/multi_sizer.py`; Test `tests/test_multi_sizer.py`.

**Interfaces:**
- `topk_sum(w: array, k: int) -> float`.
- `size_multi(books: list[dict], cap_m: float, cap_k: float, k: int, q_max: int) -> list[np.ndarray]` where each book
  is `{"P": (S_m, n_m) array, "base": (S_m,) array, "edges": (n_m,) array}`; returns whole-contract sizes per book.
  - If exactly one book: `tail_sizer.size_portfolio(edges, P, base, min(cap_m, cap_k), q_max)` (the problems are identical).
  - Otherwise LP over `[q, w, t, u]` as in spec section 6. A book whose base already breaches `cap_m` gets its
    candidates fixed at 0 and `w_m` bounded by its base loss. If the top-k of base losses already breaches `cap_k`,
    return zeros.
  - Rounding: floor; repair loop removes one contract at a time from the most-violating market (largest `w_m` among
    those breaching `cap_m`, else largest `w_m` overall while the top-k cap is breached), from the position losing
    most in that market's worst scenario; then add-back loop (best edge first) as in `size_portfolio`.
- `check_caps(books, qs, cap_m, cap_k, k) -> bool`.

- [ ] **Step 1: tests**
  - `test_single_book_matches_size_portfolio` (random books, exact equality).
  - `test_general_lp_single_book_same_objective`: calling the LP path on one book gives the same edge total as `size_portfolio` within 1 contract × max edge.
  - `test_random_books_respect_caps`: 200 random 2-4 market books, after rounding every `w_m <= cap_m + 1e-9` and topk `<= cap_k + 1e-9`.
  - `test_two_markets_limited_by_k_cap`: two identical short-YES books that each fill `cap_m = 30` alone; together with k=2, `cap_k = 40` combined worst `<= 40` and each `>= 15`.
  - `test_k1_is_max_market`: with k=1 the constraint is the largest single loss.
  - `test_breached_market_blocks_only_itself`.
- [ ] **Step 2: implement**, run tests, PASS.

### Task 5: Simulator (`multi_tail.py`, `run`)

**Files:** Create `strategies/kalshi-multi-tail/multi_tail.py`; Test `tests/test_multi_tail.py`.

**Interfaces:**
- `@dataclass MultiConfig(bankroll=1000, cap_m=0.03, k=2, cap_k=0.04, q_max=50, min_edge=0.005, slip=0.02, n0=50,
  n1=100, min_cell_n=100, warmup_days=30, fee_rate=0.07, estimator="S", hours=(6, 3), total_cap=None)`;
  `total_cap` set → option-1 baseline (k = number of markets, `cap_k = total_cap`).
- `run(obs: DataFrame, cfg) -> (trades, daily, market_daily)`:
  - trades columns = `tail_strategy.TRADE_COLS + ["market", "group"]`.
  - daily: `day, pnl, ret, n_trades, n_wins, deployed, assumed_max_market, assumed_topk, assumed_all,
    realized_max_market, realized_topk, realized_all, breach_m, breach_k`.
  - market_daily: `day, market, pnl, n_trades, deployed`.
  - Per day: rounds = unique `decision_ts` among that day's tail rows with H in cfg.hours, ascending. Per round, candidates
    are built with the base rules (`tail_strategy.best_side`, fill-quote check, `min_edge`), the market's warmup
    (prior days of that market ≥ warmup) and estimator eligibility. Cap reservation: `cap_m - 0.01·n_cands_m`,
    `cap_k - 0.01·n_cands`, as in the base.
  - After the day, `estimator.update(tail rows of the day)` and each market's day counter increments.
- [ ] **Step 1: tests**
  - `test_btc_regression`: BTC-only, `k=1, cap_k=0.03`, estimator S → trades equal committed
    `btc_tail_results_long/tail_trades.csv` on `day,ticker,H,side,q,fill,pnl` (skip if file missing).
  - `test_rounds_in_clock_order`, `test_weekend_only_btc`, `test_short_history_market`, `test_missing_fill_quote_dropped`.
  - `test_lookahead_scramble`: scrambling `y` after day D leaves trades on days ≤ D unchanged, for S and P.
- [ ] **Step 2: implement**, run tests, PASS.

### Task 6: Report, CLI, charts

**Files:** Modify `multi_tail.py` (add `summarize`, `sharpe`, `paired_sharpe_diff`, `main`); Test add to `tests/test_multi_tail.py`.

**Interfaces:** `sharpe(ret) -> float` (mean/sd·√365, 0 if sd == 0); `boot_sharpe_ci(ret, n=4000, seed=11)`;
`paired_diff_ci(ret_a, ret_b, ...)`; `main(argv)` with spec section 7 flags plus `--all` (runs S, P, BTC-only,
option-1 baseline, k sensitivity, and writes outputs).

- [ ] **Step 1: tests** `test_sharpe_constant_zero`, `test_paired_diff_identical_is_zero`, `test_cli_rejects_bad_k` (k < 1 → SystemExit), `test_cli_warns_nonbinding`.
- [ ] **Step 2: implement**: summary table per run (check and holdout: Sharpe [CI], diff vs BTC-only [CI], return,
  max DD, worst day, wins/day, traded days/week, trades), per-market contribution and correlation matrix (overall and
  stress days with any market < -1%), assumed/realized all-markets worst case, cap breaches. Writes `trades.csv`,
  `daily.csv`, `market_daily.csv`, `summary.txt`, `equity.png`, `contribution.png` to `output/`.
- [ ] **Step 3: run** `python multi_obs.py && python multi_tail.py --all` once pulls finish; record history gate and
  update spec section 2 table with markets dropped.
- [ ] **Step 4: README** `strategies/kalshi-multi-tail/README.md` with run commands and the headline results.
