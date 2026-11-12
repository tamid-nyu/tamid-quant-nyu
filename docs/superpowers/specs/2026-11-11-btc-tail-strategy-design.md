# BTC Kalshi tail strategy with portfolio worst-case sizing: design

Date: 2026-11-11
Location of code: `strategies/btc-kalshi-digital/`
Status: draft for review

## 1. Purpose and success criteria

Kalshi "BTC above $K at 5pm ET" contracts show a favorite-longshot bias in the tails at short horizons. An exploratory
analysis of `output/btc_pm_results/btc_observations.csv` (8,989 contract-hour observations, ~120 expiry days) found:

- Low tail (mid < 10%): the market overprices YES at 3h and 6h (e.g. mid 1.7% vs 0.4% realized at 3h). At 12h it is calibrated.
- High tail (mid > 90%): the market underprices YES (contracts quoted at 98c or more settled YES 100% of the time).
- Naive tail trades earned about +1.5c to +2.2c net per contract at 3h/6h, but the payoff is ~+2c most of the time and ~-97c rarely,
  so unsized results say little about survivability. That analysis tried about 24 threshold/horizon cuts with no out-of-sample test.

**Goal:** a research backtest that trades only the tails and sizes positions so that the worst possible settlement of
any single expiry day costs at most **3% of bankroll**.

**Success criteria:**
1. No expiry day's assumed worst-case loss exceeds 3% of bankroll (invariant; enforced and tested).
2. Realized worst expiry day, max drawdown, return on bankroll and capital deployed are reported with day-clustered
   bootstrap CIs and an in-sample / out-of-sample split.
3. Results are compared against two baselines (section 7) so the value of the portfolio cap is visible.
4. No look-ahead: nothing known only after time t affects a decision at time t (tested).

**Non-goals:** live execution, order-book depth, the 12h horizon, BTC delta hedging, drawdown-aware size scaling,
changes to existing scripts.

## 2. Inputs and assumptions

- Input: `output/btc_pm_results/btc_observations.csv`, produced by `btc_digital_model_backtest.py`
  (user runs that first). Columns used: `ticker, expiry, day, H, y, bid, ask, bid_f, ask_f, floor`.
  Only `strike_type == "greater"` contracts exist in the data (YES iff BTC at expiry > floor).
- One expiry per day (21:00 UTC). All positions for a day settle together.
- Decision rounds: H = 6 and H = 3 hours before expiry. A fill happens one hour after the decision, at that hour's
  bid (selling YES) or ask (buying YES), as in the existing backtest (taker fills).
- Fee per contract: `0.07 * p * (1 - p)` at the fill price p. This is an assumption copied from the existing backtest;
  confirm against Kalshi's current fee schedule.
- Bankroll: fixed `BANKROLL = 1000` dollars, no compounding in the first version.
- Position size cap: `Q_MAX = 50` contracts per ticker per round. This is a placeholder because the data has no depth.

## 3. Tail universe

An observation is a candidate when its decision-time mid `(bid + ask) / 2` is below 0.10 or above 0.90, `H in {6, 3}`,
`bid >= 0.01`, `ask <= 0.99`, and `ask - bid <= 0.06`. Side is chosen by the sign of the edge (section 4): sell YES when the
market overprices, buy YES when it underprices.

## 4. Ex-ante edge (walk-forward, no look-ahead)

Cells are `(H, price bin)` with bin edges `[0, .005, .015, .025, .035, .05, .075, .10, .90, .925, .95, .965, .975, .985, .995, 1.0001]`.

For an observation on expiry day D, use only observations from expiry days strictly before D. For its cell, with `n` prior
observations, `sum_y` YES outcomes and `m` the mean prior mid:

    f = (sum_y + N0 * m) / (n + N0),   N0 = 50

This shrinks the cell's realized rate toward the market price, so the estimate starts at "no edge" and learns.

- Skip the observation if `n < 100` or fewer than `WARMUP_DAYS = 30` prior days exist.
- Conservative prices for sizing (decision-time quotes plus a slippage buffer `SLIP = 0.02`):
  `buy_price = min(ask + SLIP, 0.99)`, `sell_price = max(bid - SLIP, 0.01)`.
- Net edge per contract:
  `buy: f - buy_price - fee(buy_price)`, `sell: sell_price - f - fee(sell_price)`.
- Trade only if the best side's net edge is at least `MIN_EDGE = 0.005`.

## 5. Sizer (`tail_sizer.py`, pure functions)

**Payoff model.** For one expiry day, let the sorted unique strikes be `K_1 < ... < K_m` (all strikes with candidates
or held positions). Settlement scenarios are `s_0` (price at or below `K_1`) and `s_j` (just above `K_j`), giving m + 1 scenarios.
A contract is YES in scenario s iff `s > K`. Per contract at fill price p:

- buy YES: payoff(s) = `1[s > K] - p - fee(p)`
- sell YES: payoff(s) = `p - 1[s > K] - fee(p)`

**Portfolio constraint.** Positions already held for the day (from the 6h round) have fixed fills and contribute a fixed
payoff vector `base(s)`. New positions `q_i` must satisfy, for every scenario s:

    base(s) + sum_i q_i * payoff_i(s)  >=  -CAP,   CAP = 0.03 * BANKROLL = 30 dollars

**Optimization.** Maximize `sum_i q_i * edge_i` subject to the constraint above and `0 <= q_i <= Q_MAX`, using
`scipy.optimize.linprog` (HiGHS). Then floor each `q_i` to a whole number and verify every scenario; if any scenario
is breached after rounding, scale all `q_i` down in 5% steps and refloor until satisfied.

Because constraints use the whole payoff curve, nested positions on one side add up, while a book with both a rally
risk (short high-strike YES) and a crash risk (long low-strike YES) is not charged for both at once.

**Slippage caveat.** Sizing uses conservative decision-time prices, but actual fills use next-hour quotes. If a fill is
worse than the buffer, the realized worst case can exceed 3%. The report counts these breaches; the target is zero but
is not guaranteed.

## 6. Simulator and outputs (`tail_strategy.py`)

For each expiry day in order: build candidates for the 6h round, size them, apply fills; then do the same for the 3h round
with the 6h positions as `base(s)`. Settle with the observed outcomes `y`. After the day, add that day's observations
(all candidates, traded or not) to the estimator cells.

CLI: `--bankroll`, `--cap`, `--q-max`, `--min-edge`, `--slip`, `--n0`, `--warmup-days`, `--out-dir`, `--no-plot`.

Outputs in `output/btc_tail_results/`: `tail_trades.csv`, `tail_daily.csv`, `tail_summary.txt`, `tail_backtest.png`.

Reported metrics: daily return on bankroll (mean, sd, cluster-bootstrap 95% CI), cumulative return, max drawdown,
worst realized day, max assumed worst-case, capital deployed, cap breaches from slippage, number of trades by side and
horizon, and first-half / second-half results.

## 7. Baselines

- **B1:** one contract per candidate that passes section 4, no cap.
- **B2:** per-trade cap: each trade sized so its own worst-case loss is at most 3% of bankroll (capped at `Q_MAX`), with no
  portfolio constraint.

Report each baseline's worst expiry-day loss as % of bankroll and the number of days above the 3% cap.

## 8. Tests (`tests/test_tail_sizer.py`)

1. For random books, the assumed worst-case loss never exceeds `CAP` after rounding.
2. Two short high-strike YES positions are sized so their combined worst case (a rally) stays within `CAP`, i.e. they get
   about half the size of one alone.
3. A book with one short high-strike YES and one long low-strike YES is allowed more total size than either alone.
4. Look-ahead: sizes and edge estimates at time t are identical if all outcomes after t are scrambled.
5. Estimator: with zero prior data, the strategy places no trades.

Tests run with `python -m pytest`; `pytest` is not in `requirements.txt` and would be added as a dev dependency.

## 9. Risks and limits

- About 120 days with very few tail losses: realized results will probably look calm. The 3% cap is the real protection,
  so the report states the assumed worst case alongside realized results.
- One market regime of about four months. A large BTC move could change the tail picture.
- Selection: price bins, `N0`, `MIN_EDGE` and `SLIP` are chosen up front here and should not be tuned on the same
  data without an out-of-sample check.
- Fill assumptions are optimistic (full taker fills at the quoted bid/ask); `Q_MAX` is not backed by depth data.
