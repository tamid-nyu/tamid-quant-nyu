# Multi-market Kalshi tail strategy with k-worst portfolio sizing: design

Date: 2026-11-12
Location of code: `strategies/kalshi-multi-tail/` (new). `strategies/btc-kalshi-digital/` is imported, never modified.
Status: draft for review
Builds on: `2026-11-11-btc-tail-strategy-design.md` (the BTC base, "Model 0") and `2026-11-11-btc-tail-variants-design.md`.

## 1. Purpose and success criteria

The BTC tail strategy earns many small wins and rare large losses. On the long history it returned +5.3% over 481 live
days, but its mean daily return CI (-0.023%, +0.041%) includes zero, and it traded on only 237 of those days. Running the
same strategy on several markets with little tail correlation should give more small wins per unit of time. It should
also give more independent days of evidence on whether the favorite-longshot bias is real.

**Goal:** a research backtest that trades the same tail rule on BTC and cross-asset financial markets on Kalshi. Each
day's combined book is sized so that (a) no single market's worst settlement loses more than `c` of bankroll and (b) the
`k` worst markets losing together lose no more than `C` of bankroll.

**Success criteria:**
1. At every decision, the assumed per-market worst case is at most `c·B` and the k-worst sum is at most `C·B` after
   rounding (invariant, enforced and tested).
2. Two edge estimators (S: separate, P: partial pooling) are reported side by side against BTC-only Model 0 on the same
   days, with a paired-bootstrap Sharpe difference.
3. The report measures the goal directly: winning trades per day, traded days per week, per-market P&L contribution, and
   realized correlation of per-market daily P&L.
4. No look-ahead (tested).
5. BTC run alone through the new pipeline with `k = 1, C = c` reproduces Model 0's committed trades exactly.

**Non-goals:** live execution, order-book depth, spot or volatility models (variants #5/#7), budget reservation between
markets, compounding, FX markets, and Nasdaq/silver in the default set.

## 2. Markets

A probe of the Kalshi API on 2026-11-12 found these "above K" ladders. FX series (EUR, JPY, GBP) have only up/down
contracts and are out.

| key | group | series | strike rule | daily expiry used | default |
|---|---|---|---|---|---|
| BTC | crypto | `KXBTCD`, `BTCD` | `>` floor | 5pm ET | on |
| SPX | equity | `KXINXU` | `>=` floor | 4pm ET event | on |
| GOLD | metals | `KXGOLDD` | `>` floor | 5pm ET | on |
| WTI | crude | `KXWTI` | `>` floor | 2:30pm ET | on |
| NATGAS | gas | `KXNATGASD` | `>` floor | 5pm ET | on |
| NDX | equity | `KXNASDAQ100U` | `>=` floor | 4pm ET event | off |
| SILVER | metals | `KXSILVERD` | `>` floor | 5pm ET | off |

Expiry times in the table come from the probe's observed close times. The pull reads each market's actual `close_time`
and does not hard-code it. Hourly series (`KXINXU`, `KXNASDAQ100U`) list one event per hour; only the event closing at
the market's regular daily close is used. NDX and SILVER are each about 0.8-0.9 correlated with SPX and GOLD, so they
add little diversification and stay off by default.

**History gate.** The first implementation task counts settled daily expiries per series. A market with fewer than
150 expiry days is dropped from the default set, and the spec's table is updated. Brent (`KXBRENTD`, since Sep 2026) is
already below this.

**History gate result (2026-11-12):** SPX 434 expiry days (from 2025-02-18) passes. GOLD 128 (from 2026-03-24),
WTI 122 (from 2026-05-18) and NATGAS 116 (from 2026-04-15) are below 150. The run by the spec is BTC + SPX. At the
user's request, a second, labeled post-hoc run includes all five with a 100-day minimum (`--min-history-days 100`).

Strike inclusivity does not change the sizer. For `>=` markets the scenarios are "below the lowest strike" and
"exactly at K_j"; contract i is YES in scenario j iff `K_i <= K_j`, the same matrix `tail_sizer.payoff_matrix` builds for
`>` markets ("just above K_j"). The outcome `y` always comes from Kalshi's `result`, so inclusivity is recorded but not
used in any calculation.

## 3. Data pipeline

- **`markets.yaml`**: one entry per market with `key`, `group`, `series` (list, first with data wins, as in the BTC
  pull), `strike_rule` (`gt` or `ge`), `daily_close_et` (used to pick the event for hourly series), and `enabled`.
- **Strike selection.** Kalshi lists many strikes per event. The pull keeps the 30 nearest the previous expiry's
  settlement value (`expiration_value`), within a per-market band (`band` in `markets.yaml`), which is known before
  the day's decisions. When there is no previous value (first day), it keeps every strike.
- **`multi_pull.py`**: a series-generic version of `btc_pm_data_pull.py`'s event-by-event market listing and hourly
  candle fetch, with a per-ticker cache. Writes `output/raw/<KEY>/markets.csv` and `output/raw/<KEY>/hourly.csv`.
  No spot data is pulled: the outcome `y` comes from Kalshi's `result` field, as in the BTC backtest. BTC reuses the
  existing long-history raw files in `strategies/btc-kalshi-digital/output/btc_pm/` and is not pulled again.
- **`multi_obs.py`**: builds one observation per contract and horizon H ∈ {6, 3}. Columns:
  `market, group, ticker, expiry, day, H, y, bid, ask, bid_f, ask_f, floor, strike_rule, decision_ts`.
  `decision_ts = floor_to_hour(expiry - H hours)` (WTI expires at :30, so its decision uses the candle ending at the
  whole hour before); `bid/ask` are the candle close quotes at `decision_ts` and `bid_f/ask_f` one hour later. For BTC
  this is the same rule as the BTC backtest, since its expiry is on the hour.
  An observation is dropped if either hour has no quote (markets that do not trade overnight lose those rounds).
  For BTC the observations are taken from `output/btc_pm_results_long/btc_observations.csv`, and the extra columns are added.
- `day` is the calendar date of expiry in New York time. All markets with the same `day` form one book.

## 4. Tail universe and trade rule

Unchanged from the BTC base: decision-time mid below 0.10 or above 0.90, `bid >= 0.01`, `ask <= 0.99`,
`ask - bid <= 0.06`. Fee `0.07·p·(1-p)` rounded up to the cent per order. `SLIP = 0.02`, `MIN_EDGE = 0.005`,
`Q_MAX = 50`, taker fills at the next hour's quotes. Same price-bin edges as the BTC base.
Whether the fee schedule is the same for financial and commodity series has to be checked against Kalshi's fee page in
the first task. If it differs, the fee becomes a per-market setting in `markets.yaml`.

## 5. Edge estimators

Both use only observations from expiry days strictly before the current day, and both train on all base-tail
observations (traded or not), as the BTC base does.

**S (separate).** The BTC base's `CellStats` is run separately for each market:
`f = (sum_y + N0·m) / (n + N0)`, with `N0 = 50` and `m` that market's mean prior mid in the cell. A cell trades if it has
`n >= 100` and the market has at least 30 prior expiry days.

**P (partial pooling).** First a pooled estimate per `(H, bin)` over all enabled markets:
`f_pool = (sum_y_all + N0·m_all) / (n_all + N0)`. Then per market:
`f_m = (sum_y_m + N1·f_pool) / (n_m + N1)`, with `N1 = 100`. A cell trades if the pooled cell has `n_all >= 100` and the
market has at least 30 prior expiry days. There is no per-market count minimum, so a new market can trade on the pooled
estimate.

All parameters (`N0`, `N1`, minimum counts, warmup) are fixed here before any run and are not tuned.

## 6. Sizer (`multi_sizer.py`)

**Payoff model per market.** For each market m in the day's book, the scenario grid comes from that market's strikes
(held or candidate), as in `tail_sizer.py`, with the inclusivity rule from section 2. Markets are independent in the
model. Each has its own scenarios, and no joint scenario grid is built.

**Variables.** Sizes `q_i` for the candidates being decided now, worst-case losses `w_m` per market, plus `t` and `u_m`
for the top-k constraint.

**Constraints:**

    w_m >= -(base_m(s) + sum_{i in m} q_i · payoff_i(s))     for every market m and scenario s
    w_m >= 0
    w_m <= c · B                                              (per-market cap)
    k · t + sum_m u_m <= C · B,  u_m >= w_m - t,  u_m >= 0    (sum of the k largest w_m <= C·B)
    0 <= q_i <= Q_MAX

**Objective:** maximize `sum_i q_i · edge_i` with `scipy.optimize.linprog` (HiGHS).

**Rounding.** Floor each `q_i`, recompute `w_m` and the top-k sum exactly, and if any limit is breached, scale all new
`q_i` down in 5% steps and floor again, as in the BTC base.

**Defaults:** `B = 1000`, `c = 0.03`, `k = 2`, `C = 0.04`. When `C >= k·c` the k-worst cap can never bind; the CLI
prints a warning and the summary says so (this is how the BTC-only regression runs, with `k = 1, C = c`). `k < 1` or
`C <= 0` raise errors. With the defaults, the second-largest `w_m` is at most `C/k = 2%`, so with five markets the
accepted worst case if every market loses at once is `C + (n - k)·C/k = 10%`. That number, computed from actual positions, is reported for every day.

**Timing.** A day's decision rounds (each market's 6h and 3h) are sorted by `decision_ts`. Rounds with the same
timestamp are sized together in one LP. Positions filled in earlier rounds, in any market, enter as fixed `base_m`.
Markets that decide earlier get first claim on the shared k-worst budget. The report shows capital used per market so
this is visible.

**Slippage.** Sizing uses decision-time quotes plus `SLIP`, and fills use next-hour quotes, so realized worst cases can
exceed `c` or `C`. Breaches of each cap are counted.

## 7. Simulator and CLI (`multi_tail.py`)

For each `day` in order: run the decision rounds as in section 6, apply fills, settle with `y`, then add the day's
observations to the estimators. Runs: `--estimator S`, `--estimator P`, and `--markets BTC --k 1 --cap-k 0.03` (BTC-only
regression).

CLI: `--markets`, `--estimator {S,P}`, `--bankroll`, `--cap` (c), `--k`, `--cap-k` (C), `--q-max`, `--min-edge`,
`--slip`, `--n0`, `--n1`, `--warmup-days`, `--obs`, `--out-dir`, `--no-plot`.

## 8. Evaluation and outputs

Every run walks forward over the full history. Scored periods match the variants work: check period 2026-01-01 to
2026-06-30, **holdout 2026-07-01 to 2026-11-11** (headline). Days without trades count as zero.

For S, P and BTC-only Model 0 on the same days:
- Sharpe (mean/sd of daily returns × sqrt(365)) with a day-bootstrap 95% CI, and the paired-bootstrap difference vs BTC-only.
- Return on $1,000, max drawdown, worst day.
- Winning trades per day, traded days per week, trades by market, side and horizon.
- Per-market P&L contribution and capital used.
- Realized correlation matrix of per-market daily P&L, overall and on days where any market lost more than 1%.
- The all-markets worst case per day (max and mean), days within 10% of `c` and of `C`, and slippage breaches of each.

Baselines and sensitivity (not tuning; results are reported and nothing is chosen from them):
- Option 1: per-market `c = 0.03` with every market's worst case summed against one total cap of 0.06.
- `k ∈ {1, 2, 3}` with `C = {0.03, 0.04, 0.05}`.

Outputs in `strategies/kalshi-multi-tail/output/`: `trades.csv`, `daily.csv`, `summary.txt`, `equity.png`
(S, P, BTC-only over the holdout from $1,000), `contribution.png` (per-market cumulative P&L).

## 9. Tests (`strategies/kalshi-multi-tail/tests/`)

1. **Regression:** BTC alone with `k = 1, C = c` reproduces `btc-kalshi-digital/output/btc_tail_results_long/tail_trades.csv`
   exactly (skipped when that file is missing).
2. **Sizer invariants:** for random multi-market books, after rounding every `w_m <= c·B` and the k-worst sum `<= C·B`.
3. **Sizer behavior:** one market alone sizes the same as `tail_sizer` with cap `c`. Two markets that each fill `c`
   alone are held to a combined worst case of `C` together. `C >= k·c` warns that the cap is non-binding, and `k < 1`
   or `C <= 0` raise errors.
4. **Inclusivity:** `multi_obs` keeps `strike_rule` and takes `y` from `result`. A WTI expiry at 18:30 UTC gives a
   6h decision at 12:00 and a fill at 13:00.
5. **Estimator P:** a market with no data gets `f_m = f_pool`. With very large `N1`, `f_m` tends to `f_pool`.
6. **Estimator S:** on BTC-only data it matches `CellStats` from the BTC base.
7. **Look-ahead:** scrambling every outcome after time t leaves all sizes and estimates at t unchanged, for S and P.
8. **Timing:** rounds within a day run in `decision_ts` order, and earlier fills appear in later rounds' `base_m`.

Tests run with `python -m pytest`, using `requirements-dev.txt`. Add `pyyaml` to `requirements.txt` if it is not already there.

## 10. Risks and limits

- **Tail correlation.** The k-worst cap assumes no more than k markets hit their worst case on the same day. A broad
  risk-off shock (equities, crude and crypto falling together while gold and gas spike) can break this. The accepted
  all-markets worst case is about 10% of bankroll and is reported daily.
- **Bias may not carry over.** The favorite-longshot bias was found in BTC. Financial markets may have tighter, better
  arbitraged pricing. S shows each market's own bias, and P assumes some sharing that may not exist.
- **History length.** The new series may have much less history than BTC, which shortens their warmup and makes the
  holdout thin for them. The history gate (section 2) and per-market trade counts make this visible.
- **Multiple comparisons.** S, P, the baseline and three k settings are all reported, so one beating BTC-only on the
  holdout is not strong evidence by itself. The parameters are fixed here before any run.
- **Fill assumptions.** As in the BTC base: full taker fills at quoted prices, `Q_MAX` not backed by depth data, and
  non-crypto order books may be thinner.
- **Order effect.** Markets deciding earlier in the day get first claim on the shared budget.
