# BTC tail strategy: numbered model variants

## Goal
Find variants of the tail strategy (`tail_strategy.py`) that earn a better return for the risk taken. The score is the
annualized Sharpe ratio of daily returns.

## Base
Model 0 is the current strategy, run on the long history: 815 expiries, Jul 2024 - Nov 2026, read from
`output/btc_pm_results_long/btc_observations.csv`. It must reproduce `output/btc_tail_results_long/` exactly.

## Periods
- Tuning: 2026-01-01 to 2026-06-30. Settings are chosen here only.
- Holdout: 2026-07-01 to 2026-11-11. Each chosen setting runs here once, and nothing changes afterwards.
- Every run walks forward over the whole history, and estimates use prior expiry days only. The periods only decide
  which days are scored.
- Known contamination: the 7 base losing days, 4 of them in the holdout, were inspected before this design. To limit
  the damage, the variants and their tuning grids are fixed here, before any run.

## Models
Each model changes exactly one thing from Model 0. Every option is a `Config` field whose default reproduces the base.

| # | Name | Change | Tuning grid |
|---|---|---|---|
| 0 | Base | none | none |
| 1 | Narrow tails | candidates only at mid < lo or > 1 - lo | lo in {0.03, 0.05, 0.075} |
| 2 | Strike distance | candidates only where \|d2\| from the EWMA-normal model >= z | z in {1.5, 2.0, 2.5} |
| 3 | 6h only | decide at 6h only | none |
| 4 | Model agrees | on the chosen side, the EWMA-FHS model's net edge must be >= m | m in {-0.01, 0.0, 0.005} |
| 5 | Model prior | shrink toward the contract's EWMA-FHS probability instead of the bin's mean mid | none |
| 6 | Cautious | size on the q-quantile of the Beta posterior of f (pessimistic for the chosen side) | q in {0.75, 0.9, 0.95} |
| 7 | Learned | walk-forward logistic regression on logit(mid), logit(model p) and 1[H=3], refit every 7 days | none |
| 8 | Recent | cell counts decay with a half-life in expiry days | h in {60, 90, 180} |
| 9 | Price moved | skip if the fill-hour mid moved more than x toward the money | x in {0.01, 0.02, 0.03} |

Two more models were added after seeing the results of 0-9. They are labeled post-hoc everywhere, and their
holdout scores are optimistic:

| # | Name | Change | Tuning grid |
|---|---|---|---|
| 10 | Learned + moved | Model 7 + Model 9 | x in {0.01, 0.02, 0.03} |
| 11 | Prior + moved | Model 5 + Model 9 | x in {0.01, 0.02, 0.03} |

Rules that apply to all models:
- The estimator always trains on the base tail set (mid < 10c or > 90c). Models 1 and 2 narrow the candidates only.
- The minimum cell count (100) uses raw counts in every model, including Model 8.
- Model 7 trains on every prior base-tail observation. Its probabilities are clipped to [0.001, 0.999].
- Candidate eligibility (cell count >= 100) is the same as in the base.

## Scoring
Each model reports the following for both the tuning period and the holdout:
- Sharpe ratio: mean / sd of daily returns x sqrt(365), counting days without trades.
- A 95% CI for the Sharpe ratio, from a bootstrap over days.
- The Sharpe ratio minus the base's, with a paired bootstrap CI over the same days.
- Return on a fixed $1,000.
- Max drawdown.
- Tail days: days losing more than half the cap.
- Trade count.

If a grid value makes no trades in the tuning period, its Sharpe ratio counts as -inf.

A model counts as better only if its Sharpe ratio beats the base in both periods. Nine models are tested, so one is
likely to beat the base on the holdout by luck.

## Outputs
`variants.py` writes four files to `output/btc_tail_variants/`:
- `tuning.csv`: every grid value's tuning-period metrics
- `comparison.csv`: the chosen settings' tuning and holdout metrics
- `variants_summary.txt`
- `holdout_equity.png`: account value from $1,000 over the holdout, one line per model

## Tests
- Model 0 reproduces the committed long-history trades when that file exists.
- Each option has a test on synthetic data showing it changes what it should.
- Model 7 passes the scrambled-future look-ahead test.
- The Sharpe and paired-bootstrap helpers are tested.
