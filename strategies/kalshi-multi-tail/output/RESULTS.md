# Multi-market tail strategy: first results (2026-11-12)

Spec: `docs/superpowers/specs/2026-11-12-multi-market-tail-design.md`. Bankroll $1,000; per-market cap c = 3%;
the k = 2 worst markets together capped at C = 4%. Holdout 2026-07-01 → 2026-11-11 is the headline. Sharpe is
annualized from daily returns, with days without trades counted as 0. "diff" is Sharpe minus BTC-only on the same
days, with a paired-bootstrap 95% CI.

## Data
| market | expiry days | range | status |
|---|---|---|---|
| BTC | 815 | 2024-07-05 → 2026-11-11 | existing long history |
| SPX (`KXINXU`) | 434 | 2025-02-18 → 2026-11-11 | passes 150-day gate |
| GOLD (`KXGOLDD`) | 128 | 2026-03-24 → 2026-11-11 | below gate |
| WTI (`KXWTI`) | 122 | 2026-05-18 → 2026-11-10 | below gate |
| NATGAS (`KXNATGASD`) | 116 | 2026-04-15 → 2026-11-10 | below gate |

Kalshi only began listing the commodity daily ladders in 2026. The Jan 19, 2026 SPX event (MLK Day) was settled
early and is excluded. Nasdaq-100 and silver were not pulled. Fees: Kalshi's API lists every series at the standard
0.07 × P × (1 − P).

## Run A: by the spec (BTC + SPX), `gate150/`
| holdout | Sharpe [95% CI] | diff vs BTC [CI] | return | max DD | wins/day | trades |
|---|---|---|---|---|---|---|
| S multi | 0.86 [−2.16, 7.26] | −0.21 [−2.55, 0.87] | +3.37% | −4.85% | 1.96 | 274 |
| P multi | 1.24 [−1.83, 6.57] | +0.17 [−5.07, 3.57] | +5.98% | −6.68% | 2.56 | 361 |
| option 1 (S, 6% total) | 1.23 [−1.90, 7.90] | +0.16 [−2.09, 1.38] | +4.83% | −4.19% | 1.99 | 276 |
| BTC only (Model 0) | 1.07 [−1.90, 7.34] | 0 | +3.93% | −3.03% | 1.82 | 254 |

Check period (Jan–Jun 2026): P multi Sharpe 3.81 [1.15, 8.81], +16.8%; S multi −0.19; BTC only 0.29.

## Run B: rule broken on purpose (all five, minimum 100 days), `gate100_posthoc/`
| holdout | Sharpe [95% CI] | diff vs BTC [CI] | return | max DD | wins/day | trades |
|---|---|---|---|---|---|---|
| S multi | 1.66 [−1.53, 10.06] | +0.59 [−1.31, 3.14] | +5.47% | −2.87% | 2.13 | 298 |
| P multi | 0.67 [−2.27, 5.46] | −0.40 [−6.73, 4.00] | +3.79% | −7.25% | 4.55 | 635 |
| option 1 (S, 6% total) | 0.72 [−2.24, 6.16] | −0.35 [−3.96, 1.55] | +3.09% | −4.35% | 2.09 | 291 |
| BTC only (Model 0) | 1.07 [−1.90, 7.34] | 0 | +3.93% | −3.03% | 1.82 | 254 |

Holdout P&L by market, Run B, P: BTC +$23.70, SPX +$26.29, GOLD +$10.57, NATGAS −$2.90, **WTI −$19.73**
(142 trades, both sides lost). Under S the commodity markets barely trade (cells rarely reach 100 observations).

## What it means
1. **More wins: yes.** P on five markets made 4.6 winning trades a day against 1.8 for BTC alone, and daily P&L
   correlation between markets is about 0 (largest: SPX–WTI 0.32). No day had two markets lose more than 1% together.
2. **Better risk-adjusted return: not shown.** No run's holdout Sharpe difference vs BTC-only excludes zero. Eight
   settings were compared, so the best holdout number (Run B S, +0.59) is expected by chance.
3. **SPX looks like the best addition.** It was positive under P in both periods ($126 over its history), mostly from
   selling longshot YES. Under P, BTC's buy-YES side lost money in the holdout.
4. **WTI hurt.** The longshot bias may not exist there, or 122 days is too few to tell.
5. **The caps held.** Assumed worst cases never exceeded the caps. Realized k-worst breaches came from one-hour
   fill slippage, at most $0.76 over the $40 cap.

## Caveats
- The holdout had been seen before (BTC variants work), and Run B breaks the spec's history rule. Treat Run B as
  exploratory.
- P on BTC alone is a different model from S (it shrinks twice), so some of P's gain is not from pooling.
- Gold, WTI and gas have about 4–6 months of trading each, so their results are thin.
- The next clean test is data after 2026-11-11.
