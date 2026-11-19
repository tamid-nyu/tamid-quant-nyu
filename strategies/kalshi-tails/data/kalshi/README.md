# Kalshi data

Hourly quotes and contract metadata for the Kalshi "above $K" ladders used in this project, pulled from Kalshi's
public API. A copy of the pulled data; the code still reads from each strategy's own `output/` folders.

| Market | Series | Contracts | Hourly rows | First day | Last day |
|---|---|---|---|---|---|
| BTC | KXBTCD | 19,299 | 266,501 | 2024-07-05 | 2026-11-11 |
| SPX (S&P 500) | KXINXU | 9,901 | 114,955 | 2025-02-18 | 2026-11-11 |
| GOLD | KXGOLDD | 3,835 | 58,098 | 2026-03-24 | 2026-11-11 |
| WTI | KXWTI | 2,900 | 41,673 | 2026-05-18 | 2026-11-10 |
| NATGAS | KXNATGASD | 3,480 | 52,096 | 2026-04-15 | 2026-11-10 |

Nothing after 2026-11-11 is included; the forward test (2026-11-12 to 2027-05-12) is not in this data.

## Files (one folder per market)
- `hourly.csv`: one row per contract per hour, `ticker, ts (UTC), bid, ask, close, volume, open_interest`, prices in dollars
- `markets.csv`: one row per contract with strike, expiry and settlement `result` (yes/no). BTC was pulled by an earlier
  script (`btc-kalshi-digital/btc_pm_data_pull.py`), so its columns differ slightly from the other four
  (`kalshi-multi-tail/multi_pull.py`)

- `BTC/spot_hourly_coinbase.csv`: hourly BTC-USD candles from Coinbase (`time, open, high, low, close`, UTC), 2024-03-17 to
  2026-11-11, 23,267 rows; used for the volatility model in the BTC digital backtest. Not Kalshi data

Not included: per-contract JSON caches (duplicates of these CSVs).
