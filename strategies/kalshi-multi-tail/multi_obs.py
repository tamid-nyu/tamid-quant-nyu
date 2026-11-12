#!/usr/bin/env python3
"""
Observations for the multi-market tail strategy: one row per contract and decision horizon H (6h, 3h before expiry).

decision_ts = floor_to_hour(expiry - H). bid/ask are the hourly candle close quotes at decision_ts, bid_f/ask_f one hour
later (the fill). Rows with a missing quote at either hour are dropped. day = expiry date in New York time.
BTC rows come from the BTC backtest's long-history observations, unchanged.

Spec: docs/superpowers/specs/2026-11-12-multi-market-tail-design.md (section 3)
Run:  python multi_obs.py     (writes output/obs/<KEY>.csv and prints the history gate)
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BTC_DIR = HERE.parent / "btc-kalshi-digital"
sys.path.insert(0, str(BTC_DIR))
import tail_strategy as ts  # noqa: E402

from multi_pull import RAW, load_markets  # noqa: E402

OBS_DIR = HERE / "output" / "obs"
BTC_OBS = BTC_DIR / "output" / "btc_pm_results_long" / "btc_observations.csv"
MIN_HISTORY_DAYS = 150
COLS = ["market", "group", "ticker", "expiry", "day", "H", "y", "bid", "ask", "bid_f", "ask_f", "floor",
        "strike_rule", "strike_type", "decision_ts", "mid"]


def build_market_obs(markets, hourly, key, group, hours=ts.HOURS):
    mk = markets[markets.result.isin(["yes", "no"])].copy()
    mk["expiry"] = pd.to_datetime(mk.expiry, utc=True, format="ISO8601")
    # an event that closes off the :00 / :30 schedule was settled early (e.g. a holiday), so it is not a normal expiry
    on_schedule = mk.expiry.dt.minute.isin([0, 30]) & (mk.expiry.dt.second == 0) & (mk.expiry.dt.microsecond == 0)
    mk = mk[on_schedule]
    hr = hourly.copy()
    hr["ts"] = pd.to_datetime(hr.ts, utc=True)
    book = {tk: g.drop_duplicates("ts").set_index("ts").sort_index()[["bid", "ask"]] for tk, g in hr.groupby("ticker")}
    rows = []
    for r in mk.itertuples(index=False):
        bk = book.get(r.ticker)
        if bk is None:
            continue
        for H in hours:
            t = (r.expiry - pd.Timedelta(hours=H)).floor("h")
            tf = t + pd.Timedelta(hours=1)
            if t not in bk.index or tf not in bk.index:
                continue
            b0, a0, b1, a1 = bk.at[t, "bid"], bk.at[t, "ask"], bk.at[tf, "bid"], bk.at[tf, "ask"]
            if any(pd.isna(x) for x in (b0, a0, b1, a1)):
                continue
            rows.append(dict(market=key, group=group, ticker=r.ticker, expiry=r.expiry,
                             day=r.expiry.tz_convert("America/New_York").date(), H=H,
                             y=1.0 if r.result == "yes" else 0.0, bid=float(b0), ask=float(a0), bid_f=float(b1),
                             ask_f=float(a1), floor=float(r.floor), strike_rule=r.strike_rule, strike_type="greater",
                             decision_ts=t, mid=(float(b0) + float(a0)) / 2))
    return pd.DataFrame(rows, columns=COLS)


def load_btc_obs(path=BTC_OBS):
    o = ts.load_obs(path)                      # same filtering as the BTC base (H in (6, 3), greater, settled)
    o["market"], o["group"], o["strike_rule"] = "BTC", "crypto", "gt"
    o["expiry"] = pd.to_datetime(o.expiry, utc=True)
    o["decision_ts"] = o.expiry - pd.to_timedelta(o.H, unit="h")
    return o[COLS].reset_index(drop=True)


def load_all(keys, obs_dir=OBS_DIR, btc_path=BTC_OBS):
    parts = []
    for k in keys:
        if k == "BTC":
            parts.append(load_btc_obs(btc_path))
            continue
        f = Path(obs_dir) / f"{k}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run multi_obs.py after the {k} pull")
        o = pd.read_csv(f)
        o["expiry"] = pd.to_datetime(o.expiry, utc=True)
        o["decision_ts"] = pd.to_datetime(o.decision_ts, utc=True)
        o["day"] = pd.to_datetime(o.day).dt.date
        parts.append(o[COLS])
    return pd.concat(parts, ignore_index=True)


def main():
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, m in load_markets().items():
        if key == "BTC":
            o = load_btc_obs()
        else:
            d = RAW / key
            if not (d / "markets.csv").exists() or not (d / "hourly.csv").exists():
                print(f"{key}: no raw data, skipped")
                continue
            o = build_market_obs(pd.read_csv(d / "markets.csv"), pd.read_csv(d / "hourly.csv"), key, m["group"])
            o.to_csv(OBS_DIR / f"{key}.csv", index=False)
        tails = ts.tail_candidates(o) if len(o) else o
        nd = o.day.nunique() if len(o) else 0
        rows.append(dict(market=key, enabled=m["enabled"], expiry_days=nd,
                         first=o.day.min() if nd else None, last=o.day.max() if nd else None,
                         observations=len(o), tail_obs=len(tails),
                         gate="ok" if nd >= MIN_HISTORY_DAYS else f"< {MIN_HISTORY_DAYS} days"))
    gate = pd.DataFrame(rows)
    print(gate.to_string(index=False))
    gate.to_csv(OBS_DIR / "history_gate.csv", index=False)


if __name__ == "__main__":
    main()
