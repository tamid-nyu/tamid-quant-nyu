"""Predictor v2 features: settlement series and realized vol (spec section 5, F-vol)."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def settlement_series(markets):
    """Settlement value per expiry for non-BTC markets, from raw markets.csv rows.

    Keeps settled rows with a non-null expiration_value and an on-schedule expiry (same rule as
    multi_obs.build_market_obs); one row per expiry (first value), sorted."""
    mk = markets[markets.result.isin(["yes", "no"]) & markets.expiration_value.notna()].copy()
    mk["expiry"] = pd.to_datetime(mk.expiry, utc=True, format="ISO8601")
    on_schedule = mk.expiry.dt.minute.isin([0, 30]) & (mk.expiry.dt.second == 0) & (mk.expiry.dt.microsecond == 0)
    mk = mk[on_schedule].drop_duplicates("expiry", keep="first")
    out = pd.DataFrame({"expiry": mk.expiry, "value": mk.expiration_value.astype(float)})
    return out.sort_values("expiry").reset_index(drop=True)


def btc_settlement_series(spot, expiries):
    """BTC settlement = open of the hourly candle whose time (candle open) equals the expiry."""
    sp = spot.copy()
    sp["time"] = pd.to_datetime(sp["time"], utc=True)
    opens = sp.drop_duplicates("time").set_index("time")["open"]
    exp = pd.Series(pd.to_datetime(pd.Series(expiries), utc=True).unique())
    exp = exp[exp.isin(opens.index)]
    out = pd.DataFrame({"expiry": exp.values, "value": opens.reindex(exp).astype(float).values})
    out["expiry"] = pd.to_datetime(out.expiry, utc=True)
    return out.sort_values("expiry").reset_index(drop=True)


def realized_sigma(series, t, n_changes=20, min_changes=10):
    """Std (ddof=1) of the last n_changes log changes among values with expiry < t; NaN if too few."""
    s = series[series.expiry < t].sort_values("expiry")
    v = s.value.to_numpy(dtype=float)[-(n_changes + 1):]
    if len(v) - 1 < min_changes:
        return np.nan
    return float(np.std(np.diff(np.log(v)), ddof=1))


def pav_decreasing(p):
    """Pool-adjacent-violators fit of a non-increasing sequence (unit weights)."""
    blocks = []  # [sum, count]
    for v in np.asarray(p, dtype=float):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    return np.concatenate([np.full(n, s / n) for s, n in blocks]) if blocks else np.array([])


def ladder_implied(strikes, bid, ask, max_spread=0.10):
    """Implied median M and sigma from a ladder of P(settle > K) quotes; (NaN, NaN) if unusable.

    Only two-sided quotes with ask - bid <= max_spread are used. Mids are PAV-smoothed
    (non-increasing in K); K at p = 0.5/0.25/0.75 is interpolated on the collapsed fit."""
    K, b, a = (np.asarray(x, dtype=float) for x in (strikes, bid, ask))
    ok = np.isfinite(K) & np.isfinite(b) & np.isfinite(a) & (a - b <= max_spread + 1e-9)
    K, mid = K[ok], ((b + a) / 2)[ok]
    if len(K) < 2:
        return np.nan, np.nan
    order = np.argsort(K, kind="stable")
    K, fit = K[order], pav_decreasing(mid[order])
    if fit.max() < 0.75 or fit.min() > 0.25:
        return np.nan, np.nan
    levels = np.unique(fit)  # increasing p; mean K per level collapses flat runs
    kbar = np.array([K[fit == lv].mean() for lv in levels])
    k50, k25, k75 = (float(np.interp(q, levels, kbar)) for q in (0.5, 0.25, 0.75))
    return k50, (k25 - k75) / (2 * 0.6745 * k50)


def ladder_table(hourly, markets, decision_times):
    """Ladder (M, sigma_imp) per (event_ticker, decision_ts), from hourly quotes at exactly decision_ts.

    Duplicate (ticker, ts) quote rows are dropped (first kept). Output has one row per decision_times row."""
    mk = markets[markets.strike_type == "greater"].drop_duplicates("ticker")[["ticker", "event_ticker", "floor"]]
    dt = decision_times[["event_ticker", "decision_ts"]].copy()
    dt["decision_ts"] = pd.to_datetime(dt.decision_ts, utc=True)
    need = dt.drop_duplicates()
    h = hourly[["ticker", "ts", "bid", "ask"]].copy()
    h["ts"] = pd.to_datetime(h.ts, utc=True)
    h = h.drop_duplicates(["ticker", "ts"], keep="first")
    h = h.merge(mk, on="ticker").merge(need, left_on=["event_ticker", "ts"],
                                       right_on=["event_ticker", "decision_ts"]).drop(columns="decision_ts")
    res = {}
    for key, g in h.groupby(["event_ticker", "ts"], sort=False):
        res[key] = ladder_implied(g.floor.to_numpy(float), g.bid.to_numpy(float), g.ask.to_numpy(float))
    nan = (np.nan, np.nan)
    got = [res.get((ev, ts), nan) for ev, ts in zip(dt.event_ticker, dt.decision_ts)]
    out = pd.DataFrame({"event_ticker": dt.event_ticker.to_numpy(), "M": [g[0] for g in got],
                        "sigma_imp": [g[1] for g in got]})
    out.insert(1, "decision_ts", dt.decision_ts.reset_index(drop=True))
    return out


def load_events(path):
    """Event calendar: market -> set of NY dates with a scheduled release (union over families)."""
    with open(path) as fh:
        fams = yaml.safe_load(fh)["families"]
    out = {}
    for fam in fams.values():
        days = {d if isinstance(d, date) else date.fromisoformat(str(d)) for d in fam["dates"]}
        for m in fam["markets"]:
            out.setdefault(m, set()).update(days)
    return out


def event_day(market, day, events):
    """1 if `day` (NY date) is a scheduled event day for `market`, else 0 (unmapped markets give 0)."""
    return int(day in events.get(market, ()))


HERE = Path(__file__).resolve().parent.parent
BTC_PM = HERE.parent / "btc-kalshi-digital" / "output" / "btc_pm"


def load_raw(keys):
    """Raw (markets_df, hourly_df) per market key, plus the BTC hourly spot frame (empty if BTC not requested)."""
    raw, spot = {}, pd.DataFrame(columns=["time", "open", "high", "low", "close"])
    for k in keys:
        if k == "BTC":
            raw[k] = (pd.read_csv(BTC_PM / "btc_kalshi_markets.csv"), pd.read_csv(BTC_PM / "btc_kalshi_hourly.csv"))
            spot = pd.read_csv(BTC_PM / "btc_spot_hourly.csv")
        else:
            d = HERE / "output" / "raw" / k
            raw[k] = (pd.read_csv(d / "markets.csv"), pd.read_csv(d / "hourly.csv"))
    return raw, spot


def build_features(obs, raw, btc_spot, events, session_hours):
    """obs plus sigma_H, M, sigma_imp, vol_prem, vol_missing, event_day (spec section 5).

    Every input is restricted to information at the row's decision_ts: settlements with expiry < decision_ts and
    ladder quotes at exactly decision_ts."""
    out = obs.copy()
    for c in ("sigma_H", "M", "sigma_imp", "vol_prem"):
        out[c] = np.nan
    out["vol_missing"] = 1
    out["event_day"] = [event_day(m, d, events) for m, d in zip(out.market, out.day)]
    for mk, idx in out.groupby("market").groups.items():
        sub = out.loc[idx]
        markets, hourly = raw[mk]
        series = (btc_settlement_series(btc_spot, sub.expiry) if mk == "BTC" else settlement_series(markets))
        ev_of = markets.drop_duplicates("ticker").set_index("ticker").event_ticker
        dts = sub.decision_ts
        sig_d = {t: realized_sigma(series, t) for t in dts.unique()}
        frac = np.minimum(sub.H.to_numpy(float) / session_hours[mk], 1.0)
        sigma_H = dts.map(sig_d).to_numpy(float) * np.sqrt(frac)
        need = pd.DataFrame({"event_ticker": sub.ticker.map(ev_of).to_numpy(), "decision_ts": dts.reset_index(drop=True)})
        lt = ladder_table(hourly, markets, need.drop_duplicates())
        j = need.merge(lt, on=["event_ticker", "decision_ts"], how="left")
        M, sig_imp = j.M.to_numpy(float), j.sigma_imp.to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vp = np.log(sig_imp / sigma_H)
        vp[~np.isfinite(vp)] = np.nan
        out.loc[idx, "sigma_H"], out.loc[idx, "M"], out.loc[idx, "sigma_imp"] = sigma_H, M, sig_imp
        out.loc[idx, "vol_prem"] = vp
    out["vol_missing"] = out.vol_prem.isna().astype(int)
    return out


def main(argv=None):
    import argparse
    import time
    import sys
    sys.path.insert(0, str(HERE))
    import multi_obs

    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    with open(HERE / "markets.yaml") as fh:
        hours = {m["key"]: float(m["session_hours"]) for m in yaml.safe_load(fh)["markets"]}
    obs = multi_obs.load_all(args.markets)
    raw, spot = load_raw(args.markets)
    feats = build_features(obs, raw, spot, load_events(Path(__file__).resolve().parent / "events.yaml"), hours)
    out_dir = Path(__file__).resolve().parent / "output" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_dir / "features.csv", index=False)
    print(f"{'market':8s} {'rows':>7s} {'vol_ok':>7s}")
    for mk in args.markets:
        f = feats[feats.market == mk]
        print(f"{mk:8s} {len(f):7d} {(f.vol_missing == 0).mean():7.1%}")
    print(f"runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
