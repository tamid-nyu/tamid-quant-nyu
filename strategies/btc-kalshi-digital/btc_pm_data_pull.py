#!/usr/bin/env python3
"""
Bitcoin prediction-market data pull (Kalshi BTC price contracts + BTC spot)
===========================================================================
Step 1 of the "contract price vs model probability" strategy.

Kalshi lists Bitcoin price contracts that settle at 5pm ET each day ("BTC above $X?" and "BTC between $X and $Y?").
This script downloads, for each day in the window:
  * the settled contracts for that day's 5pm-ET expiry (strike, result)
  * hourly bid/ask candles for the contracts nearest the money, for the ~14 hours before expiry
and BTC spot (hourly) from Coinbase, going back far enough to estimate volatility.

Outputs (backtests/output/btc_pm/):
  btc_kalshi_markets.csv, btc_kalshi_hourly.csv, btc_spot_hourly.csv, run_log.txt

Needs:  pip install requests pandas
Run:    python btc_pm_data_pull.py                (last 120 days)
        python btc_pm_data_pull.py --days 365
        python btc_pm_data_pull.py --discover     (only list Kalshi Bitcoin series and sample contracts, then stop)
It caches every candle download, so re-running after an interruption resumes where it stopped.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "output" / "btc_pm"
CACHE = OUT / "cache"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
KRAKEN = "https://api.kraken.com/0/public/OHLC"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

SERIES_CANDIDATES = ["KXBTCD", "BTCD"]         # hourly above/below ladder (current), older daily ladder (fallback)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "TAMID-Quant-BTC-Pull/1.0"})
_log = []


def log(msg=""):
    print(msg, flush=True)
    _log.append(str(msg))


def get_json(url, params=None, retries=4, timeout=30):
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                last = "429 rate limited"
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(0.7 * (attempt + 1))
    log(f"    ! request failed: {url} -> {last}")
    return None


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def price(d, key, dollars=False):
    """Kalshi price -> 0-1. '<key>_dollars' is dollars; bare '<key>' is cents on the live endpoint and
    already dollars on the /historical endpoint (pass dollars=True)."""
    if not isinstance(d, dict):
        return None
    v = fnum(d.get(key + "_dollars"))
    if v is not None:
        return v
    v = fnum(d.get(key))
    if v is None:
        return None
    return v if dollars else v / 100.0


# ----------------------------------------------------------------------------
# SPOT
# ----------------------------------------------------------------------------
def spot_coinbase(start, end):
    rows, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(hours=299), end)
        d = get_json(COINBASE, {"granularity": 3600, "start": cur.isoformat(), "end": nxt.isoformat()})
        if d:
            rows.extend(d)
        cur = nxt + timedelta(hours=1)
        time.sleep(0.15)
    # [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    return df


def spot_kraken(start, end):
    rows, since = [], int(start.timestamp())
    while since < end.timestamp():
        d = get_json(KRAKEN, {"pair": "XBTUSD", "interval": 60, "since": since})
        if not d or not d.get("result"):
            break
        key = [k for k in d["result"] if k != "last"][0]
        chunk = d["result"][key]
        if not chunk:
            break
        rows.extend(chunk)
        nxt = int(d["result"].get("last", chunk[-1][0]))
        if nxt <= since:
            break
        since = nxt
        time.sleep(0.5)
    df = pd.DataFrame([r[:5] for r in rows], columns=["t", "open", "high", "low", "close"])
    return df


def get_spot(start, end, source):
    log(f"  spot: {source} hourly {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    df = spot_coinbase(start, end) if source == "coinbase" else spot_kraken(start, end)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time")
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time", "open", "high", "low", "close"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# KALSHI
# ----------------------------------------------------------------------------
def discover():
    log("\n=== KALSHI BITCOIN SERIES ===")
    found = []
    d = get_json(f"{KALSHI}/series", {"category": "Crypto"})
    for s in (d or {}).get("series", []) or []:
        title = (s.get("title") or "")
        if "bitcoin" in title.lower() or "btc" in (s.get("ticker") or "").lower():
            log(f"  {s.get('ticker'):<18} {title}  [freq={s.get('frequency')}]")
            found.append(s.get("ticker"))
    for t in SERIES_CANDIDATES:
        d = get_json(f"{KALSHI}/series/{t}")
        s = (d or {}).get("series", d) or {}
        log(f"  candidate {t}: {'FOUND  ' + str(s.get('title')) if s.get('ticker') else 'not found'}")
    return found


def list_markets(series, min_ts, max_ts):
    """Markets closing in a time window (used by --discover only; bounded)."""
    out, cursor = [], None
    for _ in range(3):
        p = {"series_ticker": series, "min_close_ts": int(min_ts), "max_close_ts": int(max_ts), "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = get_json(KALSHI + "/markets", p)
        if not d:
            break
        out.extend(d.get("markets", []))
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def list_event_markets(event_ticker):
    """All markets of ONE event (e.g. KXBTCD-26OCT1617). Live endpoint first, then the historical one.
    Querying by event ticker keeps every call small; paging the whole historical series would never finish."""
    for path in ("/markets", "/historical/markets"):
        out, cursor = [], None
        for _ in range(5):                                   # at most 5 pages (5,000 markets) per event
            p = {"event_ticker": event_ticker, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = get_json(KALSHI + path, p, retries=2, timeout=20)
            if not d:
                break
            out.extend([dict(m, _hist=path.startswith("/hist")) for m in d.get("markets", [])])
            cursor = d.get("cursor")
            if not cursor:
                break
        if out:
            return out
    return []


def parse_strike(m):
    """-> (strike_type, floor, cap) with strike_type in {'greater','less','between'} or None."""
    st = (m.get("strike_type") or "").lower()
    fl, cp = fnum(m.get("floor_strike")), fnum(m.get("cap_strike"))
    if st in ("greater", "greater_or_equal") and fl is not None:
        return "greater", fl, None
    if st in ("less", "less_or_equal") and cp is not None:
        return "less", None, cp
    if st == "between" and fl is not None and cp is not None:
        return "between", fl, cp
    # fallback: ticker suffix  ...-T117999.99 (above)  ...-B117250 (bucket midpoint, width unknown -> skip)
    suf = (m.get("ticker") or "").split("-")[-1]
    if suf.startswith("T") and fnum(suf[1:]) is not None:
        return "greater", fnum(suf[1:]), None
    return None


def candles(series, m, start_ts, end_ts):
    tk = m["ticker"]
    cf = CACHE / f"{tk}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    params = {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": 60}
    paths = [f"/series/{series}/markets/{tk}/candlesticks", f"/historical/markets/{tk}/candlesticks"]
    if m.get("_hist"):
        paths.reverse()
    d, used = None, None
    for p in paths:
        d = get_json(KALSHI + p, params)
        if d and d.get("candlesticks"):
            used = p
            break
    rows = []
    hist = bool(used and used.startswith("/historical"))
    for c in (d or {}).get("candlesticks", []):
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        pr, bid, ask = c.get("price", {}), c.get("yes_bid", {}), c.get("yes_ask", {})
        rows.append({"ticker": tk, "ts": int(ts), "bid": price(bid, "close", hist), "ask": price(ask, "close", hist),
                     "close": price(pr, "close", hist), "volume": fnum(c.get("volume") or c.get("volume_fp")),
                     "open_interest": fnum(c.get("open_interest") or c.get("open_interest_fp"))})
    if d is not None:                      # cache only successful responses (a failed request must be retried next run)
        CACHE.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--expiry-et-hour", type=int, default=17, help="settlement hour in New York time (5pm = 17)")
    ap.add_argument("--series", nargs="*", default=None)
    ap.add_argument("--band", type=float, default=0.06, help="keep strikes within +/- this fraction of spot")
    ap.add_argument("--max-per-expiry", type=int, default=30, help="max contracts per expiry (closest to the money)")
    ap.add_argument("--spot-source", choices=["coinbase", "kraken"], default="coinbase")
    ap.add_argument("--all-series", action="store_true", help="use every listed series each day (duplicates economically identical contracts)")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--vol-history-days", type=int, default=110)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"Output folder: {OUT}")
    found = discover()
    series_list = a.series or list(SERIES_CANDIDATES)
    if a.discover:
        # show sample contracts for the first candidate so we can check field names
        for s in series_list:
            now = time.time()
            ms = list_markets(s, now - 5 * 86400, now + 2 * 86400)
            log(f"\n  {s}: {len(ms)} markets closing in the last 5 days; first 3:")
            for m in ms[:3]:
                log("   " + json.dumps({k: m.get(k) for k in ("ticker", "event_ticker", "title", "yes_sub_title", "strike_type",
                                                              "floor_strike", "cap_strike", "close_time", "status", "result")}))
        (OUT / "run_log.txt").write_text("\n".join(_log))
        return

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    first_day = (end - timedelta(days=a.days)).date()
    # ---- spot first (needed to pick near-the-money strikes and for volatility history)
    spot = get_spot(end - timedelta(days=a.days + a.vol_history_days), end + timedelta(hours=1), a.spot_source)
    if spot.empty:
        log("  ! no spot data returned; cannot continue")
        return
    spot.to_csv(OUT / "btc_spot_hourly.csv", index=False)
    log(f"  saved btc_spot_hourly.csv ({len(spot)} rows, {spot.time.min()} -> {spot.time.max()})")
    sp = spot.set_index("time").close

    def spot_at(t):                     # close of the hourly bar that ENDS at t
        k = t - timedelta(hours=1)
        return float(sp.loc[k]) if k in sp.index else None

    # ---- contracts per day
    mrows, hrows = [], []

    def save(final=False):
        if not mrows:
            return
        pd.DataFrame(mrows).to_csv(OUT / "btc_kalshi_markets.csv", index=False)
        h = pd.DataFrame(hrows)
        h["ts"] = pd.to_datetime(h["ts"], unit="s", utc=True)
        h.to_csv(OUT / "btc_kalshi_hourly.csv", index=False)

    days = [first_day + timedelta(days=i) for i in range(a.days)]
    t0 = time.time()
    n_days_ok = 0
    for i, d in enumerate(days, 1):
        E = datetime(d.year, d.month, d.day, a.expiry_et_hour, 0, tzinfo=ET).astimezone(UTC)
        if E > end - timedelta(hours=1):
            continue
        S0 = spot_at(E - timedelta(hours=12))
        if S0 is None:
            continue
        day_markets = []
        local = datetime(d.year, d.month, d.day, a.expiry_et_hour)
        for s in series_list:
            if day_markets and not a.all_series:
                break                   # the first series with data for this day wins (avoids duplicate contracts)
            evt = f"{s}-{local:%y}{local:%b}{local:%d}{local:%H}".upper()
            for m in list_event_markets(evt):
                if m.get("result") not in ("yes", "no"):
                    continue
                ps = parse_strike(m)
                if not ps:
                    continue
                st, fl, cp = ps
                ref = fl if st == "greater" else cp if st == "less" else (fl + cp) / 2
                dist = abs(math.log(ref / S0))
                if dist > a.band:
                    continue
                day_markets.append((dist, s, m, st, fl, cp))
        day_markets.sort(key=lambda x: x[0])
        got = 0
        for dist, s, m, st, fl, cp in day_markets[: a.max_per_expiry]:
            rows = candles(s, m, E.timestamp() - 14 * 3600, E.timestamp() + 3600)
            time.sleep(0.1)
            if not rows:
                continue
            got += 1
            hrows.extend(rows)
            mrows.append({"ticker": m["ticker"], "event_ticker": m.get("event_ticker"), "series": s,
                          "strike_type": st, "floor": fl, "cap": cp, "expiry": E.isoformat(),
                          "close_time": m.get("close_time"), "result": m.get("result"),
                          "volume": fnum(m.get("volume") or m.get("volume_fp")), "title": m.get("title")})
        n_days_ok += 1 if got else 0
        if i <= 3 or i % 5 == 0:
            log(f"    day {i}/{len(days)} ({d}): {len(day_markets)} candidate contracts, {got} kept | total {len(mrows)} "
                f"contracts, {len(hrows)} hourly rows | {time.time() - t0:.0f}s elapsed")
        if i % 10 == 0:
            save()
    if not mrows:
        log("  ! No contracts collected. Run with --discover and send me the output.")
        (OUT / "run_log.txt").write_text("\n".join(_log))
        return
    save(final=True)
    mdf = pd.DataFrame(mrows)
    hdf = pd.DataFrame(hrows)
    log(f"\n  saved btc_kalshi_markets.csv ({len(mdf)} contracts over {mdf.expiry.nunique()} expiries)")
    log(f"  saved btc_kalshi_hourly.csv ({len(hdf)} rows)")
    log("  contracts by type:\n" + mdf.strike_type.value_counts().to_string())
    log(f"  YES rate: {(mdf.result == 'yes').mean():.1%}")
    log(f"  hourly rows with both bid and ask: {(hdf.bid.notna() & hdf.ask.notna()).mean():.1%}")
    (OUT / "run_log.txt").write_text("\n".join(_log))
    log("\nDone. Tell Claude when it finishes.")


if __name__ == "__main__":
    sys.exit(main())
