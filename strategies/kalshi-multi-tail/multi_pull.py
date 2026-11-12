#!/usr/bin/env python3
"""
Kalshi "above K" ladder pull for one market of the multi-market tail strategy.

For each day in the window it lists the settled contracts of that day's event (e.g. KXGOLDD-26SEP1017), keeps the
strikes nearest the money and downloads hourly bid/ask candles for the 14 hours before expiry. "Nearest the money" uses
the previous expiry's settlement value (expiration_value), which is known before any decision for the new day, so the
contract selection does not look ahead. No spot data is needed: outcomes come from Kalshi's `result` field.

Spec: docs/superpowers/specs/2026-11-12-multi-market-tail-design.md (section 3)

Outputs (output/raw/<KEY>/): markets.csv, hourly.csv, run_log.txt. Candles are cached per ticker in output/raw/cache/,
so an interrupted run resumes where it stopped.

Run:  python multi_pull.py --market GOLD --days 860
"""
import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "btc-kalshi-digital"))
import btc_pm_data_pull as bp  # noqa: E402  (reuses get_json, price, fnum, parse_strike, list_event_markets)

RAW = HERE / "output" / "raw"
CACHE = RAW / "cache"
UTC = timezone.utc
bp.SESSION.headers.update({"User-Agent": "TAMID-Quant-Multi-Pull/1.0"})


def load_markets(path=HERE / "markets.yaml"):
    return {m["key"]: m for m in yaml.safe_load(Path(path).read_text())["markets"]}


def event_ticker(series, d, suffix):
    return f"{series}-{d:%y}{d.strftime('%b').upper()}{d:%d}{suffix}"


class FetchError(RuntimeError):
    pass


MIN_INTERVAL = 0.15            # seconds between requests (one process, ~6-7 requests/s), below Kalshi's public limit
_last_request = [0.0]


def get_json(url, params=None, attempts=8, timeout=40):
    """GET with patient backoff. Returns parsed JSON, None for a 404, and raises FetchError when every attempt fails
    (so a rate-limited day is retried or recorded, never silently treated as 'no event')."""
    last = None
    for i in range(attempts):
        wait = MIN_INTERVAL - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.time()
        try:
            r = bp.SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                last = "429"
                time.sleep(min(60, 2 ** i))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** i))
    raise FetchError(f"{url} {params} -> {last}")


def list_event_markets(evt):
    """All markets of one event, live endpoint then historical, with more retries than the BTC pull (the API times
    out intermittently and a missed day would also lose the next day's reference value)."""
    for path in ("/markets", "/historical/markets"):
        out, cursor = [], None
        for _ in range(5):
            p = {"event_ticker": evt, "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = get_json(bp.KALSHI + path, p)
            if not d:
                break
            out.extend([dict(m, _hist=path.startswith("/hist")) for m in d.get("markets", [])])
            cursor = d.get("cursor")
            if not cursor:
                break
        if out:
            return out
    return []


def first_listed(mk, days, step=7):
    """Index into days from which to pull: one probe per `step` days until an event exists, then back up one step
    (so holidays or a missed probe cannot skip the series' real start by more than a week of probing)."""
    for i in range(0, len(days), step):
        for s in mk["series"]:
            if list_event_markets(event_ticker(s, days[i], mk["event_suffix"])):
                return max(0, i - step)
    return None


def candles(series, m, start_ts, end_ts):
    """Hourly yes bid/ask candles for one contract, cached. Same parsing as btc_pm_data_pull.candles."""
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
        d = get_json(bp.KALSHI + p, params)
        if d and d.get("candlesticks"):
            used = p
            break
    hist = bool(used and used.startswith("/historical"))
    rows = []
    for c in (d or {}).get("candlesticks", []):
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        pr, bid, ask = c.get("price", {}), c.get("yes_bid", {}), c.get("yes_ask", {})
        rows.append({"ticker": tk, "ts": int(ts), "bid": bp.price(bid, "close", hist), "ask": bp.price(ask, "close", hist),
                     "close": bp.price(pr, "close", hist), "volume": bp.fnum(c.get("volume") or c.get("volume_fp")),
                     "open_interest": bp.fnum(c.get("open_interest") or c.get("open_interest_fp"))})
    if d is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(rows))
    return rows


def select_strikes(ms, ref, band, max_n):
    """Settled 'greater' contracts within +/- band of ref (all of them when ref is unknown), nearest first."""
    out = []
    for m in ms:
        if m.get("result") not in ("yes", "no"):
            continue
        ps = bp.parse_strike(m)
        if not ps or ps[0] != "greater":
            continue
        fl = ps[1]
        dist = abs(math.log(fl / ref)) if ref else 0.0
        if ref and dist > band:
            continue
        out.append((dist, m, fl))
    out.sort(key=lambda x: x[0])
    return out[:max_n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--days", type=int, default=860)
    ap.add_argument("--max-per-expiry", type=int, default=30)
    ap.add_argument("--hours-before", type=int, default=14)
    a = ap.parse_args(argv)

    mk = load_markets()[a.market]
    out = RAW / a.market
    out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(str(msg))

    now = datetime.now(UTC)
    first = (now - timedelta(days=a.days)).date()
    days = [first + timedelta(days=i) for i in range(a.days + 1)]
    if mk.get("weekdays_only"):
        days = [d for d in days if d.weekday() < 5]
    first_idx = first_listed(mk, days)
    if first_idx is None:
        log(f"{a.market}: no events found in the window")
        return
    days = days[first_idx:]
    log(f"{a.market}: {len(days)} days {days[0]} -> {days[-1]}, series {mk['series']}, band {mk['band']}")

    mrows, hrows = [], []
    prev_value = None
    n_events = 0
    t0 = time.time()

    def save():
        if mrows:
            pd.DataFrame(mrows).to_csv(out / "markets.csv", index=False)
        if hrows:
            h = pd.DataFrame(hrows)
            h["ts"] = pd.to_datetime(h["ts"], unit="s", utc=True)
            h.to_csv(out / "hourly.csv", index=False)
        (out / "run_log.txt").write_text("\n".join(log_lines))

    failed = []
    for i, d in enumerate(days, 1):
        ms, series = [], None
        try:
            for s in mk["series"]:
                ms = list_event_markets(event_ticker(s, d, mk["event_suffix"]))
                if ms:
                    series = s
                    break
        except FetchError as e:
            log(f"  ! {d}: listing failed ({e}); recorded in failed_days.txt")
            failed.append(str(d))
            continue
        settled = [m for m in ms if m.get("result") in ("yes", "no")]
        if not settled:
            continue
        close = max(pd.Timestamp(m["close_time"]) for m in settled)
        if close > pd.Timestamp(now) - pd.Timedelta(hours=1):
            continue
        n_events += 1
        chosen = select_strikes(settled, prev_value, mk["band"], a.max_per_expiry)
        E = close.timestamp()
        got = 0
        for dist, m, fl in chosen:
            try:
                rows = candles(series, m, E - a.hours_before * 3600, E + 3600)
            except FetchError as e:
                log(f"  ! {m['ticker']}: candles failed ({e})")
                failed.append(m["ticker"])
                continue
            if not rows:
                continue
            got += 1
            hrows.extend(rows)
            mrows.append({"market": a.market, "ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
                          "series": series, "strike_type": "greater", "strike_rule": mk["strike_rule"], "floor": fl,
                          "expiry": pd.Timestamp(m["close_time"]).isoformat(), "result": m.get("result"),
                          "expiration_value": bp.fnum(m.get("expiration_value")), "ref_value": prev_value})
        vals = [bp.fnum(m.get("expiration_value")) for m in settled if bp.fnum(m.get("expiration_value"))]
        if vals:
            prev_value = vals[0]
        if n_events <= 3 or n_events % 10 == 0:
            log(f"  {d}: {len(settled)} settled, {len(chosen)} chosen, {got} with candles | events {n_events}, "
                f"contracts {len(mrows)}, rows {len(hrows)} | {time.time() - t0:.0f}s")
        if n_events % 10 == 0:
            save()
    save()
    (out / "failed_days.txt").write_text("\n".join(failed))
    if failed:
        log(f"{a.market}: {len(failed)} days/contracts failed after all retries; re-run to retry (cached data is kept)")
    if mrows:
        mdf = pd.DataFrame(mrows)
        log(f"{a.market}: done. {len(mdf)} contracts over {mdf.expiry.nunique()} expiries "
            f"({mdf.expiry.min()[:10]} -> {mdf.expiry.max()[:10]}), {len(hrows)} hourly rows, {time.time() - t0:.0f}s")
    else:
        log(f"{a.market}: no settled events found")
    save()


if __name__ == "__main__":
    sys.exit(main())
