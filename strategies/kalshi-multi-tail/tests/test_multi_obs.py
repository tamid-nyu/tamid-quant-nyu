import datetime as dt

import numpy as np
import pandas as pd

from multi_obs import build_market_obs


def _mk(expiry, result="no", floor=100.0, ticker="X-T100"):
    return pd.DataFrame([{"market": "WTI", "ticker": ticker, "event_ticker": "X", "series": "KXWTI",
                          "strike_type": "greater", "strike_rule": "gt", "floor": floor, "expiry": expiry,
                          "result": result, "expiration_value": 99.0, "ref_value": None}])


def _hr(ticker, stamps, bid=0.03, ask=0.05):
    return pd.DataFrame([{"ticker": ticker, "ts": pd.Timestamp(s), "bid": bid, "ask": ask} for s in stamps])


def test_decision_floor_wti():
    mk = _mk("2026-09-10T18:30:00+00:00")
    hr = pd.concat([_hr("X-T100", ["2026-09-10T12:00Z"], 0.03, 0.05), _hr("X-T100", ["2026-09-10T13:00Z"], 0.02, 0.04)])
    o = build_market_obs(mk, hr, "WTI", "crude", hours=(6,))
    assert len(o) == 1
    r = o.iloc[0]
    assert r.decision_ts == pd.Timestamp("2026-09-10T12:00Z")
    assert (r.bid, r.ask, r.bid_f, r.ask_f) == (0.03, 0.05, 0.02, 0.04)


def test_on_hour_expiry():
    mk = _mk("2026-09-10T21:00:00+00:00")
    hr = pd.concat([_hr("X-T100", ["2026-09-10T18:00Z"], 0.01, 0.03), _hr("X-T100", ["2026-09-10T19:00Z"], 0.02, 0.04)])
    o = build_market_obs(mk, hr, "WTI", "crude", hours=(3,))
    assert len(o) == 1 and o.iloc[0].decision_ts == pd.Timestamp("2026-09-10T18:00Z")
    assert o.iloc[0].H == 3 and o.iloc[0].bid_f == 0.02


def test_missing_quote_dropped():
    mk = _mk("2026-09-10T21:00:00+00:00")
    hr = pd.concat([_hr("X-T100", ["2026-09-10T18:00Z"], np.nan, 0.03), _hr("X-T100", ["2026-09-10T19:00Z"])])
    assert build_market_obs(mk, hr, "WTI", "crude", hours=(3,)).empty


def test_day_is_new_york_date():
    mk = _mk("2026-11-10T01:00:00+00:00")
    hr = pd.concat([_hr("X-T100", ["2026-11-09T22:00Z", "2026-11-09T23:00Z"])])
    o = build_market_obs(mk, hr, "WTI", "crude", hours=(3,))
    assert o.iloc[0].day == dt.date(2026, 11, 9)


def test_y_from_result():
    mk = _mk("2026-09-10T21:00:00+00:00", result="yes")
    hr = _hr("X-T100", ["2026-09-10T18:00Z", "2026-09-10T19:00Z"])
    o = build_market_obs(mk, hr, "WTI", "crude", hours=(3,))
    assert o.iloc[0].y == 1.0 and o.iloc[0].market == "WTI" and o.iloc[0].group == "crude"
    assert abs(o.iloc[0].mid - 0.04) < 1e-12


def test_off_schedule_close_dropped():
    # holiday event settled early (KXINXU-26JAN19H1600 closed 2026-01-17T17:53:55.869664Z): not a normal expiry
    mk = _mk("2026-01-17T17:53:55.869664+00:00")
    hr = _hr("X-T100", ["2026-01-17T14:00Z", "2026-01-17T15:00Z", "2026-01-17T11:00Z", "2026-01-17T12:00Z"])
    assert build_market_obs(mk, hr, "SPX", "equity", hours=(3, 6)).empty
