from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from v2.logit_v2 import FEATURES, FROZEN, LogitV2Estimator

D0 = date(2025, 1, 1)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _rows(n, n_days, seed, market="BTC", event_bias=0.0, start=D0):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0.01, 0.10, n)
    mid = np.where(rng.uniform(size=n) < 0.5, lo, 1 - lo)
    event = (rng.uniform(size=n) < 0.3).astype(int)
    vp = rng.normal(0, 0.3, n)
    miss = rng.uniform(size=n) < 0.2
    vp[miss] = np.nan
    z = np.log(mid / (1 - mid)) + event_bias * event
    y = (rng.uniform(size=n) < _sigmoid(z)).astype(float)
    days = [start + timedelta(days=int(d)) for d in np.sort(rng.integers(0, n_days, n))]
    return pd.DataFrame({"market": market, "day": days, "H": rng.choice([6, 3], n), "mid": mid, "y": y,
                         "vol_prem": vp, "vol_missing": miss.astype(int), "event_day": event})


def _row(day, market="BTC", mid=0.05, H=6, vol_prem=0.1, vol_missing=0, event_day=0):
    return next(pd.DataFrame([dict(market=market, day=day, H=H, mid=mid, vol_prem=vol_prem,
                                   vol_missing=vol_missing, event_day=event_day)]).itertuples(index=False))


def _feed(est, df):
    for _, g in df.groupby("day", sort=True):
        est.update(g)


def test_feature_sets():
    assert FEATURES["F1"] == ["lm", "h3", "side", "side_lm"]
    assert FEATURES["F2"] == FEATURES["F1"] + ["vol_prem", "vol_missing"]
    assert FEATURES["F3"] == FEATURES["F2"] + ["event_day"]


def test_null_bias_near_zero():
    est = LogitV2Estimator("F3")
    _feed(est, _rows(5000, 60, 1))
    p, n = est.estimate_row(_row(D0 + timedelta(days=60)))
    assert n == 5000 and np.isfinite(p)
    assert all(abs(b) < 0.15 for b in est.beta.values()), est.beta
    assert all(abs(a) < 0.15 for a in est.alpha.values()), est.alpha


def test_planted_event_bias():
    est = LogitV2Estimator("F3")
    _feed(est, _rows(5000, 60, 2, event_bias=0.5))
    est.estimate_row(_row(D0 + timedelta(days=60)))
    assert est.beta["event_day"] / est.sd["event_day"] > 0.25


def test_min_rows_gate():
    est = LogitV2Estimator("F3")
    _feed(est, _rows(150, 20, 3))
    p, n = est.estimate_row(_row(D0 + timedelta(days=30)))
    assert np.isnan(p) and n == 0


def test_never_trains_on_same_or_future_day():
    df = _rows(3000, 40, 4)
    d = D0 + timedelta(days=30)
    est = LogitV2Estimator("F3")
    _feed(est, df[df.day < d])
    r = _row(d)
    p1, n1 = est.estimate_row(r)
    assert n1 == int((df.day < d).sum())
    _feed(est, df[df.day == d])
    p2, n2 = est.estimate_row(r)
    assert (p1, n1) == (p2, n2)
    # a refit that sees later-day rows still trains only on days before the predicted row's day
    est2 = LogitV2Estimator("F3")
    _feed(est2, df)
    assert est2.estimate_row(r)[1] == n1


def test_refit_cadence():
    df = _rows(3000, 60, 5)
    d = D0 + timedelta(days=30)
    est = LogitV2Estimator("F3")
    _feed(est, df[df.day < d])
    p1, n1 = est.estimate_row(_row(d))
    assert est.last_fit_day == d
    _feed(est, df[(df.day >= d) & (df.day < d + timedelta(days=3))])
    p2, n2 = est.estimate_row(_row(d + timedelta(days=3)))
    assert est.last_fit_day == d and n2 == n1 and p2 == p1
    _feed(est, df[(df.day >= d + timedelta(days=3)) & (df.day < d + timedelta(days=7))])
    _, n3 = est.estimate_row(_row(d + timedelta(days=7)))
    assert est.last_fit_day == d + timedelta(days=7)
    assert n3 == int((df.day < d + timedelta(days=7)).sum()) > n1
    assert len(est.cond_log) == 2 and np.isfinite(est.cond)
    assert est.fit_ok == [True, True]


def test_unseen_market_offset_zero():
    est = LogitV2Estimator("F3")
    _feed(est, _rows(2000, 30, 6, market="BTC"))
    p, n = est.estimate_row(_row(D0 + timedelta(days=30), market="SPX"))
    assert np.isfinite(p) and 0.001 <= p <= 0.999 and n == 2000
    assert "SPX" not in est.alpha


def test_warm_start_only_same_columns():
    est = LogitV2Estimator("F3")
    a = _rows(1000, 10, 7)
    a["event_day"] = 0                                   # constant in the first fit -> dropped
    _feed(est, a)
    est.estimate_row(_row(D0 + timedelta(days=10)))
    assert "event_day" not in est.cols_fit and est.warm_started is False
    _feed(est, _rows(1000, 7, 8, start=D0 + timedelta(days=10)))
    est.estimate_row(_row(D0 + timedelta(days=17)))      # column set changed -> start from zeros
    assert "event_day" in est.cols_fit and est.warm_started is False
    _feed(est, _rows(1000, 7, 9, start=D0 + timedelta(days=17)))
    est.estimate_row(_row(D0 + timedelta(days=24)))      # same column set -> warm start
    assert est.warm_started is True


def test_unsettled_rows_skipped():
    est = LogitV2Estimator("F3")
    df = _rows(300, 10, 10)
    df.loc[df.index[:50], "y"] = np.nan
    _feed(est, df)
    assert est.estimate_row(_row(D0 + timedelta(days=10)))[1] == 250


def test_estimate_raises():
    with pytest.raises(TypeError, match="V2 needs estimate_row"):
        LogitV2Estimator().estimate("BTC", 6, 0.05)


def test_make_estimator_v2():
    from estimators import make_estimator
    from multi_tail import MultiConfig
    est = make_estimator("V2", MultiConfig())
    assert isinstance(est, LogitV2Estimator) and est.feature_set == FROZEN["set"]
    with pytest.raises(ValueError, match="S, P, V2 or V3"):
        make_estimator("X", MultiConfig())


def test_constructor_defaults_are_frozen():
    est = LogitV2Estimator()
    got = dict(lam_beta=est.lam_beta, lam_alpha=est.lam_alpha, refit_days=est.refit_days, min_rows=est.min_rows)
    assert got == {k: v for k, v in FROZEN.items() if k != "set"}
