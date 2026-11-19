import datetime as dt

import numpy as np
import pandas as pd
import pytest

import estimators
from estimators import PooledEstimator, make_estimator
from multi_tail import MultiConfig
from v2.logit_v2 import FROZEN as F1_FROZEN, LogitV2Estimator
from v2.scoring import walk_forward_predict
from v3 import blend
from v3.blend import BlendEstimator

CFG = dict(warmup_days=3, min_cell_n=5, q_max=1000, hours=(6, 3), slip=0.0, n0=50, n1=100)


def _obs(markets=("A", "B"), days=30, start=dt.date(2026, 1, 5), seed=0, strikes=40):
    rng = np.random.default_rng(seed)
    rows = []
    for mi, m in enumerate(markets):
        for i in range(days):
            d = start + dt.timedelta(days=i)
            E = pd.Timestamp(d.isoformat() + f"T{19 + mi:02d}:00Z")
            for k in range(strikes):
                for H in (6, 3):
                    bid = round(float(rng.choice([0.02, 0.03, 0.04, 0.05, 0.92, 0.94])), 2)
                    ask = round(bid + 0.01, 2)
                    mid = (bid + ask) / 2
                    y = float(rng.uniform() < (0.12 if mid < 0.5 else 0.93))
                    rows.append(dict(market=m, group=m, ticker=f"{m}-{d}-{k}", expiry=E, day=d, H=H, y=y,
                                     bid=bid, ask=ask, bid_f=bid, ask_f=ask, floor=100.0 + k, strike_rule="gt",
                                     strike_type="greater", decision_ts=E - pd.Timedelta(hours=H), mid=mid,
                                     vol_prem=float(rng.normal()), vol_missing=0, event_day=int(i % 5 == 0)))
    return pd.DataFrame(rows)


def _f1():
    return LogitV2Estimator(F1_FROZEN["set"], F1_FROZEN["lam_beta"], F1_FROZEN["lam_alpha"],
                            F1_FROZEN["refit_days"], F1_FROZEN["min_rows"])


def _cfg(**kw):
    return MultiConfig(estimator="P", **{**CFG, **kw})


def _preds(obs, est):
    return walk_forward_predict(obs, est, _cfg())


def test_blend_endpoints():
    obs = _obs()
    pp, pf = _preds(obs, PooledEstimator(50, 100)), _preds(obs, _f1())
    p1, p0 = _preds(obs, BlendEstimator(1.0)), _preds(obs, BlendEstimator(0.0))
    both = (pp.n >= 100) & pp.p.notna() & pf.p.notna() & (pf.n >= 200)
    assert both.sum() > 100
    np.testing.assert_allclose(p1.p[both], pp.p[both], atol=1e-12)
    np.testing.assert_allclose(p0.p[both], pf.p[both], atol=1e-12)


def test_blend_half_is_mean():
    obs = _obs()
    pp, pf = _preds(obs, PooledEstimator(50, 100)), _preds(obs, _f1())
    ph = _preds(obs, BlendEstimator(0.5))
    both = (pp.n >= 100) & pf.p.notna()
    assert both.sum() > 100
    np.testing.assert_allclose(ph.p[both], np.clip((pp.p[both] + pf.p[both]) / 2, 0.001, 0.999), atol=1e-12)
    assert (ph.n[both] == 1).all()


@pytest.mark.parametrize("p_out, f_out", [((0.1, 50), (0.2, 300)), ((0.1, 150), (float("nan"), 0))])
def test_one_component_ineligible(p_out, f_out):
    est = BlendEstimator(0.5)
    est.p_est.estimate = lambda m, H, mid: p_out
    est.v2_est.estimate_row = lambda r: f_out
    r = next(_obs(days=1, strikes=1).itertuples(index=False))
    p, n = est.estimate_row(r)
    assert np.isnan(p) and n == 0


def test_blend_matches_after_the_fact():
    obs = _obs()
    pp, pf = _preds(obs, PooledEstimator(50, 100)), _preds(obs, _f1())
    pb = _preds(obs, BlendEstimator(0.25))
    ok = (pp.n >= 100) & pf.p.notna()
    assert ok.sum() > 100
    elig = pb.n == 1
    assert (elig == ok).all()
    np.testing.assert_allclose(pb.p[elig], np.clip(0.25 * pp.p[ok] + 0.75 * pf.p[ok], 0.001, 0.999), atol=1e-12)


def test_blend_no_lookahead():
    """Predictions for day <= d are identical whether or not data from day d onward was ever seen by update."""
    obs = _obs()
    d = sorted(obs.day.unique())[20]
    full = _preds(obs, BlendEstimator(0.5))
    cut = _preds(obs[obs.day <= d], BlendEstimator(0.5))
    ref = full[full.day <= d].reset_index(drop=True)
    assert (ref.n == 1).sum() > 50 and len(ref) == len(cut)
    pd.testing.assert_frame_equal(ref, cut.reset_index(drop=True))
    # and updating with day d after predicting it leaves that prediction unchanged
    est = BlendEstimator(0.5)
    for _, g in obs[obs.day < d].groupby("day"):
        est.update(g)
    r = next(obs[(obs.day == d) & (obs.H == 6)].itertuples(index=False))
    f1_before = est.v2_est.estimate_row(r)
    est.update(obs[obs.day == d])
    assert est.v2_est.estimate_row(r) == f1_before


def test_estimate_raises():
    with pytest.raises(TypeError):
        BlendEstimator(0.5).estimate("A", 6, 0.05)


def test_make_estimator_v3_unfrozen(monkeypatch):
    monkeypatch.setitem(blend.FROZEN, "w", None)
    with pytest.raises(ValueError, match="not frozen"):
        make_estimator("V3", _cfg())
    monkeypatch.setitem(blend.FROZEN, "w", 0.25)
    est = make_estimator("V3", _cfg())
    assert isinstance(est, BlendEstimator) and est.w == 0.25 and est.min_rows == 1
    with pytest.raises(ValueError, match="S, P, V2 or V3"):
        make_estimator("X", _cfg())
