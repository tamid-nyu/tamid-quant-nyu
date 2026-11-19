import datetime as dt

import numpy as np
import pandas as pd
import pytest

from multi_tail import MultiConfig
from v2.scoring import PRED_COLS
from v3 import select_w
from v3.select_w import NO_ARM_LINE, WEIGHTS, choose_w


def _days(D=70, per=12, start=dt.date(2025, 3, 3)):
    return np.repeat([start + dt.timedelta(days=i) for i in range(D)], per)


def _losses(half_gap, noise, seed=0):
    rng = np.random.default_rng(seed)
    day = _days()
    base = rng.uniform(0.05, 0.15, len(day))
    ll = {w: base + 1.0 for w in WEIGHTS}
    ll[0.75] = base
    ll[0.5] = base + half_gap + rng.normal(0, noise, len(day))
    return ll, day


def test_select_w_prefers_half_within_se():
    ll, day = _losses(half_gap=0.001, noise=0.2)
    w, table = choose_w(ll, day)
    r = table.set_index("w")
    assert r.loc[0.75, "delta"] == 0.0 and r.loc[0.5, "delta"] > 0
    assert r.loc[0.5, "delta"] <= r.loc[0.5, "se"]
    assert w == 0.5


def test_select_w_takes_best_outside_se():
    ll, day = _losses(half_gap=0.05, noise=0.01)
    w, table = choose_w(ll, day)
    r = table.set_index("w")
    assert r.loc[0.5, "delta"] > r.loc[0.5, "se"]
    assert w == 0.75


def _obs(start, days, flip=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, hr in (("BTC", 21), ("SPX", 20)):
        for i in range(days):
            d = start + dt.timedelta(days=i)
            E = pd.Timestamp(d.isoformat() + f"T{hr}:00Z")
            for k in range(30):
                for H in (6, 3):
                    bid = round(float(rng.choice([0.02, 0.03, 0.04, 0.05, 0.92, 0.94])), 2)
                    ask = round(bid + 0.01, 2)
                    mid = (bid + ask) / 2
                    y = float(rng.uniform() < (0.12 if mid < 0.5 else 0.93))
                    rows.append(dict(market=m, group=m, ticker=f"{m}-{d}-{k}", expiry=E, day=d, H=H,
                                     y=1.0 - y if flip else y, bid=bid, ask=ask, bid_f=bid, ask_f=ask,
                                     floor=100.0 + k, strike_rule="gt", strike_type="greater",
                                     decision_ts=E - pd.Timedelta(hours=H), mid=mid, vol_prem=float(rng.normal()),
                                     vol_missing=0, event_day=int(i % 5 == 0)))
    return pd.DataFrame(rows)


CFG = MultiConfig(estimator="P", warmup_days=3, q_max=1000, slip=0.0)


def test_select_w_ignores_post_dev_rows(monkeypatch):
    dev = _obs(dt.date(2026, 5, 22), 40)
    assert dev.day.max() <= select_w.DEV_END
    post = _obs(dt.date(2026, 7, 1), 15, flip=True, seed=1)
    seen = []
    real = select_w.walk_forward_predict

    def spy(obs, est, cfg):
        seen.append(obs.day.max())
        return real(obs, est, cfg)

    monkeypatch.setattr(select_w, "walk_forward_predict", spy)
    t_dev = select_w.run_selection(dev, CFG, score_start=dt.date(2026, 5, 22))
    n_dev = len(seen)
    t_all = select_w.run_selection(pd.concat([dev, post], ignore_index=True), CFG, score_start=dt.date(2026, 5, 22))
    assert t_all == t_dev
    assert len(seen) == 2 * n_dev and max(seen) <= select_w.DEV_END
    assert "CHOSEN w:" in t_dev


@pytest.mark.parametrize("w_star", [0.0, 1.0])
def test_select_w_degenerate_note(monkeypatch, w_star):
    rng = np.random.default_rng(3)
    day = _days(D=40, per=20, start=dt.date(2025, 2, 1))
    y = (rng.uniform(size=len(day)) < 0.1).astype(float)
    base = pd.DataFrame(dict(day=day, market=np.where(np.arange(len(day)) % 2, "BTC", "SPX"),
                             ticker=[f"t{i}" for i in range(len(day))], H=np.where(np.arange(len(day)) % 3, 6, 3),
                             mid=0.1, bid=0.095, ask=0.105, bid_f=0.095, ask_f=0.105, y=y))

    good = np.where(y == 1, 0.9, 0.02)

    def fake(obs, est, cfg):
        if type(est).__name__ == "PooledEstimator":
            return base.assign(p=good if w_star == 1.0 else 0.5, n=500)[PRED_COLS]
        return base.assign(p=good if w_star == 0.0 else 0.5, n=1000)[PRED_COLS]

    monkeypatch.setattr(select_w, "walk_forward_predict", fake)
    text = select_w.run_selection(_obs(dt.date(2025, 1, 1), 2), CFG)
    assert f"CHOSEN w: {w_star:.2f}" in text
    assert NO_ARM_LINE in text
    assert NO_ARM_LINE == "Chosen w reproduces a component; no new v3 arm is frozen."
