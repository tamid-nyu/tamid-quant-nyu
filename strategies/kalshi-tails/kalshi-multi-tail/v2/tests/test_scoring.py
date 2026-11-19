import datetime as dt

import numpy as np
import pandas as pd

from multi_tail import MultiConfig, run
from estimators import make_estimator
from v2.scoring import block_bootstrap_delta, select_set, walk_forward_predict

CFG = dict(warmup_days=3, min_cell_n=5, q_max=1000, hours=(6, 3), slip=0.0, n0=5)


def _obs(markets=("A", "B"), days=20, start=dt.date(2026, 1, 5), seed=0, strikes=10):
    """Toy tail obs: low-tail mids with y drawn above mid, so S finds buy edges."""
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
                    y = float(rng.uniform() < (0.25 if mid < 0.5 else 0.97))
                    rows.append(dict(market=m, group=m, ticker=f"{m}-{d}-{k}", expiry=E, day=d, H=H, y=y,
                                     bid=bid, ask=ask, bid_f=bid, ask_f=ask, floor=100.0 + k, strike_rule="gt",
                                     strike_type="greater", decision_ts=E - pd.Timedelta(hours=H), mid=mid,
                                     vol_prem=float(rng.normal()), vol_missing=0, event_day=int(i % 5 == 0)))
    return pd.DataFrame(rows)


def test_walk_forward_matches_run_estimates():
    obs = _obs()
    cfg = MultiConfig(estimator="S", **CFG)
    tr, _, _ = run(obs, cfg)
    assert len(tr) > 20
    pr = walk_forward_predict(obs, make_estimator("S", cfg), cfg)
    m = tr.merge(pr, on=["day", "ticker", "H"], how="left", validate="one_to_one")
    assert m.p.notna().all()
    np.testing.assert_allclose(m.p.values, m.f.values, rtol=0, atol=1e-15)
    # every live row is recorded: warm markets only, and NaN/low-n estimates kept
    assert pr.day.min() == sorted(obs.day.unique())[3]
    assert {"day", "market", "ticker", "H", "mid", "bid", "ask", "bid_f", "ask_f", "y", "p", "n"} <= set(pr.columns)


def test_block_bootstrap_row_weighted():
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    # A (one row, delta 1) at position 1 and B (9 rows, delta 0) at 6: over the 16 equally likely (start1, start2)
    # pairs the expected row-weighted resample mean is exactly 1/18, and a day-mean bootstrap's would be exactly 0.1
    lab, d = [], []
    for i, day in enumerate(days):
        if i == 1:
            lab += [day]; d += [1.0]
        elif i == 6:
            lab += [day] * 9; d += [0.0] * 9
        else:
            lab += [day]; d += [0.0]
    b = block_bootstrap_delta(np.array(lab), np.array(d))
    assert b.shape == (4000,)
    assert abs(b.mean() - 1 / 18) < 0.01
    assert abs(b.mean() - 0.1) > 0.03


def test_block_bootstrap_short_series():
    lab = np.array([dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3)] * 4)
    b = block_bootstrap_delta(lab, np.arange(12.0))
    assert b.shape == (4000,) and np.isfinite(b).all()
    assert np.allclose(b, np.arange(12.0).mean())       # D < 7: one block of all days = full-sample row-weighted mean


def _losses(gap, noise, seed=5, n_days=60, per_day=20):
    rng = np.random.default_rng(seed)
    day = np.repeat([dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n_days)], per_day)
    base = rng.exponential(0.2, len(day))
    f1 = base
    e2, e3 = rng.normal(0, noise, len(day)), rng.normal(0, noise, len(day))
    f2 = base - gap + (e2 - e2.mean())    # exact mean gaps: F2 best by gap, F3 by gap / 2
    f3 = base - gap / 2 + (e3 - e3.mean())
    return {"F1": f1, "F2": f2, "F3": f3}, day


def test_select_prefers_simpler_within_se():
    ll, day = _losses(gap=0.001, noise=0.05)
    chosen, table = select_set(ll, day)
    best = table.set_index("set").ll.idxmin()
    assert best != "F1"
    row = table.set_index("set").loc["F1"]
    assert row.delta <= row.se
    assert chosen == "F1"


def test_select_takes_better_outside_se():
    ll, day = _losses(gap=0.05, noise=0.01)
    chosen, table = select_set(ll, day)
    t = table.set_index("set")
    assert t.loc["F1"].delta > t.loc["F1"].se and t.loc["F3"].delta > t.loc["F3"].se
    assert chosen == "F2"


def test_select_ignores_post_dev_rows(tmp_path, monkeypatch):
    from v2 import select as sel
    dev = _obs(markets=("BTC", "SPX"), days=40, start=dt.date(2026, 5, 20), seed=3)
    late = _obs(markets=("BTC", "SPX"), days=15, start=dt.date(2026, 6, 29), seed=4)
    late = late[late.day >= dt.date(2026, 7, 1)].copy()
    late["y"] = 1.0 - late.y          # adversarial: would move any fit or score that saw them
    late["ticker"] = late.ticker + "-late"
    both = pd.concat([dev[dev.day <= dt.date(2026, 6, 30)], late], ignore_index=True)
    assert both.day.max() > sel.DEV_END
    p_dev, p_both = tmp_path / "dev.csv", tmp_path / "both.csv"
    both[both.day <= sel.DEV_END].to_csv(p_dev, index=False)
    both.to_csv(p_both, index=False)

    seen = []
    real = sel.walk_forward_predict
    monkeypatch.setattr(sel, "walk_forward_predict", lambda o, e, c: (seen.append(max(o.day)), real(o, e, c))[1])
    cfg = dict(warmup_days=3, min_cell_n=5, n0=5)
    res = []
    for path in (p_dev, p_both):
        obs = sel.load_dev(["BTC", "SPX"], path)
        assert max(obs.day) <= sel.DEV_END
        res.append(sel.run_selection(obs, MultiConfig(**cfg), score_start=dt.date(2026, 5, 20)))
    assert seen and max(seen) <= sel.DEV_END
    (c1, t1, p1, _), (c2, t2, p2, _) = res
    assert c1 == c2 and res[0][3] == res[1][3]
    pd.testing.assert_frame_equal(t1, t2)
    assert max(p2.day) <= sel.DEV_END and len(p1) == len(p2) and p1.common.sum() > 100
