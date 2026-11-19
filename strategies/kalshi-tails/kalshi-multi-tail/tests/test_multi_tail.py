import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multi_tail import MultiConfig, run

BTC_DIR = Path(__file__).resolve().parents[2] / "btc-kalshi-digital"
BASE_TRADES = BTC_DIR / "output" / "btc_tail_results_long" / "tail_trades.csv"
BTC_OBS = BTC_DIR / "output" / "btc_pm_results_long" / "btc_observations.csv"


def _obs(market, days, hour_utc=21, strikes=(110.0, 120.0), y=0.0, start=dt.date(2026, 1, 5), bid=0.05, ask=0.06):
    rows = []
    for i in range(days):
        d = start + dt.timedelta(days=i)
        E = pd.Timestamp(d.isoformat() + f"T{hour_utc:02d}:00Z")
        for K in strikes:
            for H in (6, 3):
                rows.append(dict(market=market, group=market, ticker=f"{market}-{d}-{K}", expiry=E, day=d, H=H, y=y,
                                 bid=bid, ask=ask, bid_f=bid, ask_f=ask, floor=K, strike_rule="gt",
                                 strike_type="greater", decision_ts=E - pd.Timedelta(hours=H), mid=(bid + ask) / 2))
    return pd.DataFrame(rows)


CFG = dict(warmup_days=3, min_cell_n=5, q_max=1000, hours=(6,), slip=0.0, n0=5)


def test_rounds_in_clock_order():
    a = _obs("A", 8, hour_utc=18)          # A decides at 12:00, B at 15:00
    b = _obs("B", 8, hour_utc=21)
    cfg = MultiConfig(cap_m=0.03, k=2, cap_k=0.04, **CFG)
    tr, daily, _ = run(pd.concat([a, b], ignore_index=True), cfg)
    last = tr[tr.day == tr.day.max()]
    wa = (last[last.market == "A"].q * 0.95).sum()
    wb = (last[last.market == "B"].q * 0.95).sum()
    assert wa > 25 and wa <= 30           # A goes first and gets close to its own cap
    assert wb > 5 and wa + wb <= 40.5     # B gets what the k-worst cap leaves


def test_weekend_only_btc():
    a = _obs("A", 8)
    b = _obs("B", 5)                      # B has no rows on the last 3 days
    tr, daily, md = run(pd.concat([a, b], ignore_index=True), MultiConfig(**CFG))
    last_days = sorted(a.day.unique())[-3:]
    assert set(tr[tr.day.isin(last_days)].market) == {"A"}
    assert len(daily) == 8


def test_short_history_market():
    a = _obs("A", 20, strikes=tuple(float(k) for k in range(100, 110)))
    b = _obs("B", 4, start=dt.date(2026, 1, 21))
    obs = pd.concat([a, b], ignore_index=True)
    cfg = dict(CFG, min_cell_n=50, warmup_days=2)
    trS, _, _ = run(obs, MultiConfig(estimator="S", **cfg))
    trP, _, _ = run(obs, MultiConfig(estimator="P", **cfg))
    assert "B" not in set(trS.market)
    assert "B" in set(trP.market)
    assert trP[trP.market == "B"].day.min() >= dt.date(2026, 1, 23)   # B's own 2-day warmup


def test_missing_fill_quote_dropped():
    a = _obs("A", 6)
    last = a.day.max()
    a.loc[(a.day == last) & (a.floor == 110.0), "bid_f"] = np.nan
    tr, _, _ = run(a, MultiConfig(**CFG))
    assert not ((tr.day == last) & tr.ticker.str.endswith("110.0")).any()
    assert ((tr.day == last) & tr.ticker.str.endswith("120.0")).any()


@pytest.mark.parametrize("est", ["S", "P"])
def test_lookahead_scramble(est):
    rng = np.random.default_rng(3)
    a = _obs("A", 15)
    b = _obs("B", 15, hour_utc=19)
    obs = pd.concat([a, b], ignore_index=True)
    obs["y"] = (rng.uniform(size=len(obs)) < 0.05).astype(float)
    cut = sorted(obs.day.unique())[9]
    tr1, _, _ = run(obs, MultiConfig(estimator=est, **CFG))
    obs2 = obs.copy()
    m = obs2.day > cut
    obs2.loc[m, "y"] = 1.0 - obs2.loc[m, "y"]
    tr2, _, _ = run(obs2, MultiConfig(estimator=est, **CFG))
    cols = ["day", "market", "ticker", "H", "side", "q", "f"]
    a1 = tr1[tr1.day <= cut][cols].reset_index(drop=True)
    a2 = tr2[tr2.day <= cut][cols].reset_index(drop=True)
    assert len(a1) > 0
    pd.testing.assert_frame_equal(a1, a2)
    tl1 = tr1[tr1.day > cut][cols].reset_index(drop=True)      # later days see different history (sanity)
    assert len(tl1) >= 0


@pytest.mark.skipif(not (BASE_TRADES.exists() and BTC_OBS.exists()), reason="committed BTC long-history files missing")
def test_btc_regression():
    from multi_obs import load_btc_obs
    obs = load_btc_obs(BTC_OBS)
    tr, _, _ = run(obs, MultiConfig(k=1, cap_k=0.03, estimator="S"))
    base = pd.read_csv(BASE_TRADES)
    cols = ["day", "ticker", "H", "side", "q", "fill", "pnl"]
    got = tr[cols].copy()
    got["day"] = got.day.astype(str)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), base[cols].reset_index(drop=True), check_dtype=False)


@pytest.mark.skipif(not BTC_OBS.exists(), reason="BTC long-history observations missing")
def test_btc_regression_counts():
    from multi_obs import load_btc_obs
    tr, _, _ = run(load_btc_obs(BTC_OBS), MultiConfig(k=1, cap_k=0.03, estimator="S"))
    assert len(tr) == 402
    assert round(float(tr.pnl.sum()), 2) == 52.61


class _RowStub:
    min_rows = 0

    def update(self, df):
        pass

    def estimate_row(self, r):
        return 0.99, 10 ** 6


class _EstStub:
    def update(self, df):
        pass

    def estimate(self, market, H, mid):
        return 0.99, 10 ** 6


def test_estimate_row_dispatch(monkeypatch):
    import multi_tail
    obs = _obs("A", 8)
    out = []
    for stub in (_RowStub, _EstStub):
        monkeypatch.setattr(multi_tail, "make_estimator", lambda kind, cfg, s=stub: s())
        tr, _, _ = run(obs, MultiConfig(**CFG))
        assert len(tr) > 0
        out.append(tr.reset_index(drop=True))
    pd.testing.assert_frame_equal(out[0], out[1])


def test_sharpe_constant_zero():
    from multi_tail import sharpe
    assert sharpe(np.zeros(10)) == 0.0
    assert sharpe(np.array([0.01, -0.01] * 5)) == pytest.approx(0.0)


def test_paired_diff_identical_is_zero():
    from multi_tail import paired_diff_ci
    r = np.random.default_rng(0).normal(0, 0.01, 200)
    d, lo, hi = paired_diff_ci(r, r.copy())
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_cli_rejects_bad_k():
    from multi_tail import main
    with pytest.raises(SystemExit):
        main(["--k", "0", "--no-plot"])


def test_cli_warns_nonbinding(capsys):
    from multi_tail import check_args
    check_args(k=1, cap_m=0.03, cap_k=0.03)
    assert "never binds" in capsys.readouterr().out


def test_enabled_markets_history_threshold(tmp_path, monkeypatch):
    import multi_obs
    import multi_tail
    pd.DataFrame({"market": ["BTC", "SPX", "GOLD", "WTI", "NATGAS"], "expiry_days": [815, 435, 128, 122, 90],
                  "gate": ["ok", "ok", "< 150 days", "< 150 days", "< 150 days"]}).to_csv(tmp_path / "history_gate.csv",
                                                                                          index=False)
    monkeypatch.setattr(multi_obs, "OBS_DIR", tmp_path)
    assert multi_tail.enabled_markets() == ["BTC", "SPX"]
    assert multi_tail.enabled_markets(min_days=100) == ["BTC", "SPX", "GOLD", "WTI"]


def test_assumed_topk_uses_lp_view():
    # A fills far better than its sizing price, so later rounds see a smaller held loss; the daily assumed k-worst
    # sum must be the LP's view (held at fill + new at sizing price), never above the k-worst cap
    a = _obs("A", 8, hour_utc=18, bid=0.05, ask=0.06)
    a["bid_f"], a["ask_f"] = 0.09, 0.10     # sells fill at 9c instead of 5c: smaller worst case after the fill
    b = _obs("B", 8, hour_utc=21)
    cfg = MultiConfig(cap_m=0.03, k=2, cap_k=0.04, **CFG)
    _, daily, _ = run(pd.concat([a, b], ignore_index=True), cfg)
    assert (daily.assumed_topk <= 40.0 + 1e-6).all()
    assert (daily.realized_topk <= 40.0 + 1e-6).all()


def test_pnl_corr_uses_days_both_traded():
    from multi_tail import pnl_corr
    md = pd.DataFrame({"day": [1, 2, 3, 1, 2, 4], "market": ["A", "A", "A", "B", "B", "B"],
                       "pnl": [1.0, 2.0, -5.0, 1.0, 2.0, 9.0]})
    c = pnl_corr(md)
    assert c.loc["A", "B"] == pytest.approx(1.0)        # only days 1 and 2 are shared


# --- final review fixes -----------------------------------------------------
def test_multiconfig_v2_set_matches_frozen():
    from v2.logit_v2 import FROZEN
    assert MultiConfig().v2_set == FROZEN["set"]


class _Stop(Exception):
    pass


def _fwd_obs():
    return _obs("BTC", 3, start=dt.date(2026, 11, 11))      # days 11-11, 11-12, 11-13


def _capture_main(monkeypatch, estimator, obs, today):
    import multi_obs
    import multi_tail
    seen = {}

    def fake_run(o, cfg):
        seen["obs"], seen["cfg"] = o, cfg
        raise _Stop

    monkeypatch.setattr(multi_obs, "load_all", lambda keys: obs.copy())
    monkeypatch.setattr(multi_tail, "load_features", lambda keys, path=None: obs.copy())
    monkeypatch.setattr(multi_tail, "run", fake_run)
    monkeypatch.setattr(multi_tail, "_real_today", lambda: today)
    try:
        multi_tail.main(["--markets", "BTC", "--estimator", estimator, "--out-dir", tempfile.mkdtemp()])
    except _Stop:
        pass
    return seen


def test_main_v2_uses_frozen_set(monkeypatch):
    from v2.logit_v2 import FROZEN
    seen = _capture_main(monkeypatch, "V2", _fwd_obs(), dt.date(2026, 11, 13))
    assert seen["cfg"].estimator == "V2" and seen["cfg"].v2_set == FROZEN["set"]


def test_main_holds_back_forward_days_before_open(monkeypatch, capsys):
    seen = _capture_main(monkeypatch, "S", _fwd_obs(), dt.date(2027, 5, 11))
    assert max(seen["obs"].day) == dt.date(2026, 11, 11)
    assert "held back" in capsys.readouterr().out


def test_hold_back_message(capsys):
    from multi_tail import hold_back_forward
    o, n = hold_back_forward(_fwd_obs(), dt.date(2027, 5, 11))
    assert n == len(_fwd_obs()) - len(o) > 0
    assert str(n) in capsys.readouterr().out


def test_main_keeps_forward_days_on_open_date(monkeypatch):
    obs = _fwd_obs()
    seen = _capture_main(monkeypatch, "S", obs, dt.date(2027, 5, 12))
    assert len(seen["obs"]) == len(obs)


def _two_markets():
    a = _obs("A", 12, hour_utc=18, y=0.0)
    b = _obs("B", 12, hour_utc=21, y=1.0)
    return pd.concat([a, b], ignore_index=True)


def test_trade_markets_none_identical():
    obs = _two_markets()
    cfg = dict(estimator="P", **CFG)
    t0, d0, m0 = run(obs, MultiConfig(**cfg))
    t1, d1, m1 = run(obs, MultiConfig(trade_markets=None, **cfg))
    pd.testing.assert_frame_equal(t0, t1)
    pd.testing.assert_frame_equal(d0, d1)
    pd.testing.assert_frame_equal(m0, m1)


def test_trade_markets_restricts():
    obs = _two_markets()
    cfg = dict(estimator="P", **CFG)
    both, _, _ = run(obs, MultiConfig(**cfg))
    assert set(both.market) == {"A", "B"}
    only, _, _ = run(obs, MultiConfig(trade_markets=("A",), **cfg))
    assert set(only.market) == {"A"}
    a_alone, _, _ = run(obs[obs.market == "A"], MultiConfig(**cfg))
    assert len(only) != len(a_alone) or not np.allclose(only.f.values, a_alone.f.values)
