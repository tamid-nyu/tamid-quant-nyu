import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tail_strategy import BINS, CellStats, Config, load_obs, run


def _synth(days=40, model_shift=0.0):
    """Same book as test_tail_strategy._synth, plus model columns. model_shift moves the model's probability."""
    rows = []
    for d in range(days):
        day = dt.date(2026, 6, 1) + dt.timedelta(days=d)
        for H in (6, 3):
            for i in range(21):
                if i <= 3:
                    bid, ask, y = 0.935, 0.945, 1.0
                elif i <= 16:
                    bid, ask, y = 0.49, 0.51, 1.0 if i < 10 else 0.0
                else:
                    bid, ask, y = 0.055, 0.065, 0.0
                mid = (bid + ask) / 2
                p = float(np.clip(mid + model_shift, 0.001, 0.999))
                rows.append(dict(ticker=f"D{d}-T{i}", expiry=f"{day} 21:00:00+00:00", day=day, H=H, y=y, S=65_000.0,
                                 bid=bid, ask=ask, bid_f=bid, ask_f=ask, strike_type="greater",
                                 floor=60_000.0 + 500 * i, mid=mid, p_ewma_fhs=p, p_ewma_normal=p))
    return pd.DataFrame(rows)


def test_default_options_reproduce_base():
    obs = _synth()
    explicit = Config(tail_lo=0.10, hours=(6, 3), z_min=0.0, agree_margin=None, prior="market", quantile=None,
                      estimator="cells", half_life=None, move_skip=None)
    a, da = run(obs, Config())
    b, db = run(obs, explicit)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(da, db)


def test_narrow_tails_drop_outer_band():
    tr, _ = run(_synth(), Config(tail_lo=0.05))          # synthetic tails sit at 6c / 94c
    assert tr.empty


def test_hours_option_limits_rounds():
    tr, _ = run(_synth(), Config(hours=(6,)))
    assert len(tr) > 0 and (tr.H == 6).all()


def test_strike_distance_filter():
    obs = _synth()                                        # model p = 6c / 94c -> |d2| about 1.55
    assert run(obs, Config(z_min=1.5))[0].shape[0] > 0
    assert run(obs, Config(z_min=2.0))[0].empty


def test_model_agreement_filter():
    agree = _synth(model_shift=-0.05)                     # model says low tail ~1c: agrees with selling at 6c
    against = _synth(model_shift=+0.05)                   # model says low tail ~11c, high tail ~99c
    tr_a, _ = run(agree, Config(agree_margin=0.0))
    tr_b, _ = run(against, Config(agree_margin=0.0))
    assert (tr_a.side == -1).any()
    assert not (tr_b.side == -1).any()


def test_model_prior_shrinks_toward_model_probability():
    cs = CellStats(50, BINS)
    cs.update(pd.DataFrame(dict(H=[6] * 100, mid=[0.02] * 100, y=[0.0] * 100)))
    f, n = cs.estimate(6, 0.02, prior=0.005)
    assert n == 100 and f == pytest.approx(50 * 0.005 / 150)


def test_cautious_quantile_is_pessimistic_for_each_side():
    cs = CellStats(50, BINS)
    cs.update(pd.DataFrame(dict(H=[6] * 200, mid=[0.02] * 200, y=[0.0] * 199 + [1.0])))
    f, _ = cs.estimate(6, 0.02)
    assert cs.quantile(6, 0.02, 0.9, side=-1) > f > cs.quantile(6, 0.02, 0.9, side=1)


def test_cautious_sizing_trades_less():
    base, _ = run(_synth(), Config())
    cautious, _ = run(_synth(), Config(quantile=0.95))
    assert cautious.q.sum() < base.q.sum()


def test_half_life_discounts_old_outcomes():
    old = pd.DataFrame(dict(H=[6] * 100, mid=[0.02] * 100, y=[1.0] * 100))
    new = pd.DataFrame(dict(H=[6] * 100, mid=[0.02] * 100, y=[0.0] * 100))
    plain, decayed = CellStats(50, BINS), CellStats(50, BINS, half_life=10)
    for cs in (plain, decayed):
        cs.update(old)
        for _ in range(30):
            cs.update(new.iloc[:5])
    f_plain, n_plain = plain.estimate(6, 0.02)
    f_dec, n_dec = decayed.estimate(6, 0.02)
    assert n_plain == n_dec == 250                       # raw counts drive eligibility
    assert f_dec < f_plain


def test_price_move_skip():
    obs = _synth()
    low = obs.mid < 0.10
    obs.loc[low, "bid_f"] = obs.loc[low, "bid"] + 0.03   # low-tail quote jumps 3c toward the money before the fill
    obs.loc[low, "ask_f"] = obs.loc[low, "ask"] + 0.03
    tr, _ = run(obs, Config(move_skip=0.02))
    assert len(tr) > 0 and not (tr.side == -1).any()


def test_logit_estimator_trades_and_has_no_lookahead():
    obs = _synth(days=60)
    tr1, _ = run(obs, Config(estimator="logit"))
    assert len(tr1) > 0
    cut = sorted(obs.day.unique())[50]
    obs2 = obs.copy()
    later = obs2.day > cut
    obs2.loc[later, "y"] = np.random.default_rng(3).permutation(obs2.loc[later, "y"].values)
    tr2, _ = run(obs2, Config(estimator="logit"))
    a = tr1[tr1.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    b = tr2[tr2.day <= cut].drop(columns=["y", "pnl"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


LONG = Path(__file__).resolve().parent.parent / "output" / "btc_pm_results_long" / "btc_observations.csv"
LONG_TRADES = LONG.parent.parent / "btc_tail_results_long" / "tail_trades.csv"


@pytest.mark.skipif(not LONG.exists(), reason="long-history observations not present")
def test_model0_reproduces_long_history():
    tr, _ = run(load_obs(LONG), Config())
    old = pd.read_csv(LONG_TRADES)
    assert len(tr) == len(old) and tr.pnl.sum() == pytest.approx(old.pnl.sum())


# ---------------------------------------------------------------- variants runner
from variants import MODELS, paired_sharpe_diff_ci, sharpe, sharpe_ci, window_metrics


def test_models_are_numbered_and_model0_is_base():
    assert sorted(MODELS) == list(range(12))
    assert MODELS[0]["overrides"] == {} and not MODELS[0]["grid"]
    assert [k for k, m in MODELS.items() if m.get("post_hoc")] == [10, 11]       # combined after seeing results
    assert MODELS[10]["overrides"]["estimator"] == "logit" and "move_skip" in MODELS[10]["grid"]
    assert MODELS[11]["overrides"]["prior"] == "model" and "move_skip" in MODELS[11]["grid"]
    for k, m in MODELS.items():
        for field in list(m["overrides"]) + list(m["grid"]):
            assert field in {f.name for f in dataclasses.fields(Config)}, (k, field)


def test_sharpe_annualizes_daily_returns():
    r = pd.Series([0.001, -0.0005, 0.002, 0.0, 0.0015])
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(365))
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_sharpe_ci_brackets_estimate():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.005, 200))
    lo, hi = sharpe_ci(r)
    assert lo < sharpe(r) < hi


def test_paired_diff_is_zero_for_identical_series():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.005, 100))
    lo, hi = paired_sharpe_diff_ci(r, r)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_window_metrics_counts_trades_and_tail_days():
    _, dy = run(_synth(), Config())
    tr, _ = run(_synth(), Config())
    m = window_metrics(tr, dy, Config(), dy.day.min(), dy.day.max())
    assert m["trades"] == len(tr) and m["tail_days"] == 0 and m["days"] == len(dy)
