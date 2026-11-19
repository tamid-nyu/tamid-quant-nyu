import datetime as dt

import numpy as np
import pytest
import pandas as pd

import multi_tail as mt
from multi_tail import MultiConfig
from v2.scoring import logloss, walk_forward_predict
from v2 import evaluate as v2_evaluate
from v3 import blend, evaluate
from v3.blend import BlendEstimator
from test_blend import CFG, _obs


def _rows(n_days=60, per_day=30, seed=3, start=dt.date(2026, 11, 12)):
    rng = np.random.default_rng(seed)
    day = np.repeat([start + dt.timedelta(days=i) for i in range(n_days)], per_day)
    p = rng.uniform(0.02, 0.08, len(day))
    y = (rng.uniform(size=len(day)) < p).astype(float)
    return pd.DataFrame({"day": day, "y": y, "p_true": p})


def _fwd_env(tmp_path, monkeypatch):
    _rows(n_days=3).assign(market="BTC").to_csv(tmp_path / "features.csv", index=False)
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(evaluate, "OUT", tmp_path / "out")
    monkeypatch.setattr(v2_evaluate, "FEATURES_PATH", tmp_path / "features.csv")      # data_quality is v2's
    monkeypatch.setattr(v2_evaluate, "RAW_DIR", tmp_path / "raw")


def test_v3_forward_refuses_early(tmp_path, monkeypatch, capsys):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2026, 11, 13))
    monkeypatch.setattr(evaluate, "forward", lambda: (_ for _ in ()).throw(AssertionError("scored early")))
    assert evaluate.main(["--period", "forward", "--today", "2027-06-01"]) == 0
    text = capsys.readouterr().out
    assert "Forward data-quality check" in text and "BTC" in text
    for word in ("Δ", "WIN", "delta", "log-loss", "Sharpe", "not shown"):
        assert word not in text
    assert not (tmp_path / "out" / "forward.txt").exists()


def test_v3_forward_opens_on_real_date(tmp_path, monkeypatch):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "forward", lambda: "FORWARD TEXT\n")
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 11))
    evaluate.main(["--period", "forward"])
    assert not (tmp_path / "out" / "forward.txt").exists()
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 12))
    assert evaluate.main(["--period", "forward"]) == 0
    assert (tmp_path / "out" / "forward.txt").read_text() == "FORWARD TEXT\n"


def test_v3_forward_report_has_multiplicity_note(tmp_path, monkeypatch):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 12))
    monkeypatch.setattr(evaluate, "_evaluate", lambda start, end, header, cfg, with_verdict: "\n".join(header))
    text = evaluate.forward()
    assert ("v2 and v3 verdicts are read on the same forward window, each at its own 5% level, "
            "with no multiplicity correction.") in text


def test_v3_forward_direct_call_guarded(tmp_path, monkeypatch):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 11))
    monkeypatch.setattr(evaluate, "_evaluate", lambda *a: (_ for _ in ()).throw(AssertionError("scored early")))
    with pytest.raises(RuntimeError):
        evaluate.forward()


def _frame(df, p_v3, p_p, p_f1):
    out = df.assign(common=True)
    for name, p in (("V3", p_v3), ("P", p_p), ("F1", p_f1), ("mid", df.p_true + 0.03)):
        out[f"ll_{name}"] = logloss(p, out.y)
    return out


def test_v3_verdict_requires_both():
    df = _rows(n_days=120, per_day=40)
    good, bad = df.p_true, (df.p_true + 0.03).clip(upper=0.999)
    label, tests = evaluate.verdict(_frame(df, good, bad, good))      # beats P, ties F1
    assert label == "not shown" and tests["P"]["pass"] and not tests["F1"]["pass"]
    label, tests = evaluate.verdict(_frame(df, good, good, bad))      # beats F1 only
    assert label == "not shown" and tests["F1"]["pass"] and not tests["P"]["pass"]
    label, tests = evaluate.verdict(_frame(df, good, bad, bad))       # beats both
    assert label == "WIN" and tests["P"]["pass"] and tests["F1"]["pass"]


def test_v3_cli_runs():
    obs = _obs()
    cfg = MultiConfig(estimator="V3", **CFG)
    trades, daily, _ = mt.run(obs, cfg)
    assert len(trades) > 0 and len(daily) > 0
    wf = walk_forward_predict(obs, BlendEstimator(blend.FROZEN["w"], cfg.n0, cfg.n1), cfg)
    f = wf.set_index(["day", "ticker", "H"]).p
    got = f.loc[list(zip(trades.day, trades.ticker, trades.H))].to_numpy()
    np.testing.assert_allclose(trades.f.to_numpy(), got, atol=1e-12)
    assert np.isfinite(got).all()


class _Stop(Exception):
    pass


def test_main_v3_loads_features_and_holds_back(tmp_path, monkeypatch):
    import multi_obs
    obs = _obs(("BTC",), 3, start=dt.date(2026, 11, 11))
    seen = {}

    def fake_run(o, cfg):
        seen["obs"], seen["cfg"] = o, cfg
        raise _Stop

    monkeypatch.setattr(multi_obs, "load_all", lambda keys: (_ for _ in ()).throw(AssertionError("v1 loader")))
    monkeypatch.setattr(mt, "load_features", lambda keys, path=None: seen.setdefault("feat", obs.copy()))
    monkeypatch.setattr(mt, "run", fake_run)
    monkeypatch.setattr(mt, "_real_today", lambda: dt.date(2027, 5, 11))
    monkeypatch.setattr(mt, "V3_OUT_DIR", tmp_path / "v3sim")
    try:
        mt.main(["--markets", "BTC", "--estimator", "V3"])
    except _Stop:
        pass
    assert "feat" in seen and seen["cfg"].estimator == "V3"
    assert max(seen["obs"].day) == dt.date(2026, 11, 11)               # forward hold-back still applies
    assert (tmp_path / "v3sim").is_dir()                               # default out-dir for V3
