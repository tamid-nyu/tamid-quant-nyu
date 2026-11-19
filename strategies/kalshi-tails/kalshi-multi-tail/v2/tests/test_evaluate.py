import datetime as dt

import numpy as np
import pandas as pd

from v2 import evaluate
from v2.scoring import block_bootstrap_delta, logloss


def _rows(n_days=60, per_day=30, seed=3, start=dt.date(2026, 11, 12)):
    rng = np.random.default_rng(seed)
    day = np.repeat([start + dt.timedelta(days=i) for i in range(n_days)], per_day)
    p = rng.uniform(0.02, 0.08, len(day))
    y = (rng.uniform(size=len(day)) < p).astype(float)
    return pd.DataFrame({"day": day, "y": y, "p_true": p})


def _frame(df, p_v2, p_p, p_s):
    out = df.assign(common=True, s_subset=True)
    for name, p in (("V2", p_v2), ("P", p_p), ("S", p_s), ("mid", df.p_true + 0.03)):
        out[f"ll_{name}"] = logloss(p, out.y)
    return out


def test_forward_refuses_early(tmp_path, monkeypatch, capsys):
    feats = _rows(n_days=3).assign(market="BTC")
    feats.to_csv(tmp_path / "features.csv", index=False)
    (tmp_path / "raw" / "SPX").mkdir(parents=True)
    (tmp_path / "raw" / "SPX" / "failed_days.txt").write_text("2026-11-12\n")
    monkeypatch.setattr(evaluate, "OUT", tmp_path / "out")
    monkeypatch.setattr(evaluate, "FEATURES_PATH", tmp_path / "features.csv")
    monkeypatch.setattr(evaluate, "RAW_DIR", tmp_path / "raw")
    assert evaluate.main(["--period", "forward", "--today", "2027-05-11"]) == 0
    text = capsys.readouterr().out
    assert "BTC" in text and "2026-11-12" in text          # data-quality counts and the failed-day file
    for word in ("Δ", "WIN", "delta", "log-loss", "Sharpe", "not shown"):
        assert word not in text
    assert not (tmp_path / "out" / "forward.txt").exists()


def test_verdict_requires_both():
    df = _rows()
    rng = np.random.default_rng(5)
    noisy = lambda: np.clip(df.p_true * np.exp(rng.normal(0, 0.8, len(df))), 0.001, 0.5)
    # v2 = truth, P noisy, S = truth: beats P, does not beat S (delta_S is exactly 0)
    label, tests = evaluate.verdict(_frame(df, df.p_true, noisy(), df.p_true))
    assert tests["P"]["pass"] and not tests["S"]["pass"]
    assert label == "not shown"
    # v2 = truth, P and S noisy: beats both
    label, tests = evaluate.verdict(_frame(df, df.p_true, noisy(), noisy()))
    assert tests["P"]["pass"] and tests["S"]["pass"]
    assert label == "WIN"


def test_mde_formula():
    # per-day mean delta alternates +/- 0.004 around 0.001 over 140 days, 10 rows a day
    days = [dt.date(2026, 7, 1) + dt.timedelta(days=i) for i in range(140)]
    per_day = 0.001 + 0.004 * np.where(np.arange(140) % 2 == 0, 1.0, -1.0)
    day = np.repeat(days, 10)
    d = np.repeat(per_day, 10)
    r = evaluate.mde(day, d)
    se_h = float(np.std(block_bootstrap_delta(day, d), ddof=1))
    assert r["D"] == 140
    assert np.isclose(r["se_h"], se_h)
    assert np.isclose(r["se_181"], se_h * np.sqrt(140 / 181))
    assert np.isclose(r["mde"], 2.5 * se_h * np.sqrt(140 / 181))
    assert np.isclose(r["day_sd"], np.std(per_day, ddof=1))
    # constant delta: no sampling noise, so the MDE is 0
    assert np.isclose(evaluate.mde(day, np.full(len(day), 0.002))["mde"], 0.0)


def _fwd_env(tmp_path, monkeypatch):
    _rows(n_days=3).assign(market="BTC").to_csv(tmp_path / "features.csv", index=False)
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(evaluate, "OUT", tmp_path / "out")
    monkeypatch.setattr(evaluate, "FEATURES_PATH", tmp_path / "features.csv")
    monkeypatch.setattr(evaluate, "RAW_DIR", tmp_path / "raw")


def test_today_cannot_move_forward(tmp_path, monkeypatch, capsys):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2026, 11, 13))
    monkeypatch.setattr(evaluate, "forward", lambda: (_ for _ in ()).throw(AssertionError("scored early")))
    assert evaluate.main(["--period", "forward", "--today", "2027-06-01"]) == 0
    assert "Forward data-quality check" in capsys.readouterr().out
    assert not (tmp_path / "out" / "forward.txt").exists()


def test_forward_opens_on_real_date(tmp_path, monkeypatch, capsys):
    _fwd_env(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluate, "forward", lambda: "FORWARD TEXT\n")
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 11))
    evaluate.main(["--period", "forward"])
    assert not (tmp_path / "out" / "forward.txt").exists()
    monkeypatch.setattr(evaluate, "_real_today", lambda: dt.date(2027, 5, 12))
    assert evaluate.main(["--period", "forward"]) == 0
    assert (tmp_path / "out" / "forward.txt").read_text() == "FORWARD TEXT\n"
    (tmp_path / "out" / "forward.txt").unlink()
    evaluate.main(["--period", "forward", "--today", "2027-05-11"])       # earlier --today still refuses
    assert not (tmp_path / "out" / "forward.txt").exists()
