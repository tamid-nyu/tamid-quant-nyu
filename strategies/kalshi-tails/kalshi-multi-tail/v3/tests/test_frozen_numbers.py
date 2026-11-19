"""Real-data pins of the frozen v1 and v2 holdout numbers (BTC + SPX, 2026-07-01 -> 2026-11-11)."""
import pytest

import multi_tail as mt
from multi_tail import MultiConfig, load_features
from v2 import evaluate as v2e
from v3 import evaluate as v3e
from v2.select import BURN_START

needs_data = pytest.mark.skipif(not v3e.FEATURES_PATH.exists(), reason="features.csv missing")


@pytest.fixture(scope="module")
def prim():
    start, end = v2e.HOLDOUT
    obs = load_features(v2e.PRIMARY, v3e.FEATURES_PATH)
    return obs[(obs.day >= BURN_START) & (obs.day <= end)].reset_index(drop=True)


@needs_data
def test_v1_p_holdout_sharpe(prim):
    start, end = v2e.HOLDOUT
    cfg = MultiConfig()
    days = sorted(prim.day.unique())
    _, daily, _ = mt.run(prim, v3e._cfg("P", cfg))
    w = mt._window(daily, days, start, end)
    assert round(mt.sharpe(w.ret), 2) == 1.24
    assert round(w.ret.sum() * 100, 2) == 5.98
    assert int(w.n_trades.sum()) == 361


@needs_data
def test_v2_f1_holdout_logloss(prim):
    start, end = v2e.HOLDOUT
    df = v2e.score(prim, start, end, MultiConfig())
    assert round(df[df.common].ll_V2.mean(), 5) == 0.08207
