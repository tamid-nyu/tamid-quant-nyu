import numpy as np
import pandas as pd
import pytest

import tail_strategy as ts
from estimators import PooledEstimator, SeparateEstimator


def _rows(market, n, seed, p_yes=0.02):
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.005, 0.099, n)
    return pd.DataFrame({"market": market, "H": rng.choice([6, 3], n), "mid": mid,
                         "y": (rng.uniform(size=n) < p_yes).astype(float)})


def test_S_matches_cellstats():
    df = _rows("BTC", 500, 1)
    cs = ts.CellStats(50, ts.BINS)
    cs.update(df)
    est = SeparateEstimator(50, ts.BINS)
    est.update(df)
    for H, mid in [(6, 0.02), (3, 0.04), (6, 0.08), (3, 0.003)]:
        np.testing.assert_equal(est.estimate("BTC", H, mid), cs.estimate(H, mid))


def test_S_markets_independent():
    est = SeparateEstimator(50, ts.BINS)
    est.update(_rows("A", 300, 2))
    before = est.estimate("B", 6, 0.02)
    est.update(_rows("A", 300, 3))
    np.testing.assert_equal(est.estimate("B", 6, 0.02), before)
    assert before[1] == 0 and np.isnan(before[0])


def test_P_no_market_data_equals_pool():
    est = PooledEstimator(50, 100, ts.BINS)
    est.update(_rows("A", 400, 4))
    pool = ts.CellStats(50, ts.BINS)
    pool.update(_rows("A", 400, 4))
    f, n = est.estimate("B", 6, 0.02)
    fp, npool = pool.estimate(6, 0.02)
    assert f == pytest.approx(fp) and n == npool


def test_P_large_n1_tends_to_pool():
    est = PooledEstimator(50, 1e9, ts.BINS)
    est.update(pd.concat([_rows("A", 400, 5), _rows("B", 200, 6, p_yes=0.2)]))
    f, _ = est.estimate("B", 6, 0.02)
    fp, _ = est.pool.estimate(6, 0.02)
    assert abs(f - fp) < 1e-6


def test_P_formula():
    # pool: 200 obs at mid 0.02 with 4 YES (market A 150 obs / 0 YES... built explicitly), market B: 50 obs, 5 YES
    a = pd.DataFrame({"market": "A", "H": 6, "mid": 0.02, "y": [1.0] * 0 + [0.0] * 150})
    b = pd.DataFrame({"market": "B", "H": 6, "mid": 0.02, "y": [1.0] * 5 + [0.0] * 45})
    est = PooledEstimator(50, 100, ts.BINS)
    est.update(pd.concat([a, b]))
    f_pool = (5 + 50 * 0.02) / (200 + 50)
    f, n = est.estimate("B", 6, 0.02)
    assert n == 200
    assert f == pytest.approx((5 + 100 * f_pool) / (50 + 100))
