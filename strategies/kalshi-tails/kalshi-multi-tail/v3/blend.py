"""
Blend estimator for predictor v3 (spec section 4): f = w * f_P + (1 - w) * f_F1, clipped to [0.001, 0.999].

P is the frozen v1 PooledEstimator(n0, n1), eligible when its pooled cell n >= 100. F1 is the frozen v2
LogitV2Estimator, eligible when it has a fit (>= 200 prior rows). The blend is eligible only when both are
eligible and finite; otherwise (NaN, 0). There is no fallback. When eligible, n = 1 and min_rows = 1.
"""
import numpy as np

from estimators import PooledEstimator
from v2.logit_v2 import FROZEN as F1, LogitV2Estimator

CLIP = (0.001, 0.999)
P_MIN_N = 100
# Set by predictor v3 freeze (Task 3). None means not frozen.
FROZEN = {"w": 0.5}


class BlendEstimator:
    min_rows = 1

    def __init__(self, w, n0=50, n1=100):
        self.w = float(w)
        self.p_est = PooledEstimator(n0, n1)
        self.v2_est = LogitV2Estimator(F1["set"], F1["lam_beta"], F1["lam_alpha"], F1["refit_days"], F1["min_rows"])

    def update(self, df):
        self.p_est.update(df)
        self.v2_est.update(df)

    def estimate(self, market, H, mid):
        raise TypeError("V3 needs estimate_row")

    def estimate_row(self, r):
        fp, np_ = self.p_est.estimate(r.market, r.H, r.mid)
        ff, nf = self.v2_est.estimate_row(r)
        if np_ < P_MIN_N or nf < self.v2_est.min_rows or not (np.isfinite(fp) and np.isfinite(ff)):
            return float("nan"), 0
        return float(np.clip(self.w * fp + (1 - self.w) * ff, *CLIP)), 1
