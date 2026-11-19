"""
Pooled logistic bias estimator for the multi-market tail strategy (predictor v2, spec section 4).

    z = logit(mid) + b0 + beta . x_std + alpha[market],   p = sigmoid(z) clipped to [0.001, 0.999]

logit(mid) is an offset with coefficient fixed at 1 and is also the standardized feature "lm". Loss is total log-loss
+ 1/2 lam_beta |beta|^2 + 1/2 lam_alpha |alpha|^2 with b0 unpenalized, minimized by L-BFGS-B. Training rows are the
settled tail candidates passed to update(); a fit uses only rows from days before the predicted row's day and is
redone when refit_days or more calendar days have passed since the last fit.
"""
import numpy as np

CLIP = (0.001, 0.999)
FEATURES = {"F1": ["lm", "h3", "side", "side_lm"]}
FEATURES["F2"] = FEATURES["F1"] + ["vol_prem", "vol_missing"]
FEATURES["F3"] = FEATURES["F2"] + ["event_day"]

# Frozen v2 configuration (spec section 6 freeze; chosen by v2/output/selection.txt). Do not change: the forward
# verdict (spec section 7) is scored with exactly these settings.
FROZEN = {"set": "F1", "lam_beta": 25, "lam_alpha": 25, "refit_days": 7, "min_rows": 200}


def _logit(p):
    p = np.clip(np.asarray(p, float), *CLIP)
    return np.log(p / (1 - p))


def _features(cols, mid, H, vol_prem=None, vol_missing=None, event_day=None):
    """Raw (unstandardized) feature matrix, one column per name in `cols`; vol_prem may hold NaN."""
    mid, H = np.atleast_1d(np.asarray(mid, float)), np.atleast_1d(np.asarray(H, float))
    lm = _logit(mid)
    side = (mid > 0.5).astype(float)
    src = {"lm": lm, "h3": (H == 3).astype(float), "side": side, "side_lm": side * lm,
           "vol_prem": vol_prem, "vol_missing": vol_missing, "event_day": event_day}
    return np.column_stack([np.atleast_1d(np.asarray(src[c], float)) for c in cols]), lm


class LogitV2Estimator:
    def __init__(self, feature_set="F3", lam_beta=FROZEN["lam_beta"], lam_alpha=FROZEN["lam_alpha"],
                 refit_days=FROZEN["refit_days"], min_rows=FROZEN["min_rows"]):
        self.feature_set, self.cols = feature_set, FEATURES[feature_set]
        self.lam_beta, self.lam_alpha, self.refit_days, self.min_rows = lam_beta, lam_alpha, refit_days, min_rows
        self._X, self._lm, self._y, self._mkt, self._day = [], [], [], [], []
        self.last_fit_day, self.n_fit = None, 0
        self.cond, self.cond_log = float("nan"), []
        self.fit_ok = []                      # L-BFGS-B convergence flag per fit (diagnostic only)
        self.warm_started = False
        self._w, self._key = None, None

    def update(self, df):
        df = df[df.y.notna()]
        if not len(df):
            return
        get = lambda c: df[c].values if c in df else None
        X, lm = _features(self.cols, df.mid.values, df.H.values, get("vol_prem"), get("vol_missing"), get("event_day"))
        self._X.append(X)
        self._lm.append(lm)
        self._y.append(df.y.values.astype(float))
        self._mkt.append(df.market.values.astype(str))
        self._day.append(np.array(list(df.day), dtype="datetime64[D]"))

    def estimate(self, market, H, mid):
        raise TypeError("V2 needs estimate_row")

    def _n_before(self, day):
        return int(sum((d < day).sum() for d in self._day)) if self._day else 0

    def _fit(self, day, label):
        from scipy.optimize import minimize
        days = np.concatenate(self._day)
        keep = days < day
        X, lm, y = np.vstack(self._X)[keep], np.concatenate(self._lm)[keep], np.concatenate(self._y)[keep]
        mk = np.concatenate(self._mkt)[keep]
        X = X.copy()
        fill = {}
        if "vol_prem" in self.cols:
            j = self.cols.index("vol_prem")
            mu = np.nanmean(X[:, j]) if np.isfinite(X[:, j]).any() else 0.0
            X[np.isnan(X[:, j]), j] = mu
            fill[j] = mu
        mean, sd = X.mean(axis=0), X.std(axis=0)
        use = sd > 1e-12
        Xs = (X[:, use] - mean[use]) / sd[use]
        markets = sorted(str(m) for m in set(mk))
        Dm = (mk[:, None] == np.array(markets)[None, :]).astype(float)
        A = np.column_stack([np.ones(len(y)), Xs, Dm])
        pen = np.concatenate([[0.0], np.full(Xs.shape[1], self.lam_beta), np.full(len(markets), self.lam_alpha)])

        def loss(w):
            z = lm + A @ w
            p = 1 / (1 + np.exp(-z))
            return (np.logaddexp(0, z) - y * z).sum() + 0.5 * (pen * w * w).sum(), A.T @ (p - y) + pen * w

        cols_fit = [c for c, u in zip(self.cols, use) if u]
        key = (tuple(cols_fit), tuple(markets))
        self.warm_started = self._w is not None and key == self._key
        w0 = self._w if self.warm_started else np.zeros(A.shape[1])
        res = minimize(loss, w0, jac=True, method="L-BFGS-B")
        w = res.x
        self.fit_ok.append(bool(res.success))
        self._w, self._key = w, key
        self.cols_fit, self._use, self._mean, self._sd, self._fill = cols_fit, use, mean, sd, fill
        self.b0 = float(w[0])
        self.beta = dict(zip(cols_fit, map(float, w[1:1 + len(cols_fit)])))
        self.alpha = dict(zip(markets, map(float, w[1 + len(cols_fit):])))
        self.mean = dict(zip(cols_fit, mean[use]))
        self.sd = dict(zip(cols_fit, sd[use]))
        self.cond = float(np.linalg.cond(Xs)) if Xs.shape[1] else 1.0
        self.cond_log.append(self.cond)
        self.n_fit, self.last_fit_day = int(len(y)), label

    def estimate_row(self, r):
        day = np.datetime64(r.day, "D")
        if self.last_fit_day is None:
            if self._n_before(day) >= self.min_rows:
                self._fit(day, r.day)
        elif (day - np.datetime64(self.last_fit_day, "D")).astype(int) >= self.refit_days:
            self._fit(day, r.day)
        if self.last_fit_day is None:
            return float("nan"), 0
        g = lambda c: getattr(r, c, np.nan)
        X, lm = _features(self.cols, r.mid, r.H, g("vol_prem"), g("vol_missing"), g("event_day"))
        x = X[0]
        for j, mu in self._fill.items():
            if np.isnan(x[j]):
                x[j] = mu
        xs = (x[self._use] - self._mean[self._use]) / self._sd[self._use]
        z = lm[0] + self.b0 + float(xs @ self._w[1:1 + len(self.cols_fit)]) + self.alpha.get(str(r.market), 0.0)
        return float(np.clip(1 / (1 + np.exp(-z)), *CLIP)), self.n_fit
