"""
Walk-forward edge estimators for the multi-market tail strategy (spec section 5).

S (separate): one BTC-base CellStats per market; f shrinks toward that market's mean mid in the cell (N0).
P (partial pooling): a pooled CellStats over all markets gives f_pool; each market's cell is shrunk toward it:
    f_m = (sum_y_m + N1 * f_pool) / (n_m + N1)
Both return (f, n) where n is the count that decides eligibility (S: the market's cell, P: the pooled cell).
"""
import tail_strategy as ts


class SeparateEstimator:
    def __init__(self, n0, bins=ts.BINS):
        self.n0, self.bins = n0, bins
        self.by_market = {}

    def _cs(self, market):
        if market not in self.by_market:
            self.by_market[market] = ts.CellStats(self.n0, self.bins)
        return self.by_market[market]

    def update(self, df):
        for m, g in df.groupby("market"):
            self._cs(m).update(g)

    def estimate(self, market, H, mid):
        cs = self.by_market.get(market)
        if cs is None:
            return float("nan"), 0
        return cs.estimate(H, mid)


class PooledEstimator:
    def __init__(self, n0, n1, bins=ts.BINS):
        self.n1 = n1
        self.pool = ts.CellStats(n0, bins)
        self.sep = SeparateEstimator(n0, bins)

    def update(self, df):
        self.pool.update(df)
        self.sep.update(df)

    def estimate(self, market, H, mid):
        f_pool, n_pool = self.pool.estimate(H, mid)
        if n_pool == 0:
            return float("nan"), 0
        cs = self.sep.by_market.get(market)
        n_m, sy_m = 0, 0.0
        if cs is not None:
            c = cs.cells.get((int(H), cs._bin(mid)))
            if c is not None:
                n_m, sy_m = c[0], c[2]
        return (sy_m + self.n1 * f_pool) / (n_m + self.n1), n_pool


def make_estimator(kind, cfg):
    if kind == "S":
        return SeparateEstimator(cfg.n0)
    if kind == "P":
        return PooledEstimator(cfg.n0, cfg.n1)
    raise ValueError(f"estimator must be S or P, got {kind!r}")
