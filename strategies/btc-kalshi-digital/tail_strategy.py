#!/usr/bin/env python3
"""
BTC Kalshi tail strategy with portfolio worst-case sizing
=========================================================
Trades only tail contracts (mid < 10% or > 90%) 6h and 3h before the daily 5pm ET expiry, and sizes each expiry
day's book so that the worst possible BTC settlement loses at most CAP_FRAC of bankroll (see tail_sizer.py).

Edge is estimated walk-forward: for each (horizon, price bin) cell, the realized YES rate on prior expiry days,
shrunk toward the market price. Reads output/btc_pm_results/btc_observations.csv written by
btc_digital_model_backtest.py.

Spec: docs/superpowers/specs/2026-11-11-btc-tail-strategy-design.md
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tail_sizer import fee, order_fee, payoff_matrix, size_per_trade, size_portfolio

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "btc_pm_results" / "btc_observations.csv"
OUT_DIR = HERE / "output" / "btc_tail_results"

BANKROLL = 1000.0
CAP_FRAC = 0.03
Q_MAX = 50                 # placeholder per ticker per round: the data has no order-book depth
MIN_EDGE = 0.005
SLIP = 0.02                # conservative buffer on decision-time quotes, since fills happen one hour later
N0 = 50                    # prior strength (observations) shrinking each cell toward the market price
MIN_CELL_N = 100
WARMUP_DAYS = 30
HOURS = (6, 3)
MAX_SPREAD = 0.06
FEE_RATE = 0.07            # ASSUMPTION copied from the main backtest; check Kalshi's fee schedule
TAIL_LO, TAIL_HI = 0.10, 0.90
BINS = [0, .005, .015, .025, .035, .05, .075, .10, .90, .925, .95, .965, .975, .985, .995, 1.0001]


@dataclass
class Config:
    bankroll: float = BANKROLL
    cap_frac: float = CAP_FRAC
    q_max: int = Q_MAX
    min_edge: float = MIN_EDGE
    slip: float = SLIP
    n0: float = N0
    min_cell_n: int = MIN_CELL_N
    warmup_days: int = WARMUP_DAYS
    fee_rate: float = FEE_RATE
    # model-variant options (spec: docs/superpowers/specs/2026-11-11-btc-tail-variants-design.md); every default
    # reproduces the base strategy
    tail_lo: float = TAIL_LO           # trade only mid < tail_lo or > 1 - tail_lo (the estimator still trains on 10c)
    hours: tuple = HOURS               # decision rounds
    z_min: float = 0.0                 # min |d2| implied by dist_col (strike distance in forecast sd)
    agree_margin: float = None         # if set, model_col's net edge on the chosen side must be >= this
    prior: str = "market"              # "market": shrink toward the bin's mean mid; "model": toward model_col
    quantile: float = None             # if set, size on this pessimistic quantile of the Beta posterior of f
    estimator: str = "cells"           # "cells" (per-bin rates) or "logit" (walk-forward logistic regression)
    half_life: float = None            # if set, cell outcomes decay with this half-life in expiry days
    move_skip: float = None            # if set, skip when the fill-hour mid moved more than this toward the money
    model_col: str = "p_ewma_fhs"
    dist_col: str = "p_ewma_normal"

    @property
    def cap(self) -> float:
        return self.bankroll * self.cap_frac


def load_obs(path):
    o = pd.read_csv(path)
    o = o[o.H.isin(HOURS) & (o.strike_type == "greater")].copy()
    o["day"] = pd.to_datetime(o.day).dt.date
    o["mid"] = (o.bid + o.ask) / 2
    o = o[o.y.notna()]           # an unsettled row would poison its estimator cell (NaN outcome) for the whole run
    return o.reset_index(drop=True)


def tail_candidates(obs):
    m = (((obs.mid < TAIL_LO) | (obs.mid > TAIL_HI)) & (obs.bid >= 0.01) & (obs.ask <= 0.99)
         & ((obs.ask - obs.bid) <= MAX_SPREAD + 1e-12))
    return obs[m]


class CellStats:
    """Per-(horizon, price bin) running counts of prior observations: raw n, and weighted count, outcomes and mids.

    Without a half-life the weights are all 1. With one, each update() call (one expiry day) first decays the
    weighted sums by 2^(-1/half_life). Eligibility always uses the raw count.
    """

    def __init__(self, n0, bins, half_life=None):
        self.n0 = n0
        self.bins = np.asarray(bins, float)
        self.decay = None if half_life is None else 2.0 ** (-1.0 / half_life)
        self.cells = {}

    def _bin(self, mid):
        return int(np.searchsorted(self.bins, mid, side="right") - 1)

    def update(self, df):
        if self.decay is not None:
            for c in self.cells.values():
                c[1] *= self.decay
                c[2] *= self.decay
                c[3] *= self.decay
        for H, mid, y in zip(df.H, df.mid, df.y):
            c = self.cells.setdefault((int(H), self._bin(mid)), [0, 0.0, 0.0, 0.0])
            c[0] += 1
            c[1] += 1.0
            c[2] += y
            c[3] += mid

    def _posterior(self, H, mid, prior):
        """(raw n, weighted n, Beta alpha, Beta beta) of the shrunk YES rate."""
        n, w, sy, sm = self.cells.get((int(H), self._bin(mid)), (0, 0.0, 0.0, 0.0))
        if n == 0:
            return n, 0.0, float("nan"), float("nan")
        m = sm / w if prior is None else prior
        return n, w, sy + self.n0 * m, (w - sy) + self.n0 * (1 - m)

    def estimate(self, H, mid, prior=None):
        """Shrunk YES rate and raw count. prior=None shrinks toward the bin's mean mid, else toward `prior`."""
        n, w, a, _ = self._posterior(H, mid, prior)
        if n == 0:
            return float("nan"), 0
        return a / (w + self.n0), n             # = (sum y + n0*m) / (n + n0), same arithmetic as the base

    def quantile(self, H, mid, q, side, prior=None):
        """Pessimistic Beta-posterior quantile of the YES rate for `side`: upper for a seller, lower for a buyer."""
        from scipy.stats import beta
        n, _, a, b = self._posterior(H, mid, prior)
        if n == 0:
            return float("nan")
        return float(beta.ppf(q if side == -1 else 1 - q, a, b))


class LogitModel:
    """Walk-forward logistic regression of y on [1, logit(mid), logit(model p), 1[H == 3]], L2-penalized.

    update() stores one expiry day's training rows; refit happens in maybe_fit() at most every `refit_days` days.
    """

    def __init__(self, model_col, refit_days=7, l2=1.0, min_rows=200):
        self.model_col, self.refit_days, self.l2, self.min_rows = model_col, refit_days, l2, min_rows
        self.X, self.y = [], []
        self.w = None
        self.last_fit = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 0.001, 0.999)
        return np.log(p / (1 - p))

    def _features(self, mid, pm, H):
        mid, pm, H = np.atleast_1d(mid), np.atleast_1d(pm), np.atleast_1d(H)
        return np.column_stack([np.ones(len(mid)), self._logit(mid), self._logit(pm), (H == 3).astype(float)])

    def update(self, df):
        if len(df):
            self.X.append(self._features(df.mid.values, df[self.model_col].values, df.H.values))
            self.y.append(df.y.values.astype(float))

    def maybe_fit(self, day_index):
        if self.last_fit is not None and day_index - self.last_fit < self.refit_days:
            return
        if not self.X or sum(len(v) for v in self.y) < self.min_rows:
            return
        from scipy.optimize import minimize
        X, y = np.vstack(self.X), np.concatenate(self.y)
        pen = np.array([0.0, 1.0, 1.0, 1.0]) * self.l2

        def loss(w):
            z = X @ w
            ll = np.logaddexp(0, z) - y * z
            p = 1 / (1 + np.exp(-z))
            return ll.sum() + 0.5 * (pen * w * w).sum(), X.T @ (p - y) + pen * w

        w0 = self.w if self.w is not None else np.zeros(X.shape[1])
        self.w = minimize(loss, w0, jac=True, method="L-BFGS-B").x
        self.last_fit = day_index

    def predict(self, mid, pm, H):
        if self.w is None:
            return float("nan")
        z = float((self._features(mid, pm, H) @ self.w)[0])
        return float(np.clip(1 / (1 + np.exp(-z)), 0.001, 0.999))


def best_side(bid, ask, f, cfg):
    """(side, sizing price, net edge per contract) for the better of buying or selling YES."""
    buy_price = min(ask + cfg.slip, 0.99)
    sell_price = max(bid - cfg.slip, 0.01)
    buy = f - buy_price - fee(buy_price, cfg.fee_rate)
    sell = sell_price - f - fee(sell_price, cfg.fee_rate)
    return (1, buy_price, buy) if buy >= sell else (-1, sell_price, sell)


SIZINGS = ("portfolio", "one", "per_trade")
TRADE_COLS = ["day", "ticker", "H", "side", "q", "size_price", "fill", "fee", "edge", "f", "y", "pnl"]
DAILY_COLS = ["day", "pnl", "ret", "assumed_worst", "realized_worst", "breach", "deployed", "n_trades"]


def _book_payoffs(grid, pos, price_col, fee_rate):
    if not pos:
        return np.zeros((len(grid) + 1, 0))
    return payoff_matrix(grid, np.array([p["floor"] for p in pos]), np.array([p["side"] for p in pos]),
                         np.array([p[price_col] for p in pos]), fee_rate)


def _held_vector(grid, held, price_col, fee_rate):
    """Book payoff per scenario at the given prices, net of each order's fee rounded up to the cent."""
    if not held:
        return np.zeros(len(grid) + 1)
    gross = _book_payoffs(grid, held, price_col, 0.0) @ np.array([p["q"] for p in held], float)
    return gross - sum(order_fee(p["q"], p[price_col], fee_rate) for p in held)


def run(obs, cfg, sizing="portfolio"):
    """Walk forward over expiry days. Returns (trades, daily); see the plan's Task 4 for the column meanings."""
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}")
    cs = CellStats(cfg.n0, BINS, half_life=cfg.half_life)
    lm = LogitModel(cfg.model_col) if cfg.estimator == "logit" else None
    trades, daily = [], []
    days_seen = 0
    for day, g in obs.groupby("day", sort=True):
        held = []
        assumed = 0.0              # sizer's view at the last round that placed positions
        tails = tail_candidates(g)
        if lm is not None:
            lm.maybe_fit(days_seen)
        if days_seen >= cfg.warmup_days:
            for H in cfg.hours:
                cands = []
                for r in tails[tails.H == H].itertuples(index=False):
                    if not (r.mid < cfg.tail_lo or r.mid > 1 - cfg.tail_lo):
                        continue
                    f, n = cs.estimate(H, r.mid)
                    if n < cfg.min_cell_n:
                        continue
                    pm = getattr(r, cfg.model_col) if (cfg.prior == "model" or cfg.agree_margin is not None
                                                       or lm is not None) else None
                    if cfg.z_min > 0:
                        from scipy.stats import norm
                        pd2 = min(max(float(getattr(r, cfg.dist_col)), 1e-9), 1 - 1e-9)
                        if abs(norm.ppf(pd2)) < cfg.z_min:
                            continue
                    prior = pm if cfg.prior == "model" else None
                    if prior is not None:
                        f, _ = cs.estimate(H, r.mid, prior=prior)
                    if lm is not None:
                        f = lm.predict(r.mid, pm, H)
                    if not np.isfinite(f):
                        continue
                    side, price, edge = best_side(r.bid, r.ask, f, cfg)
                    if cfg.quantile is not None:
                        f = cs.quantile(H, r.mid, cfg.quantile, side, prior=prior)
                        edge = side * (f - price) - fee(price, cfg.fee_rate)
                    if cfg.agree_margin is not None and side * (pm - price) - fee(price, cfg.fee_rate) < cfg.agree_margin:
                        continue
                    fill = r.ask_f if side == 1 else r.bid_f
                    # a leg whose fill-hour quote is unusable is never placed; leaving it out BEFORE sizing keeps
                    # the LP from counting on a hedge that cannot exist
                    if (not np.isfinite(fill)) or (fill >= 1.0 if side == 1 else fill <= 0.0):
                        continue
                    if cfg.move_skip is not None:
                        moved = ((r.bid_f + r.ask_f) / 2 - r.mid) * (-side)   # > 0: toward the money for this side
                        if moved > cfg.move_skip:
                            continue
                    if edge >= cfg.min_edge:
                        cands.append(dict(day=day, ticker=r.ticker, H=H, side=side, size_price=price, edge=edge, f=f,
                                          y=r.y, floor=r.floor, bid_f=r.bid_f, ask_f=r.ask_f))
                if not cands:
                    continue
                grid = np.unique([p["floor"] for p in held + cands])
                base = _held_vector(grid, held, "fill", cfg.fee_rate)
                P = _book_payoffs(grid, cands, "size_price", cfg.fee_rate)
                if sizing == "portfolio":
                    # the LP uses the linear per-contract fee; rounding up adds at most 1c per order, so reserve
                    # 1c of cap per candidate order to keep the worst case within the cap after rounding
                    q = size_portfolio(np.array([c["edge"] for c in cands]), P, base, cfg.cap - 0.01 * len(cands),
                                       cfg.q_max)
                elif sizing == "one":
                    q = np.ones(len(cands), dtype=int)
                else:
                    q = size_per_trade(P, cfg.cap, cfg.q_max)
                placed = []
                for c, qi in zip(cands, q):
                    fill = c["ask_f"] if c["side"] == 1 else c["bid_f"]
                    if qi <= 0:
                        continue
                    fe = order_fee(int(qi), fill, cfg.fee_rate)      # whole order's fee, in dollars
                    placed.append(dict(c, q=int(qi), fill=float(fill), fee=fe,
                                       pnl=qi * c["side"] * (c["y"] - fill) - fe))
                if placed:
                    assumed = float((base + _held_vector(grid, placed, "size_price", cfg.fee_rate)).min())
                    held.extend(placed)
        if held:
            grid = np.unique([p["floor"] for p in held])
            realized = float(_held_vector(grid, held, "fill", cfg.fee_rate).min())
        else:
            realized = 0.0
        trades.extend(held)
        pnl = float(sum(p["pnl"] for p in held))
        daily.append(dict(day=day, pnl=pnl, ret=pnl / cfg.bankroll, assumed_worst=assumed, realized_worst=realized,
                          breach=realized < -cfg.cap - 1e-9,
                          deployed=float(sum(p["q"] * (p["fill"] if p["side"] == 1 else 1 - p["fill"]) + p["fee"]
                                             for p in held)),
                          n_trades=len(held)))
        cs.update(tails)
        if lm is not None:
            lm.update(tails)
        days_seen += 1
    tr = pd.DataFrame(trades, columns=TRADE_COLS) if trades else pd.DataFrame(columns=TRADE_COLS)
    return tr[TRADE_COLS].reset_index(drop=True), pd.DataFrame(daily, columns=DAILY_COLS)


# ----------------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------------
def day_ci(ret, n=4000, seed=11):
    """Bootstrap over expiry days of the mean of a daily series: (2.5%, 97.5%)."""
    v = np.asarray(ret, float)
    if len(v) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(m, [2.5, 97.5]))


def summarize(trades, daily, cfg, label, start=None):
    """Report from `start` (default: this run's first traded day). Pass one common start to compare runs."""
    if trades.empty:
        return f"{label}: no trades"
    start = trades.day.min() if start is None else start
    live = daily[daily.day >= start].reset_index(drop=True)
    cum = live.ret.cumsum()
    dd = (cum - np.maximum.accumulate(np.maximum(cum, 0.0))).min()
    lo, hi = day_ci(live.ret)
    dlo, dhi = day_ci(live.deployed)
    worst_traded = live.ret[live.n_trades > 0].min()
    half = len(live) // 2
    pct = lambda x: f"{x * 100:+.3f}%"
    lines = [f"{label}: {len(trades)} trades over {int((live.n_trades > 0).sum())} of {len(live)} days "
             f"(cap {cfg.cap_frac:.1%} of ${cfg.bankroll:,.0f} = ${cfg.cap:,.2f})"]
    for (side, H), g in trades.groupby(["side", "H"]):
        lines.append(f"  {'buy YES ' if side == 1 else 'sell YES'} {H}h: n={len(g):>4}  contracts={int(g.q.sum()):>6}  "
                     f"pnl=${g.pnl.sum():+,.2f}  hit={g.y.mean():.3f}")
    lines += [
        f"  mean daily return {pct(live.ret.mean())} [95% CI {pct(lo)}, {pct(hi)}], sd {live.ret.std() * 100:.3f}%",
        f"  cumulative return {pct(cum.iloc[-1])}, max drawdown {dd * 100:.3f}%, worst traded day {pct(worst_traded)}",
        f"  max assumed worst case {-live.assumed_worst.min() / cfg.bankroll:.3%} of bankroll, "
        f"max realized worst case {-live.realized_worst.min() / cfg.bankroll:.3%}, "
        f"breaches (realized worst past cap) {int(live.breach.sum())}",
        f"  mean capital deployed incl. fees ${live.deployed.mean():,.2f} per day [95% CI ${dlo:,.2f}, ${dhi:,.2f}]",
        f"  {int(((trades.size_price <= 0.01 + 1e-9) | (trades.size_price >= 0.99 - 1e-9)).sum())} of {len(trades)} "
        f"trades sized at the 1c/99c clip (slippage buffer inactive there)",
        f"  first half mean daily {pct(live.ret.iloc[:half].mean())} | second half {pct(live.ret.iloc[half:].mean())} "
        f"(not an out-of-sample test)",
    ]
    return "\n".join(lines)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--bankroll", type=float, default=BANKROLL)
    ap.add_argument("--cap", type=float, default=CAP_FRAC, help="worst-case loss cap as a fraction of bankroll")
    ap.add_argument("--q-max", type=int, default=Q_MAX)
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE)
    ap.add_argument("--slip", type=float, default=SLIP)
    ap.add_argument("--n0", type=float, default=N0)
    ap.add_argument("--warmup-days", type=int, default=WARMUP_DAYS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config(bankroll=a.bankroll, cap_frac=a.cap, q_max=a.q_max, min_edge=a.min_edge, slip=a.slip, n0=a.n0,
                 warmup_days=a.warmup_days)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    obs = load_obs(a.data)
    res = {s: run(obs, cfg, s) for s in SIZINGS}
    text = [f"Loaded {len(obs)} observations over {obs.day.nunique()} expiry days "
            f"({obs.day.min()} -> {obs.day.max()})" if len(obs) else "Loaded 0 observations at the 6h/3h horizons", ""]
    names = {"portfolio": "PORTFOLIO (worst-case cap)", "one": "B1 one contract per signal",
             "per_trade": "B2 per-trade cap"}
    firsts = [res[s][0].day.min() for s in SIZINGS if not res[s][0].empty]
    start = min(firsts) if firsts else None        # one common window so the three modes are comparable
    if start is not None:
        text += [f"All modes reported from {start} (first day any mode traded)", ""]
    for s in SIZINGS:
        text += [summarize(*res[s], cfg, names[s], start=start), ""]
    text.append("Worst expiry day vs the cap:")
    for s in SIZINGS:
        d = res[s][1]
        if d.empty or not (d.n_trades > 0).any():
            text.append(f"  {names[s]:<28} no trades")
            continue
        text.append(f"  {names[s]:<28} worst traded day {d.ret[d.n_trades > 0].min() * 100:+.3f}% of bankroll, "
                    f"days with realized worst case past cap: {int((d.realized_worst < -cfg.cap - 1e-9).sum())}")
    report = "\n".join(text)
    print(report)

    tr, dy = res["portfolio"]
    tr.to_csv(out / "tail_trades.csv", index=False)
    dy.to_csv(out / "tail_daily.csv", index=False)
    (out / "tail_summary.txt").write_text(report + "\n")

    if not a.no_plot and not dy.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for s in SIZINGS:
            d = res[s][1]
            ax[0].plot(pd.to_datetime(d.day), d.ret.cumsum() * 100, label=names[s])
        ax[0].set_title("Cumulative return, % of bankroll"); ax[0].legend()
        ax[1].hist(dy.ret[dy.n_trades > 0] * 100, bins=30)
        ax[1].set_title("Daily return on traded days, % (portfolio)")
        fig.tight_layout(); fig.savefig(out / "tail_backtest.png", dpi=130)
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
