"""
Walk-forward scoring and the pre-registered selection rule for predictor v2 (spec section 6).

walk_forward_predict mirrors the day loop of multi_tail.run without the sizing: per expiry day, rows with H in
ts.HOURS -> ts.tail_candidates -> rows of markets past the warmup (and H in cfg.hours) are predicted ->
est.update(tails) -> days_seen is bumped for every market present that day. It records every live row, with
p = NaN when the estimator has no estimate. Eligibility (n thresholds) is applied by the caller.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARENT = Path(__file__).resolve().parent.parent
for _p in (PARENT, PARENT.parent / "btc-kalshi-digital"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import tail_strategy as ts  # noqa: E402
from tail_sizer import order_fee  # noqa: E402

CLIP = (0.001, 0.999)
PRED_COLS = ["day", "market", "ticker", "H", "mid", "bid", "ask", "bid_f", "ask_f", "y", "p", "n"]


def walk_forward_predict(obs, est, cfg):
    out = []
    days_seen = {}
    for day, g in obs.groupby("day", sort=True):
        g = g[g.H.isin(ts.HOURS)]
        tails = ts.tail_candidates(g)
        live = tails[tails.H.isin(cfg.hours) & tails.market.map(lambda m: days_seen.get(m, 0) >= cfg.warmup_days)]
        for r in live.sort_values("decision_ts", kind="stable").itertuples(index=False):
            f, n = est.estimate_row(r) if hasattr(est, "estimate_row") else est.estimate(r.market, r.H, r.mid)
            out.append((day, r.market, r.ticker, int(r.H), r.mid, r.bid, r.ask, r.bid_f, r.ask_f, r.y,
                        float(f) if np.isfinite(f) else np.nan, int(n)))
        est.update(tails)
        for m in g.market.unique():
            days_seen[m] = days_seen.get(m, 0) + 1
    return pd.DataFrame(out, columns=PRED_COLS)


def clip_p(p):
    return np.clip(np.asarray(p, float), *CLIP)


def logloss(p, y):
    p, y = clip_p(p), np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_bootstrap_delta(day, d, n=4000, seed=11, block=7):
    """n resampled row-weighted means of d: moving blocks of `block` consecutive distinct days (sorted), start
    indices uniform on [0, D - block], ceil(D / block) blocks concatenated and truncated to D days. D < block uses
    a single block of all D days."""
    day, d = np.asarray(day), np.asarray(d, float)
    u, inv = np.unique(day, return_inverse=True)
    D = len(u)
    S = np.bincount(inv, weights=d, minlength=D)
    C = np.bincount(inv, minlength=D).astype(float)
    if D < block:
        idx = np.tile(np.arange(D), (n, 1))
    else:
        nb = math.ceil(D / block)
        starts = np.random.default_rng(seed).integers(0, D - block + 1, (n, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, nb * block)[:, :D]
    return S[idx].sum(axis=1) / C[idx].sum(axis=1)


def paired(a, b, day):
    """(row-weighted mean of a - b, block-bootstrap SE)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(np.std(block_bootstrap_delta(day, d), ddof=1))


def select_set(ll, day):
    """Spec section 6 step 4. `ll` maps set name -> per-row log-loss on the common rows, in order of simplicity.
    Pick the lowest mean log-loss; if a simpler set's paired delta to the best is within 1 SE, take the simplest
    such set. Returns (chosen, table[set, ll, delta, se, within])."""
    names = list(ll)
    means = {s: float(np.mean(ll[s])) for s in names}
    best = min(names, key=lambda s: means[s])
    rows = []
    for s in names:
        delta, se = (0.0, 0.0) if s == best else paired(ll[s], ll[best], day)
        rows.append(dict(set=s, ll=means[s], delta=delta, se=se, within=bool(delta <= se)))
    table = pd.DataFrame(rows)
    chosen = next(r["set"] for r in rows if r["within"])
    return chosen, table


def _side_fill(df, cfg):
    p = df.p.to_numpy(float)
    side = np.zeros(len(df), int)
    edge = np.full(len(df), -np.inf)
    for i, (b, a, f) in enumerate(zip(df.bid.to_numpy(float), df.ask.to_numpy(float), p)):
        if np.isfinite(f):
            side[i], _, edge[i] = ts.best_side(b, a, f, cfg)
    fill = np.where(side == 1, df.ask_f.to_numpy(float), df.bid_f.to_numpy(float))
    ok = np.isfinite(fill) & np.where(side == 1, fill < 1.0, fill > 0.0) & (side != 0)
    return side, fill, edge, ok


def would_trade(df, cfg):
    """Rows where best_side's net edge >= cfg.min_edge and the fill quote is valid, as in multi_tail._candidates
    (the caller applies the estimator's n threshold)."""
    _, _, edge, ok = _side_fill(df, cfg)
    return ok & (edge >= cfg.min_edge)


def one_contract_pnl(df, cfg):
    """Hypothetical P&L of one contract on the best_side at the fill quote, fee rounded up as in run; 0 where the
    model would not trade."""
    side, fill, edge, ok = _side_fill(df, cfg)
    trade = ok & (edge >= cfg.min_edge)
    y = df.y.to_numpy(float)
    pnl = np.zeros(len(df))
    for i in np.flatnonzero(trade):
        pnl[i] = side[i] * (y[i] - fill[i]) - order_fee(1, fill[i], cfg.fee_rate)
    return pnl
