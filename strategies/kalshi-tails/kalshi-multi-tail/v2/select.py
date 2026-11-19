"""
Predictor v2 selection run (spec section 6), BTC + SPX, development period only.

Loads features for expiry days 2024-07-05 -> 2026-06-30 and drops every later row on load, before any estimator
or scoring code sees them. Runs S, P and V2 F1/F2/F3 walk-forward (v2.scoring.walk_forward_predict), scores the
common rows (development days 2025-01-01 -> 2026-06-30 where every v2 set has n >= min_rows and v1 P has
n >= min_cell_n), applies the pre-registered rule (v2.scoring.select_set) and writes v2/output/selection.txt.
Predictions go to v2/output/preds/dev_<estimator>.csv (git-ignored).

    cd strategies/kalshi-tails/kalshi-multi-tail && python -m v2.select
"""
import dataclasses
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multi_tail import V2_FEATURES, MultiConfig, load_features
from estimators import make_estimator
from v2.scoring import (clip_p, logloss, one_contract_pnl, paired, select_set,
                        walk_forward_predict, would_trade)

BURN_START = dt.date(2024, 7, 5)
SCORE_START = dt.date(2025, 1, 1)
DEV_END = dt.date(2026, 6, 30)
MARKETS = ["BTC", "SPX"]
SETS = ["F1", "F2", "F3"]
ESTIMATORS = ["S", "P"] + SETS
OUT = Path(__file__).resolve().parent / "output"
KEY = ["day", "market", "ticker", "H", "mid", "bid", "ask", "bid_f", "ask_f", "y"]


def dev_only(obs):
    return obs[(obs.day >= BURN_START) & (obs.day <= DEV_END)].reset_index(drop=True)


def load_dev(keys=MARKETS, path=V2_FEATURES):
    """Features for `keys`, typed as multi_tail.load_features types them, restricted to day <= DEV_END."""
    return dev_only(load_features(keys, path))


def _predict_all(obs, cfg):
    preds, cond, min_rows = {}, {}, None
    for name in ESTIMATORS:
        kind = name if name in ("S", "P") else "V2"
        c = dataclasses.replace(cfg, estimator=kind, v2_set=name if kind == "V2" else cfg.v2_set)
        est = make_estimator(kind, c)
        preds[name] = walk_forward_predict(obs, est, c)
        if kind == "V2":
            cond[name] = max(est.cond_log) if est.cond_log else float("nan")
            min_rows = est.min_rows
    return preds, cond, min_rows


def _frame(preds):
    """One row per walk-forward row, with p_/n_/ll_ columns for every estimator in `preds` plus mid."""
    names = list(preds)
    base = preds[names[0]][KEY].reset_index(drop=True)
    for name, p in preds.items():
        if not p[["day", "ticker", "H"]].reset_index(drop=True).equals(base[["day", "ticker", "H"]]):
            raise RuntimeError(f"walk-forward rows for {name} differ from {names[0]}")
        base[f"p_{name}"] = p.p.values
        base[f"n_{name}"] = p.n.values
    base["p_mid"] = base.mid
    for name in names + ["mid"]:
        base[f"ll_{name}"] = logloss(base[f"p_{name}"], base.y)
    return base


def _ll_table(df, names, by=None):
    cols = [f"{'n':>6}"] + [f"{n:>9}" for n in names]
    lines = [f"{(by or 'all'):<8}" + "".join(cols)]
    groups = [("all", df)] if by is None else list(df.groupby(by))
    for k, g in groups:
        lines.append(f"{str(k):<8}{len(g):6d}" + "".join(f"{g[f'll_{n}'].mean():9.5f}" for n in names))
    return lines


def _pair_line(label, a, b, day):
    d, se = paired(a, b, day)
    return f"  {label:<26}{d:+10.5f}{se:10.5f}{d / se if se > 0 else float('nan'):8.2f}"


def _reliability(df, names):
    out = []
    for tail, lo, hi in (("low tail (mid < 0.10)", 0.0, 0.10), ("high tail (mid > 0.90)", 0.90, 1.0)):
        g = df[(df.mid > lo) & (df.mid < hi)] if lo == 0.0 else df[(df.mid > lo) & (df.mid <= hi)]
        edges = np.linspace(lo, hi, 11)
        b = np.clip(np.digitize(g.mid, edges) - 1, 0, 9)
        out.append(f"{tail}: 10 equal-width mid bins; realized YES rate vs mean clipped p")
        out.append(f"  {'mid bin':<14}{'n':>6}{'y':>8}" + "".join(f"{n:>8}" for n in names))
        for i in range(10):
            h = g[b == i]
            if not len(h):
                out.append(f"  [{edges[i]:.3f},{edges[i + 1]:.3f}){0:6d}")
                continue
            out.append(f"  [{edges[i]:.3f},{edges[i + 1]:.3f}){len(h):6d}{h.y.mean():8.4f}"
                       + "".join(f"{clip_p(h[f'p_{n}']).mean():8.4f}" for n in names))
    return out


def run_selection(obs, cfg=None, score_start=SCORE_START):
    """Returns (chosen set, selection table, per-row frame, report text). Rows after DEV_END are dropped first."""
    cfg = cfg or MultiConfig()
    obs = obs[obs.day <= DEV_END].reset_index(drop=True)
    preds, cond, min_rows = _predict_all(obs, cfg)
    df = _frame(preds)
    dev = df.day >= score_start
    elig = dev & np.isfinite(df.p_P) & (df.n_P >= cfg.min_cell_n)
    for s in SETS:
        elig &= np.isfinite(df[f"p_{s}"]) & (df[f"n_{s}"] >= min_rows)
    df["common"] = elig
    df["s_subset"] = elig & np.isfinite(df.p_S) & (df.n_S >= cfg.min_cell_n)
    c, s = df[df.common], df[df.s_subset]
    chosen, table = select_set({k: c[f"ll_{k}"].to_numpy() for k in SETS}, c.day.to_numpy())

    names = SETS + ["P", "mid"]
    L = [f"Predictor v2 selection (spec section 6), markets {' + '.join(sorted(df.market.unique()))}",
         f"walk-forward over expiry days {min(obs.day)} -> {max(obs.day)}; scored days {score_start} -> {DEV_END}",
         f"common rows: dev rows where every v2 set has n >= {min_rows} and v1 P has n >= {cfg.min_cell_n}; "
         f"log-loss uses p clipped to [0.001, 0.999]",
         "SE = std of 4,000 moving-block bootstrap resamples (7-day blocks, seed 11), row-weighted delta inside each", "",
         "Row counts per market (live = warm tail candidates on scored days)",
         f"  {'market':<8}{'live':>8}{'common':>8}{'S subset':>10}"]
    for m, g in df[dev].groupby("market"):
        L.append(f"  {m:<8}{len(g):8d}{int(g.common.sum()):8d}{int(g.s_subset.sum()):10d}")
    L.append(f"  {'total':<8}{int(dev.sum()):8d}{len(c):8d}{len(s):10d}")
    L += ["", "Mean log-loss on common rows"] + ["  " + x for x in _ll_table(c, names)]
    L += ["", "by market"] + ["  " + x for x in _ll_table(c, names, "market")]
    L += ["", "by H"] + ["  " + x for x in _ll_table(c, names, "H")]
    L += ["", "v1 S subset (common rows where v1 S's own cell n >= 100), labeled as that subset"]
    L += ["  " + x for x in _ll_table(s, ["S"] + names)]
    L += ["  " + x for x in _ll_table(s, ["S"] + names, "market")[1:]]

    L += ["", "Paired delta = LL(a) - LL(b), same rows", f"  {'pair':<26}{'delta':>10}{'SE':>10}{'d/SE':>8}"]
    cd = c.day.to_numpy()
    for a, b in (("F2", "F1"), ("F3", "F2"), ("F3", "F1")):
        L.append(_pair_line(f"{a} - {b}", c[f"ll_{a}"], c[f"ll_{b}"], cd))
    for ref in ("P", "mid"):
        for a in SETS:
            L.append(_pair_line(f"{a} - {ref}", c[f"ll_{a}"], c[f"ll_{ref}"], cd))
    L.append(_pair_line("P - mid", c.ll_P, c.ll_mid, cd))
    sd = s.day.to_numpy()
    for a in SETS + ["P"]:
        L.append(_pair_line(f"{a} - S (S subset)", s[f"ll_{a}"], s.ll_S, sd))

    L += ["", "Selection (step 4): lowest log-loss; simplest set within 1 SE of the best on the paired delta",
          f"  {'set':<6}{'LL':>10}{'delta vs best':>15}{'SE':>10}  within 1 SE"]
    for r in table.itertuples(index=False):
        L.append(f"  {r.set:<6}{r.ll:10.5f}{r.delta:+15.5f}{r.se:10.5f}  {'yes' if r.within else 'no'}")
    L += [f"CHOSEN SET: {chosen}", ""]

    L += ["Reliability on common rows (report only)"] + _reliability(c, names)
    L += ["", "Max condition number of the standardized design matrix over all fits"]
    L += [f"  {k}: {v:.2f}" for k, v in cond.items()] + [f"  max: {max(cond.values()):.2f}", ""]

    L += [f"Would-trade secondary (report only): common rows where v2 or v1 P would trade (net edge >= "
          f"{cfg.min_edge} after slip {cfg.slip} and fees, by best_side, valid fill quote)",
          f"  {'v2 set':<8}{'rows':>6}{'delta_P':>10}{'SE':>10}{'v2 n/pnl':>16}{'P n/pnl':>16}"]
    for a in SETS:
        wa = would_trade(c.assign(p=c[f"p_{a}"]), cfg)
        wp = would_trade(c.assign(p=c.p_P), cfg)
        w = c[wa | wp]
        if not len(w):
            L.append(f"  {a:<8}{0:6d}")
            continue
        d, se = paired(w[f"ll_{a}"], w.ll_P, w.day.to_numpy())
        pa = one_contract_pnl(w.assign(p=w[f"p_{a}"]), cfg)
        pp = one_contract_pnl(w.assign(p=w.p_P), cfg)
        L.append(f"  {a:<8}{len(w):6d}{d:+10.5f}{se:10.5f}{int(wa.sum()):7d} ${pa.sum():+7.2f}"
                 f"{int(wp.sum()):7d} ${pp.sum():+7.2f}")
    L.append("  one-contract P&L at the fill quote, fee rounded up to the cent")
    return chosen, table, df, "\n".join(L) + "\n"


def main():
    t0 = time.time()
    obs = load_dev(MARKETS)
    chosen, table, df, text = run_selection(obs)
    text += f"\nruntime {time.time() - t0:.0f}s\n"
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    for name in ESTIMATORS:
        df[KEY + [f"p_{name}", f"n_{name}", "common", "s_subset"]].to_csv(OUT / "preds" / f"dev_{name}.csv",
                                                                          index=False)
    (OUT / "selection.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
