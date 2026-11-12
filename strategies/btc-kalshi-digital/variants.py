#!/usr/bin/env python3
"""
Numbered model variants of the BTC Kalshi tail strategy
=======================================================
Each model changes exactly one thing from Model 0, the base tail strategy. A model's setting is tuned on the tuning
period (Jan - Jun 2026) by annualized Sharpe ratio, frozen, then scored once on the holdout (Jul 1 - Nov 11 2026).
Every run walks forward over the whole history; the periods only decide which days are scored.

Spec: docs/superpowers/specs/2026-11-11-btc-tail-variants-design.md
"""
import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from tail_strategy import Config, load_obs, run

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "btc_pm_results_long" / "btc_observations.csv"
OUT_DIR = HERE / "output" / "btc_tail_variants"

TUNE = (dt.date(2026, 1, 1), dt.date(2026, 6, 30))
TEST = (dt.date(2026, 7, 1), dt.date(2026, 11, 11))
ANNUAL = np.sqrt(365)

MODELS = {
    0: dict(name="Base", change="current tail strategy", overrides={}, grid={}),
    1: dict(name="Narrow tails", change="trade only mid < lo or > 1 - lo", overrides={},
            grid={"tail_lo": [0.03, 0.05, 0.075]}),
    2: dict(name="Strike distance", change="strike at least z forecast sd from spot", overrides={},
            grid={"z_min": [1.5, 2.0, 2.5]}),
    3: dict(name="6h only", change="drop the 3h round", overrides={"hours": (6,)}, grid={}),
    4: dict(name="Model agrees", change="volatility model's edge on our side >= m", overrides={},
            grid={"agree_margin": [-0.01, 0.0, 0.005]}),
    5: dict(name="Model prior", change="shrink toward the model probability", overrides={"prior": "model"}, grid={}),
    6: dict(name="Cautious", change="size on a pessimistic quantile of the loss chance", overrides={},
            grid={"quantile": [0.75, 0.9, 0.95]}),
    7: dict(name="Learned", change="walk-forward logistic regression", overrides={"estimator": "logit"}, grid={}),
    8: dict(name="Recent", change="outcomes decay with a half-life (days)", overrides={},
            grid={"half_life": [60, 90, 180]}),
    9: dict(name="Price moved", change="skip if the quote moved > x toward the money before the fill", overrides={},
            grid={"move_skip": [0.01, 0.02, 0.03]}),
    # POST-HOC: combined after seeing the results of 0-9, so their holdout scores are optimistic
    10: dict(name="Learned+moved", change="#7 learned + #9 price-move skip", overrides={"estimator": "logit"},
             grid={"move_skip": [0.01, 0.02, 0.03]}, post_hoc=True),
    11: dict(name="Prior+moved", change="#5 model prior + #9 price-move skip", overrides={"prior": "model"},
             grid={"move_skip": [0.01, 0.02, 0.03]}, post_hoc=True),
}


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def _sharpe_rows(m):
    """Annualized Sharpe ratio of each row of a 2-D array of daily returns (0 where the row has no variance)."""
    sd = m.std(axis=1, ddof=1)
    return np.where(sd > 0, m.mean(axis=1) / np.where(sd > 0, sd, 1) * ANNUAL, 0.0)


def sharpe(ret):
    v = np.asarray(ret, float)
    if len(v) < 2:
        return 0.0
    return float(_sharpe_rows(v[None, :])[0])


def sharpe_ci(ret, n=4000, seed=7):
    """95% bootstrap CI over days of the annualized Sharpe ratio."""
    v = np.asarray(ret, float)
    if len(v) < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    return tuple(float(x) for x in np.percentile(_sharpe_rows(v[idx]), [2.5, 97.5]))


def paired_sharpe_diff_ci(ret, base_ret, n=4000, seed=7):
    """95% CI of Sharpe(ret) - Sharpe(base_ret), resampling the same days for both series."""
    a, b = np.asarray(ret, float), np.asarray(base_ret, float)
    if len(a) < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    return tuple(float(x) for x in np.percentile(_sharpe_rows(a[idx]) - _sharpe_rows(b[idx]), [2.5, 97.5]))


def window_metrics(trades, daily, cfg, start, end):
    """Metrics over expiry days in [start, end], on a fixed bankroll."""
    live = daily[(daily.day >= start) & (daily.day <= end)]
    tr = trades[(trades.day >= start) & (trades.day <= end)] if len(trades) else trades
    cum = live.ret.cumsum()
    dd = float((cum - np.maximum.accumulate(np.maximum(cum, 0.0))).min()) if len(cum) else 0.0
    lo, hi = sharpe_ci(live.ret)
    return dict(days=len(live), traded_days=int((live.n_trades > 0).sum()), trades=len(tr),
                contracts=int(tr.q.sum()) if len(tr) else 0, ret=float(live.ret.sum()), sharpe=sharpe(live.ret),
                sharpe_lo=lo, sharpe_hi=hi, max_dd=dd, tail_days=int((live.pnl < -0.5 * cfg.cap).sum()),
                pnl=float(live.pnl.sum()))


# ----------------------------------------------------------------------------
# TUNING AND SCORING
# ----------------------------------------------------------------------------
def configs(k):
    """(label, Config) for every tuning value of model k (one entry when it has no grid)."""
    m = MODELS[k]
    if not m["grid"]:
        return [("-", Config(**m["overrides"]))]
    (field, values), = m["grid"].items()
    return [(f"{field}={v}", Config(**m["overrides"], **{field: v})) for v in values]


def evaluate(obs):
    """Tune every model on TUNE, then score the chosen setting on TUNE and TEST. Returns (tuning, comparison, runs)."""
    tuning, chosen = [], {}
    for k in MODELS:
        best = None
        for label, cfg in configs(k):
            tr, dy = run(obs, cfg)
            m = window_metrics(tr, dy, cfg, *TUNE)
            score = m["sharpe"] if m["trades"] > 0 else -np.inf
            tuning.append(dict(model=k, name=MODELS[k]["name"], setting=label, tune_score=score, **m))
            if best is None or score > best[0]:
                best = (score, label, cfg, tr, dy)
        chosen[k] = best
    base_dy = chosen[0][4]
    rows = []
    for k, (_, label, cfg, tr, dy) in chosen.items():
        row = dict(model=k, name=MODELS[k]["name"], change=MODELS[k]["change"], setting=label)
        for tag, (s, e) in (("tune", TUNE), ("test", TEST)):
            m = window_metrics(tr, dy, cfg, s, e)
            r = dy[(dy.day >= s) & (dy.day <= e)].ret.values
            rb = base_dy[(base_dy.day >= s) & (base_dy.day <= e)].ret.values
            dlo, dhi = paired_sharpe_diff_ci(r, rb)
            row.update({f"{tag}_{key}": v for key, v in m.items()})
            row.update({f"{tag}_diff": m["sharpe"] - sharpe(rb), f"{tag}_diff_lo": dlo, f"{tag}_diff_hi": dhi})
        row["post_hoc"] = bool(MODELS[k].get("post_hoc", False))
        row["beats_base_both"] = bool(k != 0 and row["tune_diff"] > 0 and row["test_diff"] > 0)
        rows.append(row)
    return pd.DataFrame(tuning), pd.DataFrame(rows), {k: v[4] for k, v in chosen.items()}


def report(comp, obs):
    pct = lambda x: f"{x * 100:+.2f}%"
    lines = [f"Numbered tail-strategy models on {obs.day.nunique()} expiry days ({obs.day.min()} -> {obs.day.max()})",
             f"Tuning {TUNE[0]} -> {TUNE[1]}; holdout {TEST[0]} -> {TEST[1]}; $1,000 bankroll, 3% daily worst-case cap",
             "Sharpe = annualized mean/sd of daily returns (days without trades count as 0). diff = model - base, "
             "paired bootstrap 95% CI.", ""]
    hdr = (f"{'#':>2} {'model':<16} {'setting':<18} | {'tune Sharpe':>11} {'diff':>6} | {'HOLDOUT Sharpe [95% CI]':>24} "
           f"{'diff [95% CI]':>22} | {'return':>7} {'max DD':>7} {'tail':>4} {'trades':>6}")
    lines += [hdr, "-" * len(hdr)]
    for r in comp.itertuples():
        if r.post_hoc and not comp.post_hoc.iloc[:r.Index].any():
            lines.append("-- post-hoc combinations (chosen after seeing 0-9; holdout scores are optimistic) --")
        lines.append(f"{r.model:>2} {r.name:<16} {r.setting:<18} | {r.tune_sharpe:>11.2f} {r.tune_diff:>+6.2f} | "
                     f"{r.test_sharpe:>6.2f} [{r.test_sharpe_lo:>+6.2f}, {r.test_sharpe_hi:>+6.2f}]   "
                     f"{r.test_diff:>+6.2f} [{r.test_diff_lo:>+5.2f}, {r.test_diff_hi:>+5.2f}] | "
                     f"{pct(r.test_ret):>7} {r.test_max_dd * 100:>6.2f}% {r.test_tail_days:>4} {r.test_trades:>6}")
    wins = comp[comp.beats_base_both & ~comp.post_hoc]
    lines += ["", "Pre-registered models that beat the base on BOTH the tuning period and the holdout: "
              + (", ".join(f"#{r.model} {r.name}" for r in wins.itertuples()) if len(wins) else "none"),
              "Nine models were tried, so one beating the base on the holdout alone is expected by chance."]
    ph = comp[comp.post_hoc]
    if len(ph):
        lines.append("Post-hoc combinations beating the base on both: "
                     + (", ".join(f"#{r.model} {r.name}" for r in ph[ph.beats_base_both].itertuples()) or "none")
                     + " (needs data after Nov 11 2026 to confirm)")
    return "\n".join(lines)


def plot(comp, runs, path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.parse_math"] = False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for r in comp.itertuples():
        dy = runs[r.model]
        live = dy[(dy.day >= TEST[0]) & (dy.day <= TEST[1])]
        eq = 1000 + live.pnl.cumsum()
        if r.model == 0:
            style = dict(color="black", lw=2.6, zorder=5)
        elif r.post_hoc:
            style = dict(color=cmap(r.model % 10), lw=1.8, ls="--")
        else:
            style = dict(color=cmap(r.model), lw=1.3)
        ax.plot(pd.to_datetime(live.day), eq, label=f"#{r.model} {r.name} (Sharpe {r.test_sharpe:.2f})", **style)
    ax.axhline(1000, color="#888", lw=0.8, ls="--")
    ax.set_title("Holdout (Jul 1 - Nov 11 2026): account value from $1,000, by model", loc="left")
    ax.set_ylabel("$")
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs = load_obs(a.data)
    tuning, comp, runs = evaluate(obs)
    tuning.to_csv(out / "tuning.csv", index=False)
    comp.to_csv(out / "comparison.csv", index=False)
    text = report(comp, obs)
    (out / "variants_summary.txt").write_text(text + "\n")
    print(text)
    if not a.no_plot:
        plot(comp, runs, out / "holdout_equity.png")
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
