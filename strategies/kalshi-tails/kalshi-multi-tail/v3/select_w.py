"""
Predictor v3 pre-registered choice of the blend weight w (spec section 6), BTC + SPX, development period only.

Rows after DEV_END (2026-06-30) are dropped on load, before any estimator or scoring code sees them. The two
frozen components (v1 P and v2 F1, built exactly as BlendEstimator builds them) are each run walk-forward once;
the blend f = w * f_P + (1 - w) * f_F1 (clipped) is formed in memory for every w in WEIGHTS, which equals running
BlendEstimator(w) walk-forward (Task 1). Scored rows: days >= 2025-01-01 where P n >= 100 and F1 n >= 200, both
finite. Rule: lowest mean log-loss; take w = 0.5 if its paired delta to the best is within 1 SE.
Writes v3/output/selection.txt; predictions go to v3/output/preds/ (git-ignored).

    cd strategies/kalshi-tails/kalshi-multi-tail && python -m v3.select_w
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multi_tail import MultiConfig
from v2.scoring import logloss, paired, walk_forward_predict
from v2.select import DEV_END, KEY, MARKETS, SCORE_START, _frame, _ll_table, _pair_line, dev_only, load_dev
from v3.blend import CLIP, P_MIN_N, BlendEstimator

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_W = 0.5
OUT = Path(__file__).resolve().parent / "output"
NO_ARM_LINE = "Chosen w reproduces a component; no new v3 arm is frozen."


def _name(w):
    return f"w={w:.2f}"


def choose_w(ll, day):
    """`ll` maps w -> per-row log-loss on the scored rows. Pick the lowest mean log-loss; if w = 0.5's paired delta
    to the best is within 1 SE, take 0.5. Returns (w, table[w, ll, delta, se, within])."""
    ws = list(ll)
    means = {w: float(np.mean(ll[w])) for w in ws}
    best = min(ws, key=lambda w: means[w])
    rows = []
    for w in ws:
        delta, se = (0.0, 0.0) if w == best else paired(ll[w], ll[best], day)
        rows.append(dict(w=w, ll=means[w], delta=delta, se=se, within=bool(delta <= se)))
    table = pd.DataFrame(rows)
    half = table[table.w == DEFAULT_W]
    chosen = DEFAULT_W if len(half) and bool(half.within.iloc[0]) else best
    return chosen, table


def _components(cfg):
    b = BlendEstimator(DEFAULT_W, cfg.n0, cfg.n1)
    return b.p_est, b.v2_est


def _predict(obs, cfg):
    p_est, f_est = _components(cfg)
    return {"P": walk_forward_predict(obs, p_est, cfg), "F1": walk_forward_predict(obs, f_est, cfg)}, f_est.min_rows


def _scored_frame(obs, cfg, score_start):
    preds, f1_min = _predict(obs, cfg)
    df = _frame(preds)
    dev = df.day >= score_start
    df["scored"] = (dev & np.isfinite(df.p_P) & (df.n_P >= P_MIN_N)
                    & np.isfinite(df.p_F1) & (df.n_F1 >= f1_min))
    for w in WEIGHTS:
        p = np.clip(w * df.p_P + (1 - w) * df.p_F1, *CLIP)
        df[f"p_{_name(w)}"] = np.where(df.scored, p, np.nan)
        df[f"ll_{_name(w)}"] = np.where(df.scored, logloss(p.fillna(0.5), df.y), np.nan)
    return df, dev, f1_min


def run_selection(obs, cfg=None, score_start=SCORE_START, return_frame=False):
    cfg = cfg or MultiConfig()
    obs = dev_only(obs)
    df, dev, f1_min = _scored_frame(obs, cfg, score_start)
    c = df[df.scored]
    cd = c.day.to_numpy()
    chosen, table = choose_w({w: c[f"ll_{_name(w)}"].to_numpy() for w in WEIGHTS}, cd)
    names = [_name(w) for w in WEIGHTS] + ["mid"]

    L = [f"Predictor v3 weight selection (spec section 6), markets {' + '.join(sorted(df.market.unique()))}",
         f"walk-forward over expiry days {min(obs.day)} -> {max(obs.day)}; scored days {score_start} -> {DEV_END}",
         f"f_v3 = w * f_P + (1 - w) * f_F1, clipped to [{CLIP[0]}, {CLIP[1]}]; w = 1 is v1 P, w = 0 is v2 F1",
         f"scored rows: dev rows where v1 P has n >= {P_MIN_N} and v2 F1 has n >= {f1_min}, both finite; "
         "log-loss uses p clipped to [0.001, 0.999]",
         "SE = std of 4,000 moving-block bootstrap resamples (7-day blocks, seed 11), row-weighted delta inside each",
         "", "Row counts per market (live = warm tail candidates on scored days)",
         f"  {'market':<8}{'live':>8}{'scored':>8}"]
    for m, g in df[dev].groupby("market"):
        L.append(f"  {m:<8}{len(g):8d}{int(g.scored.sum()):8d}")
    L.append(f"  {'total':<8}{int(dev.sum()):8d}{len(c):8d}")
    L += ["", "Mean log-loss on scored rows"] + ["  " + x for x in _ll_table(c, names)]
    L += ["", "by market"] + ["  " + x for x in _ll_table(c, names, "market")]
    L += ["", "by H"] + ["  " + x for x in _ll_table(c, names, "H")]

    L += ["", "Paired delta vs mid = LL(w) - LL(mid), same rows", f"  {'pair':<26}{'delta':>10}{'SE':>10}{'d/SE':>8}"]
    for w in WEIGHTS:
        L.append(_pair_line(f"{_name(w)} - mid", c[f"ll_{_name(w)}"], c.ll_mid, cd))

    L += ["", "Selection: lowest log-loss; take w = 0.5 if within 1 SE of the best on the paired delta",
          f"  {'w':<6}{'LL':>10}{'delta vs best':>15}{'SE':>10}  within 1 SE"]
    for r in table.itertuples(index=False):
        L.append(f"  {r.w:<6.2f}{r.ll:10.5f}{r.delta:+15.5f}{r.se:10.5f}  {'yes' if r.within else 'no'}")
    L.append(f"CHOSEN w: {chosen:.2f}")
    if chosen in (0.0, 1.0):
        L.append(NO_ARM_LINE)
    text = "\n".join(L) + "\n"
    return (chosen, df, text) if return_frame else text


def main():
    t0 = time.time()
    obs = load_dev(MARKETS)
    chosen, df, text = run_selection(obs, return_frame=True)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    df[KEY + ["p_P", "n_P", "p_F1", "n_F1", "scored"] + [f"p_{_name(w)}" for w in WEIGHTS]].to_csv(
        OUT / "preds" / "dev_blend.csv", index=False)
    text += "\n"
    (OUT / "selection.txt").write_text(text)
    print(text)
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
