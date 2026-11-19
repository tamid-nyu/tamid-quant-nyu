"""
Predictor v3 evaluation after the freeze (spec section 7).

    cd strategies/kalshi-tails/kalshi-multi-tail
    python -m v3.evaluate --period holdout             # -> v3/output/holdout.txt (seen period, not the verdict)
    python -m v3.evaluate --period forward [--today D]  # --today can only move earlier; before 2027-05-12: data-quality counts only

Both periods walk v1 P, v2 F1 (v2.logit_v2.FROZEN) and the frozen v3 blend (v3.blend.FROZEN) forward over every row
up to the period's last expiry day and score only the period's days. Common rows: period rows where P has n >= 100,
F1 has n >= 200 and v3 is eligible (it is exactly when both components are). One-sided test of a paired delta: the
95th percentile of v2.scoring.block_bootstrap_delta (4,000 resamples, seed 11, 7-day blocks) is < 0. Forward
verdict: WIN only if both delta_P = LL(v3) - LL(P) and delta_F1 = LL(v3) - LL(F1) pass, else "not shown". Primary
scope BTC + SPX; the five-market block is exploratory. Helpers, periods and the forward guard come from v2.evaluate.
"""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np

import multi_tail as mt
from estimators import make_estimator
from multi_tail import V2_FEATURES, MultiConfig, load_features
from v2.evaluate import FIVE, FORWARD, FORWARD_OPEN, HOLDOUT, MDE_DAYS, PRIMARY, _check_frozen, mde, one_sided
from v2.evaluate import data_quality  # reads v2.evaluate.FEATURES_PATH / RAW_DIR
from v2.evaluate import _cfg as _v2_cfg
from v2.logit_v2 import FROZEN as F1_FROZEN
from v2.scoring import one_contract_pnl, walk_forward_predict, would_trade
from v2.select import BURN_START, _frame, _ll_table
from v3.blend import FROZEN, P_MIN_N, BlendEstimator

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FEATURES_PATH = V2_FEATURES
MODELS = {"V3": "V3", "P": "P", "F1": "V2"}          # report name -> multi_tail estimator kind


# ----------------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------------
def verdict(df):
    """Spec section 7 forward verdict on a scored frame (ll_V3, ll_P, ll_F1, day, common)."""
    c = df[df.common]
    day = c.day.to_numpy()
    tests = {"P": one_sided(day, (c.ll_V3 - c.ll_P).to_numpy()),
             "F1": one_sided(day, (c.ll_V3 - c.ll_F1).to_numpy())}
    return ("WIN" if tests["P"]["pass"] and tests["F1"]["pass"] else "not shown"), tests


# ----------------------------------------------------------------------------
# walk-forward and scoring
# ----------------------------------------------------------------------------
def _cfg(name, cfg):
    return _v2_cfg(MODELS[name], cfg)


def _estimator(name, cfg):
    est = make_estimator(MODELS[name], _cfg(name, cfg))
    if name == "F1":
        _check_frozen(est)
    if name == "V3" and not (isinstance(est, BlendEstimator) and est.w == FROZEN["w"]):
        raise RuntimeError(f"V3 estimator differs from FROZEN {FROZEN}")
    return est


def score(obs, start, end, cfg):
    """Walk V3, P, F1 forward over rows with day <= end; mark common rows on days in [start, end]."""
    obs = obs[obs.day <= end].reset_index(drop=True)
    preds = {name: walk_forward_predict(obs, _estimator(name, cfg), _cfg(name, cfg)) for name in MODELS}
    df = _frame(preds)
    df["period"] = (df.day >= start) & (df.day <= end)
    df["common"] = (df.period & np.isfinite(df.p_P) & (df.n_P >= P_MIN_N)
                    & np.isfinite(df.p_F1) & (df.n_F1 >= F1_FROZEN["min_rows"])
                    & np.isfinite(df.p_V3) & (df.n_V3 >= BlendEstimator.min_rows))
    return df


def _test_line(label, g, a, b):
    if not len(g):
        return f"  {label:<14}{0:6d}"
    t = one_sided(g.day.to_numpy(), (g[f"ll_{a}"] - g[f"ll_{b}"]).to_numpy())
    return (f"  {label:<14}{t['n']:6d}{t['D']:6d}{t['delta']:+11.5f}{t['se']:10.5f}{t['p95']:+11.5f}"
            f"  {'yes' if t['pass'] else 'no'}")


def report(df, cfg, mde_block=True):
    c = df[df.common]
    names = ["V3", "P", "F1", "mid"]
    L = ["Row counts per market (live = warm tail candidates on the period's days)",
         f"  {'market':<8}{'live':>8}{'common':>8}{'days':>6}"]
    for m, g in df[df.period].groupby("market"):
        L.append(f"  {m:<8}{len(g):8d}{int(g.common.sum()):8d}{g[g.common].day.nunique():6d}")
    L.append(f"  {'total':<8}{int(df.period.sum()):8d}{len(c):8d}{c.day.nunique():6d}")
    if not len(c):
        return L + ["", "no common rows"]
    L += ["", f"Mean log-loss on common rows (V3 = blend w = {FROZEN['w']}, P = v1 pooled, "
              f"F1 = v2 set {F1_FROZEN['set']})"] + ["  " + x for x in _ll_table(c, names)]
    L += ["", "by market"] + ["  " + x for x in _ll_table(c, names, "market")]
    L += ["", "by H"] + ["  " + x for x in _ll_table(c, names, "H")]

    L += ["", "Paired delta = LL(a) - LL(b), same rows; one-sided test passes iff the bootstrap 95th pct < 0",
          f"  {'pair':<14}{'rows':>6}{'days':>6}{'delta':>11}{'SE':>10}{'95th pct':>11}  p95 < 0"]
    for a, b in (("V3", "P"), ("V3", "F1"), ("V3", "mid"), ("F1", "P"), ("P", "mid"), ("F1", "mid")):
        L.append(_test_line(f"{a} - {b}", c, a, b))

    if mde_block:
        L += ["", f"Minimum detectable effect for a {MDE_DAYS}-day forward window: SE_h = std of the bootstrap "
                  f"delta, D = distinct days,",
              f"SE_{MDE_DAYS} = SE_h * sqrt(D / {MDE_DAYS}), MDE = 2.5 * SE_{MDE_DAYS}; per-day SD = SD of the "
              "per-day mean delta (information only)",
              f"  {'comparator':<14}{'D':>5}{'SE_h':>10}{'SE_' + str(MDE_DAYS):>10}{'MDE':>10}{'per-day SD':>12}"]
        for b in ("P", "F1"):
            r = mde(c.day.to_numpy(), (c.ll_V3 - c[f"ll_{b}"]).to_numpy())
            L.append(f"  {'V3 - ' + b:<14}{r['D']:5d}{r['se_h']:10.5f}{r['se_181']:10.5f}{r['mde']:10.5f}"
                     f"{r['day_sd']:12.5f}")

    wt = {n: would_trade(c.assign(p=c[f"p_{n}"]), cfg) for n in ("V3", "P", "F1")}
    w = c[wt["V3"] | wt["P"] | wt["F1"]]
    L += ["", f"Would-trade secondary (report only): common rows where v3, v1 P or v2 F1 would trade (net edge >= "
              f"{cfg.min_edge} after slip {cfg.slip} and fees, by best_side, valid fill quote)",
          f"  {'rows':>6}{'days':>6}{'delta_P':>11}{'95th pct':>11}{'delta_F1':>11}{'95th pct':>11}"
          f"{'v3 n/pnl':>17}{'P n/pnl':>17}{'F1 n/pnl':>17}"]
    if len(w):
        tp = one_sided(w.day.to_numpy(), (w.ll_V3 - w.ll_P).to_numpy())
        tf = one_sided(w.day.to_numpy(), (w.ll_V3 - w.ll_F1).to_numpy())
        pnl = "".join(f"{int(wt[n].sum()):8d} ${one_contract_pnl(w.assign(p=w[f'p_{n}']), cfg).sum():+7.2f}"
                      for n in ("V3", "P", "F1"))
        L.append(f"  {tp['n']:6d}{tp['D']:6d}{tp['delta']:+11.5f}{tp['p95']:+11.5f}{tf['delta']:+11.5f}"
                 f"{tf['p95']:+11.5f}{pnl}")
    else:
        L.append(f"  {0:6d}")
    L.append("  one-contract P&L at the fill quote, fee rounded up to the cent")
    return L


def sharpe_block(obs, start, end, cfg):
    """Sharpe of v3, v1 P and v2 F1 through multi_tail.run (frozen sizer) on the period's days, paired_diff_ci
    of v3 against each, and trades and winning trades per day."""
    obs = obs[obs.day <= end].reset_index(drop=True)
    days = sorted(obs.day.unique())
    w = {}
    for name in MODELS:
        _, daily, _ = mt.run(obs, _cfg(name, cfg))
        w[name] = mt._window(daily, days, start, end)
    nd = len(w["P"])
    L = [f"Sharpe through the frozen sizer (multi_tail.run: c = {cfg.cap_m:.0%}, k = {cfg.k}, C = {cfg.cap_k:.0%}, "
         f"${cfg.bankroll:,.0f}); daily returns on {nd} period days, days without trades count as 0",
         f"  {'model':<8}{'Sharpe':>8}  {'95% CI':<16}{'return':>9}{'trades':>8}{'wins':>7}{'trades/day':>12}"
         f"{'wins/day':>10}"]
    for name in MODELS:
        r = w[name]
        lo, hi = mt.boot_sharpe_ci(r.ret)
        nt, nw = int(r.n_trades.sum()), int(r.n_wins.sum())
        L.append(f"  {name:<8}{mt.sharpe(r.ret):8.2f}  [{lo:5.2f}, {hi:5.2f}]  {r.ret.sum() * 100:+8.2f}%"
                 f"{nt:8d}{nw:7d}{nt / max(nd, 1):12.2f}{nw / max(nd, 1):10.2f}")
    for b in ("P", "F1"):
        d, lo, hi = mt.paired_diff_ci(w["V3"].ret.values, w[b].ret.values)
        L.append(f"  Sharpe(V3) - Sharpe({b}) = {d:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] (paired_diff_ci, same days)")
    return L


def _evaluate(start, end, header, cfg, with_verdict):
    t0 = time.time()
    obs5 = load_features(FIVE, FEATURES_PATH)
    obs5 = obs5[(obs5.day >= BURN_START) & (obs5.day <= end)].reset_index(drop=True)
    prim = obs5[obs5.market.isin(PRIMARY)].reset_index(drop=True)
    df = score(prim, start, end, cfg)
    L = list(header) + [
        f"Frozen v3: {FROZEN} (f = w * f_P + (1 - w) * f_F1, clipped to [0.001, 0.999]); "
        f"P = PooledEstimator(n0={cfg.n0:g}, n1={cfg.n1:g}); F1 = {F1_FROZEN}",
        f"scored expiry days {start} -> {end}; walk-forward over all rows from {BURN_START} up to {end}",
        f"common rows: period rows where v1 P has n >= {P_MIN_N}, v2 F1 has n >= {F1_FROZEN['min_rows']} and v3 is "
        "eligible; log-loss uses p clipped to [0.001, 0.999]",
        "bootstrap: 4,000 moving-block resamples (7-day blocks, seed 11), row-weighted delta inside each", ""]
    if with_verdict:
        label, tests = verdict(df)
        L += ["PRIMARY VERDICT (BTC + SPX): " + label,
              f"  delta_P = {tests['P']['delta']:+.5f} (95th pct {tests['P']['p95']:+.5f}, "
              f"{'pass' if tests['P']['pass'] else 'fail'}), "
              f"delta_F1 = {tests['F1']['delta']:+.5f} (95th pct {tests['F1']['p95']:+.5f}, "
              f"{'pass' if tests['F1']['pass'] else 'fail'}); WIN only if both pass", ""]
    L += ["=== PRIMARY: BTC + SPX ==="] + report(df, cfg, mde_block=not with_verdict)
    L += [""] + sharpe_block(prim, start, end, cfg)
    df5 = score(obs5, start, end, cfg)
    L += ["", f"=== EXPLORATORY: five markets ({' + '.join(FIVE)}), no decision attached ==="]
    L += report(df5, cfg, mde_block=False) + [""] + sharpe_block(obs5, start, end, cfg)
    L.append("")
    print(f"runtime {time.time() - t0:.0f}s")
    return "\n".join(L) + "\n"


def holdout(cfg=None):
    cfg = cfg or MultiConfig()
    header = ["HOLDOUT: seen period, not the verdict",
              "Predictor v3 holdout evaluation (spec section 7), run once right after the freeze (w = "
              f"{FROZEN['w']}). Both components' holdout results were already published (v1 and v2 READMEs)."]
    return _evaluate(*HOLDOUT, header, cfg, with_verdict=False)


MULTIPLICITY = ("v2 and v3 verdicts are read on the same forward window, each at its own 5% level, "
                "with no multiplicity correction.")


def forward(cfg=None):
    if _real_today() < FORWARD_OPEN:
        raise RuntimeError(f"forward period is sealed until {FORWARD_OPEN}")
    cfg = cfg or MultiConfig()
    header = [f"FORWARD: the verdict (spec section 7), scored once on or after {FORWARD_OPEN}", MULTIPLICITY]
    return _evaluate(*FORWARD, header, cfg, with_verdict=True)


def _real_today():
    return dt.date.today()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predictor v3 holdout / forward evaluation (spec section 7)")
    ap.add_argument("--period", choices=["holdout", "forward"], required=True)
    ap.add_argument("--today", type=dt.date.fromisoformat, default=None,
                    help="test an earlier date; later than the real date is ignored")
    a = ap.parse_args(argv)
    if a.period == "holdout":
        text = holdout()
        out = Path(OUT) / "holdout.txt"
    else:
        real = _real_today()
        today = min(a.today, real) if a.today else real      # --today can only move earlier, never open the verdict
        if today < FORWARD_OPEN:
            print(data_quality(), end="")
            return 0
        text = forward()
        out = Path(OUT) / "forward.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
