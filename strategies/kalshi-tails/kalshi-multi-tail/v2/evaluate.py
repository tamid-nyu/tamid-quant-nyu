"""
Predictor v2 evaluation after the freeze (spec section 7).

    cd strategies/kalshi-tails/kalshi-multi-tail
    python -m v2.evaluate --period holdout             # -> v2/output/holdout.txt (seen period, not the verdict)
    python -m v2.evaluate --period forward [--today D]  # --today can only move earlier; before 2027-05-12: data-quality counts only

Both periods walk S, P and the frozen v2 (v2.logit_v2.FROZEN) forward over every row up to the period's last
expiry day (training uses earlier rows as usual) and score only the period's days, with the common-row rule of
spec section 6 step 2 (v2 n >= min_rows, v1 P n >= min_cell_n; v1 S on the subset where its own cell n >= 100).
One-sided test of a paired delta: the 95th percentile of v2.scoring.block_bootstrap_delta (4,000 resamples, seed
11, 7-day blocks) is < 0. Forward verdict: WIN only if both delta_P (common rows) and delta_S (S subset) pass,
else "not shown". Primary scope BTC + SPX; the five-market block is exploratory.
"""
import argparse
import dataclasses
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

import multi_tail as mt
from estimators import make_estimator
from multi_tail import V2_FEATURES, MultiConfig, load_features
from v2.logit_v2 import FROZEN
from v2.scoring import block_bootstrap_delta, one_contract_pnl, walk_forward_predict, would_trade
from v2.select import BURN_START, _frame, _ll_table

HOLDOUT = (dt.date(2026, 7, 1), dt.date(2026, 11, 11))
FORWARD = (dt.date(2026, 11, 12), dt.date(2027, 5, 12))
FORWARD_OPEN = dt.date(2027, 5, 12)
PRIMARY = ["BTC", "SPX"]
FIVE = ["BTC", "SPX", "GOLD", "WTI", "NATGAS"]
MDE_DAYS = 181
ESTIMATORS = ["S", "P", "V2"]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FEATURES_PATH = V2_FEATURES
RAW_DIR = HERE.parent / "output" / "raw"


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def one_sided(day, d):
    """Row-weighted mean of d, bootstrap SE and 95th percentile; passes iff the 95th percentile is < 0."""
    day, d = np.asarray(day), np.asarray(d, float)
    if not len(d):
        return dict(delta=float("nan"), se=float("nan"), p95=float("nan"), **{"pass": False}, n=0, D=0)
    boot = block_bootstrap_delta(day, d)
    p95 = float(np.percentile(boot, 95))
    return dict(delta=float(d.mean()), se=float(np.std(boot, ddof=1)), p95=p95, **{"pass": bool(p95 < 0)},
                n=len(d), D=len(np.unique(day)))


def verdict(df):
    """Spec section 7 forward verdict on a scored frame (ll_V2, ll_P, ll_S, day, common, s_subset)."""
    c, s = df[df.common], df[df.s_subset]
    tests = {"P": one_sided(c.day.to_numpy(), (c.ll_V2 - c.ll_P).to_numpy()),
             "S": one_sided(s.day.to_numpy(), (s.ll_V2 - s.ll_S).to_numpy())}
    return ("WIN" if tests["P"]["pass"] and tests["S"]["pass"] else "not shown"), tests


def mde(day, d, days=MDE_DAYS):
    """SE_h = std of the block-bootstrap row-weighted delta, D = distinct days, SE_181 = SE_h sqrt(D / 181),
    MDE = 2.5 SE_181; day_sd = SD of the per-day mean delta (for information)."""
    day, d = np.asarray(day), np.asarray(d, float)
    se_h = float(np.std(block_bootstrap_delta(day, d), ddof=1))
    per_day = pd.Series(d).groupby(day).mean()
    D = len(per_day)
    se = se_h * np.sqrt(D / days)
    return dict(se_h=se_h, D=D, se_181=float(se), mde=float(2.5 * se),
                day_sd=float(per_day.std(ddof=1)) if D > 1 else float("nan"))


# ----------------------------------------------------------------------------
# walk-forward and scoring
# ----------------------------------------------------------------------------
def _cfg(kind, cfg):
    return dataclasses.replace(cfg, estimator=kind, v2_set=FROZEN["set"])


def _check_frozen(est):
    got = dict(set=est.feature_set, lam_beta=est.lam_beta, lam_alpha=est.lam_alpha, refit_days=est.refit_days,
               min_rows=est.min_rows)
    if got != FROZEN:
        raise RuntimeError(f"V2 estimator {got} differs from FROZEN {FROZEN}")


def score(obs, start, end, cfg):
    """Walk S, P, V2 forward over rows with day <= end; mark common rows and the S subset on days in [start, end]."""
    obs = obs[obs.day <= end].reset_index(drop=True)
    preds = {}
    for kind in ESTIMATORS:
        c = _cfg(kind, cfg)
        est = make_estimator(kind, c)
        if kind == "V2":
            _check_frozen(est)
        preds[kind] = walk_forward_predict(obs, est, c)
    df = _frame(preds)
    df["period"] = (df.day >= start) & (df.day <= end)
    elig = (df.period & np.isfinite(df.p_P) & (df.n_P >= cfg.min_cell_n)
            & np.isfinite(df.p_V2) & (df.n_V2 >= FROZEN["min_rows"]))
    df["common"] = elig
    df["s_subset"] = elig & np.isfinite(df.p_S) & (df.n_S >= cfg.min_cell_n)
    return df


def _test_line(label, g, a, b):
    if not len(g):
        return f"  {label:<22}{0:6d}"
    t = one_sided(g.day.to_numpy(), (g[f"ll_{a}"] - g[f"ll_{b}"]).to_numpy())
    return (f"  {label:<22}{t['n']:6d}{t['D']:6d}{t['delta']:+11.5f}{t['se']:10.5f}{t['p95']:+11.5f}"
            f"  {'yes' if t['pass'] else 'no'}")


def report(df, cfg, mde_block=True):
    c, s = df[df.common], df[df.s_subset]
    names = ["V2", "P", "mid"]
    L = ["Row counts per market (live = warm tail candidates on the period's days)",
         f"  {'market':<8}{'live':>8}{'common':>8}{'S subset':>10}{'days':>6}"]
    for m, g in df[df.period].groupby("market"):
        L.append(f"  {m:<8}{len(g):8d}{int(g.common.sum()):8d}{int(g.s_subset.sum()):10d}"
                 f"{g[g.common].day.nunique():6d}")
    L.append(f"  {'total':<8}{int(df.period.sum()):8d}{len(c):8d}{len(s):10d}{c.day.nunique():6d}")
    if not len(c):
        return L + ["", "no common rows"]
    L += ["", f"Mean log-loss on common rows (V2 = frozen set {FROZEN['set']})"] + ["  " + x for x in _ll_table(c, names)]
    L += ["", "by market"] + ["  " + x for x in _ll_table(c, names, "market")]
    L += ["", "by H"] + ["  " + x for x in _ll_table(c, names, "H")]
    L += ["", "v1 S subset (common rows where v1 S's own cell n >= 100), labeled as that subset"]
    if len(s):
        L += ["  " + x for x in _ll_table(s, ["S"] + names)]
        L += ["  " + x for x in _ll_table(s, ["S"] + names, "market")[1:]]

    L += ["", "Paired delta = LL(a) - LL(b), same rows; one-sided test passes iff the bootstrap 95th pct < 0",
          f"  {'pair':<22}{'rows':>6}{'days':>6}{'delta':>11}{'SE':>10}{'95th pct':>11}  p95 < 0"]
    L.append(_test_line("V2 - P", c, "V2", "P"))
    L.append(_test_line("V2 - S (S subset)", s, "V2", "S"))
    L.append(_test_line("V2 - mid", c, "V2", "mid"))
    L.append(_test_line("P - mid", c, "P", "mid"))
    L.append(_test_line("P - S (S subset)", s, "P", "S"))

    if mde_block:
        L += ["", f"Minimum detectable effect for a {MDE_DAYS}-day forward window: SE_h = std of the bootstrap "
                  f"delta, D = distinct days,",
              f"SE_{MDE_DAYS} = SE_h * sqrt(D / {MDE_DAYS}), MDE = 2.5 * SE_{MDE_DAYS}; per-day SD = SD of the "
              "per-day mean delta (information only)",
              f"  {'comparator':<20}{'D':>5}{'SE_h':>10}{'SE_' + str(MDE_DAYS):>10}{'MDE':>10}{'per-day SD':>12}"]
        for label, g, b in (("P (common rows)", c, "P"), ("S (S subset)", s, "S")):
            if len(g):
                r = mde(g.day.to_numpy(), (g.ll_V2 - g[f"ll_{b}"]).to_numpy())
                L.append(f"  {label:<20}{r['D']:5d}{r['se_h']:10.5f}{r['se_181']:10.5f}{r['mde']:10.5f}"
                         f"{r['day_sd']:12.5f}")

    wv, wp = would_trade(c.assign(p=c.p_V2), cfg), would_trade(c.assign(p=c.p_P), cfg)
    w = c[wv | wp]
    L += ["", f"Would-trade secondary (report only): common rows where v2 or v1 P would trade (net edge >= "
              f"{cfg.min_edge} after slip {cfg.slip} and fees, by best_side, valid fill quote)",
          f"  {'rows':>6}{'days':>6}{'delta_P':>11}{'SE':>10}{'95th pct':>11}{'v2 n/pnl':>17}{'P n/pnl':>17}"]
    if len(w):
        t = one_sided(w.day.to_numpy(), (w.ll_V2 - w.ll_P).to_numpy())
        pv, pp = one_contract_pnl(w.assign(p=w.p_V2), cfg), one_contract_pnl(w.assign(p=w.p_P), cfg)
        L.append(f"  {t['n']:6d}{t['D']:6d}{t['delta']:+11.5f}{t['se']:10.5f}{t['p95']:+11.5f}"
                 f"{int(wv.sum()):8d} ${pv.sum():+7.2f}{int(wp.sum()):8d} ${pp.sum():+7.2f}")
    else:
        L.append(f"  {0:6d}")
    L.append("  one-contract P&L at the fill quote, fee rounded up to the cent")
    return L


def sharpe_block(obs, start, end, cfg):
    """Sharpe of v2 and v1 P through multi_tail.run (frozen sizer), on the period's days, and paired_diff_ci."""
    obs = obs[obs.day <= end].reset_index(drop=True)
    days = sorted(obs.day.unique())
    w = {}
    for kind in ("V2", "P"):
        _, daily, _ = mt.run(obs, _cfg(kind, cfg))
        w[kind] = mt._window(daily, days, start, end)
    L = [f"Sharpe through the frozen sizer (multi_tail.run: c = {cfg.cap_m:.0%}, k = {cfg.k}, C = {cfg.cap_k:.0%}, "
         f"${cfg.bankroll:,.0f}); daily returns on {len(w['P'])} period days, days without trades count as 0",
         f"  {'model':<8}{'Sharpe':>8}  {'95% CI':<16}{'return':>9}{'trades':>8}"]
    for kind in ("V2", "P"):
        r = w[kind]
        lo, hi = mt.boot_sharpe_ci(r.ret)
        L.append(f"  {kind:<8}{mt.sharpe(r.ret):8.2f}  [{lo:5.2f}, {hi:5.2f}]  {r.ret.sum() * 100:+8.2f}%"
                 f"{int(r.n_trades.sum()):8d}")
    d, lo, hi = mt.paired_diff_ci(w["V2"].ret.values, w["P"].ret.values)
    L.append(f"  Sharpe(V2) - Sharpe(P) = {d:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}] (paired_diff_ci, same days)")
    return L


def _evaluate(start, end, header, cfg, with_verdict):
    t0 = time.time()
    obs5 = load_features(FIVE, FEATURES_PATH)
    obs5 = obs5[(obs5.day >= BURN_START) & (obs5.day <= end)].reset_index(drop=True)
    prim = obs5[obs5.market.isin(PRIMARY)].reset_index(drop=True)
    df = score(prim, start, end, cfg)
    L = list(header) + [
        f"Frozen v2: {FROZEN}",
        f"scored expiry days {start} -> {end}; walk-forward over all rows from {BURN_START} up to {end}",
        f"common rows: period rows where v2 has n >= {FROZEN['min_rows']} and v1 P has n >= {cfg.min_cell_n}; "
        "log-loss uses p clipped to [0.001, 0.999]",
        "bootstrap: 4,000 moving-block resamples (7-day blocks, seed 11), row-weighted delta inside each", ""]
    if with_verdict:
        label, tests = verdict(df)
        L += ["PRIMARY VERDICT (BTC + SPX): " + label,
              f"  delta_P = {tests['P']['delta']:+.5f} (95th pct {tests['P']['p95']:+.5f}, "
              f"{'pass' if tests['P']['pass'] else 'fail'}), "
              f"delta_S = {tests['S']['delta']:+.5f} (95th pct {tests['S']['p95']:+.5f}, "
              f"{'pass' if tests['S']['pass'] else 'fail'}); WIN only if both pass", ""]
    L += ["=== PRIMARY: BTC + SPX ==="] + report(df, cfg, mde_block=not with_verdict)
    L += [""] + sharpe_block(prim, start, end, cfg)
    df5 = score(obs5, start, end, cfg)
    L += ["", f"=== EXPLORATORY: five markets ({' + '.join(FIVE)}), no decision attached ==="]
    L += report(df5, cfg, mde_block=False) + [""] + sharpe_block(obs5, start, end, cfg)
    L += ["", f"runtime {time.time() - t0:.0f}s"]
    return "\n".join(L) + "\n"


def holdout(cfg=None):
    cfg = cfg or MultiConfig()
    header = ["HOLDOUT: seen period, not the verdict",
              "Predictor v2 holdout evaluation (spec section 7), run once right after the freeze.",
              "Disclosure: a Task 6 smoke run had already printed a holdout Sharpe for v2 with set F3 before "
              "selection; nothing was changed in response (v2/README.md)."]
    return _evaluate(*HOLDOUT, header, cfg, with_verdict=False)


def forward(cfg=None):
    cfg = cfg or MultiConfig()
    header = ["FORWARD: the verdict (spec section 7), scored once on or after 2027-05-12"]
    return _evaluate(*FORWARD, header, cfg, with_verdict=True)


def data_quality():
    """Row and day counts per market on forward expiry days, and the failed-day files. Nothing is scored."""
    L = [f"Forward data-quality check (no scoring before {FORWARD_OPEN}); expiry days >= {FORWARD[0]}"]
    if Path(FEATURES_PATH).exists():
        f = pd.read_csv(FEATURES_PATH, usecols=["market", "day"])
        f = f[pd.to_datetime(f.day).dt.date >= FORWARD[0]]
        L += [f"features: {FEATURES_PATH}", f"  {'market':<8}{'rows':>8}{'days':>6}"]
        for m in FIVE:
            g = f[f.market == m]
            L.append(f"  {m:<8}{len(g):8d}{g.day.nunique():6d}")
    else:
        L.append(f"features file missing: {FEATURES_PATH} (run python -m v2.features)")
    L.append(f"failed-day files under {RAW_DIR}:")
    files = sorted(Path(RAW_DIR).glob("*/failed_days.txt")) if Path(RAW_DIR).exists() else []
    if not files:
        L.append("  none")
    for p in files:
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        L.append(f"  {p.parent.name}: {len(lines)} line(s)" + "".join(f"\n    {x}" for x in lines))
    return "\n".join(L) + "\n"


def _real_today():
    return dt.date.today()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predictor v2 holdout / forward evaluation (spec section 7)")
    ap.add_argument("--period", choices=["holdout", "forward"], required=True)
    ap.add_argument("--today", type=dt.date.fromisoformat, default=None, help="test an earlier date; later than the real date is ignored")
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
