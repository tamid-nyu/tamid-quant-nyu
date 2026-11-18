#!/usr/bin/env python3
"""
Multi-market Kalshi tail strategy with k-worst portfolio sizing
===============================================================
Trades the BTC tail rule (mid < 10% or > 90%, 6h and 3h before expiry) on several Kalshi "above K" ladders at once.
Each decision round is sized so that no market's worst settlement loses more than cap_m of bankroll and the k worst
markets together lose no more than cap_k (multi_sizer.py). Edge comes from estimator S (each market alone) or
P (each market shrunk toward the cross-market pooled rate), walk-forward on prior expiry days only.

Spec: docs/superpowers/specs/2026-11-12-multi-market-tail-design.md
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "btc-kalshi-digital"))
import tail_strategy as ts  # noqa: E402
from tail_sizer import order_fee  # noqa: E402

from estimators import make_estimator  # noqa: E402
from multi_sizer import book_losses, size_multi, topk_sum  # noqa: E402

OUT_DIR = HERE / "output"
CHECK = ("2026-01-01", "2026-06-30")
HOLDOUT = ("2026-07-01", "2026-11-11")
TRADE_COLS = ts.TRADE_COLS + ["market", "group"]
DAILY_COLS = ["day", "pnl", "ret", "n_trades", "n_wins", "n_markets", "deployed",
              "assumed_max_market", "assumed_topk", "assumed_all",
              "realized_max_market", "realized_topk", "realized_all", "breach_m", "breach_k"]
MD_COLS = ["day", "market", "pnl", "n_trades", "deployed"]


@dataclass
class MultiConfig:
    bankroll: float = 1000.0
    cap_m: float = 0.03
    k: int = 2
    cap_k: float = 0.04
    q_max: int = ts.Q_MAX
    min_edge: float = ts.MIN_EDGE
    slip: float = ts.SLIP
    n0: float = ts.N0
    n1: float = 100.0
    min_cell_n: int = ts.MIN_CELL_N
    warmup_days: int = ts.WARMUP_DAYS
    fee_rate: float = ts.FEE_RATE
    estimator: str = "S"
    hours: tuple = ts.HOURS
    total_cap: float = None        # option-1 baseline: every market's worst case summed against this one cap

    @property
    def k_eff(self):
        return 10 ** 6 if self.total_cap is not None else self.k

    @property
    def cap_k_frac(self):
        return self.total_cap if self.total_cap is not None else self.cap_k


def _candidates(rows, est, cfg, day):
    out = []
    for r in rows.itertuples(index=False):
        f, n = est.estimate(r.market, r.H, r.mid)
        if n < cfg.min_cell_n or not np.isfinite(f):
            continue
        side, price, edge = ts.best_side(r.bid, r.ask, f, cfg)
        fill = r.ask_f if side == 1 else r.bid_f
        if (not np.isfinite(fill)) or (fill >= 1.0 if side == 1 else fill <= 0.0):
            continue
        if edge >= cfg.min_edge:
            out.append(dict(day=day, ticker=r.ticker, H=int(r.H), side=side, size_price=price, edge=edge, f=f, y=r.y,
                            floor=r.floor, bid_f=r.bid_f, ask_f=r.ask_f, market=r.market, group=r.group))
    return out


def run(obs, cfg):
    """Walk forward over expiry days. Returns (trades, daily, market_daily)."""
    est = make_estimator(cfg.estimator, cfg)
    days_seen = {}
    trades, daily, mdaily = [], [], []
    B = cfg.bankroll
    cap_m, cap_k = cfg.cap_m * B, cfg.cap_k_frac * B
    for day, g in obs.groupby("day", sort=True):
        g = g[g.H.isin(ts.HOURS)]
        tails = ts.tail_candidates(g)
        live = tails[tails.H.isin(cfg.hours) & tails.market.map(lambda m: days_seen.get(m, 0) >= cfg.warmup_days)]
        held = {}
        assumed = {}
        for t_dec in sorted(live.decision_ts.unique()):
            cands = _candidates(live[live.decision_ts == t_dec], est, cfg, day)
            if not cands:
                continue
            by_m = {}
            for c in cands:
                by_m.setdefault(c["market"], []).append(c)
            mkts = sorted(set(by_m) | set(held))
            books, grids = [], {}
            for m in mkts:
                hm, cm = held.get(m, []), by_m.get(m, [])
                grid = np.unique([p["floor"] for p in hm + cm])
                grids[m] = grid
                books.append({"P": ts._book_payoffs(grid, cm, "size_price", cfg.fee_rate),
                              "base": ts._held_vector(grid, hm, "fill", cfg.fee_rate),
                              "edges": np.array([c["edge"] for c in cm], float)})
            # 1c per candidate order reserved against each cap, as in the BTC base (fees round up to the cent)
            caps = np.array([cap_m - 0.01 * len(by_m.get(m, [])) for m in mkts])
            qs = size_multi(books, caps, cap_k - 0.01 * len(cands), cfg.k_eff, cfg.q_max)
            for m, b, q in zip(mkts, books, qs):
                placed = []
                for c, qi in zip(by_m.get(m, []), q):
                    if qi <= 0:
                        continue
                    fill = c["ask_f"] if c["side"] == 1 else c["bid_f"]
                    fe = order_fee(int(qi), fill, cfg.fee_rate)
                    placed.append(dict(c, q=int(qi), fill=float(fill), fee=fe,
                                       pnl=qi * c["side"] * (c["y"] - fill) - fe))
                # the LP's view after this round, for every market in the book: held at fill + new at sizing price
                assumed[m] = max(0.0, -float((b["base"] + ts._held_vector(grids[m], placed, "size_price",
                                                                           cfg.fee_rate)).min()))
                if placed:
                    held.setdefault(m, []).extend(placed)
        realized = {}
        for m, hm in held.items():
            grid = np.unique([p["floor"] for p in hm])
            realized[m] = max(0.0, -float(ts._held_vector(grid, hm, "fill", cfg.fee_rate).min()))
        allpos = [p for hm in held.values() for p in hm]
        trades.extend(allpos)
        pnl = float(sum(p["pnl"] for p in allpos))
        dep = lambda ps: float(sum(p["q"] * (p["fill"] if p["side"] == 1 else 1 - p["fill"]) + p["fee"] for p in ps))
        av, rv = np.array(list(assumed.values()) or [0.0]), np.array(list(realized.values()) or [0.0])
        daily.append(dict(day=day, pnl=pnl, ret=pnl / B, n_trades=len(allpos), n_wins=sum(p["pnl"] > 0 for p in allpos),
                          n_markets=len(held), deployed=dep(allpos),
                          assumed_max_market=float(av.max()), assumed_topk=topk_sum(av, cfg.k), assumed_all=float(av.sum()),
                          realized_max_market=float(rv.max()), realized_topk=topk_sum(rv, cfg.k),
                          realized_all=float(rv.sum()), breach_m=bool(rv.max() > cap_m + 1e-9),
                          breach_k=bool(topk_sum(rv, cfg.k_eff) > cap_k + 1e-9)))
        for m, hm in held.items():
            mdaily.append(dict(day=day, market=m, pnl=float(sum(p["pnl"] for p in hm)), n_trades=len(hm),
                               deployed=dep(hm)))
        est.update(tails)
        for m in g.market.unique():
            days_seen[m] = days_seen.get(m, 0) + 1
    tr = pd.DataFrame(trades, columns=TRADE_COLS) if trades else pd.DataFrame(columns=TRADE_COLS)
    return (tr[TRADE_COLS].reset_index(drop=True), pd.DataFrame(daily, columns=DAILY_COLS),
            pd.DataFrame(mdaily, columns=MD_COLS))


# ----------------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------------
def sharpe(ret):
    r = np.asarray(ret, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


def boot_sharpe_ci(ret, n=2000, seed=11):
    r = np.asarray(ret, float)
    if len(r) < 3:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(r), (n, len(r)))
    s = np.array([sharpe(r[i]) for i in idx])
    return tuple(float(x) for x in np.percentile(s, [2.5, 97.5]))


def paired_diff_ci(a, b, n=2000, seed=11):
    """Sharpe(a) - Sharpe(b) and a bootstrap CI resampling the same days for both."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = sharpe(a) - sharpe(b)
    if len(a) < 3:
        return d, float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    s = np.array([sharpe(a[i]) - sharpe(b[i]) for i in idx])
    lo, hi = np.percentile(s, [2.5, 97.5])
    return d, float(lo), float(hi)


def check_args(k, cap_m, cap_k):
    if k < 1:
        raise SystemExit("--k must be >= 1")
    if cap_k <= 0 or cap_m <= 0:
        raise SystemExit("--cap and --cap-k must be > 0")
    if cap_k >= k * cap_m - 1e-12:
        print(f"warning: cap-k {cap_k:.3f} >= k x cap {k * cap_m:.3f}, so the k-worst cap never binds")


def _window(daily, days, start, end):
    d = daily.set_index("day").reindex(days)
    d[["pnl", "ret", "n_trades", "n_wins", "deployed", "assumed_all", "realized_all"]] = \
        d[["pnl", "ret", "n_trades", "n_wins", "deployed", "assumed_all", "realized_all"]].fillna(0.0)
    d[["breach_m", "breach_k"]] = d[["breach_m", "breach_k"]].astype("boolean").fillna(False).astype(bool)
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return d[(d.index >= lo) & (d.index <= hi)]


def period_row(name, daily, ref_daily, days, start, end, bankroll):
    w, ref = _window(daily, days, start, end), _window(ref_daily, days, start, end)
    if w.empty:
        return dict(model=name)
    cum = w.ret.cumsum()
    dd = float((cum - np.maximum.accumulate(np.maximum(cum, 0.0))).min())
    lo, hi = boot_sharpe_ci(w.ret)
    d, dlo, dhi = paired_diff_ci(w.ret.values, ref.ret.values)
    traded = w.n_trades > 0
    return dict(model=name, sharpe=sharpe(w.ret), sharpe_lo=lo, sharpe_hi=hi, diff=d, diff_lo=dlo, diff_hi=dhi,
                ret=float(cum.iloc[-1]), max_dd=dd, worst_day=float(w.ret.min()), trades=int(w.n_trades.sum()),
                wins_per_day=float(w.n_wins.mean()), traded_days_per_week=float(traded.mean() * 7),
                max_assumed_all=float(w.assumed_all.max() / bankroll), breach_m=int(w.breach_m.sum()),
                breach_k=int(w.breach_k.sum()))


def _fmt_table(rows, title):
    out = [title, f"{'model':<26}| Sharpe [95% CI]          diff vs BTC [95% CI]   | return  maxDD  worst | trades wins/d "
                  f"days/wk | max all-mkt worst  breaches c/C"]
    for r in rows:
        if "sharpe" not in r:
            out.append(f"{r['model']:<26}| no days")
            continue
        out.append(f"{r['model']:<26}| {r['sharpe']:5.2f} [{r['sharpe_lo']:5.2f}, {r['sharpe_hi']:5.2f}]   "
                   f"{r['diff']:+5.2f} [{r['diff_lo']:+5.2f}, {r['diff_hi']:+5.2f}] | {r['ret'] * 100:+6.2f}% "
                   f"{r['max_dd'] * 100:6.2f}% {r['worst_day'] * 100:6.2f}% | {r['trades']:5d} {r['wins_per_day']:6.2f} "
                   f"{r['traded_days_per_week']:6.2f} | {r['max_assumed_all'] * 100:13.2f}%   {r['breach_m']:3d}/{r['breach_k']:<3d}")
    return out


def market_report(md, days, start, end, bankroll):
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    m = md[(md.day >= lo) & (md.day <= hi)]
    if m.empty:
        return ["  no market trades"]
    out = ["  per-market P&L and capital:"]
    for k, g in m.groupby("market"):
        out.append(f"    {k:<7} pnl ${g.pnl.sum():+9.2f}  trades {int(g.n_trades.sum()):5d}  traded days {len(g):4d}  "
                   f"mean capital ${g.deployed.mean():7.2f}")
    piv = m.pivot_table(index="day", columns="market", values="pnl", aggfunc="sum")
    if piv.shape[1] > 1:
        out.append("  correlation of daily market P&L (each pair over days both traded; n = shared days):")
        c, n = pnl_corr(m), piv.notna().astype(int).T @ piv.notna().astype(int)
        out += ["    " + line for line in c.round(2).astype(str).add(" (n=").add(n.astype(str)).add(")")
                .to_string().splitlines()]
        losing = (piv < -0.01 * bankroll).sum(axis=1)
        out.append(f"  stress days (any market lost > 1% of bankroll): {int((losing > 0).sum())}; by number of markets "
                   "losing > 1% that day: " + ", ".join(f"{int(k)} mkt x{int(v)}" for k, v in
                                                       losing[losing > 0].value_counts().sort_index().items()))
    return out


def pnl_corr(md):
    """Pairwise correlation of per-market daily P&L, each pair over the days both markets traded."""
    return md.pivot_table(index="day", columns="market", values="pnl", aggfunc="sum").corr(min_periods=2)


def enabled_markets(min_days=None):
    """Enabled markets whose observations cover at least min_days expiry days (default: the spec's 150)."""
    import multi_obs
    from multi_pull import load_markets
    min_days = multi_obs.MIN_HISTORY_DAYS if min_days is None else min_days
    gate_f = multi_obs.OBS_DIR / "history_gate.csv"
    gate = pd.read_csv(gate_f).set_index("market") if gate_f.exists() else None
    keys = []
    for k, m in load_markets().items():
        if not m["enabled"]:
            continue
        if k != "BTC" and (gate is None or k not in gate.index or gate.loc[k, "expiry_days"] < min_days):
            print(f"{k}: skipped (no observations or fewer than {min_days} expiry days)")
            continue
        keys.append(k)
    return keys


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--markets", nargs="*", default=None, help="market keys (default: enabled markets passing the gate)")
    ap.add_argument("--estimator", choices=["S", "P"], default="S")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--cap", type=float, default=0.03, help="per-market worst-case cap c, fraction of bankroll")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cap-k", type=float, default=0.04, help="cap C on the k worst markets together")
    ap.add_argument("--q-max", type=int, default=ts.Q_MAX)
    ap.add_argument("--min-edge", type=float, default=ts.MIN_EDGE)
    ap.add_argument("--slip", type=float, default=ts.SLIP)
    ap.add_argument("--n0", type=float, default=ts.N0)
    ap.add_argument("--n1", type=float, default=100.0)
    ap.add_argument("--warmup-days", type=int, default=ts.WARMUP_DAYS)
    ap.add_argument("--min-history-days", type=int, default=None,
                    help="drop markets with fewer expiry days (default 150, spec section 2)")
    ap.add_argument("--all", action="store_true", help="S, P, BTC-only, option-1 baseline and k sensitivity")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    check_args(a.k, a.cap, a.cap_k)

    from multi_obs import load_all
    keys = a.markets or enabled_markets(a.min_history_days)
    obs = load_all(keys)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = dict(bankroll=a.bankroll, cap_m=a.cap, k=a.k, cap_k=a.cap_k, q_max=a.q_max, min_edge=a.min_edge,
                slip=a.slip, n0=a.n0, n1=a.n1, warmup_days=a.warmup_days)
    btc = obs[obs.market == "BTC"]
    runs = {}
    if a.all:
        runs["S multi"] = (obs, MultiConfig(**dict(base, estimator="S")))
        runs["P multi"] = (obs, MultiConfig(**dict(base, estimator="P")))
        runs["option 1 (S, total 6%)"] = (obs, MultiConfig(**dict(base, estimator="S", total_cap=0.06)))
        for kk, ck in ((1, 0.03), (3, 0.05)):
            for e in ("S", "P"):
                runs[f"{e} k={kk} C={ck:.0%}"] = (obs, MultiConfig(**dict(base, estimator=e, k=kk, cap_k=ck)))
    else:
        runs[f"{a.estimator} multi"] = (obs, MultiConfig(**dict(base, estimator=a.estimator)))
    runs["BTC only (Model 0)"] = (btc, MultiConfig(**dict(base, k=1, cap_k=a.cap, estimator="S")))
    res = {name: run(o, cfg) for name, (o, cfg) in runs.items()}

    days = sorted(obs.day.unique())
    ref = res["BTC only (Model 0)"][1]
    text = [f"Multi-market tail strategy: markets {', '.join(keys)}; {len(obs)} observations, "
            f"{len(days)} days ({days[0]} -> {days[-1]})",
            f"bankroll ${a.bankroll:,.0f}; per-market cap c = {a.cap:.1%}, k = {a.k}, k-worst cap C = {a.cap_k:.1%}; "
            f"Sharpe = mean/sd of daily returns x sqrt(365), days without trades count as 0",
            "obs per market: " + ", ".join(f"{k} {n} ({obs[obs.market == k].day.min()} -> {obs[obs.market == k].day.max()})"
                                          for k, n in obs.market.value_counts().items()), ""]
    for label, (s0, e0) in (("CHECK PERIOD", CHECK), ("HOLDOUT (headline)", HOLDOUT)):
        rows = [period_row(n, r[1], ref, days, s0, e0, a.bankroll) for n, r in res.items()]
        text += _fmt_table(rows, f"=== {label} {s0} -> {e0} ===") + [""]
        pd.DataFrame(rows).to_csv(out / f"summary_{label.split()[0].lower()}.csv", index=False)
    for n in [x for x in res if x.endswith("multi")]:
        text += [f"--- {n}: holdout detail ---"] + market_report(res[n][2], days, *HOLDOUT, a.bankroll)
        text += [f"--- {n}: full-history detail ---"] + market_report(res[n][2], days, days[0], days[-1], a.bankroll)
        d = res[n][1]
        text += [f"  all-markets assumed worst case: max {d.assumed_all.max() / a.bankroll:.2%}, mean on traded days "
                 f"{d.assumed_all[d.n_trades > 0].mean() / a.bankroll:.2%}; realized max {d.realized_all.max() / a.bankroll:.2%}", ""]
    text += ["Notes: S, P, option 1 and the k runs were all fixed before running; several are compared, so one beating "
             "BTC-only on the holdout is weak evidence on its own."]
    report = "\n".join(text)
    print(report)
    (out / "summary.txt").write_text(report + "\n")
    for n in [x for x in res if x.endswith("multi")]:
        tag = n.split()[0]
        tr, dy, md = res[n]
        tr.to_csv(out / f"trades_{tag}.csv", index=False)
        dy.to_csv(out / f"daily_{tag}.csv", index=False)
        md.to_csv(out / f"market_daily_{tag}.csv", index=False)

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for n in [x for x in res if x.endswith("multi") or x.startswith("BTC") or x.startswith("option")]:
            w = _window(res[n][1], days, *HOLDOUT)
            ax.plot(pd.to_datetime(w.index), a.bankroll + w.pnl.cumsum(), label=n)
        ax.set_title("Account value from $1,000, holdout"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "equity.png", dpi=130)
        multis = [x for x in res if x.endswith("multi")]
        fig, axs = plt.subplots(1, len(multis), figsize=(6 * len(multis), 4.2), squeeze=False)
        for axx, n in zip(axs[0], multis):
            piv = res[n][2].pivot_table(index="day", columns="market", values="pnl", aggfunc="sum")
            piv = piv.reindex(days).fillna(0.0).cumsum()
            for c in piv.columns:
                axx.plot(pd.to_datetime(piv.index), piv[c], label=c)
            axx.set_title(f"Cumulative P&L by market, {n}"); axx.legend()
        fig.tight_layout(); fig.savefig(out / "contribution.png", dpi=130)
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
