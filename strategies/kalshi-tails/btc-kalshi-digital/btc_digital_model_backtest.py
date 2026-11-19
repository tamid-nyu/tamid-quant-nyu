#!/usr/bin/env python3
"""
Bitcoin: Kalshi contract price vs model probability (with optional delta hedge)
================================================================================
A Kalshi "BTC above $X at 5pm ET" contract is a digital option. Given spot S, time to expiry and a volatility
forecast, we estimate P(BTC finishes above X) and compare it with the contract's price.

Pipeline (reads backtests/output/btc_pm/ written by btc_pm_data_pull.py):
  1. Model probability: normal digital price, or a filtered-historical ('fhs') version that uses the empirical
     distribution of past standardized H-hour returns (so fat tails and short-horizon kurtosis are handled
     from data, not assumed). Volatility comes from
       rv   = trailing 168h realized vol
       ewma = exponentially weighted hourly variance
       garch= GARCH(1,1) refit every 14 days on the trailing 90 days (own MLE, no extra packages)
     Every input at time t uses data up to t only.
  2. Calibration test: does the model predict outcomes better than the market price? (Brier score)
  3. Trading rule: at H hours before expiry, buy YES if model - ask > edge, sell YES if bid - model > edge.
     Fill at the NEXT hour's bid/ask (EXEC_LAG), pay the Kalshi fee, hold to settlement.
  4. Optional delta hedge: hold -side * delta BTC (hourly rebalanced, spot or perp), pay hedge costs.
     delta = d(model prob)/dS  ~  pdf(z) / (S * sigma * sqrt(tau))
  5. Contracts that expire together are highly correlated, so confidence intervals resample whole expiry days.

Usage:  python btc_digital_model_backtest.py
        python btc_digital_model_backtest.py --h 6 --edge 0.04 --model garch --dist fhs --hedge-bps 5
"""
import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "output" / "btc_pm"
OUT_DIR = HERE / "output" / "btc_pm_results"

# ----------------------------------------------------------------------------
# CONFIG (primary settings chosen before looking at results)
# ----------------------------------------------------------------------------
H_PRIMARY = 6                 # hours before expiry at which we decide
H_GRID = [3, 6, 12]
EDGE = 0.04                   # required edge (model vs executable price), probability points
EDGE_GRID = [0.02, 0.04, 0.06, 0.10]
EXEC_LAG = 1                  # fill one hour after the signal
MAX_SPREAD = 0.06             # Kalshi bid-ask must be <= 6c at signal time
KALSHI_FEE_RATE = 0.07        # taker fee 0.07*P*(1-P) per contract (ASSUMPTION - check Kalshi's crypto fee schedule)
HEDGE_BPS = 5.0               # one-way cost of trading the hedge, in basis points of notional (ASSUMPTION)
MODELS = ["rv", "ewma", "garch"]
DISTS = ["normal", "fhs"]
PRIMARY_MODEL, PRIMARY_DIST = "ewma", "fhs"
FHS_DAYS = 90               # trailing window of standardized returns for the fhs distribution
FHS_BW = 0.30               # kernel bandwidth (in standard deviations) used to smooth the empirical CDF
HEDGE_BPS_GRID = [0.0, 2.0, 5.0, 10.0]
IS_FRAC = 0.6                 # first 60% of expiry days = in-sample, rest = out-of-sample
N_BOOT = 4000
SEED = 11
HOURS_PER_YEAR = 8760.0


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def load(data_dir):
    mk = pd.read_csv(data_dir / "btc_kalshi_markets.csv", parse_dates=["expiry"])
    hr = pd.read_csv(data_dir / "btc_kalshi_hourly.csv", parse_dates=["ts"])
    sp = pd.read_csv(data_dir / "btc_spot_hourly.csv", parse_dates=["time"])
    mk["expiry"] = pd.to_datetime(mk.expiry, utc=True)
    hr["ts"] = pd.to_datetime(hr.ts, utc=True)
    sp["time"] = pd.to_datetime(sp.time, utc=True)
    # spot close "as of t" = close of the hourly bar that ends at t
    S = pd.Series(sp.close.values, index=sp.time + pd.Timedelta(hours=1)).sort_index()
    S = S[~S.index.duplicated()]
    mk = mk[mk.result.isin(["yes", "no"])].copy()
    mk["y"] = (mk.result == "yes").astype(float)
    book = {tk: g.drop_duplicates("ts").set_index("ts").sort_index()[["bid", "ask"]] for tk, g in hr.groupby("ticker")}
    return mk.reset_index(drop=True), book, S


# ----------------------------------------------------------------------------
# VOLATILITY MODELS  (each returns total variance of the log return over the next h hours, using data <= t)
# ----------------------------------------------------------------------------
class VolModels:
    def __init__(self, S):
        self.S = S
        # reindex to a regular hourly grid; returns over missing hours are NaN (dropped by rolling windows)
        full = S.reindex(pd.date_range(S.index.min(), S.index.max(), freq="1h"))
        self.logS = np.log(full)
        self.r = self.logS.diff()
        self.idx = full.index
        self._z = {}
        r = self.r
        self.rv = (r.rolling(168, min_periods=120).std() ** 2).rename("rv")                 # hourly variance
        self.ewma = (r.fillna(0) ** 2).ewm(alpha=0.03, adjust=False).mean().rename("ewma")  # ~33h effective memory
        self._garch = None

    # ---- GARCH(1,1) with own MLE, refit every 14 days on trailing 90 days, filtered forward (causal)
    @staticmethod
    def _nll(params, x):
        w, a, b = params
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9995:
            return 1e12
        v = np.empty(len(x))
        v[0] = x.var()
        for i in range(1, len(x)):
            v[i] = w + a * x[i - 1] ** 2 + b * v[i - 1]
        return 0.5 * np.sum(np.log(v) + x ** 2 / v)

    def _fit_garch(self, window):
        x = (window.dropna().values) * 100.0          # percent returns for numerical stability
        if len(x) < 500:
            return None
        v0 = x.var()
        res = optimize.minimize(self._nll, [0.05 * v0, 0.08, 0.88], args=(x,), method="Nelder-Mead",
                                options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-4})
        w, a, b = res.x
        if not (w > 0 and a >= 0 and b >= 0 and a + b < 0.9995):
            return (0.05 * v0, 0.08, 0.88)
        return (w, a, b)

    def build_garch(self):
        r = self.r
        n = len(r)
        sig2_next = np.full(n, np.nan)       # forecast of next-hour variance (decimal^2), known at t
        phi = np.full(n, np.nan)
        vbar = np.full(n, np.nan)
        W, STEP = 24 * 90, 24 * 14
        start = W
        for seg in range(start, n, STEP):
            params = self._fit_garch(r.iloc[seg - W: seg])
            if params is None:
                continue
            w, a, b = params
            vb = w / (1 - a - b)
            v = r.iloc[seg - W: seg].dropna().var() * 1e4
            for i in range(seg, min(seg + STEP, n)):
                x = r.iloc[i]
                x = 0.0 if np.isnan(x) else x * 100.0
                v = w + a * x * x + b * v          # variance for hour i+1 given data through i
                sig2_next[i] = v / 1e4
                phi[i] = a + b
                vbar[i] = vb / 1e4
        self._garch = (pd.Series(sig2_next, self.idx), pd.Series(phi, self.idx), pd.Series(vbar, self.idx))

    def tv_array(self, model, h):
        """Total-variance forecast over the next h hours, for every hour on the grid (causal)."""
        if model == "rv":
            return self.rv * h
        if model == "ewma":
            return self.ewma * h
        if self._garch is None:
            self.build_garch()
        s2, ph, vb = self._garch
        return h * vb + (s2 - vb) * (1 - ph ** h) / (1 - ph)

    def zwin(self, model, h, t):
        """Standardized h-hour returns that were fully known by t (window of FHS_DAYS)."""
        key = (model, h)
        if key not in self._z:
            tv = self.tv_array(model, h)
            ret = self.logS.shift(-h) - self.logS
            z = (ret + 0.5 * tv) / np.sqrt(tv)
            z.index = z.index + pd.Timedelta(hours=h)            # known only at s+h
            self._z[key] = z.replace([np.inf, -np.inf], np.nan).dropna()
        Z = self._z[key]
        return Z.loc[t - pd.Timedelta(days=FHS_DAYS): t].values

    def total_var(self, model, t, h):
        """Variance of ln(S_{t+h}/S_t) forecast at time t (t is a timestamp on the hourly grid)."""
        if t not in self.idx:
            return np.nan
        if model == "rv":
            v = self.rv.get(t, np.nan)
            return v * h
        if model == "ewma":
            return self.ewma.get(t, np.nan) * h
        if model == "garch":
            if self._garch is None:
                self.build_garch()
            s2, ph, vb = self._garch[0].get(t, np.nan), self._garch[1].get(t, np.nan), self._garch[2].get(t, np.nan)
            if np.isnan(s2):
                return np.nan
            if abs(1 - ph) < 1e-9:
                return s2 * h
            return h * vb + (s2 - vb) * (1 - ph ** h) / (1 - ph)
        raise ValueError(model)


# ----------------------------------------------------------------------------
# DIGITAL PRICING
# ----------------------------------------------------------------------------
def _above(S, K, totvar, dist, zs=None):
    """P(S_T > K) and d/dS of it, with E[S_T] = S (drift = -totvar/2)."""
    sd = math.sqrt(totvar)
    x = (math.log(K / S) + 0.5 * totvar) / sd
    if dist == "normal":
        return 1.0 - stats.norm.cdf(x), stats.norm.pdf(x) / (S * sd)
    # filtered historical simulation: kernel-smoothed empirical survival function of past standardized returns
    if zs is None or len(zs) < 300:
        return np.nan, np.nan
    u = (zs - x) / FHS_BW
    return float(stats.norm.cdf(u).mean()), float(stats.norm.pdf(u).mean() / FHS_BW / (S * sd))


def prob_delta(row, S, totvar, dist, zs=None):
    """Model probability of YES and its delta wrt spot ($ per $)."""
    if not (totvar > 0) or S <= 0:
        return np.nan, np.nan
    st = row["strike_type"]
    if st == "greater":
        return _above(S, row["floor"], totvar, dist, zs)
    if st == "less":
        p, d = _above(S, row["cap"], totvar, dist, zs)
        return 1 - p, -d
    pf, df_ = _above(S, row["floor"], totvar, dist, zs)
    pc, dc = _above(S, row["cap"], totvar, dist, zs)
    return pf - pc, df_ - dc


def kfee(p, mult=1.0):
    return KALSHI_FEE_RATE * mult * p * (1 - p)


# ----------------------------------------------------------------------------
# OBSERVATIONS (one per contract per decision hour)
# ----------------------------------------------------------------------------
def build_obs(mk, book, S, vm, hours, models, dists):
    rows = []
    for r in mk.itertuples(index=False):
        row = r._asdict()
        bk = book.get(row["ticker"])
        if bk is None:
            continue
        E = row["expiry"]
        for H in hours:
            t = E - pd.Timedelta(hours=H)
            tf = t + pd.Timedelta(hours=EXEC_LAG)
            if t not in S.index or t not in bk.index or tf not in bk.index:
                continue
            b0, a0 = bk.at[t, "bid"], bk.at[t, "ask"]
            b1, a1 = bk.at[tf, "bid"], bk.at[tf, "ask"]
            if any(pd.isna(x) for x in (b0, a0, b1, a1)):
                continue
            o = dict(ticker=row["ticker"], expiry=E, day=E.date(), H=H, y=row["y"], S=float(S.loc[t]),
                     bid=b0, ask=a0, bid_f=b1, ask_f=a1, strike_type=row["strike_type"], floor=row["floor"], cap=row["cap"])
            for m in models:
                tv = vm.total_var(m, t, H)
                for d in dists:
                    zs = vm.zwin(m, H, t) if d == "fhs" else None
                    p, dl = prob_delta(row, o["S"], tv, d, zs) if not np.isnan(tv) else (np.nan, np.nan)
                    o[f"p_{m}_{d}"] = p
            rows.append(o)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# HEDGE  (per unit LONG YES; sign flips for a short)
# ----------------------------------------------------------------------------
_hedge_cache = {}


def hedge_path(o, mkrow, S, vm, model, dist):
    """Hourly rebalanced hedge from the fill hour to expiry. Returns (pnl_per_contract_for_long_yes,
    turnover_in_dollars). Long YES has delta D>0 per $ of spot, so the hedge holds -D BTC."""
    key = (o["ticker"], o["H"], model, dist)
    if key in _hedge_cache:
        return _hedge_cache[key]
    E = o["expiry"]
    f = E - pd.Timedelta(hours=o["H"] - EXEC_LAG)
    n_prev, pnl, turn = 0.0, 0.0, 0.0
    t = f
    ok = True
    while t < E:
        nxt = t + pd.Timedelta(hours=1)
        if t not in S.index or nxt not in S.index:
            ok = False
            break
        h_left = (E - t).total_seconds() / 3600.0
        tv = vm.total_var(model, t, h_left)
        zs = vm.zwin(model, int(round(h_left)), t) if dist == "fhs" else None
        _, dl = prob_delta(mkrow, float(S.loc[t]), tv, dist, zs) if not np.isnan(tv) else (np.nan, np.nan)
        if np.isnan(dl):
            ok = False
            break
        n = -dl                                        # BTC to hold per long-YES contract
        turn += abs(n - n_prev) * float(S.loc[t])
        pnl += n * (float(S.loc[nxt]) - float(S.loc[t]))
        n_prev = n
        t = nxt
    if ok:
        turn += abs(n_prev) * float(S.loc[E])          # unwind at expiry
    res = (pnl, turn) if ok else (np.nan, np.nan)
    _hedge_cache[key] = res
    return res


# ----------------------------------------------------------------------------
# TRADING
# ----------------------------------------------------------------------------
def trades(obs, mk_lookup, S, vm, H, edge, model, dist, fee_mult=1.0, hedge_bps=HEDGE_BPS, hedge=True):
    col = f"p_{model}_{dist}"
    d = obs[(obs.H == H) & obs[col].notna() & ((obs.ask - obs.bid) <= MAX_SPREAD)
            & (obs.bid >= 0.01) & (obs.ask <= 0.99) & (obs.bid_f >= 0.0) & (obs.ask_f <= 1.0)].copy()
    d["edge_buy"] = d[col] - d.ask
    d["edge_sell"] = d.bid - d[col]
    d["side"] = 0
    d.loc[(d.edge_buy > edge) & (d.edge_buy >= d.edge_sell), "side"] = 1
    d.loc[(d.edge_sell > edge) & (d.edge_sell > d.edge_buy), "side"] = -1
    d = d[d.side != 0].copy()
    if d.empty:
        return d
    d["fill"] = np.where(d.side == 1, d.ask_f, d.bid_f)
    d["fee"] = d.fill.map(lambda p: kfee(p, fee_mult))
    d["gross"] = np.where(d.side == 1, d.y - d.fill, d.fill - d.y)
    d["net"] = d.gross - d.fee
    if hedge:
        hp, ht = [], []
        for r in d.itertuples():
            pnl, turn = hedge_path(r._asdict(), mk_lookup[r.ticker], S, vm, model, dist)
            hp.append(pnl * r.side)
            ht.append(turn)
        d["hedge_pnl"] = hp
        d["hedge_turn"] = np.array(ht)
        d["hedge_cost"] = d.hedge_turn * hedge_bps * 1e-4
        d["net_hedged"] = d.net + d.hedge_pnl - d.hedge_cost
        d = d[d.net_hedged.notna()]
    return d


# ----------------------------------------------------------------------------
# STATS
# ----------------------------------------------------------------------------
def cluster_ci(df, col, n=N_BOOT, seed=SEED):
    g = [x[col].values for _, x in df.groupby("day")]
    if len(g) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [np.concatenate([g[k] for k in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarize(d, label, hedged=True):
    if d.empty:
        print(f"  {label:<30} no trades")
        return {}
    lo, hi = cluster_ci(d, "net")
    s = dict(label=label, n=len(d), days=d.day.nunique(), win=(d.net > 0).mean(), mean=d.net.mean(), sd=d.net.std(),
             lo=lo, hi=hi, fee=d.fee.mean())
    line = (f"  {label:<30} n={s['n']:>4} days={s['days']:>3} win={s['win']:.0%} "
            f"unhedged: mean={s['mean']*100:+.2f}c sd={s['sd']*100:.1f}c [95% CI {lo*100:+.2f}, {hi*100:+.2f}]")
    if hedged and "net_hedged" in d:
        hlo, hhi = cluster_ci(d, "net_hedged")
        s.update(h_mean=d.net_hedged.mean(), h_sd=d.net_hedged.std(), h_lo=hlo, h_hi=hhi)
        line += (f" | hedged: mean={s['h_mean']*100:+.2f}c sd={s['h_sd']*100:.1f}c [{hlo*100:+.2f}, {hhi*100:+.2f}]")
    print(line)
    return s


def calibration(obs, H, model, dist):
    col = f"p_{model}_{dist}"
    d = obs[(obs.H == H) & obs[col].notna() & ((obs.ask - obs.bid) <= MAX_SPREAD)].copy()
    d["mid"] = (d.bid + d.ask) / 2
    if d.empty:
        return d, None
    d["bs_mkt"] = (d.mid - d.y) ** 2
    d["bs_mod"] = (d[col] - d.y) ** 2
    d["diff"] = d.bs_mkt - d.bs_mod                      # >0 means the model beat the market
    lo, hi = cluster_ci(d, "diff")
    return d, dict(n=len(d), bs_mkt=d.bs_mkt.mean(), bs_mod=d.bs_mod.mean(), diff=d["diff"].mean(), lo=lo, hi=hi)


def leakage_check(S, vm):
    """Forecasts at t must not change if all spot data after t is scrambled (variance forecasts and FHS window)."""
    rng = np.random.default_rng(3)
    ts = S.index[len(S) // 2: len(S) // 2 + 200: 25]
    bad = 0
    for t in ts:
        base = (vm.total_var("ewma", t, 6), vm.total_var("rv", t, 6), vm.zwin("ewma", 6, t))
        S2 = S.copy()
        fut = S2.index > t
        S2[fut] = rng.permutation(S2[fut].values)
        vm2 = VolModels(S2)
        alt = (vm2.total_var("ewma", t, 6), vm2.total_var("rv", t, 6), vm2.zwin("ewma", 6, t))
        same = (np.allclose(base[0], alt[0], equal_nan=True) and np.allclose(base[1], alt[1], equal_nan=True)
                and len(base[2]) == len(alt[2]) and np.allclose(base[2], alt[2], equal_nan=True))
        bad += 0 if same else 1
    return len(ts), bad


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--h", type=int, default=H_PRIMARY)
    ap.add_argument("--edge", type=float, default=EDGE)
    ap.add_argument("--model", default=PRIMARY_MODEL, choices=MODELS)
    ap.add_argument("--dist", default=PRIMARY_DIST, choices=DISTS)
    ap.add_argument("--hedge-bps", type=float, default=HEDGE_BPS)
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if not (a.no_garch and m == "garch")]
    if a.model not in models:
        models.append(a.model)

    mk, book, S = load(Path(a.data_dir))
    mk_lookup = {r["ticker"]: r for r in mk.to_dict("records")}
    print(f"Loaded {len(mk)} settled contracts over {mk.expiry.nunique()} expiry days; spot {S.index.min():%Y-%m-%d} -> {S.index.max():%Y-%m-%d}")
    vm = VolModels(S)
    hours = sorted(set(H_GRID + [a.h]))
    print("Building observations (garch refits every 14 days, this can take a minute)...")
    obs = build_obs(mk, book, S, vm, hours, models, DISTS)
    if obs.empty:
        print("No usable observations (check that hourly candles have bid/ask at the decision hours).")
        return
    days = sorted(obs.day.unique())
    cut = days[int(len(days) * IS_FRAC)]
    obs["period"] = np.where(obs.day < cut, "IS", "OOS")
    print(f"  {len(obs)} observations, {len(days)} expiry days; in-sample before {cut}, out-of-sample from {cut}")

    n_chk, n_bad = leakage_check(S, vm)
    print(f"Look-ahead check: scrambled all spot data after t at {n_chk} timestamps -> {n_bad} variance forecasts changed (must be 0)")

    # ---------------- calibration ----------------
    print(f"\n=== CALIBRATION at H={a.h}h before expiry: does the model beat the market price? (Brier score, lower is better) ===")
    for m in models:
        for dname in DISTS:
            _, c = calibration(obs, a.h, m, dname)
            if c:
                print(f"  {m:<6} {dname:<6} n={c['n']:>5}  market={c['bs_mkt']:.4f}  model={c['bs_mod']:.4f}  "
                      f"(market - model = {c['diff']*1e4:+.1f}e-4, 95% CI {c['lo']*1e4:+.1f}, {c['hi']*1e4:+.1f}; >0 = model better)")
    cal, _ = calibration(obs, a.h, a.model, a.dist)
    bins = [0, .02, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98, 1.0001]
    cal["bin"] = pd.cut(cal.mid, bins, right=False)
    tab = cal.groupby("bin").agg(n=("y", "size"), mkt_price=("mid", "mean"), model_p=(f"p_{a.model}_{a.dist}", "mean"),
                                 realized=("y", "mean")).dropna()
    print(f"\n  Reliability table ({a.model}/{a.dist}): market price vs model vs what actually happened")
    print(tab.round(3).to_string())

    # ---------------- primary trading result ----------------
    print(f"\n=== PRIMARY: H={a.h}h, edge>{a.edge*100:.0f}c, model={a.model}/{a.dist}, fill at +{EXEC_LAG}h bid/ask, "
          f"Kalshi fee {KALSHI_FEE_RATE}, hedge {a.hedge_bps:.0f}bps ===")
    T = trades(obs, mk_lookup, S, vm, a.h, a.edge, a.model, a.dist, 1.0, a.hedge_bps)
    sP = summarize(T, "ALL")
    if not T.empty:
        summarize(T[T.period == "IS"], "  in-sample")
        summarize(T[T.period == "OOS"], "  out-of-sample")
        summarize(T[T.side == 1], "  buy YES")
        summarize(T[T.side == -1], "  sell YES")
        T["mny"] = pd.cut(np.abs(np.log(T.S / T.floor.fillna(T.cap))), [0, .01, .02, .04, 1])
        for k, g in T.groupby("mny"):
            summarize(g, f"  |ln(K/S)| in {k}")
        print(f"  mean gross edge captured {T.gross.mean()*100:+.2f}c, fee {T.fee.mean()*100:.2f}c, "
              f"hedge pnl {T.hedge_pnl.mean()*100:+.3f}c, hedge cost {T.hedge_cost.mean()*100:.3f}c per contract")
        print(f"  hedge changes P&L std from {T.net.std()*100:.1f}c to {T.net_hedged.std()*100:.1f}c per contract")

    # ---------------- sensitivity ----------------
    print("\n=== SENSITIVITY: mean net cents per contract, unhedged / hedged, [trades] (primary model) ===")
    print("edge \\ H      " + "".join(f"{h:>2}h                     " for h in H_GRID))
    for e in EDGE_GRID:
        cells = []
        for h in H_GRID:
            t = trades(obs, mk_lookup, S, vm, h, e, a.model, a.dist, 1.0, a.hedge_bps)
            cells.append(f"{t.net.mean()*100:+6.2f}/{t.net_hedged.mean()*100:+6.2f} [{len(t):>4}]  " if len(t) else "        n/a            ")
        print(f"{e*100:>4.0f}c        " + "".join(cells))
    print("\nBy volatility model / distribution (H and edge at primary):")
    for m in models:
        for dname in DISTS:
            t = trades(obs, mk_lookup, S, vm, a.h, a.edge, m, dname, 1.0, a.hedge_bps)
            summarize(t, f"  {m}/{dname}")
    print("\nFee multiplier at primary (0 = no Kalshi fee):")
    for fm in (0.0, 1.0, 2.0):
        t = trades(obs, mk_lookup, S, vm, a.h, a.edge, a.model, a.dist, fm, a.hedge_bps, hedge=False)
        if len(t):
            print(f"  fee x{fm:.0f}: mean net {t.net.mean()*100:+.2f}c over {len(t)} trades")

    if not T.empty and "hedge_turn" in T:
        print("\nHedge cost sensitivity (primary trades): the hedge holds about $%.0f of BTC per $1 contract" % (T.hedge_turn.mean() / 2))
        for bps in HEDGE_BPS_GRID:
            nh = (T.net + T.hedge_pnl - T.hedge_turn * bps * 1e-4)
            print(f"  hedge cost {bps:>4.0f}bps: hedged mean net {nh.mean()*100:+.2f}c (sd {nh.std()*100:.1f}c) vs unhedged {T.net.mean()*100:+.2f}c (sd {T.net.std()*100:.1f}c)")

    T.drop(columns=[c for c in ("mny",) if c in T]).to_csv(out / "btc_trades_primary.csv", index=False)
    obs.to_csv(out / "btc_observations.csv", index=False)
    tab.to_csv(out / "btc_reliability_table.csv")

    if not a.no_plot and not T.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
            ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
            ax[0].plot(tab.mkt_price, tab.realized, "o-", label="market price")
            ax[0].plot(tab.model_p, tab.realized, "s-", label="model prob")
            ax[0].set_title("Reliability: predicted vs realized"); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("realized"); ax[0].legend()
            ax[1].hist([T.net * 100, T.net_hedged * 100], bins=40, label=["unhedged", "hedged"])
            ax[1].set_title("Net cents per contract"); ax[1].legend()
            byd = T.groupby("day")[["net", "net_hedged"]].sum().cumsum() * 100
            byd.plot(ax=ax[2]); ax[2].set_title("Cumulative net cents (1 contract per trade)")
            fig.tight_layout(); fig.savefig(out / "btc_backtest.png", dpi=130)
            print(f"\nPlot saved: {out / 'btc_backtest.png'}")
        except Exception as e:  # noqa: BLE001
            print("plot skipped:", e)
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
