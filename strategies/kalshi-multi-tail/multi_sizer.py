"""
Multi-market worst-case sizer (spec section 6).

Each book is one market's positions for the current decision round:
    {"P": (scenarios, n) per-contract payoffs, "base": (scenarios,) payoff of positions already held, "edges": (n,)}
Market m's worst-case loss is w_m = max(0, -min_s(base_m + P_m q_m)). Sizes maximize total edge subject to
    w_m <= cap_m           for every market
    sum of k largest w_m <= cap_k
    0 <= q <= q_max, whole contracts.
A market whose held positions already lose more than cap_m gets no new size and keeps its held loss.
"""
import numpy as np

from tail_sizer import size_portfolio


def topk_sum(w, k):
    w = np.sort(np.asarray(w, float))[::-1]
    return float(w[:k].sum())


def book_losses(books, qs):
    return np.array([max(0.0, -float((b["base"] + b["P"] @ np.asarray(q, float)).min())) for b, q in zip(books, qs)])


def _caps(books, cap_m):
    caps = np.broadcast_to(np.asarray(cap_m, float), (len(books),)).astype(float)
    held = book_losses(books, [np.zeros(len(b["edges"])) for b in books])
    blocked = held > caps + 1e-9
    return np.where(blocked, held, caps), blocked, held


def check_caps(books, qs, cap_m, cap_k, k):
    wcap, _, _ = _caps(books, cap_m)
    w = book_losses(books, qs)
    return bool((w <= wcap + 1e-9).all() and topk_sum(w, k) <= cap_k + 1e-9)


def _size_lp(books, cap_m, cap_k, k, q_max):
    """General LP over [q, w, t, u] followed by the same floor / repair / add-back rounding as tail_sizer."""
    from scipy.optimize import linprog

    M = len(books)
    ns = [len(b["edges"]) for b in books]
    N = sum(ns)
    zeros = [np.zeros(n, dtype=int) for n in ns]
    if N == 0:
        return zeros
    wcap, blocked, held = _caps(books, cap_m)
    if topk_sum(held, k) > cap_k + 1e-9:
        return zeros
    off = np.concatenate([[0], np.cumsum(ns)])
    nv = N + M + 1 + M                                    # q, w, t, u
    iw, it, iu = N, N + M, N + M + 1
    A, bvec = [], []
    for m, b in enumerate(books):
        P = np.asarray(b["P"], float)
        for s in range(P.shape[0]):                       # -P q - w <= base
            row = np.zeros(nv)
            row[off[m]:off[m + 1]] = -P[s]
            row[iw + m] = -1.0
            A.append(row)
            bvec.append(float(b["base"][s]))
        row = np.zeros(nv)                                # w - t - u <= 0
        row[iw + m], row[it], row[iu + m] = 1.0, -1.0, -1.0
        A.append(row)
        bvec.append(0.0)
    row = np.zeros(nv)                                    # k t + sum u <= cap_k
    row[it] = k
    row[iu:iu + M] = 1.0
    A.append(row)
    bvec.append(cap_k)
    c = np.zeros(nv)
    c[:N] = -np.concatenate([np.asarray(b["edges"], float) for b in books])
    bounds = []
    for m, n in enumerate(ns):
        bounds += [(0, 0 if blocked[m] else q_max)] * n
    bounds += [(0, float(wcap[m])) for m in range(M)] + [(0, None)] + [(0, None)] * M
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(bvec), bounds=bounds, method="highs")
    if res.status != 0:
        return zeros
    x = np.floor(np.clip(res.x[:N], 0, q_max) + 1e-9).astype(int)
    qs = [x[off[m]:off[m + 1]].copy() for m in range(M)]
    for m in range(M):
        if blocked[m]:
            qs[m][:] = 0

    def ok():
        return check_caps(books, qs, cap_m, cap_k, k)

    # 1) repair: remove one contract at a time from the most-violating market's worst-scenario loser
    while not ok():
        w = book_losses(books, qs)
        over = np.where(w > wcap + 1e-9)[0]
        order = over[np.argsort(-(w - wcap)[over])] if len(over) else np.argsort(-w)
        for m in order:
            P, q = np.asarray(books[m]["P"], float), qs[m]
            row = int(np.argmin(books[m]["base"] + P @ q))
            losers = np.where((q > 0) & (P[row] < 0))[0]
            if len(losers):
                q[losers[np.argmin(P[row, losers])]] -= 1
                break
        else:
            return zeros
    # 2) add back single contracts, best edge first, while every cap still holds
    flat = [(m, i, float(books[m]["edges"][i])) for m in range(M) if not blocked[m] for i in range(ns[m])]
    flat.sort(key=lambda t: -t[2])
    added = True
    while added:
        added = False
        for m, i, _ in flat:
            if qs[m][i] < q_max:
                qs[m][i] += 1
                if ok():
                    added = True
                    break
                qs[m][i] -= 1
    return qs


def size_multi(books, cap_m, cap_k, k, q_max):
    """Whole-contract sizes per book. cap_m is a scalar or one cap per book (after any fee reservation)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if cap_k <= 0:
        raise ValueError("cap_k must be > 0")
    if len(books) == 1:
        # one market: the k-worst constraint is a cap of cap_k on that market, so this is exactly tail_sizer's problem
        cm = float(np.broadcast_to(np.asarray(cap_m, float), (1,))[0])
        b = books[0]
        return [size_portfolio(b["edges"], b["P"], b["base"], min(cm, cap_k), q_max)]
    return _size_lp(books, cap_m, cap_k, k, q_max)
