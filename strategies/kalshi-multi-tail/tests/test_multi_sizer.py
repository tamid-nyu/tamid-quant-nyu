import numpy as np
import pytest

from tail_sizer import payoff_matrix, size_portfolio
from multi_sizer import _size_lp, book_losses, check_caps, size_multi, topk_sum


def _book(rng, n=None, held=False):
    n = n or int(rng.integers(1, 6))
    strikes = np.sort(rng.uniform(90, 110, n))
    sides = rng.choice([-1, 1], n)
    prices = np.where(sides == -1, rng.uniform(0.01, 0.08, n), rng.uniform(0.92, 0.99, n))
    grid = np.unique(strikes)
    P = payoff_matrix(grid, strikes, sides, prices)
    base = np.zeros(len(grid) + 1)
    if held:
        base = P[:, 0] * int(rng.integers(1, 10))
    edges = rng.uniform(0.005, 0.03, n)
    return {"P": P, "base": base, "edges": edges}


def test_topk_sum():
    assert topk_sum(np.array([1.0, 5.0, 3.0]), 2) == 8.0
    assert topk_sum(np.array([1.0]), 3) == 1.0


def test_single_book_matches_size_portfolio():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = _book(rng)
        q = size_multi([b], 30.0, 40.0, 2, 50)[0]
        assert np.array_equal(q, size_portfolio(b["edges"], b["P"], b["base"], 30.0, 50))


def test_general_lp_single_book_same_objective():
    rng = np.random.default_rng(1)
    for _ in range(50):
        b = _book(rng)
        q_lp = _size_lp([b], np.array([30.0]), 40.0, 2, 50)[0]
        q_ref = size_portfolio(b["edges"], b["P"], b["base"], 30.0, 50)
        assert check_caps([b], [q_lp], 30.0, 40.0, 2)
        # rounding may land on different vertices; objectives agree to within a few contracts of the best edge
        assert abs(b["edges"] @ q_lp - b["edges"] @ q_ref) <= 3 * b["edges"].max() + 1e-9


def test_random_books_respect_caps():
    rng = np.random.default_rng(2)
    n_traded = 0
    for _ in range(200):
        books = [_book(rng, held=bool(rng.integers(0, 2))) for _ in range(int(rng.integers(2, 5)))]
        qs = size_multi(books, 30.0, 40.0, 2, 50)
        held = book_losses(books, [np.zeros(len(b["edges"])) for b in books])
        w = book_losses(books, qs)
        if topk_sum(held, 2) > 40.0:
            assert not any(q.any() for q in qs)
            continue
        for m in range(len(books)):
            if held[m] > 30.0:
                assert not qs[m].any() and w[m] == pytest.approx(held[m])
            else:
                assert w[m] <= 30.0 + 1e-9
        assert topk_sum(w, 2) <= 40.0 + 1e-9
        n_traded += any(q.any() for q in qs)
    assert n_traded > 50


def _short_yes_book():
    # one short-YES contract at 5c on strike 100: worst case -0.95 - fee per contract
    P = payoff_matrix(np.array([100.0]), np.array([100.0]), np.array([-1]), np.array([0.05]))
    return {"P": P, "base": np.zeros(2), "edges": np.array([0.02])}


def test_two_markets_limited_by_k_cap():
    a, b = _short_yes_book(), _short_yes_book()
    alone = size_multi([a], 30.0, 40.0, 2, 1000)[0]
    assert book_losses([a], [alone])[0] == pytest.approx(30.0, abs=1.0)
    qa, qb = size_multi([a, b], 30.0, 40.0, 2, 1000)
    w = book_losses([a, b], [qa, qb])
    assert w.sum() <= 40.0 + 1e-9 and w.sum() >= 38.0
    assert w.min() >= 8.0


def test_k1_is_max_market():
    a, b = _short_yes_book(), _short_yes_book()
    qa, qb = size_multi([a, b], 30.0, 20.0, 1, 1000)
    w = book_losses([a, b], [qa, qb])
    assert w.max() <= 20.0 + 1e-9 and w.min() >= 19.0       # each market alone is held to 20, not their sum


def test_breached_market_blocks_only_itself():
    a, b = _short_yes_book(), _short_yes_book()
    a = dict(a, base=a["P"][:, 0] * 35)                      # held loss ~35 > cap 30
    qa, qb = size_multi([a, b], 30.0, 100.0, 2, 1000)
    assert qa.sum() == 0 and qb.sum() > 0
    assert book_losses([b], [qb])[0] <= 30.0 + 1e-9
