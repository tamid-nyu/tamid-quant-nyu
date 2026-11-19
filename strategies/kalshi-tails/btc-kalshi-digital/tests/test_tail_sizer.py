import numpy as np
import pytest

from tail_sizer import fee, payoff_matrix, size_per_trade, size_portfolio, worst_case

CAP = 30.0


def _book(strikes, sides, prices):
    grid = np.unique(strikes)
    return payoff_matrix(grid, np.asarray(strikes, float), np.asarray(sides), np.asarray(prices, float))


def test_fee_values():
    assert fee(0.5) == pytest.approx(0.0175)
    assert fee(0.05) == pytest.approx(0.07 * 0.05 * 0.95)


def test_payoff_matrix_short_and_long():
    grid = np.array([100.0, 200.0])
    P = payoff_matrix(grid, np.array([200.0, 100.0]), np.array([-1, 1]), np.array([0.05, 0.95]))
    assert P.shape == (3, 2)
    # short YES @0.05 on K=200: wins 0.05 unless settlement > 200
    assert P[:, 0] == pytest.approx([0.05 - fee(0.05), 0.05 - fee(0.05), 0.05 - 1 - fee(0.05)])
    # long YES @0.95 on K=100: loses 0.95 only when settlement <= 100
    assert P[:, 1] == pytest.approx([-0.95 - fee(0.95), 0.05 - fee(0.95), 0.05 - fee(0.95)])


def test_worst_case_with_base():
    P = np.array([[1.0], [-2.0]])
    assert worst_case(P, np.array([3.0])) == pytest.approx(-6.0)
    assert worst_case(P, np.array([3.0]), base=np.array([0.0, 4.0])) == pytest.approx(-2.0)


def test_random_books_never_exceed_cap():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 8)
        strikes = rng.choice(np.arange(60_000, 70_000, 250), n)
        sides = rng.choice([-1, 1], n)
        prices = np.where(sides == -1, rng.uniform(0.01, 0.10, n), rng.uniform(0.90, 0.99, n))
        P = _book(strikes, sides, prices)
        q = size_portfolio(rng.uniform(0.005, 0.03, n), P, np.zeros(P.shape[0]), CAP, 50)
        assert q.dtype.kind == "i" and (q >= 0).all() and (q <= 50).all()
        assert worst_case(P, q) >= -CAP - 1e-9


def test_nested_shorts_share_one_budget():
    P1 = _book([66_000], [-1], [0.05])
    alone = size_portfolio(np.array([0.01]), P1, np.zeros(2), CAP, 100)
    P2 = _book([65_000, 66_000], [-1, -1], [0.05, 0.05])
    pair = size_portfolio(np.array([0.01, 0.01]), P2, np.zeros(3), CAP, 100)
    assert alone[0] == 31 and pair.sum() <= alone[0]


def test_two_sided_book_gets_more_room():
    P = _book([66_000, 60_000], [-1, 1], [0.05, 0.95])
    q = size_portfolio(np.array([0.01, 0.01]), P, np.zeros(3), CAP, 100)
    assert q.sum() > 31 and worst_case(P, q) >= -CAP - 1e-9


def test_base_already_breaching_returns_zeros():
    P = _book([66_000], [-1], [0.05])
    assert size_portfolio(np.array([0.01]), P, np.array([0.0, -31.0]), CAP, 50).sum() == 0


def test_per_trade_ignores_nesting():
    P = _book([65_000, 66_000], [-1, -1], [0.05, 0.05])
    q = size_per_trade(P, CAP, 100)
    assert list(q) == [31, 31] and worst_case(P, q) < -CAP


def test_order_fee_rounds_up_per_order():
    from tail_sizer import order_fee
    assert order_fee(1, 0.01) == pytest.approx(0.01)      # 0.000693 rounds up to 1c
    assert order_fee(30, 0.01) == pytest.approx(0.03)     # 0.02079 rounds up to 3c
    assert order_fee(100, 0.5) == pytest.approx(1.75)     # exact cents stay put despite float noise
    assert order_fee(0, 0.5) == 0.0


def test_rounding_leaves_no_free_contract():
    """After rounding, no single extra contract on any candidate would still fit under the cap."""
    rng = np.random.default_rng(5)
    for _ in range(150):
        n = rng.integers(2, 7)
        strikes = rng.choice(np.arange(60_000, 70_000, 250), n)
        sides = rng.choice([-1, 1], n)
        prices = np.where(sides == -1, rng.uniform(0.01, 0.10, n), rng.uniform(0.90, 0.99, n))
        P = _book(strikes, sides, prices)
        q = size_portfolio(rng.uniform(0.005, 0.03, n), P, np.zeros(P.shape[0]), CAP, 50)
        assert worst_case(P, q) >= -CAP - 1e-9
        for i in range(n):
            if q[i] < 50:
                bumped = q.copy()
                bumped[i] += 1
                assert worst_case(P, bumped) < -CAP - 1e-9
