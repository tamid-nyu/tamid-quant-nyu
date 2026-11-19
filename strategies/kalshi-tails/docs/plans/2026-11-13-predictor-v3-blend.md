# Predictor v3 (Blend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a blend estimator f = w·f_P + (1 − w)·f_F1, choose w on development data by a pre-registered rule,
freeze it, add holdout and forward evaluation with v2's guards, and expose it in the CLI and the dashboard.

**Architecture:**
- **`v3/blend.py`:** `BlendEstimator` wraps the frozen `PooledEstimator` (v1 P) and `LogitV2Estimator(FROZEN)` (v2
  F1). It plugs into the existing `estimate_row` hook, so `multi_tail.run` doesn't change.
- **`v3/select_w.py`:** blends the two components' walk-forward predictions after the fact and scores the weight
  grid with v2's bootstrap.
- **`v3/evaluate.py`:** reuses v2's evaluation helpers and its forward guard.

**Tech Stack:** Python 3, numpy, pandas, pytest; Streamlit AppTest for the dashboard task.

**Spec:** `strategies/kalshi-tails/docs/specs/2026-11-13-predictor-v3-blend-design.md`

## Global Constraints

- **Components frozen:**
  - v1 P = `PooledEstimator(n0=50, n1=100)`, eligible when n ≥ 100
  - v2 F1 = `LogitV2Estimator` built from `v2.logit_v2.FROZEN`, eligible when n ≥ 200
  - Neither component's code changes.
- **Blend:**
  - f_v3 = w·f_P + (1 − w)·f_F1, clipped to [0.001, 0.999]
  - Eligible only when both components are eligible and finite; otherwise (NaN, 0). No fallback.
  - `min_rows = 1`; n = 1 when eligible.
- **Weight grid:** w ∈ {0, 0.25, 0.5, 0.75, 1}. Pick the lowest development log-loss, but take 0.5 if it is within
  1 SE of the best on the paired Δ.
- **Bootstrap:** `v2.scoring.block_bootstrap_delta` (7-day blocks, 4,000 resamples, seed 11).
- **Degenerate pick:** if w ∈ {0, 1} is chosen, there is no new arm. Write the report, skip the freeze, and stop
  after Task 2. Tasks 3 and 4 are then not executed. This is a valid outcome, not a failure.
- **Periods:** development 2025-01-01 → 2026-06-30, with walk-forward from 2024-07-05. Holdout 2026-07-01 →
  2026-11-11. Forward 2026-11-12 → 2027-05-12, opening on 2027-05-12. Primary markets BTC + SPX; five markets are
  exploratory.
- **Selection sees no data after 2026-06-30.** Drop those rows on load, as `v2.select.load_dev` does.
- **v1/v2 neutrality:** these must stay green:
  - Model 0 (402 trades, +$52.61)
  - the v1 P holdout (Sharpe 1.24, +5.98%, 361 trades)
  - the v2 holdout log-loss 0.08207
  - suites: dashboard 51, `tests v2/tests` 81
- **Never commit data.** `v3/output/preds/` is git-ignored; only summary text files are committed. Commit locally
  on branch predictor-v2, never push.
- **Tests:** run `cd strategies/kalshi-tails/kalshi-multi-tail && python3 -m pytest v3/tests -q`, the dashboard suite and
  `tests v2/tests`, each separately.

## Review Focus

1. **A row where only one component is eligible** (e.g. early SPX, when v1 P's pooled cell has n < 100) must return
   (NaN, 0) and never fall back to one model. `test_one_component_ineligible` (Task 1).
2. **A tie between w = 0.5 and the best w within 1 SE** must pick 0.5, and a clear winner must win.
   `test_select_w_prefers_half_within_se` and `test_select_w_takes_best_outside_se` (Task 2).
3. **Post-development rows in the input** must not change the selection. `test_select_w_ignores_post_dev_rows`
   (Task 2).
4. **`v3.evaluate --period forward` before 2027-05-12**, including with `--today` set later, must print only
   data-quality counts. `test_v3_forward_refuses_early` (Task 3).
5. **The dashboard with "v3 blend (frozen)" and learn-only markets** must run without error, with the learn markets
   feeding both components. `test_app_v3_learn_only` (Task 4).

---

### Task 1: BlendEstimator

**Files:**
- Create: `strategies/kalshi-tails/kalshi-multi-tail/v3/__init__.py`, `v3/blend.py`, `v3/tests/conftest.py` (same path setup
  as `v2/tests/conftest.py`), `v3/tests/test_blend.py`
- Modify: `strategies/kalshi-tails/kalshi-multi-tail/estimators.py` (`make_estimator` kind "V3"), repo `.gitignore` (add
  `strategies/kalshi-tails/kalshi-multi-tail/v3/output/preds/`)

**Interfaces:**
- Produces:
  - `FROZEN: dict = {"w": None}` in `v3/blend.py`. It is set by Task 3; None means not frozen.
  - `BlendEstimator(w: float, n0=50, n1=100)` with `min_rows = 1` and attributes `p_est`, `v2_est`, `w`:
    - `update(df)` passes `df` to both components.
    - `estimate_row(row) -> (p, n)`.
    - `estimate(...)` raises TypeError, as V2 does.
  - `make_estimator("V3", cfg)` returns `BlendEstimator(FROZEN["w"], cfg.n0, cfg.n1)`, or raises
    `ValueError("v3 not frozen")` when `FROZEN["w"]` is None. The ValueError text lists "S, P, V2 or V3".

- [ ] **Step 1: Write failing tests:**
  - `test_blend_endpoints` (synthetic obs with feature columns, enough rows for both to be eligible): for every row
    where both are eligible, w = 1 gives p == the P estimate and w = 0 gives p == the F1 estimate (atol 1e-12).
  - `test_blend_half_is_mean`: w = 0.5 gives the mean of the two.
  - `test_one_component_ineligible`: with few enough rows that P's pooled cell has n < 100 while F1 has n ≥ 200,
    or the reverse, the result is (NaN, 0).
  - `test_blend_matches_after_the_fact`: `walk_forward_predict(obs, BlendEstimator(0.25), cfg)` equals
    0.25·P_preds + 0.75·F1_preds on the eligible rows.
  - `test_blend_no_lookahead`: calling update with day d after predicting day d leaves that prediction unchanged.
  - `test_make_estimator_v3_unfrozen`: with `FROZEN["w"] = None`, it raises ValueError mentioning "not frozen".
- [ ] **Step 2: Run** `python3 -m pytest v3/tests -q`. Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the v3, dashboard and `tests v2/tests` suites. Expected: v3 6 passed, the others unchanged
  (51, 81).
- [ ] **Step 5: Commit** — "v3: blend estimator".

### Task 2: Pre-registered choice of w

**Files:** Create `v3/select_w.py`, `v3/tests/test_select_w.py`

**Interfaces:**
- Consumes: Task 1. Also `v2.select.load_dev`, `_predict_all` (or `walk_forward_predict` per component),
  `v2.scoring.logloss`, `paired`, `block_bootstrap_delta`.
- Produces:
  - `WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)`
  - `choose_w(ll: dict[float, ndarray], day) -> (w, table)` implementing the Global Constraints rule
  - `run_selection(obs, cfg) -> str` (the report text), which:
    - runs P and F1 walk-forward over BTC + SPX up to 2026-06-30
    - keeps rows with day ≥ 2025-01-01 where P n ≥ 100 and F1 n ≥ 200
    - scores every w, plus the mid
    - writes per-w log-loss, paired Δ vs best with SE, by-market and by-H log-loss, row counts, the chosen w, and,
      when w ∈ {0, 1}, the line "Chosen w reproduces a component; no new v3 arm is frozen."
  - CLI `python -m v3.select_w` writes `v3/output/selection.txt`, plus predictions to `v3/output/preds/`.

- [ ] **Step 1: Write failing tests:**
  - `test_select_w_prefers_half_within_se`: synthetic losses where 0.75 is best by less than 1 SE over 0.5 → 0.5.
  - `test_select_w_takes_best_outside_se`: 0.75 better than 0.5 by more than 1 SE → 0.75.
  - `test_select_w_degenerate_note`: when 1.0 is chosen, the report contains the "no new v3 arm" line.
  - `test_select_w_ignores_post_dev_rows`: adding rows dated ≥ 2026-07-01 with flipped y leaves the report
    unchanged, and spying on the prediction input shows none of those rows.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the suites, then `python -m v3.select_w`. Record the chosen w and the Δ table in the report.
  **If w ∈ {0, 1}: stop here.** Commit, report it, and Tasks 3–4 are not executed.
- [ ] **Step 5: Commit** `v3/select_w.py v3/tests/ v3/output/selection.txt` — "v3: pre-registered weight selection".

### Task 3: Freeze, holdout evaluation, forward script

**Files:** Modify `v3/blend.py` (FROZEN). Create `v3/evaluate.py`, `v3/tests/test_evaluate.py`, `v3/README.md`.
Modify `strategies/kalshi-tails/kalshi-multi-tail/README.md` (one link line).

**Interfaces:**
- Consumes: Task 2's chosen w. Also v2.evaluate's `one_sided`, `mde`, `data_quality`, `_real_today`,
  `FORWARD_OPEN`, `HOLDOUT`, `FORWARD` and `sharpe_block`-style logic. `multi_tail.run` with estimator "V3".
- Produces:
  - `FROZEN = {"w": <chosen>}` and the README's "Frozen v3 configuration", committed **before** any holdout scoring
    ("v3: freeze w"). A follow-up commit records that hash.
  - `evaluate.py --period holdout|forward [--today]` with `today = min(--today, real)`:
    - **holdout** → `v3/output/holdout.txt`: the header line "HOLDOUT: seen period, not the verdict"; log-loss for
      v3, P, F1 and mid; Δ_P and Δ_F1 with one-sided 95th percentiles; MDE; the would-trade secondary; Sharpe for
      v3 vs P and vs F1; trades and wins per day; an exploratory five-market block
    - **forward** before 2027-05-12 → data-quality counts only, with no forward.txt. Otherwise the verdict:
      "WIN" only if both Δ_P and Δ_F1 pass, else "not shown".

- [ ] **Step 1: Freeze** (FROZEN, README) and commit. Do not run any holdout code before this commit.
- [ ] **Step 2: Write failing tests:**
  - `test_v3_forward_refuses_early`: real date 2026-11-13 with `--today 2027-06-01` → no Δ/WIN text and no
    forward.txt.
  - `test_v3_verdict_requires_both`: synthetic predictions that beat P but not F1 → "not shown"; beat both → "WIN".
  - `test_v3_cli_runs`: `multi_tail.run` with estimator "V3" on synthetic feature obs completes, and its trades'
    f values equal the blend.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement** `evaluate.py`, and allow `--estimator V3` in `multi_tail.main`. It loads features like
  V2 and writes to `v3/output/sim/`, which is git-ignored.
- [ ] **Step 5: Run** all suites, then `python -m v3.evaluate --period holdout` and `--period forward`. Record the
  key holdout numbers in the report.
- [ ] **Step 6: Commit** — "v3: holdout evaluation and forward script". Then record the freeze hash in a follow-up
  commit.

### Task 4: Dashboard option

**Files:** Modify `dashboard/engine.py` (ESTIMATORS gains "v3 blend (frozen)" → ("V3", None)),
`dashboard/README.md`, `dashboard/tests/test_engine.py`, `dashboard/tests/test_app.py`

**Interfaces:**
- Consumes: `make_estimator("V3")`, the frozen w.
- Produces: the new estimator label, working in backtests, compare mode, learn-only and the accuracy tab.

- [ ] **Step 1: Write failing tests:**
  - `test_engine_v3_matches_run` (real data): BTC + SPX holdout with "v3 blend (frozen)" gives trades identical to
    `multi_tail.run` with estimator V3 on the same rows.
  - `test_accuracy_v3` (real data): rows > 0, with finite log-losses.
  - `test_app_v3_learn_only` (AppTest, real data): markets GOLD, learn BTC and SPX, "v3 blend (frozen)" → no
    exception, no error.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** all suites. Expected: all pass.
- [ ] **Step 5: Commit** — "dashboard: v3 blend option".
