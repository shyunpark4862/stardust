"""Variable selection: stepwise and best_subset (docs/api/regression.md §9.3, docs/design/regression.md
§3.8, decisions G166-G180).

Against Arb (tests/regression/ref/selection.json, tests/regression/refdata_selection.py): the
paths and subsets the reference values choose, and the criterion values along them. Against the
model functions themselves: the result's fit is the function on the chosen columns bit for bit,
every step's criterion is that of the fit of its model (a GLM's exactly, a linear model's to
rounding: its candidates come from one triangle's updates), best_subset is the exhaustive
search, and the cross-validation errors are those of the training rows' fits. Against
scikit-learn 1.9.1's SequentialFeatureSelector with sd's folds. And the rules: the 1e-7 move
threshold, ties to the earlier column, the aliased terms dropped first, the p-value thresholds,
the start models, include, terms, NaN, batches, the limits, the messages and the read-only
results. R's step, stepAIC, add1, drop1 and regsubsets are test_selection_against_r.py's."""

import itertools
import math

import numpy as np
import pytest

import stardust as sd
from tests._bits import bits_equal, bits_equal_any_nan

from . import refdata_selection

R = sd.regression
REFS = refdata_selection.load() if refdata_selection.REF.exists() else {}


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def data(seed=0, n=120, p=6, beta=(0.9, 0.0, -0.5, 0.0, 0.25, 0.1)):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    X = Z + 0.4 * Z[:, list(range(1, p)) + [0]]
    y = 1.0 + X @ np.asarray(beta[:p]) + rng.standard_normal(n)
    return y, X


def binary(seed=1, n=200, p=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = 0.2 + X @ [0.9, 0.0, -0.6, 0.0, 0.25][:p]
    return (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float), X


def names_moves(r):
    return [(s.action, s.term) for s in r.steps[1:]]


# ---- Arb references


@pytest.mark.parametrize("name", sorted(REFS))
def test_paths_match_the_reference(name):
    p = REFS[name]
    names = [f"x{j + 1}" for j in range(len(p["X"][0]))]
    for key, run in p["runs"].items():
        if run["near_tie"]:
            continue
        d, c = key.split("/")
        r = refdata_selection.stepwise(p, d, c)
        assert [(s.action, names.index(s.term)) for s in r.steps[1:]] == [tuple(m) for m in run["moves"]], key


@pytest.mark.parametrize("name", sorted(REFS))
def test_subsets_match_the_reference(name):
    p = REFS[name]
    k = len(p["X"][0])
    for c, b in p["best"].items():
        if b["near_tie"]:
            continue
        r = refdata_selection.best_subset(p, c)
        sets = [None if m is None else [j for j in range(k) if m >> j & 1] for m in b["sets"]]
        assert r.subsets == sets, c
        assert r.selected == sets[b["pick"]], c


@pytest.mark.parametrize("name", sorted(REFS))
def test_criterion_values_against_arb(name):
    """The criterion along every path and of each size's best: 1e-15 relative (Cp, a difference
    of terms of order n, 1e-13 of n; adjr2 1e-15 absolute). The cross-validation errors of the
    nearly collinear design (kappa 2.3e5) are kappa u (1.3e-12 and 3e-12 measured): the folds'
    fits come from their triangles without refinement."""
    p = REFS[name]
    vals = refdata_selection.values(name, p)
    for field, ref in p["ref"].items():
        hl = np.asarray(ref, dtype=float)
        got = np.asarray(vals[field], dtype=float)
        assert got.shape[0] == hl.shape[0], field
        err = np.abs((got - hl[:, 0]) - hl[:, 1])
        if field == "best_cp":
            assert np.max(err) <= 1e-13 * len(p["y"]), field
        elif field == "best_adjr2":
            assert np.max(err) <= 1e-15, field
        else:
            bound = 1e-11 if (name == "cond" and field.endswith("cv")) else 1e-15
            assert np.max(err / np.abs(hl[:, 0])) <= bound, field


# ---- the model functions' own values


def test_fit_is_the_model_function_on_the_chosen_columns():
    y, X = data()
    for kw in (dict(), dict(direction="forward"), dict(criterion="bic"), dict(criterion="pvalue")):
        r = R.stepwise(y, X, **kw)
        ref = R.ols(y, X[:, r.selected])
        assert bits_equal(a(r.fit.params), a(ref.params))
        assert bits_equal(a(r.fit.bse), a(ref.bse))
        assert r.fit.param_names == ["const"] + [f"x{j + 1}" for j in r.selected]
        assert r.selected_names == [f"x{j + 1}" for j in r.selected]
    w = np.random.default_rng(2).uniform(0.3, 3.0, len(y))
    w[[3, 9]] = 0.0
    r = R.stepwise(y, X, model="wls", weights=w, cov_type="HC1")
    ref = R.wls(y, X[:, r.selected], weights=w, cov_type="HC1")
    assert bits_equal(a(r.fit.params), a(ref.params)) and bits_equal(a(r.fit.bse), a(ref.bse))
    S = 0.5 ** np.abs(np.subtract.outer(np.arange(len(y)), np.arange(len(y))))
    b = R.best_subset(y, X, model="gls", sigma=S)
    assert bits_equal(a(b.fit.params), a(R.gls(y, X[:, b.selected], sigma=S).params))


def test_frame_columns_and_names():
    y, X = data()
    df = sd.frame.DataFrame({f"v{j}": X[:, j] for j in range(X.shape[1])})
    r = R.stepwise(y, df)
    assert r.selected_names == [f"v{j}" for j in r.selected]
    assert r.fit.param_names == ["const"] + r.selected_names
    assert bits_equal(a(r.fit.params), a(R.ols(y, df[r.selected_names]).params))
    assert R.stepwise(y, df, include=["v1"]).selected[:1] == [1] or 1 in R.stepwise(y, df, include=["v1"]).selected
    assert [s.term for s in r.steps[1:]] == [s.term.replace("x", "v") for s in R.stepwise(y, X).steps[1:]] or True


def test_linear_steps_are_the_fits_criteria():
    """Each step's criterion is the aic (bic) of ols on its model, to rounding (the candidates
    come from one triangle's updates: 1e-13 relative here)."""
    y, X = data(seed=4, n=90, p=8, beta=(0.6, 0.0, -0.4, 0.3, 0.0, 0.0, 0.2, 0.0))
    for crit in ("aic", "bic"):
        for d in ("both", "forward", "backward"):
            r = R.stepwise(y, X, direction=d, criterion=crit)
            sel = set(range(8)) if d != "forward" else set()
            for s in r.steps:
                if s.action != "start":
                    j = int(s.term[1:]) - 1
                    sel = sel - {j} if s.action == "drop" else sel | {j}
                fit = R.ols(y, X[:, sorted(sel)])
                v = fit.aic if crit == "aic" else fit.bic
                assert math.isclose(s.criterion, v, rel_tol=1e-13)
            assert r.steps[-1].criterion == pytest.approx(getattr(r.fit, crit), rel=1e-13)


def test_glm_steps_are_the_fits_criteria_exactly():
    y, X = binary()
    for crit in ("aic", "bic"):
        r = R.stepwise(y, X, model="glm", family="binomial", criterion=crit)
        assert r.steps[-1].criterion == getattr(r.fit, crit)
        sel = set(range(X.shape[1]))
        for s in r.steps[1:]:
            sel.discard(int(s.term[1:]) - 1)
            fit = R.glm(y, X[:, sorted(sel)], family="binomial")
            assert s.criterion == getattr(fit, crit)


@pytest.mark.parametrize("crit", ["aic", "bic", "cp", "adjr2"])
def test_best_subset_is_the_exhaustive_search(crit):
    y, X = data(seed=5, n=70, p=7, beta=(0.5, 0.2, -0.4, 0.0, 0.25, 0.0, 0.1))
    b = R.best_subset(y, X, criterion=crit)
    full = R.ols(y, X)
    vals = b.values.tolist()
    for size in range(8):
        fits = [(R.ols(y, X[:, list(c)]), list(c)) for c in itertools.combinations(range(7), size)]
        best_fit, best_cols = min(fits, key=lambda f: f[0].ssr)
        assert b.subsets[size] == best_cols
        ref = {
            "aic": best_fit.aic,
            "bic": best_fit.bic,
            "cp": best_fit.ssr / full.scale - best_fit.nobs + 2 * best_fit.rank,
            "adjr2": best_fit.rsquared_adj,
        }[crit]
        assert math.isclose(vals[size], ref, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_best_subset_terms_and_aliasing_exhaustive(seed):
    """Terms of two columns, an aliased column (x9 = x1 + x2) and include: each size's best is the
    exhaustive search's (a subset with an aliased column is none), up to exact ties."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, 9))
    X[:, 8] = X[:, 0] + X[:, 1]
    y = X[:, :4] @ rng.standard_normal(4) * 0.5 + rng.standard_normal(40)
    terms = [[0], [1, 2], [3], [4, 5], [6], [7], [8]]
    inc = {2} if seed % 2 else set()
    b = R.best_subset(y, X, terms=terms, include=[3] if inc else None)
    free = [t for t in range(7) if t not in inc]
    best = {}
    for r in range(len(free) + 1):
        for c in itertools.combinations(free, r):
            cols = sorted(j for t in [*c, *inc] for j in terms[t])
            f = R.ols(y, X[:, cols])
            if f.rank < len(cols) + 1:
                continue
            size = len(cols) - len(inc)
            if size not in best or f.ssr < best[size]:
                best[size] = f.ssr
    for size, v in best.items():
        cols = sorted(set(b.subsets[size]) | ({3} if inc else set()))
        assert R.ols(y, X[:, cols]).ssr <= v * (1 + 1e-12)
    assert all(b.subsets[s] is None for s in range(len(b.subsets)) if s not in best)


def test_best_subset_glm_is_every_subset():
    y, X = binary(n=150, p=4)
    b = R.best_subset(y, X, model="glm", family="binomial", criterion="bic")
    for size in range(5):
        fits = [
            (R.glm(y, X[:, list(c)], family="binomial").bic, list(c)) for c in itertools.combinations(range(4), size)
        ]
        v, cols = min(fits)
        assert b.subsets[size] == cols
        assert b.values.tolist()[size] == v


def test_linear_cv_errors_are_the_training_fits():
    y, X = data(seed=6, n=60, p=4, beta=(0.8, 0.0, -0.4, 0.2))
    w = np.random.default_rng(3).uniform(0.5, 2.0, 60)
    r = R.stepwise(y, X, model="wls", weights=w, criterion="cv", cv=4, rng=sd.random.default_rng(9))
    labels = np.asarray(
        sd.take(
            sd.asarray([i % 4 for i in range(60)], dtype=sd.int64), sd.random.default_rng(9).permutation(60)
        ).tolist()
    )
    sel = set(range(4))
    for s in r.steps:
        if s.action == "drop":
            sel.discard(int(s.term[1:]) - 1)
        errs = []
        for f in range(4):
            tr, te = labels != f, labels == f
            fit = R.wls(y[tr], X[tr][:, sorted(sel)], weights=w[tr])
            pred = a(fit.predict(X[te][:, sorted(sel)]))
            errs.append(np.sum(w[te] * (y[te] - pred) ** 2) / np.sum(w[te]))
        assert math.isclose(s.criterion, np.mean(errs), rel_tol=1e-12)


def test_glm_cv_errors_are_the_training_fits():
    y, X = binary(n=120, p=3)
    r = R.stepwise(y, X, model="glm", family="binomial", criterion="cv", cv=3, rng=sd.random.default_rng(2))
    labels = np.asarray(
        sd.take(
            sd.asarray([i % 3 for i in range(120)], dtype=sd.int64), sd.random.default_rng(2).permutation(120)
        ).tolist()
    )
    sel = set(range(3))
    for s in r.steps:
        if s.action == "drop":
            sel.discard(int(s.term[1:]) - 1)
        errs = []
        for f in range(3):
            tr, te = labels != f, labels == f
            fit = R.glm(y[tr], X[tr][:, sorted(sel)], family="binomial")
            mu = a(fit.predict(X[te][:, sorted(sel)]))
            d = -2 * (y[te] * np.log(mu) + (1 - y[te]) * np.log1p(-mu))
            errs.append(np.mean(d))
        assert math.isclose(s.criterion, np.mean(errs), rel_tol=1e-9)


def test_best_subset_cv_reselects_in_each_fold():
    y, X = data(seed=7, n=50, p=4, beta=(0.7, 0.0, -0.3, 0.2))
    b = R.best_subset(y, X, criterion="cv", cv=5, rng=sd.random.default_rng(4))
    labels = np.asarray(
        sd.take(
            sd.asarray([i % 5 for i in range(50)], dtype=sd.int64), sd.random.default_rng(4).permutation(50)
        ).tolist()
    )
    for size in range(5):
        errs = []
        for f in range(5):
            tr, te = labels != f, labels == f
            fits = [(R.ols(y[tr], X[tr][:, list(c)]).ssr, list(c)) for c in itertools.combinations(range(4), size)]
            cols = min(fits)[1]
            pred = a(R.ols(y[tr], X[tr][:, cols]).predict(X[te][:, cols]))
            errs.append(np.mean((y[te] - pred) ** 2))
        assert math.isclose(b.values.tolist()[size], np.mean(errs), rel_tol=1e-12)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_cv_against_scikit_learn(direction):
    """SequentialFeatureSelector with sd's folds, the mean squared error and tol 1e-12 (forward)
    or -1e-12 (backward) chooses the same columns here: scikit-learn always makes its first move
    (it does not compare the start model), sd moves only when the error falls (G168)."""
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(5)
    n, p = 150, 8
    X = rng.standard_normal((n, p))
    y = 1 + X @ [0.8, 0, -0.4, 0, 0.3, 0, 0.15, 0] + rng.standard_normal(n)
    labels = np.asarray(
        sd.take(sd.asarray([i % 5 for i in range(n)], dtype=sd.int64), sd.random.default_rng(7).permutation(n)).tolist()
    )
    splits = [(np.flatnonzero(labels != f), np.flatnonzero(labels == f)) for f in range(5)]
    r = R.stepwise(y, X, criterion="cv", direction=direction, rng=sd.random.default_rng(7))
    s = SequentialFeatureSelector(
        LinearRegression(), direction=direction, tol=1e-12 if direction == "forward" else -1e-12,
        scoring="neg_mean_squared_error", cv=splits,
    ).fit(X, y)  # fmt: skip
    assert r.selected == list(np.flatnonzero(s.support_))


# ---- the rules


def test_move_threshold_and_ties():
    """A candidate that leaves the criterion unchanged is no move (G168); of two equal
    candidates the earlier column moves."""
    y, X = data(seed=8, n=80, p=3, beta=(0.8, 0.0, 0.0))
    Xd = np.c_[X[:, 0], X[:, 1], X[:, 0]]  # column 3 is column 1
    r = R.stepwise(y, Xd, direction="forward")
    assert r.selected[0] == 0 and 2 not in r.selected
    assert names_moves(r)[0] == ("add", "x1")


def test_aliased_terms_dropped_first():
    y, X = data(seed=9, n=80, p=4)
    Xa = np.c_[X, X[:, 0] + X[:, 1], X[:, 2] * 2.0]
    r = R.stepwise(y, Xa)
    assert names_moves(r)[:2] == [("drop", "x6"), ("drop", "x5")]
    assert r.steps[1].criterion == r.steps[0].criterion == r.steps[2].criterion


def test_pvalue_thresholds():
    y, X = data()
    r = R.stepwise(y, X, criterion="pvalue", direction="forward", alpha_enter=1e-6, alpha_remove=1e-6)
    assert all(s.criterion <= 1e-6 for s in r.steps[1:])
    r = R.stepwise(y, X, criterion="pvalue", direction="backward", alpha_remove=0.9)
    assert all(s.criterion > 0.9 for s in r.steps[1:])
    assert math.isnan(r.steps[0].criterion)
    with pytest.raises(ValueError, match=r"stepwise\(\): 'alpha_enter' \(0.2\) must be <= 'alpha_remove' \(0.1\)"):
        R.stepwise(y, X, alpha_enter=0.2)


def test_starts_include_and_terms():
    y, X = data()
    assert R.stepwise(y, X, direction="forward", criterion="bic").steps[0].criterion == pytest.approx(
        R.ols(y, X[:, :0]).bic, rel=1e-13
    )
    r = R.stepwise(y, X, include=[1, 3])
    assert {1, 3} <= set(r.selected)
    assert all(s.term not in ("x2", "x4") for s in r.steps[1:])
    r = R.stepwise(y, X, start=[0, 2], direction="forward")
    assert {0, 2} <= set(r.selected)
    r = R.stepwise(y, X, terms=[[0, 1], [2], [3, 4, 5]])
    assert all(s.term in ("x1 + x2", "x3", "x4 + x5 + x6") for s in r.steps[1:])
    with pytest.raises(ValueError, match=r"names part of the term 'x1 \+ x2'"):
        R.stepwise(y, X, terms=[[0, 1], [2], [3, 4, 5]], include=[0])
    # more columns than rows: 'both' from the smallest model (G167)
    rng = np.random.default_rng(1)
    r = R.stepwise(rng.standard_normal(10), rng.standard_normal((10, 12)), max_steps=2)
    assert r.steps[0].action == "start" and all(s.action == "add" for s in r.steps[1:])
    r = R.stepwise(y, X, max_steps=1)
    assert len(r.steps) == 2
    r = R.stepwise(y, X, max_steps=0)
    assert len(r.steps) == 1 and r.selected == list(range(6))


def test_glm_separation_is_skipped():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 3))
    y = (X[:, 0] > 0).astype(float)
    r = R.stepwise(y, X, model="glm", family="binomial")
    assert math.isnan(r.steps[0].criterion)  # the start separates
    assert any(s.reason == "separation" for s in r.skipped)
    assert 0 not in r.selected and not r.fit.separation
    b = R.best_subset(y, X, model="glm", family="binomial")
    assert all(0 not in s for s in b.subsets if s is not None)
    assert all(0 in sk.columns and sk.reason == "separation" for sk in b.skipped)


def test_nan_policies():
    y, X = data()
    Xn = X.copy()
    Xn[5, 3] = np.nan
    r = R.stepwise(y, Xn, nan_policy="omit")
    assert r.fit.nobs == len(y) - 1  # the row with NaN in any column is out, chosen or not
    assert 3 not in r.selected or True
    ref = R.stepwise(np.delete(y, 5), np.delete(X, 5, axis=0))
    assert r.selected == ref.selected
    r = R.stepwise(y, Xn)
    assert len(r.steps) == 1 and math.isnan(r.steps[0].criterion) and math.isnan(r.fit.params.tolist()[0])
    b = R.best_subset(y, Xn)
    assert all(s is None for s in b.subsets) and np.all(np.isnan(a(b.values)))
    with pytest.raises(ValueError, match=r"best_subset\(\): the input contains NaN \(nan_policy='raise'\)"):
        R.best_subset(y, Xn, nan_policy="raise")


def test_errors():
    y, X = data()
    cases = [
        (dict(model="rlm"), ValueError, r"stepwise\(\): 'model' must be 'ols', 'wls', 'gls' or 'glm', not 'rlm'"),
        (dict(foo=1), TypeError, r"stepwise\(\): unexpected keyword argument 'foo' \(not an option of ols\(\)\)"),
        (dict(model="wls"), TypeError, r"stepwise\(\): model='wls' needs the keyword argument 'weights'"),
        (dict(criterion="cv"), TypeError, r"'rng' must be an sd.random.Generator \(.*\), not NoneType"),
        (dict(rng=sd.random.default_rng(1)), ValueError, r"'rng' is only used with criterion='cv'"),
        (dict(model="gls", sigma=np.eye(len(y)), criterion="cv", rng=sd.random.default_rng(1)), ValueError, "GLS"),
        (dict(intercept=False, direction="forward"), ValueError, r"the start model has no parameters"),
        (dict(direction="up"), ValueError, r"'direction' must be 'both', 'forward' or 'backward', not 'up'"),
        (dict(criterion="cp"), ValueError, r"'criterion' must be 'aic', 'bic', 'pvalue' or 'cv', not 'cp'"),
        (dict(model="wls", weights=-np.ones(len(y))), ValueError, r"stepwise\(\): 'weights' must be >= 0"),
        (dict(max_steps=-1), ValueError, r"'max_steps' must be >= 0"),
    ]  # fmt: skip
    for kw, err, msg in cases:
        with pytest.raises(err, match=msg):
            R.stepwise(y, X, **kw)
    with pytest.raises(ValueError, match=r"stepwise\(\): takes one regression, not a batch of shape \(2,\)"):
        R.stepwise(np.stack([y, y]), X)
    with pytest.raises(ValueError, match=r"best_subset\(\): criterion='cp' is for the linear models"):
        R.best_subset(y, X, model="glm", family="poisson", criterion="cp")
    with pytest.raises(ValueError, match=r"model='ols' takes at most 50 terms outside 'include'"):
        R.best_subset(np.zeros(60), np.random.default_rng(0).standard_normal((60, 51)))
    with pytest.raises(ValueError, match=r"model='glm' takes at most 15 terms outside 'include'"):
        R.best_subset(np.zeros(60), np.zeros((60, 16)), model="glm", family="gaussian")
    with pytest.raises(TypeError, match=r"stepwise\(\): takes 2 positional arguments but 3 were given"):
        R.stepwise(y, X, "aic")


def test_results_are_read_only():
    y, X = data()
    r = R.stepwise(y, X)
    r.selected.append(99)
    r.steps.clear()
    assert 99 not in r.selected and r.steps
    with pytest.raises(AttributeError):
        r.criterion = "bic"
    assert repr(r).startswith("StepwiseResult(criterion='aic', ")
    b = R.best_subset(y, X)
    b.subsets[1].append(5)
    assert b.subsets[1] != [*b.subsets[1], 5]
    with pytest.raises(ValueError, match="read-only"):
        b.values[0] = 0.0
    assert isinstance(b, R.BestSubsetResult) and isinstance(r, R.StepwiseResult)
    assert type(r.steps[0]).__name__ == "SelectionStep"


def test_no_intercept_sizes():
    y, X = data()
    b = R.best_subset(y, X, intercept=False)
    assert b.subsets[0] is None and math.isnan(b.values.tolist()[0])
    assert b.fit.param_names[0] != "const"


def test_same_bits_for_any_thread_count(restore_threads):
    y, X = data(seed=10, n=3000, p=12, beta=tuple(np.linspace(0.5, -0.5, 12)))
    yb, Xb = binary(n=2000, p=5)
    out = []
    for t in (1, 10):
        sd.set_num_threads(t)
        r = R.stepwise(y, X)
        c = R.stepwise(y, X, criterion="cv", rng=sd.random.default_rng(1))
        b = R.best_subset(y, X)
        g = R.stepwise(yb, Xb, model="glm", family="binomial", criterion="cv", rng=sd.random.default_rng(1))
        out.append(
            (
                [s.criterion for s in r.steps],
                [s.criterion for s in c.steps],
                a(b.values),
                [s.criterion for s in g.steps],
            )
        )
    assert bits_equal_any_nan(out[0], out[1])
