"""Variable selection: stepwise and best_subset (docs/api/regression.md §9.3, docs/design/regression.md
§3.8, decisions G127-G129 and G166-G180).

A selection searches the models of one regression (ols, wls, gls or glm, called with the
model's own options) over the columns of X, as terms (columns that enter and leave together);
the intercept and the `include` terms are always in. Every candidate is that model function on
the candidate's columns, on the same rows (nan_policy='omit' drops a row with NaN in any column
of X once, for every candidate), and the result's `fit` is the function called on the columns
chosen.

The linear candidates are not refitted: the residual sum of squares of any column set comes from
one triangle of the whitened [intercept X y] (stardust._core.regression.select_*; a step's
candidates cost O(m^2) each whatever n), the cross-validation's from one triangle per fold's
training and test rows, and best_subset searches one triangle by leaps and bounds. A GLM's
candidates are its IRLS fits (glm's kernel, the candidates of a step as one batch).
"""

import collections
import inspect
import itertools
import math
import numbers
import operator

from .. import _calls
from .. import _core as _sd
from .. import frame as _frame
from .. import random as _random
from ..linalg import LinAlgError as _LinAlgError
from ..stats import ChiSquared as _ChiSquared
from ..stats import F as _F
from ._linear import (
    _LOG_2PI,
    _SUMLOGW,
    _as_real,
    _check_bool,
    _check_choice,
    _check_policy,
    _has_nan,
    _log,
    _read_y,
    _ro,
    _type_name,
)

_k = _sd.regression

_MODELS = ("ols", "wls", "gls", "glm")
_DIRECTIONS = ("both", "forward", "backward")
_STEP_CRITERIA = ("aic", "bic", "pvalue", "cv")
_SUBSET_CRITERIA = ("aic", "bic", "cv", "cp", "adjr2")
# a step moves when the criterion falls by more than this (AIC, BIC; G168)
_DROP = 1e-7
# best_subset's limits (G177): the terms a search takes
_SUBSET_LINEAR_MAX = 50
_SUBSET_GLM_MAX = 15
# the doubles of the design held at once by a batch of GLM candidates
_GLM_BATCH_DOUBLES = 1 << 22

# the GLM kernel's per-regression scalars (csrc/regression/glm.cpp, enum GStat)
_G_NOBS, _G_RANK, _G_DEV, _G_LLF, _G_CONV, _G_SEP = 0, 1, 2, 6, 9, 10

SelectionStep = collections.namedtuple("SelectionStep", "action term criterion")
SelectionStep.__doc__ = (
    "One step of stepwise(): action 'start', 'add' or 'drop', the term moved (its columns' names joined "
    "by ' + '; None at the start) and the criterion after it (the p-value of the move's test for "
    "criterion='pvalue')."
)
SkippedFit = collections.namedtuple("SkippedFit", "step action term reason")
SkippedFit.__doc__ = (
    "A GLM candidate of stepwise() left out (G173): the step whose candidates it was among (0 for the "
    "first), its action and term, and why: 'separation' or 'not converged'."
)
SkippedSubset = collections.namedtuple("SkippedSubset", "columns reason")
SkippedSubset.__doc__ = (
    "A GLM subset of best_subset() left out (G173): its columns of X and why: 'separation' or 'not converged'."
)


# ---- the results (G178)


class StepwiseResult:
    """The result of stepwise(): selected (the chosen columns' positions in X), selected_names,
    steps (SelectionStep per step, the start first), fit (the model function on the chosen
    columns), criterion, direction and skipped (the GLM candidates left out). Read-only; the lists
    are new at each access."""

    __slots__ = ("_selected", "_names", "_steps", "_fit", "_criterion", "_direction", "_skipped")

    def __init__(self, selected, names, steps, fit, criterion, direction, skipped):
        self._selected = list(selected)
        self._names = list(names)
        self._steps = list(steps)
        self._fit = fit
        self._criterion = criterion
        self._direction = direction
        self._skipped = list(skipped)

    @property
    def selected(self):
        """The chosen columns' positions in X (ascending), the intercept and `include` among them
        as columns of X are; a new list at each access."""
        return list(self._selected)

    @property
    def selected_names(self):
        """The chosen columns' names (X's, or 'x1'...'xp'), as `selected`."""
        return list(self._names)

    @property
    def steps(self):
        """The search's steps: SelectionStep(action, term, criterion), the start first."""
        return list(self._steps)

    @property
    def fit(self):
        """The model function's result on the chosen columns (its coefficients named as X's columns)."""
        return self._fit

    @property
    def criterion(self):
        return self._criterion

    @property
    def direction(self):
        return self._direction

    @property
    def skipped(self):
        """The GLM candidates not chosen because they separate or do not converge (SkippedFit)."""
        return list(self._skipped)

    def __repr__(self):
        moves = sum(1 for s in self._steps if s.action != "start")
        return (
            f"StepwiseResult(criterion={self._criterion!r}, {len(self._selected)} column"
            f"{'' if len(self._selected) == 1 else 's'} selected in {moves} step{'' if moves == 1 else 's'})"
        )


class BestSubsetResult:
    """The result of best_subset(): selected, selected_names, fit and criterion as stepwise's;
    subsets (per size 0, 1, ...: the best subset's columns of X beyond `include`, None for a size
    no subset has), values (the criterion of each size's best: the cross-validation error for
    'cv') and skipped (the GLM subsets left out)."""

    __slots__ = ("_selected", "_names", "_fit", "_criterion", "_subsets", "_values", "_skipped")

    def __init__(self, selected, names, fit, criterion, subsets, values, skipped):
        self._selected = list(selected)
        self._names = list(names)
        self._fit = fit
        self._criterion = criterion
        self._subsets = [None if s is None else list(s) for s in subsets]
        self._values = _ro(values)
        self._skipped = list(skipped)

    @property
    def selected(self):
        """The chosen columns' positions in X (ascending), `include`'s with them."""
        return list(self._selected)

    @property
    def selected_names(self):
        return list(self._names)

    @property
    def fit(self):
        """The model function's result on the chosen columns (its coefficients named as X's columns)."""
        return self._fit

    @property
    def criterion(self):
        return self._criterion

    @property
    def subsets(self):
        """Per size (the columns added to the intercept and `include`, counted by rank): the best
        subset's columns of X beyond `include`, or None; new lists at each access."""
        return [None if s is None else list(s) for s in self._subsets]

    @property
    def values(self):
        """The criterion of each size's best subset (a read-only sd.ndarray; NaN where none)."""
        return self._values

    @property
    def skipped(self):
        """The GLM subsets not chosen because they separate or do not converge (SkippedSubset)."""
        return list(self._skipped)

    def __repr__(self):
        return (
            f"BestSubsetResult(criterion={self._criterion!r}, {len(self._selected)} column"
            f"{'' if len(self._selected) == 1 else 's'} selected, sizes 0-{len(self._subsets) - 1})"
        )


# ---- arguments


def _model_function(model):
    from ._glm import glm
    from ._linear import gls, ols, wls

    return {"ols": ols, "wls": wls, "gls": gls, "glm": glm}[model]


def _model_options(func, model, options):
    """The model function's keywords (G166): its own keyword-only options, but intercept and
    nan_policy, which the selection takes itself. An unknown keyword is a TypeError, as is a
    missing required one (wls's weights, gls's sigma, glm's family)."""
    params = inspect.signature(_model_function(model)).parameters
    allowed = {
        n: p
        for n, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and n not in ("intercept", "nan_policy")
    }
    for name in options:
        if name not in allowed:
            raise TypeError(f"{func}(): unexpected keyword argument '{name}' (not an option of {model}())")
    for name, p in allowed.items():
        if p.default is inspect.Parameter.empty and name not in options:
            raise TypeError(f"{func}(): model='{model}' needs the keyword argument '{name}' (an option of {model}())")
    return dict(options)


def _check_int(v, func, arg, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{func}(): '{arg}' must be >= {lo}")
    if v >= 2**31:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return v


def _check_level(v, func, arg):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {_type_name(v)}")
    x = float(v)
    if not 0 < x < 1:
        raise ValueError(f"{func}(): '{arg}' must be between 0 and 1 (exclusive)")
    return x


def _renamed(func, model, call):
    """call() with the model function's own errors named as the selection's (its argument checks)."""
    head = f"{model}(): "
    try:
        return call()
    except (ValueError, TypeError, OverflowError, _LinAlgError) as e:
        msg = str(e)
        if msg.startswith(head):
            raise type(e)(f"{func}(): " + msg[len(head) :]) from None
        raise


class _Columns:
    """X's columns: their count, names, and X with some of them as the model function takes it
    (a DataFrame's columns as a DataFrame, otherwise an sd array)."""

    def __init__(self, func, X):
        self.frame = isinstance(X, _frame.DataFrame)
        Xa, names = _as_real(X, func, "X")
        if Xa.ndim == 0:
            raise ValueError(f"{func}(): 'X' must have at least 1 dimension")
        if Xa.ndim == 1:
            Xa = _sd.reshape(Xa, (Xa.shape[0], 1))
        self.X = X
        self.array = Xa
        self.p = Xa.shape[-1]
        if names is None or len(names) != self.p:
            names = [f"x{j + 1}" for j in range(self.p)]
        self.names = list(names)

    def take(self, cols):
        cols = list(cols)
        if self.frame and cols:
            labels = list(self.X.columns)
            return self.X[[labels[j] for j in cols]]
        return _sd.take(self.array, _sd.asarray(cols, dtype=_sd.int64), axis=-1)


def _parse_terms(func, terms, names):
    """[[columns of X]] covering every column once (terms=None: a column per term)."""
    p = len(names)
    if terms is None:
        return [[j] for j in range(p)]
    if isinstance(terms, (str, bytes)) or not hasattr(terms, "__iter__"):
        raise TypeError(
            f"{func}(): 'terms' must be a list of terms (each a column or a list of columns), not {_type_name(terms)}"
        )
    out, seen = [], set()
    for t in terms:
        cols = [_column(func, "terms", t, names)] if _is_column(t) else [_column(func, "terms", c, names) for c in t]
        if not cols:
            raise ValueError(f"{func}(): a term must have at least one column")
        for c in cols:
            if c in seen:
                raise ValueError(f"{func}(): the column {names[c]!r} is in more than one term")
            seen.add(c)
        out.append(sorted(cols))
    if len(seen) != p:
        missing = [names[c] for c in range(p) if c not in seen]
        raise ValueError(f"{func}(): 'terms' must put every column of 'X' in a term; missing {missing}")
    return out


def _is_column(c):
    return isinstance(c, str) or (hasattr(c, "__index__") and not isinstance(c, bool))


def _column(func, arg, c, names):
    p = len(names)
    if isinstance(c, str):
        if c not in names:
            raise ValueError(f"{func}(): '{arg}' names a column {c!r} that 'X' does not have")
        return names.index(c)
    if isinstance(c, bool) or not hasattr(c, "__index__"):
        raise TypeError(f"{func}(): '{arg}' must give columns as positions (int) or names (str), not {_type_name(c)}")
    j = c.__index__()
    if not -p <= j < p:
        raise ValueError(f"{func}(): '{arg}' has the column {j}, out of range for {p} columns")
    return j % p


def _term_set(func, arg, v, names, terms):
    """The terms (their indices) that the columns `v` name; every column of a named term must be
    named (a term enters and leaves whole)."""
    if v is None:
        return set()
    if _is_column(v):
        v = [v]
    if not hasattr(v, "__iter__"):
        raise TypeError(f"{func}(): '{arg}' must be a list of columns (positions or names), not {_type_name(v)}")
    cols = {_column(func, arg, c, names) for c in v}
    out = set()
    for i, t in enumerate(terms):
        hit = [c for c in t if c in cols]
        if hit and len(hit) != len(t):
            miss = [names[c] for c in t if c not in cols]
            raise ValueError(
                f"{func}(): '{arg}' names part of the term {_term_name(t, names)!r}; give all of it ({miss} too)"
            )
        if hit:
            out.add(i)
    return out


def _term_name(t, names):
    return " + ".join(names[c] for c in t)


# ---- the evaluators: a model's and its candidates' statistics


class _Stat:
    """A fitted candidate: value (the residual sum of squares or deviance in the fit's units, or
    the cross-validation error), rank, llf (the log-likelihood in the data's units), ok (False: a
    GLM fit that separated or did not converge, with its reason)."""

    __slots__ = ("value", "rank", "llf", "ok", "reason", "used", "folds")

    def __init__(self, value, rank, llf, ok=True, reason=None, used=None, folds=None):
        self.value, self.rank, self.llf, self.ok, self.reason = value, rank, llf, ok, reason
        self.used = used  # the design columns' flags: 1 kept, 0 or 2 aliased
        self.folds = folds  # cv: (the folds' training deviances or None, their test errors)


def _i64(v):
    return _sd.asarray(list(v), dtype=_sd.int64) if v else _sd.zeros((0,), dtype=_sd.int64)


def _folds(func, cv, rng, n):
    """lasso_cv's folds (G46, G172): (0, 1, ..., n - 1 mod cv) permuted by rng.permutation(n)."""
    if cv > n:
        raise ValueError(f"{func}(): 'cv' must be <= the number of rows ({n}), not {cv}")
    labels = _sd.asarray([i % cv for i in range(n)], dtype=_sd.int64)
    return _sd.take(labels, rng.permutation(n))


class _Linear:
    """ols, wls and gls: every candidate from the triangle R of the whitened [intercept X y]
    (§3.8), or with cv the triangles of each fold's training and test rows."""

    def __init__(self, func, full, intercept, cv=None, rng=None):
        _, ys, Xs, w, u, tol, _, _ = full._fitargs
        self.glm = False
        self.full = full
        self.intercept = intercept
        self.rtol = tol
        self.nobs = float(full.nobs)
        self.fy = float(full._fy())
        self.sumlogw = float(full._stat(_SUMLOGW))
        self.logdet = float(full._extra.get("logdet", 0.0))
        # the triangle of [intercept X y - e] (e the full model's residual, in the fit's units,
        # orthogonal to every column): a model's residual sum of squares is that triangle's plus
        # the full model's (refined, summed in double-double), so its rounding scales with the
        # increase over the full model, not with the sum itself
        e = full.wresid if full._fitargs[0] == "gls" else full.resid
        if self.fy != 1.0:
            e = e / self.fy
        self.ssr_full = float(full._ssr_s())
        Rf = full._raw[2]  # the fit's triangle (kept columns), in the fit's units
        fx = full._xfactor()
        if fx is not None:
            Rf = Rf / _sd.expand_dims(fx, axis=-2)
        e2 = _orthogonal(e, Xs, w, u, intercept, Rf, full._raw[5])
        self.R = _k.select_qr(ys - (e if e2 is None else e2), Xs, w, u, intercept, None, 1)
        self.cv = cv
        if cv is not None:
            n = ys.shape[-1]
            labels = _folds(func, cv, rng, n)
            self.T = _k.select_qr(ys, Xs, w, u, intercept, labels, cv)
            self.Rtr = _k.select_merge(self.T, True)
            base = w if w is not None else _sd.ones((n,))
            used = _sd.where(base > 0.0, 1.0, 0.0)
            self.sw, cnt = [], []
            for f in range(cv):
                te = labels == f
                self.sw.append(float(_sd.sum(_sd.where(te, base, 0.0))))
                cnt.append(float(_sd.sum(_sd.where(te, 0.0, used))))
            self.nobs_tr = cnt

    def llf(self, ssr):
        """The result's llf (_LinearResult.llf) of a residual sum of squares in the fit's units."""
        n = self.nobs
        ls = _log(ssr / n) if n else math.nan
        if self.fy != 1.0:
            ls = ls + 2.0 * math.log(self.fy)
        v = -n / 2 * (_LOG_2PI + ls + 1.0) + self.sumlogw / 2
        return v - self.logdet / 2

    @staticmethod
    def _packed(cands):
        cols = [c for cs, _ in cands for c in cs]
        ptr = [0]
        for cs, _ in cands:
            ptr.append(ptr[-1] + len(cs))
        return _i64(cols), _i64(ptr), _i64([1 if d else 0 for _, d in cands])

    def evaluate(self, sel, cands, all_rows=False):
        """(the model's _Stat, [each candidate's _Stat]): sel the model's design columns
        (ascending), cands (design columns, drop) pairs. With cv (and not all_rows) the value is
        the cross-validation error in the data's units; else the residual sum of squares in the
        fit's units. The model's `used` flags (1 kept, 2 aliased) are those of all the rows."""
        cc, cp, cd = self._packed(cands)
        cur, crk, cin, vals, rks = _k.select_eval(
            self.R, None, _sd.asarray([self.nobs]), self.rtol, _i64(sel), cc, cp, cd
        )
        used = cin[0].tolist()
        if self.cv is None or all_rows:
            f = self.ssr_full
            v = float(cur[0]) + f
            out = [_Stat(a + f, b, self.llf(a + f)) for a, b in zip(vals[0].tolist(), rks[0].tolist())]
            return _Stat(v, float(crk[0]), self.llf(v), used=used), out
        rank = float(crk[0])
        rks0 = rks[0].tolist()
        cur, _, _, vals, _ = _k.select_eval(
            self.Rtr, self.T, _sd.asarray(self.nobs_tr), self.rtol, _i64(sel), cc, cp, cd
        )
        vt = vals.tolist()
        out = [_Stat(self.error([vt[f][q] for f in range(self.cv)]), rks0[q], math.nan) for q in range(len(cands))]
        return _Stat(self.error(cur.tolist()), rank, math.nan, used=used), out

    def error(self, sse):
        """The folds' mean of the test rows' weighted mean squared error (G172; a fold whose test
        rows all have weight 0 measures nothing, G107), in the data's units."""
        m = [s / w for s, w in zip(sse, self.sw) if w > 0]
        return self.fy * self.fy * (sum(m) / len(m)) if m else math.nan

    def fold_sse(self, f, sel):
        cur = _k.select_eval(
            self.Rtr[f], self.T[f], _sd.asarray([self.nobs_tr[f]]), self.rtol, _i64(sel), *self._packed([])
        )[0]
        return float(cur[0])


def _orthogonal(e, Xs, w, u, intercept, R, kept):
    """The full model's residual e (the fit's units, unwhitened for ols and wls) after one step of
    the corrected seminormal equations: d = D^T e (the whitened design D, in double-double), then
    e - D (R^T R)^-1 d (once rounded), with R the triangle's design block. The refined fit leaves
    D^T e at kappa u of |D| |e| (its coefficients' rounding through D^T D); the step takes it to
    about kappa^2 u of that, which the residual sums of squares of the models far from the full
    one need (they add e's projections). None when no column is kept."""
    n = e.shape[-1]
    sw = None if w is None else _sd.sqrt(w)
    cols = []
    if intercept:
        cols.append(u if u is not None else _sd.ones((n,)))
    cols += [Xs[..., j] for j in range(Xs.shape[-1])]
    keep = [j for j, k in enumerate(kept.tolist()) if k]
    if not keep:
        return None
    D = _sd.stack([cols[j] for j in keep], axis=-1)
    if sw is not None:
        D = D * _sd.expand_dims(sw, axis=-1)
    et = e if sw is None else e * sw
    d = _k.apply("stepwise", _sd.matrix_transpose(D), et, False, None)[0]
    Rk = _sd.take(_sd.take(R, _sd.asarray(keep, dtype=_sd.int64), axis=0), _sd.asarray(keep, dtype=_sd.int64), axis=1)
    from .. import linalg as _linalg

    t = _linalg.solve_triangular(_sd.matrix_transpose(Rk), _sd.expand_dims(d, axis=-1), upper=False)
    db = _linalg.solve_triangular(Rk, t, upper=True)[..., 0]
    e2 = _k.apply("stepwise", D, db, False, et)[1]
    if sw is not None:
        e2 = _sd.where(w > 0.0, e2 / _sd.where(w > 0.0, sw, 1.0), e)
    return e2


class _GLM:
    """glm: every candidate is the IRLS fit of glm's kernel on its columns (the rows, options
    and units of glm() on those columns, so its llf is that fit's), the candidates of a call as
    batches of the same column count."""

    def __init__(self, func, full, intercept, cv=None, rng=None):
        self.glm = True
        self.full = full
        self.fa = full._fitargs
        self.intercept = intercept
        self.nobs = float(full.nobs)
        sh = full._extra.get("llshift")
        self.llshift = None if sh is None else float(sh)
        d2 = full._extra.get("dev2")
        self.dev2 = 1.0 if d2 is None else float(d2)
        self.cv = cv
        self.cache = {}
        if cv is not None:
            n = self.fa["y"].shape[-1]
            labels = _folds(func, cv, rng, n)
            aux = self.fa["aux"]
            freq = aux[..., 1, :]
            folds = []
            test = []
            for f in range(cv):
                te = _sd.where(labels == f, 1.0, 0.0)
                test.append(te)
                folds.append(_sd.stack([aux[..., 0, :], freq * (1.0 - te), aux[..., 2, :], aux[..., 3, :]], axis=-2))
            self.aux_cv = _sd.stack(folds, axis=0)  # (cv, 4, n): the training rows' frequency weights
            te = _sd.stack(test, axis=0)
            self.test_f = te * freq  # (cv, n)
            self.test_fm = self.test_f * (aux[..., 0, :] * aux[..., 2, :])  # x the prior weights
            self.fold_den = _sd.sum(self.test_fm, axis=-1).tolist()

    def _fold_used(self):
        """Per fold: whether its test rows have weight (G107)."""
        return [d > 0 for d in self.fold_den]

    def _fit(self, colsets, cvfolds):
        """[(raw, index)] of the kernel's fits of each X column set (a batch per column count)."""
        fa = self.fa
        X = fa["X"]
        n = X.shape[-2]
        aux = self.aux_cv if cvfolds else fa["aux"]
        nf = self.cv if cvfolds else 1
        out = [None] * len(colsets)
        groups = collections.defaultdict(list)
        for i, cs in enumerate(colsets):
            groups[len(cs)].append(i)
        for k, idx in sorted(groups.items()):
            chunk = max(1, _GLM_BATCH_DOUBLES // max(1, n * (k + 8 * nf)))
            for s in range(0, len(idx), chunk):
                part = idx[s : s + chunk]
                if k:
                    Xb = _sd.stack(
                        [_sd.take(X, _sd.asarray(colsets[i], dtype=_sd.int64), axis=-1) for i in part], axis=0
                    )
                else:
                    Xb = _sd.zeros((len(part), n, 0))
                if cvfolds:
                    Xb = _sd.expand_dims(Xb, axis=1)
                raw = _k.glm_fit(
                    "glm",
                    fa["y"],
                    Xb,
                    aux,
                    None,
                    0,
                    fa["family_code"],
                    fa["link_code"],
                    fa["has"][0],
                    True if cvfolds else fa["has"][1],
                    fa["has"][2],
                    fa["has"][3],
                    self.intercept,
                    fa["rtol"],
                    0,
                    -1,
                    fa["max_iter"],
                    fa["tol"],
                    fa["smode"],
                    fa["svalue"],
                )
                for b, i in enumerate(part):
                    out[i] = (raw, b)
        return out

    def stats(self, colsets, all_rows=False):
        """_Stat per X column set: on every row the deviance (fit's units) as value; with cv (and not
        all_rows) the cross-validation error, with per-fold training deviances and test errors."""
        folds = self.cv is not None and not all_rows
        keys = [(folds, tuple(cs)) for cs in colsets]
        need = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if need:
            for key, (raw, b) in zip(need, self._fit([list(k[1]) for k in need], folds)):
                self.cache[key] = self._cv_stat(raw, b) if folds else self._stat(raw, b)
        return [self.cache[k] for k in keys]

    def _stat(self, raw, b):
        st = raw[6][b].tolist()
        ok, reason = True, None
        if st[_G_SEP] == 1.0:
            ok, reason = False, "separation"
        elif st[_G_CONV] != 1.0:
            ok, reason = False, "not converged"
        llf = st[_G_LLF] if self.llshift is None else st[_G_LLF] + self.llshift
        used = [1 if v else 0 for v in raw[5][b].tolist()]
        return _Stat(st[_G_DEV], st[_G_RANK], llf, ok, reason, used=used)

    def _cv_stat(self, raw, b):
        rows = raw[6][b].tolist()  # (cv, nstat)
        ok, reason = True, None
        if any(r[_G_SEP] == 1.0 for r in rows):
            ok, reason = False, "separation"
        elif any(r[_G_CONV] != 1.0 for r in rows):
            ok, reason = False, "not converged"
        rd = raw[9][b]  # (cv, n): sign(y - mu) sqrt(m d), the test rows' too
        num = _sd.sum(self.test_f * rd * rd, axis=-1).tolist()
        errs = [self.dev2 * a / d if d > 0 else None for a, d in zip(num, self.fold_den)]
        m = [e for e in errs if e is not None]
        err = sum(m) / len(m) if m else math.nan
        if ok and math.isnan(err):
            ok, reason = False, "not converged"
        return _Stat(err, rows[0][_G_RANK], math.nan, ok, reason, folds=([r[_G_DEV] for r in rows], errs))

    def evaluate(self, sel, cands, all_rows=False):
        j0 = 1 if self.intercept else 0
        cur_x = [c - j0 for c in sel if c >= j0]
        sets = [cur_x]
        for cs, drop in cands:
            xs = [c - j0 for c in cs if c >= j0]
            sets.append([c for c in cur_x if c not in xs] if drop else sorted(cur_x + xs))
        st = self.stats(sets, all_rows)
        cur = st[0]
        # the model's kept columns (its fit on every row) as flags of the whole design's columns
        kept = (st[0] if self.cv is None or all_rows else self.stats([cur_x], True)[0]).used
        used = [0] * (j0 + self.fa["X"].shape[-1])
        if self.intercept:
            used[0] = kept[0]
        for i, c in enumerate(cur_x):
            used[j0 + c] = kept[j0 + i]
        return _Stat(cur.value, cur.rank, cur.llf, cur.ok, cur.reason, used, cur.folds), st[1:]


# ---- the criteria


def _criterion_value(ev, crit, s):
    """A model's criterion value (AIC, BIC as the result's aic and bic; the cv error)."""
    if crit == "cv":
        return s.value if s.ok else math.nan
    if not s.ok:
        return math.nan
    k = 2.0 if crit == "aic" else _log(ev.nobs)
    return -2.0 * s.llf + k * s.rank


def _delta(ev, crit, cur, cand):
    """cand's criterion minus cur's, formed without the criterion's large constants."""
    if crit == "cv":
        return cand.value - cur.value
    k = 2.0 if crit == "aic" else _log(ev.nobs)
    dr = cand.rank - cur.rank
    if ev.glm:
        a, b = cand.llf, cur.llf
    else:
        a, b = cand.value, cur.value
        if a == b:
            return k * dr
        if b == 0.0 or a == 0.0:
            return math.inf if a > b else -math.inf
        return ev.nobs * math.log(a / b) + k * dr
    if a == b:
        return k * dr
    return -2.0 * (a - b) + k * dr


def _pvalue(ev, big, small):
    """The p-value of small within big (G170): the partial F of the linear models (big's scale),
    the likelihood ratio of a GLM."""
    dr = big.rank - small.rank
    if not dr > 0 or not (big.ok and small.ok):
        return math.nan
    if ev.glm:
        lr = -2.0 * (small.llf - big.llf)
        if math.isnan(lr):
            return math.nan
        return float(_ChiSquared(df=dr).ccdf(max(lr, 0.0)))
    df = ev.nobs - big.rank
    if not df > 0 or not big.value > 0:
        return math.nan
    f = ((small.value - big.value) / dr) / (big.value / df)
    return float(_F(dfn=dr, dfd=df).ccdf(max(f, 0.0)))


# ---- shared setup


def _setup(func, y, X, model, options, intercept, nan_policy, terms, include):
    _check_choice(model, func, "model", _MODELS)
    _check_bool(intercept, func, "intercept")
    _check_policy(nan_policy, func)
    opts = _model_options(func, model, options)
    cols = _Columns(func, X)
    ya = _read_y(y, func)
    if ya.ndim == 0:
        raise ValueError(f"{func}(): 'y' must have at least 1 dimension")
    if ya.ndim > 1 or cols.array.ndim > 2:
        shape = tuple(ya.shape[:-1]) if ya.ndim > 1 else tuple(cols.array.shape[:-2])
        raise ValueError(f"{func}(): takes one regression, not a batch of shape {shape}")
    tl = _parse_terms(func, terms, cols.names)
    inc = _term_set(func, "include", include, cols.names, tl)
    # NaN in a column of X: every candidate leaves the row out under 'omit' (§3.8), so y carries it
    yfit = y
    if nan_policy == "omit" and cols.p and _has_nan(cols.array):
        bad = _sd.any(_sd.isnan(cols.array), axis=-1)
        if ya.shape[-1] == bad.shape[-1]:
            yfit = _sd.where(bad, math.nan, ya)
    fn = _model_function(model)

    def call(cs):
        r = _renamed(func, model, lambda: fn(yfit, cols.take(cs), intercept=intercept, nan_policy=nan_policy, **opts))
        names = (["const"] if intercept else []) + [cols.names[c] for c in cs]
        if len(r._param_names) == len(names):
            r._param_names = names
        return r

    full = call(list(range(cols.p)))
    if full._batch:
        raise ValueError(f"{func}(): takes one regression, not a batch of shape {tuple(full._batch)}")
    return cols, tl, inc, full, call


def _check_cv(func, model, crit, cv, rng):
    """The folds' count for criterion='cv' (an sd.random Generator required), else None (and no rng)."""
    if crit != "cv":
        if rng is not None:
            raise ValueError(f"{func}(): 'rng' is only used with criterion='cv' (the other criteria draw no folds)")
        return None
    cv = _check_int(cv, func, "cv", 2)
    if not isinstance(rng, _random.Generator):
        raise TypeError(
            f"{func}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {_type_name(rng)}"
        )
    if model == "gls":
        raise ValueError(
            f"{func}(): criterion='cv' takes ols, wls or glm: the GLS whitening mixes the rows the folds split"
        )
    return cv


def _evaluator(func, model, full, intercept, cv, rng):
    cls = _GLM if model == "glm" else _Linear
    return cls(func, full, intercept, cv, rng)


def _design(intercept, terms, chosen):
    """The design columns (ascending) of the model with the terms `chosen`."""
    j0 = 1 if intercept else 0
    cs = sorted(c for t in chosen for c in terms[t])
    return ([0] if intercept else []) + [j0 + c for c in cs]


def _x_columns(terms, chosen):
    return sorted(c for t in chosen for c in terms[t])


# ---- stepwise


@_calls.function
def stepwise(
    y,
    X,
    /,
    *,
    model="ols",
    direction="both",
    criterion="aic",
    start=None,
    include=None,
    terms=None,
    alpha_enter=0.05,
    alpha_remove=0.10,
    cv=5,
    rng=None,
    max_steps=1000,
    intercept=True,
    nan_policy="propagate",
    **options,
):
    """Stepwise selection of X's columns (terms) for the model `model` ('ols', 'wls', 'gls' or
    'glm', its options as keywords): from the start model, each step adds or drops the term that
    improves the criterion most ('aic', 'bic', 'pvalue' or 'cv'), until none does
    (docs/api/regression.md §9.3)."""
    func = "stepwise"
    _check_choice(direction, func, "direction", _DIRECTIONS)
    _check_choice(criterion, func, "criterion", _STEP_CRITERIA)
    a_in = _check_level(alpha_enter, func, "alpha_enter")
    a_out = _check_level(alpha_remove, func, "alpha_remove")
    if a_in > a_out:
        raise ValueError(f"{func}(): 'alpha_enter' ({a_in:g}) must be <= 'alpha_remove' ({a_out:g})")
    max_steps = _check_int(max_steps, func, "max_steps", 0)
    _check_choice(model, func, "model", _MODELS)
    cv = _check_cv(func, model, criterion, cv, rng)
    cols, terms_l, inc, full, call = _setup(func, y, X, model, options, intercept, nan_policy, terms, include)
    K = len(terms_l)
    if start is None:
        if direction == "forward":
            chosen = set(inc)
        elif direction == "backward":
            chosen = set(range(K))
        else:
            # from the full model, or from the smallest when the full model leaves no residual
            # degree of freedom (G167)
            k_full = cols.p + (1 if intercept else 0)
            chosen = set(inc) if k_full >= full.nobs else set(range(K))
    else:
        chosen = _term_set(func, "start", start, cols.names, terms_l) | inc
    if not intercept and not chosen:
        raise ValueError(
            f"{func}(): the start model has no parameters (intercept=False and no columns); give 'start' or 'include'"
        )
    tname = [_term_name(t, cols.names) for t in terms_l]

    def result(chosen, steps, skipped):
        xc = _x_columns(terms_l, chosen)
        return StepwiseResult(xc, [cols.names[c] for c in xc], steps, call(xc), criterion, direction, skipped)

    if full._isnan():  # NaN data under 'propagate': no step, the start model's NaN fit (G175)
        return result(chosen, [SelectionStep("start", None, math.nan)], [])
    ev = _evaluator(func, model, full, intercept, cv, rng)
    j0 = 1 if intercept else 0
    order = sorted(range(K), key=lambda t: terms_l[t][0])
    drops_ok = direction in ("both", "backward")
    adds_ok = direction in ("both", "forward")
    steps, skipped = [], []
    visited = {frozenset(chosen)}
    moves = 0
    while True:
        cands = []  # in the terms' column order: a tie goes to the earlier column (G168)
        for t in order:
            if t in chosen:
                if drops_ok and t not in inc and (intercept or len(chosen) > 1):
                    cands.append((t, True))
            elif adds_ok:
                cands.append((t, False))
        cur, stats = ev.evaluate(
            _design(intercept, terms_l, chosen), [([j0 + c for c in terms_l[t]], d) for t, d in cands]
        )
        if not steps:
            steps.append(
                SelectionStep(
                    "start", None, math.nan if criterion == "pvalue" else _criterion_value(ev, criterion, cur)
                )
            )
        if moves >= max_steps:
            break
        # a term of the model that adds no rank goes first, the last one first (R's step)
        zero = []
        if drops_ok:
            zero = [t for t in sorted(chosen, key=lambda t: terms_l[t][0], reverse=True)
                    if t not in inc and not any(cur.used[j0 + c] == 1 for c in terms_l[t])]  # fmt: skip
        if zero:
            t = zero[0]
            chosen = chosen - {t}
            visited.add(frozenset(chosen))
            steps.append(SelectionStep("drop", tname[t], math.nan if criterion == "pvalue" else steps[-1].criterion))
            moves += 1
            continue
        for (t, d), s in zip(cands, stats):
            if not s.ok:
                skipped.append(SkippedFit(moves, "drop" if d else "add", tname[t], s.reason))
        pick, value = _step_choice(ev, criterion, cur, cands, stats, a_in, a_out)
        if pick is None:
            break
        t, d = pick
        nxt = chosen - {t} if d else chosen | {t}
        if frozenset(nxt) in visited:  # back at a model passed before (G170)
            break
        chosen = nxt
        visited.add(frozenset(chosen))
        steps.append(SelectionStep("drop" if d else "add", tname[t], value))
        moves += 1
    return result(chosen, steps, skipped)


def _step_choice(ev, criterion, cur, cands, stats, a_in, a_out):
    """((term, drop) of the move, the step's criterion value), or (None, None) to stop."""
    if criterion == "pvalue":
        # a term to drop first: the largest p above alpha_remove; else the smallest p of a term
        # to add, at most alpha_enter (G170)
        best = None
        for (t, d), s in zip(cands, stats):
            if d and s.ok:
                pv = _pvalue(ev, cur, s)
                if pv > a_out and (best is None or pv > best[0]):
                    best = (pv, t, d)
        if best is None:
            for (t, d), s in zip(cands, stats):
                if not d and s.ok:
                    pv = _pvalue(ev, s, cur)
                    if pv <= a_in and (best is None or pv < best[0]):
                        best = (pv, t, d)
        return (None, None) if best is None else ((best[1], best[2]), best[0])
    best = None
    for (t, d), s in zip(cands, stats):
        if not s.ok:
            continue
        # from a start that separated or did not converge (no criterion): the best candidate
        dv = _delta(ev, criterion, cur, s) if cur.ok else _criterion_value(ev, criterion, s)
        if math.isnan(dv):
            continue
        if best is None or dv < best[0]:
            best = (dv, t, d, s)
    if best is None:
        return None, None
    if cur.ok and not best[0] < (-_DROP if criterion in ("aic", "bic") else 0.0):  # G168
        return None, None
    return (best[1], best[2]), _criterion_value(ev, criterion, best[3])


# ---- best_subset


@_calls.function
def best_subset(
    y,
    X,
    /,
    *,
    model="ols",
    criterion="aic",
    max_size=None,
    include=None,
    terms=None,
    cv=5,
    rng=None,
    intercept=True,
    nan_policy="propagate",
    **options,
):
    """The best subset of X's columns (terms) of each size for the model `model` ('ols', 'wls',
    'gls' or 'glm', its options as keywords), and among the sizes the one the criterion ('aic',
    'bic', 'cv', and for the linear models 'cp' and 'adjr2') prefers (docs/api/regression.md
    §9.3). The linear models search by leaps and bounds, a GLM fits every subset."""
    func = "best_subset"
    _check_choice(criterion, func, "criterion", _SUBSET_CRITERIA)
    if criterion in ("cp", "adjr2") and model == "glm":
        raise ValueError(f"{func}(): criterion={criterion!r} is for the linear models (ols, wls, gls), not glm")
    if max_size is not None:
        max_size = _check_int(max_size, func, "max_size", 0)
    _check_choice(model, func, "model", _MODELS)
    cv = _check_cv(func, model, criterion, cv, rng)
    cols, terms_l, inc, full, call = _setup(func, y, X, model, options, intercept, nan_policy, terms, include)
    free = [t for t in range(len(terms_l)) if t not in inc]
    limit = _SUBSET_GLM_MAX if model == "glm" else _SUBSET_LINEAR_MAX
    if len(free) > limit:
        how = "fits every subset" if model == "glm" else "searches by leaps and bounds"
        raise ValueError(
            f"{func}(): model='{model}' takes at most {limit} terms outside 'include' ({how}), not {len(free)}"
        )
    nfree = sum(len(terms_l[t]) for t in free)
    max_size = nfree if max_size is None else min(max_size, nfree)
    base = set(inc)

    def result(chosen, subsets, values, skipped, fit=None):
        xc = _x_columns(terms_l, chosen)
        fit = call(xc) if fit is None else fit
        return BestSubsetResult(xc, [cols.names[c] for c in xc], fit, criterion, subsets, values, skipped)

    if full._isnan():  # (G175): no search, the full model's NaN fit
        return result(set(range(len(terms_l))), [None] * (max_size + 1), _sd.full((max_size + 1,), math.nan), [], full)
    ev = _evaluator(func, model, full, intercept, cv, rng)
    if model == "glm":
        sets, values, skipped = _subsets_glm(ev, criterion, terms_l, free, base, intercept, max_size)
    else:
        sets, values = _subsets_linear(ev, criterion, terms_l, free, base, intercept, max_size)
        skipped = []
    values = [math.nan if (v is None or sets[i] is None) else v for i, v in enumerate(values)]
    pick = None  # the size of the best value; ties to the smaller size
    for size, v in enumerate(values):
        if math.isnan(v):
            continue
        if pick is None or (v > values[pick] if criterion == "adjr2" else v < values[pick]):
            pick = size
    chosen = base | set(sets[pick]) if pick is not None else set(base)
    if not intercept and not chosen:
        chosen = set(range(len(terms_l)))
    subsets = [None if s is None else _x_columns(terms_l, s) for s in sets]
    return result(chosen, subsets, _sd.asarray(values), skipped)


def _linear_values(ev, criterion, rss, ranks, full_stat):
    """The criterion of each size's best subset from its residual sum of squares (fit's units)."""
    out = []
    df_full = ev.nobs - full_stat.rank
    s2 = full_stat.value / df_full if df_full > 0 else math.nan
    tss = ev.full._tss()
    icpt = 1.0 if ev.intercept else 0.0
    for r, k in zip(rss, ranks):
        if r is None:
            out.append(None)
        elif criterion in ("aic", "bic"):
            out.append(_criterion_value(ev, criterion, _Stat(r, k, ev.llf(r))))
        elif criterion == "cp":  # leaps' Cp: the full model's scale
            out.append(r / s2 - ev.nobs + 2.0 * k if s2 > 0 else math.nan)
        elif k == icpt:  # adjr2 of the intercept alone: 0 (the result's rule)
            out.append(0.0 if tss > 0 else math.nan)
        else:  # adjr2: the result's rsquared_adj
            df = ev.nobs - k
            out.append(1.0 - (ev.nobs - icpt) / df * (r / tss) if df > 0 and tss > 0 else math.nan)
    return out


def _leaps(R, nobs, rtol, terms_l, free, base_cols, j0, max_size, order):
    """([free terms] or None per size, [rss or None per size]) of one triangle."""
    bc, bp = [], [0]
    for t in free:
        bc.extend(j0 + c for c in terms_l[t])
        bp.append(len(bc))
    rss, sets, _ = _k.select_subsets(R, nobs, rtol, _i64(base_cols), _i64(bc), _i64(bp), _i64(order), max_size)
    rl, sl = rss.tolist(), sets.tolist()
    out_sets = [
        None if math.isnan(rl[k]) else [free[b] for b in range(len(free)) if sl[k][b]] for k in range(max_size + 1)
    ]
    return out_sets, [None if math.isnan(v) else v for v in rl]


def _subsets_linear(ev, criterion, terms_l, free, base, intercept, max_size):
    j0 = 1 if intercept else 0
    base_cols = _design(intercept, terms_l, base)
    # the search's starting order: first the terms whose drop from the full model costs most
    full_stat, drops = ev.evaluate(
        _design(intercept, terms_l, set(range(len(terms_l)))),
        [([j0 + c for c in terms_l[t]], True) for t in free],
        True,
    )
    gain = [s.value - full_stat.value for s in drops]
    order = sorted(range(len(free)), key=lambda b: (-gain[b] if not math.isnan(gain[b]) else math.inf, b))
    # an aliased column of the base takes no part (its triangle row is not the base's span)
    bstat = ev.evaluate(base_cols, [], True)[0]
    base_cols = [c for c in base_cols if bstat.used[c] == 1]
    base_rank = bstat.rank
    empty = not intercept and not base  # size 0 is no model
    sets, rss = _leaps(ev.R[0], ev.nobs, ev.rtol, terms_l, free, base_cols, j0, max_size, order)
    rss = [None if r is None else r + ev.ssr_full for r in rss]
    if empty:
        sets[0] = rss[0] = None
    if ev.cv is None:
        return sets, _linear_values(ev, criterion, rss, [base_rank + k for k in range(max_size + 1)], full_stat)
    # cv (G172): each fold's best subset of each size from its training rows, measured on its test
    # rows; the folds' mean per size, and the size's best of all the rows
    sse = [[None] * ev.cv for _ in range(max_size + 1)]
    for f in range(ev.cv):
        fsets, _ = _leaps(ev.Rtr[f], ev.nobs_tr[f], ev.rtol, terms_l, free, base_cols, j0, max_size, order)
        for k, s in enumerate(fsets):
            if s is not None and not (k == 0 and empty) and ev.sw[f] > 0:
                sse[k][f] = ev.fold_sse(f, _design(intercept, terms_l, base | set(s)))
    values = []
    for k in range(max_size + 1):
        if any(v is None for v, w in zip(sse[k], ev.sw) if w > 0):
            values.append(None)  # a fold without a subset of this size
        else:
            values.append(ev.error([0.0 if v is None else v for v in sse[k]]))
    return sets, values


def _subsets_glm(ev, criterion, terms_l, free, base, intercept, max_size):
    """Every subset of the free terms fitted (G177). A size's best: the lowest criterion (aic, bic),
    or for cv the lowest deviance (each fold's training deviance in the fold, every row's for the
    size's subset); a subset with an aliased column is not one (as the linear search)."""
    base_x = _x_columns(terms_l, base)
    j0 = 1 if intercept else 0
    subsets, xsets = [], []
    for r in range(len(free) + 1):
        for combo in itertools.combinations(free, r):
            if sum(len(terms_l[t]) for t in combo) > max_size or (not intercept and not base_x and not combo):
                continue
            subsets.append(combo)
            xsets.append(sorted(base_x + [c for t in combo for c in terms_l[t]]))
    allrows = ev.stats(xsets, True)
    folds = ev.stats(xsets) if ev.cv is not None else allrows
    base_rank = ev.stats([base_x], True)[0].rank if (intercept or base_x) else 0.0
    skipped = []
    best = [None] * (max_size + 1)  # (key, subset index)
    fold_best = [[None] * (ev.cv or 0) for _ in range(max_size + 1)]
    for i, (combo, xs) in enumerate(zip(subsets, xsets)):
        a, s = allrows[i], folds[i]
        if not (a.ok and s.ok):
            skipped.append(SkippedSubset(xs, a.reason if not a.ok else s.reason))
            continue
        if any(not a.used[j0 + xs.index(c)] for t in combo for c in terms_l[t]):
            continue  # a subset with an aliased column (G177)
        size = int(a.rank - base_rank)
        if not 0 <= size <= max_size:
            continue
        key = a.value if criterion == "cv" else _criterion_value(ev, criterion, a)
        if not math.isnan(key) and (best[size] is None or key < best[size][0]):
            best[size] = (key, i)
        if ev.cv is not None:
            for f, (dev, err) in enumerate(zip(*s.folds)):
                if err is not None and (fold_best[size][f] is None or dev < fold_best[size][f][0]):
                    fold_best[size][f] = (dev, err)
    sets = [None if b is None else list(subsets[b[1]]) for b in best]
    if ev.cv is None:
        values = [None if b is None else b[0] for b in best]
    else:
        values = []
        for k in range(max_size + 1):
            fb = [v for v in fold_best[k]]
            used = [v for v, d in zip(fb, ev._fold_used()) if d]
            values.append(None if (not used or any(v is None for v in used)) else sum(v[1] for v in used) / len(used))
    return sets, values, skipped
