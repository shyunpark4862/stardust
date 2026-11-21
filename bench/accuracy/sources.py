"""The sweeps of each area and how a sweep is measured.

A sweep is one method of a reference file: its inputs, the reference (hi + lo), the
guaranteed region, the bound's scale (the multiplier of the 1e-14 bound at each point:
max(1, kappa) for the inverses, max(1, M / |f|) near the zeros of the oscillatory Bessel and
Airy functions and of the complex error functions, else 1) and the area's contract measure.
Every area reads its references and judges its contract with the functions its tests use
(tests/<area>/refdata.py; special: tests/special/test_reference.py), so a table and the tests
agree about what "within the contract" means.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from bench import areas as _areas

from .common import Custom, isolated, ulps

# the areas measured, in bench/areas.py's order (frame and random have no references yet)
AREAS = tuple(a for a in _areas.AREAS if a not in ("frame", "random"))
GUARANTEED = 1e-14
SWEEP_SECONDS = 30.0  # a reference sweep's comparator call: its time limit (common.isolated)
REG_SECONDS = 120.0  # a regression problem's comparator fit


def check_areas(names):
    """The chosen areas in AREAS order (all when none); ValueError for an unknown one."""
    return _areas.check(names, AREAS)


@dataclasses.dataclass
class Sweep:
    area: str
    name: str  # the reference file's name
    load: Callable  # () -> (inputs, hi, lo, guaranteed, scale)
    call: Callable  # (inputs) -> numpy result
    contract: Callable  # (name, got, hi, lo, scale) -> error in contract units (GUARANTEED is the bound)
    loose_contract: Callable  # same, for the points outside the guaranteed region
    loose_bound: float | None  # the bound outside the guaranteed region (None: none documented)
    loose_label: str  # "loose" or "mixed" (the measure outside the guaranteed region)
    mixed: tuple = ()  # the regions whose contract is the mixed measure |error| / max(1, |value|)
    comparators: Callable | None = None  # () -> {library: fn(inputs) -> values, or "not run"} (compare.py)

    @property
    def id(self):
        return f"{self.area}/{self.name}"


def _cmp(area, name):
    def fn():
        from . import compare

        return getattr(compare, area)(name)

    return fn


def special_sweeps():
    import stardust as sd
    from tests.special import refdata
    from tests.special import test_reference as T

    out = []
    for name in refdata.names():

        def load(name=name):
            xs, hi, lo, g = refdata.load(name)
            return xs, hi, lo, g, refdata.scale(name)

        def call(xs, name=name):
            return np.asarray(T.function(name)(*(sd.asarray(x) for x in xs)))

        def contract(name, got, hi, lo, scale):
            return T.contract_error(got, hi, lo) / scale

        if name in T.MIXED:

            def loose(name, got, hi, lo, scale):
                return T.mixed_error(got, hi, lo)

            bound, label, mixed = GUARANTEED, "mixed", ("mixed",)
        elif name in T.LOOSE:
            loose, bound, label, mixed = contract, T.LOOSE[name], "loose", ()
        else:  # no documented bound outside the guaranteed region: a point there fails (as the test)

            def loose(name, got, hi, lo, scale):
                return np.full(np.shape(hi), np.inf)

            bound, label, mixed = GUARANTEED, "loose", ()
        out.append(Sweep("special", name, load, call, contract, loose, bound, label, mixed, _cmp("special", name)))
    return out


def stats_sweeps():
    from tests.stats import refdata
    from tests.stats import test_reference as T

    out = []
    for name in refdata.names():

        def load(name=name):
            return refdata.load(name)

        def call(xs, name=name):
            return np.asarray(refdata.call(name, xs))

        def contract(name, got, hi, lo, scale):
            return refdata.contract_error(name, got, hi, lo, scale, refdata.interval(name))

        mixed = (
            ("guaranteed", "loose") if refdata.split(name)[1] in refdata.MIXED or name in refdata.MIXED_FILES else ()
        )
        out.append(
            Sweep("stats", name, load, call, contract, contract, T.LOOSE.get(name, GUARANTEED), "loose", mixed,
                  _cmp("stats", name))
        )  # fmt: skip
    return out


# --------------------------------------------------------------------------------------------
# regression: one sweep per problem of tests/regression/ref/{linear,glm,penalized,robust,model_tests,selection}.json,
# one region per reference field (model_tests: the tests on fitted models, their values from
# tests/regression/refdata_model_tests.values, as the tests take them; selection: the criteria along
# stepwise's paths and of best_subset's sizes, from tests/regression/refdata_selection.values; their
# comparators are regcompare's r_model_tests_values, sm_model_tests_values, r_selection_values and
# sm_selection_values). The documented bound (docs/design/regression.md §3.1) is the linear
# models' coefficient vector to 1e-14 normwise for kappa(X) <= 1e11; the other fields have none
# (contract None), so only the baseline judges them.

PARAMS_BOUND, PARAMS_KAPPA = 1e-14, 1e11

REG_KINDS = ("linear", "glm", "penalized", "robust", "model_tests", "selection")
_BP = {"bp_lm": "lm", "bp_lm_pvalue": "lm_pvalue", "bp_fvalue": "fvalue", "bp_f_pvalue": "f_pvalue"}


@dataclasses.dataclass
class RegSweep:
    area: str
    name: str  # kind/problem
    kind: str
    problem: dict

    @property
    def id(self):
        return f"{self.area}/{self.name}"


def regression_sweeps():
    from tests.regression import (
        refdata,
        refdata_glm,
        refdata_model_tests,
        refdata_penalized,
        refdata_robust,
        refdata_selection,
    )

    loaders = {"linear": refdata, "glm": refdata_glm, "penalized": refdata_penalized, "robust": refdata_robust,
               "model_tests": refdata_model_tests, "selection": refdata_selection}  # fmt: skip
    out = []
    for kind in REG_KINDS:
        for name, p in loaders[kind].load().items():
            out.append(RegSweep("regression", f"{kind}/{name}", kind, p))
    return out


def _reg_fit(sw, **kw):
    import stardust as sd
    from tests.regression import refdata_glm, refdata_penalized, refdata_robust

    p = sw.problem
    if sw.kind == "linear":
        if p["weights"] is not None:
            return sd.regression.wls(p["y"], p["X"], weights=p["weights"], intercept=p["intercept"], **kw)
        return sd.regression.ols(p["y"], p["X"], intercept=p["intercept"], **kw)
    if sw.kind == "glm":
        return refdata_glm.fit(p, **kw)
    if sw.kind == "penalized":
        return refdata_penalized.fit(p)
    return refdata_robust.fit(p)


def _reg_value(sw, r, field, cache, fit=None):
    """The fitted value a reference field holds, or None when this tool does not know it; fit:
    the refit with a covariance type (sd's by default, or a comparator's with sd's names)."""
    p = sw.problem
    fit = fit or _reg_fit

    def refit(**kw):
        key = kw["cov_type"]  # the groups are the problem's own
        if key not in cache:
            cache[key] = fit(sw, **kw)
        return cache[key]

    if field.startswith(("bse_", "pvalues_", "fvalue_", "f_pvalue_")):
        attr, ct = field.rsplit("_", 1)
        kw = {"cov_type": ct, "groups": p["groups"]} if ct == "cluster" else {"cov_type": ct}
        return getattr(refit(**kw), attr)
    if field in ("conf_int_lower", "conf_int_upper"):
        return np.asarray(r.conf_int(alpha=0.05))[:, 0 if field.endswith("lower") else 1]
    if field == "hat":
        return r.leverage
    if field == "centered_tss":  # regref's tss is the one R^2 uses
        return r.centered_tss if p["intercept"] else r.uncentered_tss
    if field in ("jarque_bera", "jarque_bera_pvalue"):
        jb = r.jarque_bera()
        return jb.statistic if field == "jarque_bera" else jb.pvalue
    if field in _BP:
        return getattr(r.breusch_pagan(), _BP[field])
    if field.startswith("bp_lm_nonstudentized"):
        bp = r.breusch_pagan(studentize=False)
        return bp.lm_pvalue if field.endswith("pvalue") else bp.lm
    return getattr(r, field, None)


def design_kappa(problem) -> float:
    """kappa_2 of a linear problem's design as the fit sees it (the constant column, the rows of
    positive weight scaled by sqrt(w)), from its singular values in double (numpy): the
    contract's condition, independent of the fit measured."""
    X = np.asarray(problem["X"], dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    if problem["intercept"]:
        X = np.c_[np.ones(len(X)), X]
    if problem["weights"] is not None:
        w = np.asarray(problem["weights"], dtype=float)
        X = X[w > 0] * np.sqrt(w[w > 0])[:, None]
    return float(np.linalg.cond(X))


def measure_regression(sw: RegSweep, compare=False) -> dict:
    if sw.kind in ("model_tests", "selection"):
        from tests.regression import refdata_model_tests, refdata_selection

        mod = refdata_model_tests if sw.kind == "model_tests" else refdata_selection
        vals = mod.values(sw.name.split("/", 1)[1], sw.problem)
        # a field sd computes as another quantity (None) has no region
        ref = {f: v for f, v in sw.problem["ref"].items() if vals.get(f) is not None}
        sw = dataclasses.replace(sw, problem={**sw.problem, "ref": ref})
        get = vals.get
    else:
        r = _reg_fit(sw)
        cache = {}

        def get(f):
            return _reg_value(sw, r, f, cache)

    out = {"id": sw.id, "area": sw.area, "name": sw.name, "points": 0, "regions": _reg_regions(sw, get)}
    if sw.kind == "linear" and "params" in out["regions"] and out["regions"]["params"].get("points"):
        e = out["regions"]["params"]
        if design_kappa(sw.problem) <= PARAMS_KAPPA:
            nr = e["norm_rel"] if e["norm_rel"] is not None else e["rel_max"]  # one coefficient: its error
            e["contract"], e["bound"] = nr / PARAMS_BOUND, PARAMS_BOUND
    out["points"] = sum(x["points"] for x in out["regions"].values())
    if compare:
        out["comparators"] = _reg_comparators(sw)
        out["comparator_seconds"] = REG_SECONDS
    return out


def _reg_comparators(sw) -> dict:
    from . import rsession

    out = {}
    for lib in ("statsmodels", "scikit-learn", "R"):
        if lib == "R" and not rsession.available():
            out[lib] = "not run"
            continue
        try:
            r = isolated(lambda lib=lib: _reg_library(sw, lib), seconds=REG_SECONDS)
        except Exception as e:
            out[lib] = {"error": f"{type(e).__name__}: {e}"[:300]}
            continue
        if r is not None:
            out[lib] = r
    return out


def _reg_library(sw, lib):
    """A comparator library's regions for a regression problem (None: it has no such model)."""
    from . import regcompare as RC

    if sw.kind == "selection" and lib == "scikit-learn":  # its selectors score another way
        return None
    with np.errstate(all="ignore"):
        if lib == "statsmodels" and sw.kind == "selection":
            vals = RC.sm_selection_values(sw)
            return _reg_regions(sw, vals.get, comparator=True) if vals else None
        if lib == "statsmodels" and sw.kind == "model_tests":
            vals = RC.sm_model_tests_values(sw)
            return _reg_regions(sw, vals.get, comparator=True) if vals else None
        if lib == "statsmodels":
            res = RC.sm_fit(sw)
            if res is None:
                return None
            cache = {}

            def get(f):
                try:
                    return _reg_value(sw, res, f, cache, fit=RC.sm_fit)
                except AttributeError, ValueError, TypeError, np.linalg.LinAlgError:
                    return None

        else:
            vals = RC.sklearn_values(sw) if lib == "scikit-learn" else RC.r_values(sw)
            if vals is None:
                return None

            def get(f):
                return vals.get(f)

        return _reg_regions(sw, get, comparator=True)


def _reg_regions(sw, get, comparator=False) -> dict:
    from tests.special import refdata as R

    regions = {}
    for field, ref in sw.problem["ref"].items():
        v = get(field)
        if v is None:
            if not comparator:
                regions[field] = {"points": 0, "unmapped": True}
            continue
        hl = np.asarray(ref, dtype=np.float64).reshape(-1, 2)
        hi, lo = hl[:, 0], hl[:, 1]
        got = np.asarray(v, dtype=np.float64).reshape(-1)
        if got.size != hi.size:
            continue
        excluded = None
        if comparator:
            keep = np.isfinite(got) | ~np.isfinite(hi)
            excluded = int((~keep).sum())
            got, hi, lo = got[keep], hi[keep], lo[keep]
            if not got.size:
                regions[field] = {"points": 0, "excluded": excluded}
                continue
        with np.errstate(all="ignore"):
            ulp = ulps(got, hi)
            rel = R.rel_error(got, hi, lo)
            den = np.linalg.norm(hi + lo)
            norm = float(np.linalg.norm((got - hi) - lo) / den) if den > 0 else float(np.linalg.norm(got - hi))
        worst = int(np.argmax(np.where(np.isnan(rel), np.inf, rel)))
        nonfinite = int((np.isfinite(hi) & ~np.isfinite(got)).sum())  # counted as an inf error above
        regions[field] = {
            "points": int(hi.size), "ulp_max": _finite_max(ulp), "ulp_p999": _p999(ulp),
            "rel_max": _finite_max(rel), "norm_rel": norm if hi.size > 1 else None, "nonfinite": nonfinite,
            "contract": None, "bound": None, "worst": [worst],
        }  # fmt: skip
        if excluded is not None:
            regions[field]["excluded"] = excluded
    return regions


def sweeps(areas=None):
    """The sweeps of the areas (default all), in AREAS order."""
    builders = {
        "special": special_sweeps,
        "stats": lambda: stats_sweeps() + _mv_sweeps(),
        "regression": regression_sweeps,
        "array": lambda: _module("array").sweeps(),
        "linalg": lambda: _module("linalg").sweeps(),
        "optimize": lambda: _module("optimize").sweeps(),
        "htest": lambda: _module("htest").sweeps(),
        "sparse": lambda: _module("sparse").sweeps(),
    }
    out = []
    for a in check_areas(areas):
        out += builders[a]()
    return out


def _module(name):
    import importlib

    return importlib.import_module(f"{__package__}.{name}")


def _mv_sweeps():
    return _module("multivariate").sweeps()


# --------------------------------------------------------------------------------------------
# measuring


def _finite_max(e):
    e = np.where(np.isnan(e), np.inf, e)  # a NaN error (a NaN where the reference has a value) is the worst
    return float(e.max()) if e.size else None


def _p999(e):
    e = np.where(np.isnan(e), np.inf, e)
    return float(np.quantile(e, 0.999, method="higher")) if e.size else None


def measure(sw, compare=False) -> dict:
    """One sweep's entry; with compare, also "comparators": {library: {region: entry}, or
    "not run" (R absent), or {"error": message}} for the libraries that compute the same
    quantity (compare.py)."""
    if isinstance(sw, RegSweep):
        return measure_regression(sw, compare)
    elif isinstance(sw, Custom):
        regions = sw.run()
        out = {"id": sw.id, "area": sw.area, "name": sw.name, "points": sum(r["points"] for r in regions.values()),
               "regions": regions}  # fmt: skip
        if compare and sw.compare is not None:
            out["comparators"] = sw.compare()
            out["comparator_seconds"] = getattr(sw.compare, "seconds", None)
        return out
    else:
        xs, hi, lo, g, scale = sw.load()
        scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), hi.shape)
        out = {"id": sw.id, "area": sw.area, "name": sw.name, "points": int(hi.size),
               "regions": _regions(sw, xs, sw.call(xs), hi, lo, g, scale)}  # fmt: skip
        if compare and sw.comparators is not None:
            comps = {}
            for lib, fn in sw.comparators().items():
                if isinstance(fn, str):
                    comps[lib] = fn
                    continue
                try:
                    with np.errstate(all="ignore"):
                        vals = isolated(lambda fn=fn: _call_comparator(fn, xs, hi.shape), seconds=SWEEP_SECONDS)
                except Exception as e:  # past the time or memory limit (Guarded), or no point evaluates
                    comps[lib] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    continue
                keep = np.isfinite(vals) | ~np.isfinite(hi)  # NaN or inf where the reference is finite: left out
                comps[lib] = _regions(sw, xs, vals, hi, lo, g, scale, keep)
            out["comparators"] = comps
            out["comparator_seconds"] = SWEEP_SECONDS
        return out


def _call_comparator(fn, xs, shape):
    """fn over all the points; when it raises for the whole array (scipy's Boost errors: one bad
    point fails the call), in chunks of 256 and then point by point, a failing point NaN (left
    out and counted like a NaN result)."""
    try:
        return np.broadcast_to(np.asarray(fn(xs)), shape)  # complex stays complex
    except Exception as e:
        from .common import Guarded

        if isinstance(e, Guarded):
            raise
    cols = [np.broadcast_to(np.asarray(x), shape).ravel() for x in xs]
    cplx = any(np.iscomplexobj(c) for c in cols)
    out = np.full(int(np.prod(shape)), np.nan, dtype=complex if cplx else float)
    ok = 0
    for i in range(0, out.size, 256):
        part = [c[i : i + 256] for c in cols]
        try:
            out[i : i + 256] = np.asarray(fn(part))
            ok += 1
            continue
        except Exception as e:
            from .common import Guarded

            if isinstance(e, Guarded):
                raise
        for j in range(len(part[0])):
            try:
                out[i + j] = np.asarray(fn([c[j : j + 1] for c in part]))[0]
                ok += 1
            except Exception as e:
                from .common import Guarded

                if isinstance(e, Guarded):
                    raise
    if not ok:
        raise ValueError("every point fails")
    return out.reshape(shape)


def _regions(sw, xs, got, hi, lo, g, scale, keep=None) -> dict:
    """The guaranteed and loose (or mixed) regions of a reference sweep for the values got;
    keep: the points measured (the rest are counted as "excluded")."""
    from tests.special import refdata as R

    with np.errstate(all="ignore"):
        ulp = ulps(got, hi)
        rel = R.rel_error(got, hi, lo)
    regions = {}
    for label, region_mask, fn, bound in (("guaranteed", g, sw.contract, GUARANTEED),
                                          (sw.loose_label, ~g, sw.loose_contract, sw.loose_bound)):  # fmt: skip
        if not region_mask.any():
            continue
        mask = region_mask if keep is None else region_mask & keep
        excluded = int((region_mask & ~mask).sum())
        if not mask.any():
            regions[label] = {"points": 0, "excluded": excluded}
            continue
        with np.errstate(all="ignore"):
            c = np.asarray(fn(sw.name, got, hi, lo, scale), dtype=np.float64)[mask]
        u, r, s = ulp[mask], rel[mask], scale[mask]
        cm = np.where(np.isnan(c), np.inf, c)
        worst = int(np.argmax(cm))
        entry = {
            "points": int(mask.sum()),
            "ulp_max": _finite_max(u),
            "ulp_p999": _p999(u),
            "rel_max": _finite_max(r),
            "contract": (_finite_max(c) / bound) if bound else None,
            "bound": bound,
            "worst": [float(x[mask][worst]) if np.isrealobj(x) else str(x[mask][worst]) for x in xs],
        }
        if keep is not None:
            entry["excluded"] = excluded
        if label in sw.mixed:  # |error| / max(1, |value|) in units of 2^-52 (45 is the 1e-14 bound)
            with np.errstate(all="ignore"):
                m = R.mixed_error(got, hi, lo)[mask] / 2.0**-52
            entry["mixed_max"] = _finite_max(m)
            entry["mixed_p999"] = _p999(m)
            # the relative error says nothing where the value crosses 0: the points where the
            # reference is finite and the value is not are counted instead
            del entry["rel_max"]
            g_ = np.asarray(got)[mask]
            entry["nonfinite"] = int((np.isfinite(hi[mask]) & ~np.isfinite(g_)).sum())
        if (s > 1).any():  # every function with a scale above 1 somewhere: ULP / scale, the contract's unit
            entry["ulp_scale_max"] = _finite_max(u / s)
            entry["ulp_scale_p999"] = _p999(u / s)
        regions[label] = entry
    return regions
