"""The sweeps of each area and how a sweep is measured.

A sweep is one reference file: its inputs, the reference (hi + lo), the guaranteed region,
the bound's scale (the multiplier of the 1e-14 bound at each point: max(1, kappa) for the
inverses, max(1, M / |f|) near the zeros of the oscillatory Bessel and Airy functions and of
the complex error functions, else 1) and the area's contract measure.
The measures are the tests' own (tests/special/test_reference.py, tests/stats/refdata.py),
so a table and the tests can never disagree about what "within the contract" means.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from .common import Custom, isolated

AREAS = ("special", "stats", "multivariate", "regression", "array", "linalg", "optimize")
GUARANTEED = 1e-14


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
        else:
            loose, bound, label, mixed = contract, T.LOOSE.get(name), "loose", ()
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
# regression: one sweep per problem of tests/regression/ref/{linear,glm,penalized,robust}.json, one
# region per reference field; no documented bound (contract None), so only the baseline judges.

REG_KINDS = ("linear", "glm", "penalized", "robust")
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
    import json

    from tests.regression import refdata

    out = []
    for kind in REG_KINDS:
        data = json.loads((refdata.pathlib.Path(refdata.__file__).parent / "ref" / f"{kind}.json").read_text())
        for name, p in data.items():
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


def measure_regression(sw: RegSweep, compare=False) -> dict:
    r = _reg_fit(sw)
    cache = {}
    out = {"id": sw.id, "area": sw.area, "name": sw.name, "points": 0,
           "regions": _reg_regions(sw, lambda f: _reg_value(sw, r, f, cache))}  # fmt: skip
    out["points"] = sum(x["points"] for x in out["regions"].values())
    if compare:
        out["comparators"] = _reg_comparators(sw)
    return out


def _reg_comparators(sw) -> dict:
    from . import rsession

    out = {}
    for lib in ("statsmodels", "scikit-learn", "R"):
        if lib == "R" and not rsession.available():
            out[lib] = "not run"
            continue
        try:
            r = isolated(lambda lib=lib: _reg_library(sw, lib), seconds=120)
        except Exception as e:
            out[lib] = {"error": f"{type(e).__name__}: {e}"[:300]}
            continue
        if r is not None:
            out[lib] = r
    return out


def _reg_library(sw, lib):
    """A comparator library's regions for a regression problem (None: it has no such model)."""
    from . import regcompare as RC

    with np.errstate(all="ignore"):
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
            ulp = R.ulp_error(got, hi, lo)
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
    areas = areas or AREAS
    out = []
    if "special" in areas:
        out += special_sweeps()
    if "stats" in areas:
        out += stats_sweeps()
    if "multivariate" in areas:
        from . import multivariate

        out += multivariate.sweeps()
    if "regression" in areas:
        out += regression_sweeps()
    if "array" in areas:
        from . import array

        out += array.sweeps()
    if "linalg" in areas:
        from . import linalg

        out += linalg.sweeps()
    if "optimize" in areas:
        from . import optimize

        out += optimize.sweeps()
    return out


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
                        vals = isolated(lambda fn=fn: _call_comparator(fn, xs, hi.shape))
                except Exception as e:  # past the time or memory limit (Guarded), or no point evaluates
                    comps[lib] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    continue
                keep = np.isfinite(vals) | ~np.isfinite(hi)  # NaN or inf where the reference is finite: left out
                comps[lib] = _regions(sw, xs, vals, hi, lo, g, scale, keep)
            out["comparators"] = comps
        return out
    if compare and sw.compare is not None:
        out["comparators"] = sw.compare()
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
        ulp = R.ulp_error(got, hi, lo)
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
        if (s > 1).any():  # every function with a scale above 1 somewhere: ULP / scale, the contract's unit
            entry["ulp_cond_max"] = _finite_max(u / s)
            entry["ulp_cond_p999"] = _p999(u / s)
        regions[label] = entry
    return regions
