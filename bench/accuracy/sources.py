"""The sweeps of each area and how a sweep is measured.

A sweep is one reference file: its inputs, the reference (hi + lo), the guaranteed region,
the bound's scale (max(1, kappa) for the inverses, else 1) and the area's contract measure.
The measures are the tests' own (tests/special/test_reference.py, tests/stats/refdata.py),
so a table and the tests can never disagree about what "within the contract" means.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

AREAS = ("special", "stats", "regression")
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

    @property
    def id(self):
        return f"{self.area}/{self.name}"


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
        out.append(Sweep("special", name, load, call, contract, loose, bound, label, mixed))
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
        out.append(Sweep("stats", name, load, call, contract, contract, T.LOOSE.get(name, GUARANTEED), "loose", mixed))
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


def _reg_value(sw, r, field, cache):
    """The fitted value a reference field holds, or None when this tool does not know it."""
    p = sw.problem

    def refit(**kw):
        key = kw["cov_type"]  # the groups are the problem's own
        if key not in cache:
            cache[key] = _reg_fit(sw, **kw)
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


def measure_regression(sw: RegSweep) -> dict:
    from tests.special import refdata as R

    r = _reg_fit(sw)
    cache = {}
    regions = {}
    for field, ref in sw.problem["ref"].items():
        v = _reg_value(sw, r, field, cache)
        if v is None:
            regions[field] = {"points": 0, "unmapped": True}
            continue
        hl = np.asarray(ref, dtype=np.float64).reshape(-1, 2)
        hi, lo = hl[:, 0], hl[:, 1]
        got = np.asarray(v, dtype=np.float64).reshape(-1)
        with np.errstate(all="ignore"):
            ulp = R.ulp_error(got, hi, lo)
            rel = R.rel_error(got, hi, lo)
            den = np.linalg.norm(hi + lo)
            norm = float(np.linalg.norm((got - hi) - lo) / den) if den > 0 else float(np.linalg.norm(got - hi))
        worst = int(np.argmax(np.where(np.isnan(rel), np.inf, rel)))
        regions[field] = {
            "points": int(hi.size), "ulp_max": _finite_max(ulp), "ulp_p999": _p999(ulp),
            "rel_max": _finite_max(rel), "norm_rel": norm if hi.size > 1 else None,
            "contract": None, "bound": None, "worst": [worst],
        }  # fmt: skip
    return {"id": sw.id, "area": sw.area, "name": sw.name, "points": sum(x["points"] for x in regions.values()),
            "regions": regions}  # fmt: skip


def sweeps(areas=None):
    areas = areas or AREAS
    out = []
    if "special" in areas:
        out += special_sweeps()
    if "stats" in areas:
        out += stats_sweeps()
    if "regression" in areas:
        out += regression_sweeps()
    return out


# --------------------------------------------------------------------------------------------
# measuring


def _finite_max(e):
    e = np.where(np.isnan(e), np.inf, e)  # a NaN error (a NaN where the reference has a value) is the worst
    return float(e.max()) if e.size else None


def _p999(e):
    e = np.where(np.isnan(e), np.inf, e)
    return float(np.quantile(e, 0.999, method="higher")) if e.size else None


def measure(sw) -> dict:
    if isinstance(sw, RegSweep):
        return measure_regression(sw)
    from tests.special import refdata as R

    xs, hi, lo, g, scale = sw.load()
    got = sw.call(xs)
    scale = np.broadcast_to(np.asarray(scale, dtype=np.float64), hi.shape)
    with np.errstate(all="ignore"):
        ulp = R.ulp_error(got, hi, lo)
        rel = R.rel_error(got, hi, lo)
    regions = {}
    for label, mask, fn, bound in (("guaranteed", g, sw.contract, GUARANTEED),
                                   (sw.loose_label, ~g, sw.loose_contract, sw.loose_bound)):  # fmt: skip
        if not mask.any():
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
        if label in sw.mixed:  # |error| / max(1, |value|) in units of 2^-52 (45 is the 1e-14 bound)
            with np.errstate(all="ignore"):
                m = R.mixed_error(got, hi, lo)[mask] / 2.0**-52
            entry["mixed_max"] = _finite_max(m)
            entry["mixed_p999"] = _p999(m)
        if (s > 1).any():  # the inverses: ULP / max(1, kappa), their contract's unit
            entry["ulp_cond_max"] = _finite_max(u / s)
            entry["ulp_cond_p999"] = _p999(u / s)
        regions[label] = entry
    return {"id": sw.id, "area": sw.area, "name": sw.name, "points": int(hi.size), "regions": regions}
