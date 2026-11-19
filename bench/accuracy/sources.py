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

AREAS = ("special", "stats")
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


def sweeps(areas=None):
    areas = areas or AREAS
    out = []
    if "special" in areas:
        out += special_sweeps()
    if "stats" in areas:
        out += stats_sweeps()
    return out


# --------------------------------------------------------------------------------------------
# measuring


def _finite_max(e):
    e = np.where(np.isnan(e), np.inf, e)  # a NaN error (a NaN where the reference has a value) is the worst
    return float(e.max()) if e.size else None


def _p999(e):
    e = np.where(np.isnan(e), np.inf, e)
    return float(np.quantile(e, 0.999, method="higher")) if e.size else None


def measure(sw: Sweep) -> dict:
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
