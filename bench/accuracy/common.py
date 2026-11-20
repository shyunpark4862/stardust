"""What every area's measurement shares: the summary of one region and the generic sweep."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass
class Custom:
    """A sweep that measures itself: run() -> {region: entry} (entries from region())."""

    area: str
    name: str
    run: Callable

    @property
    def id(self):
        return f"{self.area}/{self.name}"


def finite_max(e):
    e = np.where(np.isnan(e), np.inf, np.asarray(e, dtype=np.float64))  # a NaN error is the worst
    return float(e.max()) if e.size else None


def p999(e):
    e = np.where(np.isnan(e), np.inf, np.asarray(e, dtype=np.float64))
    return float(np.quantile(e, 0.999, method="higher")) if e.size else None


def region(where, *, ulp=None, rel=None, mixed=None, abs_err=None, contract=None, bound=None, **extra) -> dict:
    """One region's entry: the maxima (and 99.9% quantiles of ULP and mixed) of the per-point
    errors given, contract = max(contract) / bound (None without a bound), and worst = the
    description (where) of the point with the largest contract error, else relative error.
    extra: further per-point measures, stored as <name>_max."""
    where = np.asarray(where)
    entry = {"points": int(where.size)}
    if ulp is not None:
        entry["ulp_max"], entry["ulp_p999"] = finite_max(ulp), p999(ulp)
    if mixed is not None:
        entry["mixed_max"], entry["mixed_p999"] = finite_max(mixed), p999(mixed)
    if rel is not None:
        entry["rel_max"] = finite_max(rel)
    if abs_err is not None:
        entry["abs_max"] = finite_max(abs_err)
    for k, v in extra.items():
        entry[f"{k}_max"] = finite_max(v)
    entry["contract"] = finite_max(contract) / bound if bound and contract is not None else None
    entry["bound"] = bound
    key = next((e for e in (contract, rel, mixed, ulp, abs_err, *extra.values()) if e is not None), None)
    if key is not None and where.size:
        k = np.where(np.isnan(key), np.inf, np.asarray(key, dtype=np.float64))
        entry["worst"] = [str(where[int(np.argmax(k))])]
    else:
        entry["worst"] = []
    return entry
