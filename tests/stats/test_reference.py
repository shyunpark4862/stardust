"""Every distribution method with a reference file (tests/stats/ref, made by
tools/stats_ref.py) meets its accuracy contract (docs/design/stats.md §2.3): relative error
at most 1e-14 in the guaranteed region (times max(1, kappa) for the inverses;
|error| <= 1e-14 max(1, |value|) for logpdf and entropy; for the discrete inverses an exact
quantile of a target within relative 1e-14); outside it, at most the documented loose bound.
The multivariate and matrix densities (tests/stats/ref/mv) are held to theirs the same way."""

import numpy as np
import pytest

from . import refdata

# Outside the guaranteed region: the measured bound each method documents
# (docs/api/stats.md §2.2). Points outside the parameter ranges that still measure within
# 1e-14 are held to 1e-14; the files listed here measured more.
LOOSE: dict[str, float] = {
    # (empty since 2026-11-19: the t's deep inverse near x = 1, 1.4e-14 before, solves in
    # log(1 - x) and measures 3.2e-16)
}


# Methods without a guaranteed point, and why (none at present: a method with no points, or
# none in the guaranteed region, fails, so a generator that dropped them all cannot pass).
NO_GUARANTEED: dict[str, str] = {}


@pytest.mark.parametrize("name", refdata.names())
def test_accuracy(name):
    xs, hi, lo, guaranteed, scale = refdata.load(name)
    assert hi.size, name
    assert guaranteed.any() or name in NO_GUARANTEED, name
    got = refdata.call(name, xs)
    err = refdata.contract_error(name, got, hi, lo, scale, refdata.interval(name))
    if guaranteed.any():
        e = err[guaranteed]
        i = int(np.argmax(np.where(np.isnan(e), np.inf, e)))
        assert e[i] <= refdata.GUARANTEED, (name, [float(x[guaranteed][i]) for x in xs], float(e[i]))
    if (~guaranteed).any():
        e = np.where(np.isnan(err[~guaranteed]), np.inf, err[~guaranteed])  # a NaN error is the worst
        assert e.max() <= LOOSE.get(name, refdata.GUARANTEED), (name, float(e.max()))


@pytest.mark.parametrize("name", refdata.names())
def test_scalar_path_gives_the_same_bits(name):
    # f(x)[i] == f(x[i]): the same value whether the method sees an array or Python floats
    xs, *_ = refdata.load(name)
    got = refdata.call(name, xs)
    idx = np.linspace(0, got.size - 1, 100).astype(int)
    for i in idx:
        v = refdata.call(name, tuple(float(x[i]) for x in xs))
        assert np.array([v]).view(np.uint64).tolist() == got[i : i + 1].view(np.uint64).tolist(), (name, i)


@pytest.mark.parametrize(
    ("family", "method"), [(f, m) for f, (_, ms) in refdata.MV_FAMILIES.items() for m in ms], ids=lambda v: str(v)
)
def test_multivariate_accuracy(family, method):
    # every point of tests/stats/ref/mv/<family>.npz, judged as the accuracy measurement judges
    for i, g in enumerate(refdata.mv_groups(family)):
        got = refdata.mv_values(family, method, g)
        lp = g["logpdf_hi"] if method == "pdf" else None
        mask, err, bound = refdata.mv_density_contract(method, got, g[f"{method}_hi"], g[f"{method}_lo"], lp)[
            "guaranteed"
        ]
        e = np.where(np.isnan(err), np.inf, err)[mask]
        assert e.max(initial=0.0) <= bound, (family, method, i, float(e.max()))
