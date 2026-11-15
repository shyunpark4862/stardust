"""Every distribution method with a reference file (tests/stats/ref, made by
tools/stats_ref.py) meets its accuracy contract (docs/design/stats.md §2.3, T20):
relative error at most 1e-14 in the guaranteed region (times max(1, kappa) for the
inverses; |error| <= 1e-14 max(1, |value|) for logpdf and entropy); outside it, at most
the documented loose bound."""

import numpy as np
import pytest

from . import refdata

# Outside the guaranteed region: the measured bound each method documents
# (docs/api/stats.md §2.2). Points outside the parameter ranges that still measure within
# 1e-14 are held to 1e-14; the files listed here measured more.
LOOSE: dict[str, float] = {
    # gamma a > 1e6 (sd.special's gammaincinv/gammainccinv beyond their measured range):
    # 1.1e-14, 2.4e-14 and 2.5e-14 measured
    "gamma_icdf": 2e-14, "gamma_ilogcdf": 3e-14, "gamma_ilogccdf": 3e-14,
    # t, df >= 1e5 with p below the normal range (no log form of the beta side's BGRAT
    # branch; user decision 2026-11-15): 1.4e-14 measured
    "studentt_icdf": 2e-14, "studentt_iccdf": 2e-14,
}


@pytest.mark.parametrize("name", refdata.names())
def test_accuracy(name):
    xs, hi, lo, guaranteed, scale = refdata.load(name)
    got = refdata.call(name, xs)
    err = refdata.contract_error(name, got, hi, lo, scale)
    if guaranteed.any():
        e = err[guaranteed]
        i = int(np.argmax(np.where(np.isnan(e), np.inf, e)))
        assert e[i] <= refdata.GUARANTEED, (name, [float(x[guaranteed][i]) for x in xs], float(e[i]))
    if (~guaranteed).any():
        e = err[~guaranteed]
        assert np.nanmax(e) <= LOOSE.get(name, refdata.GUARANTEED), (name, float(np.nanmax(e)))


@pytest.mark.parametrize("name", refdata.names())
def test_scalar_path_gives_the_same_bits(name):
    # f(x)[i] == f(x[i]): the same value whether the method sees an array or Python floats
    xs, *_ = refdata.load(name)
    got = refdata.call(name, xs)
    idx = np.linspace(0, got.size - 1, 100).astype(int)
    for i in idx:
        v = refdata.call(name, tuple(float(x[i]) for x in xs))
        assert np.array([v]).view(np.uint64).tolist() == got[i:i + 1].view(np.uint64).tolist(), (name, i)
