"""Every univariate distribution's methods end on a grid of invalid and extreme parameters
(docs/design/stats.md §3.4: the chunk kernels see only valid elements). A child process per
distribution with a time limit: a call that loops holds the GIL and would hang the run (the
Pareto with scale <= 0 did, 2026-11-20)."""

import subprocess
import sys

import pytest

import stardust as sd

DISTS = sorted(n for n in sd.stats.__all__ if getattr(getattr(sd.stats, n), "_name", ""))  # the 31 univariate

_CHILD = """
import itertools, sys, numpy as np, stardust as sd
V = [float('nan'), -np.inf, -1.0, -0.0, 0.0, 5e-324, 1e-310, 1e-300, 0.5, 1.0, 3.0, 1e300, np.inf]
X = [float('nan'), -np.inf, -1e300, -1.0, -0.0, 0.0, 5e-324, 1e-310, 0.25, 0.5, 1.0, 2.0, 1e300, np.inf]
cls = getattr(sd.stats, sys.argv[1])
k = len(cls._params)
combos = list(itertools.product(V, repeat=k))
d = cls(**{p: sd.asarray(np.array([c[i] for c in combos]).reshape(-1, 1)) for i, p in enumerate(cls._params)})
x = sd.asarray(np.array(X).reshape(1, -1))
for m in ("pdf", "pmf", "logpdf", "logpmf", "cdf", "ccdf", "logcdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf"):
    if hasattr(d, m):
        getattr(d, m)(x)
for m in ("mean", "median", "mode", "variance", "skewness", "kurtosis", "entropy"):
    getattr(d, m)()
print("ok")
"""


@pytest.mark.deep
@pytest.mark.parametrize("name", DISTS)
def test_invalid_parameters_end(name):
    r = subprocess.run([sys.executable, "-c", _CHILD, name], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
