"""Every univariate distribution's methods end on a grid of invalid and extreme parameters
(docs/design/stats.md §3.4: the chunk kernels see only valid elements). A child process per
distribution with a time limit: a call that loops holds the GIL and would hang the run (as
the Pareto with scale <= 0 did). Then the Pareto cases themselves, and a chunk that mixes
invalid, NaN and limit elements with valid ones."""

import math
import subprocess
import sys

import pytest

import stardust as sd

S = sd.stats
A = sd.asarray

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


PARETO_HANGS = [
    (-1.0, 0.0, 0.0, "pdf"),
    (-1.0, 0.0, -0.0, "pdf"),
    (1.0, 0.0, 0.0, "pdf"),
    (0.0, 0.0, 0.0, "pdf"),
    (1.0, 0.0, 0.0, "logpdf"),
    (1.0, -1.0, -0.5, "pdf"),
    (1.0, -1.0, -0.5, "cdf"),
    (1.0, -1.0, -0.5, "logpdf"),
]

_PARETO_CHILD = """
import math, sys, stardust as sd
S = sd.stats
for b, s, x, m in {cases!r}:
    v = getattr(S.Pareto(b=b, scale=s), m)(x)
    assert math.isnan(v), (b, s, x, m, v)
    # an array of them beside a valid element: the valid one keeps its value
    d = S.Pareto(b=sd.asarray([b, 2.0, b]), scale=sd.asarray([s, 1.0, s]))
    r = getattr(d, m)(sd.asarray([x, 1.5, x])).tolist()
    assert math.isnan(r[0]) and math.isnan(r[2]) and r[1] == getattr(S.Pareto(b=2.0, scale=1.0), m)(1.5), r
# the log ratio of a Weibull-like scale <= 0 (lifetime and weibull detail) through the engine
for m in ("pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf"):
    for s in (-math.inf, -1.0, -0.0, 0.0):
        for x in (-1e300, -1.0, -0.5, -0.0, 0.0, 5e-324, 1e-310):
            assert math.isnan(getattr(S.Pareto(b=1.0, scale=s), m)(x))
            assert math.isnan(getattr(S.Weibull(c=1.0, scale=s), m)(x))
print("ok")
"""


# ---- ST-17: invalid parameters ------------------------------------------------------------


def test_invalid_pareto_ends_with_nan():
    # each call used to loop forever holding the GIL: a child process with a time limit
    r = subprocess.run(
        [sys.executable, "-c", _PARETO_CHILD.format(cases=PARETO_HANGS)], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def test_chunk_kernels_see_only_good_elements():
    # invalid, NaN-input and limit elements mixed with valid ones in one chunk: the valid ones
    # keep the bits of their scalar calls, the others are NaN or the limit's value
    df = A([3.0, -1.0, math.inf, 3.0, math.nan, 7.0])
    x = A([0.5, 0.5, 0.5, math.nan, 0.5, -2.0])
    got = S.StudentT(df=df).cdf(x).tolist()
    want = [
        S.StudentT(df=3.0).cdf(0.5),
        math.nan,
        S.Normal().cdf(0.5),
        math.nan,
        math.nan,
        S.StudentT(df=7.0).cdf(-2.0),
    ]
    assert all(g == w or (math.isnan(g) and math.isnan(w)) for g, w in zip(got, want)), (got, want)
