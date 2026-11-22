"""kneighbors and radius_neighbors do not depend on the thread count or on Accelerate's thread mode
(docs/design/spatial.md §1.3, §3.2, §3.5): at 1, 3 and 10 threads every path gives the same bytes on
inputs large enough to split into many tasks (XA chunks, XB segments, GEMM blocks), and a process with
VECLIB_MAXIMUM_THREADS=1 gives the same bytes as one with Accelerate's default mode (the GEMM only
pre-selects; the pair kernel decides)."""

import hashlib
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal

pytestmark = pytest.mark.usefixtures("restore_threads")

S = sd.spatial
N = S._neighbors
INF = float("inf")


def _data():
    g = np.random.default_rng(21)
    a, b = g.standard_normal((900, 20)), g.standard_normal((5000, 20))
    a[:, ::3], b[:, ::3] = np.round(a[:, ::3]), np.round(b[:, ::3])
    b[100:200] = b[:100]  # duplicates: ties
    far = 1e8 + g.random((40, 16)), 1e8 + g.random((3000, 16))
    return a, b, far


def _results():
    a, b, (fa, fb) = _data()
    out = []
    for path in (0, 1, 2):
        for metric in ("euclidean", "sqeuclidean"):
            out.extend(N._knn("kneighbors", a, b, 7, metric, None, None, None, None, INF, path))
            out.extend(N._knn("kneighbors", a[:3], b, 5, metric, None, None, None, None, INF, path))  # segments
            out.extend(N._knn("kneighbors", fa, fb, 3, metric, None, None, None, None, INF, path))
            r = N._radius("radius_neighbors", a[:200], b, 4.0, metric, None, None, None, None, False, path)
            out.extend([r.indptr, r.indices, r.distances])
    for metric, kw in (("cityblock", {}), ("minkowski", {"p": 3.0}), ("cosine", {}), ("mahalanobis", {})):
        out.extend(S.kneighbors(a[:300], b[:2000], k=4, metric=metric, **kw))
        r = S.radius_neighbors(a[:100], b[:2000], 0.5 if metric == "cosine" else 6.0, metric=metric, **kw)
        out.extend([r.indptr, r.indices, r.distances])
    return out


def test_one_three_ten_threads():
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got[t] = _results()
    for t in (3, 10):
        for k, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, k)


_SCRIPT = """
import hashlib, sys
import numpy as np
sys.path.insert(0, {root!r})
import stardust as sd
from tests.spatial import test_neighbors_threads as T
h = hashlib.sha256()
for x in T._results():
    h.update(np.asarray(x).tobytes())
print(h.hexdigest())
"""


def _digest(env_value):
    root = str(pathlib.Path(__file__).parents[2])
    env = dict(os.environ)
    env.pop("VECLIB_MAXIMUM_THREADS", None)
    if env_value is not None:
        env["VECLIB_MAXIMUM_THREADS"] = env_value
    out = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=root)], env=env, capture_output=True, text=True,
                         timeout=600, check=True)  # fmt: skip
    return out.stdout.strip().splitlines()[-1]


def test_accelerate_thread_mode():
    one, default = _digest("1"), _digest(None)
    assert len(one) == 64 and one == default
    here = hashlib.sha256()
    for x in _results():
        here.update(np.asarray(x).tobytes())
    assert here.hexdigest() == default
