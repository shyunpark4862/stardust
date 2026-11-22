"""Results of pdist, cdist and the distance functions do not depend on the thread count
(docs/design/spatial.md §1.3, §3.5): at 1, 3 and 10 threads every metric gives the same bytes on inputs large enough to
split into many tasks (the tiles, the row groups of pdist, the scalar drivers and the prepared rows)."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal

pytestmark = pytest.mark.usefixtures("restore_threads")

S = sd.spatial
CASES = (("euclidean", {}), ("sqeuclidean", {}), ("cityblock", {}), ("chebyshev", {}), ("minkowski", {"p": 3.0}),
         ("seuclidean", {}), ("mahalanobis", {}), ("cosine", {}), ("correlation", {}), ("canberra", {}),
         ("braycurtis", {}), ("hamming", {}), ("jaccard", {}))  # fmt: skip


def _data():
    g = np.random.default_rng(11)
    a, b = g.standard_normal((301, 37)), g.standard_normal((157, 37))
    a[:, ::3], b[:, ::3] = np.round(a[:, ::3]), np.round(b[:, ::3])
    a[7, 5] = np.nan
    return a, b


def _results():
    a, b = _data()
    w = np.random.default_rng(12).uniform(0.0, 2.0, 37)
    out = []
    for metric, kw in CASES:
        out.append(S.pdist(a, metric=metric, **kw))
        out.append(S.cdist(a, b, metric=metric, **kw))
        if metric not in ("seuclidean", "mahalanobis"):
            out.append(S.cdist(a, b, metric=metric, w=w, **kw))
    bits = np.random.default_rng(13).random((260, 130)) < 0.3
    out.append(S.pdist(bits, metric="hamming"))
    out.append(S.pdist(bits, metric="jaccard"))
    out.append(S.pdist(bits.astype(np.int64), metric="jaccard"))
    h = np.random.default_rng(14).uniform(-1.5, 1.5, (400, 2))
    out.append(S.pdist(h, metric="haversine"))
    out.append(S.euclidean(np.tile(a, (40, 1)), np.tile(a[::-1], (40, 1))))
    out.append(S.cosine(np.tile(a, (40, 1)), a[0]))
    return out


def test_one_three_ten_threads():
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got[t] = _results()
    for t in (3, 10):
        for k, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, k)
