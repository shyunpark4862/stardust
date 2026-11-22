"""sd.spatial against scipy.spatial 1.18.1, scikit-learn 1.9.1 and R 4.6.2's stats::dist
(docs/design/spatial.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the
sd.spatial names it needs, and cases() keeps a spec once its names are public (SPT1 brings the
distances, SPT2 kneighbors and radius_neighbors, SPT3 KDTree and query, SPT4 the radius, pair and
count queries). The sizes are those of docs/design/spatial.md §5, where SPT0 timed the comparators.
scipy's distance and tree kernels run on one thread; the tree queries are also compared with
scipy's workers=-1, since sd runs on its pool by default. scikit-learn's expanded-form functions
use the BLAS's threads (threads="follow"). R's dist runs in the one R session.

Data: standard normal points (distances, one-call searches) and uniform points in [0, 1)^m (trees),
numpy arrays made in the setup from fixed seeds; sd gets them as sd.ndarray, outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Ref, RRef


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""


def _normal(ma, mb, n, seed=20261122):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, ma, mb, n]))
        a, b = g.standard_normal((ma, n)), g.standard_normal((mb, n))
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b), "r_a": a.ravel(order="F"), "r_m": float(ma),
                "r_n": float(n)}  # fmt: skip

    return setup


def _bool(ma, mb, n, seed=20261124):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, ma, mb, n]))
        a, b = g.random((ma, n)) < 0.3, g.random((mb, n)) < 0.3
        return {"a": a, "b": b, "sa": sd.asarray(a), "sb": sd.asarray(b)}

    return setup


def _latlon(m, seed=20261125):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, m]))
        a = np.column_stack([g.uniform(-1.5, 1.5, m), g.uniform(-3.1, 3.1, m)])
        return {"a": a, "sa": sd.asarray(a)}

    return setup


def _uniform(npts, m, nq, seed=20261123):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, npts, m, nq]))
        x, q = g.random((npts, m)), g.random((nq, m))
        return {"x": x, "q": q, "sx": sd.asarray(x), "sq": sd.asarray(q)}

    return setup


def _tree_setup(npts, m, nq):
    """The uniform points with the trees built in the setup (the query cases time the query only)."""
    base = _uniform(npts, m, nq)

    def setup():
        import scipy.spatial as S
        from sklearn.neighbors import KDTree as SKDTree

        import stardust as sd

        d = base()
        d["T"], d["ST"], d["sT"] = S.KDTree(d["x"]), SKDTree(d["x"]), sd.spatial.KDTree(d["sx"])
        return d

    return setup


def _sp():
    import stardust as sd

    return sd.spatial


def _scipy_d():
    import scipy.spatial.distance as D

    return D


R_DIST = "dist(matrix(r_a, nrow = r_m), method = '{}')"
R_DATA = {"r_a": "r_a", "r_m": "r_m"}
BALL_R = (10 / 10**6 * 3 / (4 * np.pi)) ** (1 / 3)  # about 10 points of 10^6 uniform ones in the unit cube


def _distance_specs():
    out = []
    for ma, n in ((1000, 3), (1000, 64), (4000, 3), (4000, 64), (4000, 512)):
        size = f"({ma}, {n})"
        refs = (
            Ref("scipy", lambda d: lambda: _scipy_d().cdist(d["a"], d["b"])),
            Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"], d["b"]))(_sk("euclidean_distances")), threads="follow",
                differs="the expanded form (cancellation; docs/design/spatial.md §3.2)"),
        )  # fmt: skip
        out.append(Spec("SPT1", "distance", "cdist euclidean", ("cdist",), size, _normal(ma, ma, n),
                        lambda d: lambda: _sp().cdist(d["sa"], d["sb"]), refs, quick=(ma, n) == (1000, 64)))  # fmt: skip
    for metric in ("cityblock", "chebyshev", "minkowski", "cosine", "correlation", "canberra", "braycurtis"):
        kw = {"p": 3.0} if metric == "minkowski" else {}
        out.append(Spec("SPT1", "distance", f"cdist {metric}", ("cdist",), "(1000, 64)", _normal(1000, 1000, 64),
                        lambda d, m=metric, kw=kw: lambda: _sp().cdist(d["sa"], d["sb"], metric=m, **kw),
                        (Ref("scipy", lambda d, m=metric, kw=kw: lambda: _scipy_d().cdist(d["a"], d["b"], m, **kw)),)))  # fmt: skip
    for metric, kw in (("seuclidean", {}), ("mahalanobis", {})):
        n = 16 if metric == "mahalanobis" else 64
        out.append(Spec("SPT1", "distance", f"cdist {metric}", ("cdist",), f"(1000, {n})", _normal(1000, 1000, n),
                        lambda d, m=metric: lambda: _sp().cdist(d["sa"], d["sb"], metric=m),
                        (Ref("scipy", lambda d, m=metric: lambda: _scipy_d().cdist(d["a"], d["b"], m)),),
                        note="default V / VI from the data"))  # fmt: skip
    out.append(Spec("SPT1", "distance", "cdist hamming bool", ("cdist",), "(1000, 256)", _bool(1000, 1000, 256),
                    lambda d: lambda: _sp().cdist(d["sa"], d["sb"], metric="hamming"),
                    (Ref("scipy", lambda d: lambda: _scipy_d().cdist(d["a"], d["b"], "hamming")),),
                    note="bit-packed rows (SPT23)"))  # fmt: skip
    out.append(Spec("SPT1", "distance", "cdist haversine", ("cdist",), "(2000, 2)", _latlon(2000),
                    lambda d: lambda: _sp().cdist(d["sa"], d["sa"], metric="haversine"),
                    (Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"]))(_sk("haversine_distances")),
                         threads="follow"),)))  # fmt: skip
    for n in (3, 64):
        out.append(Spec("SPT1", "distance", "pdist cosine", ("pdist",), f"(2000, {n})", _normal(2000, 1, n),
                        lambda d: lambda: _sp().pdist(d["sa"], metric="cosine"),
                        (Ref("scipy", lambda d: lambda: _scipy_d().pdist(d["a"], "cosine")),)))  # fmt: skip
    for n in (3, 64):
        refs = [Ref("scipy", lambda d: lambda: _scipy_d().pdist(d["a"]))]
        refs.append(RRef("R", R_DIST.format("euclidean"), data=R_DATA))
        out.append(Spec("SPT1", "distance", "pdist euclidean", ("pdist",), f"(2000, {n})", _normal(2000, 1, n),
                        lambda d: lambda: _sp().pdist(d["sa"]), tuple(refs), quick=n == 3))  # fmt: skip
    return out


def _sk(name):
    import sklearn.metrics.pairwise as K

    return getattr(K, name)


def _neighbor_specs():
    out = []
    for n in (3, 64, 512):
        refs = (
            Ref("sklearn", lambda d: (lambda f: lambda: f(d["a"], d["b"]))(_sk("pairwise_distances_argmin_min")),
                threads="follow", differs="the expanded form (cancellation; docs/design/spatial.md §3.2)"),
        )  # fmt: skip
        out.append(Spec("SPT2", "neighbors", "kneighbors k=1", ("kneighbors",), f"(4000, {n})", _normal(4000, 4000, n),
                        lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"]), refs, quick=n == 64))  # fmt: skip
        out.append(Spec("SPT2", "neighbors", "kneighbors k=10", ("kneighbors",), f"(4000, {n})", _normal(4000, 4000, n),
                        lambda d: lambda: _sp().kneighbors(d["sa"], d["sb"], k=10),
                        (Ref("sklearn brute", _nn_brute(10), threads="follow"),)))  # fmt: skip
    return out


def _nn_brute(k):
    def make(d):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(d["b"])
        return lambda: nn.kneighbors(d["a"])

    return make


def _tree_specs():
    out = []
    for npts in (10**5, 10**6):
        exp = int(round(np.log10(npts)))
        out.append(Spec("SPT3", "tree", "KDTree build m=3", ("KDTree",), f"10^{exp} points", _uniform(npts, 3, 1),
                        lambda d: lambda: _sp().KDTree(d["sx"]),
                        (Ref("scipy", lambda d: (lambda S: lambda: S.KDTree(d["x"]))(_scipy_s())),
                         Ref("sklearn", lambda d: (lambda K: lambda: K(d["x"]))(_sk_tree())))))  # fmt: skip
    for m, npts in ((3, 10**6), (16, 10**5)):
        for k in (1, 10):
            refs = (
                Ref("scipy", lambda d, k=k: lambda: d["T"].query(d["q"], k=k)),
                Ref("scipy workers=-1", lambda d, k=k: lambda: d["T"].query(d["q"], k=k, workers=-1), lib="scipy"),
                Ref("sklearn", lambda d, k=k: lambda: d["ST"].query(d["q"], k=k)),
            )
            out.append(Spec("SPT3", "tree", f"query m={m} k={k}", ("KDTree",), "10^4 points", _tree_setup(npts, m, 10**4),
                            lambda d, k=k: lambda: d["sT"].query(d["sq"], k=k), refs, quick=(m, k) == (3, 10)))  # fmt: skip
    out.append(Spec("SPT4", "tree", "query_ball_point m=3", ("KDTree",), "10^4 points", _tree_setup(10**6, 3, 10**4),
                    lambda d: lambda: d["sT"].query_ball_point(d["sq"], BALL_R),
                    (Ref("scipy", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R)),
                     Ref("scipy workers=-1", lambda d: lambda: d["T"].query_ball_point(d["q"], BALL_R, workers=-1),
                         lib="scipy")), note="mean about 10 points per query"))  # fmt: skip
    return out


def _scipy_s():
    import scipy.spatial as S

    return S


def _sk_tree():
    from sklearn.neighbors import KDTree

    return KDTree


SPECS = _distance_specs() + _neighbor_specs() + _tree_specs()


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.spatial.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(area="spatial", group=s.group, name=s.name, size=s.size, setup=s.setup, sd=s.sd, refs=s.refs,
             quick=s.quick, note=s.note)
        for s in SPECS
        if _public(s.needs)
    ]  # fmt: skip


TITLES = {
    "distance": "거리 (`pdist`, `cdist`)",
    "neighbors": "한 번에 끝나는 이웃 찾기",
    "tree": "KD 트리",
}
