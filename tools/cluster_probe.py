"""Probes of sd.cluster's foundations and of the libraries it is compared with (docs/design/cluster.md §2, §3, §4, §5,
§7.4). The records the tests read go to tests/cluster/data/; the timings go to the terminal and to
build/logs/cluster_probe/<stamp>.log, and docs/design/cluster.md quotes them.

    .venv/bin/python tools/cluster_probe.py                  # every section
    .venv/bin/python tools/cluster_probe.py proto scipy r    # only these sections
    .venv/bin/python tools/cluster_probe.py proto.kmeans     # one measurement (the others' records kept)

Sections (positional; default all):
  proto    the measurements of the area's kernels -> prototype.json (milestone CL0's prototypes in prototype.cpp,
           which milestone CL2 removed, are gone: their records distance/*, bounds/*, centers/*, nnchain/* and
           heights stay as they were):
             kmeans    (milestone CL2: csrc/cluster/kmeans.cpp) the assignment's difference form, its proven filter
                       by the expanded form (decision CL49) and Accelerate's expanded form alone, at 1 and 10
                       threads; whole fits by path (Lloyd, bounds, filtered) with the same bits; the key order's cost
             mst       (milestone CL1: the area's kernels, csrc/cluster/mst.cpp) Prim without a matrix, Boruvka
                       with each point's search of the KD tree and Boruvka by the dual tree: the same tree, and
                       the times by n and p (where Boruvka starts to win; decisions CL47, CL54)
             tree_heights  the Lance-Williams recurrence evaluated on the final tree (path-free), every
                       linkage, in ULPs of the exact heights, beside scipy's
  scipy    scipy 1.18.1's hierarchy and vq on the test data and at the edges the design departs from -> scipy.json
  sklearn  scikit-learn 1.9.1's KMeans, DBSCAN, HDBSCAN, OPTICS, GaussianMixture and indices -> sklearn.json
  r        R 4.6.2's stats (hclust, cutree, kmeans) and cluster 2.1.8.3 (pam, clara, fanny, agnes, diana,
           silhouette, clusGap, maxSE, daisy) through bench/accuracy/rsession.py -> r.json
  time     the comparators' timings of §5 (best of three unless marked *)

Records named rule/... are the behaviours the design departs from or relies on (§2.8). Every number is taken from
one run on a machine shared with other work; the log notes the load average.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import cli  # noqa: E402

DATA = ROOT / "tests" / "cluster" / "data"
SECTIONS = ("proto", "scipy", "sklearn", "r", "time")
TOOL = "tools/cluster_probe.py"
U = 2.0**-53
HIER = ("single", "complete", "average", "weighted", "centroid", "median", "ward")


def S():
    from tests.cluster import samples

    return samples


def RD():
    from tests.cluster import refdata

    return refdata


def P():
    from stardust import _core

    return _core.cluster


class Log:
    def __init__(self):
        self.path = cli.log_file("cluster_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


def outcome(fn):
    """{"value": ...} or {"error": type, "msg": message}, and the warnings fn() raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001 -- recording what the library raises
            out = {"error": type(e).__name__, "msg": str(e)[:300]}
        else:
            out = {"value": plain(v)}
    if w:
        out["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:200]}" for x in w})
    return out


def plain(v):
    if isinstance(v, np.ndarray):
        return {"array": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (float, bool, int, str)) or v is None:
        if isinstance(v, float) and not math.isfinite(v):
            return repr(v)
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def write_records(name, meta, records):
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(plain(v))}" for k, v in records.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {(DATA / name).relative_to(ROOT)}"


def best(fn, reps=3):
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


def threads(n):
    import stardust as sd

    sd.set_num_threads(n)


def sda(a):
    import stardust as sd

    return sd.asarray(np.ascontiguousarray(a, dtype=float))


def arr(x):
    return np.asarray(x.tolist())


def partition_same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return len(set(zip(a.tolist(), b.tolist()))) == len(set(a.tolist())) == len(set(b.tolist()))


# ---------------------------------------------------------------- proto: k-means (milestone CL2)


def proto_kmeans(log, rec):
    """Milestone CL2's measurements of k-means' kernels (csrc/cluster/kmeans.cpp) on Gaussian blobs of 10^5 points
    (decision CL49 and the paths' thresholds): the assignment alone (kmeans_predict with path 1, the difference form
    on spatial's tile kernel; path 2, the expanded form's proven filter with Accelerate's dgemm and the difference
    form on the centres it keeps) against Accelerate's expanded form alone (_kmeans_assign_expanded, scikit-learn's
    form, Accelerate's own threads), at 1 and 10 threads; then whole fits from the same start by path (1 Lloyd,
    2 Lloyd filtered, 3 Hamerly's bounds, 4 bounds filtered), which must give the same bits; and the gather of the
    rows in key order against one assignment."""
    C = sd_core()
    g = np.random.Generator(np.random.PCG64(2026))
    n = 10**5
    log("kmeans/assign: ms of one assignment of 10^5 points (1 / 10 threads); Accelerate's expanded form")
    assign = {}
    for p in (2, 10, 50, 100, 300):
        for k in (10, 100):
            cs = 4.0 * g.standard_normal((k, p))
            X = cs[g.integers(0, k, n)] + g.standard_normal((n, p))
            sX, sC = sda(X), sda(cs)
            row = {}
            for th in (1, 10):
                threads(th)
                for path, name in ((1, "diff"), (2, "filtered")):
                    row[f"{name}_{th}"] = best(lambda: C.kmeans_predict(sX, sC, sC, 0, path))
                row[f"expanded_{th}"] = best(lambda: C._kmeans_assign_expanded(sX, sC))
            threads(10)
            a = arr(C.kmeans_predict(sX, sC, sC, 0, 1)[0])
            row["same_labels"] = bool(np.array_equal(a, arr(C.kmeans_predict(sX, sC, sC, 0, 2)[0])))
            row["filtered_over_expanded_1"] = row["filtered_1"] / row["expanded_1"]
            row["diff_over_expanded_1"] = row["diff_1"] / row["expanded_1"]
            assign[f"p{p}k{k}"] = row
            log(
                f"  p={p:>3} k={k:>3}: "
                + ", ".join(f"{q} {v:.2f}" for q, v in row.items() if q != "same_labels")
                + f", same labels {row['same_labels']}"
            )
    rec["kmeans/assign"] = assign
    log("kmeans/paths: ms of a fit from the data's first k rows (1 / 10 threads), tol 1e-4")
    paths = {}
    for p, k in ((2, 10), (10, 10), (10, 100), (50, 100), (100, 100), (2, 100), (50, 10)):
        cs = 4.0 * g.standard_normal((k, p))
        X = cs[g.integers(0, k, n)] + g.standard_normal((n, p))
        sX, I = sda(X), sda(X[:k])
        order = C.lex_order(sX)
        row = {}
        outs = []
        for th in (1, 10):
            threads(th)
            for path in (1, 2, 3, 4):
                row[f"path{path}_{th}"] = best(lambda: C.kmeans(sX, order, None, k, 2, I, 1, 300, 1e-4, 1, 2, path), 1)
                if th == 10:
                    outs.append(C.kmeans(sX, order, None, k, 2, I, 1, 300, 1e-4, 1, 2, path))
        threads(10)
        row["n_iter"] = int(outs[0][1])
        row["same_bits"] = all(
            np.array_equal(arr(o[0][0]), arr(outs[0][0][0]))
            and np.array_equal(arr(o[0][1]), arr(outs[0][0][1]))
            and o[0][4] == outs[0][0][4]
            for o in outs
        )
        paths[f"p{p}k{k}"] = row
        log(
            f"  p={p:>3} k={k:>3}: "
            + ", ".join(f"{q} {v:.1f}" if isinstance(v, float) else f"{q} {v}" for q, v in row.items())
        )
    rec["kmeans/paths"] = paths
    threads(1)
    X = g.standard_normal((n, 10))
    sX = sda(X)
    order = C.lex_order(sX)
    rec["kmeans/gather"] = {
        "lex_order_ms": best(lambda: C.lex_order(sX)),
        "one_iteration_ms": best(lambda: C.kmeans(sX, order, None, 10, 2, sda(X[:10]), 1, 1, 0.0, 1, 2, 1)),
    }
    threads(10)
    log(f"  key order and gather of 10^5 x 10 (1 thread): {rec['kmeans/gather']}")


# ---------------------------------------------------------------- proto: MST


def proto_mst(log, rec):
    """Milestone CL1's measurement of decision CL47 with the area's kernels (csrc/cluster/mst.cpp through
    _core.cluster._mst): Prim without a matrix (0), Boruvka with each point's own search of the KD tree (1, what
    linkage uses) and Boruvka by the dual tree (2): the same tree, and the times by n and p (where Boruvka starts
    to win). The data in key order as linkage gives it."""
    C = sd_core()
    g = np.random.Generator(np.random.PCG64(12))
    log("mst: Prim without a matrix (0), Boruvka per point (1), Boruvka by the dual tree (2); ms at 1 / 10 threads")
    same = {}
    for p in (2, 5, 16):
        X = keyed(g.random((3000, p)))
        outs = [C._mst(sda(X), m) for m in (0, 1, 2)]
        same[f"p{p}"] = all(
            np.array_equal(arr(outs[0][0]), arr(o[0])) and np.array_equal(arr(outs[0][1]), arr(o[1])) for o in outs[1:]
        )
    X = keyed(np.array([[i, j] for i in range(40) for j in range(40)], float))
    outs = [C._mst(sda(X), m) for m in (0, 1, 2)]
    same["lattice40"] = all(np.array_equal(arr(outs[0][0]), arr(o[0])) for o in outs[1:])
    rec["mst/same_tree"] = same
    log(f"  the same edges and weights: {same}")
    times = {}
    for p in (2, 3, 4, 5, 6, 8, 16):
        for n in (512, 1024, 2048, 4096, 10**4, 3 * 10**4, 10**5):
            if p >= 8 and n > 3 * 10**4:
                continue
            X = keyed(g.random((n, p)))
            sX = sda(X)
            row = {}
            for th in (1, 10):
                threads(th)
                reps = 1 if n > 10**4 else 3
                if n <= 3 * 10**4:
                    row[f"prim_{th}"] = best(lambda: C._mst(sX, 0), reps)
                row[f"boruvka_{th}"] = best(lambda: C._mst(sX, 1), reps)
                if n <= 3 * 10**4 and p <= 8:
                    row[f"dual_{th}"] = best(lambda: C._mst(sX, 2), reps)
            times[f"p{p}n{n}"] = row
            log(f"  p={p:>2} n={n:>6}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items()))
    threads(10)
    rec["mst/time"] = times


def sd_core():
    import stardust as sd

    return sd._core.cluster


def keyed(X):
    """The points in key order (decision CL7), as linkage passes them to its kernels."""
    return np.ascontiguousarray(X[np.asarray(sd_core().lex_order(sda(X)))])


# ---------------------------------------------------------------- proto: heights


def tree_heights(X, Z, method):
    """The heights of a tree by the Lance-Williams recurrence evaluated on the tree itself (float64): the value of
    a merge's two children P = P1 + P2 and Q is the recurrence on (P1, Q), (P2, Q) and P's own height, and
    (x, Q) for a point x recurs down Q the same way to the points' distances (spatial's kernel). The children in
    Z's order; ward, centroid and median on squared values. Path-free: the tree and the data fix every
    operation. Data far from 1 in magnitude are scaled by a power of two first (exact), the heights back."""
    m = float(np.max(np.abs(X)))
    e = -math.frexp(m)[1] if m > 0 else 0
    if e:
        return np.ldexp(tree_heights(np.ldexp(X, e), Z, method), -e)
    n = X.shape[0]
    kids = {n + i: (int(r[0]), int(r[1])) for i, r in enumerate(Z)}
    size = {i: 1 for i in range(n)}
    for i, r in enumerate(Z):
        size[n + i] = size[int(r[0])] + size[int(r[1])]
    sq = method in ("ward", "centroid", "median")
    H = {}

    def base(x, y):
        d = X[x] - X[y]
        s2 = float(np.dot(d, d))
        return s2 if sq else math.sqrt(s2)

    def val(a, b):
        """The linkage value of clusters a and b (a expanded first)."""
        if a < n and b < n:
            return base(a, b)
        if a < n:
            a, b = b, a
        a1, a2 = kids[a]
        v1, v2, hab = val(a1, b), val(a2, b), H[a]
        n1, n2, nb_ = size[a1], size[a2], size[b]
        if method == "single":
            return min(v1, v2)
        if method == "complete":
            return max(v1, v2)
        if method == "average":
            return (n1 * v1 + n2 * v2) / (n1 + n2)
        if method == "weighted":
            return 0.5 * (v1 + v2)
        if method == "centroid":
            t = n1 + n2
            return (n1 * v1 + n2 * v2) / t - (n1 * n2) * hab / (t * t)
        if method == "median":
            return 0.5 * v1 + 0.5 * v2 - 0.25 * hab
        t = 1.0 / (n1 + n2 + nb_)
        return (n1 + nb_) * t * v1 + (n2 + nb_) * t * v2 - nb_ * t * hab

    out = np.empty(len(Z))
    for i, r in enumerate(Z):
        v = val(int(r[0]), int(r[1]))
        H[n + i] = v
        out[i] = math.sqrt(max(v, 0.0)) if sq else v
    return out


def proto_tree_heights(log, rec):
    import scipy.cluster.hierarchy as sch

    R = RD().ref("hierarchy")
    D = RD()
    log("tree heights: the Lance-Williams recurrence on the final tree, against the exact (ULPs of h*)")
    out = {}
    sys.setrecursionlimit(10000)
    for method in HIER:
        worst, where, worst_lw = 0.0, None, 0.0
        for sid in R.group_inputs["set"].tolist():
            s = S().points(sid)
            Zr, gaps, lo = D.tree(R, method, sid)
            ex = Zr[:, 2] + lo
            h = tree_heights(s.X, Zr, method)
            ulp = float(np.max(np.abs(h - ex) / np.maximum(np.spacing(np.abs(ex)), 2.0**-1074)))
            if ulp > worst:
                worst, where = ulp, sid
            if np.min(gaps) > D.NEAR_TIE:
                Zs = sch.linkage(s.X, method)
                if D.merges(Zs, s.n) == D.merges(Zr, s.n):
                    mem = {i: frozenset([i]) for i in range(s.n)}
                    key = {}
                    for i, row in enumerate(Zs):
                        mem[s.n + i] = mem[int(row[0])] | mem[int(row[1])]
                        key[mem[s.n + i]] = row[2]
                    memr = {i: frozenset([i]) for i in range(s.n)}
                    lw = np.empty(len(Zr))
                    for i, row in enumerate(Zr):
                        memr[s.n + i] = memr[int(row[0])] | memr[int(row[1])]
                        lw[i] = key[memr[s.n + i]]
                    worst_lw = max(
                        worst_lw, float(np.max(np.abs(lw - ex) / np.maximum(np.spacing(np.abs(ex)), 2.0**-1074)))
                    )
        out[method] = {"tree_ulp": worst, "where": where, "scipy_ulp": worst_lw}
        log(f"  {method}: tree recurrence {worst:.3g} ULP ({where}), scipy's Lance-Williams {worst_lw:.3g} ULP")
    rec["heights/tree_recurrence"] = out


PROTO = {
    "kmeans": lambda: proto_kmeans,
    "mst": lambda: proto_mst,
    "tree_heights": lambda: proto_tree_heights,
}


def proto(log, only=None):
    """Every prototype measurement, or only those named (proto.distance ...). The records of the sections not run
    (and of milestone CL0's sections milestones CL1 and CL2 retired: nnchain/*, heights, distance/*, bounds/*,
    centers/*) are kept."""
    rec = {}
    if (DATA / "prototype.json").exists():
        old = json.loads((DATA / "prototype.json").read_text())
        old.pop("_provenance")
        rec.update(old)
    for name, fn in PROTO.items():
        if only and name not in only:
            continue
        fn()(log, rec)
        log()
    import flint
    import scipy
    import sklearn

    meta = {
        "tool": TOOL,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "python-flint": flint.__version__,
        "machine": "Apple M4 Pro, macOS 26.6.2",
        "load_average": round(os.getloadavg()[0], 1),
        "note": "milestone CL0's prototypes (prototype.cpp, removed by milestone CL2) and the area's kernels",
    }
    log(write_records("prototype.json", meta, rec))


# ---------------------------------------------------------------- scipy


def _scipy_tree(Z, n):
    return sorted(sorted(m) for m in RD().merges(Z, n))


def scipy_records(log):
    import scipy
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.vq import kmeans2, vq
    from scipy.spatial.distance import pdist

    rec = {}
    R = RD().ref("hierarchy")
    for sid in R.group_inputs["set"].tolist():
        s = S().points(sid)
        for m in HIER:
            rec[f"linkage/{sid}/{m}"] = outcome(lambda m=m: sch.linkage(s.X, m))
    # input order: the same lattice shuffled
    a, b = S().points("lattice/4x4"), S().points("lattice/4x4-shuffled")
    perm = [int(np.nonzero((a.X == row).all(axis=1))[0][0]) for row in b.X]
    for m in ("single", "complete", "average", "ward"):
        Za, Zb = sch.linkage(a.X, m), sch.linkage(b.X, m)
        tb = RD().merges(Zb, 16)
        mapped = {frozenset(perm[i] for i in c) for c in tb}
        rec[f"rule/linkage_input_order/{m}"] = {"same_tree": mapped == RD().merges(Za, 16)}
    # fcluster(maxclust) below the asked count at tied heights; cut_tree exact
    line = S().points("line/7").X
    Z = sch.linkage(line, "single")
    rec["rule/fcluster_maxclust_ties"] = {
        "fcluster_maxclust_2": sch.fcluster(Z, 2, "maxclust").tolist(),
        "cut_tree_2": sch.cut_tree(Z, n_clusters=2).ravel().tolist(),
    }
    inv = S().points("inversion/4").X
    Z = sch.linkage(inv, "centroid")
    rec["rule/inversion_cuts"] = {
        "heights": Z[:, 2].tolist(),
        "is_monotonic": bool(sch.is_monotonic(Z)),
        "fcluster_maxclust_3": sch.fcluster(Z, 3, "maxclust").tolist(),
        "cut_tree_3": sch.cut_tree(Z, n_clusters=3).ravel().tolist(),
        "fcluster_distance_1.9": sch.fcluster(Z, 1.9, "distance").tolist(),
        "cut_tree_height_1.9": sch.cut_tree(Z, height=1.9).ravel().tolist(),
    }
    g = np.random.Generator(np.random.PCG64(3))
    bad = {}
    for _ in range(200):
        X = g.standard_normal((30, 2))
        for m in ("single", "complete", "average", "ward", "centroid", "median"):
            Z = sch.linkage(X, m)
            for k in (2, 3, 5, 8):
                if not partition_same(sch.cut_tree(Z, n_clusters=k).ravel(), sch.fcluster(Z, k, "maxclust")):
                    bad[m] = bad.get(m, 0) + 1
    rec["rule/cut_tree_vs_maxclust"] = {"cases": 800, "differ": bad}
    X = S().points("inversion/random30").X
    rec["rule/precomputed_ward_cityblock"] = outcome(lambda: sch.linkage(pdist(X, "cityblock"), "ward")[:3])
    rec["rule/linkage_square_input"] = outcome(lambda: sch.linkage(np.array([[0.0, 1.0], [1.0, 0.0]]), "single"))
    rec["rule/observations_noneuclidean_ward"] = outcome(lambda: sch.linkage(X, "ward", metric="cityblock"))
    rec["rule/is_valid_linkage_signature"] = {
        "params": list(__import__("inspect").signature(sch.is_valid_linkage).parameters)
    }
    for sid in ("blobs/k2p1", "lattice/3x3", "inversion/random30"):
        s = S().points(sid)
        Z = sch.linkage(s.X, "average")
        y = pdist(s.X)
        rec[f"tools/{sid}"] = {
            "leaves_list": sch.leaves_list(Z).tolist(),
            "optimal_leaf_ordering": sch.leaves_list(sch.optimal_leaf_ordering(Z, y)).tolist(),
            "cophenet": sch.cophenet(Z).tolist(),
            "cophenet_correlation": float(sch.cophenet(Z, y)[0]),
            "inconsistent": sch.inconsistent(Z).tolist(),
            "maxdists": sch.maxdists(Z).tolist(),
            "fcluster_inconsistent_default": sch.fcluster(Z, 1.0).tolist(),
            "dendrogram": {k: plain(v) for k, v in sch.dendrogram(Z, no_plot=True).items()},
        }
    C = np.array([[-1.0, 0.0], [1.0, 0.0]])
    Pq = np.array([[0.0, 0.0], [0.0, 5.0]])
    rec["rule/vq_tie"] = {"vq": vq(Pq, C)[0].tolist(), "vq_swapped": vq(Pq, C[::-1])[0].tolist()}
    E = np.array([[0.0], [0.1], [10.0], [10.1]])
    rec["rule/kmeans2_empty"] = outcome(lambda: kmeans2(E, np.array([[0.0], [10.0], [100.0]]), minit="matrix")[1])
    meta = {"tool": TOOL, "scipy": scipy.__version__, "numpy": np.__version__}
    log(write_records("scipy.json", meta, rec))


# ---------------------------------------------------------------- scikit-learn


def sklearn_records(log):
    import sklearn
    from sklearn import metrics as M
    from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

    rec = {}
    R = RD().ref("kmeans")
    for sid in R.group_inputs["set"].tolist():
        gi = RD().group_of(R, sid)
        I = R.group_inputs["init"][gi]
        s = S().points(sid)
        if I is None or np.size(I) == 0 or np.isnan(np.asarray(I, float)).all():
            continue
        I = np.asarray(I, float).reshape(-1, s.p)

        def fit(I=I, s=s):
            km = KMeans(len(I), init=I, n_init=1, tol=0.0, max_iter=300).fit(s.X, sample_weight=s.w)
            return {"labels": km.labels_, "centers": km.cluster_centers_, "inertia": km.inertia_, "n_iter": km.n_iter_}

        rec[f"kmeans/{sid}"] = outcome(fit)
    X = np.random.default_rng(5).normal(size=(200, 2))
    rec["rule/kmeans_n_init_auto"] = {
        "k-means++": KMeans(3, random_state=0).fit(X)._n_init,
        "random": KMeans(3, init="random", random_state=0).fit(X)._n_init,
    }
    C = np.array([[-1.0, 0.0], [1.0, 0.0]])
    Pq = np.array([[0.0, 0.0], [0.0, 5.0]])
    km = KMeans(2, init=C, n_init=1, max_iter=1).fit(C)
    km2 = KMeans(2, init=C[::-1], n_init=1, max_iter=1).fit(C[::-1])
    rec["rule/kmeans_tie"] = {"predict": km.predict(Pq).tolist(), "predict_swapped": km2.predict(Pq).tolist()}
    fd = S().points("fewdistinct/10").X
    rec["rule/kmeans_fewdistinct"] = outcome(lambda: KMeans(3, n_init=1, random_state=0).fit(fd).labels_)
    E = np.array([[0.0], [0.1], [10.0], [10.1]])
    rec["rule/kmeans_empty"] = outcome(
        lambda: KMeans(3, init=np.array([[0.0], [10.0], [100.0]]), n_init=1).fit(E).labels_
    )
    rec["rule/kmeans_nan"] = outcome(lambda: KMeans(2, n_init=1).fit(np.array([[0.0], [np.nan], [1.0]])))
    Xe = np.random.default_rng(12).normal(size=(20000, 4))
    a = KMeans(8, init=Xe[:8], n_init=1, algorithm="lloyd", tol=0).fit(Xe)
    b = KMeans(8, init=Xe[:8], n_init=1, algorithm="elkan", tol=0).fit(Xe)
    rec["rule/kmeans_lloyd_elkan"] = {
        "labels_equal": bool(np.array_equal(a.labels_, b.labels_)),
        "centers_max_diff": float(np.abs(a.cluster_centers_ - b.cluster_centers_).max()),
    }
    from threadpoolctl import threadpool_limits

    Xt = np.random.default_rng(7).normal(size=(200_000, 8))
    out = {}
    for t in (1, 8):
        with threadpool_limits(limits=t, user_api="openmp"):
            out[t] = KMeans(20, init=Xt[:20].copy(), n_init=1, max_iter=50, tol=0).fit(Xt)
    rec["rule/kmeans_threads"] = {
        "labels_equal": bool(np.array_equal(out[1].labels_, out[8].labels_)),
        "centers_bit_equal": bool(np.array_equal(out[1].cluster_centers_, out[8].cluster_centers_)),
        "centers_max_diff": float(np.abs(out[1].cluster_centers_ - out[8].cluster_centers_).max()),
        "inertia": [out[1].inertia_, out[8].inertia_],
    }
    # DBSCAN
    D = RD().ref("density")
    for sid in D.group_inputs["set"].tolist():
        if "|eps" not in sid:
            continue
        base, e, m = sid.split("|")
        s = S().points(base)
        rec[f"dbscan/{sid}"] = outcome(
            lambda s=s, e=float(e[3:]), m=int(m[2:]): DBSCAN(eps=e, min_samples=m).fit(s.X, sample_weight=s.w).labels_
        )
    left = np.array([[-1.0 - 0.05 * i, 0.0] for i in range(4)])
    right = np.array([[1.0 + 0.05 * i, 0.0] for i in range(4)])
    Xb = np.vstack([left, right, [[0.0, 0.0]]])
    rows = {}
    for name, idx in (("left_first", np.arange(9)), ("right_first", np.r_[4:8, 0:4, 8])):
        lab = DBSCAN(eps=1.0, min_samples=4).fit(Xb[idx]).labels_
        back = np.empty(9, int)
        back[idx] = lab
        rows[name] = "left" if back[8] == back[0] else "right"
    rec["rule/dbscan_border"] = rows
    rec["rule/dbscan_eps_inclusive"] = DBSCAN(eps=1.0, min_samples=2).fit(np.array([[0.0], [1.0]])).labels_
    for sid in ("blobs/k3p2", "density/k3p2", "rings/p2", "moons/p2"):
        s = S().points(sid)

        def hd(s=s):
            h = HDBSCAN(copy=True).fit(s.X)
            return {"labels": h.labels_, "probabilities": h.probabilities_}

        rec[f"hdbscan/{sid}"] = outcome(hd)

        def op(s=s):
            o = OPTICS().fit(s.X)
            return {
                "labels": o.labels_,
                "ordering": o.ordering_,
                "reachability": o.reachability_,
                "core": o.core_distances_,
            }

        rec[f"optics/{sid}"] = outcome(op)
    s = S().points("blobs/k3p2")
    rec["rule/hdbscan_copy_warning"] = outcome(lambda: HDBSCAN().fit(s.X).labels_.max())
    Xd = np.vstack([s.X, np.repeat(s.X[:1], 30, axis=0)])
    rec["rule/hdbscan_duplicates"] = outcome(lambda: sorted(set(HDBSCAN(copy=True).fit(Xd).labels_.tolist())))
    # mixtures
    for mid in S().mixture_ids():
        mx = S().mixture(mid)

        def ss(mx=mx):
            gm = GaussianMixture(mx.k, covariance_type=mx.covariance_type)
            gm.weights_, gm.means_, gm.covariances_ = mx.weights, mx.means, mx.covariances
            gm.precisions_cholesky_ = _compute_precision_cholesky(mx.covariances, mx.covariance_type)
            return {"score_samples": gm.score_samples(mx.X), "predict_proba": gm.predict_proba(mx.X)}

        rec[f"gmm_given/{mid}"] = outcome(ss)
    g = np.random.default_rng(9)
    X2 = np.vstack([g.normal(0, 1, (200, 2)), g.normal(5, 1, (200, 2))])
    gm = GaussianMixture(2, random_state=0).fit(X2)
    rec["rule/gmm_lower_bound_vs_score"] = {
        "converged": bool(gm.converged_),
        "n_iter": int(gm.n_iter_),
        "lower_bound": float(gm.lower_bound_),
        "score": float(gm.score(X2)),
    }
    # indices
    for lid in S().labeling_ids():
        lb = S().labeling(lid)
        fs = {
            "ari": M.adjusted_rand_score, "ri": M.rand_score, "fm": M.fowlkes_mallows_score,
            "mi": M.mutual_info_score, "homogeneity": M.homogeneity_score, "completeness": M.completeness_score,
            "v": M.v_measure_score,
        }  # fmt: skip
        row = {k: outcome(lambda f=f: f(lb.a, lb.b)) for k, f in fs.items()}
        for avg in ("min", "geometric", "arithmetic", "max"):
            row[f"nmi_{avg}"] = outcome(lambda avg=avg: M.normalized_mutual_info_score(lb.a, lb.b, average_method=avg))
            if lb.n <= 10**5:
                row[f"ami_{avg}"] = outcome(
                    lambda avg=avg: M.adjusted_mutual_info_score(lb.a, lb.b, average_method=avg)
                )
        rec[f"metrics/{lid}"] = row
    for sid in S().point_ids():
        s = S().points(sid)
        if s.truth is None or s.n > 300:
            continue
        rec[f"internal/{sid}"] = {
            "silhouette_samples": outcome(lambda s=s: M.silhouette_samples(s.X, s.truth)),
            "silhouette_score": outcome(lambda s=s: M.silhouette_score(s.X, s.truth)),
            "calinski_harabasz": outcome(lambda s=s: M.calinski_harabasz_score(s.X, s.truth)),
            "davies_bouldin": outcome(lambda s=s: M.davies_bouldin_score(s.X, s.truth)),
        }
    Xs = np.array([[0.0], [0.1], [5.0], [9.0], [9.2]])
    rec["rule/silhouette_singleton"] = M.silhouette_samples(Xs, np.array([0, 0, 1, 2, 2]))
    rec["rule/internal_one_cluster"] = {
        f.__name__: outcome(lambda f=f: f(Xs, np.zeros(5, int)))
        for f in (M.silhouette_score, M.calinski_harabasz_score, M.davies_bouldin_score)
    }
    rec["rule/label_types"] = {
        "float_integral": outcome(lambda: M.adjusted_rand_score([0.0, 0.0, 1.0], [0, 1, 1])),
        "float_fractional": outcome(lambda: M.adjusted_rand_score([0.5, 0.5, 1.5], [0, 1, 1])),
        "strings": outcome(lambda: M.adjusted_rand_score(["a", "a", "b"], [0, 1, 1])),
        "lengths_differ": outcome(lambda: M.adjusted_rand_score([0, 0, 1], [0, 1])),
    }
    meta = {"tool": TOOL, "scikit-learn": sklearn.__version__, "numpy": np.__version__}
    log(write_records("sklearn.json", meta, rec))


# ---------------------------------------------------------------- R

_R_HELP = r"""
.sd_rec <- function(f) {
  op <- options(warn = 0); on.exit(options(op))
  ws <- character(0)
  keep <- function(w) { ws <<- c(ws, conditionMessage(w)); invokeRestart("muffleWarning") }
  v <- tryCatch(withCallingHandlers(f(), warning = keep),
                error = function(e) structure(conditionMessage(e), class = "sderr"))
  if (inherits(v, "sderr")) return(c(-1, utf8ToInt(enc2utf8(as.character(v)))))
  msg <- enc2utf8(paste(ws, collapse = " | "))
  v <- as.double(unlist(v))
  c(length(v), v, nchar(msg, type = "chars"), utf8ToInt(msg))
}
"""


def r_call(body, data):
    """{"value": [...], "warnings": "..."} or {"error": "..."} of an R function body over named vectors."""
    from bench.accuracy import rsession

    try:
        out = rsession.evaluate_named(_R_HELP + f".sd_rec(function() {{ {body} }})", data).tolist()
    except RuntimeError as e:
        return {"error": str(e)[:300]}
    if out and out[0] == -1:
        return {"error": "".join(chr(int(c)) for c in out[1:])}
    n = int(out[0])
    vals = out[1 : 1 + n]
    m = int(out[1 + n])
    msg = "".join(chr(int(c)) for c in out[2 + n : 2 + n + m])
    res = {"value": vals}
    if msg:
        res["warnings"] = msg
    return res


def rmat(X):
    X = np.asarray(X, float)
    return {"xv": X.T.ravel(), "dims": np.array(X.shape, float)}


XR = "x <- matrix(xv, nrow = dims[1])"


def r_records(log):
    from bench.accuracy import rsession

    if not rsession.available():
        log("r: R is not on PATH; skipped")
        return
    rec = {}
    R = RD().ref("hierarchy")
    names = {
        "single": "single",
        "complete": "complete",
        "average": "average",
        "weighted": "mcquitty",
        "ward": "ward.D2",
    }
    for sid in R.group_inputs["set"].tolist():
        s = S().points(sid)
        for m, rn in names.items():
            rec[f"hclust/{sid}/{m}"] = r_call(
                f"{XR}; h <- hclust(dist(x), '{rn}'); c(h$merge, h$height, h$order)", rmat(s.X)
            )
        for m in ("centroid", "median"):
            rec[f"hclust/{sid}/{m}"] = r_call(
                f"{XR}; h <- hclust(dist(x)^2, '{m}'); c(h$merge, sqrt(h$height), h$order)", rmat(s.X)
            )
    X = S().points("inversion/random30").X
    rec["rule/ward_D_vs_D2"] = r_call(
        f"{XR}; c(hclust(dist(x), 'ward.D')$height, hclust(dist(x), 'ward.D2')$height)", rmat(X)
    )
    rec["rule/centroid_plain_distances"] = r_call(f"{XR}; hclust(dist(x), 'centroid')$height", rmat(X))
    inv = S().points("inversion/4").X
    rec["rule/cutree_inversion_k3"] = r_call(f"{XR}; cutree(hclust(dist(x)^2, 'centroid'), k = 3)", rmat(inv))
    rec["rule/cutree_inversion_h"] = r_call(f"{XR}; cutree(hclust(dist(x)^2, 'centroid'), h = 1.9^2)", rmat(inv))
    rec["rule/cutree_ties_k2"] = r_call(f"{XR}; cutree(hclust(dist(x), 'single'), k = 2)", rmat(S().points("line/7").X))
    Y = S().points("probe/300").X
    rec["rule/kmeans_hw_vs_lloyd"] = r_call(
        f"{XR}; c0 <- x[c(1, 2, 3), ]; "
        "c(kmeans(x, c0)$tot.withinss, kmeans(x, c0, algorithm = 'Lloyd', iter.max = 100)$tot.withinss)",
        rmat(Y),
    )
    fd = S().points("fewdistinct/10").X
    rec["rule/kmeans_fewdistinct"] = r_call(f"{XR}; kmeans(x, 3)$size", rmat(fd))
    E = np.array([[0.0], [0.1], [10.0], [10.1]])
    rec["rule/kmeans_empty_lloyd"] = r_call(
        f"{XR}; kmeans(x, matrix(c(0, 10, 100)), algorithm = 'Lloyd')$cluster", rmat(E)
    )
    rec["rule/kmeans_empty_hw"] = r_call(f"{XR}; kmeans(x, matrix(c(0, 10, 100)))$cluster", rmat(E))
    # cluster: pam on the kmedoids reference sets, the rest on the probe's data
    K = RD().ref("kmedoids")
    for sid in K.group_inputs["set"].tolist():
        base, k = sid.split("|")
        s = S().points(base)
        rec[f"pam/{sid}"] = r_call(
            f"library(cluster); {XR}; p <- pam(x, {int(k[1:])}); c(sort(p$id.med) - 1, p$objective)", rmat(s.X)
        )
    # milestone CL2: stats::kmeans with Lloyd's algorithm from the kmeans references' starts (centres, cluster,
    # tot.withinss and iter), and clara with one sample of every point and pam's swap (pamLike) against pam
    KM = RD().ref("kmeans")
    for sid in KM.group_inputs["set"].tolist():
        g = RD().group_of(KM, sid)
        init = np.asarray(KM.group_inputs["init"][g], float)
        s = S().points(sid)
        if s.w is not None or init.ndim != 2 or not np.isfinite(init).all():
            continue
        rec[f"kmeans_lloyd/{sid}"] = r_call(
            f"{XR}; c0 <- matrix(cv, nrow = {init.shape[0]}); "
            "km <- kmeans(x, c0, iter.max = 300, algorithm = 'Lloyd'); "
            "c(km$cluster - 1, as.vector(t(km$centers)), km$tot.withinss, km$iter)",
            {**rmat(s.X), "cv": np.ascontiguousarray(init.T).ravel()},
        )
    for sid in ("blobs/k3p2", "probe/300", "sizes/k3p2", "overlap/k3p2"):
        s = S().points(sid)
        rec[f"clara_pamlike/{sid}"] = r_call(
            f"library(cluster); {XR}; cl <- clara(x, 3, samples = 1, sampsize = nrow(x), pamLike = TRUE); "
            "p <- pam(x, 3); c(sort(cl$i.med) - 1, cl$objective, sort(p$id.med) - 1, p$objective[2])",
            rmat(s.X),
        )
    Y = S().points("blobs/k3p2").X
    rec["cluster/clara"] = r_call(
        f"library(cluster); {XR}; set.seed(1); cl <- clara(x, 3); c(sort(cl$i.med) - 1, cl$objective)", rmat(Y)
    )
    rec["cluster/fanny"] = r_call(f"library(cluster); {XR}; f <- fanny(x, 3); c(f$coeff, f$objective)", rmat(Y))
    rec["cluster/agnes_ac"] = r_call(f"library(cluster); {XR}; a <- agnes(x); c(a$ac, sort(a$height))", rmat(Y))
    rec["cluster/hclust_average_heights"] = r_call(f"{XR}; sort(hclust(dist(x), 'average')$height)", rmat(Y))
    rec["cluster/diana"] = r_call(f"library(cluster); {XR}; d <- diana(x); c(d$dc, d$height, d$order - 1)", rmat(Y))
    rec["cluster/clusgap"] = r_call(
        f"library(cluster); {XR}; set.seed(1); g <- clusGap(x, kmeans, K.max = 6, B = 20, verbose = FALSE); "
        "c(g$Tab[, 'gap'], g$Tab[, 'SE.sim'], maxSE(g$Tab[, 'gap'], g$Tab[, 'SE.sim']))",
        rmat(Y),
    )
    rec["cluster/daisy_gower"] = r_call(
        "library(cluster); df <- data.frame(a = c(1, 2, 3), b = factor(c('u', 'v', 'u')), c = c(TRUE, FALSE, TRUE)); "
        "as.vector(daisy(df, metric = 'gower'))",
        {"z": np.zeros(1)},
    )
    Xs = np.array([[0.0], [0.1], [5.0], [9.0], [9.2]])
    rec["cluster/silhouette_singleton"] = r_call(
        f"library(cluster); {XR}; s <- silhouette(c(1, 1, 2, 3, 3), dist(x)); s[, 3]", rmat(Xs)
    )
    # milestone CL5: silhouette on the sets with true labels, clusGap's log W with PAM (deterministic) and its
    # reference mean (random: compared statistically), maxSE's five rules, daisy's Gower on mixed frames
    for sid in S().point_ids():
        s = S().points(sid)
        if s.truth is None or s.n > 300 or not 2 <= len(set(s.truth.tolist())) <= s.n - 1:
            continue
        rec[f"silhouette/{sid}"] = r_call(
            f"library(cluster); {XR}; s <- silhouette(cl, dist(x)); s[, 3]",
            {**rmat(s.X), "cl": np.unique(s.truth, return_inverse=True)[1] + 1.0},
        )
    for sid in ("blobs/k3p2", "overlap/k3p2", "probe/300"):
        X = S().points(sid).X
        for dp in (1, 2):
            for sp in ("scaledPCA", "original"):
                rec[f"clusgap_pam/{sid}/{dp}/{sp}"] = r_call(
                    f"library(cluster); {XR}; set.seed(1); g <- clusGap(x, pam, K.max = 6, B = 100, d.power = {dp}, "
                    f"spaceH0 = '{sp}'); c(g$Tab[, 'logW'], g$Tab[, 'E.logW'], g$Tab[, 'SE.sim'])",
                    rmat(X),
                )
    g = np.random.default_rng(20261127)
    cases = []
    for _ in range(60):
        K = int(g.integers(2, 10))
        f = np.round(g.normal(size=K), 1)
        se = np.abs(np.round(g.normal(scale=0.3, size=K), 2))
        cases.append((f, se))
    out = []
    for f, se in cases:
        row = [f.tolist(), se.tolist()]
        for m in ("firstSEmax", "Tibs2001SEmax", "globalSEmax", "firstmax", "globalmax"):
            for fac in (1.0, 0.5):
                v = r_call(f"library(cluster); maxSE(f, s, method = '{m}', SE.factor = {fac})", {"f": f, "s": se})
                row.append(v["value"][0])
        out.append(row)
    rec["cluster/maxse"] = {"cases": out, "methods": ["firstSEmax", "Tibs2001SEmax", "globalSEmax", "firstmax",
                                                      "globalmax"], "factors": [1.0, 0.5]}  # fmt: skip
    frame = ("df <- data.frame(a = c(1, 2, 3, NA, 7.5, 2), b = factor(c('u', 'v', 'u', 'w', NA, 'v')), "
             "c = c(TRUE, FALSE, TRUE, TRUE, FALSE, NA), d = c(1, 0, 0, 1, 0, 0), e = c(3, 1, 2, 2, 5, 1), "
             "f = c(0.5, 2, 8, 1, 1, 4))")  # fmt: skip
    rec["cluster/daisy_mixed"] = r_call(
        f"library(cluster); {frame}; as.vector(daisy(df, metric = 'gower', type = list(asymm = 'd', symm = 'c', "
        "ordratio = 'e', logratio = 'f'), weights = c(1, 2, 1, 0.5, 1, 3)))",
        {"z": np.zeros(1)},
    )
    rec["cluster/daisy_default"] = r_call(
        f"library(cluster); {frame}; as.vector(daisy(df, metric = 'gower'))", {"z": np.zeros(1)}
    )
    Y = S().points("blobs/k3p2").X
    rec["cluster/daisy_numeric"] = r_call(f"library(cluster); {XR}; as.vector(daisy(x, metric = 'gower'))", rmat(Y))
    meta = {"tool": TOOL, "R": f"R version {rsession.version()}"}
    rec["_packages"] = r_call(
        "c(as.double(R.version$major), as.double(packageVersion('cluster')[[1, 1]]))", {"z": np.zeros(1)}
    )
    log(write_records("r.json", meta, rec))


# ---------------------------------------------------------------- timings


def timings(log):
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
    from sklearn.mixture import GaussianMixture

    log(f"timings in ms (best of three; one run when marked *), load average {os.getloadavg()[0]:.1f}")
    g = np.random.default_rng(11)
    X = g.normal(size=(5000, 5))
    for m in HIER:
        log(f"  scipy linkage {m} n=5000 p=5: {best(lambda m=m: sch.linkage(X, m)):.0f}")
    Y = g.normal(size=(100_000, 10))
    log(f"  sklearn KMeans n=1e5 p=10 k=10 n_init=1: {best(lambda: KMeans(10, n_init=1, random_state=0).fit(Y)):.0f}")
    Z = g.normal(size=(20_000, 2))
    log(f"  sklearn DBSCAN n=2e4 p=2 eps=0.05: {best(lambda: DBSCAN(eps=0.05, min_samples=5).fit(Z)):.0f}")
    log(f"  sklearn HDBSCAN n=2e4 p=2 *: {best(lambda: HDBSCAN(copy=True).fit(Z), 1):.0f}")
    log(
        "  sklearn GaussianMixture full k=5 n=2e4 p=10: "
        f"{best(lambda: GaussianMixture(5, random_state=0).fit(Y[:20000])):.0f}"
    )
    from sklearn.metrics import silhouette_score

    lab = KMeans(5, n_init=1, random_state=0).fit(Y[:20000]).labels_
    log(f"  sklearn silhouette_score n=2e4 p=10 *: {best(lambda: silhouette_score(Y[:20000], lab), 1):.0f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    parts = [s for s in args.sections if s.startswith("proto.")]
    unknown = [s for s in args.sections if s not in SECTIONS and s.removeprefix("proto.") not in PROTO]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)}, or proto.<{'|'.join(PROTO)}>)")
    log = Log()
    table = {"proto": proto, "scipy": scipy_records, "sklearn": sklearn_records, "r": r_records, "time": timings}
    if parts and "proto" not in args.sections:
        proto(log, {s.removeprefix("proto.") for s in parts})
    for s in SECTIONS:
        if (not args.sections and not parts) or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
