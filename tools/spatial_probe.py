"""Probes of sd.spatial's foundations and of the libraries it is compared with
(docs/design/spatial.md §3.1, §3.5, §5, §7.4). The records the tests read go to
tests/spatial/data/; the timings and the kernel measurements go to the terminal and to
build/logs/spatial_probe/<stamp>.log, and docs/design/spatial.md §3.1, §3.5 and §5 quote them.

    .venv/bin/python tools/spatial_probe.py                    # every section
    .venv/bin/python tools/spatial_probe.py scipy sklearn r    # only these sections

Sections (positional; default all):
  kernels  the two pair orders of csrc/spatial/pair_kernel.hpp timed in a cdist-like loop (decision
           SPT22; tools/spatial_probe_kernels.cpp), and their distances on small cases against
           sd.linalg.vector_norm(x - y): NormOrder must give its bits
  ddot     whether Accelerate's cblas_ddot(z, z) moves with the address of z (the BLAS chunks of the
           vector_norm order)
  gemm     the screening product of §3.2 called from 1-10 threads at once, in both of Accelerate's
           thread modes (the shared AMX units)
  pow      the error of the C library's pow (std::pow; Python's math.pow calls the same libm)
           against Arb, in ULP: the c_pow of the minkowski contract (§4)
  scipy    scipy 1.18.1's distance functions on the edge pairs, its KDTree's edge behaviours (a call
           that kills the process runs in a child), its summation orders, and its k nearest
           neighbours on the neighbour sets -> tests/spatial/data/scipy.json
  sklearn  scikit-learn 1.9.1's expanded-form distances, nearest-neighbour ties and errors, and
           haversine_distances -> tests/spatial/data/sklearn.json
  r        R 4.6.2's stats::dist on a few point sets (values as hex floats) -> tests/spatial/data/r.json
  time     the scipy and scikit-learn timings of §5 (best of three, one call each; scipy's distance
           and tree kernels on one thread unless workers=-1, scikit-learn's BLAS at its default)

The C++ program is compiled into build/spatial_probe/ with sd's flags (-O3 -mcpu=native
-ffp-contract=off) under the shared build lock (build/.agent-build-lock). Every child process runs
under a time limit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench import cli  # noqa: E402

BUILD = ROOT / "build" / "spatial_probe"
DATA = ROOT / "tests" / "spatial" / "data"
LOCK = ROOT / "build" / ".agent-build-lock"
FLAGS = [
    "-std=c++20",
    "-O3",
    "-mcpu=native",
    "-ffp-contract=off",
    "-DACCELERATE_NEW_LAPACK",
    "-DACCELERATE_LAPACK_ILP64",
]
SRC = ROOT / "tools" / "spatial_probe_kernels.cpp"
SECTIONS = ("kernels", "ddot", "gemm", "pow", "scipy", "sklearn", "r", "time")
CHILD_SECONDS = 1200
METRICS = ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
           "correlation", "canberra", "braycurtis", "hamming", "jaccard")  # fmt: skip


def P():
    from tests.spatial import points

    return points


class Log:
    def __init__(self):
        self.path = cli.log_file("spatial_probe")
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------- the C++ program


def build() -> pathlib.Path:
    exe = BUILD / "kernels"
    deps = [SRC, ROOT / "csrc" / "spatial" / "pair_kernel.hpp", ROOT / "csrc" / "base" / "pairwise.hpp"]
    if exe.exists() and exe.stat().st_mtime >= max(d.stat().st_mtime for d in deps):
        return exe
    BUILD.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    try:
        subprocess.run(["clang++", *FLAGS, f"-I{ROOT / 'csrc'}", "-o", str(exe), str(SRC), "-framework", "Accelerate"],
                       check=True, timeout=600)  # fmt: skip
    finally:
        LOCK.rmdir()
    return exe


def run(cmd, env_extra=None) -> str:
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=CHILD_SECONDS)
    out = p.stdout
    if p.returncode:
        out += f"(exit status {p.returncode})\n{p.stderr[-2000:]}"
    return out


def load(arg):
    return f"load average {os.getloadavg()[0]:.1f} {arg}"


def kernels(log):
    exe = build()
    with tempfile.TemporaryDirectory() as d:
        log(f"## kernels ({load('at start')})")
        log(run([exe, "kernels", d]))
        log(vector_norm_bits(pathlib.Path(d)))


def vector_norm_bits(d: pathlib.Path) -> str:
    """NormOrder's dumped distances against sd.linalg.vector_norm(x - y), bit for bit; TileOrder's
    against the same, as a count."""
    import stardust as sd

    lines = ["NormOrder and TileOrder against sd.linalg.vector_norm(x - y) (16 x 16 pairs per n):"]
    for f in sorted(d.glob("*.norm"), key=lambda p: int(p.stem[1:])):
        n = int(f.stem[1:])
        xa = np.fromfile(d / f"n{n}.xa").reshape(16, n)
        xb = np.fromfile(d / f"n{n}.xb").reshape(16, n)
        dn = np.fromfile(f).reshape(16, 16)
        dt = np.fromfile(d / f"n{n}.tile").reshape(16, 16)
        ref = np.array([[float(sd.linalg.vector_norm(sd.asarray(xa[i]) - sd.asarray(xb[j]))) for j in range(16)]
                        for i in range(16)])  # fmt: skip
        same_n = int((dn.view(np.int64) == ref.view(np.int64)).sum())
        same_t = int((dt.view(np.int64) == ref.view(np.int64)).sum())
        lines.append(f"  n = {n:5d}: NormOrder {same_n}/256 equal, TileOrder {same_t}/256 equal")
    return "\n".join(lines)


def ddot(log):
    log(f"## ddot ({load('')})")
    log(run([build(), "ddot"]))


def gemm(log):
    exe = build()
    for mode in ({"VECLIB_MAXIMUM_THREADS": "1"}, {}):
        log(f"## gemm ({load('')})")
        log(run([exe, "gemm"], mode))


# ---------------------------------------------------------------- pow


def pow_error(log):
    from arbref import rigorous, to_dd
    from flint import arb

    g = P().rng("pow")
    worst = {}
    for p in (0.5, 1.5, 3.0, 7.5, 2000.0, None):
        errs = []
        for _ in range(4000):
            x = float(10 ** g.uniform(-3, 3)) if p != 2000.0 else float(g.uniform(0.7, 1.4))
            q = float(g.uniform(0.25, 12)) if p is None else p
            try:
                got = math.pow(x, q)
            except OverflowError:
                continue
            if not 0 < got < math.inf:
                continue
            hi, lo = to_dd(rigorous(lambda x=x, q=q: arb(x) ** arb(q)))
            errs.append(abs((got - hi) - lo) / math.ulp(hi))
        worst["random p" if p is None else f"p = {p:g}"] = (max(errs), float(np.mean(errs)), len(errs))
    log("## pow: math.pow (the C library's pow) against Arb, in ULP of the true value")
    for k, (mx, mean, n) in worst.items():
        log(f"  {k:10s}: max {mx:.3f}, mean {mean:.3f} over {n} points")


# ---------------------------------------------------------------- records


def outcome(fn):
    """{"value": ...} or {"error": type, "msg": message} of fn()."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = fn()
    except Exception as e:  # noqa: BLE001 -- recording what the library raises
        return {"error": type(e).__name__, "msg": str(e)[:300]}
    return {"value": plain(v)}


def plain(v):
    if isinstance(v, np.ndarray):
        return {"array": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return {"np_int": int(v)}
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, set):
        return {"set": sorted([list(map(int, t)) for t in v])}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def write_records(name, meta, records):
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in records.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {(DATA / name).relative_to(ROOT)}"


def metric_kwargs(metric, n, k=0):
    M = P()
    if metric == "minkowski":
        return {"p": 3.0}
    if metric == "seuclidean":
        return {"V": M.variances(n)}
    if metric == "mahalanobis":
        return {"VI": M.inverse_covariances(n)}
    return {}


def scipy_records(log):
    import scipy
    import scipy.spatial as S
    import scipy.spatial.distance as D

    M = P()
    rec = {}
    for eid, u, v in M.edge_pairs():
        n = len(u)
        dig = M.digest(u, v)
        for metric in METRICS:
            kw = metric_kwargs(metric, n)
            fn = getattr(D, metric)
            if metric in ("seuclidean", "mahalanobis"):
                arg = kw["V"] if metric == "seuclidean" else kw["VI"]
                r = outcome(lambda fn=fn, arg=arg: fn(u, v, arg))
                rp = outcome(lambda metric=metric, kw=kw: D.pdist(np.vstack([u, v]), metric, **kw)[0])
            else:
                r = outcome(lambda fn=fn, kw=kw: fn(u, v, **kw))
                rp = outcome(lambda metric=metric, kw=kw: D.pdist(np.vstack([u, v]), metric, **kw)[0])
            rec[f"distance/{metric}/{eid}"] = {"digest": dig, **r}
            rec[f"pdist/{metric}/{eid}"] = {"digest": dig, **rp}
            if metric not in ("seuclidean", "mahalanobis"):
                w = M.weights(n, "zeros")
                rw = outcome(lambda metric=metric, kw=kw, w=w: D.pdist(np.vstack([u, v]), metric, w=w, **kw)[0])
                rec[f"pdist_w/{metric}/{eid}"] = {"digest": dig, **rw}
    # the documented edges
    rec["pdist/minkowski_p-1"] = outcome(lambda: D.pdist(np.array([[0.0, 0.0], [3.0, 4.0]]), "minkowski", p=-1.0)[0])
    rec["pdist/minkowski_p0"] = outcome(lambda: D.pdist(np.array([[0.0, 0.0], [3.0, 4.0]]), "minkowski", p=0.0)[0])
    rec["minkowski/p-1"] = outcome(lambda: D.minkowski([0.0, 0.0], [3.0, 4.0], p=-1.0))
    rec["minkowski/p2000"] = outcome(lambda: D.minkowski([0.0, 0.0], [3.0, 4.0], p=2000.0))
    rec["euclidean/1e200"] = outcome(lambda: D.euclidean([1e200, 0.0], [-1e200, 0.0]))
    rec["cdist/chebyshev_nan"] = outcome(lambda: D.cdist([[math.nan, 0.0]], [[1.0, 5.0]], "chebyshev")[0, 0])
    rec["canberra/nan_function"] = outcome(lambda: D.canberra([math.nan, 1.0], [1.0, 2.0]))
    rec["canberra/nan_pdist"] = outcome(lambda: D.pdist([[math.nan, 1.0], [1.0, 2.0]], "canberra")[0])
    rec["canberra/nan_pdist_w"] = outcome(lambda: D.pdist([[math.nan, 1.0], [1.0, 2.0]], "canberra", w=[1.0, 1.0])[0])
    rec["pdist/complex"] = outcome(lambda: D.pdist(np.array([[1 + 1j, 0], [0, 1]])))
    rec["pdist/positional_metric"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "cityblock")[0])
    rec["pdist/unused_p"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "euclidean", p=3))
    rec["pdist/negative_V"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "seuclidean", V=[-1.0, 1.0])[0])
    rec["pdist/zero_V"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "seuclidean", V=[0.0, 1.0])[0])
    rec["pdist/nan_w"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "euclidean", w=[math.nan, 1.0])[0])
    rec["pdist/inf_w"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "euclidean", w=[math.inf, 1.0])[0])
    rec["pdist/negative_w"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "euclidean", w=[-1.0, 1.0])[0])
    rec["pdist/alias_e"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "e")[0])
    rec["pdist/upper"] = outcome(lambda: D.pdist([[0.0, 0.0], [1.0, 1.0]], "Euclidean")[0])
    rec["pdist/n0"] = outcome(lambda: D.pdist(np.zeros((3, 0)), "euclidean"))
    rec["pdist/n0_cosine"] = outcome(lambda: D.pdist(np.zeros((3, 0)), "cosine"))
    rec["pdist/one_row"] = outcome(lambda: D.pdist(np.zeros((1, 3))))
    rec["hamming/int_2^60"] = outcome(lambda: D.hamming(np.array([2**60], np.int64), np.array([2**60 + 1], np.int64)))
    rec["cosine/batch"] = outcome(lambda: D.cosine(np.ones((2, 3)), np.ones((2, 3))))
    rec["euclidean/batch"] = outcome(lambda: D.euclidean(np.ones((2, 3)), np.zeros((2, 3))))
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    nan_m = np.array([[0.0, math.nan], [math.nan, 0.0]])
    rec["squareform/nan"] = outcome(lambda: D.squareform(nan_m))
    rec["squareform/asym"] = outcome(lambda: D.squareform(np.array([[0.0, 1.0], [2.0, 0.0]])))
    rec["squareform/diag"] = outcome(lambda: D.squareform(np.array([[1.0, 1.0], [1.0, 0.0]])))
    rec["squareform/empty"] = outcome(lambda: D.squareform(np.zeros(0)))
    rec["squareform/1x1"] = outcome(lambda: D.squareform(np.zeros((1, 1))))
    rec["squareform/0x0"] = outcome(lambda: D.squareform(np.zeros((0, 0))))
    rec["squareform/bool"] = outcome(lambda: D.squareform(np.array([True, False, True])))
    rec["squareform/int"] = outcome(lambda: D.squareform(np.array([1, 2, 3], np.int64)))
    rec["squareform/bad_length"] = outcome(lambda: D.squareform(np.zeros(4)))
    rec["squareform/3d"] = outcome(lambda: D.squareform(np.zeros((2, 2, 2))))
    rec["squareform/tovector"] = outcome(lambda: D.squareform(a, force="tovector"))
    # summation orders
    g = M.rng("scipy-order")
    seq_same = kd_diff = total = 0
    kd_by_n = {}
    for n in (2, 3, 8, 64, 512, 3000):
        diff_n = 0
        for _ in range(130):
            x, y = g.standard_normal(n), g.standard_normal(n)
            d = D.cdist(x[None], y[None])[0, 0]
            s = 0.0
            for a_, b_ in zip(x.tolist(), y.tolist()):
                z = a_ - b_
                s = s + z * z  # product and sum rounded apart, in order
            seq_same += d == math.sqrt(s)
            dk = S.KDTree(y[None]).query(x, k=1)[0]
            diff_n += dk != d
            total += 1
        kd_by_n[n] = diff_n
        kd_diff += diff_n
    rec["order/cdist_sequential"] = {"pairs": total, "same": int(seq_same)}
    rec["order/kdtree_vs_cdist"] = {"pairs_per_n": 130, "differ": {str(k): int(v) for k, v in kd_by_n.items()}}
    # KDTree edges
    data = M.get("nn/normal3").x[:50]
    T = S.KDTree(data)
    x0 = np.zeros(3)
    rec["kdtree/k_true"] = outcome(lambda: T.query(x0, k=True))
    rec["kdtree/k_float"] = outcome(lambda: T.query(x0, k=2.0))
    rec["kdtree/k_zero"] = outcome(lambda: T.query(x0, k=0))
    rec["kdtree/k_list_zero"] = child_outcome(
        "import numpy as np, scipy.spatial as S; "
        "print(S.KDTree(np.random.default_rng(0).random((50, 3))).query(np.zeros(3), k=[0]))"
    )
    rec["kdtree/r_negative_same_as_1"] = outcome(
        lambda: sorted(T.query_ball_point(x0, -1.0)) == sorted(T.query_ball_point(x0, 1.0))
    )
    rec["kdtree/r_nan"] = outcome(lambda: T.query_ball_point(x0, math.nan))
    rec["kdtree/dub_negative_same_as_1"] = outcome(
        lambda: (
            [a.tolist() for a in T.query(x0, k=3, distance_upper_bound=-1.0)]
            == [a.tolist() for a in T.query(x0, k=3, distance_upper_bound=1.0)]
        )
    )
    rec["kdtree/dub_nan"] = outcome(lambda: T.query(x0, k=3, distance_upper_bound=math.nan))
    rec["kdtree/k_gt_n"] = outcome(lambda: S.KDTree(data[:2]).query(x0, k=4))
    rec["kdtree/leafsize_float"] = outcome(lambda: S.KDTree(data, leafsize=2.5).leafsize)
    rec["kdtree/m0_query"] = outcome(lambda: S.KDTree(np.zeros((5, 0))).query(np.zeros(0)))
    rec["kdtree/n0_query"] = outcome(lambda: S.KDTree(np.zeros((0, 3))).query(x0, k=2))
    rec["kdtree/complex"] = outcome(lambda: S.KDTree(np.ones((3, 2), complex)))
    rec["kdtree/nan_data"] = outcome(lambda: S.KDTree(np.array([[math.nan, 0.0], [1.0, 1.0]])))
    rec["kdtree/1d_data"] = outcome(lambda: S.KDTree(np.zeros(3)))
    rec["kdtree/ball_point_one"] = outcome(lambda: type(T.query_ball_point(x0, 0.5)).__name__)
    rec["kdtree/ball_point_many"] = outcome(lambda: type(T.query_ball_point(np.zeros((2, 3)), 0.5)).__name__)
    rec["kdtree/ball_point_one_sorted"] = outcome(lambda: (lambda r: r == sorted(r))(T.query_ball_point(x0, 1.5)))
    rec["kdtree/query_pairs_type"] = outcome(lambda: type(T.query_pairs(0.3)).__name__)
    rec["kdtree/query_one_types"] = outcome(lambda: [type(v).__name__ for v in T.query(x0)])
    same = np.zeros((2, 3))
    dok = lambda: S.KDTree(same).sparse_distance_matrix(S.KDTree(same), 1.0, output_type="dok_matrix")  # noqa: E731
    rec["kdtree/sdm_zero_dok"] = outcome(lambda: [dok().nnz, int(dok().count_nonzero())])
    rec["kdtree/sdm_zero_coo"] = outcome(
        lambda: S.KDTree(same).sparse_distance_matrix(S.KDTree(same), 1.0, output_type="coo_matrix").nnz
    )
    six = np.ones((6, 2))
    rec["kdtree/tie_order"] = outcome(lambda: S.KDTree(six).query(np.ones(2), k=6)[1])
    rec["kdtree/count_self"] = outcome(lambda: T.count_neighbors(T, 0.5))
    # the boundary: sqrt(d^2) <= r against d^2 <= r^2 at r = the computed distance
    mismatch = trials = 0
    for _ in range(2000):
        x, y = g.standard_normal(3), g.standard_normal(3)
        s = float(((x - y) ** 2).sum())
        r = math.sqrt(s)
        trials += 1
        mismatch += (math.sqrt(s) <= r) != (s <= r * r)
    rec["kdtree/boundary_square_vs_root"] = {"trials": trials, "differ": int(mismatch)}
    # k nearest neighbours of the neighbour sets (for the set comparisons of §7.3)
    for sid in M.ids("neighbors"):
        s = M.get(sid)
        k = min(5, s.m)
        dist, idx = S.KDTree(s.x).query(s.queries, k=k)
        rec[f"knn/{sid}"] = {"digest": M.set_digest(s), "k": k, "indices": idx.tolist(), "distances": dist.tolist()}
    meta = {"tool": "tools/spatial_probe.py", "scipy": scipy.__version__, "numpy": np.__version__}
    log(write_records("scipy.json", meta, rec))


def child_outcome(code):
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    if p.returncode < 0 or p.returncode > 128:
        return {"crash": p.returncode}
    if p.returncode:
        last = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""
        return {"error": last.split(":")[0], "msg": last[:300]}
    return {"stdout": p.stdout[:300]}


def sklearn_records(log):
    import sklearn
    from sklearn.metrics.pairwise import euclidean_distances, haversine_distances, pairwise_distances_argmin_min
    from sklearn.neighbors import KDTree, NearestNeighbors

    M = P()
    rec = {}
    rec["euclidean_distances/1e8"] = outcome(lambda: euclidean_distances([[1e8, 1e8], [1e8 + 1, 1e8]])[0, 1])
    g = M.rng("sklearn-argmin")
    A = 1e4 + g.uniform(0, 1e-3, (5, 3))
    i, d = pairwise_distances_argmin_min(A, A[::-1])
    rec["argmin_min/1e4"] = {"A": A.tolist(), "distances": d.tolist(), "indices": i.tolist()}
    six = np.ones((6, 2))
    rec["nn/tie_order"] = outcome(lambda: NearestNeighbors(n_neighbors=6).fit(six).kneighbors(np.ones((1, 2)))[1][0])
    rec["nn/tie_order_brute"] = outcome(
        lambda: NearestNeighbors(n_neighbors=6, algorithm="brute").fit(six).kneighbors(np.ones((1, 2)))[1][0]
    )
    rec["nn/k_gt_n"] = outcome(lambda: NearestNeighbors(n_neighbors=4).fit(six[:2]).kneighbors(np.ones((1, 2))))
    rec["nn/nan"] = outcome(lambda: NearestNeighbors().fit(np.array([[math.nan, 0.0], [1.0, 1.0]])))
    rec["kdtree/tie_order"] = outcome(lambda: KDTree(six).query(np.ones((1, 2)), k=6)[1][0])
    for hid, u, v in M.haversine_pairs():
        rec[f"haversine/{hid}"] = {
            "digest": M.digest(u, v),
            **outcome(lambda u=u, v=v: haversine_distances([u], [v])[0, 0]),
        }
    s = M.get("nn/normal3")
    i, d = pairwise_distances_argmin_min(s.queries, s.x)
    rec["argmin_min/nn/normal3"] = {"digest": M.set_digest(s), "indices": i.tolist(), "distances": d.tolist()}
    s = M.get("far/1e8")
    i, d = pairwise_distances_argmin_min(s.queries, s.x)
    rec["argmin_min/far/1e8"] = {"digest": M.set_digest(s), "indices": i.tolist(), "distances": d.tolist()}
    meta = {"tool": "tools/spatial_probe.py", "sklearn": sklearn.__version__, "numpy": np.__version__}
    log(write_records("sklearn.json", meta, rec))


R_SETS = ("lattice/int3d", "normal/8", "uniform/3", "binary/bool64")
R_METHODS = ("euclidean", "manhattan", "maximum", "canberra", "binary", "minkowski")


def r_records(log):
    M = P()
    rec = {}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        script = ["out <- list()"]
        for k, sid in enumerate(R_SETS):
            x = M.get(sid).x[:20].astype(np.float64)
            np.savetxt(d / f"x{k}.txt", x, fmt="%.17g")
            script.append(f'x <- as.matrix(read.table("{d / f"x{k}.txt"}"))')
            for meth in R_METHODS:
                p = ", p = 3" if meth == "minkowski" else ""
                script.append(f'out[["{sid}|{meth}"]] <- sprintf("%a", as.vector(dist(x, method = "{meth}"{p})))')
        script.append(
            'x <- rbind(c(0, 1, 0, 2), c(0, 3, 0, -2)); out[["edge|canberra_0/0"]] <- '
            'sprintf("%a", as.vector(dist(x, "canberra")))'
        )
        script.append(
            'x <- rbind(c(NA, 1, 2), c(1, 2, 3)); out[["edge|euclidean_NA"]] <- sprintf("%a", as.vector(dist(x)))'
        )
        script.append(
            'x <- rbind(c(1e200, 0), c(-1e200, 0)); out[["edge|euclidean_1e200"]] <- sprintf("%a", as.vector(dist(x)))'
        )
        script.append('for (k in names(out)) cat(k, "\\t", paste(out[[k]], collapse = " "), "\\n", sep = "")')
        (d / "run.R").write_text("\n".join(script))
        p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))  # fmt: skip
        if p.returncode:
            log(f"R failed: {p.stderr[-2000:]}")
            return
        ver = subprocess.run(["Rscript", "-e", "cat(R.version.string)"], capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        key, _, vals = line.partition("\t")
        sid, _, meth = key.partition("|")
        values = [float.fromhex(v) if v not in ("NA", "NaN") else math.nan for v in vals.split()]
        entry = {"values": values, "rows": 20 if sid != "edge" else 2}
        if sid != "edge":
            entry["digest"] = M.set_digest(M.get(sid))
        rec[f"dist/{sid}/{meth}"] = entry
    log(write_records("r.json", {"tool": "tools/spatial_probe.py", "R": ver.stdout.strip()}, rec))


# ---------------------------------------------------------------- timings


def best(fn, reps=3):
    out = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t0)
    return out * 1e3


def timings(log):
    import scipy.spatial as S
    import scipy.spatial.distance as D
    from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min
    from sklearn.neighbors import KDTree as SKDTree

    g = np.random.Generator(np.random.PCG64(20261122))
    log(f"## time ({load('at start')}); ms, best of 3")
    rows = []

    def row(what, scipy_ms, sklearn_ms, note=""):
        rows.append((what, scipy_ms, sklearn_ms, note))
        f = lambda v: "-" if v is None else (f"{v:.3g}" if v < 100 else f"{v:.0f}")  # noqa: E731
        log(f"  {what:58s} {f(scipy_ms):>8s} {f(sklearn_ms):>8s}  {note}")

    for m, n in ((1000, 3), (1000, 64), (4000, 3), (4000, 64), (4000, 512)):
        a, b = g.standard_normal((m, n)), g.standard_normal((m, n))
        note = ""
        if (m, n) == (4000, 512):
            note = f"cityblock scipy {best(lambda: D.cdist(a, b, 'cityblock')):.0f}"
        row(
            f"cdist euclidean {m} x {m}, n = {n}",
            best(lambda: D.cdist(a, b)),
            best(lambda: euclidean_distances(a, b)),
            note,
        )
    for n in (3, 64, 512):
        a, b = g.standard_normal((4000, n)), g.standard_normal((4000, n))
        row(
            f"pairwise_distances_argmin_min 4000 x 4000, n = {n}",
            None,
            best(lambda: pairwise_distances_argmin_min(a, b)),
        )
    for n in (3, 64):
        x = g.standard_normal((2000, n))
        row(f"pdist euclidean m = 2000, n = {n}", best(lambda: D.pdist(x)), None,
            f"cosine {best(lambda: D.pdist(x, 'cosine')):.3g}")  # fmt: skip
    for npts in (10**5, 10**6):
        x = g.random((npts, 3))
        row(f"KDTree build n = {npts:.0e}, m = 3", best(lambda: S.KDTree(x)), best(lambda: SKDTree(x)))
    x = g.random((10**6, 3))
    q = g.random((10**4, 3))
    T, ST = S.KDTree(x), SKDTree(x)
    for k in (1, 10):
        note = f"workers=-1 {best(lambda: T.query(q, k=k, workers=-1)):.3g}"
        row(f"query 1e4 points, n = 1e6, m = 3, k = {k}", best(lambda: T.query(q, k=k)),
            best(lambda: ST.query(q, k=k)), note)  # fmt: skip
    r = (10 / 10**6 * 3 / (4 * math.pi)) ** (1 / 3)
    counts = T.query_ball_point(q, r, return_length=True)
    row(
        f"query_ball_point 1e4 points (mean {counts.mean():.1f}), n = 1e6, m = 3",
        best(lambda: T.query_ball_point(q, r)),
        None,
    )
    x = g.random((10**5, 16))
    q = g.random((10**4, 16))
    T, ST = S.KDTree(x), SKDTree(x)
    for k in (1, 10):
        note = f"workers=-1 {best(lambda: T.query(q, k=k, workers=-1), reps=1):.0f}"
        row(f"query 1e4 points, n = 1e5, m = 16, k = {k}", best(lambda: T.query(q, k=k), reps=1),
            best(lambda: ST.query(q, k=k), reps=1), note + " (one run each)")  # fmt: skip
    log(f"({load('at end')})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools/spatial_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {"kernels": kernels, "ddot": ddot, "gemm": gemm, "pow": pow_error, "scipy": scipy_records,
             "sklearn": sklearn_records, "r": r_records, "time": timings}  # fmt: skip
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
