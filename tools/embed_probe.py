"""Probes of the libraries sd.embed is compared with (docs/design/embed.md §1.2, §5, §7.4). The records the tests
and later milestones read go to tests/embed/data/; the comparators' timings of §5 go to tests/embed/data/times.json
(and the terminal and build/logs/embed_probe/<stamp>.log), and docs/design/embed.md §5 quotes them.

    .venv/bin/python tools/embed_probe.py                      # every record section (not time)
    .venv/bin/python tools/embed_probe.py sklearn r            # only these sections
    .venv/bin/python tools/embed_probe.py time                 # the timings, 1 and 10 threads
    .venv/bin/python tools/embed_probe.py time --match tsne --threads 10

Sections (positional; default every record section):
  sklearn      scikit-learn 1.9.1's decomposition, manifold and random_projection: defaults, sign conventions,
               small exact-ish outputs, ties, degenerate inputs, disconnected graphs, warnings, and the thread
               dependence of TSNE's Barnes-Hut -> sklearn.json
  scipy        scipy 1.18.1's svd, svds and eigsh: defaults, order and signs, repeatability -> scipy.json
  statsmodels  statsmodels 0.15.0's PCA, Factor and factor_rotation -> statsmodels.json
  r            R 4.6.2's stats (prcomp, princomp, cmdscale, factanal, varimax, promax), MASS (sammon, isoMDS),
               irlba, RSpectra, uwot, Rtsne and fastICA through bench/accuracy/rsession.py -> r.json
  umap         umap-learn 0.6.0: defaults and warnings, find_ab_params, smooth_knn_dist, fuzzy_simplicial_set,
               degenerate and disconnected inputs, run-to-run and thread dependence -> umap.json
  opentsne     openTSNE 1.0.4: defaults, the affinities P, fits from a given start, thread dependence
               -> opentsne.json
  time         the comparators of bench/speed/cases/embed.py at their §5 sizes, 1 and 10 threads -> times.json

Every record names the test data it used (tests/embed/samples.py) by id and digest. Records named rule/... are the
behaviours the design departs from or relies on (§2.8). The timings come from one run on a machine shared with other
work; each one notes the load average.

Threads in the time section: each thread count runs in a child process of its own, since Accelerate (numpy's and
scipy's BLAS) reads VECLIB_MAXIMUM_THREADS once per process: at 1 thread VECLIB_MAXIMUM_THREADS=1 and every OpenMP
pool limited to 1 (threadpoolctl, after the comparators are imported: cluster CL103); at 10 threads Accelerate's
default and the OpenMP pools limited to 10. umap-learn and openTSNE take their thread count as n_jobs (the
comparator's label names it), R runs on one thread (bench/speed/rsession.R_ENV) except where uwot's n_threads says
otherwise. A call of 3 s or more is timed once (after a warm-up of the library on a small input); faster calls take
the best batch of bench/speed/harness.measure.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
# tools/ (the script's folder) holds coverage.py, which would shadow the coverage package numba imports
sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != ROOT / "tools"]
sys.path.insert(0, str(ROOT))

from bench import cli  # noqa: E402

DATA = ROOT / "tests" / "embed" / "data"
SECTIONS = ("sklearn", "scipy", "statsmodels", "r", "umap", "opentsne", "time")
RECORDS = SECTIONS[:-1]
TOOL = "tools/embed_probe.py"
MACHINE = "Apple M4 Pro (14 cores), macOS 26.6"
DEGENERATE = ("tiny/1", "tiny/2", "tiny/3", "allsame/8", "fewdistinct/12")


def S():
    from tests.embed import samples

    return samples


class Log:
    def __init__(self, name="embed_probe"):
        self.path = cli.log_file(name)
        print(f"log: {self.path.relative_to(ROOT)}", flush=True)

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------- records


def plain(v):
    """A JSON value: arrays as {"array", "dtype", "shape"} (full doubles), non-finite floats as their repr."""
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "c":
            return {"real": plain(v.real.copy()), "imag": plain(v.imag.copy())}
        if v.dtype.kind == "f":
            vals = [plain(float(x)) for x in v.ravel()] if not np.isfinite(v).all() else None
            if vals is not None:
                return {"array": vals, "dtype": str(v.dtype), "shape": list(v.shape)}
        return {"array": v.tolist(), "dtype": str(v.dtype), "shape": list(v.shape)}
    if hasattr(v, "tocsr") and hasattr(v, "nnz"):  # a scipy sparse matrix or array
        c = v.tocsr()
        c.sort_indices()
        return {
            "csr": {"data": plain(c.data), "indices": c.indices.tolist(), "indptr": c.indptr.tolist()},
            "shape": list(c.shape),
        }
    if isinstance(v, (complex, np.complexfloating)):
        return {"real": plain(float(v.real)), "imag": plain(float(v.imag))}
    if isinstance(v, np.floating):
        return plain(float(v))
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def outcome(fn):
    """{"value": ...} or {"error": type, "msg": message}, and the warnings fn() raised (category: message)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            v = fn()
        except Exception as e:  # noqa: BLE001 -- recording what the library raises
            out = {"error": type(e).__name__, "msg": str(e)[:400]}
        else:
            out = {"value": plain(v)}
    if w:
        out["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:400]}" for x in w})
    return out


def on(sid, fn):
    """outcome(fn(points)) with the sample's id and digest."""
    s = S().points(sid)
    return {"sample": sid, "digest": s.digest, **outcome(lambda: fn(s))}


def isolated(sid, code):
    """outcome() of a call that may crash the interpreter (scikit-learn's Barnes-Hut quad tree recurses without end
    on coincident points): `code` is an expression over `s` (the sample) run in a child process; a crash is
    recorded as {"crash": "signal N"}."""
    s = S().points(sid)
    prog = (
        "import json, sys, warnings; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tools']\n"
        "import numpy as np\nimport embed_probe as P\n"
        "from sklearn import manifold as M\n"
        f"s = P.S().points({sid!r})\n"
        f"print('@@out@@' + json.dumps(P.outcome(lambda: {code})))\n"
    )
    r = subprocess.run([sys.executable, "-c", prog, str(ROOT)], capture_output=True, text=True, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith("@@out@@"):
            return {"sample": sid, "digest": s.digest, **json.loads(line[7:])}
    return {"sample": sid, "digest": s.digest, "crash": f"signal {-r.returncode}" if r.returncode < 0 else
            f"exit {r.returncode}", "call": code}  # fmt: skip


def defaults(obj):
    """The parameters of a callable or class and their defaults (repr when not plain)."""
    out = {}
    for name, p in inspect.signature(obj).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or name == "self":
            continue
        d = p.default
        if d is inspect.Parameter.empty:
            out[name] = "<required>"
        elif isinstance(d, (bool, int, float, str)) or d is None:
            out[name] = plain(d)
        else:
            out[name] = repr(d)
    return out


def write_records(name, meta, records):
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(plain(v))}" for k, v in records.items()]
    (DATA / name).write_text("{\n" + ",\n".join(lines) + "\n}\n")
    return f"{len(records)} records -> {(DATA / name).relative_to(ROOT)}"


def maxabs_positive(M, axis=1):
    """For each row (axis=1) or column (axis=0) of M: is its entry of largest magnitude positive (decision EM14;
    scikit-learn's svd_flip(u_based_decision=False) on the rows of V^T)?"""
    M = np.asarray(M, float)
    if axis == 0:
        M = M.T
    i = np.argmax(np.abs(M), axis=1)
    return [bool(M[r, i[r]] > 0) for r in range(M.shape[0])]


def same_up_to_perm(A, B, perm):
    """B[j] = A[perm[j]] row by row, bit for bit, and the largest difference."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    D = np.abs(A[perm] - B)
    return {"bitwise": bool(np.array_equal(A[perm], B)), "max_diff": float(D.max()) if D.size else 0.0}


def lattice_perm():
    """perm with lattice/5x5-shuffled[j] = lattice/5x5[perm[j]]."""
    a, b = S().points("lattice/5x5").X, S().points("lattice/5x5-shuffled").X
    return np.array([int(np.nonzero((a == row).all(axis=1))[0][0]) for row in b])


def knn_exact(X, k, self_first=True):
    """Exact k nearest neighbours by brute force (squared distances, then (distance, index) order), the point itself
    first when self_first (umap-learn's convention)."""
    X = np.asarray(X, float)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    n = len(X)
    idx = np.empty((n, k), np.int64)
    dist = np.empty((n, k))
    for i in range(n):
        order = np.lexsort((np.arange(n), D[i]))
        if self_first:
            order = np.concatenate([[i], order[order != i]])
        idx[i] = order[:k]
        dist[i] = np.sqrt(D[i, order[:k]])
    return idx, dist


def thread_compare(runs):
    """Bitwise and largest-difference comparison of every pair of named runs' arrays."""
    names = list(runs)
    out = {}
    for i, a_name in enumerate(names):
        a = np.asarray(runs[a_name], float)
        for b_name in names[i + 1 :]:
            b = np.asarray(runs[b_name], float)
            out[f"{a_name} vs {b_name}"] = {
                "bitwise": bool(np.array_equal(a, b)),
                "max_abs_diff": float(np.abs(a - b).max()) if a.shape == b.shape else None,
            }
    return out


# ---------------------------------------------------------------- scikit-learn


def sklearn_records(log):
    import sklearn
    import threadpoolctl
    from sklearn import decomposition as D
    from sklearn import manifold as M
    from sklearn import random_projection as RP
    from sklearn.manifold import _t_sne
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits

    rec = {}
    classes = (D.PCA, D.TruncatedSVD, D.KernelPCA, D.FactorAnalysis, D.FastICA, D.NMF, RP.GaussianRandomProjection,
               RP.SparseRandomProjection, M.ClassicalMDS, M.MDS, M.Isomap, M.LocallyLinearEmbedding,
               M.SpectralEmbedding, M.TSNE)  # fmt: skip
    funcs = (RP.johnson_lindenstrauss_min_dim, M.trustworthiness, D.fastica, D.non_negative_factorization, M.smacof,
             M.spectral_embedding, M.locally_linear_embedding)  # fmt: skip
    rec["defaults"] = {c.__name__: defaults(c) for c in classes + funcs}

    # -- PCA: the four solvers, their signs (svd_flip on V^T's rows), the outputs of 'full'
    for sid in ("lowrank/r3p10", "equalvar/p4", "wide/n10p40", "shifted/p6", "constcol/p6", "tinygap/p5"):
        s = S().points(sid)
        for solver in ("full", "covariance_eigh", "arpack", "randomized"):
            if solver == "arpack" and s.d >= min(s.n, s.p):
                continue

            def fit(s=s, solver=solver):
                pca = D.PCA(s.d, svd_solver=solver, random_state=0)
                Z = pca.fit_transform(s.X)
                out = {
                    "components": pca.components_,
                    "explained_variance": pca.explained_variance_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "singular_values": pca.singular_values_,
                    "mean": pca.mean_,
                    "noise_variance": pca.noise_variance_,
                    "components_maxabs_positive": maxabs_positive(pca.components_),
                    "scores_maxabs_positive": maxabs_positive(Z, axis=0),
                }
                if solver == "full":
                    out["embedding"] = Z
                    out["transform_Xe"] = pca.transform(s.Xe)
                    out["score_samples_Xe"] = pca.score_samples(s.Xe)
                return out

            rec[f"pca/{sid}/{solver}"] = on(sid, lambda s, fit=fit: fit())
    for sid in ("lowrank/r3p10", "lowrank/r5p30", "factor/p8k2", "equalvar/p4"):
        rec[f"pca_n_components/{sid}"] = on(
            sid,
            lambda s: {
                "mle_full": D.PCA("mle", svd_solver="full").fit(s.X).n_components_,
                "mle_covariance_eigh": D.PCA("mle", svd_solver="covariance_eigh").fit(s.X).n_components_,
                "0.9": D.PCA(0.9).fit(s.X).n_components_,
                "0.99": D.PCA(0.99).fit(s.X).n_components_,
                "None": D.PCA().fit(s.X).n_components_,
            },
        )
    rec["rule/pca_auto_solver"] = {
        "data": "numpy default_rng(1).standard_normal((n, p)), PCA(10)",
        **{
            f"{n}x{p}": D.PCA(10).fit(np.random.default_rng(1).standard_normal((n, p)))._fit_svd_solver
            for n, p in ((1000, 50), (1000, 200), (200, 1000), (5000, 600), (600, 600), (20000, 1200))
        },
    }
    rec["rule/pca_mle_arpack"] = on("lowrank/r3p10", lambda s: D.PCA("mle", svd_solver="arpack").fit(s.X))
    # -- TruncatedSVD
    for sid in ("lowrank/r3p10", "sparse/n100p50", "nonneg/r4"):
        s = S().points(sid)
        for alg in ("arpack", "randomized"):

            def fit(s=s, alg=alg):
                t = D.TruncatedSVD(s.d, algorithm=alg, random_state=0)
                Z = t.fit_transform(s.X)
                return {
                    "components": t.components_,
                    "singular_values": t.singular_values_,
                    "explained_variance": t.explained_variance_,
                    "explained_variance_ratio": t.explained_variance_ratio_,
                    "components_maxabs_positive": maxabs_positive(t.components_),
                    "scores_maxabs_positive": maxabs_positive(Z, axis=0),
                }

            rec[f"truncated_svd/{sid}/{alg}"] = on(sid, lambda s, fit=fit: fit())
    # -- random projection
    rec["johnson_lindenstrauss_min_dim"] = {
        f"n={n} eps={e}": int(RP.johnson_lindenstrauss_min_dim(n, eps=e))
        for n in (10, 100, 10**4, 10**6)
        for e in (0.1, 0.3, 0.5, 0.9)
    }
    rec["rule/random_projection_auto_too_big"] = on(
        "lowrank/r3p10", lambda s: RP.GaussianRandomProjection(random_state=0).fit(s.X).n_components_
    )
    rec["rule/sparse_projection_density_auto"] = on(
        "lowrank/r5p30", lambda s: RP.SparseRandomProjection(5, random_state=0).fit(s.X).density_
    )
    # -- KernelPCA
    for kernel in ("linear", "rbf", "poly", "sigmoid", "cosine"):

        def fit(s, kernel=kernel):
            k = D.KernelPCA(3, kernel=kernel)
            Z = k.fit_transform(s.X)
            return {
                "eigenvalues": k.eigenvalues_,
                "embedding": Z,
                "transform_Xe": k.transform(s.Xe),
                "gamma": k.gamma_,
                "eigenvectors_maxabs_positive": maxabs_positive(k.eigenvectors_, axis=0),
            }

        rec[f"kernel_pca/blobs/k3p10/{kernel}"] = on("blobs/k3p10", fit)
    rec["kernel_pca/lowrank/r3p10/None_remove_zero_eig"] = on(
        "lowrank/r3p10", lambda s: {"n_components": D.KernelPCA().fit(s.X).eigenvalues_.size}
    )
    rec["kernel_pca/blobs/k3p10/rbf_inverse"] = on(
        "blobs/k3p10",
        lambda s: D.KernelPCA(3, kernel="rbf", fit_inverse_transform=True).fit(s.X).inverse_transform(np.zeros((2, 3))),
    )
    # -- FactorAnalysis
    for rot in (None, "varimax", "quartimax"):

        def fa(s, rot=rot):
            f = D.FactorAnalysis(2, rotation=rot).fit(s.X)
            return {
                "components": f.components_,
                "noise_variance": f.noise_variance_,
                "mean": f.mean_,
                "loglike_last": f.loglike_[-1],
                "n_iter": f.n_iter_,
                "transform_first5": f.transform(s.X[:5]),
            }

        rec[f"factor_analysis/factor/p8k2/{rot}"] = on("factor/p8k2", fa)
    rec["rule/factor_analysis_convergence_warning"] = on(
        "factor/p8k2", lambda s: D.FactorAnalysis(2, max_iter=2).fit(s.X).n_iter_
    )
    # -- FastICA: a given start, then the order and signs from different random starts
    s = S().points("ica/s4")
    w0 = S().rng("probe", "w_init").standard_normal((4, 4))
    for alg in ("parallel", "deflation"):
        for fun in ("logcosh", "exp", "cube"):

            def ica(s, alg=alg, fun=fun):
                f = D.FastICA(4, algorithm=alg, fun=fun, w_init=w0, whiten="unit-variance", max_iter=1000)
                Z = f.fit_transform(s.X)
                return {
                    "components": f.components_,
                    "mixing": f.mixing_,
                    "mean": f.mean_,
                    "whitening": f.whitening_,
                    "n_iter": f.n_iter_,
                    "sources_first5": Z[:5],
                }

            rec[f"fastica/ica/s4/{alg}/{fun}"] = on("ica/s4", ica)
    rec["fastica/w_init"] = {"from": "samples.rng('probe', 'w_init').standard_normal((4, 4))", "w_init": w0}

    def ica_orders(s):
        true = np.column_stack([np.sin(2 * s.t), np.sign(np.sin(3 * s.t))])
        rows = []
        for seed in range(6):
            Z = D.FastICA(4, random_state=seed, max_iter=1000).fit_transform(s.X)
            C = np.corrcoef(np.column_stack([Z, true]).T)[:4, 4:]
            kurt = ((Z - Z.mean(0)) ** 4).mean(0) / Z.var(0) ** 2 - 3
            rows.append(
                {
                    "seed": seed,
                    "sine_component": int(np.argmax(np.abs(C[:, 0]))),
                    "sine_sign": int(np.sign(C[np.argmax(np.abs(C[:, 0])), 0])),
                    "square_component": int(np.argmax(np.abs(C[:, 1]))),
                    "excess_kurtosis": kurt,
                }
            )
        return rows

    rec["rule/fastica_order_sign"] = on("ica/s4", ica_orders)
    rec["rule/fastica_convergence_warning"] = on(
        "ica/s4", lambda s: D.FastICA(4, random_state=0, max_iter=1).fit(s.X).n_iter_
    )
    # -- NMF
    s = S().points("nonneg/r4")
    g = S().rng("probe", "nmf_init")
    W0, H0 = g.random((s.n, 4)), g.random((4, s.p))
    for solver in ("cd", "mu"):
        for init in ("nndsvda", "nndsvd", "nndsvdar", "random", "custom"):
            if solver == "mu" and init == "nndsvd":
                continue

            def nmf(s, solver=solver, init=init):
                m = D.NMF(4, init=init, solver=solver, random_state=0, max_iter=1000)
                W = m.fit_transform(s.X, W=W0, H=H0) if init == "custom" else m.fit_transform(s.X)
                out = {"reconstruction_err": m.reconstruction_err_, "n_iter": m.n_iter_, "components": m.components_}
                if init == "custom":
                    out["W"] = W
                return out

            rec[f"nmf/nonneg/r4/{solver}/{init}"] = on("nonneg/r4", nmf)
    rec["nmf/init_custom"] = {"from": "samples.rng('probe', 'nmf_init').random, W0 then H0", "W0": W0, "H0": H0}
    rec["nmf/nonneg/r4/mu/kullback-leibler"] = on(
        "nonneg/r4",
        lambda s: (lambda m: {"err": m.reconstruction_err_, "n_iter": m.n_iter_})(
            D.NMF(4, solver="mu", beta_loss="kullback-leibler", init="nndsvda", max_iter=1000).fit(s.X)
        ),
    )
    rec["rule/nmf_init_none"] = on(
        "nonneg/r4",
        lambda s: {
            "d=4_seeded_same_as_nndsvda": bool(
                np.array_equal(D.NMF(4, max_iter=50, random_state=0).fit(s.X).components_,
                               D.NMF(4, init="nndsvda", max_iter=50, random_state=0).fit(s.X).components_)
            ),
            "nndsvda_unseeded_twice_same (NNDSVD's SVD is _randomized_svd)": bool(
                np.array_equal(D.NMF(4, init="nndsvda", max_iter=50).fit(s.X).components_,
                               D.NMF(4, init="nndsvda", max_iter=50).fit(s.X).components_)
            ),
            "d=30_same_as_random": bool(
                np.array_equal(D.NMF(30, max_iter=50, random_state=0).fit(s.X).components_,
                               D.NMF(30, init="random", max_iter=50, random_state=0).fit(s.X).components_)
            ),
        },
    )  # fmt: skip
    rec["rule/nmf_negative"] = on("lowrank/r3p10", lambda s: D.NMF(2).fit(s.X))
    rec["rule/nmf_convergence_warning"] = on("nonneg/r4", lambda s: D.NMF(4, max_iter=1).fit(s.X).n_iter_)
    # -- classical MDS and SMACOF
    for sid in ("circle/60", "lattice/5x5", "lattice/5x5-shuffled", "swiss/300", "blobs/k3p10"):

        def cmds(s):
            c = M.ClassicalMDS(2).fit(s.X)
            return {
                "embedding": c.embedding_,
                "eigenvalues": c.eigenvalues_,
                "columns_maxabs_positive": maxabs_positive(c.embedding_, axis=0),
            }

        rec[f"classical_mds/{sid}"] = on(sid, cmds)
    for sid in ("circle/60", "blobs/k3p10", "lattice/5x5"):
        for metric_mds in (True, False):

            def smacof(s, metric_mds=metric_mds):
                m = M.MDS(2, metric_mds=metric_mds, init="classical_mds", n_init=1).fit(s.X)
                return {"embedding": m.embedding_, "stress": m.stress_, "n_iter": m.n_iter_}

            rec[f"mds/{sid}/metric_mds={metric_mds}"] = on(sid, smacof)
    rec["rule/mds_init_warning"] = on("blobs/k3p10", lambda s: M.MDS(2, random_state=0).fit(s.X).stress_)
    rec["rule/mds_random_n_init4"] = on(
        "blobs/k3p10", lambda s: M.MDS(2, init="random", n_init=4, random_state=0).fit(s.X).stress_
    )
    rec["rule/mds_classical_n_init4"] = on(
        "blobs/k3p10", lambda s: M.MDS(2, init="classical_mds", n_init=4).fit(s.X).stress_
    )
    rec["rule/mds_normalized_stress_nonmetric"] = on(
        "blobs/k3p10",
        lambda s: {
            "auto": M.MDS(2, metric_mds=False, init="classical_mds").fit(s.X).stress_,
            "False": outcome(
                lambda: M.MDS(2, metric_mds=False, init="classical_mds", normalized_stress=False).fit(s.X).stress_
            ),
        },
    )
    # -- Isomap, LLE, spectral embedding
    perm = lattice_perm()
    for sid in ("swiss/300", "scurve/300", "lattice/5x5", "lattice/5x5-shuffled", "twoblobs/80", "dups/10x4"):
        nn = 4 if "lattice" in sid else 10

        def iso(s, nn=nn):
            m = M.Isomap(n_neighbors=nn).fit(s.X)
            return {
                "embedding": m.embedding_,
                "eigenvalues": m.kernel_pca_.eigenvalues_,
                "reconstruction_error": m.reconstruction_error(),
                "transform_Xe": m.transform(s.Xe),
                "columns_maxabs_positive": maxabs_positive(m.embedding_, axis=0),
            }

        rec[f"isomap/{sid}/k={nn}"] = on(sid, iso)
    a, b = rec["isomap/lattice/5x5/k=4"], rec["isomap/lattice/5x5-shuffled/k=4"]
    if "value" in a and "value" in b:
        rec["rule/isomap_input_order"] = {
            "samples": {k: S().points(k).digest for k in ("lattice/5x5", "lattice/5x5-shuffled")},
            **same_up_to_perm(a["value"]["embedding"]["array"], b["value"]["embedding"]["array"], perm),
        }
    for method in ("standard", "modified", "hessian", "ltsa"):

        def lle(s, method=method):
            m = M.LocallyLinearEmbedding(n_neighbors=10, method=method, eigen_solver="dense").fit(s.X)
            return {"embedding": m.embedding_, "reconstruction_error": m.reconstruction_error_}

        rec[f"lle/scurve/300/{method}"] = on("scurve/300", lle)
    rec["lle/scurve/300/standard_transform"] = on(
        "scurve/300",
        lambda s: M.LocallyLinearEmbedding(n_neighbors=10, eigen_solver="dense").fit(s.X).transform(s.Xe),
    )
    rec["rule/lle_arpack_random_state"] = on(
        "scurve/300",
        lambda s: {
            "two_unseeded_runs_bitwise": bool(
                np.array_equal(
                    M.LocallyLinearEmbedding(n_neighbors=10, eigen_solver="arpack").fit(s.X).embedding_,
                    M.LocallyLinearEmbedding(n_neighbors=10, eigen_solver="arpack").fit(s.X).embedding_,
                )
            )
        },
    )
    for sid in ("swiss/300", "twoblobs/80", "lattice/5x5", "blobs/k3p10"):
        for aff in ("nearest_neighbors", "rbf"):

            def se(s, aff=aff):
                m = M.SpectralEmbedding(2, affinity=aff, eigen_solver="arpack", random_state=0).fit(s.X)
                return {
                    "embedding": m.embedding_,
                    "n_neighbors": getattr(m, "n_neighbors_", None),
                    "columns_maxabs_positive": maxabs_positive(m.embedding_, axis=0),
                }

            rec[f"spectral_embedding/{sid}/{aff}"] = on(sid, se)
    # -- neighbour ties on the lattice
    rec["rule/kneighbors_ties"] = on(
        "lattice/5x5",
        lambda s: {
            "kneighbors_4_of_point_12": NearestNeighbors(n_neighbors=4).fit(s.X).kneighbors(s.X[12:13])[1],
            "kneighbors_4_brute": NearestNeighbors(n_neighbors=4, algorithm="brute").fit(s.X).kneighbors(s.X[12:13])[1],
            "kneighbors_4_kd_tree": NearestNeighbors(n_neighbors=4, algorithm="kd_tree")
            .fit(s.X)
            .kneighbors(s.X[12:13])[1],
            "exact_index_order": knn_exact(s.X, 5)[0][12, 1:],
        },
    )
    # -- trustworthiness
    for sid, n_neighbors in (("blobs/k3p10", 5), ("blobs/k3p10", 10), ("digits/500", 5), ("lattice/5x5", 5)):

        def tw(s, k=n_neighbors):
            Y = D.PCA(2, svd_solver="full").fit_transform(s.X) if s.p > 2 else s.X[:, :1]
            return {"embedding": Y, "trustworthiness": M.trustworthiness(s.X, Y, n_neighbors=k)}

        rec[f"trustworthiness/{sid}/k={n_neighbors}"] = on(sid, tw)
    rec["trustworthiness/swiss/300/truth"] = on("swiss/300", lambda s: M.trustworthiness(s.X, s.t, n_neighbors=10))
    rec["rule/trustworthiness_k_limit"] = on("tiny/3", lambda s: M.trustworthiness(s.X, s.X, n_neighbors=2))
    # -- t-SNE: P, a given start (exact and Barnes-Hut), perplexity limits
    rec["tsne/P_exact/lowrank/r1p2/perplexity=5"] = on(
        "lowrank/r1p2",
        lambda s: _t_sne._joint_probabilities(
            np.ascontiguousarray(((s.X[:, None] - s.X[None]) ** 2).sum(-1), np.float32), 5.0, 0
        ),
    )

    def p_nn(s, perp=10.0):
        k = min(s.n - 1, int(3.0 * perp + 1))
        nn = NearestNeighbors(n_neighbors=k).fit(s.X)
        Dg = nn.kneighbors_graph(mode="distance")
        Dg.data **= 2
        return _t_sne._joint_probabilities_nn(Dg, perp, 0)

    rec["tsne/P_nn/blobs/k3p10/perplexity=10"] = on("blobs/k3p10", p_nn)

    def tsne_given(s, method):
        Y0 = 1e-4 * D.PCA(2, svd_solver="full").fit_transform(s.X)
        Y0 = Y0 / np.std(Y0[:, 0]) * 1e-4
        t = M.TSNE(2, perplexity=10.0, init=Y0, method=method, random_state=0, max_iter=500)
        Y = t.fit_transform(s.X)
        return {"init": Y0, "embedding": Y, "kl_divergence": t.kl_divergence_, "n_iter": t.n_iter_,
                "learning_rate": t.learning_rate_}  # fmt: skip

    for method in ("exact", "barnes_hut"):
        rec[f"tsne/given_init/blobs/k3p10/{method}"] = on("blobs/k3p10", lambda s, m=method: tsne_given(s, m))
    rec["rule/tsne_pca_init"] = on(
        "blobs/k3p10",
        lambda s: (lambda t: {"kl": t.kl_divergence_, "embedding_first5": t.embedding_[:5]})(
            M.TSNE(2, perplexity=10.0, random_state=0).fit(s.X)
        ),
    )
    rec["rule/tsne_perplexity_too_large"] = on("blobs/k3p10", lambda s: M.TSNE(perplexity=200.0).fit(s.X))
    rec["rule/tsne_bh_4d"] = on("blobs/k3p10", lambda s: M.TSNE(4).fit(s.X))
    rec["rule/tsne_duplicates"] = {
        m: isolated("dups/10x4", f"M.TSNE(perplexity=5.0, method={m!r}, random_state=0).fit(s.X).kl_divergence_")
        for m in ("barnes_hut", "exact")
    }

    # -- degenerate inputs
    def deg(s):
        d = s.d
        nonneg = np.abs(s.X)
        tests = {
            "PCA": lambda: D.PCA(d).fit_transform(s.X),
            "PCA_None": lambda: D.PCA().fit(s.X).explained_variance_,
            "TruncatedSVD": lambda: D.TruncatedSVD(d, algorithm="arpack").fit_transform(s.X),
            "TruncatedSVD_randomized": lambda: D.TruncatedSVD(d, random_state=0).fit_transform(s.X),
            "KernelPCA_rbf": lambda: D.KernelPCA(d, kernel="rbf").fit_transform(s.X),
            "FactorAnalysis": lambda: D.FactorAnalysis(d).fit_transform(s.X),
            "FastICA": lambda: D.FastICA(d, random_state=0).fit_transform(s.X),
            "NMF_abs": lambda: D.NMF(d, max_iter=500).fit_transform(nonneg),
            "ClassicalMDS": lambda: M.ClassicalMDS(d).fit_transform(s.X),
            "MDS": lambda: M.MDS(d, init="classical_mds").fit_transform(s.X),
            "Isomap": lambda: M.Isomap(n_neighbors=min(5, s.n - 1) if s.n > 1 else 1, n_components=d).fit_transform(
                s.X
            ),
            "LLE": lambda: M.LocallyLinearEmbedding(
                n_neighbors=min(5, s.n - 1) if s.n > 1 else 1, n_components=d, eigen_solver="dense"
            ).fit_transform(s.X),
            "SpectralEmbedding": lambda: M.SpectralEmbedding(d, random_state=0).fit_transform(s.X),
            "trustworthiness": lambda: M.trustworthiness(s.X, s.X[:, :1], n_neighbors=1),
        }
        return {k: outcome(f) for k, f in tests.items()}

    for sid in DEGENERATE:
        rec[f"degenerate/{sid}"] = on(sid, deg)
        d = S().points(sid).d
        perp = max(0.5, (S().points(sid).n - 1) / 3.0 - 0.1)
        for method in ("barnes_hut", "exact"):
            rec[f"degenerate/{sid}/TSNE/{method}"] = isolated(
                sid, f"M.TSNE({min(d, 3)}, method={method!r}, random_state=0).fit_transform(s.X)"
            )
            rec[f"degenerate/{sid}/TSNE_perplexity_small/{method}"] = isolated(
                sid, f"M.TSNE({min(d, 3)}, perplexity={perp!r}, method={method!r}, random_state=0).fit_transform(s.X)"
            )
    # -- disconnected graphs
    rec["disconnected/twoblobs/80/isomap"] = on(
        "twoblobs/80", lambda s: (lambda m: {"embedding": m.embedding_, "eigenvalues": m.kernel_pca_.eigenvalues_})(
            M.Isomap(n_neighbors=5).fit(s.X)
        )
    )  # fmt: skip
    rec["disconnected/twoblobs/80/spectral_embedding"] = on(
        "twoblobs/80", lambda s: M.SpectralEmbedding(2, n_neighbors=5, random_state=0).fit(s.X).embedding_
    )
    rec["disconnected/twoblobs/80/lle"] = on(
        "twoblobs/80", lambda s: M.LocallyLinearEmbedding(n_neighbors=5, eigen_solver="dense").fit(s.X).embedding_
    )
    # -- thread dependence of TSNE's Barnes-Hut (OpenMP; same random_state and init='pca')
    s = S().points("large/blobs")
    runs = {}
    for label, t in (("threads=1", 1), ("threads=10", 10), ("threads=10 again", 10), ("threads=3", 3)):
        with threadpool_limits(limits=t, user_api="openmp"):
            tt = M.TSNE(2, random_state=0).fit(s.X)
        runs[label] = (tt.embedding_.copy(), float(tt.kl_divergence_), int(tt.n_iter_))
        log(f"  sklearn TSNE BH large/blobs {label}: KL {runs[label][1]!r}")
    rec["threads/tsne_barnes_hut/large/blobs"] = {
        "sample": "large/blobs",
        "digest": s.digest,
        "kl_divergence": {k: v[1] for k, v in runs.items()},
        "n_iter": {k: v[2] for k, v in runs.items()},
        "embedding": thread_compare({k: v[0] for k, v in runs.items()}),
        "openmp": [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "openmp"],
    }
    meta = {"tool": TOOL, "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "threadpoolctl": threadpoolctl.__version__, "machine": MACHINE}  # fmt: skip
    log(write_records("sklearn.json", meta, rec))


# ---------------------------------------------------------------- scipy


def scipy_records(log):
    import scipy
    import scipy.linalg as la
    import scipy.sparse.linalg as sl

    rec = {"defaults": {f.__name__: defaults(f) for f in (la.svd, la.eigh, sl.svds, sl.eigsh)}}
    for sid in ("lowrank/r3p10", "equalvar/p4", "sparse/n100p50"):
        s = S().points(sid)
        Xc = s.X - s.X.mean(axis=0)
        rec[f"svd/{sid}"] = on(
            sid,
            lambda s, Xc=Xc: (lambda U, sv, Vt: {"singular_values": sv, "Vt_rows_maxabs_positive": maxabs_positive(Vt),
                                                 "U_cols_maxabs_positive": maxabs_positive(U, axis=0)})(
                *la.svd(Xc, full_matrices=False)
            ),
        )  # fmt: skip
        for solver in ("arpack", "propack", "lobpcg"):

            def svds(s, Xc=Xc, solver=solver):
                U, sv, Vt = sl.svds(Xc, k=s.d, solver=solver, rng=0)
                U2, sv2, Vt2 = sl.svds(Xc, k=s.d, solver=solver, rng=0)
                U3, sv3, Vt3 = sl.svds(Xc, k=s.d, solver=solver)
                U4, sv4, Vt4 = sl.svds(Xc, k=s.d, solver=solver)
                return {
                    "singular_values": sv,
                    "ascending": bool(np.all(np.diff(sv) >= 0)),
                    "Vt_rows_maxabs_positive": maxabs_positive(Vt),
                    "same_rng_bitwise": bool(np.array_equal(Vt, Vt2) and np.array_equal(sv, sv2)),
                    "unseeded_twice_bitwise": bool(np.array_equal(Vt3, Vt4) and np.array_equal(sv3, sv4)),
                    "unseeded_twice_max_sv_diff": float(np.abs(sv3 - sv4).max()),
                }

            rec[f"svds/{sid}/{solver}"] = on(sid, svds)
    s = S().points("swiss/300")
    from sklearn.neighbors import kneighbors_graph

    def lap(s):
        import scipy.sparse.csgraph as cg

        A = kneighbors_graph(s.X, 10, include_self=False)
        A = 0.5 * (A + A.T)
        L = cg.laplacian(A, normed=True)
        vals, vecs = sl.eigsh(L, k=3, sigma=0.0, which="LM", rng=0)
        vals2, _ = sl.eigsh(L, k=3, sigma=0.0, which="LM")
        vals3, _ = sl.eigsh(L, k=3, sigma=0.0, which="LM")
        return {"eigenvalues": vals, "ascending": bool(np.all(np.diff(vals) >= 0)),
                "unseeded_twice_bitwise": bool(np.array_equal(vals2, vals3))}  # fmt: skip

    rec["eigsh/swiss/300/normed_laplacian_shift_invert"] = on("swiss/300", lap)
    meta = {"tool": TOOL, "scipy": scipy.__version__, "numpy": np.__version__}
    log(write_records("scipy.json", meta, rec))


# ---------------------------------------------------------------- statsmodels


def statsmodels_records(log):
    import statsmodels
    import statsmodels.multivariate.factor as F
    import statsmodels.multivariate.factor_rotation as FR
    from statsmodels.multivariate.pca import PCA

    rec = {
        "defaults": {
            "PCA": defaults(PCA),
            "Factor": defaults(F.Factor),
            "Factor.fit": defaults(F.Factor.fit),
            "rotate_factors": defaults(FR.rotate_factors),
            "promax": defaults(FR.promax),
        }
    }
    for sid in ("lowrank/r3p10", "factor/p8k2", "equalvar/p4"):
        for standardize in (True, False):

            def pca(s, st=standardize):
                p = PCA(s.X, ncomp=s.d, standardize=st)
                return {"eigenvals": p.eigenvals, "loadings": np.asarray(p.loadings), "factors_first5": np.asarray(
                    p.factors)[:5], "coeff": np.asarray(p.coeff),
                    "loadings_cols_maxabs_positive": maxabs_positive(np.asarray(p.loadings), axis=0)}  # fmt: skip

            rec[f"pca/{sid}/standardize={standardize}"] = on(sid, pca)
    for method in ("ml", "pa"):

        def fac(s, method=method):
            r = F.Factor(s.X, n_factor=2, method=method).fit()
            out = {"loadings": r.loadings, "uniqueness": r.uniqueness}
            for rot in ("varimax", "quartimax", "promax"):

                def rotated(rot=rot):
                    rr = F.Factor(s.X, n_factor=2, method=method).fit()
                    rr.rotate(rot)
                    return rr.loadings

                out[f"loadings_{rot}"] = outcome(rotated)
            return out

        rec[f"factor/factor/p8k2/{method}"] = on("factor/p8k2", fac)

    def ml_twice(s):
        a = np.asarray(F.Factor(s.X, n_factor=2, method="ml").fit().loadings)
        b = np.asarray(F.Factor(s.X, n_factor=2, method="ml").fit().loadings)
        return {"dtype": str(a.dtype), "max_abs_imag": float(np.abs(np.imag(a)).max()),
                "twice_bitwise": bool(np.array_equal(a, b)), "twice_max_diff": float(np.abs(a - b).max())}  # fmt: skip

    rec["rule/factor_ml_complex_and_repeat"] = on("factor/p8k2", ml_twice)
    try:
        L = np.real(F.Factor(S().points("factor/p8k2").X, n_factor=2, method="ml").fit().loadings).astype(float)
    except Exception:  # noqa: BLE001 -- the fit's failure is recorded above
        L = None
    if L is not None:
        rec["rotate_factors/input_loadings"] = {
            "from": "Factor(factor/p8k2, n_factor=2, method='ml').fit().loadings (real part)",
            "sample": "factor/p8k2",
            "digest": S().points("factor/p8k2").digest,
            "loadings": L,
        }
        for m in ("varimax", "quartimax"):
            rec[f"rotate_factors/{m}"] = outcome(
                lambda m=m: dict(zip(("loadings", "rotation"), FR.rotate_factors(L, m)))
            )
        rec["rotate_factors/promax"] = outcome(lambda: dict(zip(("loadings", "rotation"), FR.promax(L, 4))))
    meta = {"tool": TOOL, "statsmodels": statsmodels.__version__, "numpy": np.__version__}
    log(write_records("statsmodels.json", meta, rec))


# ---------------------------------------------------------------- R

_R_HELP = r"""
.sd_enc <- function(s) { s <- enc2utf8(as.character(s)); c(nchar(s, type = "chars"), utf8ToInt(s)) }
.sd_rec <- function(f) {
  op <- options(warn = 0); on.exit(options(op))
  ws <- character(0)
  keep <- function(w) { ws <<- c(ws, conditionMessage(w)); invokeRestart("muffleWarning") }
  v <- tryCatch(withCallingHandlers(f(), warning = keep),
                error = function(e) structure(conditionMessage(e), class = "sderr"))
  if (inherits(v, "sderr")) return(c(-1, .sd_enc(v)))
  if (!is.list(v) || is.data.frame(v)) v <- list(value = v)
  out <- length(v)
  for (nm in names(v)) {
    x <- v[[nm]]
    out <- c(out, .sd_enc(nm))
    if (is.character(x)) {
      out <- c(out, 1, .sd_enc(paste(x, collapse = "\x1f")))
    } else {
      d <- dim(x); x <- as.double(unclass(x))
      out <- c(out, 0, length(d), d, length(x), x)
    }
  }
  c(out, .sd_enc(paste(unique(ws), collapse = " | ")))
}
"""


def r_call(body, data=None):
    """{"value": {name: array or string}, "warnings": "..."} or {"error": "..."}: an R function body returning a
    named list (numbers, matrices or strings) over the named vectors in data."""
    from bench.accuracy import rsession

    try:
        out = rsession.evaluate_named(_R_HELP + f".sd_rec(function() {{ {body} }})", data or {"z": np.zeros(1)})
    except RuntimeError as e:
        return {"error": str(e)[:400]}
    pos = [0]

    def take(k):
        v = out[pos[0] : pos[0] + k]
        pos[0] += k
        return v

    def text():
        n = int(take(1)[0])
        return "".join(chr(int(c)) for c in take(n))

    head = take(1)[0]
    if head == -1:
        return {"error": text()}
    vals = {}
    for _ in range(int(head)):
        name = text()
        kind = int(take(1)[0])
        if kind == 1:
            vals[name] = text().split("\x1f")
            continue
        nd = int(take(1)[0])
        dims = [int(x) for x in take(nd)]
        n = int(take(1)[0])
        x = np.asarray(take(n), float)
        vals[name] = x.reshape(dims, order="F") if nd >= 2 else x
    res = {"value": vals["value"] if list(vals) == ["value"] else vals}
    msg = text()
    if msg:
        res["warnings"] = msg
    return res


def rmat(X):
    X = np.asarray(X, float)
    return {"xv": X.T.ravel(), "dims": np.array(X.shape, float)}


XR = "x <- matrix(xv, nrow = dims[1])"


def r_on(sid, body, extra=None):
    s = S().points(sid)
    return {"sample": sid, "digest": s.digest, **r_call(f"{XR}; {body}", {**rmat(s.X), **(extra or {})})}


R_FUNCS = {
    "prcomp": "stats:::prcomp.default",
    "princomp": "stats:::princomp.default",
    "cmdscale": "stats::cmdscale",
    "factanal": "stats::factanal",
    "varimax": "stats::varimax",
    "promax": "stats::promax",
    "sammon": "MASS::sammon",
    "isoMDS": "MASS::isoMDS",
    "irlba": "irlba::irlba",
    "prcomp_irlba": "irlba::prcomp_irlba",
    "svds": "RSpectra::svds",
    "eigs_sym": "RSpectra::eigs_sym",
    "umap": "uwot::umap",
    "Rtsne": "Rtsne:::Rtsne.default",
    "fastICA": "fastICA::fastICA",
}
R_PACKAGES = ("MASS", "irlba", "RSpectra", "uwot", "Rtsne", "fastICA", "Matrix", "RcppAnnoy")


def r_records(log):
    from bench.accuracy import rsession

    if not rsession.available():
        log("r: R is not on PATH; skipped")
        return
    rec = {}
    vers = r_call(
        "c(list(R = paste(R.version$major, R.version$minor, sep = '.')), "
        + "setNames(lapply(c("
        + ", ".join(f"'{p}'" for p in R_PACKAGES)
        + "), function(p) as.character(packageVersion(p))), c("
        + ", ".join(f"'{p}'" for p in R_PACKAGES)
        + ")))"
    )
    rec["versions"] = vers
    rec["defaults"] = r_call(
        "fs <- list("
        + ", ".join(f"{k} = {v}" for k, v in R_FUNCS.items())
        + "); lapply(fs, function(f) { a <- formals(f); "
        "paste(names(a), vapply(a, function(x) paste(deparse(x), collapse = ''), ''), sep = '=') })"
    )
    # -- prcomp, princomp
    for sid in ("lowrank/r3p10", "equalvar/p4", "wide/n10p40", "shifted/p6", "constcol/p6"):
        rec[f"prcomp/{sid}"] = r_on(
            sid, "p <- prcomp(x); list(sdev = p$sdev, rotation = p$rotation, x = p$x, center = p$center)"
        )
    rec["prcomp/lowrank/r3p10/scale"] = r_on(
        "lowrank/r3p10", "p <- prcomp(x, scale. = TRUE, rank. = 3); list(sdev = p$sdev, rotation = p$rotation, "
        "scale = p$scale)"
    )  # fmt: skip
    rec["prcomp/constcol/p6/scale"] = r_on("constcol/p6", "prcomp(x, scale. = TRUE)$sdev")
    for sid in ("lowrank/r3p10", "factor/p8k2"):
        for cor in ("FALSE", "TRUE"):
            rec[f"princomp/{sid}/cor={cor}"] = r_on(
                sid,
                f"p <- princomp(x, cor = {cor}); list(sdev = p$sdev, loadings = unclass(p$loadings), "
                "center = p$center, scale = p$scale, scores_first5 = p$scores[1:5, ])",
            )
    # -- cmdscale
    for sid in ("circle/60", "lattice/5x5", "swiss/300", "blobs/k3p10"):
        rec[f"cmdscale/{sid}"] = r_on(
            sid, "m <- cmdscale(dist(x), k = 2, eig = TRUE); list(points = m$points, eig = m$eig, GOF = m$GOF)"
        )
    rec["cmdscale/blobs/k3p10/cityblock_add"] = r_on(
        "blobs/k3p10",
        "d <- dist(x, 'manhattan'); m <- cmdscale(d, k = 2, eig = TRUE); a <- cmdscale(d, k = 2, eig = TRUE, "
        "add = TRUE); list(eig = m$eig, GOF = m$GOF, ac = a$ac, points_add = a$points, eig_add = a$eig, "
        "GOF_add = a$GOF)",
    )
    rec["cmdscale/lattice/5x5/k3"] = r_on("lattice/5x5", "cmdscale(dist(x), k = 3)")
    # -- factanal, varimax, promax
    s = S().points("factor/p8k2")
    for rot in ("varimax", "promax", "none"):
        rec[f"factanal/factor/p8k2/{rot}"] = r_on(
            "factor/p8k2",
            f"f <- factanal(x, 2, rotation = '{rot}', scores = 'regression'); "
            "fb <- factanal(x, 2, rotation = 'none', scores = 'Bartlett'); "
            "list(loadings = unclass(f$loadings), uniquenesses = f$uniquenesses, statistic = f$STATISTIC, "
            "dof = f$dof, pvalue = f$PVAL, criteria = f$criteria, converged = f$converged, "
            "rotmat = if (is.null(f$rotmat)) numeric(0) else f$rotmat, scores_first5 = f$scores[1:5, ], "
            "bartlett_first5 = fb$scores[1:5, ], start = (1 - 0.5 * 2 / ncol(x)) / diag(solve(cor(x))))",
        )
    rec["factanal/factor/p8k2/3factors"] = r_on(
        "factor/p8k2", "f <- factanal(x, 3); list(dof = f$dof, statistic = f$STATISTIC, uniq = f$uniquenesses)"
    )
    rec["factanal/factor/p8k2/5factors"] = r_on("factor/p8k2", "factanal(x, 5)$dof")
    rec["varimax_promax/factor/p8k2"] = r_on(
        "factor/p8k2",
        "L <- unclass(factanal(x, 2, rotation = 'none')$loadings); v <- varimax(L); p <- promax(L); "
        "v2 <- varimax(L, normalize = FALSE); list(input = L, varimax = unclass(v$loadings), "
        "varimax_rotmat = v$rotmat, "
        "varimax_nonorm = unclass(v2$loadings), promax = unclass(p$loadings), promax_rotmat = p$rotmat)",
    )
    # -- sammon, isoMDS
    for sid in ("circle/60", "blobs/k3p10", "lattice/5x5"):
        rec[f"sammon/{sid}"] = r_on(
            sid, "library(MASS); m <- sammon(dist(x), k = 2, trace = FALSE); list(points = m$points, stress = m$stress)"
        )
        rec[f"isoMDS/{sid}"] = r_on(
            sid, "library(MASS); m <- isoMDS(dist(x), k = 2, trace = FALSE); list(points = m$points, stress = m$stress)"
        )
    rec["rule/sammon_zero_distance"] = r_on("dups/10x4", "library(MASS); sammon(dist(x), trace = FALSE)$stress")
    rec["rule/isomds_zero_distance"] = r_on("dups/10x4", "library(MASS); isoMDS(dist(x), trace = FALSE)$stress")
    # -- irlba, RSpectra
    rec["irlba/lowrank/r3p10"] = r_on(
        "lowrank/r3p10",
        "library(irlba); xc <- scale(x, scale = FALSE); list(irlba_d = irlba(xc, nv = 3)$d, "
        "prcomp_irlba_sdev = prcomp_irlba(x, n = 3)$sdev, svd_d = svd(xc)$d[1:3])",
    )
    rec["RSpectra/lowrank/r3p10"] = r_on(
        "lowrank/r3p10",
        "library(RSpectra); xc <- scale(x, scale = FALSE); s <- svds(xc, k = 3); "
        "e <- eigs_sym(crossprod(xc), k = 3); list(svds_d = s$d, svds_v = s$v, eigs_values = e$values)",
    )
    # -- uwot
    u = (
        "umap(x, n_neighbors = 15, min_dist = 0.1, n_epochs = 200, batch = TRUE, seed = 1, "
        "nn_method = 'fnn', verbose = FALSE, n_threads = {t}, n_sgd_threads = {t})"
    )
    rec["uwot/blobs/k3p10/batch"] = r_on("blobs/k3p10", "library(uwot); " + u.format(t=1))
    rec["rule/uwot_batch_threads"] = r_on(
        "blobs/k3p10",
        f"library(uwot); a <- {u.format(t=1)}; b <- {u.format(t=4)}; c <- {u.format(t=4)}; "
        "list(threads1_vs_4_bitwise = identical(a, b), threads4_twice_bitwise = identical(b, c), "
        "max_diff = max(abs(a - b)))",
    )
    rec["rule/uwot_hogwild_seeded"] = r_on(
        "blobs/k3p10",
        "library(uwot); f <- function(t) umap(x, seed = 1, nn_method = 'fnn', verbose = FALSE, n_sgd_threads = t); "
        "a <- f(1); b <- f(1); c <- f(4); d <- f(4); list(sgd1_twice_bitwise = identical(a, b), "
        "sgd4_twice_bitwise = identical(c, d), sgd4_max_diff = max(abs(c - d)))",
    )
    rec["uwot/find_ab_params"] = r_call(
        "f <- uwot:::find_ab_params; list(" + ", ".join(
            f"'{md}_{sp}' = f(spread = {sp}, min_dist = {md})"
            for md, sp in ((0.1, 1.0), (0.01, 1.0), (0.5, 1.0), (0.001, 1.0), (0.1, 2.0), (0.25, 0.5))
        ) + ")"
    )  # fmt: skip
    rec["uwot/similarity_graph/blobs/k3p10"] = r_on(
        "blobs/k3p10",
        "library(uwot); g <- similarity_graph(x, n_neighbors = 15, nn_method = 'fnn', verbose = FALSE); "
        "g <- as(g, 'TsparseMatrix'); list(i = g@i, j = g@j, x = g@x)",
    )
    rec["uwot/disconnected/twoblobs/80"] = r_on(
        "twoblobs/80",
        "library(uwot); umap(x, n_neighbors = 5, seed = 1, batch = TRUE, nn_method = 'fnn', verbose = FALSE)",
    )
    for sid in DEGENERATE:
        rec[f"uwot/degenerate/{sid}"] = r_on(
            sid, "library(uwot); umap(x, n_neighbors = 15, seed = 1, batch = TRUE, verbose = FALSE)"
        )
        rec[f"uwot/degenerate/{sid}/n_neighbors=n-1"] = r_on(
            sid, "library(uwot); umap(x, n_neighbors = max(nrow(x) - 1, 1), seed = 1, batch = TRUE, verbose = FALSE)"
        )
    # -- Rtsne
    s = S().points("blobs/k3p10")
    Y0 = np.ascontiguousarray(1e-4 * (s.X - s.X.mean(0)) @ np.linalg.svd(s.X - s.X.mean(0))[2][:2].T)
    rec["Rtsne/blobs/k3p10/given_init"] = r_on(
        "blobs/k3p10",
        "library(Rtsne); y0 <- matrix(yv, nrow = dims[1]); "
        "r <- Rtsne(x, perplexity = 10, pca = FALSE, normalize = FALSE, Y_init = y0, max_iter = 500, theta = 0.5); "
        "e <- Rtsne(x, perplexity = 10, pca = FALSE, normalize = FALSE, Y_init = y0, max_iter = 500, theta = 0); "
        "list(Y = r$Y, itercosts = r$itercosts, Y_exact = e$Y, itercosts_exact = e$itercosts)",
        {"yv": Y0.T.ravel()},
    )
    rec["Rtsne/given_init_Y0"] = {"from": "1e-4 x the first two principal scores", "sample": "blobs/k3p10",
                                  "digest": s.digest, "Y0": Y0}  # fmt: skip
    rec["rule/rtsne_threads"] = r_on(
        "large/blobs",
        "library(Rtsne); f <- function(t) { set.seed(1); Rtsne(x[1:3000, ], pca = FALSE, num_threads = t, "
        "max_iter = 300)$Y }; a <- f(1); b <- f(10); c <- f(10); list(threads1_vs_10_bitwise = identical(a, b), "
        "threads10_twice_bitwise = identical(b, c), max_diff = max(abs(a - b)))",
    )
    for sid in ("tiny/3", "allsame/8", "dups/10x4"):
        rec[f"Rtsne/degenerate/{sid}"] = r_on(sid, "library(Rtsne); set.seed(1); Rtsne(x, perplexity = 30)$Y")
        rec[f"Rtsne/degenerate/{sid}/perplexity_small"] = r_on(
            sid,
            "library(Rtsne); set.seed(1); Rtsne(x, perplexity = (nrow(x) - 1) / 3 - 0.01, check_duplicates = FALSE)$Y",
        )
    rec["Rtsne/degenerate/dups/10x4/no_check"] = r_on(
        "dups/10x4", "library(Rtsne); set.seed(1); Rtsne(x, perplexity = 5, check_duplicates = FALSE)$Y"
    )
    # -- fastICA
    w0 = S().rng("probe", "w_init").standard_normal((4, 4))
    for alg in ("parallel", "deflation"):
        for method in ("R", "C"):
            rec[f"fastICA/ica/s4/{alg}/{method}"] = r_on(
                "ica/s4",
                f"library(fastICA); w <- matrix(wv, 4); f <- fastICA(x, 4, alg.typ = '{alg}', method = '{method}', "
                "w.init = w, maxit = 1000); list(K = f$K, W = f$W, A = f$A, S_first5 = f$S[1:5, ])",
                {"wv": w0.T.ravel()},
            )
    rec["fastICA/w_init"] = {"from": "samples.rng('probe', 'w_init').standard_normal((4, 4))", "w_init": w0}
    # -- degenerate inputs
    for sid in DEGENERATE:
        d = S().points(sid).d
        rec[f"degenerate/{sid}"] = {
            "prcomp": r_on(sid, "p <- prcomp(x); list(sdev = p$sdev, x = p$x)"),
            "prcomp_scale": r_on(sid, "prcomp(x, scale. = TRUE)$sdev"),
            "princomp": r_on(sid, "princomp(x)$sdev"),
            "cmdscale": r_on(sid, f"cmdscale(dist(x), k = {d}, eig = TRUE)[c('points', 'eig')]"),
            "factanal": r_on(sid, "factanal(x, 1)$uniquenesses"),
            "sammon": r_on(sid, f"library(MASS); sammon(dist(x), k = {d}, trace = FALSE)$stress"),
            "isoMDS": r_on(sid, f"library(MASS); isoMDS(dist(x), k = {d}, trace = FALSE)$stress"),
            "fastICA": r_on(sid, f"library(fastICA); set.seed(1); fastICA(x, {d})$S"),
        }
    meta = {"tool": TOOL, "R": f"R version {rsession.version()}", "machine": MACHINE}
    log(write_records("r.json", meta, rec))


# ---------------------------------------------------------------- umap-learn


def umap_records(log):
    import numba
    import pynndescent
    import umap
    from umap import umap_ as U

    rec = {"defaults": {"UMAP": defaults(umap.UMAP), "fuzzy_simplicial_set": defaults(U.fuzzy_simplicial_set),
                        "smooth_knn_dist": defaults(U.smooth_knn_dist)}}  # fmt: skip
    rec["rule/compatibility_layout_warning"] = U.COMPATIBILITY_LAYOUT_DEFAULT_WARNING
    rec["rule/init_auto"] = "init='auto' is 'spectral' with compatibility_layout=True, 'recursive' otherwise"
    rec["rule/default_fit_warnings"] = on("blobs/k3p10", lambda s: umap.UMAP().fit(s.X).embedding_.shape)
    rec["rule/seeded_fit_warnings"] = on("blobs/k3p10", lambda s: umap.UMAP(random_state=0).fit(s.X).embedding_.shape)
    rec["rule/seeded_modern_fit_warnings"] = on(
        "blobs/k3p10", lambda s: umap.UMAP(random_state=0, compatibility_layout=False).fit(s.X).embedding_.shape
    )
    rec["find_ab_params"] = {
        f"min_dist={md} spread={sp}": list(U.find_ab_params(sp, md))
        for md, sp in ((0.1, 1.0), (0.01, 1.0), (0.5, 1.0), (0.001, 1.0), (0.0, 1.0), (0.1, 2.0), (0.25, 0.5))
    }
    for sid in ("blobs/k3p10", "circle/60", "dups/10x4"):
        s = S().points(sid)
        k = 15 if s.n > 15 else s.n
        idx, dist = knn_exact(s.X, k)
        for lc in (1.0, 2.0):
            rec[f"smooth_knn_dist/{sid}/k={k}/local_connectivity={lc}"] = on(
                sid,
                lambda s, dist=dist, k=k, lc=lc: dict(
                    zip(("sigmas", "rhos"), U.smooth_knn_dist(dist.astype(np.float32), float(k), local_connectivity=lc))
                ),
            )
        for mix in (1.0, 0.5):
            rec[f"fuzzy_simplicial_set/{sid}/k={k}/set_op_mix_ratio={mix}"] = on(
                sid,
                lambda s, idx=idx, dist=dist, k=k, mix=mix: dict(
                    zip(
                        ("graph", "sigmas", "rhos"),
                        U.fuzzy_simplicial_set(
                            s.X, k, np.random.RandomState(0), "euclidean", knn_indices=idx,
                            knn_dists=dist.astype(np.float32), set_op_mix_ratio=mix,
                        ),
                    )
                ),
            )  # fmt: skip
        rec[f"knn_exact/{sid}/k={k}"] = {"sample": sid, "digest": s.digest, "indices": idx, "distances": dist}
    rec["umap/blobs/k3p10/seeded_legacy"] = on(
        "blobs/k3p10", lambda s: umap.UMAP(random_state=0, compatibility_layout=True, n_jobs=1).fit(s.X).embedding_
    )
    rec["umap/blobs/k3p10/seeded_modern"] = on(
        "blobs/k3p10", lambda s: umap.UMAP(random_state=0, compatibility_layout=False).fit(s.X).embedding_
    )
    rec["disconnected/twoblobs/80"] = on(
        "twoblobs/80",
        lambda s: umap.UMAP(n_neighbors=5, random_state=0, compatibility_layout=True, n_jobs=1).fit(s.X).embedding_,
    )
    rec["disconnected/twoblobs/80/modern"] = on(
        "twoblobs/80",
        lambda s: umap.UMAP(n_neighbors=5, random_state=0, compatibility_layout=False).fit(s.X).embedding_,
    )
    for sid in DEGENERATE:
        rec[f"degenerate/{sid}"] = on(
            sid,
            lambda s: umap.UMAP(random_state=0, compatibility_layout=True, n_jobs=1).fit(s.X).embedding_,
        )
        rec[f"degenerate/{sid}/n_neighbors=2"] = on(
            sid,
            lambda s: umap.UMAP(n_neighbors=2, random_state=0, compatibility_layout=True, n_jobs=1).fit(s.X).embedding_,
        )
    # -- run-to-run and thread dependence on the first 5000 of large/blobs
    s = S().points("large/blobs")
    X = s.X[:5000]

    def fit(data=None, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return umap.UMAP(**kw).fit(X if data is None else data).embedding_.copy()

    runs = {
        "legacy seeded n_jobs=1": fit(random_state=0, n_jobs=1, compatibility_layout=True),
        "legacy seeded n_jobs=1 again": fit(random_state=0, n_jobs=1, compatibility_layout=True),
    }
    rec["threads/legacy_seeded"] = {"sample": "large/blobs", "digest": s.digest, "rows": 5000, **thread_compare(runs)}
    runs = {
        "legacy unseeded n_jobs=10": fit(n_jobs=10, compatibility_layout=True),
        "legacy unseeded n_jobs=10 again": fit(n_jobs=10, compatibility_layout=True),
    }
    rec["threads/legacy_unseeded"] = {"sample": "large/blobs", "digest": s.digest, "rows": 5000, **thread_compare(runs)}
    runs = {
        "modern seeded n_jobs=10": fit(random_state=0, n_jobs=10, compatibility_layout=False),
        "modern seeded n_jobs=10 again": fit(random_state=0, n_jobs=10, compatibility_layout=False),
        "modern seeded n_jobs=1": fit(random_state=0, n_jobs=1, compatibility_layout=False),
        "modern seeded n_jobs=3": fit(random_state=0, n_jobs=3, compatibility_layout=False),
    }
    rec["threads/modern_seeded"] = {"sample": "large/blobs", "digest": s.digest, "rows": 5000, **thread_compare(runs)}
    rec["threads/sample"] = {"sample": "large/blobs", "digest": s.digest, "rows": "first 5000"}

    def init_repeat(data, **kw):
        a = fit(data, random_state=0, n_jobs=1, compatibility_layout=True, **kw)
        b = fit(data, random_state=0, n_jobs=1, compatibility_layout=True, **kw)
        return {"bitwise": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max())}

    sw = S().points("large/swiss")
    rec["rule/legacy_seeded_repeat_by_init"] = {
        "note": "the same seeded 1-thread fit twice; large/blobs (first 5000) is five far apart blobs (a "
        "disconnected graph), large/swiss is connected",
        "large/blobs": {"digest": s.digest, **{f"init={i}": init_repeat(X, init=i) for i in ("spectral", "random")}},
        "large/blobs n_epochs=[0]": {"digest": s.digest, "spectral": init_repeat(X, n_epochs=[0])},
        "large/swiss": {"digest": sw.digest, "spectral": init_repeat(sw.X)},
    }
    log(f"  umap rule/legacy_seeded_repeat_by_init: {rec['rule/legacy_seeded_repeat_by_init']}")
    for k in ("threads/legacy_seeded", "threads/legacy_unseeded", "threads/modern_seeded"):
        log(f"  umap {k}: {rec[k]}")
    meta = {"tool": TOOL, "umap-learn": umap.__version__, "numba": numba.__version__,
            "pynndescent": pynndescent.__version__, "numpy": np.__version__, "machine": MACHINE}  # fmt: skip
    log(write_records("umap.json", meta, rec))


# ---------------------------------------------------------------- openTSNE


def opentsne_records(log):
    import openTSNE
    from openTSNE import affinity

    rec = {"defaults": {"TSNE": defaults(openTSNE.TSNE), "PerplexityBasedNN": defaults(affinity.PerplexityBasedNN),
                        "TSNEEmbedding.optimize": defaults(openTSNE.TSNEEmbedding.optimize)}}  # fmt: skip
    for sid, perp in (("lowrank/r1p2", 5.0), ("blobs/k3p10", 10.0), ("circle/60", 5.0)):
        rec[f"P/{sid}/perplexity={perp}/exact"] = on(
            sid, lambda s, p=perp: affinity.PerplexityBasedNN(s.X, perplexity=p, method="exact", random_state=0).P
        )

        def vs_annoy(s, p=perp):
            a = affinity.PerplexityBasedNN(s.X, perplexity=p, method="exact", random_state=0).P.toarray()
            b = affinity.PerplexityBasedNN(s.X, perplexity=p, method="annoy", random_state=0).P.toarray()
            return {"bitwise": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max())}

        rec[f"P/{sid}/perplexity={perp}/annoy_vs_exact"] = on(sid, vs_annoy)
    rec["P/blobs/k3p10/perplexity=30.0/row_sums_and_nnz"] = on(
        "blobs/k3p10",
        lambda s: (lambda P: {"nnz": P.nnz, "sum": P.sum(), "row_sums": np.asarray(P.sum(axis=1)).ravel()})(
            affinity.PerplexityBasedNN(s.X, perplexity=30.0, method="exact", random_state=0).P
        ),
    )
    s = S().points("blobs/k3p10")
    Y0 = 1e-4 * (s.X - s.X.mean(0)) @ np.linalg.svd(s.X - s.X.mean(0))[2][:2].T
    for method in ("bh", "fft"):
        rec[f"fit/blobs/k3p10/given_init/{method}"] = on(
            "blobs/k3p10",
            lambda s, m=method: (lambda e: {"embedding": np.asarray(e), "kl_divergence": e.kl_divergence})(
                openTSNE.TSNE(perplexity=10, negative_gradient_method=m, n_jobs=1, random_state=0).fit(
                    s.X, initialization=Y0
                )
            ),
        )
    rec["fit/given_init_Y0"] = {"from": "1e-4 x the first two principal scores", "sample": "blobs/k3p10",
                                "digest": s.digest, "Y0": Y0}  # fmt: skip
    rec["rule/pca_init_scaling"] = on(
        "blobs/k3p10",
        lambda s: (lambda Y: {"std_col0": float(np.std(Y[:, 0])), "first5": Y[:5]})(
            openTSNE.initialization.pca(s.X, random_state=0)
        ),
    )
    for sid in DEGENERATE + ("dups/10x4",):
        rec[f"degenerate/{sid}"] = on(sid, lambda s: np.asarray(openTSNE.TSNE(random_state=0).fit(s.X)))
    # -- thread dependence: the same random_state at n_jobs 1, 10, 10 again
    s = S().points("large/blobs")
    for method in ("fft", "bh"):
        X = s.X if method == "fft" else s.X[:5000]
        runs = {}
        for label, jobs in (("n_jobs=1", 1), ("n_jobs=10", 10), ("n_jobs=10 again", 10)):
            e = openTSNE.TSNE(negative_gradient_method=method, n_jobs=jobs, random_state=0).fit(X)
            runs[label] = (np.asarray(e).copy(), float(e.kl_divergence))
        rec[f"threads/{method}"] = {
            "sample": "large/blobs",
            "digest": s.digest,
            "rows": len(X),
            "kl_divergence": {k: v[1] for k, v in runs.items()},
            "embedding": thread_compare({k: v[0] for k, v in runs.items()}),
        }
        log(f"  openTSNE {method}: {rec[f'threads/{method}']['embedding']}")
    meta = {"tool": TOOL, "openTSNE": openTSNE.__version__, "numpy": np.__version__, "machine": MACHINE}
    log(write_records("opentsne.json", meta, rec))


# ---------------------------------------------------------------- timings

ONE_CALL_S = 3.0


def _jobs(label: str) -> int | None:
    m = re.search(r"(?:n_jobs|n_threads)=(\d+)", label)
    return int(m.group(1)) if m else None


def _timed(fn):
    from bench.speed import harness

    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first >= ONE_CALL_S:
        return {"s": first, "calls": 1}
    sm = harness.measure(fn, budget_s=2.0)
    return {"s": min(sm.times), "calls": sm.calls * len(sm.times)}


def _r_timed(sess, ref, d):
    data = {name: d[key] for key, name in ref.data.items()}
    lines, files = sess._prepare(ref.expr, data, ref.setup, ref.packages, None)
    lines.append(".sd_t0 <- .sd_now(); invisible(.sd_f()); .sd_t1 <- .sd_now() - .sd_t0; .sd_n <- 1")
    lines.append(
        f"if (.sd_t1 < {ONE_CALL_S}) {{ .sd_r <- .sd_time(.sd_f, 0.02, 7, 3, 2.0, 5.0); "
        ".sd_t1 <- min(.sd_r$times); .sd_n <- .sd_r$calls * length(.sd_r$times) }"
    )
    lines.append('cat("SDRESULT", .sd_t1, .sd_n, "\\n")')
    for ln in sess._run(lines, files, ref.packages):
        if ln.startswith("SDRESULT"):
            parts = ln.split()
            return {"s": float(parts[1]), "calls": int(float(parts[2]))}
    raise RuntimeError("R gave no time")


def _warm(libs):
    """One small call of each JIT-compiled library (umap-learn's numba kernels) before the timings."""
    if "umap" in libs:
        import umap

        X = np.random.default_rng(0).standard_normal((2000, 50))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for compat, jobs, seed in ((True, 1, 0), (True, 10, None), (False, 10, 0)):
                umap.UMAP(n_jobs=jobs, random_state=seed, compatibility_layout=compat).fit(X)


def time_child(threads: int, match):
    """One thread count's timings (a child process; the parent set VECLIB_MAXIMUM_THREADS): one JSON line per
    comparator on stdout."""
    import importlib

    from threadpoolctl import threadpool_limits

    from bench.speed.cases import embed as E
    from bench.speed.registry import RRef

    specs = [s for s in E.SPECS if cli.matches(match, f"{s.name} [{s.size}]")]
    libs = sorted({m for s in specs for m in s.libs} | {"sklearn.decomposition", "sklearn.manifold"})
    for m in libs:  # before any thread limit (cluster CL103)
        importlib.import_module(m)
    _warm(libs)
    sess = None
    for spec in specs:
        d = None
        for ref in spec.refs:
            jobs = _jobs(ref.label)
            if isinstance(ref, RRef):
                if (jobs or 1) != threads:
                    continue
            elif ref.threads == "fixed" and (jobs or 1) != threads:
                continue
            if d is None:
                d = spec.setup()
            load = round(os.getloadavg()[0], 1)
            try:
                if isinstance(ref, RRef):
                    if sess is None:
                        from bench.speed.rsession import RSession

                        sess = RSession()
                    r = _r_timed(sess, ref, d)
                else:
                    fn = ref.make(d)
                    with threadpool_limits(limits=threads), warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = _timed(fn)
            except Exception as e:  # noqa: BLE001 -- record and go on
                r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            r["load"] = load
            out = {"case": f"{spec.name} [{spec.size}]", "milestone": spec.milestone, "comparator": ref.label,
                   "threads": threads, **r}  # fmt: skip
            print("@@time@@ " + json.dumps(out), flush=True)
    if sess is not None:
        sess.close()


def timings(log, match, threads):
    """Every thread count in a child process of its own; the results merged into times.json."""
    path = DATA / "times.json"
    rec = {}
    if path.exists():
        rec = json.loads(path.read_text())
        rec.pop("_provenance", None)
    for t in threads:
        env = dict(os.environ)
        if t == 1:
            env["VECLIB_MAXIMUM_THREADS"] = "1"
        else:
            env.pop("VECLIB_MAXIMUM_THREADS", None)
        cmd = [sys.executable, str(ROOT / TOOL), "_time", "--threads", str(t)]
        for m in match:
            cmd += ["--match", m]
        log(f"time: {t} thread(s), load average {os.getloadavg()[0]:.1f}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env, cwd=ROOT)
        for line in p.stdout:
            if not line.startswith("@@time@@ "):
                continue
            r = json.loads(line[len("@@time@@ ") :])
            case = rec.setdefault(r["case"], {"milestone": r["milestone"], "comparators": {}})
            row = case["comparators"].setdefault(r["comparator"], {})
            row[str(t)] = {k: r[k] for k in ("s", "calls", "error", "load") if k in r}
            v = f"{r['s']:.3f} s ({r['calls']} calls)" if "s" in r else r["error"]
            log(f"  {r['case']} | {r['comparator']} | {t}: {v} (load {r['load']})")
            _write_times(path, rec)
        p.wait()
    _write_times(path, rec)
    log(f"{len(rec)} cases -> {path.relative_to(ROOT)}")


def _write_times(path, rec):
    import sklearn

    meta = {
        "tool": TOOL,
        "machine": MACHINE,
        "scikit-learn": sklearn.__version__,
        "rule": "seconds per call; a call of 3 s or more timed once, faster ones the best batch of harness.measure; "
        "1 thread: VECLIB_MAXIMUM_THREADS=1 and OpenMP limited to 1; 10 threads: Accelerate's default and OpenMP "
        "limited to 10; n_jobs / n_threads in a label is that comparator's own count; R on one thread otherwise",
    }
    with contextlib.suppress(ImportError):
        import openTSNE
        import umap

        meta["umap-learn"] = umap.__version__
        meta["openTSNE"] = openTSNE.__version__
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in rec.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default {', '.join(RECORDS)})")
    p.add_argument("--match", action="append", default=[], help="time: only the cases whose id matches (repeatable)")
    p.add_argument("--threads", default="1,10", help="time: thread counts (default 1,10)")
    args = p.parse_args(argv)
    threads = [int(t) for t in str(args.threads).split(",")]
    if args.sections == ["_time"]:
        time_child(threads[0], args.match)
        return 0
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS)})")
    log = Log()
    table = {"sklearn": sklearn_records, "scipy": scipy_records, "statsmodels": statsmodels_records,
             "r": r_records, "umap": umap_records, "opentsne": opentsne_records}  # fmt: skip
    for s in args.sections or RECORDS:
        if s == "time":
            timings(log, args.match, threads)
        else:
            log(f"{s}:")
            table[s](log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
