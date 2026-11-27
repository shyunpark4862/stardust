"""Milestone EM0's prototype measurements (docs/design/embed.md §5, the five "시제품" items) on the private
prototypes of csrc/embed/prototype.cpp (_core.embed._proto_*), written to tests/embed/data/prototype.json and quoted
by docs/design/embed.md §5.1.

    .venv/bin/python tools/embed_proto.py                  # every item
    .venv/bin/python tools/embed_proto.py gram bh          # only these items

Items:
  gram   (1) the Gram X^T X and the p x p eigenvalues: Accelerate (dsyrk, dsyevd) against the own fixed-order NEON
         Gram and a cyclic Jacobi; (2) the chunked dsyrk: row chunks fixed by size (2^16 rows, one call after
         another) against one chunk per sd thread called side by side (D-33)
  bh     (3) one Barnes-Hut repulsion step in 2-D: the sorted Morton tree of EM38 against an insertion tree like
         sklearn's, float64 against float32, build and walk apart; scikit-learn's own gradient (attraction and
         repulsion, float32) on the same points for scale
  umap   (4) the UMAP layout on one fuzzy graph (umap-learn's fuzzy_simplicial_set on exact neighbours, its
         spectral start, a and b): umap-learn's sequential SGD, the synchronous batch with SGD and with Adam
         (EM40); trustworthiness (scikit-learn, 15 neighbours, on a fixed sample of 5000 points) and time; also
         umap-learn's and uwot's own fits of the same data for the quality scale; the item umap_quality (run alone:
         tools/embed_proto.py umap_quality) is its quality part on noisy digits (kept 15-neighbourhoods,
         trustworthiness at 15 and 100 neighbours, 50 to 500 epochs) with the 1, 3, 10 thread bit check
  knn    (5) the exact k-nearest neighbours (sd.spatial.kneighbors, k = 15 + itself) for n = 10^4 to 10^6 and
         p = 10, 50, 784, and pynndescent's approximate neighbours with their recall, for §9's second question

Threads: every configuration runs in a child process of its own (Accelerate reads VECLIB_MAXIMUM_THREADS once per
process): "1" is sd at 1 thread with VECLIB_MAXIMUM_THREADS=1, "10" sd at 10 threads with Accelerate's default.
Times are the best of a few runs (one run for calls over 3 s). No download; the data are tests/embed/samples.py's
formulas.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != ROOT / "tools"]  # tools/coverage.py
sys.path.insert(0, str(ROOT))

OUT = ROOT / "tests" / "embed" / "data" / "prototype.json"
ITEMS = ("gram", "bh", "umap", "knn")  # umap_quality: only the quality part of umap


def best(fn, reps=3):
    t0 = time.perf_counter()
    r = fn()
    t = time.perf_counter() - t0
    if t > 3.0:
        return t, r
    for _ in range(reps - 1):
        t0 = time.perf_counter()
        r = fn()
        t = min(t, time.perf_counter() - t0)
    return t, r


def S():
    from tests.embed import samples

    return samples


# ---------------------------------------------------------------- the children


def child_gram(threads):
    import stardust as sd

    sd.set_num_threads(threads)
    P = sd._core.embed
    out = []
    for n, p in ((10**5, 20), (10**5, 100), (10**6, 100)):
        g = S().rng("proto", "gram", n, p)
        X = g.standard_normal((n, p))
        X -= X.mean(axis=0)
        sX = sd.asarray(X)
        ref = None
        row = {"n": n, "p": p}
        for form, name in ((0, "dsyrk"), (1, "neon"), (2, "dsyrk_size_chunks"), (3, "dsyrk_thread_chunks")):
            t, G = best(lambda f=form: P._proto_gram(sX, f))
            G = np.asarray(G)
            ref = G if ref is None else ref
            row[name] = t
            row[name + "_maxdiff"] = float(np.abs(G - ref).max() / np.abs(ref).max())
        out.append(row)
    eig = []
    for p in (100, 500):
        g = S().rng("proto", "eigh", p)
        A = g.standard_normal((4 * p, p))
        C = sd.asarray(A.T @ A)
        t0, w0 = best(lambda: P._proto_eigh(C, 0))
        t1, w1 = best(lambda: P._proto_eigh(C, 1))
        eig.append(
            {
                "p": p,
                "dsyevd": t0,
                "jacobi": t1,
                "maxdiff": float(np.abs(np.asarray(w0) - np.asarray(w1)).max() / np.abs(np.asarray(w0)).max()),
            }
        )
    return {"gram": out, "eigh": eig}


def _tsne_like(n):
    """2-D points like a t-SNE layout midway: 10 Gaussian clusters of different spreads, scale ~30."""
    g = S().rng("proto", "bh", n)
    lab = g.integers(0, 10, n)
    centres = 25.0 * g.standard_normal((10, 2))
    return centres[lab] + (1.0 + lab[:, None] / 5.0) * g.standard_normal((n, 2))


def child_bh(threads):
    import stardust as sd

    sd.set_num_threads(threads)
    P = sd._core.embed
    out = []
    for n in (2 * 10**4, 10**5):
        Y = _tsne_like(n)
        sY = sd.asarray(Y)
        row = {"n": n}
        F0 = None
        for form, tree in ((0, "sorted"), (1, "insertion")):
            for single in (False, True):
                runs = [P._proto_bh(sY, form, 0.5, single) for _ in range(3)]
                build = min(r[2] for r in runs) / 1e9
                walk = min(r[3] for r in runs) / 1e9
                F = np.asarray(runs[0][0])
                F0 = F if F0 is None else F0
                key = f"{tree}_{'f32' if single else 'f64'}"
                row[key + "_build"] = build
                row[key + "_walk"] = walk
                row[key + "_vs_sorted_f64"] = float(np.abs(F - F0).max() / np.abs(F0).max())
                row[key + "_same_bits_3runs"] = all(np.array_equal(np.asarray(r[0]), F) for r in runs)
        # scikit-learn's gradient (attraction on a 90-neighbour P plus the BH repulsion; float32) at this thread count
        try:
            from scipy.sparse import csr_matrix
            from sklearn.manifold import _barnes_hut_tsne
            from sklearn.neighbors import NearestNeighbors

            k = 90
            nn = NearestNeighbors(n_neighbors=k).fit(Y)
            ind = nn.kneighbors(return_distance=False)
            vals = np.full(n * k, 1.0 / (n * k), np.float32)
            Pm = csr_matrix((vals, ind.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
            Pm = Pm + Pm.T
            Pm.sort_indices()
            pos = Y.astype(np.float32)
            forces = np.zeros_like(pos)

            def grad():
                forces[:] = 0
                return _barnes_hut_tsne.gradient(
                    Pm.data.astype(np.float32),
                    pos,
                    Pm.indices.astype(np.int64),
                    Pm.indptr.astype(np.int64),
                    forces,
                    0.5,
                    2,
                    0,
                    dof=1,
                    compute_error=0,
                    num_threads=threads,
                )

            t, _ = best(grad)
            row["sklearn_gradient"] = t
        except Exception as e:  # noqa: BLE001  (recorded, not fatal)
            row["sklearn_gradient_error"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return {"bh": out}


def _umap_data(n):
    g = S().rng("proto", "umap", n)
    X, lab = S().digits(n, g)
    X = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return np.ascontiguousarray(U[:, :50] * s[:50]), lab


def _umap_graph(X, k=15):
    import umap.umap_ as uu
    from umap.spectral import spectral_layout

    import stardust as sd

    d, i = sd.spatial.kneighbors(sd.asarray(X), sd.asarray(X), k=k)
    knn_i, knn_d = np.asarray(i), np.asarray(d)
    graph, _, _ = uu.fuzzy_simplicial_set(X, k, None, "euclidean", knn_indices=knn_i, knn_dists=knn_d)
    graph = graph.tocoo()
    graph.sum_duplicates()
    rs = np.random.RandomState(0)
    init = spectral_layout(X, graph, 2, rs)
    init = 10.0 * init / np.abs(init).max() + rs.normal(scale=1e-4, size=init.shape)
    a, b = uu.find_ab_params(1.0, 0.1)
    return graph.tocsr(), init, float(a), float(b), knn_i


def _edges(graph, n_epochs):
    import umap.umap_ as uu

    g = graph.tocoo()
    g.data[g.data < g.data.max() / float(n_epochs)] = 0.0
    g.eliminate_zeros()
    order = np.lexsort((g.col, g.row))
    head, tail, w = g.row[order].astype(np.int64), g.col[order].astype(np.int64), g.data[order]
    eps = uu.make_epochs_per_sample(w, n_epochs)
    keep = eps > 0
    return head[keep], tail[keep], np.ascontiguousarray(eps[keep], float)


def _trust(X, Z, sample):
    from sklearn.manifold import trustworthiness

    return float(trustworthiness(X[sample], Z[sample], n_neighbors=15))


def child_umap(threads, n, epochs, forms):
    import stardust as sd

    sd.set_num_threads(threads)
    X, _ = _umap_data(n)
    t0 = time.perf_counter()
    graph, init, a, b, _ = _umap_graph(X)
    t_graph = time.perf_counter() - t0
    sample = S().rng("proto", "umap-sample", n).choice(n, size=min(n, 5000), replace=False)
    out = {"n": n, "threads": threads, "a": a, "b": b, "graph_time": t_graph, "edges": int(graph.nnz), "runs": []}
    out["trust_init"] = _trust(X, init, sample)
    for E in epochs:
        head, tail, eps = _edges(graph, E)
        sh, st, se, sY = sd.asarray(head), sd.asarray(tail), sd.asarray(eps), sd.asarray(np.ascontiguousarray(init))
        for form in forms:
            if form == 0 and threads != 1:
                continue
            t0 = time.perf_counter()
            Y = np.asarray(sd._core.embed._proto_umap(sh, st, se, sY, E, a, b, form, 7, 11, 5, 1.0, 1.0))
            t = time.perf_counter() - t0
            Y2 = np.asarray(sd._core.embed._proto_umap(sh, st, se, sY, E, a, b, form, 7, 11, 5, 1.0, 1.0))
            out["runs"].append(
                {
                    "form": ["sequential_sgd", "batch_sgd", "batch_adam"][form],
                    "n_epochs": E,
                    "time": t,
                    "trust": _trust(X, Y, sample),
                    "same_bits_twice": bool(np.array_equal(Y, Y2)),
                    "Y_digest": S().digest(Y),
                }
            )
    return out


def _knn_kept(X, Z, k=15):
    """The mean fraction of each point's k nearest neighbours in X that are among its k nearest in Z."""
    import stardust as sd

    _, a = sd.spatial.kneighbors(sd.asarray(X), sd.asarray(X), k=k + 1)
    _, b = sd.spatial.kneighbors(sd.asarray(Z), sd.asarray(Z), k=k + 1)
    a, b = np.asarray(a)[:, 1:], np.asarray(b)[:, 1:]
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))


def child_umap_quality(n):
    """The forms on a harder set (noisy digits: overlapping classes) at few and many epochs, judged by
    trustworthiness at 15 and 100 neighbours and the kept 15-neighbourhoods; and the thread check of the batch forms
    (the same start at 1 and 10 threads in one process: equal bits)."""
    import stardust as sd

    g = S().rng("proto", "umap-noisy", n)
    X, _ = S().digits(n, g, noise=6.0)
    X = X - X.mean(axis=0)
    graph, init, a, b, _ = _umap_graph(np.ascontiguousarray(X))
    sample = S().rng("proto", "umap-sample", n).choice(n, size=min(n, 5000), replace=False)
    from sklearn.manifold import trustworthiness

    out = {"n": n, "data": "digits noise 6", "runs": [], "thread_check": []}
    out["init"] = {"trust15": _trust(X, init, sample), "kept15": _knn_kept(X, init)}
    for E in (50, 100, 200, 500):
        head, tail, eps = _edges(graph, E)
        args = (sd.asarray(head), sd.asarray(tail), sd.asarray(eps), sd.asarray(np.ascontiguousarray(init)))
        for form in (0, 1, 2):
            sd.set_num_threads(1 if form == 0 else 10)
            t0 = time.perf_counter()
            Y = np.asarray(sd._core.embed._proto_umap(*args, E, a, b, form, 7, 11, 5, 1.0, 1.0))
            t = time.perf_counter() - t0
            out["runs"].append(
                {
                    "form": ["sequential_sgd", "batch_sgd", "batch_adam"][form],
                    "n_epochs": E,
                    "threads": 1 if form == 0 else 10,
                    "time": t,
                    "trust15": _trust(X, Y, sample),
                    "trust100": float(trustworthiness(X[sample], Y[sample], n_neighbors=100)),
                    "kept15": _knn_kept(X, Y),
                }
            )
            if E == 200 and form > 0:
                sd.set_num_threads(1)
                Y1 = np.asarray(sd._core.embed._proto_umap(*args, E, a, b, form, 7, 11, 5, 1.0, 1.0))
                sd.set_num_threads(3)
                Y3 = np.asarray(sd._core.embed._proto_umap(*args, E, a, b, form, 7, 11, 5, 1.0, 1.0))
                out["thread_check"].append(
                    {
                        "form": out["runs"][-1]["form"],
                        "same_1_3_10": bool(np.array_equal(Y1, Y) and np.array_equal(Y3, Y)),
                    }
                )
    import warnings

    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, kw in (
            ("umap_learn_seeded", {"random_state": 0}),
            ("umap_learn_adam", {"random_state": 0, "compatibility_layout": False, "n_jobs": 10}),
        ):
            t0 = time.perf_counter()
            Z = umap.UMAP(**kw).fit_transform(X)
            out[label] = {
                "time": time.perf_counter() - t0,
                "trust15": _trust(X, Z, sample),
                "trust100": float(trustworthiness(X[sample], Z[sample], n_neighbors=100)),
                "kept15": _knn_kept(X, Z),
            }
    return out


def child_umap_ref(n):
    """umap-learn's own fit (seeded: one thread, the 0.5 path) and uwot's batch fit on the same data."""
    import warnings

    import umap

    X, _ = _umap_data(n)
    sample = S().rng("proto", "umap-sample", n).choice(n, size=min(n, 5000), replace=False)
    out = {"n": n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        Z = umap.UMAP(random_state=0).fit_transform(X)
        out["umap_learn_seeded"] = {"time": time.perf_counter() - t0, "trust": _trust(X, Z, sample)}
        try:
            t0 = time.perf_counter()
            Z = umap.UMAP(random_state=0, compatibility_layout=False, n_jobs=10).fit_transform(X)
            out["umap_learn_adam"] = {"time": time.perf_counter() - t0, "trust": _trust(X, Z, sample)}
        except TypeError as e:
            out["umap_learn_adam_error"] = str(e)
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"embed_proto_uwot_{n}.bin"
    X.astype("<f8").tofile(tmp)
    out_path = tmp.with_suffix(".out")
    code = (
        f"library(uwot); x <- matrix(readBin('{tmp}', 'double', {X.size}, endian='little'), ncol=50, byrow=TRUE);"
        f"set.seed(1); t <- system.time(z <- umap(x, batch=TRUE, n_threads=10, n_sgd_threads=10, min_dist=0.1));"
        f"writeBin(c(t[['elapsed']], as.vector(t(z))), '{out_path}', endian='little')"
    )
    r = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    if r.returncode == 0:
        v = np.fromfile(out_path, "<f8")
        Z = v[1:].reshape(n, 2)
        out["uwot_batch"] = {"time": float(v[0]), "trust": _trust(X, Z, sample), "threads": 10}
    else:
        out["uwot_error"] = r.stderr[-500:]
    for f in (tmp, out_path):
        f.unlink(missing_ok=True)
    return out


def child_knn(threads):
    import stardust as sd

    sd.set_num_threads(threads)
    out = []
    budget = time.perf_counter() + 1500
    for p, ns in ((10, (10**4, 10**5, 10**6)), (50, (10**4, 10**5, 10**6)), (784, (10**4, 10**5))):
        for n in ns:
            if threads == 1 and n * n * p > 10**4 * 10**4 * 784 * 2 and p > 16:
                continue  # one thread: the exhaustive paths above ~1.6e11 flops are left to 10 threads
            if time.perf_counter() > budget:
                out.append({"n": n, "p": p, "skipped": "time budget"})
                continue
            g = S().rng("proto", "knn", n, p)
            r = min(p, 20)
            X = (g.standard_normal((n, r)) @ g.standard_normal((r, p))) + 0.1 * g.standard_normal((n, p))
            sX = sd.asarray(X)
            t, (d, i) = best(lambda: sd.spatial.kneighbors(sX, sX, k=16), reps=2)
            row = {"n": n, "p": p, "exact": t}
            if threads == 10 and n >= 10**5:
                try:
                    import pynndescent

                    t0 = time.perf_counter()
                    idx = pynndescent.NNDescent(X, n_neighbors=16, random_state=None, n_jobs=10)
                    ai, _ = idx.neighbor_graph
                    row["pynndescent"] = time.perf_counter() - t0
                    ei = np.asarray(i)
                    sample = np.arange(0, n, max(1, n // 2000))
                    hits = sum(len(set(ai[s]) & set(ei[s])) for s in sample)
                    row["pynndescent_recall"] = hits / (16 * len(sample))
                except Exception as e:  # noqa: BLE001
                    row["pynndescent_error"] = f"{type(e).__name__}: {e}"
            out.append(row)
    return {"knn": out}


CHILDREN = {"gram": child_gram, "bh": child_bh, "knn": child_knn}


def run_child(item, threads, extra=()):
    env = dict(os.environ)
    if threads == 1:
        env["VECLIB_MAXIMUM_THREADS"] = "1"
    else:
        env.pop("VECLIB_MAXIMUM_THREADS", None)
    cmd = [sys.executable, "-I", __file__, "--child", item, str(threads), *map(str, extra)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{item} at {threads} threads failed:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools/embed_proto.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("items", nargs="*", metavar="ITEM")
    ap.add_argument("--child", nargs="+", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        item, threads = a.child[0], int(a.child[1])
        if item == "umap":
            n = int(a.child[2])
            epochs = [int(v) for v in a.child[3].split(",")]
            forms = [int(v) for v in a.child[4].split(",")]
            res = child_umap(threads, n, epochs, forms)
        elif item == "umap_ref":
            res = child_umap_ref(int(a.child[2]))
        elif item == "umap_quality":
            res = child_umap_quality(int(a.child[2]))
        else:
            res = CHILDREN[item](threads)
        print(json.dumps(res))
        return 0
    items = a.items or list(ITEMS)
    rec = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec["meta"] = {
        "tool": "tools/embed_proto.py",
        "machine": "Apple M4 Pro (14 cores), " + platform.platform(),
        "date": time.strftime("%Y-%m-%d"),
        "threads": "1: sd 1 thread, VECLIB_MAXIMUM_THREADS=1; 10: sd 10 threads, Accelerate's default",
    }
    for item in items:
        print(f"[{item}]", flush=True)
        if item == "umap":
            res = {
                "n10000": [
                    run_child("umap", 1, (10**4, "100,200,500", "0,1,2")),
                    run_child("umap", 10, (10**4, "500", "1,2")),
                ],
                "n70000": [
                    run_child("umap", 1, (7 * 10**4, "200", "0,1,2")),
                    run_child("umap", 10, (7 * 10**4, "200", "1,2")),
                ],
                "reference": [run_child("umap_ref", 10, (10**4,)), run_child("umap_ref", 10, (7 * 10**4,))],
                "quality": run_child("umap_quality", 10, (10**4,)),
            }
        elif item == "umap_quality":
            rec.setdefault("umap", {})["quality"] = run_child("umap_quality", 10, (10**4,))
            OUT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
            print(json.dumps(rec["umap"]["quality"], indent=1), flush=True)
            continue
        else:
            res = {str(t): run_child(item, t) for t in (1, 10)}
        rec[item] = res
        print(json.dumps(res, indent=1)[:4000], flush=True)
        OUT.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
