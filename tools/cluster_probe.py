"""Probes of sd.cluster's foundations and of the libraries it is compared with (docs/design/cluster.md §2, §3, §4, §5,
§7.4). The records the tests read go to tests/cluster/data/; the timings go to the terminal and to
build/logs/cluster_probe/<stamp>.log, and docs/design/cluster.md quotes them.

    .venv/bin/python tools/cluster_probe.py                  # every section
    .venv/bin/python tools/cluster_probe.py proto scipy r    # only these sections
    .venv/bin/python tools/cluster_probe.py proto.mst        # one prototype measurement (the others' records kept)

Sections (positional; default all):
  proto    the prototypes of csrc/cluster/prototype.cpp (_core.cluster._proto_*) against the references of
           tests/cluster/ref (tools/cluster_ref.py must have run) -> prototype.json:
             distance  k-means' squared distances in the difference form (spatial's pair kernel) against the
                       expanded form with Accelerate's dgemm (scikit-learn's): time at 1 and 10 threads, and the
                       labels against the exact ones (the references' label_f, and sets shifted by 10^4 to 10^12)
             bounds    Hamerly's bounds with the rounding margin c (p + 4) u, c = 0, 1, 4, 16: the same bits as
                       Lloyd on every reference set, the distances skipped, the time on 10^5 points
             mst       Prim without a matrix, Boruvka on a KD tree and Boruvka by a scan: the same tree, and the
                       times by n and p (where Boruvka starts to win)
             nnchain   the nearest-neighbour chain against the exact primitive trees (does a scan meet an exact
                       tie whenever the trees differ; do equal heights flag it), and the cost of the check
             heights   heights recomputed by the definition (average: the |A||B| distances; ward: the
                       centroids) against the exact ones and against scipy's Lance-Williams heights, in units
                       of the draft contracts; the cost against the chain
             tree_heights  the Lance-Williams recurrence evaluated on the final tree (path-free), every
                       linkage, in ULPs of the exact heights, beside scipy's
             centers   k-means centres summed in chunks of 4096 points against the exact means
  scipy    scipy 1.18.1's hierarchy and vq on the test data and at the edges the design departs from -> scipy.json
  sklearn  scikit-learn 1.9.1's KMeans, DBSCAN, HDBSCAN, OPTICS, GaussianMixture and indices -> sklearn.json
  r        R 4.6.2's stats (hclust, cutree, kmeans) and cluster 2.1.8.3 (pam, clara, fanny, agnes, diana,
           silhouette, clusGap, daisy) through bench/accuracy/rsession.py -> r.json
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
CHAIN = {"complete": 1, "average": 2, "weighted": 3, "ward": 4}


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


# ---------------------------------------------------------------- proto: distances


def exact_labels(X, C):
    """The nearest centre of every point by exact squared distance (the smaller centre on a tie), and the gaps."""
    from fractions import Fraction

    Cq = [[Fraction(v) for v in c] for c in C]
    labs, gaps = [], []
    for x in X:
        xq = [Fraction(v) for v in x]
        ds = [sum((a - b) ** 2 for a, b in zip(xq, c)) for c in Cq]
        j = min(range(len(ds)), key=lambda t: (ds[t], t))
        rest = sorted(ds[t] for t in range(len(ds)) if t != j)
        labs.append(j)
        gaps.append(float((rest[0] - ds[j]) / ds[j]) if rest and ds[j] > 0 else math.inf)
    return np.array(labs), np.array(gaps)


def proto_distance(log, rec):
    C = P()
    g = np.random.Generator(np.random.PCG64(20261124))
    log("distance: k-means' squared distances, difference form (spatial's kernel) vs expanded (Accelerate dgemm)")
    log(f"  load average {os.getloadavg()[0]:.1f}; ms, best of three; Accelerate's own threads in 'expanded'")
    table = {}
    for n, p, k in ((10**5, 2, 10), (10**5, 10, 10), (10**5, 10, 100), (10**5, 50, 10), (10**5, 50, 100),
                    (10**5, 100, 100), (2 * 10**4, 300, 100)):  # fmt: skip
        X = g.standard_normal((n, p))
        Cn = X[g.choice(n, k, replace=False)].copy()
        sX, sC = sda(X), sda(Cn)
        t = {}
        for th in (1, 10):
            threads(th)
            t[f"diff_{th}"] = best(lambda: C._proto_assign(sX, sC, 0))
            t[f"filtered_{th}"] = best(lambda: C._proto_assign(sX, sC, 3))
        threads(1)
        t["expanded"] = best(lambda: C._proto_assign(sX, sC, 1))
        t["ratio_1"] = t["diff_1"] / t["expanded"]
        t["ratio_10"] = t["diff_10"] / t["expanded"]
        t["filtered_ratio_1"] = t["filtered_1"] / t["expanded"]
        t["filtered_ratio_10"] = t["filtered_10"] / t["expanded"]
        lab0 = arr(C._proto_assign(sX, sC, 0))
        t["tile_equals_scalar"] = bool(np.array_equal(lab0, arr(C._proto_assign(sX, sC, 2))))
        t["filtered_equals_diff"] = bool(np.array_equal(lab0, arr(C._proto_assign(sX, sC, 3))))
        t["filtered_candidates_per_point"] = C._proto_candidates() / n
        table[f"n{n}p{p}k{k}"] = t
        log(
            f"  n={n:>6} p={p:>3} k={k:>3}: diff {t['diff_1']:8.2f} (1) {t['diff_10']:7.2f} (10), expanded "
            f"{t['expanded']:7.2f}, filtered {t['filtered_1']:7.2f} (1) {t['filtered_10']:7.2f} (10); "
            f"/expanded: diff {t['ratio_1']:.2f} {t['ratio_10']:.2f}, filtered {t['filtered_ratio_1']:.2f} "
            f"{t['filtered_ratio_10']:.2f}; candidates {t['filtered_candidates_per_point']:.2f}, same "
            f"{t['tile_equals_scalar'] and t['filtered_equals_diff']}"
        )
    threads(10)
    rec["distance/time"] = table
    # accuracy: the labels at the references' rounded centres (label_f) and on shifted blobs
    R = RD().ref("kmeans")
    bad = {"diff": 0, "expanded": 0, "filtered": 0, "points": 0, "near": 0}
    for sid in R.group_inputs["set"].tolist():
        gi = RD().group_of(R, sid)
        cf = R.group_inputs.get("centers_f")
        if cf is None:
            continue
        Cf = cf[gi] if not isinstance(cf, list) else cf[gi]
        if Cf is None or np.size(Cf) == 0 or np.isnan(np.asarray(Cf, float)).all():
            continue
        s = S().points(sid)
        Cf = np.asarray(Cf, float).reshape(-1, s.p)
        pos, idx = RD().at(R, "label_f", sid, 0)
        m = R.methods["label_f"]
        want = np.empty(s.n)
        want[idx] = m.hi[pos]
        gap = np.empty(s.n)
        gap[idx] = m.fields["gap"][pos]
        far = gap > RD().NEAR_TIE
        for name, form in (("diff", 0), ("expanded", 1), ("filtered", 3)):
            got = arr(C._proto_assign(sda(s.X), sda(Cf), form))
            bad[name] += int(np.sum((got != want) & far))
        bad["points"] += int(far.sum())
        bad["near"] += int((~far).sum())
    rec["distance/labels_at_reference_centres"] = bad
    log(f"  labels at the references' centres (exact, gap > 2^-40): {bad}")
    shifted = {}
    for e in (4, 8, 12):
        base = np.vstack([g.standard_normal((700, 2)) + c for c in ([0.0, 0.0], [3.0, 0.0], [0.0, 3.0])])
        X = 10.0**e + base
        Cn = 10.0**e + np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]) + 0.01 * g.standard_normal((3, 2))
        want, gaps = exact_labels(X, Cn)
        row = {"points": len(X)}
        for name, form in (("diff", 0), ("expanded", 1), ("filtered", 3)):
            got = arr(C._proto_assign(sda(X), sda(Cn), form))
            row[name] = int(np.sum(got != want))
        row["filtered_candidates_per_point"] = C._proto_candidates() / len(X)
        from sklearn.cluster import KMeans

        km = KMeans(3, init=Cn, n_init=1, max_iter=1)
        km.fit(Cn)
        km.cluster_centers_ = Cn
        row["sklearn_predict"] = int(np.sum(km.predict(X) != want))
        shifted[f"1e{e}"] = row
    rec["distance/shifted_blobs"] = shifted
    log(f"  shifted blobs (2100 points, 3 centres; wrong labels against the exact): {shifted}")


# ---------------------------------------------------------------- proto: bounds


def proto_bounds(log, rec):
    C = P()
    R = RD().ref("kmeans")
    log("bounds: Hamerly with the rounding margin c (p + 4) u against Lloyd (same labels and centre bits?)")
    summary = {c: {"sets": 0, "equal": 0, "skipped": 0.0, "work": 0.0} for c in (0, 1, 4, 16)}
    for sid in R.group_inputs["set"].tolist():
        gi = RD().group_of(R, sid)
        init = R.group_inputs.get("init")
        if init is None:
            continue
        I = init[gi]
        if I is None or np.size(I) == 0 or np.isnan(np.asarray(I, float)).all():
            continue
        s = S().points(sid)
        I = np.asarray(I, float).reshape(-1, s.p)
        if s.w is not None:
            continue
        lab0, cen0, it0, nd0 = C._proto_kmeans(sda(s.X), sda(I), 300, 0, 1.0)
        for c in summary:
            lab, cen, it, nd = C._proto_kmeans(sda(s.X), sda(I), 300, 1, float(c))
            eq = np.array_equal(arr(lab), arr(lab0)) and np.array_equal(arr(cen), arr(cen0)) and it == it0
            summary[c]["sets"] += 1
            summary[c]["equal"] += int(eq)
            summary[c]["skipped"] += 1.0 - nd / nd0
            summary[c]["work"] += 1.0
    for v in summary.values():
        v["skipped_mean"] = v.pop("skipped") / max(v.pop("work"), 1)
    rec["bounds/reference_sets"] = summary
    log(f"  reference sets: {summary}")
    # near ties: a lattice and points a ulp off the bisectors
    g = np.random.Generator(np.random.PCG64(7))
    ties = {}
    L = np.array([[i, j] for i in range(60) for j in range(60)], float)
    cases = {
        "lattice60": (L, L[[0, 59, 3540, 3599, 1830]]),
        "bisector": (
            np.column_stack([np.nextafter(np.zeros(4000), g.choice([-1.0, 1.0], 4000)), g.standard_normal(4000)]),
            np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 5.0]]),
        ),
        "blobs_noise": (
            np.vstack([g.standard_normal((3000, 3)) + c for c in np.eye(3) * 3])
            * (1 + 1e-15 * g.standard_normal((9000, 1))),
            np.eye(3) * 3 + 1e-3,
        ),
    }
    for name, (X, I) in cases.items():
        lab0, cen0, it0, nd0 = C._proto_kmeans(sda(X), sda(I), 300, 0, 1.0)
        row = {}
        for c in (0, 1):
            lab, cen, it, nd = C._proto_kmeans(sda(X), sda(I), 300, 1, float(c))
            row[f"c{c}"] = {
                "equal": bool(np.array_equal(arr(lab), arr(lab0)) and np.array_equal(arr(cen), arr(cen0))),
                "labels_differ": int(np.sum(arr(lab) != arr(lab0))),
                "skipped": 1.0 - nd / nd0,
            }
        ties[name] = row
    rec["bounds/near_ties"] = ties
    log(f"  near ties: {ties}")
    # time on 10^5 points
    tt = {}
    g = np.random.Generator(np.random.PCG64(11))
    for p, k in ((2, 10), (10, 10), (2, 100), (10, 100)):
        centers = 4.0 * g.standard_normal((k, p))
        X = np.repeat(centers, 10**5 // k, axis=0) + g.standard_normal((10**5, p))
        X = X[g.permutation(len(X))]
        I = X[:k].copy()
        row = {}
        for th in (1, 10):
            threads(th)
            r0 = C._proto_kmeans(sda(X), sda(I), 300, 0, 1.0)
            r1 = C._proto_kmeans(sda(X), sda(I), 300, 1, 1.0)
            row[f"lloyd_{th}"] = best(lambda: C._proto_kmeans(sda(X), sda(I), 300, 0, 1.0), 1)
            row[f"hamerly_{th}"] = best(lambda: C._proto_kmeans(sda(X), sda(I), 300, 1, 1.0), 1)
            row["equal"] = bool(np.array_equal(arr(r0[1]), arr(r1[1])) and np.array_equal(arr(r0[0]), arr(r1[0])))
            row["iterations"] = int(r0[2])
            row["skipped"] = 1.0 - r1[3] / r0[3]
        tt[f"p{p}k{k}"] = row
        log(
            f"  n=1e5 p={p} k={k} ({row['iterations']} iterations): Lloyd {row['lloyd_1']:.0f} / "
            f"{row['lloyd_10']:.0f} ms,"
            f" Hamerly {row['hamerly_1']:.0f} / {row['hamerly_10']:.0f} ms (1 / 10 threads), skipped "
            f"{row['skipped']:.2f}, same bits {row['equal']}"
        )
    threads(10)
    rec["bounds/time"] = tt


# ---------------------------------------------------------------- proto: MST


def proto_mst(log, rec):
    C = P()
    g = np.random.Generator(np.random.PCG64(12))
    log("mst: Prim without a matrix (0), Boruvka on a KD tree (1), Boruvka by a scan (2); ms at 1 / 10 threads")
    same = {}
    for p in (2, 5, 16):
        X = g.random((3000, p))
        outs = [C._proto_mst(sda(X), m) for m in (0, 1, 2)]
        same[f"p{p}"] = all(
            np.array_equal(arr(outs[0][0]), arr(o[0])) and np.array_equal(arr(outs[0][1]), arr(o[1])) for o in outs[1:]
        )
    X = np.array([[i, j] for i in range(40) for j in range(40)], float)
    outs = [C._proto_mst(sda(X), m) for m in (0, 1, 2)]
    same["lattice40"] = all(np.array_equal(arr(outs[0][0]), arr(o[0])) for o in outs[1:])
    rec["mst/same_tree"] = same
    log(f"  the same edges and weights: {same}")
    times = {}
    for p in (2, 3, 5, 8, 16):
        for n in (10**3, 10**4, 3 * 10**4, 10**5):
            X = g.random((n, p))
            row = {}
            for th in (1, 10):
                threads(th)
                if n <= 3 * 10**4:
                    row[f"prim_{th}"] = best(lambda: C._proto_mst(sda(X), 0), 1 if n > 10**4 else 3)
                if not (n == 10**5 and p == 16):
                    row[f"boruvka_{th}"] = best(lambda: C._proto_mst(sda(X), 1), 1 if n > 10**4 else 3)
            times[f"p{p}n{n}"] = row
            log(f"  p={p:>2} n={n:>6}: " + ", ".join(f"{k} {v:.1f}" for k, v in row.items()))
    threads(10)
    rec["mst/time"] = times


# ---------------------------------------------------------------- proto: NN-chain


def _lexsorted(X):
    order = np.lexsort(tuple(X[:, j] for j in range(X.shape[1] - 1, -1, -1)))
    return X[order], order


def proto_nnchain(log, rec):
    C = P()
    R = RD().ref("hierarchy")
    log("nnchain: the chain with the (distance, index) neighbour against the exact primitive trees")
    rows = {}
    totals = {"cases": 0, "same": 0, "differ_flagged": 0, "differ_unflagged": 0, "differ_unflagged_near": 0,
              "scan_tie": 0, "equal_heights": 0}  # fmt: skip
    for sid in R.group_inputs["set"].tolist():
        s = S().points(sid)
        Xs, order = _lexsorted(s.X)  # index order = the key order (CL7)
        for method, code in CHAIN.items():
            Zref, gaps, _ = RD().tree(R, method, sid)
            Z, tie = C._proto_nnchain(sda(Xs), code, 1)
            Z = arr(Z)
            mapped = np.array(Z)
            mapped[:, :2] = np.where(
                mapped[:, :2] < s.n, order[np.minimum(mapped[:, :2], s.n - 1).astype(int)], mapped[:, :2]
            )
            same = RD().merges(mapped, s.n) == RD().merges(Zref, s.n)
            eqh = len(np.unique(Z[:, 2])) < len(Z)
            near = bool(np.min(gaps) <= RD().NEAR_TIE) if len(gaps) else False
            totals["cases"] += 1
            totals["same"] += int(same)
            totals["scan_tie"] += int(tie)
            totals["equal_heights"] += int(eqh)
            if not same:
                if tie:
                    totals["differ_flagged"] += 1
                else:
                    totals["differ_unflagged"] += 1
                    totals["differ_unflagged_near"] += int(near)
            if not same or tie:
                rows[f"{sid}/{method}"] = {"same": same, "scan_tie": int(tie), "equal_heights": eqh, "near_tie": near}
    rec["nnchain/against_primitive"] = {"totals": totals, "cases": rows}
    log(f"  {totals}")
    for k, v in rows.items():
        if not v["same"]:
            log(f"    differs: {k} {v}")
    g = np.random.Generator(np.random.PCG64(13))
    X = g.standard_normal((5000, 5))
    import scipy.cluster.hierarchy as sch

    tt = {}
    sX = sda(X)
    for method, code in CHAIN.items():
        t0, t1, ts = math.inf, math.inf, math.inf
        for _ in range(3):  # interleaved: the first call of a size pays for its pages
            t0 = min(t0, best(lambda: C._proto_nnchain(sX, code, 0), 1))
            t1 = min(t1, best(lambda: C._proto_nnchain(sX, code, 1), 1))
            ts = min(ts, best(lambda: sch.linkage(X, method), 1))
        tt[method] = {"chain": t0, "chain_checked": t1, "overhead": t1 / t0 - 1, "scipy": ts}
        log(
            f"  n=5000 p=5 {method}: chain {t0:.0f} ms, with the tie check {t1:.0f} ms "
            f"({100 * (t1 / t0 - 1):+.1f}%), scipy {ts:.0f} ms"
        )
    rec["nnchain/time"] = tt


# ---------------------------------------------------------------- proto: heights


def proto_heights(log, rec):
    import scipy.cluster.hierarchy as sch

    C = P()
    R = RD().ref("hierarchy")
    D = RD()
    log(
        "heights: recomputed by the definition vs scipy's Lance-Williams, against the exact "
        "(units of the draft bounds with constant 1)"
    )
    out = {}
    for method, code in (("average", 2), ("ward", 4)):
        worst = {"recomputed": 0.0, "lance_williams": 0.0, "recomputed_ulp": 0.0, "lance_williams_ulp": 0.0}
        where = {}
        for sid in R.group_inputs["set"].tolist():
            s = S().points(sid)
            if s.n < 2:
                continue
            Zref, gaps, lo = D.tree(R, method, sid)
            if np.min(gaps) <= D.NEAR_TIE:
                continue  # a tie: scipy's tree may be another one
            h_exact = Zref[:, 2] + lo
            rec_h = arr(C._proto_heights(sda(s.X), sda(Zref), code))
            Zs = sch.linkage(s.X, method)
            if D.merges(Zs, s.n) != D.merges(Zref, s.n):
                lw = None
            else:
                key = {}
                mem = {i: frozenset([i]) for i in range(s.n)}
                for i, row in enumerate(Zs):
                    mem[s.n + i] = mem[int(row[0])] | mem[int(row[1])]
                    key[mem[s.n + i]] = row[2]
                memr = {i: frozenset([i]) for i in range(s.n)}
                lw = np.empty(len(Zref))
                for i, row in enumerate(Zref):
                    memr[s.n + i] = memr[int(row[0])] | memr[int(row[1])]
                    lw[i] = key[memr[s.n + i]]
            sizes = {i: 1 for i in range(s.n)}
            cents = {i: s.X[i] for i in range(s.n)}
            bounds = np.empty(len(Zref))
            for i, row in enumerate(Zref):
                a, b = int(row[0]), int(row[1])
                na, nb = sizes[a], sizes[b]
                if method == "average":
                    bounds[i] = D.gamma(math.ceil(math.log2(na * nb)) + 1) * abs(h_exact[i])
                else:
                    ca, cb = np.linalg.norm(cents[a]), np.linalg.norm(cents[b])
                    bounds[i] = U * (s.p + math.log2(s.n)) * (ca + cb) * math.sqrt(2.0 * na * nb / (na + nb))
                sizes[s.n + i] = na + nb
                cents[s.n + i] = (na * cents[a] + nb * cents[b]) / (na + nb)
            bounds = np.maximum(bounds, 2.0**-1074)
            for name, h in (("recomputed", rec_h), ("lance_williams", lw)):
                if h is None:
                    continue
                err = np.abs(h - Zref[:, 2] - lo)
                r = float(np.max(err / bounds))
                ulp = float(np.max(err / np.maximum(np.spacing(np.abs(h_exact)), 2.0**-1074)))
                if r > worst[name]:
                    worst[name] = r
                    where[name] = sid
                worst[f"{name}_ulp"] = max(worst[f"{name}_ulp"], ulp)
        worst["where"] = where
        out[method] = worst
        log(f"  {method}: {worst}")
    g = np.random.Generator(np.random.PCG64(14))
    X = g.standard_normal((5000, 5))
    for method, code in (("average", 2), ("ward", 4)):
        Z = sch.linkage(X, method)
        th = best(lambda: C._proto_heights(sda(X), sda(Z), code))
        tc = best(lambda: C._proto_nnchain(sda(X), code, 1))
        out[method]["time_recompute_ms"] = th
        out[method]["time_chain_ms"] = tc
        log(f"  n=5000 p=5 {method}: recompute {th:.0f} ms, chain {tc:.0f} ms ({th / tc:.2f} of it)")
    rec["heights"] = out


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


# ---------------------------------------------------------------- proto: centres


def proto_centers(log, rec):
    R = RD().ref("kmeans")
    D = RD()
    log("centers: the chunk sums (4096 points, then a pairwise tree) of the true labels' means against the exact")
    worst, where = 0.0, None
    for sid in R.group_inputs["set"].tolist():
        s = S().points(sid)
        if s.truth is None or s.w is not None:
            continue
        pos, idx = D.at(R, "truth_mean", sid, 2)
        if not len(pos):
            continue
        m = R.methods["truth_mean"]
        exact = np.empty(len(idx))
        exact[idx] = m.hi[pos] + m.lo[pos]
        k = int(s.truth.max()) + 1
        got = np.zeros((k, s.p))
        cnt = np.zeros(k)
        for i in range(s.n):  # one chunk: a sequential sum in index order
            j = s.truth[i]
            if j >= 0:
                got[j] += s.X[i]
                cnt[j] += 1
        got = (got / cnt[:, None]).ravel()
        bound = D.gamma(math.ceil(math.log2(s.n)) + 1) * float(np.max(np.abs(s.X)))
        r = float(np.max(np.abs(got - exact)) / bound) if bound > 0 else 0.0
        if r > worst:
            worst, where = r, sid
    rec["centers/chunk_sums"] = {"worst": worst, "where": where}
    log(f"  worst {worst:.3g} of gamma(ceil(log2 n) + 1) max|x| ({where})")


PROTO = {
    "distance": lambda: proto_distance,
    "bounds": lambda: proto_bounds,
    "mst": lambda: proto_mst,
    "nnchain": lambda: proto_nnchain,
    "heights": lambda: proto_heights,
    "tree_heights": lambda: proto_tree_heights,
    "centers": lambda: proto_centers,
}


def proto(log, only=None):
    """Every prototype measurement, or only those named (proto.distance ...: the others' records are kept)."""
    rec = {}
    if only:
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
        "note": "the prototypes of csrc/cluster/prototype.cpp against tests/cluster/ref",
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
