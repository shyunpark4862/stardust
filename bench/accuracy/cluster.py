"""sd.cluster's accuracy sweeps: one per reference family of tests/cluster/ref (tools/cluster_ref.py) once the
functions it measures are public.

CL0 made the references, the comparators' records (scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2: tests/cluster/data,
from tools/cluster_probe.py) and the prototypes' measurements (tests/cluster/data/prototype.json); each milestone
adds the sweep of its functions. The plan (docs/design/cluster.md §7.5), each judged through
tests/cluster/refdata.py:

  hierarchy   CL1 (done): linkage's heights on hierarchy.npz in ULPs and in units of the contract
              (refdata.height_bound), and the trees against the exact primitive tree; comparators scipy's
              linkage and R's hclust, live
  kmeans      CL2: the centres and inertia of the exact Lloyd fixed point from the same start (kmeans.npz),
              the labels at the rounded centres, the fixed point's check; comparator scikit-learn's KMeans from
              the same start
  kmedoids    CL2: PAM's objectives and medoids (and the optimum of the small sets); comparator R's pam
  density     CL3: DBSCAN's partition (the points near eps counted apart), HDBSCAN's core distances and minimum
              spanning tree; comparators scikit-learn's DBSCAN and HDBSCAN
  mixture     CL4: the log densities and responsibilities at given parameters (relative to max(1, |value|)) and
              the M step at given responsibilities; comparator scikit-learn's GaussianMixture at the same
              parameters (its one-pass diagonal covariance loses digits: 6.6e-12 measured at CL0)
  metrics     CL5: ARI, RI (correctly rounded: 0.5 ULP), FM, MI, NMI, AMI, homogeneity, completeness, V, the
              silhouettes, Calinski-Harabasz and Davies-Bouldin; comparator scikit-learn's indices
"""

from __future__ import annotations

import math

import numpy as np

from .common import Custom, compare_libs, region

# ---- hierarchy (CL1) -----------------------------------------------------------------------------------------------
# linkage on every set of hierarchy.npz and the seven methods: the heights in ULPs of the exact ones and in units of
# the contract (tests/cluster/refdata.py height_bound: single and complete gamma(p/2 + 2) h*, the others the proved
# bound of the Lance-Williams recurrence on the final tree, gamma(c mu + p + 5) V+), regions by method and by the
# data's kind; the trees: a set whose rows differ from the exact primitive tree counts in "trees_differ" (a near
# tie of the reference may go either way: "trees_near_tie"). scipy's linkage (Lance-Williams along its own merge
# order; trees compared where they agree with the reference) and R's hclust (heights sorted, on the sets without a
# near tie; ward.D2, and centroid and median on squared distances rooted) are the comparator columns.

_METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
_R_NAME = {"ward": "ward.D2", "weighted": "mcquitty"}
_KINDS = (
    ("scale", {"scale"}),
    ("ties", {"ties", "degenerate", "tiny"}),
)


def _kind(tags):
    for name, want in _KINDS:
        if tags & want:
            return name
    return "plain"


def _linkage(lib, X, method):
    if lib == "sd":
        import stardust as sd

        return np.asarray(sd.cluster.linkage(X, method=method))
    if lib == "scipy":
        import scipy.cluster.hierarchy as sch

        return sch.linkage(X, method)
    from . import rsession

    name = _R_NAME.get(method, method)
    d = "dist(x)^2" if method in ("centroid", "median") else "dist(x)"
    expr = f"x <- matrix(xv, nrow = {X.shape[0]}); hclust({d}, '{name}')$height"
    h = np.asarray(rsession.evaluate_named(expr, {"xv": np.ascontiguousarray(X.T).ravel()}))
    if method in ("centroid", "median"):
        h = np.sqrt(np.maximum(h, 0.0))
    return h  # heights only, in R's merge order


def _hierarchy_regions(lib):
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("hierarchy")
    pts = {}
    trees = {"trees_differ": [], "trees_near_tie": []}
    for sid in r.group_inputs["set"].tolist():
        s = S.points(sid)
        kind = _kind(s.tags)
        keys = S.lex_rank(s.X)
        for method in _METHODS:
            Zr, gaps, lo = D.tree(r, method, sid)
            h_ref = Zr[:, 2] + lo
            got = _linkage(lib, s.X, method)
            if lib == "R":
                if len(gaps) and float(np.min(gaps)) <= D.NEAR_TIE:
                    continue  # R's tree at a tie is another tree: its sorted heights are not comparable
                h = np.sort(got)
                order = np.argsort(h_ref, kind="stable")
                if len(h) != len(h_ref):
                    continue
                href = np.sort(h_ref)
                bound = D.height_bound(Zr, s.X, method, keys, h_ref)[order]
            else:
                if not np.array_equal(got[:, [0, 1, 3]], Zr[:, [0, 1, 3]]):
                    near = len(gaps) and float(np.min(gaps)) <= D.NEAR_TIE
                    trees["trees_near_tie" if near else "trees_differ"].append(f"{sid} {method}")
                    if D.merges(got, s.n) != D.merges(Zr, s.n):
                        continue
                    # the same clusters in another order: compare each cluster's height
                    mem = {i: frozenset([i]) for i in range(s.n)}
                    by = {}
                    for i, row in enumerate(got):
                        mem[s.n + i] = mem[int(row[0])] | mem[int(row[1])]
                        by[mem[s.n + i]] = row[2]
                    memr = {i: frozenset([i]) for i in range(s.n)}
                    h = np.empty(len(Zr))
                    for i, row in enumerate(Zr):
                        memr[s.n + i] = memr[int(row[0])] | memr[int(row[1])]
                        h[i] = by[memr[s.n + i]]
                else:
                    h = got[:, 2]
                href = h_ref
                bound = D.height_bound(Zr, s.X, method, keys, h_ref)
            err = np.abs(h - href)
            ulp = err / np.maximum(np.spacing(np.abs(href)), 2.0**-1074)
            for key in (f"{method}/{kind}",):
                e = pts.setdefault(key, {"where": [], "ulp": [], "contract": [], "rel": []})
                for i in range(len(h)):
                    e["where"].append(f"{sid} #{i}")
                    e["ulp"].append(ulp[i])
                    e["contract"].append(err[i] / bound[i] if bound[i] > 0 else (0.0 if err[i] == 0 else math.inf))
                    e["rel"].append(err[i] / abs(href[i]) if href[i] else 0.0)
    out = {
        key: region(
            p["where"], ulp=np.array(p["ulp"]), rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0
        )  # fmt: skip
        for key, p in sorted(pts.items())
    }
    if lib != "R":
        for key, where in trees.items():
            out[key] = region(np.array(where, dtype=object), rel=np.zeros(len(where)))
    return out or None


def sweeps():
    return [
        Custom(
            "cluster",
            "hierarchy",
            lambda: _hierarchy_regions("sd"),
            compare_libs(_hierarchy_regions, ("scipy", "R")),
        ),
    ]
