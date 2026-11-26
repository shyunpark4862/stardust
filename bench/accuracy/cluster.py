"""sd.cluster's accuracy sweeps: one per reference family of tests/cluster/ref (tools/cluster_ref.py) once the
functions it measures are public.

CL0 made the references, the comparators' records (scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2: tests/cluster/data,
from tools/cluster_probe.py) and the prototypes' measurements (tests/cluster/data/prototype.json); each milestone
adds the sweep of its functions. The plan (docs/design/cluster.md §7.5), each judged through
tests/cluster/refdata.py:

  hierarchy   CL1 (done): linkage's heights on hierarchy.npz in ULPs and in units of the contract
              (refdata.height_bound), and the trees against the exact primitive tree; comparators scipy's
              linkage and R's hclust, live
  kmeans      CL2 (done): the centres and inertia against the exact Lloyd fixed point from the same start
              (kmeans.npz) in units of the contract (refdata.center_bound, inertia_bound), the labels that differ
              from the exact ones where no step met a near tie; comparators scikit-learn's KMeans and R's kmeans
              (algorithm 'Lloyd') from the same start, live
  kmedoids    CL2 (done): PAM's objective against the exact BUILD and SWAP (refdata.pam_bound) and the medoids
              that differ; comparator R's pam, live
  fuzzy       CL2 (done): the memberships at given centres for m = 1.5, 2, 3 (refdata.membership_bound) and one
              step's centres (fuzzy_centre_bound); no comparator (e1071 is not installed)
  density     CL3 (done): DBSCAN's labels against the exact partition; HDBSCAN's core distances and minimum
              spanning tree weights (refdata.distance_bound), stabilities (stability_bound), probabilities and GLOSH
              scores (membership_strength_bound), the 'eom' and 'leaf' labels against the exact definition; OPTICS's
              reachabilities and order; comparator scikit-learn's DBSCAN, HDBSCAN and OPTICS, live
  mixture     CL4 (done): the log densities, the mixture's log density and the responsibilities at given factors
              (refdata.mixture_bounds), the factors of given covariances (precision_bound) and the M step at given
              responsibilities (mstep_bound), against mixture.npz; comparator scikit-learn's _estimate_log_gaussian_prob,
              _compute_precision_cholesky and _estimate_gaussian_parameters at the same inputs (its expanded quadratic
              forms of diag and spherical and its one-pass tied, diag and spherical moments lose digits)
  metrics     CL5 (done): ARI, RI and FM (correctly rounded: 0.5 ULP), MI, the entropies, AMI's expected MI, NMI, AMI,
              homogeneity, completeness and V (refdata.mi_bound, entropy_bound, emi_bound and their quotients), the
              silhouettes, Calinski-Harabasz, Davies-Bouldin and the gap statistic's log W of a given partition
              (silhouette_bound, ch_bound, db_bound, logw_bound) against metrics.npz; comparator scikit-learn's
              indices (its entropy and expected_mutual_information too) at the same labelings, live
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


# ---- k-means (CL2) ---------------------------------------------------------------------------------------------------
# kmeans from the references' start with tol 0 (until no label changes) on every set of kmeans.npz: the centres in
# units of refdata.center_bound (and relative to max |x|), the inertia in units of inertia_bound, and the sets whose
# labels differ from the exact fixed point although no step met a near tie ("labels_differ"). scikit-learn's KMeans
# and R's kmeans (algorithm 'Lloyd') from the same start are the comparator columns (the expanded form of
# scikit-learn loses the subnormal set: all its distances are 0).


def _km_group(r, sid):
    m = r.methods["lloyd_inertia"]
    from tests.cluster import refdata as D

    at = np.nonzero(m.index == D.group_of(r, sid))[0]
    return (m.hi[at[0]] + m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}) if len(at) else None


def _fit_kmeans(lib, X, init, w):
    k = len(init)
    if lib == "sd":
        import stardust as sd

        r = sd.cluster.kmeans(X, k, rng=sd.random.default_rng(0), init=init, tol=0.0, weights=w)
        return np.asarray(r.labels), np.asarray(r.centers), r.inertia
    if lib == "sklearn":
        from sklearn.cluster import KMeans

        r = KMeans(k, init=init, n_init=1, tol=0.0, max_iter=300).fit(X, sample_weight=w)
        return r.labels_, r.cluster_centers_, float(r.inertia_)
    if w is not None:
        return None
    from . import rsession

    expr = (f"x <- matrix(xv, nrow = {X.shape[0]}); c0 <- matrix(cv, nrow = {k}); "
            "km <- kmeans(x, c0, iter.max = 300, algorithm = 'Lloyd'); "
            "c(km$cluster - 1, as.vector(t(km$centers)), km$tot.withinss)")  # fmt: skip
    v = np.asarray(rsession.evaluate_named(expr, {"xv": np.ascontiguousarray(X.T).ravel(),
                                                  "cv": np.ascontiguousarray(init.T).ravel()}))  # fmt: skip
    n, p = X.shape
    return v[:n].astype(int), v[n : n + k * p].reshape(k, p), float(v[-1])


def _kmeans_regions(lib):
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("kmeans")
    pts = {}
    differ = []
    for sid in r.group_inputs["set"].tolist():
        g = _km_group(r, sid)
        if g is None:
            continue
        I_ref, f = g
        s = S.points(sid)
        init = np.asarray(r.group_inputs["init"][D.group_of(r, sid)], float)[: s.k]
        try:
            out = _fit_kmeans(lib, s.X, init, s.w)
        except Exception:  # noqa: BLE001 -- R's error on fewer distinct points than centres
            continue
        if out is None:
            continue
        lab, C, inertia = out
        pos, idx = D.at(r, "lloyd_label", sid, 0)
        ref_lab = np.empty(s.n, int)
        ref_lab[idx] = r.methods["lloyd_label"].hi[pos].astype(int)
        pos, idx = D.at(r, "lloyd_center", sid, 1)
        cref = np.empty(len(idx))
        cref[idx] = r.methods["lloyd_center"].hi[pos] + r.methods["lloyd_center"].lo[pos]
        Cref = cref.reshape(s.k, s.p)
        if not np.array_equal(S.first_seen(lab), S.first_seen(ref_lab)):
            if f["path_gap"] > D.NEAR_TIE:
                differ.append(sid)
            continue
        num = {}
        for a, b in zip(ref_lab.tolist(), np.asarray(lab).tolist()):
            num.setdefault(b, a)
        Cm = np.array([Cref[num[j]] for j in range(s.k)]) if len(num) == s.k else None
        if Cm is None:
            continue
        mx = float(np.max(np.abs(s.X)))
        err = np.abs(np.asarray(C) - Cm).ravel()
        cb = D.center_bound(s.n, mx)
        e = pts.setdefault("centers", {"where": [], "contract": [], "rel": []})
        for q, v in enumerate(err):
            e["where"].append(f"{sid} #{q}")
            e["contract"].append(v / cb if cb > 0 else (0.0 if v == 0 else np.inf))
            e["rel"].append(v / mx if mx else 0.0)
        ib = D.inertia_bound(s.n, s.p, I_ref)
        e = pts.setdefault("inertia", {"where": [], "contract": [], "rel": []})
        e["where"].append(sid)
        ie = abs(inertia - I_ref)
        e["contract"].append(ie / ib if ib > 0 else (0.0 if ie == 0 else np.inf))
        e["rel"].append(ie / abs(I_ref) if I_ref else 0.0)
    out = {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    }
    out["labels_differ"] = region(np.array(differ, dtype=object), rel=np.zeros(len(differ)))
    return out


# ---- k-medoids (CL2) -------------------------------------------------------------------------------------------------
# kmedoids (PAM) on every set and k of kmedoids.npz: the objective after SWAP in units of refdata.pam_bound and
# relative, and the sets whose medoids differ from the exact PAM's ("medoids_differ": exact ties of the objective,
# the 1-D medians and the lattices, decide by the computed sums). R's pam is the comparator, live.


def _pam(lib, X, k):
    if lib == "sd":
        import stardust as sd

        r = sd.cluster.kmedoids(X, k)
        return sorted(np.asarray(r.medoids).tolist()), r.objective
    from . import rsession

    expr = (
        f"library(cluster); x <- matrix(xv, nrow = {X.shape[0]}); p <- pam(x, {k}); c(sort(p$id.med) - 1, p$objective)"
    )
    v = np.asarray(rsession.evaluate_named(expr, {"xv": np.ascontiguousarray(X.T).ravel()}))
    return v[:k].astype(int).tolist(), float(v[k + 1])


def _kmedoids_regions(lib):
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("kmedoids")
    where, contract, rel, differ = [], [], [], []
    for sid in r.group_inputs["set"].tolist():
        pid, kk = sid.split("|")
        k = int(kk[1:])
        s = S.points(pid)
        m = r.methods["pam_swap"]
        at = np.nonzero(m.index == D.group_of(r, sid))[0]
        ref = m.hi[at[0]] + m.lo[at[0]]
        meds, obj = _pam(lib, s.X, k)
        pos, _ = D.at(r, "pam", sid, 0)
        if meds != sorted(r.methods["pam"].hi[pos].astype(int).tolist()):
            differ.append(sid)
        b = D.pam_bound(s.n, s.p, ref)
        where.append(sid)
        contract.append(abs(obj - ref) / b if b > 0 else (0.0 if obj == ref else np.inf))
        rel.append(abs(obj - ref) / ref if ref else 0.0)
    return {
        "objective": region(np.array(where, dtype=object), rel=np.array(rel), contract=np.array(contract), bound=1.0),
        "medoids_differ": region(np.array(differ, dtype=object), rel=np.zeros(len(differ))),
    }


# ---- fuzzy c-means (CL2) ----------------------------------------------------------------------------------------------
# the memberships at the references' given centres (fuzzy.npz) for m = 2, 1.5, 3 in units of refdata.membership_bound
# (absolute: a membership is in [0, 1]), and the centres one step on at m = 2 in units of fuzzy_centre_bound.


def _fuzzy_regions():
    import itertools

    import stardust as sd
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("fuzzy")
    pts = {}
    for sid in r.group_inputs["set"].tolist():
        c = np.asarray(r.group_inputs["centers"][D.group_of(r, sid)], float)
        if not c.size or np.isnan(c).all():
            continue
        s = S.points(sid)
        C = c.reshape(s.k, -1)
        for name, m in (("u_m2", 2.0), ("u_m15", 1.5), ("u_m3", 3.0)):
            pos, idx = D.at(r, name, sid, 0)
            ref = np.full(s.n * s.k, np.nan)
            ref[idx] = r.methods[name].hi[pos] + r.methods[name].lo[pos]
            got = np.asarray(sd._core.cluster.fuzzy_memberships(sd.asarray(s.X), sd.asarray(C), m)).ravel()
            b = D.membership_bound(s.p, s.k, m)
            e = pts.setdefault(f"memberships/m={m:g}", {"where": [], "contract": [], "rel": []})
            for q in np.nonzero(~np.isnan(ref))[0]:
                e["where"].append(f"{sid} #{q}")
                e["contract"].append(abs(got[q] - ref[q]) / b)
                e["rel"].append(abs(got[q] - ref[q]))
        if s.k > 6:
            continue
        res = sd.cluster.fuzzy_cmeans(s.X, s.k, rng=sd.random.default_rng(0), init=C, max_iter=1, tol=0.0)
        pos, idx = D.at(r, "centre1", sid, 1)
        cref = np.empty(len(idx))
        cref[idx] = r.methods["centre1"].hi[pos] + r.methods["centre1"].lo[pos]
        cref = cref.reshape(s.k, -1)
        got = np.asarray(res.centers)
        err = min(np.max(np.abs(got[list(q)] - cref)) for q in itertools.permutations(range(s.k)))
        mx = float(np.max(np.abs(s.X)))
        b = D.fuzzy_centre_bound(s.n, s.p, s.k, mx)
        e = pts.setdefault("centres", {"where": [], "contract": [], "rel": []})
        e["where"].append(sid)
        e["contract"].append(err / b if b > 0 else 0.0)
        e["rel"].append(err / mx if mx else 0.0)
    return {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    }


# ---- density (CL3) ---------------------------------------------------------------------------------------------------
# density.npz: DBSCAN at each set's eps and min_samples ("dbscan_differ": the sets whose labels are not the exact
# partition; scikit-learn's border points go to the first cluster that reaches them, so its column counts the sets
# whose core points or noise differ, "dbscan_border_differ" the others); on the "|hdbscan5" sets the core
# distances and the spanning tree's sorted weights in units of refdata.distance_bound, the stabilities in units of
# stability_bound, the probabilities and GLOSH scores in units of membership_strength_bound, the sets whose labels
# differ from the exact definition's ('eom', 'leaf'); OPTICS's reachabilities (distance_bound) and the sets whose
# order differs. scikit-learn's HDBSCAN breaks equal mutual reachabilities by its Prim's order and has no GLOSH
# (its probabilities and stabilities are not compared); its OPTICS rounds to 15 decimals and breaks ties by the input
# index.


def _density_fit(lib, kind, X, **kw):
    if lib == "sd":
        import stardust as sd

        return getattr(sd.cluster, kind)(X, **kw)
    import sklearn.cluster as skc

    if kind == "dbscan":
        w = kw.pop("weights", None)
        return skc.DBSCAN(**kw).fit(X, sample_weight=w)
    if kind == "hdbscan":
        return skc.HDBSCAN(copy=True, **kw).fit(X)
    return skc.OPTICS(**kw).fit(X)


def _attr(r, sd_name, sk_name):
    v = getattr(r, sd_name, None)
    if v is None:
        v = getattr(r, sk_name)
    return np.asarray(v)


def _density_regions(lib):
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("density")
    pts = {}
    lists = {"dbscan_differ": [], "dbscan_border_differ": [], "hdbscan_eom_differ": [], "hdbscan_leaf_differ": [],
             "optics_order_differ": []}  # fmt: skip

    def add(key, where, err, bound):
        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
        for q, (v, b) in enumerate(zip(np.atleast_1d(err), np.atleast_1d(bound))):
            e["where"].append(f"{where} #{q}")
            e["contract"].append(v / b if b > 0 else (0.0 if v == 0 else math.inf))
            e["rel"].append(v)

    def per(method, sid, kind=0):
        pos, idx = D.at(r, method, sid, kind)
        o = np.argsort(idx, kind="stable")
        m = r.methods[method]
        return m.hi[pos[o]], m.lo[pos[o]], {k: v[pos[o]] for k, v in m.fields.items()}

    for g, sid in enumerate(r.group_inputs["set"].tolist()):
        base = sid.split("|")[0]
        s = S.points(base)
        if "|eps" in sid:
            eps, ms = float(r.group_inputs["eps"][g]), int(r.group_inputs["min_samples"][g])
            lab, _, f = per("label", sid)
            got = _density_fit(lib, "dbscan", s.X, eps=eps, min_samples=ms, weights=s.w)
            gl = S.first_seen(_attr(got, "labels", "labels_"))
            if not np.array_equal(gl, lab.astype(int)):
                core = f["core"] == 1
                same_core = np.array_equal(S.first_seen(np.where(core, gl, -1)), S.first_seen(np.where(core, lab, -1)))
                lists[
                    "dbscan_border_differ" if same_core and np.array_equal(gl < 0, lab < 0) else "dbscan_differ"
                ].append(sid)
            continue
        h = _density_fit(lib, "hdbscan", s.X, min_samples=5)
        hl = _density_fit(lib, "hdbscan", s.X, min_samples=5, cluster_selection_method="leaf")
        for name, res, key in (("hdbscan_eom", h, "hdbscan_eom_differ"), ("hdbscan_leaf", hl, "hdbscan_leaf_differ")):
            ref, _, _ = per(name, sid)
            if not np.array_equal(S.first_seen(_attr(res, "labels", "labels_")), ref.astype(int)):
                lists[key].append(sid)
        core_ref, _, _ = per("core_distance", sid)
        if lib == "sd":
            import stardust as sd

            Z = np.asarray(h.single_linkage)
            o = np.asarray(sd._core.cluster.lex_order(sd.asarray(s.X)))
            core = np.empty(s.n)
            core[o] = np.asarray(sd._core.cluster.core_distances(sd.asarray(s.X[o]), None, 5, 0, 0))
            st, sl, f = per("stability", sid, 2)
            got = np.asarray(h._stability)
            fin = np.isfinite(st) & np.isfinite(got)
            add("hdbscan/stability", sid, np.abs(got[fin] - st[fin] - sl[fin]),
                D.stability_bound(s.p, f["plus"][fin], f["rows"][fin]))  # fmt: skip
            for name, field in (("probability", "probabilities"), ("outlier", "outlier_scores")):
                v, lo, _ = per(name, sid)
                add(f"hdbscan/{name}", sid, np.abs(np.asarray(getattr(h, field)) - v - lo),
                    np.full(s.n, D.membership_strength_bound(s.p)))  # fmt: skip
        else:
            from sklearn.neighbors import NearestNeighbors

            Z = np.asarray([list(t) for t in h._single_linkage_tree_.tolist()], float)
            core = NearestNeighbors(n_neighbors=5).fit(s.X).kneighbors(s.X, 5)[0][:, -1]
        add("hdbscan/core_distance", sid, np.abs(core - core_ref), D.distance_bound(s.p, core_ref))
        w, wl, _ = per("mst", sid, 1)
        add("hdbscan/mst", sid, np.abs(np.sort(Z[:, 2]) - w - wl), D.distance_bound(s.p, w))
        o = _density_fit(lib, "optics", s.X, min_samples=5)
        order, _, _ = per("optics_order", sid, 3)
        if not np.array_equal(_attr(o, "ordering", "ordering_"), order.astype(int)):
            lists["optics_order_differ"].append(sid)
        rh, rl, _ = per("optics_reach", sid)
        got = _attr(o, "reachability", "reachability_")
        fin = np.isfinite(rh)
        add("optics/reachability", sid, np.abs(got[fin] - rh[fin] - rl[fin]), D.distance_bound(s.p, rh[fin]))
    out = {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    }
    for key, where in lists.items():
        out[key] = region(np.array(where, dtype=object), rel=np.zeros(len(where)))
    return out


# ---- mixtures (CL4) ---------------------------------------------------------------------------------------------------
# mixture.npz: at the given factors U_f (scikit-learn's precisions_cholesky_ of the sets' covariances) the components'
# log densities, the mixture's and the responsibilities in units of refdata.mixture_bounds (theorem 7); the factors
# of the exact covariances in units of precision_bound; the M step at the rounded responsibilities in units of
# mstep_bound (theorem 9). scikit-learn's functions at the same inputs are the comparator columns (its M step's
# weights normalized as its _m_step does).

_MIX_TYPES = {"full": 0, "tied": 1, "diag": 2, "spherical": 3}


def _mixture_regions(lib):
    import math

    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    r = D.ref("mixture")
    pts = {}

    def per(name, mid, kind, lo=True):
        pos, idx = D.at(r, name, mid, kind)
        o = np.argsort(idx, kind="stable")
        m = r.methods[name]
        return m.hi[pos[o]] + (m.lo[pos[o]] if lo else 0.0)

    def add(key, where, err, bound):
        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
        for q, (v, b) in enumerate(zip(np.ravel(err), np.ravel(bound))):
            e["where"].append(f"{where} #{q}")
            e["contract"].append(v / b if b > 0 else (0.0 if v == 0 else math.inf))
            e["rel"].append(v)

    for mid in S.mixture_ids():
        mx = S.mixture(mid)
        k, p, t = mx.k, mx.p, mx.covariance_type
        shape = {"full": (k, p, p), "tied": (p, p), "diag": (k, p), "spherical": (k,)}[t]
        U = np.asarray(r.group_inputs["prec_chol"][D.group_of(r, mid)], float).reshape(shape)
        wlp_ref = np.column_stack([per(f"logdens_u_{j}", mid, 0) + math.log(mx.weights[j]) for j in range(k)])
        bE, bs, br = D.mixture_bounds(t, mx.X, mx.weights, mx.means, U, wlp_ref)
        Rf = np.column_stack([per(f"resp_{j}", mid, 0, lo=False) for j in range(k)])  # the reference M step's input
        if lib == "sd":
            import stardust as sd

            core = sd._core.cluster
            blocks = U if t == "full" else (np.repeat(U[None], k, 0) if t == "tied" else None)
            ld = (np.array([np.sum(np.log(np.diag(B))) for B in blocks]) if blocks is not None
                  else (np.sum(np.log(U), axis=1) if t == "diag" else p * np.log(U)))  # fmt: skip
            comp = []
            for j in range(k):
                Uj = U[j : j + 1] if t != "tied" else U
                _, _, lj = core.gmm_estimate(sd.asarray(mx.X), sd.asarray(mx.means[j : j + 1]),
                                             sd.asarray(np.ascontiguousarray(Uj)), sd.asarray(ld[j : j + 1]),
                                             sd.asarray(np.zeros(1)), _MIX_TYPES[t], 0)  # fmt: skip
                comp.append(np.asarray(lj))
            comp = np.column_stack(comp)
            _, resp, lpn = core.gmm_estimate(sd.asarray(mx.X), sd.asarray(mx.means), sd.asarray(np.ascontiguousarray(U)),
                                             sd.asarray(ld), sd.asarray(np.log(mx.weights)), _MIX_TYPES[t], 0)  # fmt: skip
            resp, lpn = np.asarray(resp), np.asarray(lpn)
            fac = core.gmm_precisions(sd.asarray(np.ascontiguousarray(mx.covariances)), _MIX_TYPES[t], k, p)
            Ug = np.asarray(fac[0])
            w, m, c = core.gmm_mstep(sd.asarray(mx.X), sd.asarray(Rf), _MIX_TYPES[t], 1e-6)
            ms = np.concatenate([np.asarray(w).ravel(), np.asarray(m).ravel(), np.asarray(c).ravel()])
        else:
            from scipy.special import logsumexp
            from sklearn.mixture import _gaussian_mixture as G

            comp = G._estimate_log_gaussian_prob(mx.X, mx.means, U, t)
            wl = comp + np.log(mx.weights)
            lpn = logsumexp(wl, axis=1)
            resp = np.exp(wl - lpn[:, None])
            Ug = G._compute_precision_cholesky(np.asarray(mx.covariances, float), t)
            nk, m, c = G._estimate_gaussian_parameters(mx.X, Rf, 1e-6, t)
            ms = np.concatenate([nk / nk.sum(), m.ravel(), np.asarray(c).ravel()])
        add("mixture/logdens", mid, np.abs(comp - (wlp_ref - np.log(mx.weights))), bE)
        add("mixture/score", mid, np.abs(lpn - per("score_u", mid, 0)), bs)
        rr = np.column_stack([per(f"resp_u_{j}", mid, 0) for j in range(k)])
        add("mixture/resp", mid, np.abs(resp - rr), br)
        uref = per("prec_chol", mid, 2).reshape(shape)
        add("mixture/prec_chol", mid, np.abs(Ug - uref), D.precision_bound(t, mx.covariances, uref, k, p))
        mref = per("mstep", mid, 1)
        add("mixture/mstep", mid, np.abs(ms - mref), D.mstep_bound(t, mx.X, Rf, mref))
    return {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    }


# ---- metrics (CL5) ----------------------------------------------------------------------------------------------
# Every labeling of metrics.npz and every point set with true labels: the exact indices in ULPs (their contract is
# 0.5 ULP: correctly rounded), the others in units of their contract (tests/cluster/refdata.py). scikit-learn's
# indices at the same inputs are the comparator column (its float pair counts, lgamma differences and expanded
# distances).

_AVGS = ("min", "geometric", "arithmetic", "max")


def _norm(avg, ha, hb):
    return {"min": min(ha, hb), "max": max(ha, hb), "geometric": math.sqrt(ha * hb), "arithmetic": (ha + hb) / 2}[avg]


def _metric_values(lib, a, b, emi=True):
    """{name: value} of one labeling pair."""
    if lib == "sd":
        import stardust as sd

        cl = sd.cluster
        ca = np.unique(a, return_inverse=True)[1].astype(np.int64)
        cb = np.unique(b, return_inverse=True)[1].astype(np.int64)
        _, _, _, mi, ha, hb, e = sd._core.cluster.external(
            sd.asarray(ca), sd.asarray(cb), int(ca.max()) + 1, int(cb.max()) + 1, emi
        )
        out = {"mi": mi, "h_true": ha, "h_pred": hb, "emi": e}
        M = cl
    else:
        import sklearn.metrics as M
        from sklearn.metrics.cluster import _supervised as SV
        from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information

        C = SV.contingency_matrix(a, b, sparse=True)
        out = {"mi": M.mutual_info_score(a, b), "h_true": float(SV.entropy(a)), "h_pred": float(SV.entropy(b)),
               "emi": float(expected_mutual_information(C, len(a))) if emi else math.nan}  # fmt: skip
    out["ari"] = M.adjusted_rand_score(a, b)
    out["ri"] = M.rand_score(a, b)
    out["fm"] = M.fowlkes_mallows_score(a, b)
    out["homogeneity"] = M.homogeneity_score(a, b)
    out["completeness"] = M.completeness_score(a, b)
    out["v"] = M.v_measure_score(a, b)
    for avg in _AVGS:
        out[f"nmi_{avg}"] = M.normalized_mutual_info_score(a, b, average_method=avg)
        if emi:
            out[f"ami_{avg}"] = M.adjusted_mutual_info_score(a, b, average_method=avg)
    return out


def _metrics_regions(lib):
    from tests.cluster import refdata as D
    from tests.cluster import samples as S

    from .common import ulps

    r = D.ref("metrics")
    pts = {}

    def gv(name, sid):
        m = r.methods[name]
        at = np.nonzero(m.index == D.group_of(r, sid))[0]
        return (m.hi[at[0]], m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}) if len(at) else None

    def add(key, where, err, bound, ulp=None):
        e = pts.setdefault(key, {"where": [], "contract": [], "rel": [], "ulp": []})
        for q, (v, b) in enumerate(zip(np.ravel(err), np.ravel(bound))):
            e["where"].append(f"{where} #{q}" if np.size(err) > 1 else where)
            e["contract"].append(v / b if b > 0 else (0.0 if v == 0 else math.inf))
            e["rel"].append(v)
        if ulp is not None:
            e["ulp"].extend(np.ravel(ulp).tolist())

    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        emi_ref = gv("emi", lid)
        vals = _metric_values(lib, lb.a, lb.b, emi=lb.n <= 10**5)
        M, ra, rb, n = D.contingency(lb.a, lb.b)
        for name in ("ari", "ri", "fm"):
            hi, lo, _ = gv(name, lid)
            d = float(ulps(vals[name], hi))
            add(f"metrics/{name}", lid, d, 0.5, ulp=[d])  # correctly rounded: the nearest double, 0 ULP away
        mi_ref, ha_ref, hb_ref = gv("mi", lid), gv("h_true", lid), gv("h_pred", lid)
        dmi = D.mi_bound(M, n)
        dha, dhb = D.entropy_bound(ra, ha_ref[0]), D.entropy_bound(rb, hb_ref[0])
        add("metrics/mi", lid, abs(vals["mi"] - (mi_ref[0] + mi_ref[1])), dmi)
        add("metrics/entropy", lid + " true", abs(vals["h_true"] - ha_ref[0] - ha_ref[1]), dha)
        add("metrics/entropy", lid + " pred", abs(vals["h_pred"] - hb_ref[0] - hb_ref[1]), dhb)
        if emi_ref is not None:
            add("metrics/emi", lid, abs(vals["emi"] - emi_ref[0] - emi_ref[1]), D.emi_bound(emi_ref[2]["plus"]))
        for avg in _AVGS:
            ref = gv(f"nmi_{avg}", lid)
            N = _norm(avg, ha_ref[0], hb_ref[0])
            bound = 0.0 if ref[2]["special"] else D.ratio_bound(mi_ref[0], dmi, N, dha + dhb + 2 * D.U * N)
            add("metrics/nmi", f"{lid} {avg}", abs(vals[f"nmi_{avg}"] - ref[0]), bound)
            ref = gv(f"ami_{avg}", lid)
            if ref is None or f"ami_{avg}" not in vals:
                continue
            if ref[2]["special"]:
                add("metrics/ami", f"{lid} {avg}", abs(vals[f"ami_{avg}"] - ref[0]), 0.0)
                continue
            de = D.emi_bound(emi_ref[2]["plus"])
            bound = D.ratio_bound(mi_ref[0] - emi_ref[0], dmi + de, N - emi_ref[0], dha + dhb + de + 2 * D.U * N)
            add("metrics/ami", f"{lid} {avg}", abs(vals[f"ami_{avg}"] - ref[0]), bound)
        for name, h, dh in (("homogeneity", ha_ref[0], dha), ("completeness", hb_ref[0], dhb)):
            ref = gv(name, lid)
            add("metrics/homogeneity", f"{lid} {name}", abs(vals[name] - ref[0]),
                D.ratio_bound(mi_ref[0], dmi, h, dh) if h else 0.0)  # fmt: skip
        ref = gv("v", lid)
        s_ = ha_ref[0] + hb_ref[0]
        add("metrics/v", lid, abs(vals["v"] - ref[0]),
            D.ratio_bound(2 * mi_ref[0], 2 * dmi, s_, dha + dhb + D.U * s_) if s_ else 0.0)  # fmt: skip
    sets = set(r.group_inputs["set"].tolist())
    for sid in S.point_ids():
        s = S.points(sid)
        if sid not in sets or s.truth is None or not 2 <= len(set(s.truth.tolist())) <= s.n - 1:
            continue
        if lib == "sd":
            import stardust as sd

            cl = sd.cluster
            sil = np.asarray(cl.silhouette_samples(s.X, s.truth))
            ch, db = cl.calinski_harabasz_score(s.X, s.truth), cl.davies_bouldin_score(s.X, s.truth)
            codes = np.unique(s.truth, return_inverse=True)[1].astype(np.int64)
            o = np.asarray(sd._core.cluster.lex_order(sd.asarray(s.X)))
            K = int(codes.max()) + 1
            lw = {pw: sd._core.cluster.gap_logw(sd.asarray(np.ascontiguousarray(s.X[o])), sd.asarray(codes[o]), K,
                                                float(pw)) for pw in (1, 2)}  # fmt: skip
        else:
            import sklearn.metrics as M

            sil = M.silhouette_samples(s.X, s.truth)
            ch, db = M.calinski_harabasz_score(s.X, s.truth), M.davies_bouldin_score(s.X, s.truth)
            lw = None
        pos, idx = D.at(r, "silhouette", sid, 0)
        o2 = np.argsort(idx)
        ref = r.methods["silhouette"].hi[pos[o2]] + r.methods["silhouette"].lo[pos[o2]]
        n_max = int(np.max(np.unique(s.truth, return_counts=True)[1]))
        add("metrics/silhouette", sid, np.abs(sil - ref), np.full(len(ref), D.silhouette_bound(n_max, s.p)))
        hi, lo, _ = gv("calinski_harabasz", sid)
        add("metrics/calinski_harabasz", sid, abs(ch - hi - lo), D.ch_bound(s.X, s.truth, hi))
        hi, lo, _ = gv("davies_bouldin", sid)
        add("metrics/davies_bouldin", sid, abs(db - hi - lo), D.db_bound(s.X, s.truth, hi))
        if lw is not None:
            K = len(set(s.truth.tolist()))
            for pw in (1, 2):
                ref = gv(f"logw_{pw}", sid)
                if ref is not None:
                    add("metrics/gap_logw", f"{sid} d_power={pw}", abs(lw[pw] - ref[0] - ref[1]),
                        D.logw_bound(s.n, s.p, K, pw, ref[0]))  # fmt: skip
    return {
        key: region(
            p["where"],
            rel=np.array(p["rel"]),
            contract=np.array(p["contract"]),
            bound=1.0,
            **({"ulp": np.array(p["ulp"])} if p["ulp"] else {}),
        )  # fmt: skip
        for key, p in sorted(pts.items())
    }


def sweeps():
    return [
        Custom(
            "cluster",
            "hierarchy",
            lambda: _hierarchy_regions("sd"),
            compare_libs(_hierarchy_regions, ("scipy", "R")),
        ),
        Custom("cluster", "kmeans", lambda: _kmeans_regions("sd"), compare_libs(_kmeans_regions, ("sklearn", "R"))),
        Custom("cluster", "kmedoids", lambda: _kmedoids_regions("sd"), compare_libs(_kmedoids_regions, ("R",))),
        Custom("cluster", "fuzzy", _fuzzy_regions),
        Custom("cluster", "density", lambda: _density_regions("sd"), compare_libs(_density_regions, ("sklearn",))),
        Custom("cluster", "mixture", lambda: _mixture_regions("sd"), compare_libs(_mixture_regions, ("sklearn",))),
        Custom("cluster", "metrics", lambda: _metrics_regions("sd"), compare_libs(_metrics_regions, ("sklearn",))),
    ]
