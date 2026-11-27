"""The reference data of sd.embed (docs/design/embed.md §4, §7.2): every file belongs to the test data as they are
(digests), the references pass checks of their own (eigenvalues ordered, eigenvectors orthonormal with the sign rule
of decision EM14, the defining equations of the roots holding, P rows summing to 1, trustworthiness in [0, 1], the
isotonic fits' block means), and independent re-derivations in float64 (numpy and scipy) agree within loose bounds.
No sd.embed function is needed: the tests judge the references, not an implementation."""

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path

from . import refdata as D
from . import samples as S

REFS = {f: D.ref(f) for f in D.FAMILIES}


def obj(sid):
    return S.points(D.base(sid))


def vals(family, method, sid, kind=None):
    idx, hi, _, fields = D.per_set(REFS[family], method, sid, kind)
    return idx, hi, fields


def gval(family, method, sid):
    g = D.group_value(REFS[family], method, sid)
    return None if g is None else g[0]


def sets_of(family, suffix):
    return [s for s in REFS[family].group_inputs["set"].tolist() if s.endswith(suffix) or suffix in s]


def check_sign(col, lo, keys, gap):
    """EM14 on a vector given as double-doubles (col + lo): the entry of largest |value| is positive; at an exact
    tie (gap 0) the one of smallest key. Magnitudes compare as (|hi|, sign(hi) lo)."""
    col, lo = np.asarray(col), np.asarray(lo)
    mag = [(abs(h), math.copysign(1.0, h) * l) for h, l in zip(col, lo)]
    top = max(mag)
    if top[0] == 0:
        return
    cands = [i for i, m in enumerate(mag) if m == top]
    if gap > 0:
        if len(cands) == 1:  # else the runner-up is within 2^-106 and the double-doubles cannot tell
            assert col[cands[0]] > 0
    else:
        assert col[min(cands, key=lambda i: keys[i])] > 0


def column(family, method, sid, kind, j, width=2):
    """(hi, lo, sign_gap) of column j of an embedding stored at index i width + j."""
    idx, hi, lo, f = D.per_set(REFS[family], method, sid, kind)
    sel = (idx % width) == j
    return hi[sel], lo[sel], f["sign_gap"][sel][0]


def sqd(X):
    X = np.asarray(X, float)
    return ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)


# ---- the files belong to the data -----------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert D.names() == sorted(D.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/embed_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f
        assert all(not len(m.dropped) for m in r.methods.values()), f
        assert np.all(np.isfinite([m.hi for m in r.methods.values()][0]))


def test_points_and_groups_are_the_generators_points():
    for f, r in REFS.items():
        for sid, dig in set(zip(r.inputs["set"].tolist(), r.inputs["digest"].tolist())):
            assert obj(sid).digest == dig, (f, sid)
        for sid in r.group_inputs["set"].tolist():
            assert D.base(sid) in S.point_ids(), (f, sid)


def test_the_key_is_lexicographic_and_order_free():
    a, b = S.points("lattice/5x5"), S.points("lattice/5x5-shuffled")
    ra, rb = S.lex_rank(a.X), S.lex_rank(b.X)
    assert sorted(ra.tolist()) == list(range(25))
    for i in range(25):
        j = int(np.nonzero((a.X == b.X[i]).all(axis=1))[0][0])
        assert rb[i] == ra[j]


def test_the_values_are_finite_and_the_low_parts_small():
    for f, r in REFS.items():
        for name, m in r.methods.items():
            hi, lo = np.asarray(m.hi), np.asarray(m.lo)
            assert np.all(np.isfinite(hi)), (f, name)
            assert np.all(np.abs(lo) <= np.abs(hi) * 2.0**-52 + 1e-300), (f, name)


# ---- linear -----------------------------------------------------------------------------------


def _pca_float(s, cor):
    X = np.asarray(s.X, float)
    C = np.cov(X, rowvar=False, ddof=1)
    if cor:
        sd = np.sqrt(np.diag(C))
        C = C / sd[:, None] / sd[None, :]
    w, V = np.linalg.eigh(C)
    return C, w[::-1], V[:, ::-1]


@pytest.mark.parametrize("variant", ["cov", "cor"])
def test_pca_spectrum_and_components(variant):
    for sid in sets_of("linear", f"|{variant}"):
        s = obj(sid)
        n, p = s.n, s.p
        m = min(n, p)
        _, var, f = vals("linear", "variance", sid, 0)
        assert len(var) == m
        assert np.all(np.diff(var) <= 0) and np.all(var >= 0), sid
        l1 = var[0]
        C, w, V = _pca_float(s, variant == "cor")
        assert np.allclose(var, w[:m], rtol=0, atol=1e-9 * l1), sid
        _, sv, _ = vals("linear", "singular", sid, 0)
        assert np.allclose(sv**2, (n - 1) * var, rtol=1e-14, atol=1e-300), sid
        _, ratio, _ = vals("linear", "ratio", sid, 0)
        total = gval("linear", "total_variance", sid)
        assert np.allclose(ratio, var / total, rtol=1e-14), sid
        assert math.isclose(math.fsum(var), total, rel_tol=1e-12), sid
        if variant == "cor":
            assert total == p
        d_g = D.group_value(REFS["linear"], "noise_variance", sid)
        d = int(d_g[2]["d"])
        noise = gval("linear", "noise_variance", sid)
        assert math.isclose(noise, np.mean(var[d:]) if d < m else 0.0, rel_tol=1e-12, abs_tol=1e-300), sid
        idx, comp, cf = vals("linear", "component", sid, 1)
        if not len(idx):
            continue
        k = len(idx) // p
        Vr = D.matrix(REFS["linear"], "component", sid, (k, p), 1)
        assert np.allclose(Vr @ Vr.T, np.eye(k), atol=1e-14), sid
        gaps = cf["gap"][::p]
        sg = cf["sign_gap"][::p]
        for j in range(k):
            _, _, lo_, _ = D.per_set(REFS["linear"], "component", sid, 1)
            check_sign(Vr[j], lo_[j * p : (j + 1) * p], range(p), sg[j])
            sin = min(np.linalg.norm(Vr[j] - V[:, j]), np.linalg.norm(Vr[j] + V[:, j]))  # ~ the angle's sine
            assert sin <= 1e-9 * l1 / gaps[j] + 1e-12, (sid, j, sin)
            assert np.allclose(C @ Vr[j], var[j] * Vr[j], atol=1e-9 * l1 / max(gaps[j] / l1, 1e-6)), (sid, j)


def test_kernel_pca():
    for sid in sets_of("linear", "|kpca-"):
        s = obj(sid)
        n, p = s.n, s.p
        X = np.asarray(s.X, float)
        kernel = sid.split("kpca-")[1]
        K = X @ X.T if kernel == "linear" else ((X @ X.T) / p + 1) ** 3 if kernel == "poly" else np.exp(-sqd(X) / p)
        Kc = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
        w = np.linalg.eigvalsh(Kc)[::-1]
        idx, ev, _ = vals("linear", "kpca_eigenvalue", sid, 2)
        if len(idx):
            assert np.all(np.diff(ev) <= 1e-300)
            assert np.allclose(ev, w[: len(ev)], atol=1e-10 * abs(w[0])), sid
        idx, kv, _ = vals("linear", "kpca_kernel", sid, 4)
        for t, v in zip(idx, kv):
            i, j = divmod(int(t), n)
            assert math.isclose(v, Kc[i, j], rel_tol=1e-10, abs_tol=1e-10 * np.abs(Kc).max()), sid
        idx, emb, ef = vals("linear", "kpca_embedding", sid, 3)
        rank = S.lex_rank(X)
        for j in sorted({int(t) % 2 for t in idx}):
            col = emb[(idx % 2) == j]
            lam = ev[j]
            assert math.isclose(col @ col, lam, rel_tol=1e-12), sid
            assert np.allclose(Kc @ col, lam * col, atol=1e-8 * abs(w[0]) * np.abs(col).max()), sid
            check_sign(*column("linear", "kpca_embedding", sid, 3, j)[:2], rank, ef["sign_gap"][(idx % 2) == j][0])


def _fa_float(C, psi, q):
    R = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    sc = 1 / np.sqrt(psi)
    Ss = R * sc[:, None] * sc[None, :]
    e, V = np.linalg.eigh(Ss)
    e, V = e[::-1], V[:, ::-1]
    obj_ = float(np.sum(e[q:] - np.log(e[q:])) + q - len(psi))
    load = (V[:, :q] * np.sqrt(np.maximum(e[:q] - 1, 0))) * np.sqrt(psi)[:, None]
    g = np.diag(load @ load.T + np.diag(psi) - R) / psi**2
    return e, obj_, g


def test_factor_analysis_objective_and_gradient():
    sids = sets_of("linear", "|fa-")
    assert sids
    for sid in sids:
        s = obj(sid)
        C = np.cov(np.asarray(s.X, float), rowvar=False)
        psi = np.asarray(D.group_input(REFS["linear"], sid, "psi"), float)
        q = int(D.group_input(REFS["linear"], sid, "q"))
        e, f, g = _fa_float(C, psi, q)
        _, ev, _ = vals("linear", "fa_eigen", sid, 5)
        assert np.allclose(ev, e, rtol=1e-10), sid
        assert math.isclose(gval("linear", "fa_objective", sid), f, rel_tol=1e-9, abs_tol=1e-12), sid
        _, gr, _ = vals("linear", "fa_gradient", sid, 6)
        assert np.allclose(gr, g, rtol=1e-8, atol=1e-10), sid
        h = 1e-6 * psi  # FAgr is the gradient of FAfn (R's pair)
        fd = [
            (
                _fa_float(C, psi + h[i] * np.eye(len(psi))[i], q)[1]
                - _fa_float(C, psi - h[i] * np.eye(len(psi))[i], q)[1]
            )
            / (2 * h[i])
            for i in range(len(psi))
        ]
        assert np.allclose(gr, fd, rtol=1e-4, atol=1e-5 * np.abs(gr).max()), sid


def _sym_decorrelation(W):
    s, u = np.linalg.eigh(W @ W.T)
    return (u * (1 / np.sqrt(s))) @ u.T @ W


def test_fastica_step():
    for sid in sets_of("linear", "|ica"):
        Xw = np.asarray(D.group_input(REFS["linear"], sid, "Xw"), float)
        w0 = np.asarray(D.group_input(REFS["linear"], sid, "w_init"), float)
        d = w0.shape[0]
        W0 = D.matrix(REFS["linear"], "ica_w0", sid, (d, d), 7)
        W1 = D.matrix(REFS["linear"], "ica_w1", sid, (d, d), 7)
        assert np.allclose(W0 @ W0.T, np.eye(d), atol=1e-14)
        assert np.allclose(W1 @ W1.T, np.eye(d), atol=1e-14)
        W0f = _sym_decorrelation(w0)
        assert np.allclose(W0, W0f, atol=1e-13)
        X1 = Xw.T
        gx = np.tanh(W0 @ X1)
        g_ = (1 - gx**2).mean(axis=-1)
        W1f = _sym_decorrelation(gx @ X1.T / X1.shape[1] - g_[:, None] * W0)
        assert np.allclose(W1, W1f, atol=1e-12)
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W0)) - 1))
        assert math.isclose(gval("linear", "ica_lim", sid), lim, rel_tol=1e-8, abs_tol=1e-15)


def test_nmf_multiplicative_step():
    for sid in sets_of("linear", "|nmf-"):
        s = obj(sid)
        X = np.asarray(s.X, float)
        W0 = np.asarray(D.group_input(REFS["linear"], sid, "W0"), float)
        H0 = np.asarray(D.group_input(REFS["linear"], sid, "H0"), float)
        n, d = W0.shape
        W1 = W0 * (X @ H0.T) / (W0 @ (H0 @ H0.T))
        H1 = H0 * (W1.T @ X) / ((W1.T @ W1) @ H0)
        assert np.allclose(D.matrix(REFS["linear"], "nmf_w1", sid, (n, d), 8), W1, rtol=1e-12)
        assert np.allclose(D.matrix(REFS["linear"], "nmf_h1", sid, (d, s.p), 9), H1, rtol=1e-12)
        l0, l1 = gval("linear", "nmf_loss0", sid), gval("linear", "nmf_loss1", sid)
        assert l1 <= l0  # the multiplicative update does not increase the loss
        assert math.isclose(l0, 0.5 * np.sum((X - W0 @ H0) ** 2), rel_tol=1e-10)
        assert math.isclose(l1, 0.5 * np.sum((X - W1 @ H1) ** 2), rel_tol=1e-8)
        assert math.isclose(gval("linear", "nmf_err1", sid), math.sqrt(2 * l1), rel_tol=1e-15)


# ---- mds --------------------------------------------------------------------------------------


def mds_sets():
    return REFS["mds"].group_inputs["set"].tolist()


def _b(X):
    D2 = sqd(X)
    return -0.5 * (D2 - D2.mean(0)[None, :] - D2.mean(1)[:, None] + D2.mean())


def test_classical_mds_spectrum_and_embedding():
    for sid in mds_sets():
        s = obj(sid)
        n = s.n
        X = np.asarray(s.X, float)
        B = _b(X)
        _, ev, f = vals("mds", "eigenvalue", sid, 0)
        assert len(ev) == n and np.all(np.diff(ev) <= 0) and np.all(ev >= 0), sid
        Xc = X - X.mean(axis=0)
        assert math.isclose(math.fsum(ev), np.sum(Xc**2), rel_tol=1e-12, abs_tol=1e-300), sid
        w = np.linalg.eigvalsh(B)[::-1]
        scale = max(abs(w[0]), 1e-300)
        assert np.allclose(ev, w, atol=1e-10 * scale), sid
        assert np.sum(ev == 0) >= n - s.p
        idx, bv, _ = vals("mds", "b_matrix", sid, 2)
        for t, v in zip(idx, bv):
            i, j = divmod(int(t), n)
            assert math.isclose(v, B[i, j], rel_tol=1e-9, abs_tol=1e-10 * scale), sid
        idx, emb, ef = vals("mds", "embedding", sid, 1)
        rank = S.lex_rank(X)
        for j in sorted({int(t) % 2 for t in idx}):
            col = emb[(idx % 2) == j]
            assert len(col) == n
            assert math.isclose(col @ col, ev[j], rel_tol=1e-12, abs_tol=1e-300), sid
            if ev[j] > 0:
                assert np.allclose(B @ col, ev[j] * col, atol=1e-8 * scale * np.abs(col).max()), sid
                check_sign(*column("mds", "embedding", sid, 1, j)[:2], rank, ef["sign_gap"][(idx % 2) == j][0])
        if len(idx) == 2 * n:
            E = D.matrix(REFS["mds"], "embedding", sid, (n, 2), 1)
            assert abs(E[:, 0] @ E[:, 1]) <= 1e-10 * scale, sid


def test_guttman_transform_and_stress():
    for sid in mds_sets():
        s = obj(sid)
        n = s.n
        X = np.asarray(s.X, float)
        Z = np.asarray(D.group_input(REFS["mds"], sid, "Z"), float)
        delta = np.sqrt(sqd(X))
        dz = np.sqrt(sqd(Z))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dz > 0, delta / dz, 0.0)
        np.fill_diagonal(ratio, 0.0)
        Bz = -ratio
        Bz[np.arange(n), np.arange(n)] += ratio.sum(axis=1)
        Zn = Bz @ Z / n
        Gr = D.matrix(REFS["mds"], "guttman", sid, (n, 2), 3)
        assert np.allclose(Gr, Zn, atol=1e-11 * max(np.abs(Zn).max(), np.abs(Z).max())), sid
        iu = np.triu_indices(n, 1)
        e = dz[iu] - delta[iu]
        raw = float(np.sum(e**2))
        g = D.group_value(REFS["mds"], "stress_raw", sid)
        assert math.isclose(g[0], raw, rel_tol=1e-9, abs_tol=1e-12 * g[2]["plus"]), sid
        kr = gval("mds", "stress_kruskal", sid)
        assert math.isclose(kr, math.sqrt(raw / np.sum(dz[iu] ** 2)), rel_tol=1e-9), sid
        sam = gval("mds", "stress_sammon", sid)
        if np.all(delta[iu] > 0):
            assert math.isclose(sam, np.sum(e**2 / delta[iu]) / np.sum(delta[iu]), rel_tol=1e-9), sid
        else:
            assert sam is None


def _blocks_ok(order, yv, fit):
    """An isotonic fit in this order: non-decreasing, and each block of equal fitted values holds the block's mean."""
    f = [fit[k] for k in order]
    assert all(a <= b for a, b in zip(f, f[1:]))
    t = 0
    while t < len(f):
        e = t
        while e + 1 < len(f) and f[e + 1] == f[t]:
            e += 1
        block = [yv[order[u]] for u in range(t, e + 1)]
        assert abs(math.fsum(block) / len(block) - f[t]) <= 1e-12 * max(1.0, abs(f[t]))
        t = e + 1


def test_isotonic_fits():
    done = 0
    for sid in mds_sets():
        r = REFS["mds"]
        idx, prim, _ = vals("mds", "iso_primary", sid, 4)
        if not len(idx):
            continue
        done += 1
        X = [[Fraction(v) for v in row] for row in np.asarray(obj(sid).X, float)]
        x = [sum((a - b) ** 2 for a, b in zip(X[i], X[j])) for i in range(len(X)) for j in range(i + 1, len(X))]
        y = np.asarray(D.group_input(r, sid, "y"), float)
        _, sec, _ = vals("mds", "iso_secondary", sid, 4)
        order_p = sorted(range(len(x)), key=lambda k: (x[k], y[k], k))
        _blocks_ok(order_p, y, prim)
        for a, b in zip(order_p, order_p[1:]):  # secondary: tied x share one value
            if x[a] == x[b]:
                assert sec[a] == sec[b]
        _blocks_ok(order_p, y, sec)
        assert np.sum((prim - y) ** 2) <= np.sum((sec - y) ** 2) * (1 + 1e-12)  # fewer constraints, no worse
    assert done >= 3


# ---- graph ------------------------------------------------------------------------------------


def graph_sets():
    return REFS["graph"].group_inputs["set"].tolist()


def knn_of(sid):
    s = obj(sid)
    k = int(D.group_input(REFS["graph"], sid, "k"))
    idx, d2, lo, f = D.per_set(REFS["graph"], "knn", sid, 0)
    f = dict(f, lo=lo.reshape(s.n, k))
    return s, k, d2.reshape(s.n, k), f["nbr"].reshape(s.n, k).astype(int), f


def test_knn_graphs():
    for sid in graph_sets():
        s, k, d2, nbr, f = knn_of(sid)
        n = s.n
        Df = sqd(s.X)
        rank = S.lex_rank(s.X)
        tie = f["tie"].reshape(n, k)
        bnd = f["boundary"].reshape(n, k)
        for i in range(n):
            assert i not in nbr[i] and len(set(nbr[i])) == k
            assert np.all(np.diff(d2[i]) >= 0)
            assert np.allclose(d2[i], Df[i, nbr[i]], rtol=1e-13, atol=0)
            for r in range(1, k):
                if d2[i, r] == d2[i, r - 1] and f["lo"][i, r] == f["lo"][i, r - 1]:
                    assert tie[i, r] and tie[i, r - 1] and rank[nbr[i, r - 1]] < rank[nbr[i, r]]
            others = np.delete(np.arange(n), i)
            order = others[np.lexsort((rank[others], Df[i, others]))]
            nxt = Df[i, order[k]] if n - 1 > k else math.inf
            if not bnd[i, 0] and nxt > d2[i, -1] * (1 + 1e-12):
                assert set(order[:k]) == set(nbr[i]), (sid, i)
        _, dist, _ = vals("graph", "knn_dist", sid, 0)
        assert np.allclose(dist.reshape(n, k) ** 2, d2, rtol=1e-15)


def test_lle_weights():
    for sid in graph_sets():
        s, k, _, nbr, _ = knn_of(sid)
        X = np.asarray(s.X, float)
        _, w, _ = vals("graph", "lle_weight", sid, 0)
        w = w.reshape(s.n, k)
        for i in range(s.n):
            assert math.isclose(math.fsum(w[i]), 1.0, abs_tol=1e-14 * max(1.0, np.abs(w[i]).max()))
            C = X[nbr[i]] - X[i]
            G = C @ C.T
            tr = np.trace(G)
            G = G + (D.REG * tr if tr > 0 else D.REG) * np.eye(k)
            wf = np.linalg.solve(G, np.ones(k))
            wf /= wf.sum()
            assert np.allclose(w[i], wf, atol=D.lle_bound(k, np.abs(w[i]).max()) * 1e3), (sid, i)


def _weights(n, nbr, variant):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, nbr[i]] = 1.0
    return (A + A.T) / 2 if variant == "lapm" else np.maximum(A, A.T)


def test_laplacians():
    for sid in graph_sets():
        s, k, _, nbr, _ = knn_of(sid)
        n = s.n
        rank = S.lex_rank(s.X)
        for variant in ("lapm", "lapu"):
            W = _weights(n, nbr, variant)
            deg = W.sum(1)
            L = np.eye(n) - W / np.sqrt(np.outer(deg, deg))
            _, ev, f = vals("graph", f"{variant}_eigenvalue", sid, 4)
            assert np.all(np.diff(ev) >= 0) and ev[0] == 0 and np.all(ev <= 2), sid
            c = int(f["mult"][0])
            assert np.all(ev[:c] == 0) and (len(ev) == c or ev[c] > 0)
            w = np.linalg.eigvalsh(L)
            assert np.allclose(ev, w[: len(ev)], atol=1e-11), (sid, variant)
            idx, emb, ef = vals("graph", f"{variant}_embedding", sid, 5)
            for j in sorted({int(t) % 2 for t in idx}):
                col = emb[(idx % 2) == j]
                u = col * np.sqrt(deg)
                assert math.isclose(u @ u, 1.0, rel_tol=1e-13)
                assert np.allclose(L @ u, ev[j + 1] * u, atol=1e-10), (sid, variant, j)
                check_sign(
                    *column("graph", f"{variant}_embedding", sid, 5, j)[:2], rank, ef["sign_gap"][(idx % 2) == j][0]
                )


def test_isomap_geodesics_and_spectrum():
    done = 0
    for sid in graph_sets():
        idx, geo, _ = vals("graph", "geodesic", sid, 1)
        if not len(idx):
            continue
        done += 1
        s, k, d2, nbr, _ = knn_of(sid)
        n = s.n
        G = np.zeros((n, n))
        for i in range(n):
            G[i, nbr[i]] = np.sqrt(d2[i])
        G = np.maximum(G, G.T)
        P = shortest_path(G, method="D", directed=False)
        Df = np.sqrt(sqd(s.X))
        for t, v in zip(idx, geo):
            a, b = divmod(int(t), n)
            assert math.isclose(v, P[a, b], rel_tol=1e-12), (sid, a, b)
            assert v >= Df[a, b] * (1 - 1e-15)
        K = -0.5 * P**2
        K = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
        w = np.linalg.eigvalsh(K)[::-1]
        _, ev, _ = vals("graph", "isomap_eigenvalue", sid, 2)
        assert np.allclose(ev, w[: len(ev)], atol=1e-9 * abs(w[0])), sid
        idx, emb, ef = vals("graph", "isomap_embedding", sid, 3)
        rank = S.lex_rank(s.X)
        for j in sorted({int(t) % 2 for t in idx}):
            col = emb[(idx % 2) == j]
            assert math.isclose(col @ col, ev[j], rel_tol=1e-12)
            assert np.allclose(K @ col, ev[j] * col, atol=1e-7 * abs(w[0]) * np.abs(col).max()), sid
            check_sign(*column("graph", "isomap_embedding", sid, 3, j)[:2], rank, ef["sign_gap"][(idx % 2) == j][0])
    assert done >= 5


# ---- sne --------------------------------------------------------------------------------------


def tsne_sets():
    return [s for s in REFS["sne"].group_inputs["set"].tolist() if "|perp" in s]


def _neighbour_d2(s, k):
    Df = sqd(s.X)
    rank = S.lex_rank(s.X)
    out = []
    for i in range(s.n):
        others = np.delete(np.arange(s.n), i)
        order = others[np.lexsort((rank[others], Df[i, others]))][:k]
        out.append((order, Df[i, order]))
    return out


def _cond_p(d2, beta):
    e = np.exp(-beta * (d2 - d2.min()))
    return e / e.sum()


def _p_matrix(sid):
    s = obj(sid)
    n = s.n
    _, beta, f = vals("sne", "beta", sid, 0)
    k = int(f["k"][0])
    P = np.zeros((n, n))
    for i, (order, d2) in enumerate(_neighbour_d2(s, k)):
        P[i, order] = _cond_p(d2, beta[i])
    return P, beta, k


def test_tsne_betas_and_p():
    for sid in tsne_sets():
        s = obj(sid)
        n = s.n
        perp = float(D.group_input(REFS["sne"], sid, "perplexity"))
        P, beta, k = _p_matrix(sid)
        assert np.all(beta > 0)
        for i, (_, d2) in enumerate(_neighbour_d2(s, k)):
            p = _cond_p(d2, beta[i])
            H = -np.sum(p[p > 0] * np.log(p[p > 0]))
            assert abs(H - math.log(perp)) <= 1e-9, (sid, i)
        idx, pc, _ = vals("sne", "p_cond", sid, 1)
        rows = sorted({int(t) // n for t in idx})
        for i in rows:
            sel = (idx // n) == i
            assert math.isclose(math.fsum(pc[sel]), 1.0, abs_tol=1e-14), (sid, i)
            assert np.allclose(pc[sel], P[i, idx[sel] % n], rtol=1e-9, atol=1e-300)
        idx, ps, _ = vals("sne", "p_sym", sid, 2)
        Ps = (P + P.T) / (2 * n)
        assert np.allclose(ps, Ps[idx // n, idx % n], rtol=1e-9, atol=1e-300), sid


def _kl_and_grad(P, Y, sklearn_form=False):
    n, d = Y.shape
    dof = max(d - 1, 1)
    d2 = sqd(Y)
    base = 1 + d2 / dof
    W = base ** (-(dof + 1) / 2)
    np.fill_diagonal(W, 0.0)
    Qm = W / W.sum()
    m = P > 0
    kl = float(np.sum(P[m] * np.log(P[m] / Qm[m])))
    F = W if sklearn_form else 1 / base
    PQ = (P - Qm) * F
    np.fill_diagonal(PQ, 0.0)
    grad = (2 * (dof + 1) / dof) * (PQ.sum(1)[:, None] * Y - PQ @ Y)
    return kl, grad


def test_tsne_gradient_and_kl():
    gsets = [s for s in REFS["sne"].group_inputs["set"].tolist() if "|grad-" in s]
    assert gsets
    for sid in gsets:
        s = obj(sid)
        n = s.n
        psid = D.base(sid) + "|" + sid.split("|grad-")[1].rsplit("-", 2)[0]
        P, _, _ = _p_matrix(psid)
        P = (P + P.T) / (2 * n)
        Y = np.asarray(D.group_input(REFS["sne"], sid, "Y"), float)
        d = Y.shape[1]
        kl, grad = _kl_and_grad(P, Y)
        idx, gr, gf = vals("sne", "grad", sid, 3)
        G = gr.reshape(n, d)
        plus = gf["plus"].reshape(n, d)
        assert np.all(np.abs(G - grad) <= 1e-9 * plus + 1e-300), sid
        assert np.allclose(G.sum(axis=0), 0, atol=1e-10 * plus.max())  # translation invariance
        assert math.isclose(gval("sne", "kl", sid), kl, rel_tol=1e-9), sid
        if d == 3:  # the KL gradient, by finite differences, and sklearn's formula, which differs for dof != 1
            h = 1e-6 * np.abs(Y).max()
            for i, c in ((0, 0), (n // 2, 2)):
                E = np.zeros_like(Y)
                E[i, c] = h
                fd = (_kl_and_grad(P, Y + E)[0] - _kl_and_grad(P, Y - E)[0]) / (2 * h)
                assert math.isclose(G[i, c], fd, rel_tol=1e-4, abs_tol=1e-8 * plus.max())
            _, gsk = _kl_and_grad(P, Y, sklearn_form=True)
            Gs = vals("sne", "grad_sklearn", sid, 3)[1].reshape(n, d)
            assert np.allclose(Gs, gsk, rtol=1e-8, atol=1e-12 * plus.max())
            assert np.abs(Gs - G).max() > 1e-3 * np.abs(G).max()


def test_umap_rho_sigma_and_fuzzy_set():
    usets = [s for s in REFS["sne"].group_inputs["set"].tolist() if "|umap" in s]
    assert usets
    for sid in usets:
        s = obj(sid)
        n = s.n
        kk = int(D.group_input(REFS["sne"], sid, "n_neighbors"))
        nb = _neighbour_d2(s, kk - 1)
        dist = [np.sqrt(d2) for _, d2 in nb]
        allmean = sum(x.sum() for x in dist) / (n * kk)
        _, rho, _ = vals("sne", "rho", sid, 0)
        _, sig, sf = vals("sne", "sigma", sid, 0)
        memb = {}
        for i in range(n):
            nz = dist[i][dist[i] > 0]
            assert rho[i] == (nz[0] if len(nz) else 0.0) or math.isclose(rho[i], nz[0], rel_tol=1e-15)
            a = dist[i] - rho[i]
            if sf["rule"][i] == 0:
                psum = np.sum(np.where(a > 0, np.exp(-np.maximum(a, 0) / sig[i]), 1.0))
                assert abs(psum - math.log2(kk)) <= 1e-10, (sid, i)
            else:
                ref = D.MIN_K_DIST_SCALE * (dist[i].sum() / kk if rho[i] > 0 else allmean)
                assert math.isclose(sig[i], ref, rel_tol=1e-14), (sid, i)
            for j, x in zip(nb[i][0], a):
                memb[(i, j)] = 1.0 if x <= 0 or sig[i] == 0 else math.exp(-x / sig[i])
        idx, fz, _ = vals("sne", "fuzzy", sid, 4)
        if len(idx):
            assert np.all((fz >= 0) & (fz <= 1))
            for t, v in zip(idx, fz):
                i, j = divmod(int(t), n)
                a, b = memb.get((i, j), 0.0), memb.get((j, i), 0.0)
                assert math.isclose(v, a + b - a * b, rel_tol=1e-9, abs_tol=1e-15), (sid, i, j)


# ---- quality ----------------------------------------------------------------------------------


def _trust_float(X, E, k, rank):
    """Venna-Kaski's trustworthiness (and, swapped, continuity) with numpy: ranks by (squared distance, key)."""
    n = len(X)

    def ranks(A):
        if n <= 60:  # small sets carry exact ties that rounded squared distances would reorder: Fractions
            F = [[Fraction(v) for v in row] for row in np.asarray(A, float)]
            Dm = np.array([[sum((a - b) ** 2 for a, b in zip(F[i], F[j])) for j in range(n)] for i in range(n)])
        else:
            Dm = sqd(A)
        R = np.zeros((n, n), int)
        for i in range(n):
            others = np.delete(np.arange(n), i)
            order = others[np.lexsort((rank[others], Dm[i, others]))]
            R[i, order] = np.arange(1, n)
        return R

    RX, RE = ranks(X), ranks(E)

    def penalty(Ra, Rb):
        tot = 0
        for i in range(n):
            nb = np.nonzero((Rb[i] >= 1) & (Rb[i] <= k))[0]
            tot += int(np.sum(np.maximum(Ra[i, nb] - k, 0)))
        return tot

    return penalty(RX, RE), penalty(RE, RX)


def test_trustworthiness_and_continuity():
    for sid in REFS["quality"].group_inputs["set"].tolist():
        s = obj(sid)
        n = s.n
        k = int(D.group_input(REFS["quality"], sid, "n_neighbors"))
        E = np.asarray(D.group_input(REFS["quality"], sid, "E"), float)
        E = E.reshape(n, -1)
        denom = n * k * (2 * n - 3 * k - 1)
        st, sc = _trust_float(np.asarray(s.X, float), E, k, S.lex_rank(s.X))
        for method, tot, kind in (("trustworthiness", st, "t_point"), ("continuity", sc, "c_point")):
            hi, lo, f = D.group_value(REFS["quality"], method, sid)
            assert 0 <= hi <= 1
            assert int(f["sum"]) == tot, (sid, method)
            assert hi == float(1 - Fraction(2 * tot, denom)), (sid, method)  # correctly rounded
            _, per, _ = vals("quality", kind, sid, 0)
            assert int(per.sum()) == tot
        if sid.startswith("lattice/5x5|first2"):
            assert gval("quality", "trustworthiness", sid) == 1.0 == gval("quality", "continuity", sid)
