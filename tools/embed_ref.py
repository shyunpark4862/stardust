"""Reference values for sd.embed (docs/design/embed.md §4, §7.2) on the test data of tests/embed/samples.py:
exact rationals (python-flint's fmpq, fmpq_mat, fmpq_poly, Python's Fraction) and Arb (arb, arb_mat, acb_mat.eig).

    .venv/bin/python tools/embed_ref.py                     # every family, on 6 processes
    .venv/bin/python tools/embed_ref.py linear mds          # only these families
    .venv/bin/python tools/embed_ref.py --list              # the families and their items

writes tests/embed/ref/<family>.npz (layout: tests/reffile.py), read by tests/embed/refdata.py. A point has the
inputs "set" (the id in samples.py with "|..." for the variant: "|cov", "|kpca-rbf", "|k5", "|perp30-exact", ...),
"kind" (what the point is, per family below), "index" (its number in its kind) and "digest" (the data's digest); a
group is one item (one set and variant) and holds the given inputs the references were computed at (a starting
configuration Z, an embedding Y, a matrix W, ...). The data are doubles taken as exact rationals; a reference is
the true value for those doubles, rounded once (a double-double hi + lo). Squared Euclidean distances of doubles
are exact rationals, so every neighbour choice is made on exact values; a choice between values equal in exact
arithmetic follows the key of decision EM12 (the points' lexicographic ranks, samples.lex_rank) and is marked.

Eigenpairs. A rational symmetric matrix (or pencil A - lambda D with D a positive rational diagonal) has its exact
characteristic polynomial (fmpq_mat.charpoly of D^-1 A) whose roots (complex_roots) come with their exact
multiplicities; an Arb matrix has acb_mat.eig(multiple=True), its balls still overlapping at TIE_PREC bits counting
as a repeated eigenvalue. A simple eigenvalue's vector is the solution of the bordered system (A - lambda D) w = 0,
w_k = 1 (row and column k removed, k the largest entry), an enclosure valid over the whole eigenvalue ball, then
D^(1/2) w normalised. A repeated eigenvalue has no vector (the definition is ill-posed there) and the group's input
"note" says which were left out. The sign rule of decision EM14: the entry of largest |value| is positive, an exact
tie (balls overlapping at TIE_PREC) by the smaller key: the column index for PCA's components, the points'
lexicographic rank for an embedding's columns; field sign_gap is the runner-up's relative distance (0 a tie).
Eigenvalues are settled to 2^-110 of the largest one (an absolute floor; exact zeros are exact), vector entries to
2^-110 of the vector's largest entry.

    linear     per set (p <= 50): "|cov" the exact centred Gram C (divisor n - 1): variance (kind 0, index j <
               min(n, p): explained variance, fields mult, gap = the distance to the nearest other eigenvalue (0 for
               a repeated one): the Davis-Kahan denominator), singular (kind 0: sqrt((n - 1) lambda), the centred
               data's singular values), ratio (kind 0: lambda / trace C), component (kind 1, index j p + c, j < the
               set's d: the unit eigenvectors as rows, EM14 with the column index as key; fields sign_gap, gap), over
               the group total_variance (trace C) and noise_variance (field d: the mean of lambda_d .. lambda_{m-1},
               sklearn's noise_variance_; 0 when d = m); "|cor" the same for the correlation matrix (R prcomp
               scale. = TRUE; sets with a zero-variance column left out). Kernel PCA "|kpca-<kernel>" (linear and
               poly exact, rbf Arb; gamma = 1 / p, degree 3, coef0 1, sklearn's defaults): the doubly centred kernel
               J K J, kpca_eigenvalue (kind 2, index j < 8; fields mult, gap), kpca_embedding (kind 3, index i 2 + j:
               sqrt(lambda_j) u_j, EM14 on the column with the lex key; field sign_gap), kpca_kernel (kind 4, index
               i n + j, i <= j, n <= 40: the centred kernel). Factor analysis "|fa-q<q>-<psi>" at the group input
               psi (R factanal's start (1 - q / 2p) / diag(R^-1) rounded, or a fixed other point): fa_eigen (kind 5,
               index j: the eigenvalues of Psi^-1/2 R Psi^-1/2), fa_gradient (kind 6, index j: R's FAgr) and over
               the group fa_objective (R's FAfn: sum_{j >= q} (e_j - log e_j) + q - p). FastICA "|ica" (logcosh,
               alpha 1) at the group inputs Xw (the whitened data, n x d, as sklearn whitens, rounded) and w_init
               (d x d): ica_w0 (kind 7, index i d + j: the symmetric decorrelation (W W^T)^-1/2 W of w_init) and
               ica_w1 (kind 7: one parallel step from it, sklearn's _ica_par), over the group ica_lim (sklearn's
               max | |diag(W1 W0^T)| - 1 |). NMF "|nmf-<s>": one multiplicative-update step (Frobenius, sklearn's
               order: W, then H with the new W) from the group inputs W0, H0, exact: nmf_w1 (kind 8, index i d + j),
               nmf_h1 (kind 9, index j p + c), over the group nmf_loss0, nmf_loss1 (1/2 ||X - W H||_F^2, exact) and
               nmf_err0, nmf_err1 (||X - W H||_F, sklearn's reconstruction_err_)
    mds        per set (n <= 300): B = -1/2 J D^2 J from the exact squared distances (checked equal to Xc Xc^T),
               eigenvalue (kind 0, index j < n: all n eigenvalues, from the p x p Gram Xc^T Xc and exact zeros;
               fields mult, gap), embedding (kind 1, index i 2 + j: Xc v_j = sqrt(lambda_j) u_j, EM14 with the lex
               key, a zero eigenvalue's axis exactly 0; field sign_gap), b_matrix (kind 2, index i n + j, i <= j,
               n <= 40); at the group input Z (n x 2): guttman (kind 3, index i 2 + c: one SMACOF Guttman transform
               Z' = n^-1 B(Z) Z, b_ij = -delta_ij / d_ij(Z), b_ij = 0 where d_ij(Z) = 0) and over the group stress_raw
               (sum_{i<j} (d - delta)^2), stress_kruskal (sqrt(raw / sum d^2)), stress_sammon (sum (d - delta)^2 /
               delta / sum delta; not where some delta is 0); for n <= 80 the isotonic regression of the group input
               y (the doubles of d_ij(Z), condensed) on x = delta (its order: the exact squared distances):
               iso_primary (kind 4, index the condensed pair: Kruskal's primary approach, ties of x ordered by (y,
               pair)) and iso_secondary (kind 4: ties of x pooled first, as sklearn's IsotonicRegression does)
    graph      per set and k in (5, 15): the exact k-NN (self excluded, ties by (distance, key)): knn (kind 0,
               index i k + r: the squared distance; fields nbr (the input index), tie (the distance equals the one
               before or after it in the row, the (k+1)-th included), boundary (the row's k-th and (k+1)-th
               distances are equal: the set depends on the key)), knn_dist (kind 0: the distance), lle_weight (kind
               0: sklearn's barycenter weights, the regularised local Gram G + reg tr(G) I, reg = 1e-3, solved and
               normalised, exact); the normalised Laplacian I - D^-1/2 W D^-1/2 of two symmetrisations of the 0/1
               graph A: mean W = (A + A^T) / 2 (sklearn's SpectralEmbedding, whose kneighbors_graph includes the
               point itself, so its n_neighbors = k + 1 is this k; the Laplacian ignores the diagonal) and union
               W = max(A, A^T), the exact pencil (D - W, D): lapm_eigenvalue, lapu_eigenvalue (kind 4, index j < 6,
               ascending; fields mult, gap; 0 exact with the multiplicity of the connected components),
               lapm_embedding, lapu_embedding (kind 5, index i d + j: sklearn's columns u_{j+1} / sqrt(deg), the
               first dropped, EM14 with the lex key; field sign_gap); for k = 5 on
               connected graphs, Isomap: geodesic (kind 1, index s n + t: Dijkstra on the undirected graph in Arb,
               certified by the edge inequalities; every pair s < t for n <= 60, else the rows of sources 0 .. 7),
               isomap_eigenvalue (kind 2, index j < 10: of -1/2 J G^2 J; fields mult, gap), isomap_embedding (kind
               3, index i 2 + j; field sign_gap)
    sne        t-SNE "|perp<P>-<knn|exact>": beta (kind 0, index i: the root of H(P_i) = log(perplexity) in nats
               over the squared distances to the min(n - 1, floor(3 P + 1)) nearest (knn, sklearn's barnes_hut
               path) or to every other point (exact); field k), p_cond (kind 1, index i n + j: the conditional
               p_{j|i} at that root, for the rows of the first ROWS points), p_sym (kind 2, index i n + j: (p_{j|i} +
               p_{i|j}) / 2n on the same rows); "|grad-...-<Y>-d<d>" at the group input Y (n x d) with the P of
               the same item: grad (kind 3, index i d + c: the gradient of KL(P || Q), Student t with dof = max(d
               - 1, 1)), for d = 3 grad_sklearn (kind 3: sklearn's _kl_divergence formula, which multiplies by the
               kernel w_ij instead of (1 + d_ij^2 / dof)^-1 and so is not the gradient when dof != 1), over the group
               kl (sum_{i != j} p_ij log(p_ij / q_ij)); UMAP "|umap15": rho, sigma (kind 0, index i: umap-learn's
               smooth_knn_dist with local_connectivity 1 over the 15 nearest including the point itself, sigma the
               exact root of sum_{j >= 1} exp(-max(0, d_j - rho) / sigma) = log2(15) or, below it, umap-learn's
               floor 1e-3 mean distance; field rule (1 where the floor applies; root 0 where none exists)), fuzzy
               (kind 4, index i n + j, i < j, n <= 150: the fuzzy union a + b - a b of the membership strengths)
    quality    per (set, embedding E, k in (5, 10)): trustworthiness and continuity over the groups (exact: the
               integer rank sums once rounded; fields sum, ties (the rows where a tie at a rank that counts
               decides by the key)), t_point and c_point (kind 0, index i: the point's integer rank sum); ranks by
               the exact squared distances, ties by the lex key of X in both spaces; E the group input (a fixed
               formula of X: its first columns, the manifold's own coordinates, a rounded PCA)

Left for the milestones that implement them (they are steps of algorithms, not definitions this tool can hold
without the implementation's choices): t-SNE's Barnes-Hut and FFT gradient approximations (EM6, EM8), one UMAP
optimisation step and its a, b (a curve fit, not a truth; EM7), one Sammon step (EM4), NMF's coordinate-descent
step (EM3), LLE's modified, hessian and ltsa local steps and the LLE embedding (EM5), Isomap on disconnected graphs
(sklearn's component joining, EM5), the Cailliez constant of cmdscale(add = TRUE) (EM4), sets with p > 50 for the
linear family (lowrank/r2p200; EM1 adds the dual n x n route if it needs it).

Settling: an exact rational is exact; an Arb value is settled to 2^-110 (relative, or of its floor). Each task runs
at 256 bits and doubles to 2048 until every value settles. Workers are spawned (a fork after the parent's BLAS
threads deadlocked the replaced workers), replaced every 4 tasks (maxtasksperchild), each task under a time limit
(SIGALRM). No mpmath (the project's rule). The console output also
goes to build/logs/embed_ref/.
"""

from __future__ import annotations

import argparse
import heapq
import math
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled  # noqa: E402
from flint import acb_mat, arb, arb_mat, fmpq, fmpq_mat  # noqa: E402

OUT = ROOT / "tests" / "embed" / "ref"
TOOL = "tools/embed_ref.py"
SECONDS = 3600  # a task's time limit
PRECS = (256, 512, 1024, 2048)
TIE_PREC = 1024  # balls still overlapping at this precision count as equal (a repeated eigenvalue, a sign tie)
MAX_P = 50  # the linear family's column cap
KPCA_EIGS = 8
ISO_EIGS = 10
LAP_EIGS = 6
ROWS = 10  # the t-SNE rows of P kept for sets of more than 80 points
REG = 1e-3  # LLE's reg (sklearn's default)
MIN_K_DIST_SCALE = 1e-3  # umap-learn's floor of sigma
UMAP_K = 15


class _Timeout(Exception):
    pass


class Unresolvable(Exception):
    """acb_mat.eig cannot isolate the eigenvalues even at TIE_PREC (a large cluster of exactly equal ones, as
    exactly repeated points give): the part is left out and noted."""


def _alarm(*_):
    raise _Timeout


def S():
    from tests.embed import samples

    return samples


# ---------------------------------------------------------------- exact helpers


def Q(v) -> fmpq:
    return fmpq(*float(v).as_integer_ratio())


def fr(q) -> Fraction:
    if isinstance(q, Fraction):
        return q
    if isinstance(q, int):
        return Fraction(q)
    return Fraction(int(q.p), int(q.q))


def dd_q(q):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded."""
    q = fr(q)
    try:
        hi = float(q)
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def arb_q(q):
    if isinstance(q, Fraction):
        q = fmpq(q.numerator, q.denominator)
    if isinstance(q, int):
        return arb(q)
    return arb(q.p) / arb(q.q)


def exact_rows(X):
    return [[Q(v) for v in row] for row in np.asarray(X, float)]


def sqdist(a, b):
    s = fmpq(0)
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def sqdist_rows(P):
    """The exact squared distances as a full list of lists (0 on the diagonal)."""
    n = len(P)
    D = [[fmpq(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = sqdist(P[i], P[j])
    return D


def centred(P):
    n, p = len(P), len(P[0])
    mu = [sum((P[i][c] for i in range(n)), fmpq(0)) / n for c in range(p)]
    return [[P[i][c] - mu[c] for c in range(p)] for i in range(n)], mu


def mid(v) -> float:
    return float(v.mid()) if isinstance(v, arb) else float(fr(v))


def is_exact_value(v):
    return isinstance(v, (fmpq, Fraction, int)) or (isinstance(v, float) and not isinstance(v, bool))


def knn_rows(D2, rank, k):
    """Per point the k nearest others by (exact squared distance, key): [(d2, j)], and the (k+1)-th's d2."""
    n = len(D2)
    out, nxt = [], []
    for i in range(n):
        row = sorted(((D2[i][j], rank[j], j) for j in range(n) if j != i), key=lambda t: (t[0], t[1]))
        out.append([(d, j) for d, _, j in row[:k]])
        nxt.append(row[k][0] if len(row) > k else None)
    return out, nxt


# ---------------------------------------------------------------- the values of a task


class Out:
    """The values of one group: per point (kind, index) and per group, by method; values are exact (fmpq,
    Fraction, int, float) or arb balls with a floor (settled to 2^-110 relative or of the floor)."""

    def __init__(self, sid, digest):
        self.sid = sid
        self.digest = digest
        self.points = {}
        self.groups = {}
        self.ginputs = {}

    def at(self, method, kind, index, value, floor=0, **fields):
        self.points.setdefault((int(kind), int(index)), {})[method] = (value, floor, fields)

    def group(self, method, value, floor=0, **fields):
        self.groups[method] = (value, floor, fields)

    def result(self):
        """{"set", "digest", "ginputs", "points": [(kind, index, {method: (hi, lo, fields)})], "gmethods"};
        raises NotSettled when an arb value has not settled."""
        return {
            "set": self.sid,
            "digest": self.digest,
            "ginputs": self.ginputs,
            "points": [(k, i, {m: _entry(*e) for m, e in ms.items()}) for (k, i), ms in sorted(self.points.items())],
            "gmethods": {m: _entry(*e) for m, e in self.groups.items()},
        }


def _entry(value, floor, fields):
    fields = {k: float(v) for k, v in fields.items()}
    if is_exact_value(value):
        hi, lo = (float(value), 0.0) if isinstance(value, float) else dd_q(value)
        return hi, lo, fields
    if not arbref.settled(value, floor=float(floor)):
        raise NotSettled(f"value {value} (floor {floor})")
    hi, lo = arbref.to_dd(value)
    return hi, lo, fields


def at_precisions(fn):
    """fn(tie_ok) -> a list of Out at doubling precision until every value settles."""
    for prec in PRECS:
        with arbref.precision(prec):
            try:
                outs = fn(prec >= TIE_PREC)
                return [o.result() for o in outs]
            except (NotSettled, ZeroDivisionError, ValueError) as e:
                last = e
    raise NotSettled(f"not settled by {PRECS[-1]} bits: {last}")


# ---------------------------------------------------------------- eigenpairs


def rational_spectrum(f):
    """The roots of an exact real-rooted characteristic polynomial (fmpq_poly) at the current precision,
    descending, each repeated by its exact multiplicity: [(arb, mult)]."""
    out = []
    for z, m in f.complex_roots():
        if hasattr(z, "imag") and not z.imag.contains(0):
            raise AssertionError(f"a complex root {z}: the matrix is not symmetric")
        v = z.real if hasattr(z, "real") else arb(z)
        out.extend([(v, int(m))] * int(m))
    out.sort(key=lambda t: -mid(t[0]))
    return out


def arb_spectrum(M, want, tie_ok, zeros=0, ascending=False):
    """The eigenvalues of a symmetric arb_mat by acb_mat.eig(multiple=True), descending (ascending: ascending),
    as [(arb, mult)]: balls overlapping their neighbours count as one repeated eigenvalue when tie_ok, else
    NotSettled (a higher precision may separate them) -- only for the first `want` positions and their neighbours.
    zeros: the exact multiplicity of the eigenvalue 0 at the start (ascending; a Laplacian's components)."""
    try:
        E = acb_mat(M).eig(multiple=True)
    except ValueError:
        if tie_ok:
            raise Unresolvable("acb_mat.eig could not isolate the eigenvalues") from None
        raise NotSettled("eig") from None
    vals = sorted((z.real for z in E), key=lambda v: mid(v) if ascending else -mid(v))
    n = len(vals)
    out = [None] * n
    for j in range(min(zeros, n)):
        out[j] = (arb(0), zeros)
    j = min(zeros, n)
    if 0 < zeros < n and (vals[zeros].overlaps(vals[zeros - 1]) or vals[zeros].contains(0)):
        if tie_ok:
            raise Unresolvable("the smallest nonzero eigenvalue is not separated from 0")
        raise NotSettled("zero cluster")
    while j < n:
        e = j + 1
        while e < n and vals[e].overlaps(vals[e - 1]):
            e += 1
        size = e - j
        if size > 1 and j < want:
            if not tie_ok:
                raise NotSettled("eigenvalues not separated")
            u = vals[j]
            for t in range(j + 1, e):
                u = u.union(vals[t])
            for t in range(j, e):
                out[t] = (u, size)
        else:
            for t in range(j, e):
                out[t] = (vals[t], 1 if size == 1 else size)
        j = e
    return out


def gaps(spec):
    """Per position the distance to the nearest different eigenvalue (0 for a repeated one), in Arb (a gap far
    below the eigenvalues' size survives). spec is sorted, so the nearest is the next different one either side."""
    n = len(spec)
    out = []
    for j, (v, m) in enumerate(spec):
        if m > 1:
            out.append(0.0)
            continue
        g = None
        for t in (j - 1, j + 1):
            if 0 <= t < n:
                d = abs(v - spec[t][0])
                g = d if g is None or d.mid() < g.mid() else g
        out.append(0.0 if g is None else float(g.mid()))
    return out


def eigvec(A, lam, dg=None, guess=None):
    """The eigenvector of the simple eigenvalue lam of the pencil A - lam D (A an arb_mat, dg D's diagonal as
    arbs or None for the identity), with D^(1/2) w normalised to unit length: the bordered system with w_k = 1, k
    the largest entry of the guess (a float vector) and then of the solution itself."""
    n = A.nrows()
    if n == 1:
        return [arb(1)]
    order = list(np.argsort(-np.abs(guess), kind="stable")) if guess is not None else list(range(n))
    tried = set()
    for k in order[:4] + [None]:
        if k is None:
            break
        k = int(k)
        for _ in range(2):
            if k in tried:
                break
            tried.add(k)
            idx = [i for i in range(n) if i != k]
            rows = []
            for i in idx:
                r = [A[i, j] for j in idx]
                pos = idx.index(i)
                r[pos] = r[pos] - lam * (dg[i] if dg is not None else 1)
                rows.append(r)
            try:
                sol = arb_mat(rows).solve(arb_mat([[-A[i, k]] for i in idx]))
            except ZeroDivisionError:
                break
            w = [None] * n
            w[k] = arb(1)
            for t, i in enumerate(idx):
                w[i] = sol[t, 0]
            big = int(np.argmax([abs(mid(v)) for v in w]))
            if abs(mid(w[big])) > 1.5 and big not in tried:  # pivot on the largest entry
                k = big
                continue
            u = [w[i] * dg[i].sqrt() for i in range(n)] if dg is not None else w
            nrm = sum((x * x for x in u), arb(0)).sqrt()
            return [x / nrm for x in u]
    raise NotSettled("eigenvector: no pivot")


def sign_rule(v, keys, tie_ok):
    """(+1 or -1, sign_gap) of decision EM14 for the vector v (arbs): the entry of largest |value| positive, an
    exact tie by the smaller key; sign_gap the runner-up's relative distance, in Arb (0 for a tie)."""
    a = [abs(x) for x in v]
    order = sorted(range(len(v)), key=lambda i: a[i].mid(), reverse=True)
    top = order[0]
    cands = [i for i in order if a[i].overlaps(a[top])]
    if len(cands) > 1:
        if not tie_ok:
            raise NotSettled("sign tie")
        top = min(cands, key=lambda i: keys[i])
        gap = 0.0
    elif len(order) > 1:
        gap = float(((a[top] - a[order[1]]) / a[top]).mid())
    else:
        gap = 1.0
    if v[top].contains(0):
        raise NotSettled("sign: the largest entry is not separated from 0")
    return (1 if v[top] > 0 else -1), gap


def rat_matrix(rows):
    return fmpq_mat([[q for q in r] for r in rows])


def float_matrix(M):
    return np.array([[float(fr(M[i, j])) for j in range(M.ncols())] for i in range(M.nrows())])


def float_guesses(Mf, dgf=None):
    """numpy's eigenpairs (descending) of the symmetric form of the pencil (Mf, diag dgf): the pivot guesses for
    w (D^-1/2 u)."""
    if dgf is None:
        w, V = np.linalg.eigh(Mf)
        return w[::-1], V[:, ::-1]
    s = 1.0 / np.sqrt(dgf)
    w, V = np.linalg.eigh(Mf * s[:, None] * s[None, :])
    return w[::-1], (V * s[:, None])[:, ::-1]


# ---------------------------------------------------------------- linear


def linear_sets():
    out = []
    for pid in S().point_ids(tag="linear", exclude=("large",)):
        if S().points(pid).p <= MAX_P:
            out.append(pid)
    return out


def _gram(s):
    P = exact_rows(s.X)
    Xc, mu = centred(P)
    XcM = rat_matrix(Xc)
    C = XcM.transpose() * XcM
    C = C * fmpq(1, s.n - 1)
    return P, Xc, C


def pca_task(pid, variant):
    s = S().points(pid)
    n, p = s.n, s.p
    _, _, C = _gram(s)
    dg = [C[j, j] for j in range(p)] if variant == "cor" else None
    if dg is not None and any(v == 0 for v in dg):
        raise ValueError(f"{pid}: a zero-variance column")
    M = C if dg is None else fmpq_mat([[C[i, j] / dg[i] for j in range(p)] for i in range(p)])
    f = M.charpoly()
    m = min(n, p)
    d = min(s.d, m)
    Cf = float_matrix(C)
    _, G = float_guesses(Cf, None if dg is None else np.diag(Cf))
    trace = sum((C[j, j] for j in range(p)), fmpq(0)) if dg is None else fmpq(p)

    def compute(tie_ok):
        out = Out(f"{pid}|{variant}", s.digest)
        spec = rational_spectrum(f)
        gp = gaps(spec)
        lmax = abs(mid(spec[0][0])) or 1.0
        smax = math.sqrt((n - 1) * lmax)
        for j in range(m):
            lam, mult = spec[j]
            out.at("variance", 0, j, lam, lmax, mult=mult, gap=gp[j])
            sv = lam if (lam.is_exact() and lam.is_zero()) else ((n - 1) * lam).sqrt()
            out.at("singular", 0, j, sv, smax)
            out.at("ratio", 0, j, lam / arb_q(trace), lmax / abs(mid(arb_q(trace))))
        A = arb_mat(C)
        dga = [arb_q(v) for v in dg] if dg is not None else None
        skipped = []
        for j in range(d):
            lam, mult = spec[j]
            if mult > 1 or not lam > 0:
                skipped.append(j)
                continue
            u = eigvec(A, lam, dga, G[:, j])
            sg, gap = sign_rule(u, list(range(p)), tie_ok)
            for c in range(p):
                out.at("component", 1, j * p + c, sg * u[c], 1.0, sign_gap=gap, gap=gp[j])
        out.group("total_variance", trace)
        noise = sum((spec[j][0] for j in range(d, m)), arb(0)) / (m - d) if d < m else fmpq(0)
        out.group("noise_variance", noise, lmax, d=d)
        out.ginputs["note"] = f"no component {skipped} (repeated or zero eigenvalue)" if skipped else ""
        return [out]

    return at_precisions(compute)


def _kernel_exact(P, kernel, gamma):
    n = len(P)
    K = [[fmpq(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dot = sum((a * b for a, b in zip(P[i], P[j])), fmpq(0))
            v = dot if kernel == "linear" else (gamma * dot + 1) ** 3
            K[i][j] = K[j][i] = v
    return K


def _centre_exact(K):
    n = len(K)
    r = [sum(row, fmpq(0)) / n for row in K]
    t = sum(r, fmpq(0)) / n
    return [[K[i][j] - r[i] - r[j] + t for j in range(n)] for i in range(n)]


def kpca_task(pid, kernel):
    s = S().points(pid)
    n, p = s.n, s.p
    P = exact_rows(s.X)
    rank = S().lex_rank(s.X).tolist()
    gamma = fmpq(1, p)
    want = min(n, KPCA_EIGS)
    d = min(2, n)
    if kernel in ("linear", "poly"):
        Kc = _centre_exact(_kernel_exact(P, kernel, gamma))
        KcM = rat_matrix(Kc)
        f = KcM.charpoly()
        Kf = float_matrix(KcM)
    else:
        D2 = sqdist_rows(P)
        X = np.asarray(s.X)
        D2f = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        Kf0 = np.exp(-D2f / p)
        Kf = Kf0 - Kf0.mean(0)[None, :] - Kf0.mean(1)[:, None] + Kf0.mean()
    _, G = float_guesses(Kf)

    def compute(tie_ok):
        out = Out(f"{pid}|kpca-{kernel}", s.digest)
        if kernel in ("linear", "poly"):
            spec = rational_spectrum(f)
            A = arb_mat(KcM)
        else:
            K = [[(-arb_q(gamma) * arb_q(D2[i][j])).exp() for j in range(n)] for i in range(n)]
            r = [sum(row, arb(0)) / n for row in K]
            t = sum(r, arb(0)) / n
            Kca = [[K[i][j] - r[i] - r[j] + t for j in range(n)] for i in range(n)]
            A = arb_mat(Kca)
            try:
                spec = arb_spectrum(A, want + 1, tie_ok)
            except Unresolvable:
                spec = None
        skipped = []
        if spec is not None:
            skipped = _kpca_eigen(out, spec, A, G, rank, n, want, d, tie_ok)
        if n <= 40:
            for i in range(n):
                for j in range(i, n):
                    v = Kc[i][j] if kernel != "rbf" else A[i, j]
                    out.at("kpca_kernel", 4, i * n + j, v, 1.0)
        out.ginputs["gamma"] = np.float64(1.0 / p)
        if spec is None:
            out.ginputs["note"] = "no eigenpairs: acb_mat.eig cannot isolate the exactly repeated zero eigenvalues"
        else:
            out.ginputs["note"] = f"no embedding column {skipped} (repeated or zero eigenvalue)" if skipped else ""
        return [out]

    return at_precisions(compute)


def _kpca_eigen(out, spec, A, G, rank, n, want, d, tie_ok):
    gp = gaps(spec)
    lmax = abs(mid(spec[0][0])) or 1.0
    for j in range(want):
        lam, mult = spec[j]
        out.at("kpca_eigenvalue", 2, j, lam, lmax, mult=mult, gap=gp[j])
    skipped = []
    for j in range(d):
        lam, mult = spec[j]
        if mult > 1 or not lam > 0:
            skipped.append(j)
            continue
        u = eigvec(A, lam, None, G[:, j])
        col = [x * lam.sqrt() for x in u]
        sg, gap = sign_rule(col, rank, tie_ok)
        scale = max(abs(mid(x)) for x in col)
        for i in range(n):
            out.at("kpca_embedding", 3, i * 2 + j, sg * col[i], scale, sign_gap=gap)
    return skipped


def fa_start(C, p, q):
    """R factanal's start (1 - 0.5 q / p) / diag(R^-1) as exact rationals: diag(R^-1)_i = c_ii (C^-1)_ii."""
    Ci = C.inv()
    return [(1 - fmpq(q, 2 * p)) / (C[i, i] * Ci[i, i]) for i in range(p)]


def fa_task(pid, q, variant):
    s = S().points(pid)
    p = s.p
    _, _, C = _gram(s)
    start = [float(fr(v)) for v in fa_start(C, p, q)]
    if variant == "start":
        psi = np.array(start)
    else:  # a fixed other point inside (lower, 1)
        psi = np.array([0.5 * v + 0.2 + 0.03 * (i % 3) for i, v in enumerate(start)])
    psiq = [Q(v) for v in psi]
    gdiag = [psiq[i] * C[i, i] for i in range(p)]
    M = fmpq_mat([[C[i, j] / gdiag[i] for j in range(p)] for i in range(p)])
    f = M.charpoly()
    Cf = float_matrix(C)
    _, G = float_guesses(Cf, np.array([float(fr(v)) for v in gdiag]))

    def compute(tie_ok):
        out = Out(f"{pid}|fa-q{q}-{variant}", s.digest)
        out.ginputs["psi"] = psi
        out.ginputs["q"] = np.float64(q)
        spec = rational_spectrum(f)
        gp = gaps(spec)
        emax = abs(mid(spec[0][0]))
        for j in range(p):
            out.at("fa_eigen", 5, j, spec[j][0], emax, mult=spec[j][1], gap=gp[j])
        obj = sum((e - e.log() for e, _ in spec[q:]), arb(0)) + (q - p)
        out.group("fa_objective", obj, 1.0)
        A = arb_mat(C)
        dga = [arb_q(v) for v in gdiag]
        U = []
        for j in range(q):
            lam, mult = spec[j]
            if mult > 1:
                raise ValueError(f"{pid}: a repeated top eigenvalue")
            U.append(eigvec(A, lam, dga, G[:, j]))
        for i in range(p):
            ps = arb_q(psiq[i])
            load = sum((U[j][i] ** 2 * (spec[j][0] - 1 if spec[j][0] > 1 else arb(0)) for j in range(q)), arb(0))
            g = (ps * load + ps - 1) / (ps * ps)
            out.at("fa_gradient", 6, i, g, 1.0 / float(psi[i]) ** 2)
        return [out]

    return at_precisions(compute)


def ica_inputs(s):
    """sklearn's whitening (whiten='unit-variance', whiten_solver='svd') in float64, and w_init."""
    X = np.asarray(s.X, float)
    d = s.d
    XT = (X - X.mean(axis=0)).T
    u, sv = np.linalg.svd(XT, full_matrices=False)[:2]
    u *= np.sign(u[0])
    K = (u / sv).T[:d]
    X1 = (K @ XT) * np.sqrt(X.shape[0])
    w_init = S().rng("ica-w", s.id).standard_normal((d, d))
    return np.ascontiguousarray(X1.T), w_init


def _sym_decorrelation(M, tie_ok, Mf):
    """(M M^T)^-1/2 M for an arb_mat M (d x d): the eigenpairs of M M^T (simple; NotSettled otherwise)."""
    d = M.nrows()
    MM = M * M.transpose()
    spec = arb_spectrum(MM, d, tie_ok)
    _, G = float_guesses(Mf @ Mf.T)
    vecs = []
    for j in range(d):
        lam, mult = spec[j]
        if mult > 1:
            raise NotSettled("repeated eigenvalue in the decorrelation")
        vecs.append((lam, eigvec(MM, lam, None, G[:, j])))
    R = arb_mat(d, d)
    for i in range(d):
        for k in range(d):
            R[i, k] = sum((u[i] * u[k] / lam.sqrt() for lam, u in vecs), arb(0))
    return R * M


def ica_task(pid):
    s = S().points(pid)
    n = s.n
    Xw, w_init = ica_inputs(s)
    d = Xw.shape[1]
    XwQ = [[Q(v) for v in row] for row in Xw]

    def compute(tie_ok):
        out = Out(f"{pid}|ica", s.digest)
        out.ginputs["Xw"] = Xw
        out.ginputs["w_init"] = w_init
        W0 = _sym_decorrelation(arb_mat([[arb_q(Q(v)) for v in r] for r in w_init]), tie_ok, w_init)
        W0f = np.array([[mid(W0[i, j]) for j in range(d)] for i in range(d)])
        M1 = arb_mat(d, d)
        gw = []
        T = [[None] * n for _ in range(d)]
        for i in range(d):
            for t in range(n):
                wx = sum((W0[i, c] * arb_q(XwQ[t][c]) for c in range(d)), arb(0))
                T[i][t] = wx.tanh()
            gw.append(sum((1 - x * x for x in T[i]), arb(0)) / n)
        for i in range(d):
            for c in range(d):
                M1[i, c] = sum((T[i][t] * arb_q(XwQ[t][c]) for t in range(n)), arb(0)) / n - gw[i] * W0[i, c]
        M1f = np.array([[mid(M1[i, j]) for j in range(d)] for i in range(d)])
        W1 = _sym_decorrelation(M1, tie_ok, M1f)
        lim = None
        for i in range(d):
            v = abs(abs(sum((W1[i, c] * W0[i, c] for c in range(d)), arb(0))) - 1)
            lim = v if lim is None else lim.max(v)
            for c in range(d):
                out.at("ica_w0", 7, i * d + c, W0[i, c], 1.0)
                out.at("ica_w1", 7, i * d + c, W1[i, c], 1.0)
        out.group("ica_lim", lim, 1.0)
        del W0f
        return [out]

    return at_precisions(compute)


def nmf_inputs(s, seed):
    X = np.asarray(s.X, float)
    d = s.d
    g = S().rng("nmf-init", s.id, seed)
    avg = math.sqrt(X.mean() / d)
    H = avg * np.abs(g.standard_normal((d, s.p)))
    W = avg * np.abs(g.standard_normal((s.n, d)))
    return W, H


def nmf_task(pid, seed):
    s = S().points(pid)
    n, p, d = s.n, s.p, s.d
    W0, H0 = nmf_inputs(s, seed)
    X = rat_matrix(exact_rows(s.X))
    W = rat_matrix(exact_rows(W0))
    H = rat_matrix(exact_rows(H0))
    num, den = X * H.transpose(), W * (H * H.transpose())
    W1 = fmpq_mat([[W[i, j] * num[i, j] / den[i, j] for j in range(d)] for i in range(n)])
    num, den = W1.transpose() * X, (W1.transpose() * W1) * H
    H1 = fmpq_mat([[H[j, c] * num[j, c] / den[j, c] for c in range(p)] for j in range(d)])

    def loss(Wm, Hm):
        R = X - Wm * Hm
        return sum((R[i, c] ** 2 for i in range(n) for c in range(p)), fmpq(0)) / 2

    l0, l1 = loss(W, H), loss(W1, H1)

    def compute(tie_ok):
        out = Out(f"{pid}|nmf-{seed}", s.digest)
        out.ginputs["W0"] = W0
        out.ginputs["H0"] = H0
        for i in range(n):
            for j in range(d):
                out.at("nmf_w1", 8, i * d + j, W1[i, j])
        for j in range(d):
            for c in range(p):
                out.at("nmf_h1", 9, j * p + c, H1[j, c])
        out.group("nmf_loss0", l0)
        out.group("nmf_loss1", l1)
        out.group("nmf_err0", arb_q(2 * l0).sqrt())
        out.group("nmf_err1", arb_q(2 * l1).sqrt())
        return [out]

    return at_precisions(compute)


def linear_items():
    out = []
    for pid in linear_sets():
        out.append(("pca", pid, "cov"))
        s = S().points(pid)
        X = np.asarray(s.X)
        if not np.any(np.all(X == X[0], axis=0)):
            out.append(("pca", pid, "cor"))
    for pid in ("lowrank/r1p2", "lowrank/r3p10", "lattice/5x5", "dups/10x4", "twoblobs/80", "tiny/3"):
        for kernel in ("linear", "poly", "rbf"):
            out.append(("kpca", pid, kernel))
    for q in (1, 2, 3):
        out.append(("fa", "factor/p8k2", q, "start"))
    out.append(("fa", "factor/p8k2", 2, "other"))
    out.append(("fa", "lowrank/r3p10", 3, "start"))
    out.append(("ica", "ica/s4"))
    out.append(("nmf", "nonneg/r4", 0))
    out.append(("nmf", "nonneg/r4", 1))
    return out


def linear_task(item):
    kind, *args = item
    return {"pca": pca_task, "kpca": kpca_task, "fa": fa_task, "ica": ica_task, "nmf": nmf_task}[kind](*args)


# ---------------------------------------------------------------- mds


MDS_SETS = (
    "swiss/300",
    "scurve/300",
    "circle/60",
    "ring/150",
    "sphere/200",
    "twoblobs/80",
    "lattice/5x5",
    "lattice/5x5-shuffled",
    "dups/10x4",
    "collinear/p3",
    "lowrank/r3p10",
    "tiny/3",
    "fewdistinct/12",
    "allsame/8",
)


def mds_inputs(s, emb):
    """The given configuration Z: 0.9 times the classical embedding plus noise (sd 0.05 of its spread),
    rounded."""
    g = S().rng("mds-z", s.id)
    spread = float(np.std(emb)) if np.std(emb) > 0 else 1.0
    return 0.9 * emb + 0.05 * spread * g.standard_normal(emb.shape)


def pava(y, w=None):
    """The isotonic (non-decreasing) least-squares fit of y (Fractions) with weights w, by pooling adjacent
    violators: exact."""
    w = w or [1] * len(y)
    blocks = []  # [sum w y, sum w, count]
    for v, wt in zip(y, w):
        blocks.append([v * wt, Fraction(wt), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a = blocks.pop()
            blocks[-1][0] += a[0]
            blocks[-1][1] += a[1]
            blocks[-1][2] += a[2]
    out = []
    for sw, ww, c in blocks:
        out.extend([sw / ww] * c)
    return out


def isotonic(x, y, approach):
    """Kruskal's primary (ties of x ordered by y, then by position) or secondary (ties of x pooled) approach."""
    m = len(x)
    order = sorted(range(m), key=lambda k: (x[k], y[k], k))
    fit = [None] * m
    if approach == "primary":
        f = pava([y[k] for k in order])
        for t, k in enumerate(order):
            fit[k] = f[t]
        return fit
    blocks, vals = [], []
    for k in order:
        if blocks and x[blocks[-1][0]] == x[k]:
            blocks[-1].append(k)
        else:
            blocks.append([k])
    for b in blocks:
        vals.append(sum((y[k] for k in b), Fraction(0)) / len(b))
    f = pava(vals, [len(b) for b in blocks])
    for b, v in zip(blocks, f):
        for k in b:
            fit[k] = v
    return fit


def mds_task(pid):
    s = S().points(pid)
    n, p = s.n, s.p
    P = exact_rows(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_rows(P)
    r = [sum(row, fmpq(0)) / n for row in D2]
    t = sum(r, fmpq(0)) / n
    B = [[-(D2[i][j] - r[i] - r[j] + t) / 2 for j in range(n)] for i in range(n)]
    Xc, _ = centred(P)
    for i in range(n):
        for j in range(i, n):
            if B[i][j] != sum((Xc[i][c] * Xc[j][c] for c in range(p)), fmpq(0)):
                raise AssertionError(f"{pid}: B != Xc Xc^T at {i}, {j}")
    XcM = rat_matrix(Xc)
    Gm = XcM.transpose() * XcM
    f = Gm.charpoly()
    Gf = float_matrix(Gm)
    _, Gv = float_guesses(Gf)
    d = 2
    Bf = np.array([[float(fr(v)) for v in row] for row in B])
    w, V = np.linalg.eigh(Bf)
    w, V = w[::-1][:d], V[:, ::-1][:, :d]
    emb = V * np.sqrt(np.maximum(w, 0.0))
    Z = mds_inputs(s, emb)
    ZQ = exact_rows(Z)
    DZ2 = sqdist_rows(ZQ)
    small = n <= 80
    if small:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        y = np.array([math.sqrt(float(fr(DZ2[i][j]))) for i, j in pairs])
        yq = [fr(Q(v)) for v in y]
        xq = [fr(D2[i][j]) for i, j in pairs]
        iso_p = isotonic(xq, yq, "primary")
        iso_s = isotonic(xq, yq, "secondary")
    zero_delta = any(D2[i][j] == 0 for i in range(n) for j in range(i + 1, n))

    def compute(tie_ok):
        out = Out(pid, s.digest)
        out.ginputs["Z"] = Z
        spec_g = rational_spectrum(f)
        nonzero = [(v, m) for v, m in spec_g if not (v.is_exact() and v.is_zero())]
        spec = (nonzero + [(arb(0), n - len(nonzero))] * (n - len(nonzero)))[:n]
        if len(nonzero) < n:
            z = n - len(nonzero)
            spec = nonzero + [(arb(0), z)] * z
        gp = gaps(spec)
        lmax = abs(mid(spec[0][0])) or 1.0
        for j in range(n):
            out.at("eigenvalue", 0, j, spec[j][0], lmax, mult=spec[j][1], gap=gp[j])
        A = arb_mat(Gm)
        skipped = []
        for j in range(d):
            lam, mult = spec[j]
            if lam.is_exact() and lam.is_zero():
                for i in range(n):
                    out.at("embedding", 1, i * d + j, 0.0, 1.0, sign_gap=1.0)
                continue
            if mult > 1:
                skipped.append(j)
                continue
            v = eigvec(A, lam, None, Gv[:, j])
            col = [sum((arb_q(Xc[i][c]) * v[c] for c in range(p)), arb(0)) for i in range(n)]
            sg, gap = sign_rule(col, rank, tie_ok)
            scale = max(abs(mid(x)) for x in col)
            for i in range(n):
                out.at("embedding", 1, i * d + j, sg * col[i], scale, sign_gap=gap)
        if n <= 40:
            for i in range(n):
                for j in range(i, n):
                    out.at("b_matrix", 2, i * n + j, B[i][j])
        delta = [[arb_q(D2[i][j]).sqrt() for j in range(n)] for i in range(n)]
        dz = [[arb_q(DZ2[i][j]).sqrt() for j in range(n)] for i in range(n)]
        zscale = float(np.max(np.abs(Z))) or 1.0
        for i in range(n):
            acc = [arb(0)] * d
            for j in range(n):
                if j == i or DZ2[i][j] == 0:
                    continue
                ratio = delta[i][j] / dz[i][j]
                for c in range(d):
                    acc[c] += ratio * (arb_q(ZQ[i][c]) - arb_q(ZQ[j][c]))
            for c in range(d):
                out.at("guttman", 3, i * d + c, acc[c] / n, zscale)
        raw, sd2, sdel, sam, plus = arb(0), arb(0), arb(0), arb(0), 0.0
        for i in range(n):
            for j in range(i + 1, n):
                e = dz[i][j] - delta[i][j]
                raw += e * e
                sd2 += arb_q(DZ2[i][j])
                plus += mid(dz[i][j]) ** 2 + mid(delta[i][j]) ** 2
                if not zero_delta:
                    sdel += delta[i][j]
                    sam += e * e / delta[i][j]
        out.group("stress_raw", raw, plus, plus=plus)
        if sd2 > 0:
            out.group("stress_kruskal", (raw / sd2).sqrt(), 1.0)
        if not zero_delta:
            out.group("stress_sammon", sam / sdel, 1.0)
        if small:
            out.ginputs["y"] = y
            for k in range(len(pairs)):
                out.at("iso_primary", 4, k, iso_p[k])
                out.at("iso_secondary", 4, k, iso_s[k])
        out.ginputs["note"] = f"no embedding column {skipped} (repeated eigenvalue)" if skipped else ""
        return [out]

    return at_precisions(compute)


# ---------------------------------------------------------------- graph


GRAPH_SETS = (
    "swiss/300",
    "scurve/300",
    "circle/60",
    "ring/150",
    "sphere/200",
    "twoblobs/80",
    "lattice/5x5",
    "lattice/5x5-shuffled",
    "dups/10x4",
    "collinear/p3",
    "blobs/k3p10",
    "blobs/k5p20",
    "sizes/k3p10",
)


def components(n, adj):
    seen = [-1] * n
    c = 0
    for s0 in range(n):
        if seen[s0] >= 0:
            continue
        stack = [s0]
        seen[s0] = c
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if seen[v] < 0:
                    seen[v] = c
                    stack.append(v)
        c += 1
    return c


def dijkstra(n, adj, w, src):
    """Shortest path lengths from src (arbs): Dijkstra ordered by the labels' exact midpoints, each label the
    ball of an actual path's length, then widened by (n - 1) eps, eps the largest violation of an edge inequality
    L(v) <= L(u) + w(u, v) the balls allow: the true distance d(v) lies in [L(v) - (n - 1) eps, L(v)]."""
    L = [None] * n
    L[src] = arb(0)
    done = [False] * n
    heap = [(arb(0), src)]
    while heap:
        _, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v in adj[u]:
            if done[v]:
                continue
            cand = L[u] + w[(u, v)]
            if L[v] is None or cand.mid() < L[v].mid():
                L[v] = cand
                heapq.heappush(heap, (cand.mid(), v))
    eps = arb(0)
    for u in range(n):
        for v in adj[u]:
            viol = L[v] - L[u] - w[(u, v)]
            up = arb(viol.upper())
            if up > eps:
                eps = up
    return [arbref.add_error(x, (n - 1) * eps) if x is not None else None for x in L]


def lap_prepare(n, W):
    """The pencil (D - W, D) of the normalised Laplacian I - D^-1/2 W D^-1/2 of the rational symmetric weights W
    (dict (i, j) -> fmpq): (D - W as fmpq_mat, the degrees, the number of components, the exact characteristic
    polynomial of D^-1 (D - W), numpy's eigenvectors of L for the pivots)."""
    deg = [fmpq(0)] * n
    adj = [[] for _ in range(n)]
    for (i, j), v in W.items():
        deg[i] += v
        adj[i].append(j)
    c = components(n, adj)
    rows = [[fmpq(0)] * n for _ in range(n)]
    for i in range(n):
        rows[i][i] = deg[i]
    for (i, j), v in W.items():
        rows[i][j] = -v
    Lw = rat_matrix(rows)
    f = rat_matrix([[rows[i][j] / deg[i] for j in range(n)] for i in range(n)]).charpoly()
    Lf = float_matrix(Lw)
    sd = 1 / np.sqrt(np.array([float(fr(v)) for v in deg]))
    _, V = np.linalg.eigh(Lf * sd[:, None] * sd[None, :])
    return Lw, deg, c, f, V * sd[:, None]


def laplacian_part(out, name, prep, rank, tie_ok, d=2):
    """The LAP_EIGS smallest eigenvalues of the normalised Laplacian (exact multiplicities) and sklearn's
    embedding columns u_j / sqrt(deg), j = 1 .. d (u_j the unit eigenvectors)."""
    Lw, deg, c, f, V = prep
    n = Lw.nrows()
    spec = rational_spectrum(f)[::-1]
    gp = gaps(spec)
    want = min(n, LAP_EIGS)
    for j in range(want):
        out.at(f"{name}_eigenvalue", 4, j, spec[j][0], 2.0, mult=spec[j][1], gap=gp[j])
    A = arb_mat(Lw)
    dga = [arb_q(v) for v in deg]
    skipped = []
    for j in range(1, d + 1):
        lam, mult = spec[j]
        if mult > 1:
            skipped.append(j)
            continue
        u = eigvec(A, lam, dga, V[:, j])  # unit eigenvectors of L
        col = [u[i] / dga[i].sqrt() for i in range(n)]
        sg, gap = sign_rule(col, rank, tie_ok)
        scale = max(abs(mid(x)) for x in col)
        for i in range(n):
            out.at(f"{name}_embedding", 5, i * d + j - 1, sg * col[i], scale, sign_gap=gap)
    return skipped, c


def graph_task(item):
    pid, k = item
    s = S().points(pid)
    n = s.n
    P = exact_rows(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_rows(P)
    nb, nxt = knn_rows(D2, rank, k)
    reg = Q(REG)
    weights = []
    for i in range(n):
        C = [[P[j][c] - P[i][c] for c in range(s.p)] for _, j in nb[i]]
        G = [[sum((a * b for a, b in zip(C[r1], C[r2])), fmpq(0)) for r2 in range(k)] for r1 in range(k)]
        tr = sum((G[r][r] for r in range(k)), fmpq(0))
        R = reg * tr if tr > 0 else reg
        for r in range(k):
            G[r][r] += R
        wv = rat_matrix(G).solve(fmpq_mat([[1] for _ in range(k)]))
        tot = sum((wv[r, 0] for r in range(k)), fmpq(0))
        weights.append([wv[r, 0] / tot for r in range(k)])
    A = {}
    for i in range(n):
        for _, j in nb[i]:
            A[(i, j)] = 1
    Wm, Wu = {}, {}
    for i, j in A:
        both = (j, i) in A
        for a, b in ((i, j), (j, i)):
            Wm[(a, b)] = fmpq(1) if both else fmpq(1, 2)
            Wu[(a, b)] = fmpq(1)
    adj = [[] for _ in range(n)]
    for i, j in Wu:
        adj[i].append(j)
    connected = components(n, adj) == 1
    preps = {"lapm": lap_prepare(n, Wm), "lapu": lap_prepare(n, Wu)}
    do_iso = k == 5 and connected
    if do_iso:
        sources = list(range(n)) if n <= 60 else list(range(8))

    def compute(tie_ok):
        out = Out(f"{pid}|k{k}", s.digest)
        dmax = max(math.sqrt(float(fr(d))) for row in nb for d, _ in row) or 1.0
        for i in range(n):
            row = nb[i]
            bnd = 1 if nxt[i] is not None and nxt[i] == row[-1][0] else 0
            for r, (d2, j) in enumerate(row):
                before = r > 0 and row[r - 1][0] == d2
                after = (row[r + 1][0] == d2) if r + 1 < k else (nxt[i] is not None and nxt[i] == d2)
                out.at("knn", 0, i * k + r, d2, nbr=j, tie=int(before or after), boundary=bnd)
                out.at("knn_dist", 0, i * k + r, arb_q(d2).sqrt() if d2 != 0 else fmpq(0), dmax)
                out.at("lle_weight", 0, i * k + r, weights[i][r])
        notes = []
        for name, prep in preps.items():
            sk, c = laplacian_part(out, name, prep, rank, tie_ok)
            if sk:
                notes.append(f"{name}: no embedding column {sk} ({c} components or a repeated eigenvalue)")
        if do_iso:
            w = {(i, j): (arb_q(D2[i][j]).sqrt() if D2[i][j] != 0 else arb(0)) for (i, j) in Wu}
            G = [dijkstra(n, adj, w, src) for src in range(n)]
            for src in sources:
                for t in range(n):
                    if t == src or (n <= 60 and t < src):
                        continue
                    out.at("geodesic", 1, src * n + t, G[src][t], max(mid(G[src][t]), 1e-300))
            G2 = [[G[i][j] * G[i][j] for j in range(n)] for i in range(n)]
            r = [sum(row, arb(0)) / n for row in G2]
            tt = sum(r, arb(0)) / n
            K = arb_mat([[-(G2[i][j] - r[i] - r[j] + tt) / 2 for j in range(n)] for i in range(n)])
            want = min(n, ISO_EIGS)
            try:
                spec = arb_spectrum(K, want + 1, tie_ok)
            except Unresolvable as e:
                spec = None
                notes.append(f"isomap: no eigenpairs ({e})")
        if do_iso and spec is not None:
            gp = gaps(spec)
            lmax = abs(mid(spec[0][0]))
            for j in range(want):
                out.at("isomap_eigenvalue", 2, j, spec[j][0], lmax, mult=spec[j][1], gap=gp[j])
            Kf = np.array([[mid(K[i, j]) for j in range(n)] for i in range(n)])
            _, V = float_guesses(Kf)
            for j in range(2):
                lam, mult = spec[j]
                if mult > 1 or not lam > 0:
                    notes.append(f"isomap: no embedding column {j} (repeated eigenvalue)")
                    continue
                u = eigvec(K, lam, None, V[:, j])
                col = [x * lam.sqrt() for x in u]
                sg, gap = sign_rule(col, rank, tie_ok)
                scale = max(abs(mid(x)) for x in col)
                for i in range(n):
                    out.at("isomap_embedding", 3, i * 2 + j, sg * col[i], scale, sign_gap=gap)
        elif k == 5 and not do_iso:
            notes.append("isomap: the graph is not connected")
        out.ginputs["k"] = np.float64(k)
        out.ginputs["note"] = "; ".join(notes)
        return [out]

    return at_precisions(compute)


def graph_items():
    return [(pid, k) for pid in GRAPH_SETS for k in (5, 15) if S().points(pid).n > k + 1]


# ---------------------------------------------------------------- sne


def _entropy_parts(d, beta):
    """(H, Var) of the distribution p_j proportional to exp(-beta d_j) (d arbs, shifted by their minimum)."""
    dmin = min(d, key=mid)
    e = [(-(beta * (x - dmin))).exp() for x in d]
    Z = sum(e, arb(0))
    pr = [x / Z for x in e]
    m1 = sum((pj * (x - dmin) for pj, x in zip(pr, d)), arb(0))
    m2 = sum((pj * (x - dmin) ** 2 for pj, x in zip(pr, d)), arb(0))
    return Z.log() + beta * m1, m2 - m1 * m1, pr


def perplexity_beta(d2, perp):
    """The root beta of H(beta) = log(perp) (H in nats), settled to 2^-110 relative, for the squared
    distances d2 (fmpq)."""
    darb = None

    def g(beta):
        nonlocal darb
        darb = [arb_q(x) for x in d2]
        H, _, _ = _entropy_parts(darb, beta)
        return arb(perp).log() - H

    def xdg(beta):
        darb2 = [arb_q(x) for x in d2]
        _, var, _ = _entropy_parts(darb2, beta)
        return beta * beta * var

    df = np.array([float(fr(x)) for x in d2])
    df = df - df.min()

    def hf(beta):
        e = np.exp(-beta * df)
        z = e.sum()
        return math.log(z) + beta * float((e * df).sum()) / z - math.log(perp)

    return arbref.solve_log_settled(g, xdg, log_bisect(hf, decreasing=True), increasing=True)


def log_bisect(h, decreasing=False, lo=-700.0, hi=700.0, steps=200):
    """A float start for Newton: the root of h(exp(t)) by bisection in t (h increasing, or decreasing)."""
    for _ in range(steps):
        t = 0.5 * (lo + hi)
        v = h(math.exp(t))
        if (v > 0) != decreasing:
            hi = t
        else:
            lo = t
    return math.exp(0.5 * (lo + hi))


def sne_rows(n):
    return list(range(n)) if n <= 80 else list(range(ROWS))


def grad_inputs(s, d, which):
    g = S().rng("tsne-y", s.id, d, which)
    if which == "ya":  # sklearn's init='random' scale
        return 1e-4 * g.standard_normal((s.n, d))
    X = np.asarray(s.X, float)
    Xc = X - X.mean(axis=0)
    cols = Xc[:, :d] if X.shape[1] >= d else np.column_stack([Xc, np.zeros((s.n, d - X.shape[1]))])
    sd = cols.std(axis=0)
    sd[sd == 0] = 1.0
    return cols / sd + 0.01 * g.standard_normal((s.n, d))


def tsne_task(item):
    pid, perp, mode, grads = item
    s = S().points(pid)
    n = s.n
    P = exact_rows(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_rows(P)
    k = min(n - 1, int(3.0 * perp + 1)) if mode == "knn" else n - 1
    nb, _ = knn_rows(D2, rank, k)
    rows = sne_rows(n)
    betas = [perplexity_beta([d for d, _ in nb[i]], perp) for i in range(n)]
    Ys = [(which, d, grad_inputs(s, d, which)) for which, d in grads]

    def compute(tie_ok):
        outs = []
        out = Out(f"{pid}|perp{perp:g}-{mode}", s.digest)
        out.ginputs["perplexity"] = np.float64(perp)
        cond = []
        for i in range(n):
            darb = [arb_q(d) for d, _ in nb[i]]
            _, _, pr = _entropy_parts(darb, betas[i])
            cond.append({j: pr[r] for r, (_, j) in enumerate(nb[i])})
            out.at("beta", 0, i, betas[i], k=k)
        Psym = {}
        for i in range(n):
            for j, v in cond[i].items():
                a, b = (i, j)
                Psym[(a, b)] = Psym.get((a, b), arb(0)) + v / (2 * n)
                Psym[(b, a)] = Psym.get((b, a), arb(0)) + v / (2 * n)
        for i in rows:
            for j, v in cond[i].items():
                out.at("p_cond", 1, i * n + j, v, 1.0)
            for j in range(n):
                if (i, j) in Psym:
                    out.at("p_sym", 2, i * n + j, Psym[(i, j)], 1.0 / n)
        outs.append(out)
        for which, d, Y in Ys:
            g = Out(f"{pid}|grad-perp{perp:g}-{mode}-{which}-d{d}", s.digest)
            g.ginputs["Y"] = Y
            g.ginputs["perplexity"] = np.float64(perp)
            dof = max(d - 1, 1)
            YQ = exact_rows(Y)
            Ya = [[arb_q(v) for v in row] for row in YQ]
            Wt, Ft = {}, {}
            tot = arb(0)
            for i in range(n):
                for j in range(i + 1, n):
                    dd2 = arb_q(sqdist(YQ[i], YQ[j]))
                    base = 1 + dd2 / dof
                    wv = base ** (-arb(dof + 1) / 2) if dof != 1 else 1 / base
                    Wt[(i, j)] = Wt[(j, i)] = wv
                    Ft[(i, j)] = Ft[(j, i)] = 1 / base
                    tot += 2 * wv
            c = arb(2 * (dof + 1)) / dof
            kl = arb(0)
            for (i, j), pv in Psym.items():
                q = Wt[(i, j)] / tot
                kl += pv * (pv / q).log()
            for i in range(n):
                for cc in range(d):
                    acc, accs, plus = arb(0), arb(0), 0.0
                    for j in range(n):
                        if j == i:
                            continue
                        q = Wt[(i, j)] / tot
                        pq = Psym.get((i, j), arb(0)) - q
                        diff = Ya[i][cc] - Ya[j][cc]
                        acc += pq * Ft[(i, j)] * diff
                        accs += pq * Wt[(i, j)] * diff
                        plus += abs(mid(pq) * mid(Ft[(i, j)]) * mid(diff))
                    sc = max(mid(c) * plus, 1e-300)
                    g.at("grad", 3, i * d + cc, c * acc, sc, plus=mid(c) * plus)
                    if dof != 1:
                        g.at("grad_sklearn", 3, i * d + cc, c * accs, sc)
            g.group("kl", kl, 1.0)
            outs.append(g)
        return outs

    return at_precisions(compute)


def umap_task(item):
    pid, kk = item
    s = S().points(pid)
    n = s.n
    P = exact_rows(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_rows(P)
    nb, nxt = knn_rows(D2, rank, kk - 1)  # plus the point itself: kk in all

    def compute(tie_ok):
        out = Out(f"{pid}|umap{kk}", s.digest)
        out.ginputs["n_neighbors"] = np.float64(kk)
        dist = [[arb_q(d).sqrt() if d != 0 else arb(0) for d, _ in nb[i]] for i in range(n)]
        allmean = sum((x for row in dist for x in row), arb(0)) / (n * kk)
        floor_c = arb_q(Q(MIN_K_DIST_SCALE))
        target = arb(kk).log() / arb(2).log()
        rhos, sigmas = [], []
        for i in range(n):
            nz = [(d2, x) for (d2, _), x in zip(nb[i], dist[i]) if d2 != 0]
            rho2, rho = nz[0] if nz else (fmpq(0), arb(0))
            # the terms beyond rho, decided exactly (d_j > rho iff d_j^2 > rho^2); the others are exactly 1
            pos = [x - rho for (d2, _), x in zip(nb[i], dist[i]) if d2 > rho2]
            ones = len(nb[i]) - len(pos)
            if 2**ones >= kk or not pos:  # no root: every sigma gives at least log2(k)
                root = arb(0)
            else:

                def g(sig, pos=pos, ones=ones):
                    return sum(((-(x / sig)).exp() for x in pos), arb(0)) + ones - target

                def xdg(sig, pos=pos):
                    return sum(((-(x / sig)).exp() * x / sig for x in pos), arb(0))

                pf = np.array([mid(x) for x in pos])

                def gf(sig, pf=pf, ones=ones):
                    return float(np.exp(-pf / sig).sum()) + ones - math.log2(kk)

                root = arbref.solve_log_settled(g, xdg, log_bisect(gf), increasing=True)
            mean_i = (sum(dist[i], arb(0))) / kk  # the row with the point itself (0) included, as umap-learn's
            lim = floor_c * (mean_i if rho > 0 else allmean)
            if root < lim:
                sig, rule = lim, 1
            elif root > lim:
                sig, rule = root, 0
            else:
                raise NotSettled("sigma at its floor")
            rhos.append((rho2, rho))
            sigmas.append(sig)
            out.at("rho", 0, i, rho if not rho.is_zero() else fmpq(0), 1.0 if rho.is_zero() else 0)
            out.at("sigma", 0, i, sig, 0, rule=rule)
        if n <= 150:
            memb = {}
            for i in range(n):
                rho2, rho = rhos[i]
                for (d2, j), x in zip(nb[i], dist[i]):
                    if d2 <= rho2 or sigmas[i].is_zero():
                        memb[(i, j)] = arb(1)
                    else:
                        memb[(i, j)] = (-((x - rho) / sigmas[i])).exp()
            edges = sorted({(min(i, j), max(i, j)) for (i, j) in memb})
            for i, j in edges:
                a, b = memb.get((i, j), arb(0)), memb.get((j, i), arb(0))
                out.at("fuzzy", 4, i * n + j, a + b - a * b, 1.0)
        bnd = sum(1 for i in range(n) if nxt[i] is not None and nxt[i] == nb[i][-1][0])
        out.ginputs["boundary_ties"] = np.float64(bnd)
        return [out]

    return at_precisions(compute)


SNE_ITEMS = (
    ("blobs/k3p10", 30.0, "knn", (("ya", 2), ("yb", 2), ("yb", 3))),
    ("blobs/k3p10", 30.0, "exact", (("ya", 2), ("yb", 2), ("yb", 3))),
    ("blobs/k3p10", 5.0, "knn", ()),
    ("sizes/k3p10", 30.0, "knn", (("yb", 2),)),
    ("swiss/300", 30.0, "knn", ()),
    ("circle/60", 30.0, "exact", (("yb", 2),)),
    ("circle/60", 5.0, "knn", ()),
    ("lattice/5x5", 5.0, "knn", (("yb", 2),)),
    ("lattice/5x5", 5.0, "exact", ()),
    ("twoblobs/80", 30.0, "exact", (("yb", 2),)),
    ("dups/10x4", 5.0, "knn", (("yb", 2),)),
)
UMAP_SETS = ("blobs/k3p10", "sizes/k3p10", "swiss/300", "circle/60", "lattice/5x5", "dups/10x4", "twoblobs/80")


def sne_items():
    return [("tsne", *it) for it in SNE_ITEMS] + [("umap", (pid, UMAP_K)) for pid in UMAP_SETS]


def sne_task(item):
    if item[0] == "umap":
        return umap_task(item[1])
    return tsne_task(item[1:])


# ---------------------------------------------------------------- quality


def quality_embedding(s, name):
    X = np.asarray(s.X, float)
    if name == "first2":
        return X[:, :2].copy()
    if name == "first1":
        return X[:, :1].copy()
    if name == "own":
        t = np.asarray(s.t, float)
        return (t if t.ndim == 2 else t[:, None]).copy()
    if name == "xz":
        return X[:, [0, 2]].copy()
    if name == "tilt":
        return (X[:, 0] + 0.1 * X[:, 1])[:, None]
    if name == "pca2r":  # a rounded PCA: the top two principal axes in float64, rounded to 3 decimals
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        V = Vt[:2].T * np.sign(Vt[:2].T[np.argmax(np.abs(Vt[:2].T), axis=0), [0, 1]])
        return np.round(Xc @ V, 3)
    raise KeyError(name)


QUALITY_ITEMS = (
    ("swiss/300", "own"),
    ("swiss/300", "xz"),
    ("scurve/300", "own"),
    ("blobs/k3p10", "first2"),
    ("sizes/k3p10", "pca2r"),
    ("lattice/5x5", "first2"),
    ("lattice/5x5", "tilt"),
    ("dups/10x4", "first2"),
    ("circle/60", "own"),
)


def quality_items():
    return [(pid, e, k) for pid, e in QUALITY_ITEMS for k in (5, 10) if k < S().points(pid).n / 2]


def _ranks(D2, rank):
    """Per point i: {j: rank r(i, j) in 1 .. n - 1} by (exact squared distance, key), and the tie flags."""
    n = len(D2)
    out = []
    for i in range(n):
        row = sorted(((D2[i][j], rank[j], j) for j in range(n) if j != i), key=lambda t: (t[0], t[1]))
        r = {j: t + 1 for t, (_, _, j) in enumerate(row)}
        tied = set()
        for t in range(1, len(row)):
            if row[t][0] == row[t - 1][0]:
                tied.add(row[t][2])
                tied.add(row[t - 1][2])
        out.append((r, tied))
    return out


def quality_task(item):
    pid, ename, k = item
    s = S().points(pid)
    n = s.n
    E = quality_embedding(s, ename)
    rank = S().lex_rank(s.X).tolist()
    DX = sqdist_rows(exact_rows(s.X))
    DE = sqdist_rows(exact_rows(E))
    RX, RE = _ranks(DX, rank), _ranks(DE, rank)
    denom = n * k * (2 * n - 3 * k - 1)

    def side(Ra, Rb):  # the penalty of b's neighbours by their ranks in a
        tot, per, ties = 0, [], 0
        for i in range(n):
            ra, tieda = Ra[i]
            rb, tiedb = Rb[i]
            nb = [j for j, r in rb.items() if r <= k]
            v = sum(ra[j] - k for j in nb if ra[j] > k)
            per.append(v)
            tot += v
            if any(j in tiedb for j in nb) or any(j in tieda for j in nb if ra[j] > k):
                ties += 1
        return tot, per, ties

    st, pt, tt = side(RX, RE)
    sc, pc, tc = side(RE, RX)

    def compute(tie_ok):
        out = Out(f"{pid}|{ename}-k{k}", s.digest)
        out.ginputs["E"] = E
        out.ginputs["n_neighbors"] = np.float64(k)
        for i in range(n):
            out.at("t_point", 0, i, pt[i])
            out.at("c_point", 0, i, pc[i])
        out.group("trustworthiness", 1 - Fraction(2 * st, denom), sum=st, ties=tt)
        out.group("continuity", 1 - Fraction(2 * sc, denom), sum=sc, ties=tc)
        return [out]

    return at_precisions(compute)


# ---------------------------------------------------------------- driver


def _items(fam):
    return {
        "linear": linear_items,
        "mds": lambda: list(MDS_SETS),
        "graph": graph_items,
        "sne": sne_items,
        "quality": quality_items,
    }[fam]()


TASKS = {
    "linear": linear_task,
    "mds": mds_task,
    "graph": graph_task,
    "sne": sne_task,
    "quality": quality_task,
}
FAMILIES = tuple(TASKS)
EXTRA = {
    "linear": "exact rationals (fmpq, fmpq_mat, the characteristic polynomials' roots); tanh, log, square roots "
    "and the kernels' exp in Arb",
    "mds": "exact rationals (fmpq, fmpq_mat, Fraction); square roots in Arb",
    "graph": "exact rationals (fmpq, fmpq_mat); square roots, Dijkstra and acb_mat.eig in Arb",
    "sne": "exact rationals (fmpq); exp, log and the roots in Arb",
    "quality": "exact rationals (Fraction)",
}


def _run(task):
    fam, k, item = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        res, err = TASKS[fam](item), None
    except (_Timeout, NotSettled) as e:
        res, err = None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    return fam, k, res, time.monotonic() - t0, err, rss


def assemble(name, results):
    """Write one file from the tasks' results (in item order, each a list of groups)."""
    from tests import reffile

    inputs = {"set": [], "kind": [], "index": [], "digest": []}
    group, gin, methods = [], {}, {}
    flat = [g for r in results for g in r]

    def put(m, over, at, entry):
        e = methods.setdefault(m, {"over": over, "index": [], "hi": [], "lo": [], "dropped": []})
        if entry is None:
            e["dropped"].append(at)
            return
        hi, lo, fields = entry
        e["index"].append(at)
        e["hi"].append(hi)
        e["lo"].append(lo)
        for key, v in fields.items():
            e.setdefault(key, []).append(v)

    gkeys = sorted({key for r in flat for key in r["ginputs"]})
    for g, r in enumerate(flat):
        for key in gkeys:
            gin.setdefault(key, []).append(r["ginputs"].get(key))
        for kind, idx, ms in r["points"]:
            at = len(group)
            group.append(g)
            for key, v in (("set", r["set"]), ("kind", kind), ("index", idx), ("digest", r["digest"])):
                inputs[key].append(v)
            for m, entry in ms.items():
                put(m, "point", at, entry)
        for m, entry in r["gmethods"].items():
            put(m, "group", g, entry)
    for m, e in methods.items():
        width = len(e["index"])
        for key in list(e):
            if key in ("over", "dropped"):
                continue
            if len(e[key]) != width:
                raise ValueError(f"{name}: method {m} field {key} has {len(e[key])} of {width} values")
        for key in e:
            if key != "over":
                e[key] = np.asarray(e[key], dtype=np.int64 if key in ("index", "dropped") else np.float64)
    inputs = {
        "set": np.array(inputs["set"], dtype=str),
        "kind": np.array(inputs["kind"], np.int64),
        "index": np.array(inputs["index"], np.int64),
        "digest": np.array(inputs["digest"], dtype=str),
    }
    ginputs = {"set": np.array([r["set"] for r in flat], dtype=str)}
    for key, vs in gin.items():
        if all(isinstance(v, str) or v is None for v in vs):
            ginputs[key] = np.array(["" if v is None else v for v in vs], dtype=str)
            continue
        arrs = [np.full((0,), np.nan) if v is None else np.asarray(v, float) for v in vs]
        if len({a.shape for a in arrs}) == 1:
            ginputs[key] = np.stack(arrs)
        else:
            ginputs[key] = arrs
    reffile.write(
        OUT / f"{name}.npz",
        **arbref.provenance(extra=EXTRA.get(name)),
        tool=TOOL,
        inputs=inputs,
        methods=methods,
        group=np.array(group, np.int64),
        ngroups=len(flat),
        group_inputs=ginputs,
        n_requested=len(group),
    )
    size = (OUT / f"{name}.npz").stat().st_size / 1024
    print(
        f"{name}: {len(flat)} groups, {len(group)} points, {len(methods)} methods, {size:.0f} KiB "
        f"-> {(OUT / f'{name}.npz').relative_to(ROOT)}",
        flush=True,
    )


def _cost(task):
    fam, _, item = task
    pid = {"linear": lambda: item[1], "sne": lambda: item[1] if item[0] == "tsne" else item[1][0]}.get(
        fam, lambda: item[0] if isinstance(item, tuple) else item
    )()
    s = S().points(pid)
    return s.n**2 * (s.p + 10)


def build(names, jobs):
    start = time.monotonic()
    plans = {f: _items(f) for f in names}
    tasks = [(f, k, it) for f in names for k, it in enumerate(plans[f])]
    tasks.sort(key=lambda t: -_cost(t))  # the long ones first
    results = {f: {} for f in names}
    failed = []
    peak = 0.0
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=4) as pool:  # fork deadlocks after BLAS threads
        for fam, k, res, secs, err, rss in pool.imap_unordered(_run, tasks, chunksize=1):
            peak = max(peak, rss)
            if res is None:
                failed.append((fam, plans[fam][k], err))
                print(f"  failed {fam} {plans[fam][k]}: {err}", flush=True)
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]}: {secs:.0f} s, worker RSS {rss:.2f} GB", flush=True)
    for f in names:
        if any(fam == f for fam, _, _ in failed):
            print(f"{f}: not written (a task failed)", flush=True)
            continue
        assemble(f, [results[f][k] for k in sorted(results[f])])
    print(
        f"{len(tasks)} tasks, {len(failed)} failed, {(time.monotonic() - start):.0f} s, "
        f"largest worker RSS {peak:.2f} GB",
        flush=True,
    )
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their items")
    from bench import cli

    cli.add_jobs(ap, default=6)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    names = [f for f in FAMILIES if not a.names or f in a.names]
    if a.list:
        for f in names:
            items = _items(f)
            print(f"{f} ({len(items)}): " + ", ".join(map(str, items)))
        return 0
    with arbref.logged("embed_ref"):
        return build(names, min(a.jobs, 8))


if __name__ == "__main__":
    sys.exit(main())
