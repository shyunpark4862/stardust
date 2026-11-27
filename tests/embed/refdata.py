"""sd.embed's references and their contract (docs/design/embed.md §4, §7), as the tests and the accuracy measurement
(bench/accuracy) both judge them.

tests/embed/ref/<family>.npz (tools/embed_ref.py; layout: tests/reffile.py) holds points with the inputs "set" (the
id in tests/embed/samples.py, with "|..." for the variant), "kind", "index" and "digest" (the data's digest); one
group per variant, whose inputs are what the references were computed at; and per method the reference hi + lo
(tools/embed_ref.py's docstring defines each one):

    linear   "|cov", "|cor": variance, singular, ratio (kind 0, index j; fields mult, gap), component (kind 1, index
             j p + c; fields sign_gap, gap), total_variance and noise_variance (field d) over the groups;
             "|kpca-<kernel>": kpca_eigenvalue (kind 2), kpca_embedding (kind 3, index i 2 + j), kpca_kernel (kind 4,
             index i n + j, i <= j); "|fa-q<q>-<psi>" (group inputs psi, q): fa_eigen (kind 5), fa_gradient (kind
             6), fa_objective; "|ica" (group inputs Xw, w_init): ica_w0, ica_w1 (kind 7, index i d + j), ica_lim;
             "|nmf-<s>" (group inputs W0, H0): nmf_w1 (kind 8, index i d + j), nmf_h1 (kind 9, index j p + c),
             nmf_loss0, nmf_loss1, nmf_err0, nmf_err1
    mds      eigenvalue (kind 0, all n), embedding (kind 1, index i 2 + j), b_matrix (kind 2, index i n + j), at the
             group input Z: guttman (kind 3, index i 2 + c), stress_raw (field plus), stress_kruskal,
             stress_sammon; for n <= 80 at the group input y: iso_primary, iso_secondary (kind 4, the condensed pair)
    graph    "|k<k>": knn (kind 0, index i k + r; fields nbr, tie, boundary), knn_dist, lle_weight (kind 0),
             lapm_* and lapu_* (eigenvalue kind 4, embedding kind 5, index i 2 + j), for k = 5: geodesic (kind 1,
             index s n + t), isomap_eigenvalue (kind 2), isomap_embedding (kind 3)
    sne      "|perp<P>-<knn|exact>": beta (kind 0; field k), p_cond (kind 1, index i n + j), p_sym (kind 2);
             "|grad-...": grad (kind 3, index i d + c; field plus), grad_sklearn (d = 3 only), kl; "|umap15": rho,
             sigma (kind 0; field rule), fuzzy (kind 4, index i n + j, i < j)
    quality  "|<E>-k<k>" (group inputs E, n_neighbors): trustworthiness and continuity over the groups (fields sum,
             ties), t_point and c_point (kind 0)

A group's input "note" says what the tool left out and why (a repeated eigenvalue has no eigenvector; a graph that
is not connected has no Isomap).

The contracts (docs/design/embed.md §4). DRAFTS: the constants below are placeholders of milestone EM0, stated so
that the tests and the accuracy sweep have one place to read them; the milestones that implement each function
(EM1-EM8) prove or measure and fix them:

    singular values, 'full'     |s - s*| <= C_SVD u sqrt(min(n, p)) s_1                                  (§4.1)
    variances, covariance_eigh  |l - l*| <= C_COV u n l_1 (the squared loss: small variances are not
                                relatively accurate)                                                    (§4.1)
    components                  sin angle(v, v*) <= C_VEC u n l_1 / gap (Davis-Kahan; gap the reference's
                                field: the distance to the nearest other eigenvalue)                    (§4.1)
    kernel PCA, classical MDS,  eigenvalues |l - l*| <= C_DENSE_EIG u n l_1; the embedding's columns by the
    Isomap                      Davis-Kahan bound with C_VEC                                            (§4.1, §4.2)
    factor analysis             at given psi: |f - f*| <= C_FA gamma(p^2) (1 + |f*|); the gradient's entries
                                |g - g*| <= C_FA gamma(p^2) / psi^2                                     (§4.1)
    FastICA, NMF one step       the entries |w - w*| <= C_STEP gamma(n + d^2) max |w*| (FastICA: with the
                                whitened data's size; NMF: relative per entry)                          (§4.1)
    Guttman transform           |z - z*| <= C_STEP gamma(n + 2p + 8) max |z*|                              (§4.2)
    stress                      raw: |s - s*| <= gamma(ceil(log2 m) + p + 8) plus (m the pairs); kruskal and
                                sammon: relative gamma(ceil(log2 m) + p + 10)                          (§4.4)
    the k-NN sets               equal to the reference's where no boundary tie (field boundary); at a tie either
                                choice                                                                  (§4.3)
    LLE weights                 |w - w*| <= C_LLE u k (1 + 1 / reg) max |w*| (the regularised Gram's condition)
    Laplacian eigenvalues       |m - m*| <= C_DENSE_EIG u n 2 (|L| <= 2)                                  (§4.3)
    geodesic distances          relative gamma(n + p / 2 + 2) (a path's sum of at most n - 1 distances)
    beta, sigma (the roots)     the defining equation's residual at the computed root below the bisection's
                                tolerance: |H(beta) - log P| <= PERPLEXITY_TOL, |psum(sigma) - log2 k| <=
                                SMOOTH_K_TOL (sklearn's and umap-learn's own)                           (§4.3)
    P, fuzzy weights            relative C_P gamma(k + 10) beyond what the roots' tolerance moves them  (§4.3)
    the t-SNE gradient          |g - g*| <= C_GRAD gamma(n + d + 10) plus (field plus: the sum of the terms'
                                absolute values); KL relative gamma(n + 10)                             (§4.3)
    trustworthiness, continuity correctly rounded (the reference's hi)                                  (§4.4)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("graph", "linear", "mds", "quality", "sne")
U = 2.0**-53

# DRAFTS (milestone EM0): the milestones EM1-EM8 fix them
C_SVD = 4.0
C_COV = 4.0
C_VEC = 4.0
C_DENSE_EIG = 4.0
C_FA = 4.0
C_STEP = 4.0
C_LLE = 4.0
C_P = 4.0
C_GRAD = 4.0
PERPLEXITY_TOL = 1e-5  # sklearn's _binary_search_perplexity
SMOOTH_K_TOL = 1e-5  # umap-learn's SMOOTH_K_TOLERANCE
REG = 1e-3  # the LLE references' reg
MIN_K_DIST_SCALE = 1e-3  # umap-learn's floor of sigma


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k) -> float:
    return k * U / (1 - k * U)


def base(sid: str) -> str:
    """The samples.py id of a set label ("swiss/300|k5" -> "swiss/300")."""
    return sid.split("|")[0]


def at(r: reffile.Ref, method: str, sid: str, kind: int | None = None):
    """(positions in the method's arrays, the points' index) of one set's points (one kind)."""
    m = r.methods[method]
    sets = r.inputs["set"][m.index]
    keep = sets == sid
    if kind is not None:
        keep &= r.inputs["kind"][m.index] == kind
    pos = np.nonzero(keep)[0]
    return pos, r.inputs["index"][m.index][pos]


def per_set(r: reffile.Ref, method: str, sid: str, kind: int | None = None):
    """(index, hi, lo, fields) of a set's points with that method, ordered by the points' index."""
    pos, idx = at(r, method, sid, kind)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def group_of(r: reffile.Ref, sid: str) -> int:
    return int(np.nonzero(r.group_inputs["set"] == sid)[0][0])


def group_input(r: reffile.Ref, sid: str, name: str):
    v = r.group_inputs[name]
    return v[group_of(r, sid)]


def group_value(r: reffile.Ref, method: str, sid: str):
    """(hi, lo, fields) of a group method at one set, or None when the set has none."""
    m = r.methods[method]
    at_ = np.nonzero(m.index == group_of(r, sid))[0]
    if not len(at_):
        return None
    return m.hi[at_[0]], m.lo[at_[0]], {k: v[at_[0]] for k, v in m.fields.items()}


def sets_with(r: reffile.Ref, method: str) -> list[str]:
    """The set labels that have values of a method, in file order."""
    m = r.methods[method]
    if m.over == "group":
        return [str(r.group_inputs["set"][g]) for g in m.index]
    seen = dict.fromkeys(r.inputs["set"][m.index].tolist())
    return list(seen)


def matrix(r: reffile.Ref, method: str, sid: str, shape, kind: int | None = None):
    """The values of a method at one set laid out by their index (row-major) into an array of that shape; NaN
    where the set has no value."""
    idx, hi, _, _ = per_set(r, method, sid, kind)
    out = np.full(int(np.prod(shape)), np.nan)
    out[idx] = hi
    return out.reshape(shape)


# ------------------------------------------------------------------ the draft bounds


def svd_bound(n, p, s1):
    return C_SVD * U * math.sqrt(min(n, p)) * s1


def cov_bound(n, l1):
    return C_COV * U * n * l1


def vector_bound(n, l1, gap):
    """sin of the angle between a computed and the reference eigenvector (Davis-Kahan); inf at a zero gap."""
    return math.inf if gap <= 0 else C_VEC * U * n * l1 / gap


def dense_eig_bound(n, l1):
    return C_DENSE_EIG * U * n * abs(l1)


def fa_bound(p, value=0.0):
    return C_FA * gamma(p * p) * (1 + abs(value))


def step_bound(m, scale):
    return C_STEP * gamma(m) * scale


def stress_bound(m, p, plus):
    return gamma(math.ceil(math.log2(max(m, 2))) + p + 8) * plus


def lle_bound(k, wmax, reg=REG):
    return C_LLE * U * k * (1 + 1 / reg) * wmax


def geodesic_bound(n, p, value):
    return gamma(n + p / 2 + 2) * abs(value)


def p_bound(k, value):
    return C_P * gamma(k + 10) * abs(value)


def grad_bound(n, d, plus):
    return C_GRAD * gamma(n + d + 10) * plus
