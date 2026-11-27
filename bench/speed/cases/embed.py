"""sd.embed against scikit-learn 1.9.1, umap-learn 0.6.0, openTSNE 1.0.4 and R 4.6.2's stats, MASS, irlba, uwot,
Rtsne and fastICA (docs/design/embed.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the sd.embed names
it needs, and cases() keeps a spec once its names are public (EM1 the PCA and the truncated SVD, EM2 factor
analysis and FastICA, EM3 NMF and kernel PCA, EM4 the MDS methods, EM5 the neighbour graph methods and
trustworthiness, EM6 t-SNE, EM7 UMAP, EM8 the FFT t-SNE). The sizes are those of docs/design/embed.md §5, where
milestone EM0 timed the comparators (tools/embed_probe.py time, which times these specs' comparators on the same
data). scikit-learn runs on its OpenMP threads and Accelerate (it follows the thread count); umap-learn and openTSNE
take their thread count as an argument (n_jobs), so each count is a comparator of its own; R runs on one thread
(uwot also with ten: its RcppParallel threads are an argument).

Data: Gaussian, low-rank, factor-model, mixed-source, nonnegative and Swiss roll points from fixed seeds, made in
the setup as numpy arrays (the probe uses these setups); sd gets them as sd.ndarray (or sd.sparse.csr_array)
outside the timing. Every library starts its own way where the start is random (FastICA, t-SNE, UMAP), so those
cases are not value-checked; the deterministic ones compare singular values, eigenvalues, stress, uniquenesses or
the value itself.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref, RRef, count

SEED = 20261127


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]  # numpy arrays only (the probe calls it without stardust)
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""
    check: Check = Check()
    libs: tuple[str, ...] = ()  # comparator modules loaded before the timing starts (cluster CL103)
    timeout_s: float | None = None


# ---------------------------------------------------------------- data (numpy only)


def _g(*key) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([SEED, *key]))


def _lowrank(n, p, r=10):
    """Rank r with singular values 10 * 0.8^j * sqrt(n) on a random basis, plus noise of standard deviation 1."""

    def setup():
        g = _g(1, n, p, r)
        U = g.standard_normal((n, r))
        V = np.linalg.qr(g.standard_normal((p, r)))[0]
        s = 10.0 * 0.8 ** np.arange(r)
        X = (U * s) @ V.T + g.standard_normal((n, p)) + g.standard_normal(p)
        return {"X": X}

    return setup


def _sparse(n, p, density):
    """A random sparse matrix (CSR, about density * n * p uniform entries in (0, 1]), and its triplets for R."""

    def setup():
        import scipy.sparse as sp

        g = _g(2, n, p)
        nnz = int(round(density * n * p))
        flat = np.unique(g.integers(0, n * p, nnz))
        rows, cols = flat // p, flat % p
        vals = 1.0 - g.random(flat.size)
        S = sp.csr_array((vals, (rows, cols)), shape=(n, p))
        S.sort_indices()
        return {
            "S": S,
            "r_i": rows.astype(float) + 1.0,
            "r_j": cols.astype(float) + 1.0,
            "r_v": vals,
            "dims": np.array([n, p], float),
        }

    return setup


def _factor(n, p, k):
    """A k-factor model: loadings in +-[0.4, 0.9] on blocks, uniquenesses 1 - |l|^2; Z is X standardised with the
    population standard deviation (scikit-learn's noise variance on Z is the uniqueness)."""

    def setup():
        g = _g(3, n, p, k)
        L = g.uniform(0.4, 0.9, (p, k)) * np.sign(g.standard_normal((p, k)))
        block = np.arange(p) * k // p
        L[np.arange(k)[None, :] != block[:, None]] *= 0.25
        L *= np.minimum(1.0, 0.95 / np.sqrt((L**2).sum(axis=1)))[:, None]
        psi = 1.0 - (L**2).sum(axis=1)
        X = g.standard_normal((n, k)) @ L.T + np.sqrt(psi) * g.standard_normal((n, p))
        X = X * g.uniform(0.5, 3.0, p) + g.standard_normal(p)
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        return {"X": X, "Z": Z}

    return setup


def _sources(n, p):
    """p independent sources (Laplace, uniform, sine, square wave in turn) mixed by a random matrix."""

    def setup():
        g = _g(4, n, p)
        t = np.linspace(0.0, 200.0, n)
        cols = []
        for j in range(p):
            kind = j % 4
            if kind == 0:
                cols.append(g.laplace(size=n))
            elif kind == 1:
                cols.append(g.uniform(-1.0, 1.0, n))
            elif kind == 2:
                cols.append(np.sin((1.0 + 0.37 * j) * t))
            else:
                cols.append(np.sign(np.sin((0.5 + 0.21 * j) * t)))
        X = np.column_stack(cols) @ (g.standard_normal((p, p)) + 2.0 * np.eye(p)).T
        return {"X": X}

    return setup


def _nonneg(n, p, r):
    def setup():
        g = _g(5, n, p, r)
        X = g.random((n, r)) @ g.random((r, p)) + 0.01 * g.random((n, p))
        return {"X": X}

    return setup


def _blobs(n, p, k=10):
    """k Gaussian blobs (standard deviation 1) around centres 6 N(0, I), the points in a random order."""

    def setup():
        g = _g(6, n, p, k)
        centers = 6.0 * g.standard_normal((k, p))
        lab = g.integers(0, k, n)
        X = centers[lab] + g.standard_normal((n, p))
        return {"X": X, "labels": lab}

    return setup


def _swiss(n):
    """scikit-learn's Swiss roll formula with noise 0.05."""

    def setup():
        g = _g(7, n)
        t = 1.5 * np.pi * (1.0 + 2.0 * g.random(n))
        h = 21.0 * g.random(n)
        X = np.column_stack([t * np.cos(t), h, t * np.sin(t)]) + 0.05 * g.standard_normal((n, 3))
        return {"X": X}

    return setup


def _embedded(n, p):
    """Blobs in p dimensions and a 2-D embedding of them (their first two coordinates plus a little noise)."""

    def setup():
        d = _blobs(n, p)()
        g = _g(8, n, p)
        d["Y"] = d["X"][:, :2] + 0.5 * g.standard_normal((n, 2))
        return d

    return setup


def _with_sd(setup):
    """The setup, then the sd copies of its arrays: X -> sX, Y -> sY (sd.ndarray), S -> sS (sd.sparse.csr_array)."""

    def run():
        import stardust as sd

        d = setup()
        for k in ("X", "Y", "Z"):
            if k in d:
                d["s" + k] = sd.asarray(np.ascontiguousarray(d[k]))
        if "S" in d:
            S = d["S"]
            d["sS"] = sd.sparse.csr_array((S.data, S.indices, S.indptr), shape=S.shape)
        return d

    return run


# ---------------------------------------------------------------- helpers


def _em():
    import stardust as sd

    return sd.embed


def _rng(seed=0):
    import stardust as sd

    return sd.random.default_rng(seed)


def _sk(module, name):
    return getattr(__import__(module, fromlist=[name]), name)


def _np(a) -> np.ndarray:
    return np.asarray(a.tolist() if hasattr(a, "tolist") else a, dtype=float)


def _field(name):
    """A ref normaliser: an R value as given (an array, or a dict's `name`), a Python result's attribute."""

    def get(label, r):
        if isinstance(r, np.ndarray):
            return r
        if isinstance(r, dict):
            return np.asarray(r[name] if name in r else next(iter(r.values())), float)
        return _np(getattr(r, name))

    return get


def _sign_fixed(Z) -> np.ndarray:
    """Each column times the sign of its entry of largest magnitude (decision EM14; scikit-learn's
    _deterministic_vector_sign_flip on the columns)."""
    Z = _np(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    i = np.argmax(np.abs(Z), axis=0)
    return Z * np.sign(Z[i, np.arange(Z.shape[1])])


X_R = {"X": "x"}


# ---------------------------------------------------------------- EM1: PCA and the truncated SVD


def _pca_specs():
    out = []
    setup = _lowrank(10**5, 100)
    sk_solver = {"full": "full", "covariance_eigh": "covariance_eigh", "lanczos": "arpack", "randomized": "randomized"}
    for solver, sk in sk_solver.items():
        rng = solver == "randomized"
        refs = [
            Ref(
                f"sklearn {sk}",
                lambda d, sk=sk: (
                    lambda: _sk("sklearn.decomposition", "PCA")(10, svd_solver=sk, random_state=0).fit(d["X"])
                ),
                threads="follow",
            )
        ]
        if solver == "full":
            refs.append(RRef("R prcomp", "prcomp(x, rank. = 10)", data=X_R, value=".v$sdev[1:10] * sqrt(nrow(x) - 1)"))
        if solver == "lanczos":
            refs.append(
                RRef(
                    "R prcomp_irlba",
                    "prcomp_irlba(x, n = 10)",
                    data=X_R,
                    packages=("irlba",),
                    value=".v$sdev * sqrt(nrow(x) - 1)",
                )
            )
        out.append(
            Spec(
                "EM1",
                "linear",
                f"pca {solver} p=100 d=10",
                ("pca",),
                count(10**5),
                setup,
                lambda d, s=solver, rng=rng: (
                    lambda: _em().pca(d["sX"], 10, svd_solver=s, **({"rng": _rng()} if rng else {}))
                ),
                tuple(refs),
                quick=solver in ("full", "covariance_eigh"),
                check=Check(
                    sd=lambda r: _np(r.singular_values),
                    ref=_field("singular_values_"),
                    rtol=1e-3 if rng else 1e-8,
                ),
                libs=("sklearn.decomposition",),
            )
        )
    out.append(
        Spec(
            "EM1",
            "linear",
            "truncated_svd lanczos sparse 10^4 cols density=1e-3 d=100",
            ("truncated_svd",),
            count(10**5),
            _sparse(10**5, 10**4, 1e-3),
            lambda d: lambda: _em().truncated_svd(d["sS"], 100),
            (
                Ref(
                    "sklearn arpack",
                    lambda d: lambda: _sk("sklearn.decomposition", "TruncatedSVD")(100, algorithm="arpack").fit(d["S"]),
                    threads="follow",
                ),
                Ref(
                    "sklearn randomized",
                    lambda d: lambda: _sk("sklearn.decomposition", "TruncatedSVD")(100, random_state=0).fit(d["S"]),
                    threads="follow",
                    differs="a randomized approximation (n_iter = 5) of a flat spectrum",
                ),
                RRef(
                    "R irlba",
                    "irlba(A, nv = 100)",
                    data={"r_i": "ri", "r_j": "rj", "r_v": "rv", "dims": "dims"},
                    setup="A <- Matrix::sparseMatrix(i = ri, j = rj, x = rv, dims = dims)",
                    packages=("irlba", "Matrix"),
                    value=".v$d",
                ),
            ),
            check=Check(sd=lambda r: _np(r.singular_values), ref=_field("singular_values_"), rtol=1e-4),
            libs=("sklearn.decomposition",),
        )
    )
    return out


# ---------------------------------------------------------------- EM2: factor analysis and FastICA


def _factor_specs():
    return [
        Spec(
            "EM2",
            "linear",
            "factor_analysis p=50 d=5",
            ("factor_analysis",),
            count(10**4),
            _factor(10**4, 50, 5),
            lambda d: lambda: _em().factor_analysis(d["sX"], 5),
            (
                Ref(
                    "sklearn varimax",
                    lambda d: lambda: _sk("sklearn.decomposition", "FactorAnalysis")(5, rotation="varimax").fit(d["Z"]),
                    threads="follow",
                ),
                RRef("R factanal", "factanal(x, 5)", data=X_R, value=".v$uniquenesses"),
            ),
            quick=True,
            note="sklearn fits the standardised data, so its noise variance is the uniqueness",
            check=Check(sd=lambda r: _np(r.uniquenesses), ref=_field("noise_variance_"), rtol=1e-2),
            libs=("sklearn.decomposition",),
        ),
        Spec(
            "EM2",
            "linear",
            "fastica p=20",
            ("fastica",),
            count(10**5),
            _sources(10**5, 20),
            lambda d: lambda: _em().fastica(d["sX"], rng=_rng()),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.decomposition", "FastICA")(random_state=0).fit(d["X"]),
                    threads="follow",
                ),
                RRef("R fastICA", "fastICA(x, 20)", data=X_R, packages=("fastICA",)),
            ),
            check=Check(skip="random starts: each library's own w_init (the tests compare from the same start)"),
            libs=("sklearn.decomposition",),
        ),
    ]


# ---------------------------------------------------------------- EM3: NMF and kernel PCA


def _nmf_kernel_specs():
    return [
        Spec(
            "EM3",
            "linear",
            "nmf cd p=1000 d=20",
            ("nmf",),
            count(10**4),
            _nonneg(10**4, 1000, 20),
            lambda d: lambda: _em().nmf(d["sX"], 20, solver="cd"),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.decomposition", "NMF")(20, solver="cd").fit(d["X"]),
                    threads="follow",
                ),
            ),
            check=Check(sd=lambda r: float(r.reconstruction_err), ref=_field("reconstruction_err_"), rtol=1e-3),
            libs=("sklearn.decomposition",),
        ),
        Spec(
            "EM3",
            "linear",
            "kernel_pca rbf p=10 d=10",
            ("kernel_pca",),
            count(5000),
            _blobs(5000, 10),
            lambda d: lambda: _em().kernel_pca(d["sX"], 10, kernel="rbf"),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.decomposition", "KernelPCA")(10, kernel="rbf").fit(d["X"]),
                    threads="follow",
                ),
            ),
            quick=True,
            check=Check(sd=lambda r: _np(r.eigenvalues), ref=_field("eigenvalues_"), rtol=1e-8),
            libs=("sklearn.decomposition",),
        ),
    ]


# ---------------------------------------------------------------- EM4: MDS


def _mds_specs():
    return [
        Spec(
            "EM4",
            "distance",
            "mds smacof p=10 300 iterations",
            ("mds",),
            count(2000),
            _blobs(2000, 10),
            lambda d: lambda: _em().mds(d["sX"], 2, eps=0.0),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.manifold", "MDS")(2, init="classical_mds", eps=0.0).fit(d["X"]),
                    threads="follow",
                ),
            ),
            note="eps = 0: every library runs its 300 iterations from classical MDS",
            check=Check(sd=lambda r: float(r.stress), ref=_field("stress_"), rtol=1e-6),
            libs=("sklearn.manifold",),
        ),
        Spec(
            "EM4",
            "distance",
            "sammon p=10",
            ("sammon",),
            count(1000),
            _blobs(1000, 10),
            lambda d: lambda: _em().sammon(d["sX"], 2),
            (
                RRef(
                    "R sammon", "sammon(dist(x), k = 2, trace = FALSE)", data=X_R, packages=("MASS",), value=".v$stress"
                ),
            ),
            check=Check(sd=lambda r: float(r.stress), ref=_field("stress"), rtol=1e-6),
        ),
    ]


# ---------------------------------------------------------------- EM5: neighbour graphs and the quality metrics


def _graph_specs():
    return [
        Spec(
            "EM5",
            "graph",
            "isomap swiss roll k=5",
            ("isomap",),
            count(10**4),
            _swiss(10**4),
            lambda d: lambda: _em().isomap(d["sX"], 2),
            (Ref("sklearn", lambda d: lambda: _sk("sklearn.manifold", "Isomap")().fit(d["X"]), threads="follow"),),
            check=Check(
                sd=lambda r: _np(r.eigenvalues),
                ref=lambda label, r: _np(r.kernel_pca_.eigenvalues_),
                rtol=1e-6,
            ),
            libs=("sklearn.manifold",),
            timeout_s=30 * 60,
        ),
        Spec(
            "EM5",
            "graph",
            "locally_linear_embedding standard swiss roll k=5",
            ("locally_linear_embedding",),
            count(10**4),
            _swiss(10**4),
            lambda d: lambda: _em().locally_linear_embedding(d["sX"], 2),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.manifold", "LocallyLinearEmbedding")().fit(d["X"]),
                    threads="follow",
                ),
            ),
            check=Check(sd=lambda r: float(r.reconstruction_error), ref=_field("reconstruction_error_"), rtol=1e-3),
            libs=("sklearn.manifold",),
        ),
        Spec(
            "EM5",
            "graph",
            "spectral_embedding swiss roll k=15",
            ("spectral_embedding",),
            count(2 * 10**4),
            _swiss(2 * 10**4),
            lambda d: lambda: _em().spectral_embedding(d["sX"], 2, n_neighbors=15),
            (
                Ref(
                    "sklearn",
                    lambda d: (
                        lambda: _sk("sklearn.manifold", "SpectralEmbedding")(n_neighbors=15, random_state=0).fit(d["X"])
                    ),
                    threads="follow",
                ),
            ),
            note="n_neighbors = 15: the default n / 10 = 2000 neighbours is a dense graph",
            check=Check(
                sd=lambda r: _sign_fixed(r.embedding),
                ref=lambda label, r: _sign_fixed(r.embedding_),
                rtol=1e-4,
                agree=0.9,
            ),
            libs=("sklearn.manifold",),
        ),
        Spec(
            "EM5",
            "quality",
            "trustworthiness p=50 k=5",
            ("trustworthiness",),
            count(10**4),
            _embedded(10**4, 50),
            lambda d: lambda: _em().trustworthiness(d["sX"], d["sY"]),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.manifold", "trustworthiness")(d["X"], d["Y"]),
                    threads="follow",
                ),
            ),
            quick=True,
            note="n = 10^4, not design §5's 2 x 10^4: sklearn holds three n x n arrays (9.6 GB at 2 x 10^4)",
            check=Check(rtol=1e-12),
            libs=("sklearn.manifold",),
        ),
    ]


# ---------------------------------------------------------------- EM6, EM8: t-SNE


def _tsne_specs():
    rtsne = (
        "Rtsne(x, dims = 2, perplexity = 30, theta = 0.5, max_iter = 1000, pca = FALSE, check_duplicates = FALSE, "
        "num_threads = 1)"
    )
    return [
        Spec(
            "EM6",
            "neighbour",
            "tsne barnes_hut p=50",
            ("tsne",),
            count(2 * 10**4),
            _blobs(2 * 10**4, 50),
            lambda d: lambda: _em().tsne(d["sX"], 2),
            (
                Ref(
                    "sklearn",
                    lambda d: lambda: _sk("sklearn.manifold", "TSNE")(random_state=0).fit(d["X"]),
                    threads="follow",
                ),
                RRef(
                    "R Rtsne",
                    rtsne,
                    data=X_R,
                    setup="set.seed(1)",
                    packages=("Rtsne",),
                    differs="a random start; Rtsne normalises the data and uses eta = 200",
                ),
            ),
            check=Check(skip="a non-convex fit in float32 (sklearn) and float64 (sd): the tests compare KL"),
            libs=("sklearn.manifold",),
            timeout_s=60 * 60,
        ),
        Spec(
            "EM8",
            "neighbour",
            "tsne fft p=50",
            ("tsne",),
            count(10**5),
            _blobs(10**5, 50),
            lambda d: lambda: _em().tsne(d["sX"], 2, method="fft"),
            tuple(
                Ref(
                    f"openTSNE n_jobs={jobs}",
                    lambda d, jobs=jobs: (
                        lambda: _sk("openTSNE", "TSNE")(
                            n_iter=750, negative_gradient_method="fft", n_jobs=jobs, random_state=0
                        ).fit(d["X"])
                    ),
                    lib="openTSNE",
                )
                for jobs in (1, 10)
            ),
            note="openTSNE runs 250 + 750 iterations, as sklearn's and sd's 1000",
            check=Check(skip="a non-convex fit from each library's own start: the tests compare KL"),
            libs=("openTSNE",),
            timeout_s=60 * 60,
        ),
    ]


# ---------------------------------------------------------------- EM7: UMAP


def _umap_specs():
    def umap_learn(d, jobs, seed, compat):
        def run():
            import warnings

            UMAP = _sk("umap", "UMAP")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return UMAP(n_jobs=jobs, random_state=seed, compatibility_layout=compat).fit(d["X"])

        return run

    uwot = (
        "umap(x, n_neighbors = 15, min_dist = 0.1, batch = TRUE, n_threads = {t}, n_sgd_threads = {t}, seed = 1, "
        "verbose = FALSE)"
    )
    learn = (
        ("umap-learn n_jobs=1 seeded", 1, 0, True),
        ("umap-learn n_jobs=10", 10, None, True),
        ("umap-learn adam n_jobs=1 seeded", 1, 0, False),
        ("umap-learn adam n_jobs=10 seeded", 10, 0, False),
    )
    refs = [Ref(label, lambda d, j=j, s=s, c=c: umap_learn(d, j, s, c), lib="umap-learn") for label, j, s, c in learn]
    refs += [RRef(f"R uwot batch n_threads={t}", uwot.format(t=t), data=X_R, packages=("uwot",)) for t in (1, 10)]
    return [
        Spec(
            "EM7",
            "neighbour",
            "umap p=50",
            ("umap",),
            count(7 * 10**4),
            _blobs(7 * 10**4, 50),
            lambda d: lambda: _em().umap(d["sX"], 2, rng=_rng()),
            tuple(refs),
            note=(
                "umap-learn: the 0.5.x layout seeded (one thread), unseeded (Hogwild) and the 0.6 Adam layout (seeded: "
                "the same bytes at one thread count); uwot "
                "batch = TRUE; both find approximate neighbours (sd: exact)"
            ),
            check=Check(skip="a non-convex fit from each library's own start: the tests compare trustworthiness"),
            libs=("umap",),
            timeout_s=60 * 60,
        )
    ]


SPECS = _pca_specs() + _factor_specs() + _nmf_kernel_specs() + _mds_specs() + _graph_specs() + _tsne_specs()
SPECS += _umap_specs()


def _public(names) -> bool:
    import stardust as sd

    mod = getattr(sd, "embed", None)
    return mod is not None and all(n in getattr(mod, "__all__", ()) for n in names)


def _loaded(setup, libs):
    """The setup, after loading the comparators' libraries: threadpoolctl limits the OpenMP pools it finds when a
    measurement starts, so a library imported lazily inside the first measurement would run its threads unlimited
    at "1 thread" (cluster CL103)."""

    def run():
        import importlib

        for m in libs:
            importlib.import_module(m)
        return setup()

    return run


def cases() -> list[Case]:
    return [
        Case(
            area="embed",
            group=s.group,
            name=s.name,
            size=s.size,
            setup=_loaded(_with_sd(s.setup), s.libs),
            sd=s.sd,
            refs=s.refs,
            quick=s.quick,
            note=s.note,
            check=s.check,
            timeout_s=s.timeout_s,
        )
        for s in SPECS
        if _public(s.needs)
    ]


TITLES = {
    "linear": "선형 분해 (`pca`, `truncated_svd`, `factor_analysis`, `fastica`, `nmf`, `kernel_pca`)",
    "distance": "거리 보존 (`mds`, `sammon`)",
    "graph": "이웃 그래프 (`isomap`, `locally_linear_embedding`, `spectral_embedding`)",
    "neighbour": "확률적 이웃 임베딩 (`tsne`, `umap`)",
    "quality": "지표 (`trustworthiness`)",
}
