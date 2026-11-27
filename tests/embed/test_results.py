"""The common base of the results and the shared argument checks (src/stardust/embed/_results.py, _common.py and
the learning areas' stardust/_learn.py; docs/api/embed.md §1.2, §1.3; decisions EM7-EM11, EM44, EM45): read-only
fields and arrays, lazy fields computed once, the summary box with its component table and the one-line repr; the
data, nan_policy, precomputed distances, rng, X_new and Z. The subclasses come with milestones EM1 to EM7;
stand-in subclasses check the base here, and the checks are the same functions as sd.cluster's (EM44)."""

import math

import numpy as np
import pytest

import stardust as sd
from stardust import _learn
from stardust.cluster import _common as CC
from stardust.embed import _common as C
from stardust.embed._results import EmbeddingResult


class _Fit(EmbeddingResult):
    _TITLE = "PCA Results"
    _METHOD = "PCA ({svd_solver})"
    _TUNING = (("Solver:", "svd_solver"), ("Whiten:", "whiten"))
    _COLUMNS = (("Variance", "explained_variance"), ("Ratio", "explained_variance_ratio"))
    _CUMULATIVE = "explained_variance_ratio"
    _REPR = ("nobs", "n_components", "explained_variance_ratio")


class _Iter(EmbeddingResult):
    _TITLE = "t-SNE Results"
    _METHOD = "t-SNE ({method})"
    _OBJECTIVE = ("KL divergence:", "kl_divergence")
    _TUNING = (("Perplexity:", "perplexity"),)
    _REPR = ("nobs", "n_components", "kl_divergence")


def _fit(**extra):
    calls = []

    def covariance():
        calls.append(1)
        return sd.asarray([[2.0, 0.5], [0.5, 1.0]])

    fields = dict(
        embedding=sd.asarray([[1.0, 0.0], [-1.0, 0.5], [0.0, -0.5]]),
        n_components=2,
        nobs=3,
        omitted=sd.zeros((3,), dtype=sd.bool),
        explained_variance=sd.asarray([1.25, 0.25]),
        explained_variance_ratio=sd.asarray([0.75, 0.15]),
        svd_solver="full",
        whiten=False,
        covariance=covariance,
    )
    fields.update(extra)
    return _Fit(**fields), calls


def test_fields_and_arrays_are_read_only():
    r, _ = _fit()
    with pytest.raises(ValueError, match="read-only"):
        r.embedding[0, 0] = 5.0
    with pytest.raises(AttributeError, match="read-only"):
        r.nobs = 4
    with pytest.raises(AttributeError, match="read-only"):
        del r.nobs
    with pytest.raises(AttributeError, match="has no field 'components'"):
        r.components  # noqa: B018


def test_lazy_fields_are_computed_once_and_read_only():
    r, calls = _fit()
    assert calls == []
    a = r.covariance
    assert a is r.covariance and calls == [1]
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0] = 1.0
    assert "covariance" in dir(r) and "embedding" in dir(r)


def test_the_base_is_the_cluster_areas():
    # EM44: one field mechanics for the learning areas
    assert issubclass(EmbeddingResult, _learn.LearnResult)
    assert issubclass(sd.cluster._results.ClusterResult, _learn.LearnResult)


def test_summary_is_the_regression_box_with_a_row_per_component():
    r, _ = _fit()
    lines = r.summary().splitlines()
    assert lines[0].strip() == "PCA Results"
    assert lines[1] == "=" * 78 and lines[-1] == "=" * 78
    assert lines[2].startswith("Method:") and lines[2].split("Method:")[1].split("Solver:")[0].strip() == "PCA (full)"
    head = "\n".join(lines[2:6])
    for label in ("No. Observations:", "No. Components:", "Solver:", "Whiten:"):
        assert label in head, label
    assert "Omitted" not in head and "Iterations" not in head
    table = lines[lines.index("-" * 78) - 1 :]
    assert table[0].split() == ["component", "Variance", "Ratio", "Cumulative"]
    assert table[2].split() == ["0", "1.2500", "0.7500", "0.7500"]
    assert table[3].split() == ["1", "0.2500", "0.1500", "0.9000"]
    assert all(len(line) <= 78 for line in lines)


def test_summary_of_an_iterative_result_has_no_table():
    r = _Iter(
        embedding=sd.zeros((4, 2)),
        n_components=2,
        nobs=3,
        omitted=sd.asarray([False, True, False, False]),
        kl_divergence=0.8125,
        n_iter=1000,
        converged=False,
        method="barnes_hut",
        perplexity=30.0,
    )
    text = r.summary()
    assert "-" * 78 not in text
    assert text.split("Omitted rows (NaN):")[1].split()[0] == "1"
    assert text.split("KL divergence:")[1].split()[0] == "0.8125"
    assert text.split("Iterations:")[1].split()[0] == "1000"
    assert text.split("Converged:")[1].split()[0] == "False"
    assert repr(r) == "_Iter(nobs=3, n_components=2, kl_divergence=0.8125)"


def test_summary_cuts_a_long_table():
    d = 25
    r, _ = _fit(n_components=d, explained_variance=sd.ones((d,)), explained_variance_ratio=sd.full((d,), 0.04))
    lines = r.summary().splitlines()
    assert sum(1 for line in lines if line.split()[:1] == ["..."]) == 1
    assert lines[-2].split() == ["24", "1.0000", "0.0400", "1.0000"]


def test_repr_is_one_line():
    r, _ = _fit()
    assert repr(r) == "_Fit(nobs=3, n_components=2, explained_variance_ratio=[0.75, 0.15])"


def test_methods_report_their_own_name():
    r, _ = _fit()
    with pytest.raises(TypeError, match=r"^summary\(\): "):
        r.summary(1)


# ---- the shared checks (EM44: stardust._learn's, the same functions as sd.cluster's)


def test_the_checks_are_the_cluster_areas():
    for name in ("points", "rows", "metric", "check_rng", "no_rng", "key", "new_points", "check_alloc", "distances"):
        assert getattr(C, name) is getattr(CC, name) is getattr(_learn, name), name


def test_points_take_every_input_kind():
    from stardust import frame

    assert tuple(C.points([1, 2, 3], "f").shape) == (3, 1)
    assert C.points(np.ones((4, 2), dtype=np.float32), "f").dtype == sd.float64
    df = frame.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    assert C.points(df, "f").tolist() == [[1.0, 3.0], [2.0, 4.0]]
    with pytest.raises(TypeError, match=r"^pca\(\): 'X' must be real, not complex128$"):
        C.points([1 + 2j], "pca")


def test_nan_policy_and_rows_back():
    X = C.points([[1.0, 2.0], [math.nan, 1.0], [3.0, 4.0]], "f")
    with pytest.raises(ValueError, match=r"^f\(\): the input contains NaN \(nan_policy='raise'\)$"):
        C.rows(X, "f", "raise")
    kept, _, omitted = C.rows(X, "f", "omit")
    assert kept.tolist() == [[1.0, 2.0], [3.0, 4.0]] and omitted.tolist() == [False, True, False]
    Z = C.rows_back(sd.asarray([[0.5], [1.5]]), omitted)
    assert Z[0, 0] == 0.5 and math.isnan(Z[1, 0]) and Z[2, 0] == 1.5
    assert C.rows_back(kept, None) is kept
    with pytest.raises(ValueError, match=r"'nan_policy' must be 'raise' or 'omit', not 'propagate'"):
        C.rows(X, "f", "propagate")  # EM9: no 'propagate'


def test_new_points_and_z():
    with pytest.raises(ValueError, match=r"^transform\(\): 'X_new' must have 5 columns like the fit's 'X', not 3$"):
        C.new_points(np.zeros((2, 3)), "transform", 5)
    with pytest.raises(ValueError, match=r"^transform\(\): 'X_new' contains NaN$"):
        C.new_points([[math.nan]], "transform", 1)
    with pytest.raises(ValueError, match=r"^inverse_transform\(\): 'Z' must have 2 columns like the embedding, not 3$"):
        C.embedded(np.zeros((4, 3)), "inverse_transform", 2)
    with pytest.raises(ValueError, match=r"^inverse_transform\(\): 'Z' contains infinity$"):
        C.embedded([[math.inf, 0.0]], "inverse_transform", 2)
    assert C.embedded([[1.0, 2.0]], "inverse_transform", 2).tolist() == [[1.0, 2.0]]


def test_rng_rules():
    # EM11: the options decide; a call takes two words
    with pytest.raises(TypeError, match=r"^tsne\(\): init='random' needs 'rng' \(sd.random.default_rng\(seed\)\)$"):
        C.one_of_rng(None, "tsne", True, "init='random'")
    with pytest.raises(ValueError, match=r"^pca\(\): 'rng' is only used with svd_solver='randomized'$"):
        C.one_of_rng(sd.random.default_rng(0), "pca", False, "svd_solver='randomized'")
    assert C.one_of_rng(None, "pca", False, "svd_solver='randomized'") is None
    g = sd.random.default_rng(5)
    assert C.one_of_rng(g, "tsne", True, "init='random'") is g
    a, b = sd.random.default_rng(7), sd.random.default_rng(7)
    assert len(C.key(a)) == 2
    b.integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,))
    assert a.random() == b.random()


def test_precomputed_distances_and_metric():
    d, n = C.distances(np.array([[0.0, 1.0], [1.0, 0.0]]), "mds")
    assert n == 2 and d.tolist() == [1.0]
    with pytest.raises(ValueError, match=r"^mds\(\): the distance matrix 'X' must be symmetric$"):
        C.distances(np.array([[0.0, 1.0], [2.0, 0.0]]), "mds")
    assert C.metric("precomputed", "isomap") == "precomputed"
    with pytest.raises(ValueError, match=r"^umap\(\): 'metric' must be 'euclidean', .* not 'l2'$"):
        C.metric("l2", "umap")
