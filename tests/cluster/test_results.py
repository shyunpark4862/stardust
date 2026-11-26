"""The common base of the results and the shared argument checks (src/stardust/cluster/_results.py, _common.py;
docs/api/cluster.md §1.2, §1.3; decisions CL1, CL6, CL8, CL11, CL12, CL41, CL44): read-only fields and arrays,
lazy fields computed once, the summary box and the one-line repr; the data, nan_policy, weights, precomputed
distances and rng. The subclasses come with milestones CL1 to CL6; stand-in subclasses check the base here."""

import math

import numpy as np
import pytest

import stardust as sd
from stardust.cluster import _common as C
from stardust.cluster._results import ClusterResult


class _Fit(ClusterResult):
    _TITLE = "K-Means Results"
    _METHOD = "k-means ({init})"
    _OBJECTIVE = ("Inertia:", "inertia")
    _TUNING = (("Init:", "init"), ("No. Starts:", "n_init"), ("Tol:", "tol"))
    _COLUMNS = (("Within SS", "within"),)
    _REPR = ("nobs", "n_clusters", "inertia", "n_iter")


def _fit(**extra):
    calls = []

    def centers():
        calls.append(1)
        return sd.asarray([[0.0, 1.0], [5.0, 5.5]])

    fields = dict(
        labels=sd.asarray([0, 0, 1, 1, 1], dtype=sd.int64),
        n_clusters=2,
        nobs=5,
        omitted=sd.zeros((5,), dtype=sd.bool),
        inertia=573.1685,
        n_iter=6,
        converged=True,
        init="k-means++",
        n_init=1,
        tol=1e-4,
        within=sd.asarray([1.25, 571.9185]),
        centers=centers,
    )
    fields.update(extra)
    return _Fit(**fields), calls


def test_fields_and_arrays_are_read_only():
    r, _ = _fit()
    with pytest.raises(ValueError, match="read-only"):
        r.labels[0] = 5
    with pytest.raises(AttributeError, match="read-only"):
        r.nobs = 4
    with pytest.raises(AttributeError, match="read-only"):
        del r.nobs
    with pytest.raises(AttributeError, match="has no field 'medoids'"):
        r.medoids  # noqa: B018


def test_lazy_fields_are_computed_once_and_read_only():
    r, calls = _fit()
    assert calls == []
    a = r.centers
    b = r.centers
    assert a is b and calls == [1]
    with pytest.raises(ValueError, match="read-only"):
        a[0, 0] = 1.0
    assert "centers" in dir(r) and "labels" in dir(r)


def test_summary_is_the_regression_box_with_a_row_per_cluster():
    r, _ = _fit()
    lines = r.summary().splitlines()
    assert lines[0].strip() == "K-Means Results"
    assert lines[1] == "=" * 78 and lines[-1] == "=" * 78
    assert (
        lines[2].startswith("Method:")
        and lines[2].split("Method:")[1].split("Init:")[0].strip() == "k-means (k-means++)"
    )
    head = "\n".join(lines[2:8])
    for label in ("No. Observations:", "No. Clusters:", "Inertia:", "Iterations:", "Converged:", "No. Starts:", "Tol:"):
        assert label in head, label
    assert "573.2" in head and "Noise" not in head and "Omitted" not in head
    table = lines[lines.index("-" * 78) - 1 :]
    assert table[0].split() == ["cluster", "Size", "Within", "SS"]
    assert table[2].split() == ["0", "2", "1.2500"] and table[3].split() == ["1", "3", "571.9185"]
    assert all(len(line) <= 78 for line in lines)


def test_summary_counts_noise_and_omitted_rows_apart():
    r, _ = _fit(
        labels=sd.asarray([0, -1, 1, -1, 1, -1], dtype=sd.int64),
        nobs=5,
        omitted=sd.asarray([False, False, False, True, False, False]),
    )
    text = r.summary()
    assert "Noise points:" in text and text.split("Noise points:")[1].split()[0] == "2"
    assert "Omitted rows (NaN):" in text and text.split("Omitted rows (NaN):")[1].split()[0] == "1"


def test_summary_cuts_a_long_table():
    k = 25
    r, _ = _fit(labels=sd.asarray(list(range(k)), dtype=sd.int64), n_clusters=k, nobs=k,
                omitted=sd.zeros((k,), dtype=sd.bool), within=sd.zeros((k,)))  # fmt: skip
    lines = r.summary().splitlines()
    assert sum(1 for line in lines if line.split()[:1] == ["..."]) == 1
    assert lines[-2].split()[0] == "24"


def test_repr_is_one_line():
    r, _ = _fit()
    assert repr(r) == "_Fit(nobs=5, n_clusters=2, inertia=573.168, n_iter=6)"


def test_methods_report_their_own_name():
    r, _ = _fit()
    with pytest.raises(TypeError, match=r"^summary\(\): "):
        r.summary(1)


# ---- the shared checks


def test_points_take_every_input_kind():
    from stardust import frame

    assert tuple(C.points([1, 2, 3], "f").shape) == (3, 1)
    assert tuple(C.points(np.ones((4, 2)), "f").shape) == (4, 2)
    assert C.points(np.ones((4, 2), dtype=np.float32), "f").dtype == sd.float64
    assert C.points([[True, False]], "f").tolist() == [[1.0, 0.0]]
    df = frame.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    assert C.points(df, "f").tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert C.points(df["a"], "f").tolist() == [[1.0], [2.0]]
    t = sd.asarray(np.arange(12.0).reshape(3, 4))[:, ::2]
    assert C.points(t, "f")._flags()["c_contiguous"]


@pytest.mark.parametrize(
    "x, err, msg",
    [
        (3.0, ValueError, r"^f\(\): 'X' must be 1-D or 2-D, not 0-D$"),
        (np.ones((2, 2, 2)), ValueError, r"^f\(\): 'X' must be 1-D or 2-D, not 3-D$"),
        (np.ones((0, 2)), ValueError, r"^f\(\): 'X' must have at least 1 point$"),
        (np.ones((2, 0)), ValueError, r"^f\(\): 'X' must have at least 1 column$"),
        ([1 + 2j], TypeError, r"^f\(\): 'X' must be real, not complex128$"),
        ("abc", TypeError, r"^f\(\): 'X' must be "),
        (None, TypeError, r"^f\(\): 'X' must be "),
    ],
)
def test_points_errors(x, err, msg):
    with pytest.raises(err, match=msg):
        C.points(x, "f")


def test_rows_and_nan_policy():
    X = C.points([[1.0, 2.0], [math.nan, 1.0], [3.0, 4.0]], "f")
    with pytest.raises(ValueError, match=r"^f\(\): the input contains NaN \(nan_policy='raise'\)$"):
        C.rows(X, "f", "raise")
    kept, w, omitted = C.rows(X, "f", "omit")
    assert kept.tolist() == [[1.0, 2.0], [3.0, 4.0]] and w is None and omitted.tolist() == [False, True, False]
    assert C.labels_back(sd.asarray([0, 1], dtype=sd.int64), omitted).tolist() == [0, -1, 1]
    assert C.rows(C.points([[1.0], [2.0]], "f"), "f", "raise")[2] is None
    with pytest.raises(ValueError, match="contains infinity"):
        C.rows(C.points([[math.inf]], "f"), "f", "omit")
    with pytest.raises(ValueError, match=r"no row is left without NaN"):
        C.rows(C.points([[math.nan]], "f"), "f", "omit")
    with pytest.raises(ValueError, match=r"'nan_policy' must be 'raise' or 'omit', not 'propagate'"):
        C.rows(X, "f", "propagate")


def test_weights():
    X = C.points([[1.0], [2.0], [3.0]], "f")
    w = C.weights([1.0, math.nan, 0.0], "f", 3)
    _, kept, omitted = C.rows(X, "f", "omit", w)
    assert kept.tolist() == [1.0, 0.0] and omitted.tolist() == [False, True, False]
    with pytest.raises(ValueError, match="at least one weight must be positive"):
        C.rows(X, "f", "raise", C.weights([0.0, 0.0, 0.0], "f", 3))
    with pytest.raises(ValueError, match=r"'weights' must be >= 0"):
        C.weights([1.0, -1.0, 0.0], "f", 3)
    with pytest.raises(ValueError, match=r"one value per point \(3\), not 2"):
        C.weights([1.0, 2.0], "f", 3)


def test_precomputed_distances():
    D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    d, n = C.distances(D, "f")
    assert n == 3 and d.tolist() == [1.0, 2.0, 3.0]
    d, n = C.distances([1.0, 2.0, 3.0], "f")
    assert n == 3 and d.tolist() == [1.0, 2.0, 3.0]
    bad = D.copy()
    bad[0, 1] = np.nextafter(1.0, 2.0)
    with pytest.raises(ValueError, match="must be symmetric"):
        C.distances(bad, "f")
    bad = D + np.eye(3)
    with pytest.raises(ValueError, match="zero diagonal"):
        C.distances(bad, "f")
    with pytest.raises(ValueError, match=r"length n \(n - 1\) / 2 for some n >= 2, not 4"):
        C.distances([1.0, 2.0, 3.0, 4.0], "f")
    with pytest.raises(ValueError, match="must be >= 0"):
        C.distances([1.0, -2.0, 3.0], "f")
    with pytest.raises(ValueError, match="contain NaN"):
        C.distances(np.where(np.eye(3) == 1, 0.0, np.nan), "f")
    with pytest.raises(ValueError, match="must be square, not 2 x 3"):
        C.distances(np.zeros((2, 3)), "f")


def test_metric_names():
    assert C.metric("cityblock", "f") == "cityblock"
    assert C.metric("precomputed", "f") == "precomputed"
    with pytest.raises(ValueError, match=r"unknown metric 'precomputed'"):
        C.metric("precomputed", "f", precomputed=False)
    with pytest.raises(ValueError, match=r"unknown metric 'l2'"):
        C.metric("l2", "f")
    with pytest.raises(TypeError, match=r"'metric' must be a metric name \(str\), not int"):
        C.metric(2, "f")
    assert set(C.METRICS) <= set(__import__("stardust.spatial._common", fromlist=["x"]).METRICS)


def test_rng_takes_two_words():
    # CL8, CL41: a call moves the generator by exactly two 64-bit words, as regression's rlm(method='MM')
    a, b = sd.random.default_rng(7), sd.random.default_rng(7)
    key = C.key(a)
    assert len(key) == 2 and all(-(2**63) <= v < 2**63 for v in key)
    b.integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,))
    assert a.random() == b.random()
    with pytest.raises(TypeError, match=r"^f\(\): 'rng' must be an sd.random.Generator"):
        C.check_rng(3, "f")
    with pytest.raises(ValueError, match=r"^f\(\): 'rng' is not used with method='pam'$"):
        C.no_rng(sd.random.default_rng(0), "f", "method='pam'")


def test_numbers():
    assert C.positive_int(np.int64(3), "f", "k") == 3
    with pytest.raises(ValueError, match=r"'k' must be >= 1, not 0"):
        C.positive_int(0, "f", "k")
    with pytest.raises(TypeError, match=r"'k' must be an int, not bool"):
        C.positive_int(True, "f", "k")
    with pytest.raises(TypeError, match=r"'k' must be an int, not float"):
        C.positive_int(2.0, "f", "k")
    assert C.condensed_size(6, "f", "y") == 4
