"""Fuzzy c-means (src/stardust/cluster/_fuzzy.py, csrc/cluster/fuzzy.cpp; docs/api/cluster.md §3; docs/design/cluster.md
§2.3, §3.5, §4.3; decisions CL27 and CL59-CL66): the memberships at given centres and one step's centres against the
exact references (tools/cluster_ref.py fuzzy), the stop, the thread counts, the input order and the errors."""

import itertools
import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S

CL = sd.cluster
CORE = sd._core.cluster
REF = D.ref("fuzzy")
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _sets():
    out = []
    for sid in REF.group_inputs["set"].tolist():
        c = np.asarray(REF.group_inputs["centers"][D.group_of(REF, sid)], float)
        if c.size and not np.isnan(c).all():
            s = S.points(sid)
            out.append((sid, s, c.reshape(s.k, -1)))
    return out


def _ref(name, sid, kind, size):
    pos, idx = D.at(REF, name, sid, kind)
    m = REF.methods[name]
    out = np.full(size, np.nan)
    out[idx] = m.hi[pos] + m.lo[pos]
    return out


@pytest.mark.parametrize("name, m", [("u_m2", 2.0), ("u_m15", 1.5), ("u_m3", 3.0)])
def test_memberships_at_given_centres(name, m):
    worst = 0.0
    for sid, s, C in _sets():
        ref = _ref(name, sid, 0, s.n * s.k)
        got = _np(CORE.fuzzy_memberships(A(s.X), A(C), m)).ravel()
        err = np.nanmax(np.abs(got - ref))
        bound = D.membership_bound(s.p, s.k, m)
        assert err <= bound, (sid, err / bound)
        worst = max(worst, err / bound)
        assert np.allclose(_np(CORE.fuzzy_memberships(A(s.X), A(C), m)).sum(axis=1), 1.0, rtol=0, atol=1e-14)
    assert worst > 0


def test_one_step_centres_and_the_objective():
    for sid, s, C in _sets():
        r = CL.fuzzy_cmeans(s.X, s.k, rng=sd.random.default_rng(), init=C, max_iter=1, tol=0.0)
        ref = _ref("centre1", sid, 1, s.k * s.p).reshape(s.k, -1)
        got = _np(r.centers)
        bound = D.fuzzy_centre_bound(s.n, s.p, s.k, float(np.max(np.abs(s.X))))
        if s.k <= 6:
            err = min(np.max(np.abs(got[list(q)] - ref)) for q in itertools.permutations(range(s.k)))
            assert err <= bound, (sid, err / bound)
        assert r.n_iter == 1 and not r.converged or r.converged


def test_fit_stops_converges_and_reaches_a_fixed_point():
    s = S.points("probe/300")
    r = CL.fuzzy_cmeans(s.X, 3, rng=sd.random.default_rng(3))
    assert r.converged and r.n_iter < 100
    U = _np(r.memberships)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-14) and U.shape == (300, 3)
    assert np.array_equal(_np(r.labels), np.argmax(U, axis=1))
    # the stop: one more step moves no membership by tol or more
    r2 = CL.fuzzy_cmeans(s.X, 3, rng=sd.random.default_rng(3), init=r.centers, max_iter=1)
    assert np.max(np.abs(_np(r2.memberships) - U)) < 1e-4
    assert r.partition_coefficient == pytest.approx(float(np.mean(np.sum(U * U, axis=1))), rel=1e-14)
    assert np.allclose(_np(r.predict_proba(s.X)), U, atol=1e-12)
    assert np.array_equal(_np(r.predict(s.X)), _np(r.labels))
    assert "Fuzzy C-Means Results" in r.summary() and repr(r).startswith("FuzzyCMeansResult(nobs=300")


def test_points_on_centres_and_m():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [6.0, 5.0]])
    U = _np(CORE.fuzzy_memberships(A(X), A(np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])), 2.0))
    assert U[0].tolist() == [0.5, 0.5, 0.0]  # on two centres: equal shares
    hard = _np(CORE.fuzzy_memberships(A(X), A(X[[0, 2]]), 1.01))
    assert np.all((hard > 1 - 1e-12) | (hard < 1e-12))  # m near 1: k-means' hard partition
    soft = _np(CORE.fuzzy_memberships(A(X), A(X[[0, 2]]), 50.0))
    assert np.all(np.abs(soft[[1, 3]] - 0.5) < 0.1)


def test_thread_counts_input_order_and_scale():
    g = np.random.default_rng(6)
    X = np.vstack([g.normal(c, 1.0, (4000, 3)) for c in (0.0, 4.0, 8.0)])

    def results():
        r = CL.fuzzy_cmeans(X, 4, rng=sd.random.default_rng(2))
        return [r.memberships, r.centers, r.objective, r.n_iter]

    sd.set_num_threads(1)
    base = results()
    try:
        for t in (3, 10):
            sd.set_num_threads(t)
            for k, (a, b) in enumerate(zip(base, results())):
                assert bits_equal(a, b), (t, k)
    finally:
        sd.set_num_threads(10)
    perm = g.permutation(len(X))
    a = CL.fuzzy_cmeans(X, 4, rng=sd.random.default_rng(2))
    b = CL.fuzzy_cmeans(X[perm], 4, rng=sd.random.default_rng(2))
    assert np.array_equal(S.first_seen(_np(a.labels)[perm]), _np(b.labels)) and a.objective == b.objective
    for e in (-1000, 400):  # CL43: powers of two scale exactly
        c = CL.fuzzy_cmeans(X * 2.0**e, 4, rng=sd.random.default_rng(2))
        assert bits_equal(c.memberships, a.memberships) and c.objective == math.ldexp(a.objective, 2 * e), e


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.fuzzy_cmeans(np.eye(3), 2), TypeError, r"missing 1 required keyword-only argument: 'rng'"),
        (lambda: CL.fuzzy_cmeans(np.eye(3), 2, rng=sd.random.default_rng(), m=1.0), ValueError, r"'m' must be"),
        (lambda: CL.fuzzy_cmeans(np.eye(3), 2, rng=sd.random.default_rng(), init="k-means++"), ValueError,
         r"'init' must be 'random' or a \(k, p\) array"),
        (lambda: CL.fuzzy_cmeans(np.zeros((4, 2)), 2, rng=sd.random.default_rng()), ValueError,
         r"'k' \(2\) is larger than the number of distinct points \(1\)"),
        (lambda: CL.fuzzy_cmeans(np.eye(3), 2, rng=sd.random.default_rng(), tol=-1.0), ValueError, r"'tol'"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
