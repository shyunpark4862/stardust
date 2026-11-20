"""lstsq with rtol below eps solves by the SVD whose singular values matrix_rank reads, so its
rank is matrix_rank(rtol)'s (user decision 2026-11-19, docs/design/linalg.md §10.4).
gelsd's xLALSD moves entries below eps of the largest to eps before its divide and conquer
(min(m, n) > 25): a zero singular value counted as one."""

import numpy as np
import pytest

import stardust as sd


@pytest.mark.parametrize("n", [10, 25, 26, 30, 60])
def test_zero_column_rank_matches_matrix_rank(n):
    a = np.diag(np.arange(1.0, n + 1))
    a[:, 3] = 0.0
    A = sd.asarray(a)
    b = sd.asarray(np.ones(n))
    for rtol in (0, 0.0, 1e-300, 2.0**-53):
        r = sd.linalg.lstsq(A, b, rtol=rtol)
        assert r.rank == sd.linalg.matrix_rank(A, rtol=rtol) == n - 1
        # the singular values are svdvals' own, and the solution is the minimum-norm one
        assert np.array_equal(np.asarray(r.singular_values), np.asarray(sd.linalg.svdvals(A)))
        x = np.asarray(r.solution)
        want = np.ones(n) / np.arange(1.0, n + 1)
        want[3] = 0.0
        np.testing.assert_allclose(x, want, rtol=1e-14, atol=1e-15)


def test_tiny_singular_value_counts_at_rtol_zero():
    D = sd.asarray(np.diag([1.0, 5e-324]))
    r = sd.linalg.lstsq(D, sd.asarray([1.0, 1.0]), rtol=0)
    assert r.rank == sd.linalg.matrix_rank(D, rtol=0) == 2
    # a 2-column right-hand side and the batch path give the same rank
    rb = sd.linalg.lstsq(sd.asarray(np.stack([np.diag([1.0, 5e-324])] * 3)), sd.asarray(np.ones((3, 2, 2))), rtol=0)
    assert list(np.asarray(rb.rank)) == [2, 2, 2]


def test_rectangular_and_complex_agree_with_pinv():
    rng = np.random.default_rng(5)
    for m, n in ((40, 30), (30, 40)):
        u, _ = np.linalg.qr(rng.normal(size=(m, m)))
        v, _ = np.linalg.qr(rng.normal(size=(n, n)))
        s = np.zeros((m, n))
        k = min(m, n)
        s[np.arange(k), np.arange(k)] = np.linspace(1.0, 2.0, k)
        s[k - 1, k - 1] = 0.0
        a = u @ s @ v.T
        b = rng.normal(size=(m, 2))
        r = sd.linalg.lstsq(sd.asarray(a), sd.asarray(b), rtol=0)
        assert r.rank == sd.linalg.matrix_rank(sd.asarray(a), rtol=0)
        np.testing.assert_allclose(
            np.asarray(r.solution), np.asarray(sd.linalg.pinv(sd.asarray(a), rtol=0)) @ b, rtol=1e-9, atol=1e-9
        )
    c = rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))
    c[:, 2] = 0.0
    rc = sd.linalg.lstsq(sd.asarray(c), sd.asarray(np.ones(30) + 0j), rtol=0)
    assert rc.rank == sd.linalg.matrix_rank(sd.asarray(c), rtol=0)


LA = sd.linalg


def test_lstsq_rtol_zero_counts_every_positive_singular_value():
    # xLALSD replaces rcond <= 0 by machine epsilon; rtol = 0 must mean what it means in
    # matrix_rank (only exact zeros are zero)
    a = sd.ones((3, 2))
    b = sd.ones((3,))
    r = sd.linalg.lstsq(a, b, rtol=0.0)
    assert min(r.singular_values.tolist()) > 0.0
    assert r.rank == sd.linalg.matrix_rank(a, rtol=0.0) == 2
    assert sd.linalg.lstsq(a, b, rtol=1e-300).rank == 2
    assert sd.linalg.lstsq(a, b).rank == sd.linalg.matrix_rank(a) == 1
    z = sd.linalg.lstsq(sd.zeros((3, 2)), b, rtol=0.0)
    assert z.rank == 0 and z.solution.tolist() == [0.0, 0.0]


# ------------------------------------------------- lstsq alignment, messages --


@pytest.mark.parametrize("m, n, k", [(3, 3, 1), (5, 3, 3), (3, 5, 3), (9, 5, 1), (31, 29, 3)])
def test_lstsq_small_rtol_batch_is_one_by_one(m, n, k):
    # The SVD path writes the solution through gemm; an odd n * k puts every other
    # batch element's slot off 16 bytes (docs/design/linalg.md §6.4).
    rng = np.random.default_rng(m * 100 + n * 10 + k)
    a = rng.standard_normal((7, m, n))
    b = rng.standard_normal((7, m, k))
    whole = np.asarray(LA.lstsq(sd.asarray(a), sd.asarray(b), rtol=0.0).solution)
    for i in range(7):
        one = np.asarray(LA.lstsq(sd.asarray(a[i]), sd.asarray(b[i]), rtol=0.0).solution)
        assert np.array_equal(one, whole[i])
