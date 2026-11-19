"""Ints beyond the double range and requests beyond the machine's memory, in every area
(user decisions 2026-11-18): OverflowError and MemoryError that name the function, never
Python's bare message or a killed process."""

import pytest

import stardust as sd

BIG = 10**400


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda x, a: sd.var(x, correction=BIG), r"var\(\): 'correction' is out of range"),
        (lambda x, a: sd.quantile(x, BIG), r"quantile\(\): 'q' is out of range"),
        (lambda x, a: sd.histogram(x, bins=3, range=(0, BIG)), r"histogram\(\): 'range' is out of range"),
        (lambda x, a: sd.linalg.vector_norm(x, ord=BIG), r"vector_norm\(\): 'ord' is out of range"),
        (lambda x, a: sd.linalg.matrix_norm(a, ord=BIG), r"matrix_norm\(\): 'ord' is out of range"),
        (lambda x, a: sd.linalg.cond(sd.ones((2, 2)), ord=BIG), r"cond\(\): 'ord' is out of range"),
        (lambda x, a: sd.linalg.matrix_rank(a, rtol=BIG), r"matrix_rank\(\): 'rtol' is out of range"),
        (lambda x, a: sd.linalg.lstsq(a, x, rtol=BIG), r"lstsq\(\): 'rtol' is out of range"),
        (lambda x, a: sd.stats.Normal(mu=BIG), r"Normal\(\): 'mu' is out of range"),
        (lambda x, a: sd.asarray([BIG, 0.5]), r"asarray\(\): Python int too large to convert to float64"),
        (lambda x, a: sd.full((2,), BIG, dtype=sd.float64), r"full\(\): Python int too large to convert to float64"),
        (lambda x, a: sd.linspace(0, BIG, 3), r"linspace\(\): Python int too large to convert to float64"),
        (lambda x, a: sd.frame.Series([BIG, 0.5]), r"Series\(\): Python int too large to convert to float64"),
        (lambda x, a: x * BIG, r"Python int too large to convert to float64"),
    ],
)
def test_an_int_beyond_the_double_range(call, message):
    x = sd.asarray([1.0, 2.0, 3.0])
    a = sd.ones((3, 2))
    with pytest.raises(OverflowError, match="^" + message + "$"):
        call(x, a)


@pytest.mark.parametrize(
    "call,prefix",
    [
        (lambda: sd.zeros((2**50,)), r"zeros\(\): "),
        (lambda: sd.arange(2**50), r"arange\(\): "),
        (lambda: sd.random.default_rng(1).permutation(2**50), r"permutation\(\): "),
        (lambda: sd.random.default_rng(1).random((2**50,)), r"random\(\): "),
        (lambda: sd.ones((2**25, 1)) * sd.ones((1, 2**25)), ""),
    ],
)
def test_a_request_beyond_physical_memory_is_memory_error(call, prefix):
    # 2**53 bytes: refused before anything is allocated
    with pytest.raises(
        MemoryError, match="^" + prefix + r"cannot allocate \d+ bytes: more than this machine's physical memory$"
    ):
        call()
