"""Shared helpers. numpy and scipy are the references ("oracles"); results are
checked by residuals first, and directly against numpy where the answer is
unique (DESIGN.md §2.6, ARCHITECTURE.md §6)."""

import numpy as np
import stardust_numpy as sdnp
from hypothesis import strategies as st

EPS = np.finfo(np.float64).eps

# Residual tolerance factor c in  ||residual|| <= c * n * eps * scale  (decided in M1).
RESIDUAL_C = 10.0

DTYPES = ["bool", "int64", "float64", "complex128"]
LAYOUTS = ["C", "F", "strided", "reversed", "bcast"]


def sd(x):
    """numpy -> sdnp without a copy (the strides are kept)."""
    return sdnp.asarray(x)


def npy(x):
    """sdnp -> numpy without a copy."""
    return np.asarray(x)


def compute_dtype(*dtypes):
    return np.complex128 if "complex128" in dtypes else np.float64


def random_values(rng, shape, dtype):
    if dtype == "bool":
        return rng.integers(0, 2, shape).astype(np.bool_)
    if dtype == "int64":
        return rng.integers(-3, 4, shape).astype(np.int64)
    if dtype == "float64":
        return rng.standard_normal(shape)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def well_conditioned(rng, batch, n, dtype):
    """Square matrices that are safely nonsingular in every dtype."""
    if dtype == "bool":  # unit lower triangular: determinant 1
        low = rng.integers(0, 2, (*batch, n, n)).astype(np.bool_)
        return np.tril(low, -1) | np.eye(n, dtype=np.bool_)
    a = random_values(rng, (*batch, n, n), dtype)
    # A dominant diagonal; keeps the dtype. For int64 (entries in [-3, 3]) it is
    # strictly dominant: diagonal >= 3n - 2 > 3(n - 1) >= any off-diagonal row sum.
    return a + (3 * n + 1) * np.eye(n, dtype=np.int64)


def with_layout(x, layout):
    """The same values as `x` (numpy, ndim >= 2) in a different memory layout."""
    if layout == "C":
        return np.ascontiguousarray(x)
    if layout == "F":  # every matrix column-major
        return np.swapaxes(np.ascontiguousarray(np.swapaxes(x, -1, -2)), -1, -2)
    if layout == "strided":  # every other element along the last two axes
        big = np.zeros((*x.shape[:-2], 2 * x.shape[-2] + 1, 2 * x.shape[-1] + 1), x.dtype)
        big[..., 1::2, 1::2] = x
        return big[..., 1::2, 1::2]
    if layout == "reversed":  # negative strides
        return np.ascontiguousarray(x[..., ::-1, ::-1])[..., ::-1, ::-1]
    if layout == "bcast":  # batch axis 0 has stride 0 (values repeat)
        if x.ndim < 3 or x.shape[0] == 0:
            return x
        return np.broadcast_to(x[:1], x.shape)
    raise ValueError(layout)


@st.composite
def batch_shapes(draw, max_dims=2, max_size=3):
    return tuple(draw(st.lists(st.integers(0, max_size), max_size=max_dims)))


def residual_ok(a, x, b):
    """Normwise backward error of a solve, in the infinity norm:
    ||a x - b|| <= c * n * eps * (||a|| ||x|| + ||b||) for every batch element."""
    a, x, b = (np.asarray(t) for t in (a, x, b))
    if x.ndim == a.ndim - 1:  # vector right-hand side
        x, b = x[..., None], b[..., None]
    n = a.shape[-1]
    if n == 0 or x.size == 0:
        return True
    r = a @ x - b
    inf_norm = lambda m: np.abs(m).sum(axis=-1).max(axis=-1)  # noqa: E731
    lhs = inf_norm(r)
    rhs = RESIDUAL_C * n * EPS * (inf_norm(a) * inf_norm(x) + inf_norm(b))
    return bool(np.all(lhs <= rhs))
