"""The test matrices of sd.sparse (docs/design/sparse.md §7): generated from formulas and fixed
seeds, nothing downloaded. The tests, the reference tool (tools/sparse_ref.py), the scipy and R
records (tools/gen_sparse_scipy.py, tools/gen_sparse_r.py) and the measurements all take their
matrices from here, by id.

A matrix is held as coordinates (numpy arrays: `row`, `col` int64, `data` of one of the four
dtypes), the form every library accepts. Most are canonical (row-major order, no duplicates);
the COO inputs of the "coo/..." family are deliberately not: unsorted, with duplicates (summed
in input order, left to right: docs/design/sparse.md §2.3) and explicit zeros. `csr()` and
`csc()` give canonical compressed arrays (int64 indices), `dense()` the dense matrix.

    import tests.sparse.matrices as M
    a = M.get("lap2d/30")              # cached; M.ids(tag="spd", small=True) lists ids
    indptr, indices, data = a.csr()
    b = M.rhs(a)                       # a fixed right-hand side (n,), or M.rhs(a, k=3) (n, 3)

Tags say what a matrix is for: spd, symmetric, hermitian, unsym, indefinite, complex, singular
(exactly singular), struct_singular, near_singular, ill (condition number about 1e8-1e12),
rect (least squares), rank_deficient, banded, grid, random, coo (non-canonical input), dups,
explicit_zeros, empty (empty rows / columns or a zero extent), long_rows (rows past the
summation chunk L = 4096), cancel (rows whose sum cancels), int, bool, small (n <= 2000: exact
references exist), large (for timing; not built by the quick tests).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib

import numpy as np

SEED = 20261120


@dataclasses.dataclass(frozen=True)
class Matrix:
    id: str
    shape: tuple[int, int]
    row: np.ndarray  # int64
    col: np.ndarray  # int64
    data: np.ndarray
    tags: frozenset
    note: str = ""
    cond: float | None = None  # an estimate of the 1-norm condition number, when known (square)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self):
        return self.data.dtype

    def canonical(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, data) in row-major order without duplicates: duplicates summed left to
        right in input order (stable sort, then a sequential sum per position)."""
        return _canonical(self.shape, self.row, self.col, self.data, by_row=True)

    def csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indptr, indices, data) of the canonical CSR form; int64 indices."""
        r, c, v = _canonical(self.shape, self.row, self.col, self.data, by_row=True)
        indptr = np.zeros(self.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.bincount(r, minlength=self.shape[0]), out=indptr[1:])
        return indptr, c, v

    def csc(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indptr, indices, data) of the canonical CSC form; int64 indices."""
        r, c, v = _canonical(self.shape, self.row, self.col, self.data, by_row=False)
        indptr = np.zeros(self.shape[1] + 1, dtype=np.int64)
        np.cumsum(np.bincount(c, minlength=self.shape[1]), out=indptr[1:])
        return indptr, r, v

    def dense(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.data.dtype)
        r, c, v = self.canonical()
        out[r, c] = v
        return out

    def scipy(self, fmt="csr"):
        """The canonical matrix as a scipy.sparse array of format fmt (int64 index arrays)."""
        import scipy.sparse as sp

        if fmt == "coo":
            return sp.coo_array((self.data, (self.row, self.col)), shape=self.shape)
        if fmt == "csc":
            p, i, v = self.csc()
            return sp.csc_array((v, i, p), shape=self.shape)
        p, i, v = self.csr()
        return sp.csr_array((v, i, p), shape=self.shape)


def _canonical(shape, row, col, data, by_row):
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    major, minor = (row, col) if by_row else (col, row)
    order = np.lexsort((minor, major))  # stable: equal positions keep their input order
    r, c, v = row[order], col[order], data[order]
    if r.size == 0:
        return r, c, v
    key = (r * shape[1] + c) if by_row else (c * shape[0] + r)
    start = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
    if start.size == r.size:
        return r, c, v
    sums = v[start].copy()
    ends = np.concatenate((start[1:], [r.size]))
    for g in np.flatnonzero(ends - start > 1):  # sequential, left to right
        acc = v[start[g]]
        for k in range(start[g] + 1, ends[g]):
            acc = acc + v[k]
        sums[g] = acc
    return r[start], c[start], sums


def digest(a: Matrix) -> str:
    """SHA-256 of the matrix as given (shape, dtype, coordinates and values in input order): the
    recorded data and references name the matrix they were made from."""
    h = hashlib.sha256()
    h.update(repr((a.shape, str(a.data.dtype))).encode())
    for arr in (a.row, a.col, a.data):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def rhs(a: Matrix, k: int | None = None, *, seed: int = 0) -> np.ndarray:
    """A fixed right-hand side of length shape[0]: uniform in [-1, 1) (complex for a complex
    matrix), (n,) or (n, k)."""
    g = np.random.Generator(np.random.PCG64(SEED + 7 * seed + 1))
    shape = (a.shape[0],) if k is None else (a.shape[0], k)
    b = g.uniform(-1, 1, shape)
    if a.data.dtype == np.complex128:
        b = b + 1j * g.uniform(-1, 1, shape)
    return b


# ---------------------------------------------------------------- builders


def _rng(*key):
    return np.random.Generator(np.random.PCG64([SEED, *[hash_int(k) for k in key]]))


def hash_int(k) -> int:
    return int.from_bytes(hashlib.sha256(str(k).encode()).digest()[:8], "little")


def _coo(entries, shape):
    """Coordinates from a dict {(i, j): v} in row-major order."""
    keys = sorted(entries)
    row = np.array([k[0] for k in keys], dtype=np.int64)
    col = np.array([k[1] for k in keys], dtype=np.int64)
    data = np.array([entries[k] for k in keys])
    return row, col, data


def _from_dense_pattern(rows, cols, vals, shape, dtype=np.float64):
    """Canonical coordinates from arrays (duplicates must not occur)."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], np.asarray(vals, dtype=dtype)[order]


def _kron_lap(g, dim, neumann=False):
    """The (g^dim) x (g^dim) Laplacian of the 2*dim+1-point stencil: 2*dim on the diagonal, -1 to
    each grid neighbour (Dirichlet); Neumann subtracts the missing neighbours from the diagonal
    (rows sum to zero: exactly singular, the constants are its null space)."""
    n = g**dim
    idx = np.arange(n)
    coords = np.stack(np.unravel_index(idx, (g,) * dim), axis=1)
    rows, cols, vals = [idx], [idx], [np.zeros(n)]
    deg = np.zeros(n)
    stride = [g ** (dim - 1 - d) for d in range(dim)]
    for d in range(dim):
        for s in (-1, 1):
            ok = (coords[:, d] + s >= 0) & (coords[:, d] + s < g)
            rows.append(idx[ok])
            cols.append(idx[ok] + s * stride[d])
            vals.append(-np.ones(ok.sum()))
            deg += ok
    vals[0] = deg if neumann else np.full(n, 2.0 * dim)
    return _from_dense_pattern(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (n, n))


def _convdiff2d(g):
    """5-point convection-diffusion on a g x g grid, diagonally dominant: diagonal 4.2, west and
    east -0.6 and -1.3, south and north -0.8 and -1.1 (off-diagonal sum 3.8 < 4.2, so the
    condition number stays bounded; a non-dominant variant reaches 1e20 at g = 300)."""
    n = g * g
    idx = np.arange(n)
    x, y = idx % g, idx // g
    rows, cols, vals = [idx], [idx], [np.full(n, 4.2)]
    for ok, off, v in ((x > 0, -1, -0.6), (x < g - 1, 1, -1.3), (y > 0, -g, -0.8), (y < g - 1, g, -1.1)):
        rows.append(idx[ok])
        cols.append(idx[ok] + off)
        vals.append(np.full(ok.sum(), v))
    return _from_dense_pattern(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (n, n))


def _random_pattern(n, m, per_row, rng, window=None):
    """per_row distinct columns a row (all of them when per_row >= m), uniform, or within
    +-window of the diagonal."""
    rows, cols = [], []
    for i in range(n):
        if window is None:
            lo, hi = 0, m
        else:
            lo, hi = max(0, i - window), min(m, i + window + 1)
        k = min(per_row, hi - lo)
        c = np.sort(rng.choice(np.arange(lo, hi), size=k, replace=False))
        rows.append(np.full(k, i))
        cols.append(c)
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64)


def _random_unsym(n, per_row, key, dominant=True, complex_=False, window=None):
    rng = _rng("unsym", n, per_row, key)
    r, c = _random_pattern(n, n, per_row, rng, window)
    v = rng.uniform(-1, 1, r.size)
    if complex_:
        v = v + 1j * rng.uniform(-1, 1, r.size)
    entries = {}
    for i, j, x in zip(r.tolist(), c.tolist(), v.tolist()):
        entries[(i, j)] = x
    if dominant:
        rowsum = np.zeros(n)
        for (i, j), x in entries.items():
            if i != j:
                rowsum[i] += abs(x)
        for i in range(n):
            entries[(i, i)] = (rowsum[i] + 1.0) * (1 if rng.uniform() < 0.5 else -1)
    return _coo(entries, (n, n))


def _random_symmetric(n, per_row, key, kind):
    """kind 'spd' (diagonally dominant, positive diagonal), 'indefinite' (dominant, diagonal of
    both signs: nonsingular, inertia known), 'hermitian' (complex Hermitian positive definite)."""
    rng = _rng("sym", n, per_row, key, kind)
    r, c = _random_pattern(n, n, max(1, per_row // 2), rng)
    entries = {}
    for i, j in zip(r.tolist(), c.tolist()):
        if i == j:
            continue
        x = rng.uniform(-1, 1)
        if kind == "hermitian":
            x = x + 1j * rng.uniform(-1, 1)
        entries[(i, j)] = x
        entries[(j, i)] = np.conj(x) if kind == "hermitian" else x
    rowsum = np.zeros(n)
    for (i, _), x in entries.items():
        rowsum[i] += abs(x)
    for i in range(n):
        d = rowsum[i] + 0.5 + rng.uniform()
        if kind == "indefinite" and i % 3 == 1:
            d = -d
        entries[(i, i)] = complex(d) if kind == "hermitian" else d
    return _coo(entries, (n, n))


def _banded(n, lower, upper, key):
    rng = _rng("banded", n, lower, upper, key)
    entries = {}
    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            entries[(i, j)] = rng.uniform(-1, 1)
        entries[(i, i)] = float(lower + upper + 1) + rng.uniform()
    return _coo(entries, (n, n))


def _blockdiag(nblocks, bs, key):
    """Dense random blocks on the diagonal (pivoting needed inside a block)."""
    rng = _rng("blockdiag", nblocks, bs, key)
    entries = {}
    for b in range(nblocks):
        blk = rng.uniform(-1, 1, (bs, bs))
        blk[np.arange(bs), np.arange(bs)] *= 0.01  # small diagonal: row exchanges inside the block
        for i in range(bs):
            for j in range(bs):
                entries[(b * bs + i, b * bs + j)] = blk[i, j]
    n = nblocks * bs
    return _coo(entries, (n, n))


def _saddle(m, p, key):
    """[[A, B^T], [B, 0]]: A the 1-D Dirichlet Laplacian (m x m), B (p x m) random with full row
    rank; symmetric indefinite, nonsingular, a zero diagonal block."""
    rng = _rng("saddle", m, p, key)
    entries = {}
    for i in range(m):
        entries[(i, i)] = 2.0
        if i > 0:
            entries[(i, i - 1)] = -1.0
            entries[(i - 1, i)] = -1.0
    for k in range(p):
        cols = rng.choice(m, size=3, replace=False)
        cols = np.append(cols, (k * m) // p)  # a distinct column per row: full row rank
        for j in set(cols.tolist()):
            v = rng.uniform(0.5, 1.5)
            entries[(m + k, j)] = v
            entries[(j, m + k)] = v
    n = m + p
    return _coo(entries, (n, n))


def _graded_tridiag(n, decades):
    """D T D with T the 1-D Dirichlet Laplacian and D = diag(10^(-decades * i / (n - 1) / 2)):
    symmetric positive definite, condition number about 10^decades times T's."""
    d = 10.0 ** (-decades * np.arange(n) / (n - 1) / 2)
    entries = {}
    for i in range(n):
        entries[(i, i)] = 2.0 * d[i] * d[i]
        if i > 0:
            entries[(i, i - 1)] = -d[i] * d[i - 1]
            entries[(i - 1, i)] = -d[i] * d[i - 1]
    return _coo(entries, (n, n))


def _rect(m, n, per_row, key, rank_deficient=False):
    """m x n random with per_row entries a row plus a dominant 'diagonal' i -> i % n (full column
    rank when m >= n); rank_deficient makes the last column a copy of the first."""
    rng = _rng("rect", m, n, per_row, key, rank_deficient)
    r, c = _random_pattern(m, n, per_row, rng)
    entries = {(i, j): rng.uniform(-1, 1) for i, j in zip(r.tolist(), c.tolist())}
    for i in range(min(m, n)):
        entries[(i, i)] = 4.0 + rng.uniform()
    if rank_deficient:
        for i, j in list(entries):
            if j == n - 1:
                del entries[(i, j)]
        for (i, j), v in list(entries.items()):
            if j == 0:
                entries[(i, n - 1)] = v
    return _coo(entries, (m, n))


def _long_rows(n, long_rows, key):
    """n x n with 5 entries a row and a few rows that hold every column (rows longer than the
    summation chunk L = 4096 when n > 4096: the chunked pairwise rule of docs/design/sparse.md §2.6)."""
    rng = _rng("long", n, key)
    r, c = _random_pattern(n, n, 5, rng)
    entries = {(i, j): rng.uniform(-1, 1) for i, j in zip(r.tolist(), c.tolist())}
    for i in long_rows:
        for j in range(n):
            entries[(i, j)] = rng.uniform(-1, 1)
    for i in range(n):
        entries[(i, i)] = 10.0 + rng.uniform()
    return _coo(entries, (n, n))


def _cancel(n, key):
    """Rows whose products with x = ones cancel: +-2^k pairs and a small remainder, so a sum in
    the wrong order loses every digit (Arb gives the exact sums)."""
    rng = _rng("cancel", n, key)
    entries = {}
    for i in range(n):
        cols = np.sort(rng.choice(n, size=min(n, 8), replace=False))
        big = [2.0 ** int(rng.integers(20, 60)) for _ in range(len(cols) // 2)]
        vals = []
        for b in big:
            vals += [b, -b * (1 + 2.0**-30)]
        vals = vals[: len(cols) - 1] + [rng.uniform(-1, 1)]
        for j, v in zip(cols.tolist(), vals):
            entries[(i, j)] = v
    return _coo(entries, (n, n))


def _coo_inputs(key):
    """Non-canonical COO inputs: unsorted, with duplicates and explicit zeros. The sum
    [1, 1e16, 1, -1e16] at one position is 0 left to right (2 in exact arithmetic)."""
    rng = _rng("coo", key)
    if key == "cancel":
        row = np.array([0, 1, 0, 0, 2, 0], dtype=np.int64)
        col = np.array([1, 0, 1, 1, 2, 1], dtype=np.int64)
        data = np.array([1.0, 3.0, 1e16, 1.0, 0.0, -1e16])
        return (3, 3), row, col, data
    if key == "dups":
        m, n, nnz = 40, 30, 400  # about a third of the positions repeat
        row = rng.integers(0, m, nnz)
        col = rng.integers(0, n, nnz)
        data = rng.uniform(-1, 1, nnz)
        data[rng.choice(nnz, 20, replace=False)] = 0.0  # explicit zeros
        return (m, n), row.astype(np.int64), col.astype(np.int64), data
    if key == "unsorted":
        m, n, nnz = 50, 50, 300
        flat = rng.choice(m * n, size=nnz, replace=False)
        rng.shuffle(flat)
        return (m, n), (flat // n).astype(np.int64), (flat % n).astype(np.int64), rng.uniform(-1, 1, nnz)
    if key == "int":
        m, n, nnz = 20, 25, 120
        return (m, n), rng.integers(0, m, nnz), rng.integers(0, n, nnz), rng.integers(-5, 6, nnz).astype(np.int64)
    if key == "bool":
        m, n, nnz = 20, 25, 80
        flat = rng.choice(m * n, size=nnz, replace=False)
        return (m, n), (flat // n).astype(np.int64), (flat % n).astype(np.int64), np.ones(nnz, dtype=bool)
    if key == "complex":
        m, n, nnz = 30, 30, 200
        row = rng.integers(0, m, nnz)
        col = rng.integers(0, n, nnz)
        return (
            (m, n),
            row.astype(np.int64),
            col.astype(np.int64),
            rng.uniform(-1, 1, nnz) + 1j * rng.uniform(-1, 1, nnz),
        )
    if key == "nan_inf":
        row = np.array([0, 1, 2, 2, 3], dtype=np.int64)
        col = np.array([0, 2, 1, 3, 3], dtype=np.int64)
        return (4, 4), row, col, np.array([np.nan, np.inf, -np.inf, 0.0, -0.0])
    raise KeyError(key)


# ---------------------------------------------------------------- the collection


def _mk(id_, shape, rcd, tags, note="", cond=None):
    r, c, d = rcd
    return Matrix(id_, tuple(int(s) for s in shape), np.asarray(r, np.int64), np.asarray(c, np.int64), np.asarray(d),
                  frozenset(tags), note, cond)  # fmt: skip


def _small(n):
    return {"small"} if n <= 2000 else ({"large"} if n > 200_000 else set())


def _build(id_: str) -> Matrix:
    fam, _, arg = id_.partition("/")
    if fam in ("lap1d", "lap2d", "lap3d"):
        dim = int(fam[3])
        neumann = arg.endswith("n")
        g = int(arg.rstrip("n"))
        n = g**dim
        tags = {"symmetric", "grid", "banded" if dim == 1 else "grid"} | _small(n)
        tags |= {"singular"} if neumann else {"spd"}
        return _mk(id_, (n, n), _kron_lap(g, dim, neumann), tags,
                   f"{dim}-D Laplacian, {'Neumann' if neumann else 'Dirichlet'}, g = {g}")  # fmt: skip
    if fam == "convdiff2d":
        g = int(arg)
        return _mk(id_, (g * g, g * g), _convdiff2d(g), {"unsym", "grid"} | _small(g * g), "convection-diffusion")
    if fam == "random_unsym":
        n, per = (int(s) for s in arg.split("x"))
        window = None if n <= 4000 else 100
        return _mk(id_, (n, n), _random_unsym(n, per, "a", window=window), {"unsym", "random"} | _small(n))
    if fam == "random_spd":
        n = int(arg)
        return _mk(id_, (n, n), _random_symmetric(n, 8, "a", "spd"), {"spd", "symmetric", "random"} | _small(n))
    if fam == "random_indef":
        n = int(arg)
        return _mk(id_, (n, n), _random_symmetric(n, 8, "a", "indefinite"),
                   {"symmetric", "indefinite", "random"} | _small(n))  # fmt: skip
    if fam == "random_herm":
        n = int(arg)
        return _mk(id_, (n, n), _random_symmetric(n, 8, "a", "hermitian"),
                   {"hermitian", "spd", "complex", "random"} | _small(n))  # fmt: skip
    if fam == "random_complex":
        n = int(arg)
        return _mk(id_, (n, n), _random_unsym(n, 6, "c", complex_=True), {"unsym", "complex", "random"} | _small(n))
    if fam == "banded":
        n, lo, up = (int(s) for s in arg.split("x"))
        return _mk(id_, (n, n), _banded(n, lo, up, "a"), {"unsym", "banded"} | _small(n))
    if fam == "blockdiag":
        nb, bs = (int(s) for s in arg.split("x"))
        return _mk(id_, (nb * bs, nb * bs), _blockdiag(nb, bs, "a"), {"unsym", "random"} | _small(nb * bs))
    if fam == "saddle":
        m, p = (int(s) for s in arg.split("x"))
        return _mk(id_, (m + p, m + p), _saddle(m, p, "a"), {"symmetric", "indefinite"} | _small(m + p),
                   "saddle point [[A, B^T], [B, 0]]")  # fmt: skip
    if fam == "graded":
        n, dec = (int(s) for s in arg.split("x"))
        return _mk(id_, (n, n), _graded_tridiag(n, dec), {"spd", "symmetric", "ill", "banded"} | _small(n),
                   f"D T D, condition about 1e{dec} times T's")  # fmt: skip
    if fam == "near_singular":
        g = int(arg)
        r, c, d = _kron_lap(g, 2, neumann=True)
        d = d + np.where(r == c, 1e-8, 0.0)
        return _mk(id_, (g * g, g * g), (r, c, d), {"symmetric", "near_singular", "ill"} | _small(g * g),
                   "Neumann Laplacian + 1e-8 I")  # fmt: skip
    if fam == "struct_singular":
        n = int(arg)
        r, c, d = _random_unsym(n, 5, "s")
        keep = c != n // 2  # an empty column
        return _mk(id_, (n, n), (r[keep], c[keep], d[keep]), {"unsym", "struct_singular", "empty"} | _small(n))
    if fam == "dupcol":
        n = int(arg)
        r, c, d = _random_unsym(n, 5, "d")
        keep = c != n - 1
        r, c, d = r[keep], c[keep], d[keep]
        first = c == 0
        r2, c2, d2 = (
            np.concatenate((r, r[first])),
            np.concatenate((c, np.full(first.sum(), n - 1))),
            np.concatenate((d, d[first])),
        )
        return _mk(id_, (n, n), _from_dense_pattern(r2, c2, d2, (n, n)), {"unsym", "singular"} | _small(n),
                   "the last column a copy of the first")  # fmt: skip
    if fam == "rect":
        m, n = (int(s) for s in arg.rstrip("d").split("x"))
        deficient = arg.endswith("d")
        tags = {"rect"} | _small(max(m, n)) | ({"rank_deficient"} if deficient else set())
        return _mk(id_, (m, n), _rect(m, n, 4, "a", deficient), tags)
    if fam == "long_rows":
        n = int(arg)
        return _mk(id_, (n, n), _long_rows(n, (0, n // 2, n - 1), "a"), {"unsym", "long_rows"} | _small(n))
    if fam == "cancel":
        n = int(arg)
        return _mk(id_, (n, n), _cancel(n, "a"), {"cancel"} | _small(n))
    if fam == "empty":
        if arg == "rows":  # empty rows and columns inside
            r, c, d = _random_unsym(30, 3, "e", dominant=False)
            keep = (r % 7 != 3) & (c % 5 != 2)
            return _mk(id_, (30, 30), (r[keep], c[keep], d[keep]), {"empty", "small"})
        m, n = (int(s) for s in arg.split("x"))  # a zero extent, or no stored entry
        z = np.zeros(0, np.int64)
        return _mk(id_, (m, n), (z, z, np.zeros(0)), {"empty", "small"})
    if fam == "coo":
        shape, r, c, d = _coo_inputs(arg)
        tags = {"coo", "small"}
        tags |= {"dups"} if arg in ("cancel", "dups", "int", "complex") else set()
        tags |= {"explicit_zeros"} if arg in ("cancel", "dups", "nan_inf") else set()
        tags |= {"int"} if arg == "int" else set()
        tags |= {"bool"} if arg == "bool" else set()
        tags |= {"complex"} if arg == "complex" else set()
        return _mk(id_, shape, (r, c, d), tags)
    raise KeyError(id_)


IDS = (
    # grids: Dirichlet (SPD) and Neumann (exactly singular)
    "lap1d/10", "lap1d/1000", "lap1d/10n",
    "lap2d/4", "lap2d/10", "lap2d/30", "lap2d/100", "lap2d/300", "lap2d/700", "lap2d/1000",
    "lap2d/10n", "lap2d/60n",
    "lap3d/3", "lap3d/6", "lap3d/10", "lap3d/22", "lap3d/45",
    "convdiff2d/10", "convdiff2d/40", "convdiff2d/300",
    # random
    "random_unsym/50x5", "random_unsym/1000x8", "random_unsym/100000x10",
    "random_spd/60", "random_spd/1500",
    "random_indef/60", "random_indef/1500",
    "random_herm/40", "random_herm/800",
    "random_complex/40", "random_complex/800",
    # structure
    "banded/200x2x3", "banded/2000x5x5",
    "blockdiag/20x8", "blockdiag/200x8",
    "saddle/40x10", "saddle/600x150",
    # hard cases
    "graded/100x8", "graded/400x12",
    "near_singular/20",
    "struct_singular/40", "struct_singular/1000",
    "dupcol/40", "dupcol/1000",
    "rect/60x20", "rect/1200x300", "rect/20x60", "rect/80x30d",
    "long_rows/300", "long_rows/9000",
    "cancel/50", "cancel/400",
    "empty/rows", "empty/0x5", "empty/5x0", "empty/0x0", "empty/4x6",
    # non-canonical COO inputs and other dtypes
    "coo/cancel", "coo/dups", "coo/unsorted", "coo/int", "coo/bool", "coo/complex", "coo/nan_inf",
)  # fmt: skip


# Timing sizes (n >= 9e4): ids() leaves them out unless asked, so the tests never build them.
HEAVY = frozenset({"lap2d/300", "lap2d/700", "lap2d/1000", "lap3d/45", "convdiff2d/300", "random_unsym/100000x10"})


@functools.cache
def get(id_: str) -> Matrix:
    if id_ not in IDS:
        raise KeyError(f"unknown test matrix {id_!r}")
    return _build(id_)


def ids(tag: str | None = None, *, small: bool | None = None, exclude=(), heavy: bool = False) -> list[str]:
    """The ids carrying `tag` (all when None), optionally only the small (n <= 2000) or only the
    non-small ones, without those carrying a tag in `exclude`; the HEAVY ones only with
    heavy=True. Builds the matrices it filters."""
    out = []
    for i in IDS:
        if i in HEAVY and not heavy:
            continue
        t = get(i).tags
        if tag is not None and tag not in t:
            continue
        if small is not None and ("small" in t) != small:
            continue
        if t & set(exclude):
            continue
        out.append(i)
    return out
