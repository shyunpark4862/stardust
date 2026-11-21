"""Golden cases for sd.sparse (tools/golden.py has the contract).

The formats and conversions (milestone SP1, docs/design/sparse.md §8): every constructor input
(dense, a shape, coordinates, compressed parts, another format, a scipy array), the conversions
between the four formats and to dense, the transposes, reshape and resize, nonzero and
count_nonzero, eliminate_zeros and sum_duplicates, astype, conj, real and imag, the DOK element
access, and the wrong calls (their errors are results). The matrices come from fixed literals and
L.rng(seed): small ones with repeats, stored zeros, NaN and -0.0, and random ones of 2*10^5
entries with repeats, which take the multithreaded bucket path of the conversions
(csrc/sparse/canonical.cpp) at 3 and 10 threads and the direct path at 1 for the smaller sizes:
"big" (2000 x 1500, about 7% of the positions stored) and "long" (30 x 10^5, rows of about 6,700
entries, past the summation piece L = 4096). Their shapes keep a dense result small (3*10^6
values at most), so a worker stays under 1 GB.

Indexing and building (milestone SP3): every kind of key (ints, slices with steps, integer arrays
with repeats, masks, None, ...) read from the same matrices in every format, assignments of
scalars, dense and sparse values (on copies), diagonal / trace / setdiag, the builders and their
wrong calls; the big and long matrices take the threaded selections at 3 and 10 threads.

The operations (milestone SP2): every operator between two sparse arrays, with a scalar and with a
dense array (and the broadcast rows and columns), the element-wise functions, multiply / maximum /
minimum / power, the products (SpMV, SpMM, dense @ sparse, SpGEMM, the 1-D forms) and the
reductions with axis, keepdims, dtype and explicit, on the same matrices in every format; the big
and long matrices take the threaded line ranges at 3 and 10 threads (the scatter runs on one
thread on every count) and the long rows' pieces of L.

The direct solves and LinearOperator (milestone SP4): spsolve with dense (1-D, 2-D) and sparse
right-hand sides, splu's solve with trans N, T and H, factorized and inv on a 2-D Laplacian, a
random nonsymmetric and a complex matrix in every format (Accelerate's LU/AMD: its bits depend on
Accelerate's thread mode, never on sd's thread count), spsolve_triangular (lower, upper, unit, 64
columns: the column blocks run on the pool at 3 and 10 threads), spbandwidth and is_sptriangular,
LinearOperator's products and combinations, the singular and wrong calls, and sd.linalg's
refusals of a sparse array.

The iterative solvers (milestone SP5): cg, bicgstab, gmres, minres, lsqr and lsmr on a 2-D
Laplacian, a random nonsymmetric, a complex Hermitian, a complex nonsymmetric and a rectangular
matrix in every format, dense and as LinearOperators (aslinearoperator, and Python functions of
the same products), with x0, a Jacobi preconditioner, the caps, restart, shift and damp; long
banded systems (n = 4*10^4) where the pool splits the products, the inner products, the norms and
the updates at 3 and 10 threads; the result objects whole (every field, summary()); the wrong
calls.

The eigensolvers (milestone SP6): eigsh, eigs and svds on a 2-D Laplacian, a random symmetric, a
random nonsymmetric, a complex Hermitian, a complex nonsymmetric and a rectangular matrix in every
format, dense and as LinearOperators, every `which`, the spectral transformations (M, sigma,
buckling, Cayley, OPpart), return_eigenvectors / return_singular_vectors, v0 and rng, the
iteration limit (partial results), the result objects whole; banded matrices of n = 4*10^4 where
the pool splits the products, the inner products and the basis updates at 3 and 10 threads; the
wrong calls.

The matrix functions and norms (milestone SP7): expm on a 2-D Laplacian, a random nonsymmetric, a
complex, a triangular and a diagonal matrix in every format (the result in A's format); expm_multiply
with vector and matrix B, every path of the time interval, dense and operator A, traceA; onenormest
with t, max_iter and the vectors, on matrices, dense arrays and operators; norm with every ord and
axis on 2-D and 1-D arrays of every dtype; banded matrices of n = 4*10^4 where the pool splits the
products and the vector operations at 3 and 10 threads; the wrong calls.

A sparse array is stored whole: its class, shape, dtype and arrays (COO: coords, data and
has_canonical_format; CSR and CSC: indptr, indices and data; DOK: its entries in row-major order).
"""

import math

import numpy as np

NAN = math.nan


def _arr(enc, a):
    return enc(a)


def encode(x, L, enc):
    S = L.sd.sparse
    if isinstance(x, S.sparray):
        out = {"sparse": type(x).__name__, "shape": list(x.shape), "dtype": str(x.dtype), "nnz": x.nnz}
        if x.format == "coo":
            out["coords"] = [enc(c) for c in x.coords]
            out["data"] = enc(x.data)
            out["canonical"] = x.has_canonical_format
        elif x.format in ("csr", "csc"):
            out.update(indptr=enc(x.indptr), indices=enc(x.indices), data=enc(x.data))
        else:
            c = x.tocoo()
            out["entries"] = [[enc(i) for i in c.coords], enc(c.data)]
        return out
    sp = __import__("sys").modules.get("scipy.sparse")
    if sp is not None and sp.issparse(x):
        out = {"scipy": type(x).__name__, "shape": list(x.shape), "dtype": x.dtype.name}
        if x.format in ("csr", "csc"):
            out.update(indptr=enc(x.indptr), indices=enc(x.indices), data=enc(x.data))
        else:
            c = x.tocoo()
            out.update(coords=[enc(i) for i in c.coords], data=enc(c.data))
        return out
    return None


def data(L):
    r = L.rng(601)
    n = 2 * 10**5
    d = {
        "dense": np.array([[1.0, 0, 2, 0], [0, 0, 3, -0.0], [NAN, 0, 0, 4], [0, 0, 0, 0]]),
        "dense1": np.array([0.0, 2, 0, -1, NAN, 0]),
        "cplx": np.array([[1 + 2j, 0], [0, -3j], [2, 0]]),
        # repeats summed in input order: [1, 1e16, 1, -1e16] is 0; a stored zero; unsorted
        "coo_v": np.array([1.0, 1e16, 2.0, 1.0, -1e16, 0.0, 5.0, -0.0, 3.5]),
        "coo_r": np.array([0, 0, 2, 0, 0, 1, 2, 3, 1]),
        "coo_c": np.array([1, 1, 0, 1, 1, 2, 3, 0, 0]),
        "ints_v": np.array([2**62, 2**62, 7, -3, 0]),
        "bools_v": np.array([True, True, False, True, False]),
        "small_r": np.array([1, 1, 0, 2, 0]),
        "small_c": np.array([0, 0, 2, 1, 2]),
        "big_r": r.integers(0, 2000, n),
        "big_c": r.integers(0, 1500, n),
        "big_v": r.standard_normal(n) * 10.0 ** r.integers(-12, 13, n),
        "long_r": r.integers(0, 30, n),
        "long_c": r.integers(0, 10**5, n),
        "long_v": r.standard_normal(n),
        "parts_v": np.array([1.0, 2, 3, 4, 5, 6]),
        "parts_i": np.array([2, 0, 2, 1, 0, 3]),
        "parts_p": np.array([0, 3, 3, 5, 6]),
    }
    return d


def cases(L):
    sd, S = L.sd, L.sd.sparse
    d = data(L)
    a = L.arr

    def coo_of(v, r, c, shape=None):
        return S.coo_array((a(v), (a(r), a(c))), shape=shape)

    mats = {
        "dense": lambda: S.coo_array(a(d["dense"])),
        "repeats": lambda: coo_of(d["coo_v"], d["coo_r"], d["coo_c"], (4, 4)),
        "ints": lambda: coo_of(d["ints_v"], d["small_r"], d["small_c"]),
        "bools": lambda: coo_of(d["bools_v"], d["small_r"], d["small_c"]),
        "complex": lambda: S.coo_array(a(d["cplx"])),
        "big": lambda: coo_of(d["big_v"], d["big_r"], d["big_c"], (2000, 1500)),
        "long": lambda: coo_of(d["long_v"], d["long_r"], d["long_c"], (30, 10**5)),
        "1d": lambda: S.coo_array(a(d["dense1"])),
        "empty": lambda: S.coo_array((3, 0)),
    }
    fmts = ("coo", "csr", "csc", "dok")
    classes = {"coo": S.coo_array, "csr": S.csr_array, "csc": S.csc_array, "dok": S.dok_array}

    # ---- constructors
    for f, cls in classes.items():
        yield f"{f} from dense", lambda cls=cls: cls(a(d["dense"]))
        yield f"{f} from dense dtype=int64", lambda cls=cls: cls(a(d["dense"]), dtype=sd.int64)
        yield f"{f} from list", lambda cls=cls: cls([[0, 1, 0], [2, 0, 3]])
        yield f"{f} from shape", lambda cls=cls: cls((3, 5), dtype="complex128")
        yield f"{f} from coords", lambda cls=cls: cls((a(d["coo_v"]), (a(d["coo_r"]), a(d["coo_c"]))))
        yield f"{f} from scipy", lambda cls=cls: cls(__import__("scipy.sparse", fromlist=["x"]).coo_array(
            (d["coo_v"], (d["coo_r"], d["coo_c"])), shape=(4, 4)))  # fmt: skip
        for g in fmts:
            yield f"{f} from {g}", lambda cls=cls, g=g: cls(mats["repeats"]().asformat(g))
        if f != "csc":
            yield f"{f} from dense 1-D", lambda cls=cls: cls(a(d["dense1"]))
    for f in ("csr", "csc"):
        cls = classes[f]
        yield f"{f} from parts", lambda cls=cls: cls((a(d["parts_v"]), a(d["parts_i"]), a(d["parts_p"])))
        shape = (4, 5) if cls is S.csr_array else (5, 4)
        yield (
            f"{f} from parts shape",
            lambda cls=cls, shape=shape: cls((a(d["parts_v"]), a(d["parts_i"]), a(d["parts_p"])), shape=shape),
        )
    yield (
        "csr from parts 1-D",
        lambda: S.csr_array((a(d["parts_v"][:3]), a(d["parts_i"][:3]), a(np.array([0, 3]))), shape=(4,)),
    )
    yield "coo from coords 1-D", lambda: S.coo_array((a(d["parts_v"][:3]), (a(d["parts_i"][:3]),)))

    # ---- conversions, dense, the pattern methods
    for name, mk in mats.items():
        for f in fmts:
            if f == "csc" and name == "1d":
                continue
            yield f"{name} as {f}", lambda mk=mk, f=f: mk().asformat(f)
            yield f"{name} {f} toarray", lambda mk=mk, f=f: mk().asformat(f).toarray()
            yield f"{name} {f} nonzero", lambda mk=mk, f=f: mk().asformat(f).nonzero()
            yield f"{name} {f} count_nonzero", lambda mk=mk, f=f: mk().asformat(f).count_nonzero()
            if name != "1d":
                for ax in (0, 1):
                    yield (
                        f"{name} {f} count_nonzero axis={ax}",
                        lambda mk=mk, f=f, ax=ax: mk().asformat(f).count_nonzero(axis=ax),
                    )
                yield f"{name} {f} T", lambda mk=mk, f=f: mk().asformat(f).T
                yield f"{name} {f} mT", lambda mk=mk, f=f: mk().asformat(f).mT
                yield f"{name} {f} matrix_transpose", lambda mk=mk, f=f: S.matrix_transpose(mk().asformat(f))
            yield f"{name} {f} transpose", lambda mk=mk, f=f: mk().asformat(f).transpose()
            yield f"{name} {f} conj", lambda mk=mk, f=f: mk().asformat(f).conj()
            yield f"{name} {f} real", lambda mk=mk, f=f: mk().asformat(f).real
            yield f"{name} {f} imag", lambda mk=mk, f=f: mk().asformat(f).imag
            yield f"{name} {f} copy", lambda mk=mk, f=f: mk().asformat(f).copy()
            yield f"{name} {f} astype complex128", lambda mk=mk, f=f: mk().asformat(f).astype(sd.complex128)
            yield f"{name} {f} astype int64", lambda mk=mk, f=f: mk().asformat(f).astype("int64")
            yield f"{name} {f} to_scipy", lambda mk=mk, f=f: mk().asformat(f).to_scipy()
            for g in fmts:
                if g != f and not (g == "csc" and name == "1d"):
                    yield f"{name} {f} to {g}", lambda mk=mk, f=f, g=g: getattr(mk().asformat(f), "to" + g)()

    def reshaped(mk, f, shape):
        return mk().asformat(f).reshape(shape)

    def resized(mk, f, shape):
        A = mk().asformat(f)
        A.resize(shape)
        return A

    def eliminated(mk, f):
        A = mk().asformat(f)
        A.eliminate_zeros()
        return A

    def summed(mk):
        A = mk()
        A.sum_duplicates()
        return A

    for name in ("dense", "repeats", "big"):
        mk = mats[name]
        size = math.prod(mk().shape)
        for f in fmts:
            yield (
                f"{name} {f} reshape",
                lambda mk=mk, f=f, size=size: reshaped(mk, f, (size // 2, -1) if size % 2 == 0 else (-1,)),
            )
            yield f"{name} {f} reshape 1-D", lambda mk=mk, f=f, size=size: reshaped(mk, f, size)
            yield (
                f"{name} {f} resize smaller",
                lambda mk=mk, f=f: resized(mk, f, tuple(max(s // 2, 1) for s in mk().shape)),
            )
            yield f"{name} {f} resize larger", lambda mk=mk, f=f: resized(mk, f, tuple(s + 3 for s in mk().shape))
            if f != "dok":
                yield f"{name} {f} eliminate_zeros", lambda mk=mk, f=f: eliminated(mk, f)
        yield f"{name} coo sum_duplicates", lambda mk=mk: summed(mk)
        yield f"{name} coo has_canonical_format", lambda mk=mk: mk().has_canonical_format

    # ---- DOK element access
    def dok_built():
        A = S.dok_array((4, 5))
        for i, j, v in ((0, 1, 2.5), (3, 4, 1), (-1, 0, 7.0), (2, 2, NAN), (0, 1, 3.0), (3, 4, 0.0), (1, 1, -0.0)):
            A[i, j] = v
        return A

    yield "dok setitem", dok_built
    yield "dok getitem stored", lambda: dok_built()[0, 1]
    yield "dok getitem missing", lambda: dok_built()[1, 3]
    yield "dok getitem negative", lambda: dok_built()[-1, -5]
    yield "dok int getitem", lambda: S.dok_array(a(np.array([[0, 4], [5, 0]])))[1, 0]
    yield "dok getitem out of range", lambda: dok_built()[4, 0]
    yield "dok setitem complex into float", lambda: dok_built().__setitem__((0, 0), 1j)
    yield "dok setitem key float", lambda: dok_built().__setitem__((0.0, 0), 1.0)

    # ---- the module functions and errors
    yield (
        "issparse sparray",
        lambda: (S.issparse(mats["dense"]()), S.issparse(a(d["dense"])), isinstance(mats["dense"](), S.sparray)),
    )
    yield (
        "error index out of bounds",
        lambda: S.coo_array((a(d["coo_v"]), (a(d["coo_r"]), a(d["coo_c"]))), shape=(3, 3)),
    )
    yield "error indptr", lambda: S.csr_array((a(d["parts_v"]), a(d["parts_i"]), a(np.array([0, 3, 2, 5, 6]))))
    yield "error copy=False dense", lambda: S.csr_array(a(d["dense"]), copy=False)
    yield "error dtype", lambda: S.csr_array(a(d["dense"]), dtype="float32")
    yield "error 1-D T", lambda: mats["1d"]().T
    yield "error reshape", lambda: mats["dense"]().reshape((3, 5))
    yield "error asformat", lambda: mats["dense"]().asformat("bsr")
    yield "error astype complex to float", lambda: mats["complex"]().astype(sd.float64)
    yield "error sparray", lambda: S.sparray()

    yield from cases_sp2(L, mats, fmts)
    yield from cases_sp3(L, mats, fmts)
    yield from cases_sp4(L, fmts)
    yield from cases_sp5(L, fmts)
    yield from cases_sp6(L, fmts)
    yield from cases_sp7(L, fmts)


def partner(A, S):
    """A second array of A's shape and format with another pattern: A's entries with the values and
    the last index reversed."""
    c = A.tocoo()
    coords = tuple(c.coords[:-1]) + (c.coords[-1][::-1].copy(),)
    return S.coo_array((c.data[::-1].copy(), coords), shape=A.shape).asformat(A.format)


def cases_sp2(L, mats, fmts):
    import operator

    S = L.sd.sparse
    r = L.rng(602)
    binary = {"+": operator.add, "-": operator.sub, "*": operator.mul, "==": operator.eq, "!=": operator.ne,
              "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}  # fmt: skip
    names = ("sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "sqrt", "expm1", "log1p",
             "deg2rad", "rad2deg", "floor", "ceil", "trunc", "round", "sign")  # fmt: skip

    def dense_like(A, shape=None, cplx=False):
        shape = A.shape if shape is None else shape
        x = r.standard_normal(shape) if shape else r.standard_normal(())
        if cplx:
            x = x + 1j * r.standard_normal(shape)
        return L.arr(np.asarray(x))

    # the big and long matrices take only what stays sparse: no dense operand or result, no comparison
    # true between two implicit zeros (their dense cases are SP1's toarray)
    for name in ("dense", "repeats", "ints", "bools", "complex", "big", "long", "1d", "empty"):
        mk = mats[name]
        large = name in ("big", "long")
        for f in fmts:
            if f == "csc" and name == "1d":
                continue
            A = mk().asformat(f)
            B = partner(A, S)
            key = f"{name} {f}"
            for sym, op in binary.items():
                if large and sym in ("==", "<=", ">="):
                    continue
                yield f"{key} {sym} {f}", lambda op=op, A=A, B=B: op(A, B)
                if f != "csr":
                    yield f"{key} {sym} csr", lambda op=op, A=A, B=B: op(A, B.tocsr())
            for fn in ("maximum", "minimum"):
                yield f"{key} {fn} sparse", lambda fn=fn, A=A, B=B: getattr(A, fn)(B)
                for c in (-1.5, 0, 2):
                    if large and (c > 0) == (fn == "maximum") and c != 0:
                        continue
                    yield f"{key} {fn} {c}", lambda fn=fn, A=A, c=c: getattr(A, fn)(c)
            if not large:
                yield f"{key} / sparse", lambda A=A, B=B: A / B
            for c in (2.5, 0, -1, float("nan"), 1j):
                yield f"{key} * {c}", lambda A=A, c=c: A * c
                yield f"{c} * {key}", lambda A=A, c=c: c * A
                yield f"{key} / {c}", lambda A=A, c=c: A / c
            every = tuple((sym, c) for c in (0, 1, -0.5, float("nan")) for sym in ("==", "!=", "<", ">="))
            compares = (("!=", 0), ("<", 0), (">=", 1), ("==", -0.5)) if large else every
            for sym, c in compares:
                yield f"{key} {sym} {c}", lambda op=binary[sym], A=A, c=c: op(A, c)
            yield f"{key} + 0", lambda A=A: A + 0
            yield f"0 - {key}", lambda A=A: 0 - A
            yield f"{key} neg", lambda A=A: -A
            yield f"{key} abs", lambda A=A: abs(A)
            yield f"{key} ** 2", lambda A=A: A**2
            yield f"{key} power 0.5", lambda A=A: A.power(0.5)
            if name in ("dense", "complex", "1d", "big"):
                for fn in names:
                    yield f"{key} {fn}", lambda fn=fn, A=A: getattr(A, fn)()
            if A.ndim == 2:
                row = dense_like(A, (A.shape[1],))
                col = dense_like(A, (A.shape[0], 1))
                yield f"{key} * row", lambda A=A, row=row: A * row
                yield f"{key} * column", lambda A=A, col=col: A * col
            if large:
                yield from _products_and_reductions(L, A, B, key, name, dense_like)
                continue
            D = dense_like(A, cplx=name == "complex")
            yield f"{key} * dense", lambda A=A, D=D: A * D
            yield f"dense * {key}", lambda A=A, D=D: D * A
            yield f"{key} multiply dense", lambda A=A, D=D: A.multiply(D)
            yield f"{key} / dense", lambda A=A, D=D: A / D
            yield f"{key} + dense", lambda A=A, D=D: A + D
            yield f"dense - {key}", lambda A=A, D=D: D - A
            yield f"{key} < dense", lambda A=A, D=D: A < D
            yield f"{key} maximum dense", lambda A=A, D=D: A.maximum(D)
            yield from _products_and_reductions(L, A, B, key, name, dense_like)

    yield from _wrong_calls(L, mats)


def _products_and_reductions(L, A, B, key, name, dense_like):
    """The products and the reductions of A (B: a partner of A's shape and format)."""
    sd, S = L.sd, L.sd.sparse
    # products
    x = dense_like(A, (A.shape[-1],), cplx=name == "complex")
    yield f"{key} @ x", lambda A=A, x=x: A @ x
    w = dense_like(A, (A.shape[0],))
    yield f"w @ {key}", lambda A=A, w=w: w @ A
    if name not in ("big", "long"):
        X = dense_like(A, (A.shape[-1], 7))
        yield f"{key} @ X", lambda A=A, X=X: A @ X
        W = dense_like(A, (5, A.shape[0]))
        yield f"W @ {key}", lambda A=A, W=W: W @ A
    if A.ndim == 2:
        yield f"{key} @ {key}.T", lambda A=A: A @ A.T
        if name != "long":  # A^T B of the long matrix has ~10^9 products
            yield f"{key}.T @ csr", lambda A=A, B=B: A.T @ B.tocsr()
        v = S.csr_array(dense_like(A, (A.shape[1],)))
        yield f"{key} @ sparse vector", lambda A=A, v=v: A @ v
    else:
        yield f"{key} @ {key}", lambda A=A: A @ A
    # reductions
    for fn in ("sum", "mean", "max", "min", "nanmax", "nanmin", "argmax", "argmin"):
        yield f"{key} {fn}", lambda fn=fn, A=A: getattr(A, fn)()
        for ax in range(A.ndim):
            yield f"{key} {fn} axis={ax}", lambda fn=fn, A=A, ax=ax: getattr(A, fn)(axis=ax)
            yield (
                f"{key} {fn} axis={ax} keepdims",
                lambda fn=fn, A=A, ax=ax: getattr(A, fn)(axis=ax, keepdims=True),
            )
        if fn not in ("sum", "mean"):
            yield f"{key} {fn} explicit", lambda fn=fn, A=A: getattr(A, fn)(explicit=True)
            yield f"{key} {fn} axis=-1 explicit", lambda fn=fn, A=A: getattr(A, fn)(axis=-1, explicit=True)
    yield f"{key} sum dtype float64", lambda A=A: A.sum(dtype=sd.float64)
    yield f"{key} sum axis=0 dtype complex128", lambda A=A: A.sum(axis=0, dtype="complex128")


def _wrong_calls(L, mats):
    sd, S = L.sd, L.sd.sparse
    A = mats["dense"]().tocsr()
    yield "error + nonzero scalar", lambda: A + 1
    yield "error - nonzero scalar", lambda: 1 - A
    yield "error ** 0", lambda: A**0
    yield "error power -1", lambda: A.power(-1)
    yield "error shapes", lambda: A + S.csr_array((2, 2))
    yield "error matmul shapes", lambda: A @ L.arr(np.ones(3))
    yield "error matmul scalar", lambda: A @ 2
    yield "error numpy operand", lambda: A * np.ones((4, 4))
    yield "error numpy ==", lambda: A == np.ones((4, 4))
    yield "error multiply list", lambda: A.multiply([1, 2])
    yield "error dense broadcast", lambda: S.csr_array(L.arr(np.ones((1, 4)))) * L.arr(np.ones((3, 4)))
    yield "error sd.sin", lambda: sd.sin(A)
    yield "error sd.add", lambda: sd.add(A, A)
    yield "error complex max", lambda: mats["complex"]().max()
    yield "error complex <", lambda: mats["complex"]() < 1
    yield "error empty max", lambda: mats["empty"]().max()
    yield "error argmax explicit", lambda: S.csr_array((2, 3)).argmax(explicit=True)
    yield "error axis", lambda: A.sum(axis=2)
    yield "error keepdims", lambda: A.max(keepdims=None)


def cases_sp3(L, mats, fmts):
    """Indexing, assignment, the diagonal and the builders (SP3)."""
    sd, S = L.sd, L.sd.sparse
    r = L.rng(603)
    a = L.arr

    def keys(shape):
        if len(shape) == 1:
            (n,) = shape
            return [("int", 1), ("neg", -1), ("slice", slice(1, None, 2)), ("rev", slice(None, None, -1)),
                    ("array", a(r.integers(-n, n, 9))), ("mask", a(r.random(n) < 0.4)),
                    ("none", (None, slice(None)))]  # fmt: skip
        m, n = shape
        out = [("all", slice(None)), ("empty list", [])]
        if m == 0 or n == 0:
            return out + [("cols", (slice(None), slice(0, 0)))]
        k = min(m, 2500)
        out += [
            ("int", 0), ("neg row", -1), ("element", (m // 2, n // 3)), ("neg element", (-1, -1)),
            ("row slice", slice(m // 4, None, 2)), ("col slice", (slice(None), slice(n // 5, n // 2))),
            ("strided", (slice(None, None, -3), slice(1, None, 2))), ("column", (slice(None), n - 1)),
            ("rows", a(r.integers(-m, m, k))), ("cols", (slice(None), a(r.integers(0, n, min(n, 3000))))),
            ("sorted cols", (slice(None), a(np.sort(r.integers(0, n, min(n, 3000)))))),
            ("row mask", a(r.random(m) < 0.3)), ("pick", (a(r.integers(0, m, 500)), a(r.integers(0, n, 500)))),
            ("outer", (a(r.integers(0, m, (7, 1))), a(r.integers(0, n, 9)))), ("int and array", (1, a([n - 1, 0, 0]))),
            ("none", (None, 0)), ("ellipsis", (Ellipsis, 0)),
        ]  # fmt: skip
        if m * n <= 10**6:
            out.append(("mask 2-D", a(r.random((m, n)) < 0.2)))
        return out

    def assigned(mk, f, key, value):
        A = mk().asformat(f).copy()
        A[key] = value
        return A

    def with_diag(mk, f, values, k):
        A = mk().asformat(f).copy()
        A.setdiag(values, k=k)
        return A

    for name in ("dense", "repeats", "ints", "bools", "complex", "big", "long", "1d", "empty"):
        mk = mats[name]
        for f in fmts:
            if f == "csc" and name == "1d":
                continue
            A = mk().asformat(f)
            key0 = f"{name} {f}"
            for kn, key in keys(A.shape):
                yield f"{key0} [{kn}]", lambda A=A, key=key: A[key]
            if A.ndim != 2 or 0 in A.shape:
                continue
            m, n = A.shape
            dt = A.dtype
            one = True if dt == sd.bool else (1 if dt == sd.int64 else 1.5)
            zero = False if dt == sd.bool else 0
            row = a(np.ones(n, dtype=np.dtype(str(dt))))
            for an, key, value in (
                ("element", (m // 2, 0), one), ("zero element", (0, n - 1), zero), ("row", (0, slice(None)), row),
                ("zero row", (m - 1, slice(None)), zero), ("pick", (a([0, 0, m - 1]), a([1 % n, 1 % n, 0])), one),
                ("block", (slice(0, 2), slice(0, 2)), S.csr_array(a(np.array([[0, 1], [0, 0]]))).astype(dt)),
            ):  # fmt: skip
                yield f"{key0} [{an}] =", lambda mk=mk, f=f, key=key, value=value: assigned(mk, f, key, value)
            for k in (0, 1, -2):
                yield f"{key0} diagonal k={k}", lambda A=A, k=k: A.diagonal(k=k)
                yield f"{key0} trace k={k}", lambda A=A, k=k: A.trace(k=k)
                yield f"{key0} setdiag k={k}", lambda mk=mk, f=f, k=k, one=one: with_diag(mk, f, one, k)

    # ---- the builders
    small = mats["dense"]().tocsr()
    ints = mats["ints"]().tocsr()
    cplx = mats["complex"]()
    big = mats["big"]()
    for fmt in (None, "coo", "csr", "csc", "dok"):
        yield f"eye_array 4 5 k=1 {fmt}", lambda fmt=fmt: S.eye_array(4, 5, k=1, format=fmt)
        yield f"kron {fmt}", lambda fmt=fmt: S.kron(small, cplx, format=fmt)
        yield f"vstack {fmt}", lambda fmt=fmt: S.vstack([small, small.tocoo()], format=fmt)
        yield f"tril {fmt}", lambda fmt=fmt: S.tril(small, k=-1, format=fmt)
    yield "eye_array int64 k=-2", lambda: S.eye_array(5, dtype="int64", k=-2)
    yield "eye_array bool", lambda: S.eye_array(3, dtype=sd.bool)
    yield (
        "diags_array",
        lambda: S.diags_array([a(np.array([1.0, 0, 2])), 5.0, a(np.array([3, 4]))], offsets=[0, 1, -2], shape=(3, 3)),
    )
    yield "diags_array int", lambda: S.diags_array(a(np.array([1, 2, 3])), offsets=1)
    yield "random_array", lambda: S.random_array((40, 30), density=0.2, rng=sd.random.default_rng(7))
    yield (
        "random_array complex csr",
        lambda: S.random_array((40, 30), density=0.1, dtype="complex128", rng=sd.random.default_rng(8), format="csr"),
    )
    yield (
        "random_array int 1-D",
        lambda: S.random_array((50,), density=0.3, dtype="int64", rng=sd.random.default_rng(9)),
    )
    yield "kron ints bools", lambda: S.kron(ints, mats["bools"]())
    yield "kron big", lambda: S.kron(S.eye_array(3), big)
    yield "kron 1-D", lambda: S.kron(mats["1d"](), mats["1d"]().tocsr())
    yield "kronsum", lambda: S.kronsum(small, ints[:3, :3])
    # SP111: placed values, complex inf and -0.0 parts kept (as sd.linalg.kronsum)
    special = np.array([[complex(0, np.inf), complex(1, -0.0)], [0, complex(-np.inf, 2)]])
    yield (
        "kronsum complex special",
        lambda: S.kronsum(S.csr_array(a(special)), S.csr_array(a(np.array([[1j, 0], [2, -0.0]])))),
    )
    yield "hstack csr", lambda: S.hstack([small, small, small[:, :1]])
    yield "hstack csc", lambda: S.hstack([small.tocsc(), small.tocsc()[:, 1:3]])
    yield "hstack big", lambda: S.hstack([big.tocsr(), big.tocsr()])
    yield "vstack big csc", lambda: S.vstack([big.tocsc(), big.tocsc()])
    yield "hstack 1-D", lambda: S.hstack([mats["1d"](), mats["1d"]().tocsr()])
    yield "block_array", lambda: S.block_array([[small, None], [None, cplx]])
    yield "block_array dense", lambda: S.block_array([[small, a(np.eye(4))]], dtype="complex128")
    yield "block_diag", lambda: S.block_diag([small, ints, cplx.todok()])
    yield "triu repeats", lambda: S.triu(mats["repeats"](), k=1)
    yield "tril big", lambda: S.tril(big)
    yield "find", lambda: S.find(mats["repeats"]())
    yield "find big", lambda: S.find(big.tocsc())

    # ---- wrong calls
    A = small
    yield "error getitem out of bounds", lambda: A[7]
    yield "error index float", lambda: A[1.5]
    yield "error index 3-D", lambda: A[None, :]
    yield "error index separated", lambda: A[[0, 1], None, [0, 1]]
    yield "error index broadcast", lambda: A[[0, 1], [0, 1, 2]]
    yield "error index mask", lambda: A[a(np.array([True, False]))]
    yield "error setitem downcast", lambda: ints.copy().__setitem__((0, 0), 1.5)
    yield "error setitem broadcast", lambda: A.copy().__setitem__((0, slice(None)), a(np.ones((1, 4))))
    yield "error setitem sequence", lambda: A.copy().__setitem__((0, 0), [1.0])
    yield "error setdiag k", lambda: A.copy().setdiag(1.0, k=9)
    yield "error diagonal 1-D", lambda: mats["1d"]().diagonal()
    yield "error random_array rng", lambda: S.random_array((3, 3))
    yield "error random_array density", lambda: S.random_array((3, 3), density=2, rng=sd.random.default_rng(0))
    yield "error diags_array offsets", lambda: S.diags_array([a(np.ones(2)), a(np.ones(2))], offsets=[1, 1])
    yield "error diags_array length", lambda: S.diags_array([a(np.ones(2))], offsets=[0], shape=(4, 4))
    yield "error kron list", lambda: S.kron(A, [[1.0]])
    yield "error kronsum square", lambda: S.kronsum(S.csr_array((2, 3)), A)
    yield "error vstack dims", lambda: S.vstack([A, S.csr_array((2, 3))])
    yield "error block_array none", lambda: S.block_array([[None]])
    yield "error tril 1-D", lambda: S.tril(mats["1d"]())
    yield "error eye_array format", lambda: S.eye_array(3, format="dia")


def cases_sp4(L, fmts):
    """The direct solves, the patterns, LinearOperator and sd.linalg's refusals (SP4)."""
    sd, S = L.sd, L.sd.sparse
    r = L.rng(604)
    a = L.arr
    g = 12
    T = S.diags_array([a(np.full(g - 1, -1.0)), a(np.full(g, 2.0)), a(np.full(g - 1, -1.0))], offsets=[-1, 0, 1])
    I1 = S.eye_array(g)
    lap = (S.kron(I1, T) + S.kron(T, I1)).tocsc()  # 144 x 144
    n = 300
    rows, cols = r.integers(0, n, 2400), r.integers(0, n, 2400)
    keep = rows != cols
    diag = np.arange(n)
    uv = a(np.concatenate([r.standard_normal(keep.sum()), 10.0 + r.random(n)]))
    ur, uc = a(np.concatenate([rows[keep], diag])), a(np.concatenate([cols[keep], diag]))
    unsym = S.coo_array((uv, (ur, uc)), shape=(n, n)).tocsr()
    m = 60
    zr, zc = r.integers(0, m, 400), r.integers(0, m, 400)
    zk = zr != zc
    zv = a(np.concatenate([r.standard_normal(zk.sum()) + 1j * r.standard_normal(zk.sum()), 8.0 + 2j + np.zeros(m)]))
    zrr, zcc = a(np.concatenate([zr[zk], np.arange(m)])), a(np.concatenate([zc[zk], np.arange(m)]))
    cplx = S.coo_array((zv, (zrr, zcc)), shape=(m, m)).tocsr()
    mats = {"lap": lap, "unsym": unsym, "complex": cplx}

    for name, A in mats.items():
        k = A.shape[0]
        b1 = a(r.standard_normal(k))
        B = a(r.standard_normal((k, 5)))
        Bs = S.csc_array(a(np.where(r.random((k, 3)) < 0.1, r.standard_normal((k, 3)), 0.0)))
        for f in fmts:
            Af = A.asformat(f)
            yield f"spsolve {name} {f} b 1-D", lambda Af=Af, b1=b1: S.spsolve(Af, b1)
            yield f"spsolve {name} {f} b 2-D", lambda Af=Af, B=B: S.spsolve(Af, B)
            yield f"spsolve {name} {f} b sparse", lambda Af=Af, Bs=Bs: S.spsolve(Af, Bs)
        for trans in ("N", "T", "H"):
            yield f"splu {name} solve {trans} 1-D", lambda A=A, b1=b1, t=trans: S.splu(A).solve(b1, trans=t)
            yield f"splu {name} solve {trans} 2-D", lambda A=A, B=B, t=trans: S.splu(A).solve(B, trans=t)
        yield f"splu {name} shape", lambda A=A: S.splu(A).shape
        yield f"factorized {name}", lambda A=A, b1=b1: S.factorized(A)(b1 * 1j)
        yield f"spsolve {name} b complex", lambda A=A, b1=b1: S.spsolve(A, b1 * (1 - 2j))
        yield f"spsolve {name} b (n, 1)", lambda A=A, b1=b1: S.spsolve(A, b1.reshape((k, 1)))
        wide = a(r.standard_normal((k, 64)))
        for lower in (True, False):
            for unit in (False, True):
                tag = f"{'lower' if lower else 'upper'}{' unit' if unit else ''}"
                for f in ("csr", "csc"):
                    yield (f"spsolve_triangular {name} {f} {tag}",
                           lambda A=A, f=f, b1=b1, lower=lower, unit=unit:
                           S.spsolve_triangular(A.asformat(f), b1, upper=not lower, unit_diagonal=unit))  # fmt: skip
                yield (f"spsolve_triangular {name} {tag} 64 columns",
                       lambda A=A, w=wide, lower=lower, unit=unit:
                       S.spsolve_triangular(A, w, upper=not lower, unit_diagonal=unit))  # fmt: skip
        yield f"spbandwidth {name}", lambda A=A: S.spbandwidth(A)
        yield f"is_sptriangular {name} tril", lambda A=A: S.is_sptriangular(S.tril(A))
    yield "inv lap csc", lambda: S.inv(lap)
    yield "inv complex csr", lambda: S.inv(cplx)
    yield "inv lap coo", lambda: S.inv(lap.tocoo())
    yield "spsolve int matrix", lambda: S.spsolve(S.csr_array(a(np.array([[2, 1], [0, 3]]))), a(np.array([1, 1])))
    yield "spsolve empty", lambda: S.spsolve(S.csr_array((0, 0)), a(np.zeros((0, 2))))
    yield "inv empty", lambda: S.inv(S.csc_array((0, 0)))
    yield "spbandwidth rect", lambda: S.spbandwidth(S.csr_array(a(np.triu(np.ones((3, 6)), 2))))
    yield "is_sptriangular empty", lambda: S.is_sptriangular(S.csr_array((3, 3)))

    # singular and wrong calls
    singular = S.csr_array(a(np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])))
    dupcol = S.csr_array(a(np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 3.0, 5.0]])))
    struct = S.csr_array(a(np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 4.0], [5.0, 0.0, 6.0]])))
    for name, A in (("exact zero pivot", singular), ("dupcol", dupcol), ("structural", struct)):
        yield f"error singular {name} spsolve", lambda A=A: S.spsolve(A, a(np.ones(3)))
        yield f"error singular {name} splu", lambda A=A: S.splu(A)
        yield f"error singular {name} inv", lambda A=A: S.inv(A)
        yield f"error singular {name} factorized", lambda A=A: S.factorized(A)
    yield "error spsolve dense A", lambda: S.spsolve(a(np.eye(2)), a(np.ones(2)))
    yield "error spsolve rect", lambda: S.spsolve(S.csr_array((2, 3)), a(np.ones(2)))
    yield "error spsolve rows", lambda: S.spsolve(lap, a(np.ones(3)))
    yield "error spsolve list b", lambda: S.spsolve(lap, [1.0])
    yield "error spsolve nan", lambda: S.spsolve(S.csr_array(a(np.array([[1.0, np.nan], [0, 1]]))), a(np.ones(2)))
    yield "error spsolve permc_spec", lambda: S.spsolve(lap, a(np.ones(144)), permc_spec="COLAMD")
    yield "error splu trans", lambda: S.splu(lap).solve(a(np.ones(144)), trans="C")
    yield "error triangular zero diagonal", lambda: S.spsolve_triangular(struct, a(np.ones(3)), upper=False)
    yield "error triangular upper", lambda: S.spsolve_triangular(lap, a(np.ones(144)), upper=0)
    yield "error spbandwidth 1-D", lambda: S.spbandwidth(S.csr_array(a(np.ones(3))))

    # LinearOperator
    X = a(r.standard_normal((n, 4)))
    x1, y1 = a(r.standard_normal(n)), a(r.standard_normal(n))
    U = unsym
    yield "aslinearoperator matvec", lambda: S.aslinearoperator(U) @ x1
    yield "aslinearoperator matmat", lambda: S.aslinearoperator(U) @ X
    yield "aslinearoperator rmatvec", lambda: S.aslinearoperator(U).rmatvec(y1)
    yield "aslinearoperator dense", lambda: S.aslinearoperator(a(np.eye(3) * 2.0)).matvec(a(np.ones(3)))
    yield (
        "LinearOperator T H",
        lambda: (S.aslinearoperator(cplx).T @ a(np.ones(m) * 1j), S.aslinearoperator(cplx).H @ a(np.ones(m))),
    )
    yield (
        "LinearOperator combination",
        lambda: (2.0 * S.aslinearoperator(U) - S.aslinearoperator(U) @ S.aslinearoperator(U)) @ x1,
    )
    yield "LinearOperator power", lambda: (S.aslinearoperator(U) ** 3) @ x1
    yield (
        "LinearOperator custom",
        lambda: S.LinearOperator((n, n), lambda v: U @ v, rmatvec=lambda v: U.T @ v).matmat(X),
    )
    yield "LinearOperator custom dtype", lambda: S.LinearOperator((m, m), lambda v: cplx @ v).dtype
    yield "LinearOperator shape", lambda: S.LinearOperator((2, 3), lambda v: a(np.ones(2))).shape
    yield "error LinearOperator no rmatvec", lambda: S.LinearOperator((n, n), lambda v: U @ v).rmatvec(x1)
    yield "error LinearOperator matvec shape", lambda: S.aslinearoperator(U).matvec(a(np.ones(3)))
    yield "error aslinearoperator list", lambda: S.aslinearoperator([1.0])

    # sd.linalg refuses a sparse array, naming the function to use
    for fn, call in (("solve", lambda A: sd.linalg.solve(A, a(np.ones(144)))), ("inv", sd.linalg.inv),
                     ("lu", sd.linalg.lu), ("det", sd.linalg.det), ("eigh", sd.linalg.eigh),
                     ("kron", lambda A: sd.linalg.kron(A, A)),
                     ("solve_triangular",
                      lambda A: sd.linalg.solve_triangular(A, a(np.ones(144)), upper=True))):  # fmt: skip
        yield f"error sd.linalg.{fn} sparse", lambda call=call: call(lap)


def cases_sp5(L, fmts):
    """The iterative solvers (SP5): every solver on small matrices in every format and as
    operators, their options, long systems for the pool, and the wrong calls."""
    sd, S = L.sd, L.sd.sparse
    r = L.rng(605)
    a = L.arr
    g = 12
    T = S.diags_array([a(np.full(g - 1, -1.0)), a(np.full(g, 2.0)), a(np.full(g - 1, -1.0))], offsets=[-1, 0, 1])
    I1 = S.eye_array(g)
    lap = (S.kron(I1, T) + S.kron(T, I1)).tocsr()  # 144 x 144, SPD
    n = 200
    rows, cols = r.integers(0, n, 1600), r.integers(0, n, 1600)
    keep = rows != cols
    diag = np.arange(n)
    uv = a(np.concatenate([r.standard_normal(keep.sum()), 8.0 + r.random(n)]))
    unsym = S.coo_array((uv, (a(np.concatenate([rows[keep], diag])), a(np.concatenate([cols[keep], diag])))),
                        shape=(n, n)).tocsr()  # fmt: skip
    m = 50
    zr, zc = r.integers(0, m, 300), r.integers(0, m, 300)
    zk = zr != zc
    zv = r.standard_normal(zk.sum()) + 1j * r.standard_normal(zk.sum())
    Z = S.coo_array((a(zv), (a(zr[zk]), a(zc[zk]))), shape=(m, m)).tocsr()
    herm = (Z + Z.T.conj() + S.eye_array(m) * 12.0).tocsr()
    cplx = (Z + S.eye_array(m) * (9.0 + 1j)).tocsr()
    rr, rc = r.integers(0, 120, 500), r.integers(0, 40, 500)
    rect = S.coo_array((a(r.standard_normal(500)), (a(rr), a(rc))), shape=(120, 40)).tocsr()
    rect = (rect + S.eye_array(120, 40) * 3.0).tocsr()
    vec = {k: a(r.standard_normal(A.shape[0])) for k, A in (("lap", lap), ("unsym", unsym), ("rect", rect))}
    vec["herm"] = a(r.standard_normal(m) + 1j * r.standard_normal(m))
    vec["cplx"] = vec["herm"]
    cases_of = {
        "lap": ("cg", "minres", "gmres", "bicgstab", "lsqr", "lsmr"),
        "unsym": ("gmres", "bicgstab", "lsqr", "lsmr"),
        "herm": ("cg", "minres"),
        "cplx": ("gmres", "bicgstab", "lsmr"),
        "rect": ("lsqr", "lsmr"),
    }
    mats = {"lap": lap, "unsym": unsym, "herm": herm, "cplx": cplx, "rect": rect}
    for key, fns in cases_of.items():
        A, b = mats[key], vec[key]
        for fn in fns:
            f = getattr(S, fn)
            yield f"{fn} {key}", lambda f=f, A=A, b=b: f(A, b)
            for fm in fmts:
                if fm != "csr":
                    yield f"{fn} {key} {fm}", lambda f=f, A=A, b=b, fm=fm: f(A.asformat(fm), b).x
            yield f"{fn} {key} dense", lambda f=f, A=A, b=b: f(A.toarray(), b).x
            yield f"{fn} {key} aslinearoperator", lambda f=f, A=A, b=b: f(S.aslinearoperator(A), b).x
            AH = A.T.conj().tocsr() if A.dtype == sd.complex128 else A.T.tocsr()
            yield (f"{fn} {key} python operator",
                   lambda f=f, A=A, AH=AH, b=b: f(S.LinearOperator(A.shape, lambda v: A @ v,
                                                                     rmatvec=lambda v: AH @ v), b).x)  # fmt: skip
            yield f"{fn} {key} max_iter 3", lambda f=f, A=A, b=b: f(A, b, max_iter=3)
            if fn in ("lsqr", "lsmr"):
                x0 = a(np.cos(np.arange(A.shape[1])))
                yield f"{fn} {key} damp", lambda f=f, A=A, b=b: f(A, b, damp=0.5)
                yield f"{fn} {key} x0 tight", lambda f=f, A=A, b=b, x0=x0: f(A, b, x0=x0, atol=1e-12, btol=1e-12)
                yield f"{fn} {key} conlim", lambda f=f, A=A, b=b: f(A, b, conlim=2.0)
            else:
                x0 = a(np.cos(np.arange(A.shape[0])))
                D = S.diags_array(a(1.0 / np.abs(np.asarray(A.diagonal()))), format="csr")
                yield f"{fn} {key} x0 tight", lambda f=f, A=A, b=b, x0=x0: f(A, b, x0=x0, rtol=1e-12)
                yield f"{fn} {key} jacobi", lambda f=f, A=A, b=b, D=D: f(A, b, M=D, rtol=1e-10)
                if fn != "minres":  # (minres has rtol only, as scipy's)
                    yield f"{fn} {key} atol", lambda f=f, A=A, b=b: f(A, b, rtol=0.0, atol=1e-3)
            if fn == "gmres":
                yield f"gmres {key} restart 5", lambda A=A, b=b: S.gmres(A, b, restart=5, max_iter=4)
                yield f"gmres {key} restart 60", lambda A=A, b=b: S.gmres(A, b, restart=60, rtol=1e-11)
            if fn == "minres":
                yield f"minres {key} shift", lambda A=A, b=b: S.minres(A, b, shift=0.75, rtol=1e-10)
    yield "summary cg lap", lambda: S.cg(lap, vec["lap"]).summary()
    yield "repr lsqr rect", lambda: repr(S.lsqr(rect, vec["rect"]))
    yield "unpack lsmr rect", lambda: tuple(S.lsmr(rect, vec["rect"]))
    yield "unpack lsqr rect", lambda: tuple(S.lsqr(rect, vec["rect"]))
    yield (
        "bicgstab breakdown",
        lambda: S.bicgstab(S.csr_array(a(np.array([[0.0, 1.0], [1.0, 0.0]]))), a(np.array([1.0, 0.0]))),
    )
    yield "gmres exact", lambda: S.gmres(S.eye_array(5) * 3.0, a(np.arange(1.0, 6.0)))
    yield "cg b zero", lambda: S.cg(lap, a(np.zeros(144)))
    yield "lsmr b zero", lambda: S.lsmr(rect, a(np.zeros(120)))
    yield "cg empty", lambda: S.cg(S.csr_array((0, 0)), a(np.zeros(0)))
    yield "cg b (n, 1)", lambda: S.cg(lap, vec["lap"].reshape((144, 1))).x
    yield "cg int matrix", lambda: S.cg(lap.astype(sd.int64) * 1, vec["lap"]).x
    yield "gmres complex b real matrix", lambda: S.gmres(unsym, vec["unsym"] * (1 + 2j)).x

    # long systems: the pool splits every vector operation at 3 and 10 threads
    k = 40000
    band = S.diags_array([a(np.full(k, 4.0)), a(np.full(k - 1, -1.0)), a(np.full(k - 1, -1.0))], offsets=[0, 1, -1],
                         format="csr")  # fmt: skip
    skew = S.diags_array([a(np.full(k, 4.0)), a(np.full(k - 1, -1.0)), a(np.full(k - 1, 0.5))], offsets=[0, 1, -1],
                         format="csr")  # fmt: skip
    tall = S.diags_array([a(np.full(k, 4.0)), a(np.full(k, -1.0)), a(np.full(k, 0.5))], offsets=[0, 1, -1],
                         shape=(k + 5000, k), format="csr")  # fmt: skip
    t = np.arange(k + 5000, dtype=np.float64)
    bl, bt = a(np.sin(0.37 * t[:k]) + 0.1), a(np.sin(0.37 * t) + 0.1)
    for fn in ("cg", "minres"):
        yield f"{fn} long", lambda fn=fn: getattr(S, fn)(band, bl, rtol=1e-12)
    for fn in ("gmres", "bicgstab"):
        yield f"{fn} long", lambda fn=fn: getattr(S, fn)(skew, bl, rtol=1e-12)
    for fn in ("lsqr", "lsmr"):
        yield f"{fn} long", lambda fn=fn: getattr(S, fn)(tall, bt, atol=1e-12, btol=1e-12)
    yield "gmres long complex", lambda: S.gmres(skew * (1 + 0.5j), bl * (1 - 1j), rtol=1e-12)

    # wrong calls
    b = vec["lap"]
    yield "error cg list A", lambda: S.cg([[1.0]], b)
    yield "error cg numpy b", lambda: S.cg(lap, np.ones(144))
    yield "error cg rect", lambda: S.cg(rect, vec["rect"])
    yield "error cg b shape", lambda: S.cg(lap, a(np.ones(3)))
    yield "error cg M shape", lambda: S.cg(lap, b, M=S.eye_array(3))
    yield "error cg rtol", lambda: S.cg(lap, b, rtol=-1.0)
    yield "error cg max_iter 0", lambda: S.cg(lap, b, max_iter=0)
    yield "error cg maxiter", lambda: S.cg(lap, b, maxiter=5)
    yield "error cg callback", lambda: S.cg(lap, b, callback=print)
    yield "error gmres restart", lambda: S.gmres(lap, b, restart=0)
    yield "error minres shift", lambda: S.minres(lap, b, shift=float("nan"))
    yield "error minres indefinite M", lambda: S.minres(lap, b, M=S.eye_array(144) * -1.0)
    yield "error lsqr iter_lim", lambda: S.lsqr(rect, vec["rect"], iter_lim=3)
    yield "error lsqr calc_var", lambda: S.lsqr(rect, vec["rect"], calc_var=True)
    yield "error lsmr damp", lambda: S.lsmr(rect, vec["rect"], damp=-1.0)
    yield "error lsmr conlim", lambda: S.lsmr(rect, vec["rect"], conlim=float("nan"))
    yield "error cg nan", lambda: S.cg(lap, b * float("nan"))
    yield "error lsqr no rmatvec", lambda: S.lsqr(S.LinearOperator(rect.shape, lambda v: rect @ v), vec["rect"])


def cases_sp6(L, fmts):
    """The eigensolvers (SP6): eigsh, eigs, svds on small matrices in every format and as
    operators, their options, long banded matrices for the pool, and the wrong calls."""
    sd, S = L.sd, L.sd.sparse
    r = L.rng(606)
    a = L.arr
    g = 12
    T = S.diags_array([a(np.full(g - 1, -1.0)), a(np.full(g, 2.0)), a(np.full(g - 1, -1.0))], offsets=[-1, 0, 1])
    I1 = S.eye_array(g)
    lap = (S.kron(I1, T) + S.kron(T, I1)).tocsr()  # 144 x 144, repeated eigenvalues
    n = 120
    rows, cols = r.integers(0, n, 900), r.integers(0, n, 900)
    vals = r.standard_normal(900)
    R0 = S.coo_array((a(vals), (a(rows), a(cols))), shape=(n, n)).tocsr()
    sym = (R0 + R0.T + S.diags_array(a(np.linspace(-3.0, 3.0, n)), format="csr")).tocsr()
    unsym = (R0 + S.diags_array(a(np.linspace(-1.0, 4.0, n)), format="csr")).tocsr()
    m = 60
    zr, zc = r.integers(0, m, 400), r.integers(0, m, 400)
    zv = r.standard_normal(400) + 1j * r.standard_normal(400)
    Z = S.coo_array((a(zv), (a(zr), a(zc))), shape=(m, m)).tocsr()
    herm = (Z + Z.T.conj() + S.diags_array(a(np.linspace(0.0, 6.0, m)), format="csr")).tocsr()
    cplx = (Z + S.diags_array(a(np.linspace(0.0, 3.0, m) + 1j), format="csr")).tocsr()
    rr, rc = r.integers(0, 90, 500), r.integers(0, 40, 500)
    rect = (
        S.coo_array((a(r.standard_normal(500)), (a(rr), a(rc))), shape=(90, 40)) + S.eye_array(90, 40) * 3.0
    ).tocsr()
    Mg = S.diags_array(a(np.linspace(1.0, 2.0, 144)), format="csr")
    Mu = S.diags_array(a(np.linspace(1.0, 2.0, n)), format="csr")
    for key, A in (("lap", lap), ("sym", sym), ("herm", herm)):
        for which in ("LM", "SM", "LA", "SA") + (("BE",) if key != "herm" else ()):
            yield f"eigsh {key} {which}", lambda A=A, which=which: S.eigsh(A, k=4, which=which)
        for fm in fmts:
            if fm != "csr":
                yield f"eigsh {key} {fm}", lambda A=A, fm=fm: S.eigsh(A.asformat(fm), k=3).eigenvectors
        yield f"eigsh {key} dense", lambda A=A: S.eigsh(A.toarray(), k=3).eigenvectors
        yield (
            f"eigsh {key} python operator",
            lambda A=A: S.eigsh(S.LinearOperator(A.shape, lambda v: A @ v, dtype=A.dtype), k=3).eigenvalues,
        )
        yield f"eigsh {key} sigma", lambda A=A: S.eigsh(A, k=3, sigma=0.7)
        yield f"eigsh {key} no vectors", lambda A=A: S.eigsh(A, k=3, return_eigenvectors=False)
        yield f"eigsh {key} max_iter 2", lambda A=A: S.eigsh(A, k=4, which="SM", max_iter=2)
    yield "eigsh lap M", lambda: S.eigsh(lap, k=3, M=Mg)
    yield "eigsh lap M SA", lambda: S.eigsh(lap, k=3, M=Mg, which="SA")
    yield "eigsh lap M sigma", lambda: S.eigsh(lap, k=3, M=Mg, sigma=0.5)
    yield "eigsh lap buckling", lambda: S.eigsh(lap, k=3, sigma=0.5, mode="buckling")
    yield "eigsh lap cayley", lambda: S.eigsh(lap, k=3, sigma=0.5, mode="cayley")
    yield "eigsh lap cayley M", lambda: S.eigsh(lap, k=3, sigma=0.5, mode="cayley", M=Mg)
    yield "eigsh lap OPinv", lambda: S.eigsh(lap, k=3, sigma=0.0, OPinv=S.LinearOperator(
        lap.shape, lambda v: S.spsolve(lap.tocsc(), v)))  # fmt: skip
    yield "eigsh lap v0", lambda: S.eigsh(lap, k=3, v0=a(np.ones(144)))
    yield "eigsh lap rng", lambda: S.eigsh(lap, k=3, rng=sd.random.default_rng(3))
    yield "eigsh lap ncv tol", lambda: S.eigsh(lap, k=3, ncv=8, tol=1e-6)
    for key, A in (("unsym", unsym), ("cplx", cplx)):
        for which in ("LM", "SM", "LR", "SR", "LI", "SI"):
            yield f"eigs {key} {which}", lambda A=A, which=which: S.eigs(A, k=4, which=which)
        for fm in fmts:
            if fm != "csr":
                yield f"eigs {key} {fm}", lambda A=A, fm=fm: S.eigs(A.asformat(fm), k=3).eigenvectors
        yield f"eigs {key} dense", lambda A=A: S.eigs(A.toarray(), k=3).eigenvalues
        yield f"eigs {key} aslinearoperator", lambda A=A: S.eigs(S.aslinearoperator(A), k=3).eigenvalues
        yield f"eigs {key} sigma", lambda A=A: S.eigs(A, k=3, sigma=0.5 + 0.25j)
        yield f"eigs {key} no vectors", lambda A=A: S.eigs(A, k=3, return_eigenvectors=False)
    yield "eigs unsym M", lambda: S.eigs(unsym, k=3, M=Mu)
    yield "eigs unsym sigma real", lambda: S.eigs(unsym, k=3, sigma=0.5)
    yield "eigs unsym OPpart i", lambda: S.eigs(unsym, k=3, sigma=0.5 + 0.25j, OPpart="i")
    yield "eigs unsym max_iter 1", lambda: S.eigs(unsym, k=4, which="SM", max_iter=1)
    for key, A in (("rect", rect), ("wide", rect.T.tocsr()), ("cplx", cplx), ("lap", lap)):
        for which in ("LM", "SM"):
            yield f"svds {key} {which}", lambda A=A, which=which: S.svds(A, k=4, which=which)
        for opt in (False, "u", "vh"):
            yield f"svds {key} {opt}", lambda A=A, opt=opt: S.svds(A, k=3, return_singular_vectors=opt)
    for fm in fmts:
        if fm != "csr":
            yield f"svds rect {fm}", lambda fm=fm: S.svds(rect.asformat(fm), k=3).U
    yield "svds rect dense", lambda: S.svds(rect.toarray(), k=3).Vh
    yield (
        "svds rect python operator",
        lambda: S.svds(S.LinearOperator(rect.shape, lambda v: rect @ v, rmatvec=lambda v: rect.T @ v), k=3).S,
    )
    yield "svds rect ncv v0", lambda: S.svds(rect, k=3, ncv=6, v0=a(np.ones(40)))
    yield "svds lap SM max_iter 2", lambda: S.svds(lap, k=4, which="SM", max_iter=2)
    yield "summary eigsh lap", lambda: S.eigsh(lap, k=3).summary()
    yield "summary eigs cplx", lambda: S.eigs(cplx, k=3).summary()
    yield "summary svds rect", lambda: S.svds(rect, k=3).summary()
    yield "repr eigs unsym", lambda: repr(S.eigs(unsym, k=3))
    yield "unpack eigsh lap", lambda: tuple(S.eigsh(lap, k=3))
    yield "unpack svds rect", lambda: tuple(S.svds(rect, k=3))

    # long banded matrices: the pool splits every vector operation at 3 and 10 threads
    k = 40000
    t = np.arange(k, dtype=np.float64)
    band = S.diags_array([a(1.0 + 3.0 * t / k), a(np.full(k - 1, 0.2)), a(np.full(k - 1, 0.2))], offsets=[0, 1, -1],
                         format="csr")  # fmt: skip
    skew = (band + S.diags_array(a(np.full(k - 1, 0.1)), offsets=1, format="csr")).tocsr()
    yield "eigsh long", lambda: S.eigsh(band, k=3, ncv=12)
    yield "eigsh long complex", lambda: S.eigsh(band * (1 + 0j), k=3, ncv=12)
    yield "eigs long", lambda: S.eigs(skew, k=3, ncv=12)
    yield "svds long", lambda: S.svds(skew, k=3, ncv=12)

    # wrong calls
    yield "error eigsh list A", lambda: S.eigsh([[1.0]])
    yield "error eigsh k n", lambda: S.eigsh(lap, k=144)
    yield "error eigs k n-1", lambda: S.eigs(unsym, k=119)
    yield "error eigsh ncv", lambda: S.eigsh(lap, k=3, ncv=3)
    yield "error eigsh which", lambda: S.eigsh(lap, which="LR")
    yield "error eigs which", lambda: S.eigs(unsym, which="LA")
    yield "error eigsh mode no sigma", lambda: S.eigsh(lap, mode="buckling")
    yield "error eigsh maxiter", lambda: S.eigsh(lap, maxiter=3)
    yield "error eigsh v0 zero", lambda: S.eigsh(lap, v0=a(np.zeros(144)))
    singular = S.diags_array(a(np.r_[0.0, np.arange(1.0, 10.0)]), format="csr")
    yield "error eigsh singular sigma", lambda: S.eigsh(singular, k=2, sigma=0.0)
    yield "error svds solver", lambda: S.svds(rect, solver="arpack")
    yield "error svds k", lambda: S.svds(rect, k=40)
    yield "error svds which", lambda: S.svds(rect, which="LA")
    yield "error eigs OPpart", lambda: S.eigs(unsym, sigma=1.0, OPpart="i")


def cases_sp7(L, fmts):
    """The matrix functions and norms (SP7): expm, expm_multiply, onenormest, norm on small matrices
    in every format and as operators, long banded matrices for the pool, and the wrong calls."""
    sd, S = L.sd, L.sd.sparse
    r = L.rng(707)
    a = L.arr
    g = 12
    T = S.diags_array([a(np.full(g - 1, -1.0)), a(np.full(g, 2.0)), a(np.full(g - 1, -1.0))], offsets=[-1, 0, 1])
    I1 = S.eye_array(g)
    lap = (S.kron(I1, T) + S.kron(T, I1)).tocsr()  # 144 x 144
    n = 120
    rows, cols = r.integers(0, n, 600), r.integers(0, n, 600)
    unsym = (S.coo_array((a(r.standard_normal(600)), (a(rows), a(cols))), shape=(n, n)) * 0.3).tocsr()
    m = 40
    zr, zc = r.integers(0, m, 200), r.integers(0, m, 200)
    cplx = S.coo_array((a(r.standard_normal(200) + 1j * r.standard_normal(200)), (a(zr), a(zc))), shape=(m, m)).tocsr()
    upper = S.triu(S.coo_array((a(r.standard_normal(150)), (a(r.integers(0, 30, 150)), a(r.integers(0, 30, 150)))),
                               shape=(30, 30)), format="csr")  # fmt: skip
    diag = S.diags_array(a(np.linspace(-3.0, 2.0, 10)), format="csr")
    b144, bn, bm = a(np.sin(np.arange(144.0))), a(np.cos(np.arange(float(n)))), a(np.sin(np.arange(float(m))) + 0.5j)
    B3 = a(r.standard_normal((n, 3)))

    for key, A in (("lap", lap * 0.25), ("lap big", lap * 4.0), ("unsym", unsym), ("cplx", cplx), ("upper", upper),
                   ("lower", upper.T.tocsr()), ("diag", diag)):  # fmt: skip
        yield f"expm {key}", lambda A=A: S.expm(A)
        for fm in fmts:
            if fm != "csr":
                yield f"expm {key} {fm}", lambda A=A, fm=fm: S.expm(A.asformat(fm))
    yield "expm int", lambda: S.expm(S.csr_array(a(np.array([[1, 2], [3, 0]]))))
    yield "expm 1x1", lambda: S.expm(S.csr_array(a(np.array([[0.5]]))))
    yield "expm empty", lambda: S.expm(S.csr_array((0, 0)))
    yield "expm zero", lambda: S.expm(S.csr_array((3, 3)))
    yield "expm huge", lambda: S.expm(S.csr_array(a(np.array([[-1e40, 1e39], [1e39, -2e40]]))))

    for key, A, b in (("lap", lap * -0.1, b144), ("unsym", unsym, bn), ("cplx", cplx, bm)):
        yield f"expm_multiply {key}", lambda A=A, b=b: S.expm_multiply(A, b)
        yield f"expm_multiply {key} csc", lambda A=A, b=b: S.expm_multiply(A.tocsc(), b)
        yield f"expm_multiply {key} dense", lambda A=A, b=b: S.expm_multiply(A.toarray(), b)
        for kw in ({"start": 0.0, "stop": 1.0, "num": 5}, {"start": 0.0, "stop": 3.0, "num": 4, "endpoint": False},
                   {"start": -1.0, "stop": 2.0, "num": 31}, {"start": 0.5, "stop": 0.5, "num": 3}):  # fmt: skip
            yield (
                f"expm_multiply {key} interval {sorted(kw.items())}",
                lambda A=A, b=b, kw=kw: S.expm_multiply(A, b, **kw),
            )
    yield "expm_multiply unsym B", lambda: S.expm_multiply(unsym, B3)
    yield "expm_multiply unsym B interval", lambda: S.expm_multiply(unsym, B3, start=0.0, stop=1.0, num=3)
    yield "expm_multiply unsym traceA", lambda: S.expm_multiply(unsym, bn, traceA=2.5)
    yield "expm_multiply unsym traceA complex", lambda: S.expm_multiply(unsym, bn, traceA=1 + 1j)
    yield "expm_multiply unsym aslinearoperator", lambda: S.expm_multiply(S.aslinearoperator(unsym), bn)
    yield (
        "expm_multiply unsym python operator",
        lambda: S.expm_multiply(S.LinearOperator(unsym.shape, lambda v: unsym @ v, rmatvec=lambda v: unsym.T @ v,
                                                 dtype=sd.float64), bn),
    )  # fmt: skip
    yield (
        "expm_multiply unsym python operator interval",
        lambda: S.expm_multiply(S.LinearOperator(unsym.shape, lambda v: unsym @ v, rmatvec=lambda v: unsym.T @ v,
                                                 dtype=sd.float64), bn, start=0.0, stop=1.0, num=4),
    )  # fmt: skip
    yield "expm_multiply zero", lambda: S.expm_multiply(S.csr_array((4, 4)), a(np.arange(4.0)))
    yield "expm_multiply int B", lambda: S.expm_multiply(unsym, a(np.arange(n)))

    for key, A in (("lap", lap), ("unsym", unsym), ("cplx", cplx), ("upper", upper)):
        yield f"onenormest {key}", lambda A=A: S.onenormest(A)
        yield f"onenormest {key} vectors", lambda A=A: S.onenormest(A, compute_v=True, compute_w=True)
        yield f"onenormest {key} t1", lambda A=A: S.onenormest(A, t=1)
        yield f"onenormest {key} t4 it8", lambda A=A: S.onenormest(A, t=4, max_iter=8)
        yield f"onenormest {key} exact", lambda A=A: S.onenormest(A, t=A.shape[0], compute_w=True)
        yield f"onenormest {key} dense", lambda A=A: S.onenormest(A.toarray())
        yield f"onenormest {key} operator", lambda A=A: S.onenormest(S.aslinearoperator(A))
    yield "onenormest unsym v", lambda: S.onenormest(unsym, compute_v=True)
    yield "onenormest unsym w", lambda: S.onenormest(unsym, compute_w=True)
    yield "onenormest empty", lambda: S.onenormest(S.csr_array((0, 0)))
    yield "onenormest unsym summary", lambda: S.onenormest(unsym, compute_v=True).summary()
    yield "onenormest unsym repr", lambda: repr(S.onenormest(unsym, compute_w=True))
    yield "onenormest unsym unpack", lambda: tuple(S.onenormest(unsym, compute_v=True, compute_w=True))

    x1 = S.csr_array(a(np.array([0.0, 3.0, 0.0, -4.0, 1e-200, 0.0])))
    big = S.csr_array(a(np.array([[1e200, 1e200, 0.0], [0.0, 1e-200, 3.0], [0.0, 0.0, 0.0]])))
    nan = S.csr_array(a(np.array([[1.0, np.nan], [0.0, 2.0]])))
    ints = S.csr_array(a(np.array([[1, -2, 0], [0, 0, 5]])))
    for key, A in (("lap", lap), ("unsym", unsym), ("cplx", cplx), ("big", big), ("ints", ints)):
        for o in (None, "fro", 1, -1, 2, math.inf, -math.inf):
            yield f"norm {key} {o}", lambda A=A, o=o: S.norm(A, ord=o)
        for o in (None, 1, 0, -1, math.inf, -math.inf, 3, 0.5, -1.5):
            for ax in (0, 1):
                yield f"norm {key} {o} axis {ax}", lambda A=A, o=o, ax=ax: S.norm(A, ord=o, axis=ax)
        yield f"norm {key} transposed axes", lambda A=A: S.norm(A, ord=1, axis=(1, 0))
    for fm in fmts:
        if fm != "csr":
            yield f"norm unsym {fm}", lambda fm=fm: [S.norm(unsym.asformat(fm), ord=o) for o in (None, 1, 2, -math.inf)]
    for o in (None, 1, 0, -1, math.inf, -math.inf, 3, -0.5):
        yield f"norm 1-D {o}", lambda o=o: S.norm(x1, ord=o)
    yield "norm nan", lambda: [S.norm(nan), S.norm(nan, ord=1), S.norm(nan, axis=1)]
    yield "norm bool", lambda: S.norm(S.csr_array(a(np.array([[True, False], [True, True]]))), ord=math.inf)
    yield "norm empty", lambda: [S.norm(S.csr_array((0, 3)), ord=o) for o in (None, 1, 2, -math.inf)]
    yield "norm zero 2", lambda: S.norm(S.csr_array((3, 4)), ord=2)
    yield "norm row 2", lambda: S.norm(S.csr_array(a(np.array([[3.0, 0.0, -4.0]]))), ord=2)

    # long banded matrices: the pool splits the products and the vector operations at 3 and 10 threads
    k = 40000
    t = np.arange(k, dtype=np.float64)
    band = S.diags_array([a(-1.0 - t / k), a(np.full(k - 1, 0.3)), a(np.full(k - 1, 0.2))], offsets=[0, 1, -1],
                         format="csr")  # fmt: skip
    bk = a(np.sin(t))
    yield "expm_multiply long", lambda: S.expm_multiply(band, bk)
    yield "expm_multiply long interval", lambda: S.expm_multiply(band, bk, start=0.0, stop=2.0, num=4)
    yield "onenormest long", lambda: S.onenormest(band, compute_w=True)
    yield "norm long", lambda: [S.norm(band, ord=o) for o in (None, 1, 2, math.inf)]
    yield "norm long axis", lambda: S.norm(band, axis=0)

    # wrong calls
    yield "error expm dense", lambda: S.expm(lap.toarray())
    yield "error expm rect", lambda: S.expm(S.csr_array((2, 3)))
    yield "error expm nan", lambda: S.expm(nan)
    yield "error expm_multiply sparse B", lambda: S.expm_multiply(unsym, unsym)
    yield "error expm_multiply rows", lambda: S.expm_multiply(unsym, b144)
    yield "error expm_multiply num", lambda: S.expm_multiply(unsym, bn, start=0, stop=1, num=1)
    yield "error expm_multiply stop", lambda: S.expm_multiply(unsym, bn, start=0)
    yield "error expm_multiply traceA", lambda: S.expm_multiply(unsym, bn, traceA="1")
    yield "error onenormest max_iter", lambda: S.onenormest(lap, max_iter=1)
    yield "error onenormest itmax", lambda: S.onenormest(lap, itmax=3)
    yield "error onenormest t", lambda: S.onenormest(lap, t=0)
    yield "error onenormest rect", lambda: S.onenormest(S.csr_array((2, 3)))
    yield "error norm nuc", lambda: S.norm(lap, ord="nuc")
    yield "error norm -2", lambda: S.norm(lap, ord=-2)
    yield "error norm ord str axis", lambda: S.norm(lap, ord="fro", axis=0)
    yield "error norm bool", lambda: S.norm(lap, ord=True)
    yield "error norm axis", lambda: S.norm(lap, axis=(0, 0))
    yield "error norm dense", lambda: S.norm(lap.toarray())
    yield "error norm positional ord", lambda: S.norm(lap, 1)
    yield "error norm 2 nan", lambda: S.norm(nan, ord=2)
    yield "error sd.linalg.expm sparse", lambda: sd.linalg.expm(lap)
    yield "error sd.linalg.matrix_norm sparse", lambda: sd.linalg.matrix_norm(lap)
