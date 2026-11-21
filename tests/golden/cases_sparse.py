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
