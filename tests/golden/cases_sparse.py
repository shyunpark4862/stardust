"""Golden cases for sd.sparse (tools/golden.py has the contract).

The formats and conversions (milestone SP1, docs/design/sparse.md §8): every constructor input
(dense, a shape, coordinates, compressed parts, another format, a scipy array), the conversions
between the four formats and to dense, the transposes, reshape and resize, nonzero and
count_nonzero, eliminate_zeros and sum_duplicates, astype, conj, real and imag, the DOK element
access, and the wrong calls (their errors are results). The matrices come from fixed literals and
L.rng(seed): small ones with repeats, stored zeros, NaN and -0.0, and random ones of 2*10^5
entries with repeats, which take the multithreaded bucket path of the conversions
(csrc/sparse/canonical.cpp) at 3 and 10 threads and the direct path at 1 for the smaller sizes.

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
        "big_r": r.integers(0, 20000, n),
        "big_c": r.integers(0, 15000, n),
        "big_v": r.standard_normal(n) * 10.0 ** r.integers(-12, 13, n),
        "long_r": r.integers(0, 30, n),
        "long_c": r.integers(0, 10**6, n),
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
        "big": lambda: coo_of(d["big_v"], d["big_r"], d["big_c"], (20000, 15000)),
        "long": lambda: coo_of(d["long_v"], d["long_r"], d["long_c"], (30, 10**6)),
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
