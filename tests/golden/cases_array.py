"""Golden cases for the array area (tools/golden.py has the contract).

Every name of the array area's ``__all__`` (the top level of sd without its subpackages) plus the
ndarray attributes, methods, operators and Python protocols. Inputs come only from ``L.rng(seed)``
and fixed literals.

Layout of the file: one generator per function family, chained by cases().
Inputs are read-only numpy arrays memoized at module level, wrapped by ``L.arr`` and then copied, so every
stardust input owns its memory. Thunks that mutate always work on a
``.copy()``; the last case checks that no shared input was changed.

"Large" inputs cross the parallel threshold (2**15 elements) so the
multithreaded paths run; small ones stay below it.
"""

import hashlib
import math
import operator

INT_MAX, INT_MIN = 2**63 - 1, -(2**63)
NAN, INF = math.nan, math.inf
KINDS = "bifc"

F_SP = [
    0.0,
    -0.0,
    1.5,
    -2.5,
    0.5,
    3.0,
    INF,
    -INF,
    NAN,
    1e308,
    5e-324,
    -1e-310,
    2.0,
    100.0,
    0.7,
    -0.3,
    710.0,
    1e-8,
    -1.0,
    1.0,
    2.5,
    -0.5,
]
I_SP = [0, 1, -1, 2, -7, 3, 5, INT_MAX, INT_MIN, 12345678901, 63, 64, -64, -2]
_CF = [0.0, -0.0, 1.5, -2.5, INF, -INF, NAN, 0.7, 1e-8, 2.0]
C_SP = [complex(r, i) for r in _CF for i in _CF]
B_SP = [True, False]
SPECIAL = {"b": B_SP, "i": I_SP, "f": F_SP, "c": C_SP}
PYSCALAR = {"b": True, "i": 3, "f": 2.5, "c": complex(1.5, -2.0)}

UNARY = [
    "negative",
    "positive",
    "abs",
    "square",
    "sqrt",
    "sign",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "floor",
    "ceil",
    "trunc",
    "round",
    "logical_not",
    "bitwise_invert",
    "isnan",
    "isinf",
    "isfinite",
    "real",
    "imag",
    "conj",
    "reciprocal",
    "signbit",
    "deg2rad",
    "rad2deg",
]
BINARY = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "remainder",
    "pow",
    "atan2",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "maximum",
    "minimum",
    "copysign",
    "hypot",
    "logaddexp",
    "nextafter",
]
REDUCTIONS = ["sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax", "any", "all"]
NAN_REDUCTIONS = ["nansum", "nanprod", "nanmin", "nanmax", "nanmean", "nanvar", "nanstd", "nanargmin", "nanargmax"]
CUMULATIVE = ["cumulative_sum", "cumulative_prod"]
NAN_CUMULATIVE = ["nancumulative_sum", "nancumulative_prod"]

BINOPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
    "lshift": operator.lshift,
    "rshift": operator.rshift,
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}
INPLACE = {
    "iadd": operator.iadd,
    "isub": operator.isub,
    "imul": operator.imul,
    "itruediv": operator.itruediv,
    "ifloordiv": operator.ifloordiv,
    "imod": operator.imod,
    "ipow": operator.ipow,
    "iand": operator.iand,
    "ior": operator.ior,
    "ixor": operator.ixor,
    "ilshift": operator.ilshift,
    "irshift": operator.irshift,
}

_NP = {}  # read-only numpy inputs up to MEMO elements, memoized across calls of cases()
MEMO = 2**16  # larger inputs are made again when asked for (Ctx.S keeps one stardust copy)


def _digest(a):
    return hashlib.sha256(a.tobytes()).digest()


class Ctx:
    """Input factory: deterministic numpy data -> stardust arrays owning their memory."""

    def __init__(self, L):
        self.L, self.A, self.np = L, L.sd, L.np
        self.DT = {"b": L.sd.bool, "i": L.sd.int64, "f": L.sd.float64, "c": L.sd.complex128}
        self.NPT = {"b": L.np.bool_, "i": L.np.int64, "f": L.np.float64, "c": L.np.complex128}
        self.shared = []  # (stardust input, SHA-256 of its numpy source), checked by the guard case
        self.made = {}  # (kind, shape, seed) -> the stardust input

    def gen(self, kind, shape, seed=0):
        """kind: b/i/f/c random, 't' float with many ties, 'I' wide ints, 'n' float with nan/-0.0."""
        key = (kind, shape, seed)
        a = _NP.get(key)
        if a is None:
            np, r = self.np, self.L.rng(seed)
            if kind == "b":
                a = r.random(shape) < 0.5
            elif kind == "i":
                a = r.integers(-50, 50, shape)
            elif kind == "I":
                a = r.integers(-(10**6), 10**6, shape)
            elif kind == "f":
                a = r.standard_normal(shape)
            elif kind == "c":
                a = r.standard_normal(shape) + 1j * r.standard_normal(shape)
            elif kind == "t":
                a = r.integers(-5, 6, shape) * 0.5
            elif kind == "n":
                a = r.integers(-5, 6, shape) * 0.5
                a[r.random(shape) < 0.1] = np.nan
                a[r.random(shape) < 0.05] = -0.0
                a[r.random(shape) < 0.03] = np.inf
                a[r.random(shape) < 0.03] = -np.inf
            else:
                raise ValueError(kind)
            a.flags.writeable = False
            if a.size <= MEMO:
                _NP[key] = a
        return a

    def S(self, kind, shape, seed=0):
        """The stardust input of gen(kind, shape, seed): one array per key, shared by the cases
        that ask for it (no case writes to it: the guard case checks)."""
        key = (kind, shape, seed)
        x = self.made.get(key)
        if x is None:
            src = self.gen(kind, shape, seed)
            x = self.made[key] = self.L.arr(src).copy()
            self.shared.append((x, _digest(src)))
        return x

    def lit(self, values, kind=None):
        """A small array from literals (optionally of a given kind)."""
        np = self.np
        src = np.array(values, dtype=self.NPT[kind]) if kind else np.array(values)
        src.flags.writeable = False
        x = self.L.arr(src).copy()
        self.shared.append((x, _digest(src)))
        return x


def cases(L):
    c = Ctx(L)
    for section in (
        constants,
        creation,
        shapes,
        astype_cases,
        indexing,
        setitem,
        unary,
        binary,
        scalars_ops,
        outs,
        clip_where,
        operators,
        protocols,
        reductions,
        cumulative,
        sorting,
        unique,
        matmul,
        dtype_functions,
        creation_more,
        shapes_more,
        diag_block,
        quantiles,
        covariance,
        counting,
        index_more,
        tensordot_vecdot,
        threads,
        printing,
    ):
        yield from section(c)
    shared = c.shared

    def guard():
        return [_digest(c.np.asarray(x)) == digest for x, digest in shared].count(False)

    yield "zz guard shared inputs untouched", guard


# ---------------------------------------------------------------- constants


def constants(c):
    A, DT = c.A, c.DT
    for nm in ("e", "pi", "inf", "nan", "newaxis"):
        yield f"const {nm}", (lambda nm=nm: getattr(A, nm))
    for k in KINDS:
        yield f"dtype {k} object", (lambda k=k: DT[k])
        yield f"dtype {k} str", (lambda k=k: str(DT[k]))
    yield "dtype identity and equality", lambda: [(DT[a] is DT[b], DT[a] == DT[b]) for a in KINDS for b in KINDS]
    yield "dtype hashable", lambda: len({DT[k] for k in KINDS})
    yield (
        "ndarray isinstance",
        lambda: (
            isinstance(A.zeros(2), A.ndarray),
            isinstance(c.np.zeros(2), A.ndarray),
            isinstance(A.zeros((2, 2)).T, A.ndarray),
        ),
    )
    yield "ndarray construct err", lambda: A.ndarray()


# ---------------------------------------------------------------- creation


def creation(c):
    A, np, DT = c.A, c.np, c.DT
    lists = {
        "ints": [1, 2, 3],
        "floats": [1.5, -0.0, NAN, INF],
        "bools": [True, False, True],
        "complex": [1 + 2j, -0.0j, complex(NAN, 1)],
        "int+float": [1, 2.5],
        "bool+int": [True, 2],
        "int+complex": [1, 1j],
        "bool+float": [False, 0.5],
        "nested2d": [[1, 2, 3], [4, 5, 6]],
        "nested3d": [[[1.0], [2.0]], [[3.0], [4.0]]],
        "empty": [],
        "empty2d": [[]],
        "empty3d": [[[], []]],
        "big ints": [INT_MAX, INT_MIN],
        "tuple": (1, 2, 3),
        "nested tuple": ((1, 2), [3, 4]),
        "neg zero ints": [0, -0],
        "5d": [[[[[1, 2]]]]],
        "single": [7],
    }
    for nm, v in lists.items():
        yield f"asarray list {nm}", (lambda v=v: A.asarray(v))
        for k in KINDS:
            yield f"asarray list {nm} dtype={k}", (lambda v=v, k=k: A.asarray(v, dtype=DT[k]))
    yield (
        "asarray float->int saturate",
        lambda: A.asarray([NAN, INF, -INF, 1e300, -1e300, 2.9, -2.9, -0.0], dtype=DT["i"]),
    )
    yield "asarray complex->bool", lambda: A.asarray([0j, 1j, complex(NAN, 0), -0.0 + 0j], dtype=DT["b"])
    yield "asarray int too big err", lambda: A.asarray([2**63])
    yield "asarray int too big to float", lambda: A.asarray([2**63, 1.0])
    yield "asarray int too big dtype=float", lambda: A.asarray([2**70], dtype=DT["f"])
    yield "asarray int too small err", lambda: A.asarray([-(2**63) - 1])
    yield "asarray ragged err", lambda: A.asarray([[1, 2], [3]])
    yield "asarray ragged depth err", lambda: A.asarray([1, [2]])
    yield "asarray scalar int err", lambda: A.asarray(3)
    yield "asarray scalar float err", lambda: A.asarray(2.5)
    yield "asarray str err", lambda: A.asarray("abc")
    yield "asarray list of str err", lambda: A.asarray(["a"])
    yield "asarray None err", lambda: A.asarray(None)
    yield "asarray dict err", lambda: A.asarray({1: 2})
    yield "asarray dtype python type err", lambda: A.asarray([1], dtype=float)
    yield "asarray dtype string err", lambda: A.asarray([1], dtype="float64")
    yield "asarray dtype numpy err", lambda: A.asarray([1], dtype=np.float64)
    yield "asarray positional dtype err", lambda: A.asarray([1], DT["f"])
    yield "asarray copy bad err", lambda: A.asarray([1], copy=1)
    yield "asarray nested ndarrays", lambda: A.asarray([A.asarray([1, 2]), A.asarray([3.5, 4])])
    yield "asarray nested ndarray + list", lambda: A.asarray([A.arange(3), [4, 5, 6]])

    x = c.S("f", (3, 4))
    yield "asarray ndarray is same", lambda: A.asarray(x) is x
    yield "asarray ndarray same dtype is same", lambda: A.asarray(x, dtype=DT["f"]) is x
    yield "asarray ndarray copy=True", lambda: (A.asarray(x, copy=True) is x, A.asarray(x, copy=True))
    yield "asarray ndarray copy=False", lambda: A.asarray(x, copy=False) is x
    yield "asarray ndarray to int", lambda: A.asarray(x, dtype=DT["i"])
    yield "asarray ndarray to int copy=False err", lambda: A.asarray(x, dtype=DT["i"], copy=False)
    yield "asarray list copy=False err", lambda: A.asarray([1, 2], copy=False)
    yield "asarray complex->float err", lambda: A.asarray([1 + 1j], dtype=DT["f"])
    yield "asarray complex nd->int err", lambda: A.asarray(c.lit([1j]), dtype=DT["i"])

    # buffer protocol import
    src = c.gen("f", (6, 5), 3)
    np_inputs = {
        "f64": src,
        "f64 F-order": np.asfortranarray(src),
        "f64 strided": src[::-2, 1::2],
        "i64": (src * 10).astype(np.int64),
        "bool": src > 0,
        "c128": src + 1j * src[::-1],
        "i32": (src * 10).astype(np.int32),
        "i16": (src * 10).astype(np.int16),
        "u8": (src * 10).astype(np.uint8),
        "u32": np.abs(src * 10).astype(np.uint32),
        "f32": src.astype(np.float32),
        "c64": (src + 1j).astype(np.complex64),
        "f16": src.astype(np.float16),
        "i8": (src * 10).astype(np.int8),
        "empty": np.zeros((0, 3)),
        "5d": src.reshape(1, 2, 3, 5, 1),
    }
    for nm, a in np_inputs.items():
        yield f"asarray numpy {nm}", (lambda a=a: A.asarray(a))
        yield f"asarray numpy {nm} dtype=c", (lambda a=a: A.asarray(a, dtype=DT["c"]))
    yield "asarray numpy copy=True", lambda: A.asarray(np_inputs["f64 strided"], copy=True)
    yield "asarray numpy u64 err", lambda: A.asarray(np.abs(src * 10).astype(np.uint64))
    yield "asarray numpy u64 dtype=i err", lambda: A.asarray(np.abs(src * 10).astype(np.uint64), dtype=DT["i"])
    yield "asarray numpy u64 dtype=f err", lambda: A.asarray(np.abs(src * 10).astype(np.uint64), dtype=DT["f"])
    yield "asarray numpy 0d err", lambda: A.asarray(np.float64(2.0).reshape(()))
    yield "asarray bytes", lambda: A.asarray(b"\x00\x01\xff")
    yield "asarray bytearray", lambda: A.asarray(bytearray(b"\x05\x06"))
    yield "asarray numpy bool bytes normalized", lambda: A.asarray(np.frombuffer(b"\x00\x02\x01\xff", dtype=np.bool_))
    yield (
        "asarray numpy misaligned",
        lambda: A.asarray(np.frombuffer(b"\x00" + bytes(range(64)), dtype=np.uint8)[1:].view(np.float64)),
    )
    yield "asarray numpy object err", lambda: A.asarray(np.array([1, "a"], dtype=object))
    yield "asarray numpy datetime err", lambda: A.asarray(np.array(["2020-01-01"], dtype="datetime64[D]"))
    yield "asarray numpy readonly view write err", lambda: _setitem(A.asarray(c.gen("f", (3,), 1)), 0, 1.0)
    yield "asarray numpy large", lambda: A.asarray(c.gen("f", (1_000_000,), 1)[::-3])

    shapes = [(), 0, 3, (0,), (2, 3), (3, 0, 2), (1, 1), (2, 1, 3, 1, 2), (1,)]
    for fn in ("zeros", "ones", "empty"):
        for s in shapes:
            if fn == "empty":
                yield f"{fn} {s!r}" + (" err" if s == () else ""), (lambda s=s: _layout(A.empty(s)))
            else:
                yield f"{fn} {s!r}" + (" err" if s == () else ""), (lambda fn=fn, s=s: getattr(A, fn)(s))
        for k in KINDS:
            if fn == "empty":
                yield f"{fn} dtype={k}", (lambda k=k: _layout(A.empty((2, 3), dtype=DT[k])))
            else:
                yield f"{fn} dtype={k}", (lambda fn=fn, k=k: getattr(A, fn)((2, 3), dtype=DT[k]))
        for bad in ((-1,), (2, -3), 2.0, (2.0,), "3", [2, 3], None, (True,)):
            yield f"{fn} bad shape {bad!r} err", (lambda fn=fn, bad=bad: getattr(A, fn)(bad))
        yield f"{fn} dtype python type err", (lambda fn=fn: getattr(A, fn)(3, dtype=int))
        yield (
            f"{fn} large",
            (lambda fn=fn: _layout(getattr(A, fn)((1000, 1000))) if fn == "empty" else getattr(A, fn)((1000, 1000))),
        )
    yield "zeros huge err", lambda: A.zeros((2**40, 2**40))
    for v in (True, 3, -2.5, 1 + 2j, NAN, -0.0, INT_MAX):
        yield f"full fill {v!r}", (lambda v=v: A.full((2, 3), v))
        for k in KINDS:
            yield f"full fill {v!r} dtype={k}", (lambda v=v, k=k: A.full((2, 3), v, dtype=DT[k]))
    yield "full 0d err", lambda: A.full((), 2.0)
    yield "full large", lambda: A.full(1_000_000, 0.25)
    yield "full fill too big err", lambda: A.full(3, 2**64)
    yield "full fill array err", lambda: A.full(3, A.asarray([1.0]))
    yield "full fill str err", lambda: A.full(3, "a")
    yield "full fill None err", lambda: A.full(3, None)
    yield "full keyword fill err", lambda: A.full(3, fill_value=1.0)

    srcs = {
        "C": c.S("f", (3, 4)),
        "F": c.S("f", (4, 3)).T,
        "strided": c.S("f", (6, 8))[::2, ::-2],
        "int": c.S("i", (2, 5)),
        "bool": c.S("b", (5,)),
        "complex": c.S("c", (2, 2)),
        "empty": c.S("f", (0, 3)),
        "5d": c.S("f", (1, 2, 3, 2, 1)),
        "large F": c.S("f", (1000, 600)).T,
    }
    for nm, s in srcs.items():
        for fn in ("zeros_like", "ones_like", "empty_like"):
            if fn == "empty_like":
                yield f"{fn} {nm}", (lambda s=s: _layout(A.empty_like(s)))
                yield f"{fn} {nm} dtype=c", (lambda s=s: _layout(A.empty_like(s, dtype=DT["c"])))
            else:
                yield f"{fn} {nm}", (lambda fn=fn, s=s: getattr(A, fn)(s))
                yield f"{fn} {nm} dtype=b", (lambda fn=fn, s=s: getattr(A, fn)(s, dtype=DT["b"]))
        yield f"full_like {nm} 1.5", (lambda s=s: A.full_like(s, 1.5))
        yield f"full_like {nm} 7 dtype=i", (lambda s=s: A.full_like(s, 7, dtype=DT["i"]))
        yield f"full_like {nm} True", (lambda s=s: A.full_like(s, True))
    yield "full_like int with 1.5", lambda: A.full_like(srcs["int"], 1.5)
    yield "full_like int with 1j err", lambda: A.full_like(srcs["int"], 1j)
    yield "full_like float with 1j err", lambda: A.full_like(srcs["C"], 1j)
    yield "zeros_like list err", lambda: A.zeros_like([1, 2])
    yield "ones_like numpy err", lambda: A.ones_like(np.zeros(2))

    aranges = [
        (5,),
        (0,),
        (-3,),
        (2, 10),
        (10, 2),
        (10, 2, -3),
        (0, 1, 0.1),
        (1.5, 4),
        (0.5,),
        (-1, 1, 0.25),
        (0, 10, 3),
        (10, 0, -1),
        (3, 3),
        (0, -5, -0.5),
        (2**62, 2**62 + 5),
        (1, 2, 1e-1),
        (0, 1_000_000),
    ]
    for args in aranges:
        yield f"arange {args!r}", (lambda args=args: A.arange(*args))
    for k in KINDS:
        yield f"arange (0, 6, 2) dtype={k}", (lambda k=k: A.arange(0, 6, 2, dtype=DT[k]))
        yield f"arange (0.5, 3) dtype={k}", (lambda k=k: A.arange(0.5, 3, dtype=DT[k]))
    yield "arange keyword stop err", lambda: A.arange(2, stop=5)
    yield "arange keyword step err", lambda: A.arange(0, 10, step=4)
    yield "arange float large", lambda: A.arange(0.0, 100.0, 1e-4)
    yield "arange (nan,)", lambda: A.arange(NAN)
    for bad in [(0, 5, 0), (0.0, 1.0, 0.0), (1j,), ("a",), (None,), (0, INF), (0, 2**70)]:
        yield f"arange {bad!r} err", (lambda bad=bad: A.arange(*bad))
    yield "arange start keyword err", lambda: A.arange(start=3)

    lins = [
        (0, 1, 5),
        (0, 1, 1),
        (0, 1, 0),
        (-1.5, 2.5, 9),
        (1, 0, 4),
        (0, 1e300, 3),
        (0, 10, 11),
        (5, 5, 3),
        (0, 1, 7),
        (1 + 1j, 2 - 3j, 4),
        (True, False, 3),
        (0, 1, 1_000_000),
    ]
    for args in lins:
        yield f"linspace {args!r}", (lambda args=args: A.linspace(*args))
        yield f"linspace {args!r} endpoint=False", (lambda args=args: A.linspace(*args, endpoint=False))
    for k in KINDS:
        yield f"linspace dtype={k}", (lambda k=k: A.linspace(0, 4, 5, dtype=DT[k]))
    yield "linspace complex dtype=f err", lambda: A.linspace(0, 1j, 3, dtype=DT["f"])
    yield "linspace negative num err", lambda: A.linspace(0, 1, -1)
    yield "linspace float num err", lambda: A.linspace(0, 1, 2.0)
    yield "linspace missing num err", lambda: A.linspace(0, 1)
    yield "linspace nan", lambda: A.linspace(0, NAN, 3)

    eyes = [(3,), (0,), (1,), (3, 5), (5, 3), (4, 4, 1), (4, 4, -2), (3, 5, 4), (3, 5, -3), (3, 3, 10), (2, 0)]
    for args in eyes:
        yield (
            f"eye {args!r}",
            (
                lambda n=args[0], m=(args[1] if len(args) > 1 else None), k=(args[2] if len(args) > 2 else 0): A.eye(  # noqa: B008 -- inputs bound when the case is made
                    n, m, k=k
                )
            ),
        )
    for k in KINDS:
        yield f"eye dtype={k}", (lambda k=k: A.eye(3, 4, k=1, dtype=DT[k]))
    yield "eye large", lambda: A.eye(1000)
    yield "eye negative err", lambda: A.eye(-1)
    yield "eye float err", lambda: A.eye(2.0)
    yield "eye k float err", lambda: A.eye(3, k=1.0)
    yield "eye k positional err", lambda: A.eye(3, 3, 1)


def _rev(A, x):
    """All axes reversed: a C-order array becomes an F-order view (any ndim)."""
    return A.permute_dims(x, tuple(range(x.ndim))[::-1])


def _layout(x):
    """For empty/empty_like: the contents are undefined, so only the layout."""
    return (x.shape, x.dtype, x.strides, x.ndim, x.size)


def _setitem(x, key, value):
    x[key] = value
    return x


# ---------------------------------------------------------------- shapes and views


def shapes(c):
    A = c.A
    srcs = {
        "C": c.S("f", (4, 6)),
        "F": c.S("f", (6, 4)).T,
        "strided": c.S("f", (8, 12))[::2, ::-2],
        "3d": c.S("i", (2, 3, 4)),
        "empty": c.S("f", (0, 4)),
        "5d": c.S("c", (2, 1, 3, 1, 2)),
        "1elem": c.S("b", (1, 1)),
        "bcast": A.broadcast_to(c.S("f", (6,)), (4, 6)),
    }
    for nm, s in srcs.items():
        for shp in [(-1,), (s.size,), (2, -1), (-1, 2), (1, -1, 1), (s.size, 1)]:
            if s.size % 2 and 2 in shp:
                continue
            yield f"reshape {nm} {shp!r}", (lambda s=s, shp=shp: _view(A.reshape(s, shp), s))
            yield f"reshape {nm} {shp!r} copy=True", (lambda s=s, shp=shp: _view(A.reshape(s, shp, copy=True), s))
            yield (
                f"reshape {nm} {shp!r} copy=False" + (" err" if nm in ("F", "strided", "bcast") else ""),
                (lambda s=s, shp=shp: _view(A.reshape(s, shp, copy=False), s)),
            )
        yield f"ravel {nm}", (lambda s=s: _view(A.ravel(s), s))
        yield f"T {nm}" + ("" if s.ndim == 2 else " err"), (lambda s=s: _view(s.T, s))
        yield f"mT {nm}", (lambda s=s: _view(s.mT, s))
        yield f"matrix_transpose {nm}", (lambda s=s: _view(A.matrix_transpose(s), s))
        yield f"permute_dims {nm} reversed", (lambda s=s: _view(A.permute_dims(s, tuple(range(s.ndim))[::-1]), s))
        yield f"moveaxis {nm} 0->-1", (lambda s=s: _view(A.moveaxis(s, 0, -1), s))
        yield f"expand_dims {nm} 0", (lambda s=s: _view(A.expand_dims(s, axis=0), s))
        yield f"expand_dims {nm} -1", (lambda s=s: _view(A.expand_dims(s, axis=-1), s))
        yield f"expand_dims {nm} (0,2)", (lambda s=s: _view(A.expand_dims(s, axis=(0, 2)), s))
        yield f"broadcast_to {nm} (3,)+shape", (lambda s=s: _view(A.broadcast_to(s, (3,) + s.shape), s))
        yield f"method reshape {nm}", (lambda s=s: s.reshape((-1,)))
    x = srcs["C"]
    x3 = srcs["3d"]
    yield "reshape method tuple args err", lambda: x.reshape(3, 8)
    yield "reshape int shape", lambda: A.reshape(x, 24)
    yield "reshape empty (0,-1) err", lambda: A.reshape(srcs["empty"], (0, -1))
    yield "reshape empty -1 ambiguous err", lambda: A.reshape(srcs["empty"], (-1, 0))
    yield "reshape F copy=False err", lambda: A.reshape(srcs["F"], (-1,), copy=False)
    yield "reshape strided copy=False err", lambda: A.reshape(srcs["strided"], (-1,), copy=False)
    yield "reshape size mismatch err", lambda: A.reshape(x, (5, 5))
    yield "reshape two -1 err", lambda: A.reshape(x, (-1, -1))
    yield "reshape -2 err", lambda: A.reshape(x, (-2, -12))
    yield "reshape float err", lambda: A.reshape(x, (2.0, 12))
    yield "reshape to 0d err", lambda: A.reshape(c.lit([1.0]), ())
    yield "reshape copy keyword positional err", lambda: A.reshape(x, (24,), True)
    yield "reshape large F", lambda: A.reshape(c.S("f", (1000, 500)).T, (1000, 50, 10))
    yield "reshape high ndim", lambda: A.reshape(A.arange(64), (2,) * 6)
    yield "reshape 33 dims err", lambda: A.reshape(A.arange(1), (1,) * 33)
    yield "ravel large strided", lambda: A.ravel(c.S("f", (1000, 1000))[::-1, ::3])
    yield "ravel list err", lambda: A.ravel([1, 2])
    yield "T 1d err", lambda: _view(c.lit([1, 2, 3]).T, None)
    yield "mT 1d err", lambda: c.lit([1, 2, 3]).mT
    yield "matrix_transpose 1d err", lambda: A.matrix_transpose(c.lit([1, 2, 3]))
    for axes in [(0, 1, 2), (2, 0, 1), (1, 2, 0), (-1, 0, 1), (0, 1), (0, 0, 1), (0, 1, 3), (0, 1, 2.0)]:
        yield (
            f"permute_dims 3d {axes!r}"
            + ("" if repr(axes) in ("(0, 1, 2)", "(2, 0, 1)", "(1, 2, 0)", "(-1, 0, 1)") else " err"),
            (lambda axes=axes: _view(A.permute_dims(x3, axes), x3)),
        )
    yield "permute_dims list axes err", lambda: A.permute_dims(x3, [2, 1, 0])
    for s, d in [
        (0, 2),
        (-1, 0),
        ((0, 1), (1, 0)),
        ((0, 2), (2, 0)),
        ((0,), (1,)),
        (0, 3),
        (0, (1, 2)),
        ((0, 0), (1, 2)),
        (1, -1),
    ]:
        bad = (s, d) in ((0, 3), (0, (1, 2)), ((0, 0), (1, 2)))
        yield f"moveaxis 3d {s!r}->{d!r}" + (" err" if bad else ""), (lambda s=s, d=d: A.moveaxis(x3, s, d))
    sq = c.S("f", (1, 3, 1, 2, 1))
    for ax in [0, 2, -1, (0, 2), (0, 2, 4), (), 1, (0, 0), 5]:
        yield (
            f"squeeze {ax!r}" + (" err" if ax in (1, (0, 0), 5) else ""),
            (lambda ax=ax: _view(A.squeeze(sq, axis=ax), sq)),
        )
    yield "squeeze no axis err", lambda: A.squeeze(sq)
    yield "squeeze to 0d err", lambda: A.squeeze(c.lit([[2.5]]), axis=(0, 1))
    for ax in [1, 3, -4, (1, 3), (0, 1, 2, 3, 4), 4, -5, (0, 0), 2.0]:
        yield (
            f"expand_dims 3d {ax!r}" + (" err" if ax in (4, -5, (0, 0), 2.0) else ""),
            (lambda ax=ax: A.expand_dims(x3, axis=ax)),
        )
    yield "expand_dims no axis err", lambda: A.expand_dims(x3)
    v = c.S("f", (3, 1))
    for shp in [(3, 1), (3, 4), (2, 3, 4), (0, 3, 1), (3, 0), (4, 3), (3,), (1, 1)]:
        yield (
            f"broadcast_to (3,1)->{shp!r}" + (" err" if shp in ((4, 3), (3,), (1, 1)) else ""),
            (lambda shp=shp: _view(A.broadcast_to(v, shp), v)),
        )
    yield "broadcast_to readonly write err", lambda: _setitem(A.broadcast_to(v, (3, 4)), (0, 0), 1.0)
    yield "broadcast_to int shape", lambda: A.broadcast_to(c.lit([1.0]), 5)
    yield "broadcast_to list err", lambda: A.broadcast_to([1.0], (3,))
    a1, a2, a3 = c.S("f", (3, 1)), c.S("i", (4,)), c.S("b", (2, 1, 1))
    yield "broadcast_arrays 3", lambda: A.broadcast_arrays(a1, a2, a3)
    yield "broadcast_arrays 1", lambda: A.broadcast_arrays(a1)
    yield "broadcast_arrays 0", lambda: A.broadcast_arrays()
    yield "broadcast_arrays empty dim", lambda: A.broadcast_arrays(a1, c.S("f", (1, 0)))
    yield "broadcast_arrays incompatible err", lambda: A.broadcast_arrays(a1, c.S("f", (2, 1)))
    yield "broadcast_arrays readonly err", lambda: _setitem(A.broadcast_arrays(a1, a2)[0], (0, 0), 1.0)
    yield "broadcast_arrays list err", lambda: A.broadcast_arrays(a1, [1, 2])

    parts = {k: c.S(k, (2, 3), 5) for k in KINDS}
    for ka in KINDS:
        for kb in KINDS:
            yield f"concat {ka}{kb}", (lambda ka=ka, kb=kb: A.concat([parts[ka], parts[kb]]))
            yield f"stack {ka}{kb}", (lambda ka=ka, kb=kb: A.stack((parts[ka], parts[kb])))
    p = parts["f"]
    q = c.S("f", (3, 2), 6).T
    for ax in [0, 1, -1, -2, 2, -3]:
        yield (
            f"concat axis={ax}" + (" err" if ax in (2, -3) else ""),
            (lambda ax=ax: A.concat([p, q, p[:, ::-1]], axis=ax)),
        )
    for ax in [0, 1, 2, -1, -3, 3, -4]:
        yield f"stack axis={ax}" + (" err" if ax in (3, -4) else ""), (lambda ax=ax: A.stack([p, q], axis=ax))
    yield "concat single", lambda: A.concat([p])
    yield "concat empty parts", lambda: A.concat([p, c.S("f", (0, 3))])
    yield "concat 1d", lambda: A.concat((c.lit([1, 2]), c.lit([3.5])))
    yield "concat axis None", lambda: A.concat([p, q], axis=None)
    yield "concat large", lambda: A.concat([c.S("f", (500, 1000)), c.S("f", (1000, 300), 2).T], axis=0)
    yield "concat large axis1", lambda: A.concat([c.S("f", (1000, 300)), c.S("i", (1000, 200))], axis=1)
    yield "stack large", lambda: A.stack([c.S("f", (300, 300)), c.S("f", (300, 300), 2).T], axis=1)
    yield "concat empty list err", lambda: A.concat([])
    yield "stack empty list err", lambda: A.stack([])
    yield "concat shape mismatch err", lambda: A.concat([p, c.S("f", (2, 4))])
    yield "concat ndim mismatch err", lambda: A.concat([p, c.S("f", (3,))])
    yield "concat numpy member err", lambda: A.concat([p, c.np.zeros((2, 3))])
    yield "concat list member err", lambda: A.concat([p, [[1, 2, 3]]])
    yield "concat not sequence err", lambda: A.concat(p)
    yield "concat generator err", lambda: A.concat(x for x in [p, p])
    yield "stack shape mismatch err", lambda: A.stack([p, q.T])
    yield "concat 0d-like 1elem", lambda: A.concat([c.lit([1.0]), c.lit([True])])


def _view(r, src):
    """A result plus whether it shares the source's owner (view or copy)."""
    if src is None:
        return r
    return (r, r.base is src or (r.base is not None and r.base is src.base))


# ---------------------------------------------------------------- astype


def astype_cases(c):
    A, DT = c.A, c.DT
    for s in KINDS:
        x = c.lit(SPECIAL[s], s)
        xs = c.lit(SPECIAL[s], s)[::-2]
        for d in KINDS:
            yield f"astype {s}->{d}", (lambda x=x, d=d: A.astype(x, DT[d]))
            yield f"astype {s}->{d} strided", (lambda x=xs, d=d: A.astype(x, DT[d]))
            yield f"astype method {s}->{d}", (lambda x=x, d=d: x.astype(DT[d]))
    f = c.lit([1.0, 2.0])
    yield "astype copy default is new", lambda: A.astype(f, DT["f"]) is f
    yield "astype copy=None same", lambda: A.astype(f, DT["f"], copy=None) is f
    yield "astype copy=False same", lambda: A.astype(f, DT["f"], copy=False) is f
    yield "astype copy=False needs copy err", lambda: A.astype(f, DT["i"], copy=False)
    yield "astype copy=None converts", lambda: A.astype(f, DT["i"], copy=None)
    yield "astype dtype None err", lambda: A.astype(f, None)
    yield "astype dtype python err", lambda: f.astype(int)
    yield "astype copy positional err", lambda: f.astype(DT["i"], False)
    yield "astype list err", lambda: A.astype([1.0], DT["i"])
    yield "astype F-order keeps layout", lambda: c.S("f", (5, 3)).T.astype(DT["i"])
    yield "astype strided 3d", lambda: c.S("f", (4, 6, 8))[::-1, ::2, 1::3].astype(DT["c"])
    yield "astype 5d", lambda: c.S("i", (2, 3, 1, 2, 3)).astype(DT["b"])
    yield "astype empty", lambda: c.S("f", (0, 5)).astype(DT["i"])
    for s in "bifc":
        big = c.S(s, (1_000_000,) if s != "c" else (300_000,), 7)
        for d in KINDS:
            yield f"astype large {s}->{d}", (lambda big=big, d=d: A.astype(big, DT[d]))
    yield "astype large F", lambda: c.S("f", (1000, 1000), 8).T.astype(DT["i"])
    yield "astype large float->int scaled", lambda: A.astype(c.S("f", (1_000_000,), 9) * 1e18, DT["i"])
    yield "copy C", lambda: c.S("f", (3, 4)).copy()
    yield "copy F", lambda: c.S("f", (4, 3)).T.copy()
    yield "copy strided", lambda: c.S("c", (6, 6))[::-2, 1::2].copy()
    yield "copy base none", lambda: c.S("f", (4, 3))[1:].copy().base is None
    yield "copy large transposed 3d", lambda: A.permute_dims(c.S("f", (60, 70, 80)), (2, 0, 1)).copy()


# ---------------------------------------------------------------- indexing (get)


def indexing(c):
    A, np, DT = c.A, c.np, c.DT
    base_np = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    xs = {
        "C": c.lit(base_np),
        "F": c.L.arr(np.asfortranarray(base_np)),
        "Ft": _rev(A, c.lit(base_np.transpose(2, 1, 0).copy())),
        "strided": c.lit(np.arange(8 * 10 * 6.0).reshape(8, 10, 6))[::2, ::-2, ::-1],
    }
    m = base_np > 50
    keys = {
        "0": 0,
        "-1": -1,
        "int3": (1, 2, 3),
        "neg int3": (-1, -2, -3),
        "slice": slice(1, 3),
        "rev": slice(None, None, -1),
        "step2": slice(None, None, 2),
        "5:1:-2": slice(5, 1, -2),
        "empty slice": slice(10, 20),
        "(:,2)": (slice(None), 2),
        "(...,1)": (Ellipsis, 1),
        "(1,...)": (1, Ellipsis),
        "(None,)": (None,),
        "(None,1,None)": (None, 1, None),
        "(...,None)": (Ellipsis, None),
        "()": (),
        "...": Ellipsis,
        "None": None,
        "(-1,-1,::-1)": (-1, -1, slice(None, None, -1)),
        "(0:0,)": (slice(0, 0),),
        "(3:0:1,)": (slice(3, 0, 1),),
        "(::-2,...,4:0:-2)": (slice(None, None, -2), Ellipsis, slice(4, 0, -2)),
        "(0,...,None,1:4)": (0, Ellipsis, None, slice(1, 4)),
        "big slice bounds": (slice(-100, 100), slice(None, -100, -1)),
        "[0,2]": [0, 2],
        "[-1,0,-1]": [-1, 0, -1],
        "2d int": [[0, 1], [2, 3]],
        "([0,1],[1,2])": ([0, 1], [1, 2]),
        "([[0],[2]],[1,3])": ([[0], [2]], [1, 3]),
        "(...,[0,-1])": (Ellipsis, [0, -1]),
        "([0,2],...)": ([0, 2], Ellipsis),
        "(1,[0,2])": (1, [0, 2]),
        "([1,2],3)": ([1, 2], 3),
        "(1:,[0,0],[4,1])": (slice(1, None), [0, 0], [4, 1]),
        "(None,[0,1])": (None, [0, 1]),
        "([0,1],:,:)": ([0, 1], slice(None), slice(None)),
        "(:,[4,0],[1,5])": (slice(None), [4, 0], [1, 5]),
        "(::-1,[3,1])": (slice(None, None, -1), [3, 1]),
        "mask full": m,
        "(0,mask0)": (0, m[0]),
        "(:,mask0)": (slice(None), m[0]),
        "(mask2d,)": (m[:, :, 0],),
        "(mask2d,:)": (m[:, :, 0], slice(None)),
        "([0,2],mask5)": ([0, 2], base_np[0, :, 0] > 12),
        "[]": [],
        "([],:)": ([], slice(None)),
        "np empty int": np.array([], dtype=np.int64),
        "[T,F,T,F]": [True, False, True, False],
        "(2,3,[0,4])": (2, 3, [0, 4]),
        "np int idx": np.array([[3, 0], [1, 2]]),
        "sd int idx": "SD_INT",
        "sd mask": "SD_MASK",
        "strided sd idx": "SD_STRIDED",
        "np int64 scalar": np.int64(2),
        "mask all false": base_np < -1,
        "(...,mask6)": (Ellipsis, base_np[0, 0] > 2),
        "(None,mask4)": (None, [True, False, False, True]),
        # errors
        "oob 4 err": 4,
        "oob -5 err": -5,
        "too many err": (0, 0, 0, 0),
        "two ellipsis err": (Ellipsis, Ellipsis),
        "float err": 1.0,
        "str err": "a",
        "step0 err": slice(None, None, 0),
        "huge int err": 2**70,
        "True err": True,
        "False err": False,
        "(0,True) err": (0, True),
        "float list err": [1.0, 2.0],
        "sd float idx err": "SD_FLOAT",
        "[0,4] oob err": [0, 4],
        "([0],[9]) err": ([0], [9]),
        "[-5] err": [-5],
        "mask wrong shape err": np.ones((3, 5), bool),
        "mask short err": [True, False],
        "sep int-slice-arr err": (0, slice(None), [0, 1]),
        "sep arr-slice-int err": ([0, 1], slice(None), 0),
        "sep arr-slice-arr err": ([0, 1], slice(None), [0, 1]),
        "sep arr-None-arr err": ([0, 1], None, [0, 1]),
        "sep arr-...-arr err": ([0, 1], Ellipsis, [0, 1]),
        "ragged list err": [[0, 1], [2]],
        "dict err": {0: 1},
        "tuple of float err": (0, 1.5),
    }
    sd_keys = {
        "SD_INT": A.asarray([[3, 0], [1, 2]]),
        "SD_MASK": A.asarray(m),
        "SD_STRIDED": A.asarray([0, 1, 2, 3, 0, 1])[::-2],
        "SD_FLOAT": A.asarray([0.5]),
    }
    for lay, x in xs.items():
        for kn, key in keys.items():
            key = sd_keys.get(key, key) if isinstance(key, str) else key
            yield f"getitem {lay} {kn}", (lambda x=x, key=key: _get(x, key))
    for k in "bic":
        x = c.S(k, (4, 5, 6), 11)
        for kn in ("int3", "(:,2)", "rev", "[0,2]", "mask full", "(1,[0,2])", "(...,None)", "sd mask"):
            key = keys[kn]
            key = sd_keys.get(key, key) if isinstance(key, str) else key
            yield f"getitem {k} {kn}", (lambda x=x, key=key: _get(x, key))
        yield f"getitem {k} scalar type", (lambda x=x: type(x[1, 2, 3]).__name__)
    e = c.S("f", (3, 0, 2))
    for kn, key in {
        "0": 0,
        "(:,[])": (slice(None), []),
        "(:,:,[1])": (slice(None), slice(None), [1]),
        "(:,[0]) err": (slice(None), [0]),
        "mask": A.zeros((3, 0, 2), dtype=DT["b"]),
    }.items():
        yield f"getitem empty {kn}", (lambda key=key: e[key])
    hi = c.S("f", (2, 3, 2, 3, 2), 12)
    for kn, key in {
        "(1,...,0)": (1, Ellipsis, 0),
        "(:,::-1,0,[2,0],:)": (slice(None), slice(None, None, -1), 0, [2, 0], slice(None)),
        "mask": "M",
        "(None,1,None,...,None)": (None, 1, None, Ellipsis, None),
    }.items():
        if key == "M":
            key = hi > 0.0
        yield f"getitem 5d {kn}", (lambda key=key: hi[key])
    yield "getitem 32d", lambda: A.zeros((1,) * 32)[(0,) * 31]
    yield "getitem 33d newaxis err", lambda: A.zeros((1,) * 32)[None]
    # large: gather, masks, take along an axis
    big = c.S("f", (1_000_000,), 21)
    idx = c.S("I", (400_000,), 22) % 1_000_000
    M = c.S("f", (1000, 1000), 23)
    MF = c.S("f", (1000, 1000), 24).T
    T3 = A.permute_dims(c.S("f", (60, 70, 80), 25), (2, 0, 1))
    yield "getitem large gather", lambda: big[idx]
    yield "getitem large gather negative", lambda: big[idx - 1_000_000]
    yield "getitem large gather strided src", lambda: big[::-2][idx % 500_000]
    yield "getitem large mask", lambda: big[big > 0.3]
    yield "getitem large mask sparse", lambda: big[big > 2.5]
    yield "getitem large mask dense", lambda: big[big > -3.0]
    yield "getitem large 2d mask", lambda: M[M > 0]
    yield "getitem large row mask", lambda: M[M[:, 0] > 0]
    yield "getitem large col mask", lambda: M[:, M[0] > 0]
    yield "getitem large cols", lambda: M[:, idx[:500] % 1000]
    yield "getitem large rows", lambda: M[idx[:300] % 1000]
    yield "getitem large F cols", lambda: MF[:, idx[:500] % 1000]
    yield "getitem large F rows", lambda: MF[idx[:300] % 1000]
    yield "getitem large 3d mid", lambda: T3[:, :, idx[:40] % 70]
    yield "getitem large 3d pair", lambda: T3[idx[:1000] % 80, idx[1000:2000] % 60]
    yield "getitem large leading int", lambda: T3[5, idx[:50] % 60]
    yield "getitem large basic view", lambda: M[::-3, 7::2].copy()
    yield "getitem large oob err", lambda: big[idx + 999_999]


def _get(x, key):
    return x[key]


# ---------------------------------------------------------------- setitem


def setitem(c):
    A, np, DT = c.A, c.np, c.DT
    xf = c.S("f", (4, 5, 6), 31)
    xi = c.S("i", (4, 5), 32)
    xb = c.S("b", (4, 5), 33)
    xc = c.S("c", (4, 5), 34)
    xF = _rev(A, c.S("f", (6, 5, 4), 35))
    yield_cases = {
        "scalar int idx": (xf, 0, 5.0),
        "scalar elem": (xf, (1, 2, 3), 2.5),
        "slice bcast row": (xf, (slice(None), 1), A.arange(6.0)),
        "neg step": (xf, (Ellipsis, slice(None, None, -2)), -1.0),
        "full list": (xf, (0, 0), [1, 2, 3, 4, 5, 6]),
        "numpy value": (xf, 1, np.ones((5, 6))),
        "ellipsis": (xf, Ellipsis, 0.5),
        "newaxis": (xf, (None, 0), 9.0),
        "int value into float": (xf, 2, 7),
        "bool value into float": (xf, 2, True),
        "int arr value": (xf, 3, A.arange(6)),
        "adv [0,2]": (xf, [0, 2], 7.0),
        "adv repeated": (xf, [0, 0, 0], 3.0),
        "mask -1": (xf, "MASK", -1.0),
        "mask values": (xf, "MASK", "MASKVALS"),
        "adv pair": (xf, ([0, 1], [1, 2]), A.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        "adv cols": (xf, (slice(None), [1, 3]), -2.0),
        "adv bcast": (xf, ([[0], [2]], [1, 3]), 0.0),
        "empty adv": (xf, [], 1.0),
        "nan": (xf, 0, NAN),
        "inf": (xf, (0, 0, 0), -INF),
        "neg zero": (xf, (0, 0), -0.0),
        # dtype rules
        "int float value err": (xi, 0, 1.5),
        "int float arr err": (xi, 0, A.asarray([1.5] * 5)),
        "int bool ok": (xi, 0, True),
        "int too big err": (xi, 0, 2**63),
        "int max": (xi, (0, 0), INT_MAX),
        "int min": (xi, (0, 1), INT_MIN),
        "float complex err": (xf, 0, 1j),
        "bool int err": (xb, 0, 2),
        "bool true": (xb, 0, True),
        "bool 1 err": (xb, (0, 0), 1),
        "complex float ok": (xc, 0, 2.5),
        "complex value": (xc, (1, 1), 1 - 1j),
        "complex nan": (xc, 2, complex(NAN, -0.0)),
        # broadcasting
        "strict bcast err": (xf, (0, 0), A.ones((1, 6))),
        "bcast mismatch err": (xf, 0, A.ones(4)),
        "elem with array err": (xf, (1, 2, 3), A.asarray([1.0])),
        "elem with list err": (xf, (1, 2, 3), [1.0]),
        "str value err": (xf, 0, "a"),
        "None value err": (xf, 0, None),
        # index errors
        "adv oob err": (xf, [0, 9], 1.0),
        "oob int err": (xf, 4, 1.0),
        "sep adv err": (xf, ([0, 1], slice(None), [0, 1]), 1.0),
        "float idx err": (xf, 1.5, 1.0),
        "mask shape err": (xf, "BADMASK", 1.0),
        # layouts
        "F row": (xF, 0, 3.0),
        "F col adv": (xF, (slice(None), [0, 4]), 1.0),
        "F mask": (xF, "FMASK", 0.0),
    }
    for nm, (x, key, val) in yield_cases.items():

        def thunk(x=x, key=key, val=val):
            y = _rev(A, _rev(A, x).copy()) if x is xF else x.copy()  # keep the F layout of xF
            if isinstance(key, str):
                key = {"MASK": y > 0.5, "BADMASK": A.ones((5, 4), dtype=DT["b"]), "FMASK": y < 0}[key]
            if isinstance(val, str):
                val = A.arange(float(A.sum(key)))  # "MASKVALS"
            try:
                y[key] = val
            except Exception as e:  # noqa: BLE001 -- the error and the untouched array
                return (e, y)
            return y

        yield f"setitem {nm}", thunk
    # repeated advanced indices: the last write wins
    yield (
        "setitem adv repeated last wins",
        lambda: _set_copy(c.lit([0.0] * 5), [1, 3, 1, 1], c.lit([1.0, 2.0, 3.0, 4.0])),
    )
    yield "setitem adv repeated 2d", lambda: _set_copy(xf, [0, 0, 1], A.asarray(np.arange(90.0).reshape(3, 5, 6)))
    # overlap: value read before writing
    yield "setitem overlap shift right", lambda: _self_assign(c.S("f", (10,), 36), slice(1, None), slice(None, -1))
    yield "setitem overlap shift left", lambda: _self_assign(c.S("f", (10,), 36), slice(None, -1), slice(1, None))
    yield "setitem overlap reverse", lambda: _self_assign(c.S("f", (10,), 36), slice(None), slice(None, None, -1))
    yield "setitem overlap adv value", lambda: _self_assign_adv(c.S("f", (10,), 37))
    yield "setitem overlap transposed", lambda: _self_assign_T(c.S("f", (5, 5), 38))
    yield "setitem adv oob leaves x err", lambda: _set_copy(c.lit([0.0] * 5), [0, 1, 7], 1.0)
    yield "setitem readonly numpy err", lambda: _setitem(c.L.arr(c.gen("f", (4,), 39)), 0, 1.0)
    yield "setitem real view of complex", lambda: _set_real(A, xc)
    yield "setitem imag of real err", lambda: _setitem(A.imag(c.S("f", (3,), 40)), 0, 1.0)
    yield "setitem iteration views write", lambda: _iter_write(c.S("f", (3, 4), 41))
    # large scatter
    big = c.S("f", (1_000_000,), 42)
    idx = c.S("I", (400_000,), 43) % 1_000_000
    M = c.S("f", (1000, 1000), 44)
    yield "setitem large mask scalar", lambda: _set_copy(big, big > 0.3, -1.0)
    yield "setitem large mask values", lambda: _set_copy_maskvals(big)
    yield "setitem large scatter scalar", lambda: _set_copy(big, idx, 0.0)
    yield "setitem large scatter values", lambda: _set_copy(big, idx, A.arange(400_000.0))
    yield "setitem large scatter int values", lambda: _set_copy(big, idx, A.arange(400_000))
    yield "setitem large cols", lambda: _set_copy(M, (slice(None), idx[:300] % 1000), 5.0)
    yield "setitem large F cols", lambda: _set_copy_F(M, idx)
    yield "setitem large 2d mask", lambda: _set_copy(M, M < -1.0, M[M < -1.0] * 2.0)
    yield "setitem large slice bcast", lambda: _set_copy(M, (slice(None, None, -2), slice(None)), A.arange(1000.0))
    yield "setitem large int to float", lambda: _set_copy(M, slice(None), c.S("I", (1000, 1000), 45))
    yield "setitem large transposed self", lambda: _self_assign_T(c.S("f", (1000, 1000), 46))


def _set_copy(x, key, val):
    y = x.copy()
    try:
        y[key] = val
    except Exception as e:  # noqa: BLE001
        return (e, y)
    return y


def _set_copy_maskvals(x):
    y = x.copy()
    mask = y > 0.3
    y[mask] = y[mask] * -1.0 + 0.5
    return y


def _set_copy_F(M, idx):
    y = M.T.copy().T
    y[:, idx[:300] % 1000] = 1.0
    return y


def _self_assign(x, dst, src):
    y = x.copy()
    y[dst] = y[src]
    return y


def _self_assign_adv(x):
    y = x.copy()
    y[[0, 2, 4, 6]] = y[1:5]
    return y


def _self_assign_T(x):
    y = x.copy()
    y[...] = y.T
    return y


def _set_real(A, xc):
    y = xc.copy()
    r = A.real(y)
    r[...] = 0.0
    A.imag(y)[0] = -1.0
    return (y, r)


def _iter_write(x):
    y = x.copy()
    for row in y:
        row[0] = 99.0
    return y


# ---------------------------------------------------------------- diag, block


def diag_block(c):
    A, np = c.A, c.np
    # diag: construction only
    for k in KINDS:
        x = c.S(k, (5,), 60)
        yield f"diag {k}", (lambda x=x: A.diag(x))
        for off in (2, -3):
            yield f"diag {k} k={off}", (lambda x=x, off=off: A.diag(x, k=off))
    yield "diag special", lambda: A.diag(c.S("n", (40,), 61), k=-1)
    yield "diag strided", lambda: A.diag(c.S("f", (11,), 62)[::-2], k=1)
    yield "diag empty", lambda: A.diag(c.S("f", (0,), 62))
    yield "diag empty k=3", lambda: A.diag(c.S("i", (0,), 62), k=3)
    yield "diag 1elem k=-1", lambda: A.diag(c.lit([2.5]), k=-1)
    yield "diag large", lambda: A.diag(c.S("f", (2000,), 63))
    yield "diag large k=500", lambda: A.diag(c.S("c", (600,), 63), k=500)
    yield "diag bcast view", lambda: A.diag(A.broadcast_to(c.S("f", (1,), 64), (4,)))
    yield "diag 2d err", lambda: A.diag(c.S("f", (3, 3), 65))
    yield "diag 3d err", lambda: A.diag(c.S("f", (2, 2, 2), 65))
    yield "diag list err", lambda: A.diag([1.0, 2.0])
    yield "diag numpy err", lambda: A.diag(np.ones(3))
    yield "diag scalar err", lambda: A.diag(2.0)
    yield "diag k float err", lambda: A.diag(c.S("f", (3,), 66), k=1.0)
    yield "diag k bool err", lambda: A.diag(c.S("f", (3,), 66), k=True)
    yield "diag k None err", lambda: A.diag(c.S("f", (3,), 66), k=None)
    yield "diag k int min err", lambda: A.diag(c.S("f", (3,), 66), k=INT_MIN)
    yield "diag k overflow err", lambda: A.diag(c.S("f", (3,), 66), k=2**63)
    yield "diag k too big err", lambda: A.diag(c.S("f", (3,), 66), k=INT_MAX)
    yield "diag k positional err", lambda: A.diag(c.S("f", (3,), 66), 1)
    yield "diag sparse err", lambda: A.diag(A.sparse.csr_array(c.S("f", (3, 3), 65)))
    # block: numpy's nested lists
    p, q = c.S("f", (2, 3), 70), c.S("f", (2, 1), 71)
    r, t = c.S("f", (1, 3), 72), c.S("f", (1, 1), 73)
    yield "block 2x2", lambda: A.block([[p, q], [r, t]])
    for ka in KINDS:
        for kb in KINDS:
            a, b = c.S(ka, (2, 2), 74), c.S(kb, (2, 3), 75)
            yield f"block {ka}{kb}", (lambda a=a, b=b: A.block([a, b]))
    yield "block 1d", lambda: A.block([c.lit([1, 2]), c.lit([3.5])])
    yield "block 1d rows", lambda: A.block([[c.lit([1, 2, 3])], [c.lit([4.0, 5.0, 6.0])]])
    row1, row2 = [c.lit([1.0, 2.0]), c.lit([3.0])], [c.S("f", (2, 2), 76), c.S("f", (2, 1), 77)]
    yield "block 1d and 2d", lambda: A.block([row1, row2])
    yield "block 3d depth 2", lambda: A.block([[c.S("f", (2, 2, 3), 78), c.S("i", (2, 2, 1), 79)]])
    yield "block 2d depth 3", lambda: A.block([[[p], [r]], [[p], [r]]])
    yield "block single array", lambda: A.block(p)
    yield "block single list", lambda: A.block([p])
    yield "block tuple", lambda: A.block(((p, q), (r, t)))
    yield "block layouts", lambda: A.block([[c.S("f", (3, 2), 90).T, q[::-1]], [r[:, ::-1], t]])
    yield "block F-order", lambda: A.block([c.S("f", (30, 40), 80).T, c.S("f", (40, 10), 81)])
    yield "block bcast view", lambda: A.block([A.broadcast_to(c.S("f", (3,), 82), (2, 3)), q])
    yield "block special", lambda: A.block([[c.S("n", (3, 4), 83)], [c.S("n", (2, 4), 84)]])
    e20, e00, e01 = c.S("f", (2, 0), 85), c.S("f", (0, 0), 85), c.S("f", (0, 1), 85)
    yield "block empty parts", lambda: A.block([[e20, q], [e00, e01]])
    big = [[c.S("f", (300, 200), 86), c.S("i", (300, 100), 87)], [c.S("f", (100, 300), 88)]]
    yield "block large", lambda: A.block(big)
    yield "block empty list err", lambda: A.block([])
    yield "block nested empty err", lambda: A.block([[p], []])
    yield "block uneven depth err", lambda: A.block([[p, q], r])
    yield "block uneven depth deep err", lambda: A.block([[[p]], [q]])
    yield "block shape mismatch err", lambda: A.block([[p, c.S("f", (3, 1), 89)]])
    yield "block row mismatch err", lambda: A.block([[p], [c.S("f", (1, 2), 89)]])
    yield "block scalar leaf err", lambda: A.block([p, 1.0])
    yield "block numpy leaf err", lambda: A.block([p, np.ones((2, 1))])
    yield "block sparse leaf err", lambda: A.block([p, A.sparse.csr_array(q)])
    yield "block not a list err", lambda: A.block(3)
    yield "block numpy arg err", lambda: A.block(np.ones((2, 2)))
    yield "block too many dims err", lambda: A.block(_nest(p, 33))
    yield "block 32 dims", lambda: A.block(_nest(c.lit([1.0]), 32))


def _nest(x, depth):
    for _ in range(depth):
        x = [x]
    return x


# ---------------------------------------------------------------- elementwise


def unary(c):
    A = c.A
    for name in UNARY:
        fn = getattr(A, name)
        for k in KINDS:
            x = c.lit(SPECIAL[k], k)
            yield f"ufunc {name} {k} special", (lambda fn=fn, x=x: fn(x))
            yield f"ufunc {name} {k} special reversed", (lambda fn=fn, x=x: fn(x[::-1]))
        for k, shape in (("f", (1_000_000,)), ("c", (200_000,)), ("i", (200_000,)), ("b", (200_000,))):
            x = c.S(k, shape, 50)
            yield f"ufunc {name} {k} large", (lambda fn=fn, x=x: fn(x))
        kf = {"logical_not": "b", "bitwise_invert": "i"}.get(name, "f")  # a dtype the function takes
        xs = c.S(kf, (300, 400), 51)
        yield f"ufunc {name} {kf} large F-order", (lambda fn=fn, x=xs: fn(x.T))
        yield f"ufunc {name} {kf} large strided", (lambda fn=fn, x=xs: fn(x[::-2, 1::3]))
        yield f"ufunc {name} {kf} 5d", (lambda fn=fn, x=c.S(kf, (2, 3, 1, 4, 2), 52): fn(x))  # noqa: B008 -- inputs bound when the case is made
        yield f"ufunc {name} {kf} empty", (lambda fn=fn, x=c.S(kf, (0, 3), 52): fn(x))  # noqa: B008 -- inputs bound when the case is made
        yield f"ufunc {name} {kf} 1elem", (lambda fn=fn, x=c.S(kf, (1, 1), 52): fn(x))  # noqa: B008 -- inputs bound when the case is made
        yield f"ufunc {name} {kf} bcast view", (lambda fn=fn, x=A.broadcast_to(c.S(kf, (4,), 53), (3, 4)): fn(x))  # noqa: B008 -- inputs bound when the case is made
        for k, s in PYSCALAR.items():
            yield (
                f"ufunc {name} pyscalar {k}" + (" err" if name in ("real", "imag") else ""),
                (lambda fn=fn, s=s: fn(s)),
            )
        yield f"ufunc {name} list err", (lambda fn=fn: fn([1.0, 2.0]))
        yield f"ufunc {name} numpy err", (lambda fn=fn: fn(c.np.ones(2)))
    yield (
        "ufunc round half even",
        lambda: A.round(c.lit([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, 1e16 + 1, -0.0, 4.5000001])),
    )
    yield "ufunc round complex half even", lambda: A.round(c.lit([0.5 + 1.5j, 2.5 - 0.5j]))
    yield "ufunc abs int min wraps", lambda: A.abs(c.lit([INT_MIN, -1, 0]))
    yield "ufunc negative int min wraps", lambda: A.negative(c.lit([INT_MIN, INT_MAX]))
    yield "ufunc square int overflow", lambda: A.square(c.lit([2**32, 3037000500]))
    yield "ufunc sign nan complex", lambda: A.sign(c.lit([complex(NAN, 0), 3 + 4j, 0j, complex(INF, 1)]))
    yield "ufunc real is self", lambda: (lambda x: A.real(x) is x)(c.lit([1.0]))
    yield "ufunc real complex view", lambda: _view(A.real(c.S("c", (3, 4), 54)), None).strides
    yield "ufunc imag complex strided", lambda: A.imag(c.S("c", (6, 4), 55)[::-2, 1::2])
    yield "ufunc imag of real readonly err", lambda: _setitem(A.imag(c.lit([1, 2])), 0, 5)
    yield "ufunc bitwise_invert bool is not", lambda: A.bitwise_invert(c.lit([True, False]))


def binary(c):
    A = c.A
    for name in BINARY:
        fn = getattr(A, name)
        for ka in KINDS:
            for kb in KINDS:
                a = c.lit(SPECIAL[ka], ka)[:, None] if ka != "c" else c.lit(SPECIAL[ka], ka)[::3, None]
                b = c.lit(SPECIAL[kb], kb)[None, :]
                yield f"ufunc {name} {ka}{kb} grid", (lambda fn=fn, a=a, b=b: fn(a, b))
        for ka, kb in (("f", "f"), ("i", "i"), ("i", "f"), ("b", "b"), ("c", "c"), ("f", "c")):
            sa = (1_000_000,) if "c" not in ka + kb else (200_000,)
            a, b = c.S(ka, sa, 60), c.S(kb, sa, 61)
            if name in ("pow",) and ka == "i" and kb == "i":
                b = c.S("i", sa, 61) % 7
            yield f"ufunc {name} {ka}{kb} large", (lambda fn=fn, a=a, b=b: fn(a, b))
        # a dtype the function takes, and a second (mixed) one
        kf = "b" if name.startswith("logical") else "i" if name.startswith("bitwise") else "f"
        k2 = {"f": "i", "i": "b", "b": "b"}[kf] if "shift" not in name else "i"
        M = c.S(kf, (1000, 1000), 62)
        v = c.S(kf, (1000,), 63)
        Mi = c.S(k2, (1000, 1000), 64)
        sc = {"f": 0.5, "i": 3, "b": True}[kf]
        yield f"ufunc {name} large bcast row", (lambda fn=fn, M=M, v=v: fn(M, v))
        yield f"ufunc {name} large bcast col", (lambda fn=fn, M=M, v=v: fn(v[:, None], M))
        yield f"ufunc {name} large C x F", (lambda fn=fn, M=M: fn(M, M.T))
        yield f"ufunc {name} large F x F mixed", (lambda fn=fn, M=M, Mi=Mi: fn(M.T, Mi.T))
        yield f"ufunc {name} large strided mixed", (lambda fn=fn, M=M, Mi=Mi: fn(M[::-2, ::3], Mi[1::2, ::-3]))
        yield f"ufunc {name} large scalar", (lambda fn=fn, M=M, sc=sc: fn(M, sc))
        yield f"ufunc {name} large scalar reflected", (lambda fn=fn, M=M, sc=sc: fn(sc, M))
        yield (
            f"ufunc {name} 5d bcast",
            (lambda fn=fn, kf=kf, k2=k2: fn(c.S(k2, (2, 1, 3, 1, 2), 65), c.S(kf, (4, 1, 1, 2), 66))),
        )
        yield f"ufunc {name} empty bcast", (lambda fn=fn, kf=kf: fn(c.S(kf, (0, 3), 67), c.S(kf, (3,), 68)))
        yield f"ufunc {name} bcast err", (lambda fn=fn: fn(c.S("f", (3,), 69), c.S("f", (4,), 69)))
        yield f"ufunc {name} list err", (lambda fn=fn: fn(c.lit([1.0]), [1.0]))
        yield f"ufunc {name} numpy err", (lambda fn=fn: fn(c.np.ones(1), c.lit([1.0])))
        yield f"ufunc {name} str err", (lambda fn=fn: fn(c.lit([1.0]), "a"))
        yield f"ufunc {name} big int", (lambda fn=fn: fn(c.lit([1]), 2**64))
        yield f"ufunc {name} keyword err", (lambda fn=fn: fn(x1=c.lit([1.0]), x2=c.lit([1.0])))
        for ka in "if":
            a = c.S(ka, (3, 4), 70)
            for ks, s in PYSCALAR.items():
                yield f"ufunc {name} {ka} op pyscalar {ks}", (lambda fn=fn, a=a, s=s: fn(a, s))
                yield f"ufunc {name} pyscalar {ks} op {ka}", (lambda fn=fn, a=a, s=s: fn(s, a))
        for ka in "bc":
            a = c.S(ka, (3, 4), 71)
            s = PYSCALAR[ka]
            yield f"ufunc {name} {ka} op pyscalar {ka}", (lambda fn=fn, a=a, s=s: fn(a, s))
            yield f"ufunc {name} pyscalar {ka} op {ka}", (lambda fn=fn, a=a, s=s: fn(s, a))
        for s1, s2 in ((7, 0), (2**62, 2**62), (True, False), (-7, 2), (2.5, -0.0), (1j, 2), (3, 2.0), (INT_MIN, -1)):
            yield f"ufunc {name} pyscalars {s1!r},{s2!r}", (lambda fn=fn, s1=s1, s2=s2: fn(s1, s2))
    # documented integer edge rules, spelled out
    i = c.lit([7, -7, 0, INT_MIN, INT_MAX, 1, -1])
    yield "int floor_divide by 0", lambda: A.floor_divide(i, 0)
    yield "int remainder by 0", lambda: A.remainder(i, 0)
    yield "int floor_divide min by -1", lambda: A.floor_divide(i, -1)
    yield "int remainder by -3", lambda: A.remainder(i, -3)
    yield "int pow negative exponent", lambda: A.pow(i, -1)
    yield "int pow big wraps", lambda: A.pow(i, 63)
    yield "int add wraps", lambda: A.add(i, INT_MAX)
    yield "int multiply wraps", lambda: A.multiply(i, 2**40)
    yield "int divide is float", lambda: A.divide(i, 2)
    yield "int shifts", lambda: [A.bitwise_left_shift(i, s) for s in (0, 1, 63, 64, 100, -1)]
    yield "int right shifts", lambda: [A.bitwise_right_shift(i, s) for s in (0, 1, 63, 64, 100, -1)]
    yield "bool add is int", lambda: A.add(c.lit([True, True, False]), c.lit([True, False, False]))
    yield "bool shift err", lambda: A.bitwise_left_shift(c.lit([True]), 1)
    yield "float floor_divide by 0", lambda: A.floor_divide(c.lit([1.0, -1.0, 0.0, NAN, INF]), 0.0)
    yield (
        "float remainder signs",
        lambda: A.remainder(c.lit([5.5, -5.5, 5.5, -5.5, -0.0, 0.0]), c.lit([2.0, 2.0, -2.0, -2.0, 3.0, -3.0])),
    )
    yield (
        "maximum nan and signed zero",
        lambda: (
            A.maximum(c.lit([NAN, 1.0, -0.0, 0.0]), c.lit([1.0, NAN, 0.0, -0.0])),
            A.minimum(c.lit([NAN, 1.0, -0.0, 0.0]), c.lit([1.0, NAN, 0.0, -0.0])),
        ),
    )
    yield "complex compare err", lambda: A.less(c.lit([1j]), c.lit([2j]))
    yield "pow complex zero", lambda: A.pow(c.lit([0j, 0j, 0j]), c.lit([0j, 1 + 0j, -1 + 0j]))


def scalars_ops(c):
    """Python-scalar-only calls of a few functions (Z1: scalar in, scalar out)."""
    A = c.A
    yield "pyscalar sqrt 4", lambda: A.sqrt(4)
    yield "pyscalar sqrt -1.0", lambda: A.sqrt(-1.0)
    yield "pyscalar sqrt -1j", lambda: A.sqrt(-1 + 0j)
    yield "pyscalar where", lambda: A.where(True, 1, 2.5)
    yield "pyscalar clip", lambda: A.clip(5, min=0, max=3)
    yield "pyscalar equal int float", lambda: A.equal(1, 1.0)
    yield "pyscalar isnan nan", lambda: A.isnan(NAN)
    yield "pyscalar real complex err", lambda: A.real(3 - 4j)
    yield "pyscalar abs complex", lambda: A.abs(3 - 4j)


def outs(c):
    A, DT = c.A, c.DT
    x, y = c.S("f", (3, 4), 80), c.S("f", (3, 4), 81)
    xi = c.S("i", (3, 4), 82)
    yield "out add returns out", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((3, 4)))
    yield "out sin", lambda: _out(lambda o: A.sin(x, out=o), A.zeros((3, 4)))
    yield "out int result into float", lambda: _out(lambda o: A.add(xi, xi, out=o), A.zeros((3, 4)))
    yield (
        "out int result into complex",
        lambda: _out(lambda o: A.multiply(xi, 3, out=o), A.zeros((3, 4), dtype=DT["c"])),
    )
    yield "out float into int err", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((3, 4), dtype=DT["i"]))
    yield "out compare into int", lambda: _out(lambda o: A.less(x, y, out=o), A.zeros((3, 4), dtype=DT["i"]))
    yield "out compare into bool", lambda: _out(lambda o: A.less(x, y, out=o), A.zeros((3, 4), dtype=DT["b"]))
    yield "out shape mismatch err", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((4, 3)))
    yield "out bcast shape err", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((4,)))
    yield "out bigger shape", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((2, 3, 4)))
    yield "out readonly err", lambda: A.add(x, y, out=A.broadcast_to(A.zeros(4), (3, 4)))
    yield "out not array err", lambda: A.add(x, y, out=[0.0] * 12)
    yield "out numpy err", lambda: A.add(x, y, out=c.np.zeros((3, 4)))
    yield "out None", lambda: A.add(x, y, out=None)
    yield "out aliases input", lambda: _alias(x, lambda z: A.add(z, 1.0, out=z))
    yield "out aliases input 2", lambda: _alias(x, lambda z: A.multiply(2.0, z, out=z))
    yield "out overlap shifted", lambda: _alias(c.S("f", (20,), 83), lambda z: A.add(z[:-1], 1.0, out=z[1:]))
    yield "out overlap reversed", lambda: _alias(c.S("f", (20,), 84), lambda z: A.negative(z, out=z[::-1]))
    yield "out overlap transposed", lambda: _alias(c.S("f", (30, 30), 85), lambda z: A.add(z, z.T, out=z))
    yield "out F-order", lambda: _out(lambda o: A.add(x, y, out=o), A.zeros((4, 3)).T)
    yield "out strided", lambda: _out(lambda o: A.exp(x, out=o), A.zeros((6, 8))[::2, ::-2])
    yield "out large", lambda: _out(lambda o: A.add(A.ones((1000, 1000)), 2.0, out=o), A.zeros((1000, 1000)).T)
    yield "out large overlap", lambda: _alias(c.S("f", (1_000_000,), 86), lambda z: A.add(z[1:], z[:-1], out=z[:-1]))
    yield "out complex abs", lambda: _out(lambda o: A.abs(c.S("c", (3, 4), 87), out=o), A.zeros((3, 4)))
    yield "out real", lambda: _out(lambda o: A.real(c.S("c", (2, 2), 88), out=o), A.zeros((2, 2)))
    yield "out imag of real", lambda: _out(lambda o: A.imag(c.S("f", (2, 2), 88), out=o), A.ones((2, 2)))
    yield "out real of real", lambda: _out(lambda o: A.real(c.S("i", (2, 2), 88), out=o), A.ones((2, 2), dtype=DT["i"]))
    yield "out pyscalars", lambda: _out(lambda o: A.add(1, 2, out=o), A.zeros(1))
    yield "out positional err", lambda: A.add(x, y, A.zeros((3, 4)))


def _out(call, o):
    try:
        r = call(o)
    except Exception as e:  # noqa: BLE001
        return (e, o)
    return (r is o, o)


def _alias(x, call):
    z = x.copy()
    r = call(z)
    return (r, z)


def clip_where(c):
    A, DT = c.A, c.DT
    f = c.lit(F_SP, "f")
    i = c.lit(I_SP, "i")
    for nm, kw in {
        "min": dict(min=-1.0),
        "max": dict(max=1.0),
        "both": dict(min=-1.0, max=1.0),
        "none": {},
        "min>max": dict(min=2.0, max=-2.0),
        "nan min": dict(min=NAN),
        "nan max": dict(max=NAN),
        "-0.0 min": dict(min=-0.0),
        "0.0 max": dict(max=0.0),
        "int bounds": dict(min=-1, max=1),
        "bool bound": dict(min=False),
        "inf": dict(min=-INF, max=INF),
    }.items():
        yield f"clip f {nm}", (lambda kw=kw: A.clip(f, **kw))
    for nm, kw in {
        "both": dict(min=-7, max=63),
        "float min err": dict(min=0.5),
        "bool": dict(min=True),
        "big": dict(min=INT_MIN, max=INT_MAX),
        "complex err": dict(max=1j),
        "min>max": dict(min=5, max=1),
    }.items():
        yield f"clip i {nm}", (lambda kw=kw: A.clip(i, **kw))
    yield "clip b", lambda: A.clip(c.lit([True, False]), min=True)
    yield "clip b int err", lambda: A.clip(c.lit([True, False]), min=0)
    yield "clip c", lambda: A.clip(c.lit([1 + 1j, -1j]), min=0.0)
    yield (
        "clip array bounds",
        lambda: A.clip(c.S("f", (3, 4), 90), min=c.S("f", (4,), 91), max=c.S("f", (3, 1), 92) + 2.0),
    )
    yield "clip array bound bcast err", lambda: A.clip(c.S("f", (3, 4), 90), min=c.S("f", (5,), 91))
    yield "clip int array bound float err", lambda: A.clip(c.S("i", (3, 4), 90), min=c.S("f", (4,), 91))
    yield "clip large", lambda: A.clip(c.S("f", (1_000_000,), 93), min=-0.5, max=0.5)
    yield "clip large F", lambda: A.clip(c.S("f", (1000, 1000), 94).T, min=c.S("f", (1000,), 95), max=1.0)
    yield "clip out", lambda: _out(lambda o: A.clip(f, min=0.0, out=o), A.zeros(len(F_SP)))
    yield "clip list err", lambda: A.clip([1.0], min=0.0)
    yield "clip positional err", lambda: A.clip(f, 0.0, 1.0)

    cond = c.S("b", (3, 4), 96)
    for ka in KINDS:
        for kb in KINDS:
            yield f"where {ka}{kb}", (lambda ka=ka, kb=kb: A.where(cond, c.S(ka, (3, 4), 97), c.S(kb, (4,), 98)))
    for ks, s in PYSCALAR.items():
        yield f"where scalar {ks}", (lambda s=s: A.where(cond, s, c.S("i", (3, 4), 99)))
        yield f"where scalars {ks}", (lambda s=s: A.where(cond, s, 0))
    yield "where nan", lambda: A.where(c.lit([True, False, True]), NAN, -0.0)
    yield "where 3-way bcast", lambda: A.where(c.S("b", (2, 1, 4), 100), c.S("f", (3, 1), 101), c.S("i", (4,), 102))
    yield "where int cond err", lambda: A.where(c.S("i", (3, 4), 103), 1.0, 0.0)
    yield "where float cond err", lambda: A.where(c.S("f", (3, 4), 103), 1.0, 0.0)
    yield "where bcast err", lambda: A.where(cond, c.S("f", (5,), 104), 0.0)
    yield "where list err", lambda: A.where(cond, [1.0], 0.0)
    yield "where large", lambda: A.where(c.S("b", (1_000_000,), 105), c.S("f", (1_000_000,), 106), 0.0)
    yield (
        "where large F mixed",
        lambda: A.where(c.S("b", (1000, 1000), 107).T, c.S("i", (1000, 1000), 108), c.S("f", (1000,), 109)),
    )
    yield "where out", lambda: _out(lambda o: A.where(cond, 1.0, 2.0, out=o), A.zeros((3, 4)))
    yield "where out dtype err", lambda: _out(lambda o: A.where(cond, 1.5, 2.0, out=o), A.zeros((3, 4), dtype=DT["i"]))


# ---------------------------------------------------------------- operators


def operators(c):
    A = c.A
    arrs = {k: c.S(k, (3, 4), 110 + n) for n, k in enumerate(KINDS)}
    vecs = {k: c.S(k, (4,), 120 + n) for n, k in enumerate(KINDS)}
    for on, op in BINOPS.items():
        for ka in KINDS:
            for kb in KINDS:
                yield f"op {on} {ka}{kb}", (lambda op=op, a=arrs[ka], b=vecs[kb]: op(a, b))
            for ks, s in PYSCALAR.items():
                yield f"op {on} {ka} pyscalar {ks}", (lambda op=op, a=arrs[ka], s=s: op(a, s))
                yield f"op {on} pyscalar {ks} {ka} reflected", (lambda op=op, a=arrs[ka], s=s: op(s, a))
        tail = "" if on in ("eq", "ne") else " err"  # == and != fall back to identity
        yield f"op {on} numpy right{tail}", (lambda op=op: op(arrs["f"], c.np.ones(4)))
        yield f"op {on} numpy left{tail}", (lambda op=op: op(c.np.ones(4), arrs["f"]))
        yield f"op {on} list{tail}", (lambda op=op: op(arrs["i"], [1, 2, 3, 4]))
        yield f"op {on} list reflected{tail}", (lambda op=op: op([1, 2, 3, 4], arrs["i"]))
        yield f"op {on} None{tail}", (lambda op=op: op(arrs["i"], None))
        kl = "I" if on in ("and", "or", "xor", "lshift", "rshift") else "f"
        yield (
            f"op {on} large",
            (
                lambda op=op, kl=kl: op(
                    c.S(kl, (1000, 1000), 130),
                    c.S(kl, (1000, 1000), 131).T % 70 if kl == "I" else c.S(kl, (1000, 1000), 131).T,
                )
            ),
        )
        yield f"op {on} large int", (lambda op=op: op(c.S("I", (1_000_000,), 132), 7))
    for on, op in INPLACE.items():
        for ka in KINDS:
            for kb in KINDS:
                yield f"iop {on} {ka}{kb}", (lambda op=op, a=arrs[ka], b=vecs[kb]: _inplace(op, a, b))
            for ks, s in PYSCALAR.items():
                yield f"iop {on} {ka} pyscalar {ks}", (lambda op=op, a=arrs[ka], s=s: _inplace(op, a, s))
        yield f"iop {on} bcast to self err", (lambda op=op: _inplace(op, vecs["f"], arrs["f"]))
        yield f"iop {on} self alias", (lambda op=op: _inplace_self(op, arrs["i"]))
        yield (
            f"iop {on} large",
            (lambda op=op: _inplace(op, c.S("I", (1000, 1000), 133), c.S("I", (1000,), 134) % 5 + 1)),
        )
        yield f"iop {on} readonly err", (lambda op=op: op(A.broadcast_to(vecs["i"], (3, 4)), 1))
    M = c.S("f", (3, 4), 135)
    N = c.S("f", (4, 2), 136)
    yield "op matmul", lambda: M @ N
    yield "op matmul pyscalar err", lambda: M @ 2.0
    yield "op rmatmul pyscalar err", lambda: 2.0 @ M
    yield "op matmul numpy err", lambda: M @ c.np.ones((4, 2))
    yield "op matmul list err", lambda: M @ [[1.0]] * 4
    yield "op imatmul", lambda: _inplace(operator.imatmul, c.S("f", (4, 4), 137), c.S("f", (4, 4), 138))
    yield "op imatmul shape change err", lambda: _inplace(operator.imatmul, M, N)
    yield (
        "op imatmul int by float err",
        lambda: _inplace(operator.imatmul, c.S("i", (4, 4), 139), c.S("f", (4, 4), 138)),
    )
    for k, a in arrs.items():
        yield f"op neg {k}", (lambda a=a: -a)
        yield f"op pos {k}", (lambda a=a: +a)
        yield f"op abs {k}", (lambda a=a: abs(a))
        yield f"op invert {k}", (lambda a=a: ~a)
    sp = c.lit(F_SP, "f")
    yield "op eq special", lambda: (sp == sp, sp != sp, sp < 0.0, sp >= -0.0)
    cp = c.lit(C_SP, "c")
    yield "op eq complex special", lambda: (cp == cp, cp != 0)
    yield "op eq list is identity", lambda: (arrs["i"] == [1, 2], arrs["i"] != [1, 2])
    yield "op eq numpy is identity", lambda: (arrs["i"] == c.np.ones(4), arrs["i"] != c.np.ones(4))
    yield "op divmod err", lambda: divmod(arrs["i"], 2)
    yield "op chain", lambda: ((arrs["f"] * 2.0 + 1.0) / 3.0 - arrs["i"]) ** 2 // 0.5 % 1.25


def _inplace(op, a, b):
    y = a.copy()
    try:
        r = op(y, b)
    except Exception as e:  # noqa: BLE001
        return (e, y)
    return (r is y, y)


def _inplace_self(op, a):
    y = a.copy() % 5 + 1
    r = op(y, y)
    return (r is y, y)


# ---------------------------------------------------------------- python protocols and attributes


def protocols(c):
    A, np = c.A, c.np
    xs = {
        "f C": c.S("f", (3, 4), 140),
        "f F": c.S("f", (4, 3), 141).T,
        "f strided": c.S("f", (8, 9), 142)[::-3, 1::2],
        "i": c.S("i", (2, 3, 4), 143),
        "b": c.S("b", (5,), 144),
        "c": c.S("c", (2, 2), 145),
        "empty": c.S("f", (0, 3), 146),
        "empty1d": c.S("i", (0,), 146),
        "1elem": c.lit([1.5]),
        "5d": c.S("f", (2, 1, 3, 1, 2), 147),
        "bcast": A.broadcast_to(c.S("f", (3,), 148), (2, 3)),
    }
    for nm, x in xs.items():
        yield f"attrs {nm}", (lambda x=x: (x.dtype, x.shape, x.ndim, x.size, x.strides, x.itemsize, x.nbytes))
        yield (
            f"base {nm}",
            (lambda x=x: (x.base is None, x.copy().base is None, x[...].base is (x.base if x.base is not None else x))),
        )
        yield f"tolist {nm}", (lambda x=x: x.tolist())
        yield f"len {nm}", (lambda x=x: len(x))
        yield f"iter {nm}", (lambda x=x: list(x))
        yield f"proto repr {nm}", (lambda x=x: repr(x))
        yield f"proto str {nm}", (lambda x=x: str(x))
        yield f"numpy roundtrip {nm}", (lambda x=x: A.asarray(np.asarray(x)))
        yield (
            f"memoryview {nm}",
            (lambda x=x: (memoryview(x).format, memoryview(x).shape, memoryview(x).strides, memoryview(x).readonly)),
        )
        for conv in ("bool", "int", "float", "complex", "index", "hash", "in"):
            fn = {
                "bool": bool,
                "int": int,
                "float": float,
                "complex": complex,
                "index": operator.index,
                "hash": hash,
                "in": lambda x: 1.0 in x,
            }[conv]
            yield f"{conv}() {nm} err", (lambda fn=fn, x=x: fn(x))
    yield "tolist special f", lambda: c.lit(F_SP, "f").tolist()
    yield "tolist special c", lambda: c.lit(C_SP[:12], "c").tolist()
    yield "tolist special i", lambda: c.lit(I_SP, "i").tolist()
    yield "iter yields views", lambda: [r.base is not None for r in c.S("f", (3, 2), 149)]
    yield "iter 1d types", lambda: [type(v).__name__ for k in KINDS for v in c.lit(SPECIAL[k][:2], k)]
    yield "len keepdims result", lambda: len(A.sum(c.lit([1.0]), keepdims=True))
    yield "base final owner", lambda: (lambda x: x[1:][::2].T.base is x)(c.S("f", (6, 4), 150))
    yield "base of numpy import", lambda: A.asarray(c.gen("f", (3,), 151)).base is None
    yield "attrs 32d", lambda: (lambda x: (x.shape, x.strides, x.ndim))(A.zeros((1,) * 32))
    yield "attrs 33d err", lambda: A.zeros((1,) * 33)
    yield "setattr shape err", lambda: setattr(c.S("f", (2,), 152), "shape", (1, 2))
    yield "setattr dtype err", lambda: setattr(c.S("f", (2,), 152), "dtype", A.int64)
    yield "no array method", lambda: hasattr(c.S("f", (2,), 152), "flatten")


# ---------------------------------------------------------------- reductions


def reductions(c):
    A, DT = c.A, c.DT
    small = {k: c.S(k, (4, 5, 6), 160) for k in KINDS}
    small["t"] = c.S("t", (4, 5, 6), 161)  # ties for argmin/argmax
    small["n"] = c.S("n", (4, 5, 6), 162)  # nan, -0.0, inf
    axes_opts = {
        "None": dict(),
        "0": dict(axis=0),
        "1": dict(axis=1),
        "-1": dict(axis=-1),
        "(0,2)": dict(axis=(0, 2)),
        "(0,1,2)": dict(axis=(0, 1, 2)),
        "()": dict(axis=()),
        "keep": dict(keepdims=True),
        "1 keep": dict(axis=1, keepdims=True),
        "(2,0) keep": dict(axis=(2, 0), keepdims=True),
    }
    for fn_name in REDUCTIONS + NAN_REDUCTIONS:
        fn = getattr(A, fn_name)
        for k, x in small.items():
            for an, kw in axes_opts.items():
                if fn_name.endswith(("argmin", "argmax")) and isinstance(kw.get("axis"), tuple):
                    continue
                yield f"red {fn_name} {k} axis={an}", (lambda fn=fn, x=x, kw=kw: fn(x, **kw))
        yield f"red {fn_name} f F-order axis=0", (lambda fn=fn: fn(_rev(A, c.S("f", (6, 5, 4), 163)), axis=0))
        yield f"red {fn_name} f F-order axis=None", (lambda fn=fn: fn(_rev(A, c.S("f", (6, 5, 4), 163))))
        yield f"red {fn_name} f strided axis=1", (lambda fn=fn: fn(c.S("f", (8, 10, 12), 164)[::2, ::-2, 1::3], axis=1))
        yield f"red {fn_name} i strided", (lambda fn=fn: fn(c.S("i", (8, 10), 165)[::-3, ::2]))
        yield (
            f"red {fn_name} 5d axis=(1,3)"
            if not fn_name.endswith(("argmin", "argmax"))
            else f"red {fn_name} 5d axis=3",
            (
                lambda fn=fn, fn_name=fn_name: fn(
                    c.S("f", (2, 3, 2, 4, 2), 166), axis=(1, 3) if not fn_name.endswith(("argmin", "argmax")) else 3
                )
            ),
        )
        yield f"red {fn_name} bcast view", (lambda fn=fn: fn(A.broadcast_to(c.S("f", (5,), 167), (4, 5)), axis=0))
        yield f"red {fn_name} 1elem", (lambda fn=fn: fn(c.lit([[2.5]])))
        yield f"red {fn_name} complex special", (lambda fn=fn: fn(c.lit(C_SP, "c")))
        yield f"red {fn_name} float special", (lambda fn=fn: fn(c.lit(F_SP, "f")))
        yield f"red {fn_name} signed zeros", (lambda fn=fn: fn(c.lit([-0.0, 0.0, -0.0])))
        yield f"red {fn_name} neg zeros only", (lambda fn=fn: fn(c.lit([-0.0, -0.0])))
        yield f"red {fn_name} int extremes", (lambda fn=fn: fn(c.lit([INT_MAX, 1, INT_MIN, -1])))
        # empty inputs
        for k in "bifc":
            e1, e2 = c.S(k, (0,), 168), c.S(k, (3, 0), 168)
            yield f"red {fn_name} {k} empty (0,)", (lambda fn=fn, e=e1: fn(e))
            yield f"red {fn_name} {k} empty (3,0) axis=0", (lambda fn=fn, e=e2: fn(e, axis=0))
            yield f"red {fn_name} {k} empty (3,0) axis=1", (lambda fn=fn, e=e2: fn(e, axis=1))
            yield f"red {fn_name} {k} empty (3,0) keep", (lambda fn=fn, e=e2: fn(e, axis=1, keepdims=True))
        # large: above the parallel threshold
        for k, shape in (
            ("f", (1_000_000,)),
            ("I", (1_000_000,)),
            ("b", (1_000_000,)),
            ("c", (300_000,)),
            ("n", (1_000_000,)),
        ):
            big = c.S(k, shape, 170)
            yield f"red {fn_name} {k} large", (lambda fn=fn, x=big: fn(x))
            yield f"red {fn_name} {k} large strided", (lambda fn=fn, x=big: fn(x[::-3]))
        for k in "fIc":
            M = c.S(k, (1000, 1000) if k != "c" else (500, 400), 171)
            for ax in (0, 1):
                yield f"red {fn_name} {k} large 2d axis={ax}", (lambda fn=fn, M=M, ax=ax: fn(M, axis=ax))
                yield f"red {fn_name} {k} large 2d F axis={ax}", (lambda fn=fn, M=M, ax=ax: fn(M.T, axis=ax))
        T3 = A.permute_dims(c.S("f", (60, 70, 80), 172), (2, 0, 1))
        if not fn_name.endswith(("argmin", "argmax")):
            yield f"red {fn_name} large 3d axis=(0,2)", (lambda fn=fn: fn(T3, axis=(0, 2)))
        yield f"red {fn_name} large 3d axis=1 keep", (lambda fn=fn: fn(T3, axis=1, keepdims=True))
        yield f"red {fn_name} large tall axis=0", (lambda fn=fn: fn(c.S("f", (100_000, 3), 173), axis=0))
        yield f"red {fn_name} large wide axis=1", (lambda fn=fn: fn(c.S("f", (3, 100_000), 174), axis=1))
        # errors
        yield f"red {fn_name} axis oob err", (lambda fn=fn: fn(small["f"], axis=3))
        yield f"red {fn_name} axis -4 err", (lambda fn=fn: fn(small["f"], axis=-4))
        yield f"red {fn_name} axis float err", (lambda fn=fn: fn(small["f"], axis=1.0))
        yield f"red {fn_name} axis dup err", (lambda fn=fn: fn(small["f"], axis=(0, 0)))
        yield f"red {fn_name} axis list err", (lambda fn=fn: fn(small["f"], axis=[0]))
        yield f"red {fn_name} keepdims positional err", (lambda fn=fn: fn(small["f"], None, True))
        yield f"red {fn_name} list err", (lambda fn=fn: fn([1.0, 2.0]))
        yield f"red {fn_name} pyscalar err", (lambda fn=fn: fn(1.0))
        yield f"red method {fn_name}", (lambda fn_name=fn_name: getattr(small["f"], fn_name)(axis=1))
        yield f"red method {fn_name} none", (lambda fn_name=fn_name: getattr(small["i"], fn_name)())
    yield "red argmin axis tuple err", lambda: A.argmin(small["f"], axis=(0, 1))
    yield "red argmax axis tuple err", lambda: A.argmax(small["f"], axis=(0,))
    yield "red nanargmin axis tuple err", lambda: A.nanargmin(small["f"], axis=(0, 1))
    yield "red nanargmax axis tuple err", lambda: A.nanargmax(small["f"], axis=(0,))
    all_nan = c.lit([[NAN, 1.0, NAN], [NAN, NAN, NAN]])
    for fn_name in NAN_REDUCTIONS:
        fn = getattr(A, fn_name)
        yield f"red {fn_name} all-nan lane axis=1", (lambda fn=fn: fn(all_nan, axis=1))
        yield f"red {fn_name} all-nan lane axis=0", (lambda fn=fn: fn(all_nan, axis=0))
        yield f"red {fn_name} no nan is plain", (lambda fn=fn: fn(small["f"], axis=1))
        yield f"red {fn_name} complex nan", (lambda fn=fn: fn(c.lit([1 + 1j, complex(NAN, 0), complex(0, NAN), 2j])))
        yield f"red {fn_name} large mostly nan", (lambda fn=fn: fn(c.S("n", (1_000_000,), 183) * c.lit([NAN])))
    for fn_name in ("sum", "prod", "nansum", "nanprod"):
        fn = getattr(A, fn_name)
        for s in KINDS:
            for d in KINDS:
                yield f"red {fn_name} {s} dtype={d}", (lambda fn=fn, s=s, d=d: fn(small[s], dtype=DT[d]))
                yield f"red {fn_name} {s} dtype={d} axis=1", (lambda fn=fn, s=s, d=d: fn(small[s], axis=1, dtype=DT[d]))
        yield f"red {fn_name} large dtype=f from int", (lambda fn=fn: fn(c.S("I", (1_000_000,), 175), dtype=DT["f"]))
        yield (
            f"red {fn_name} large dtype=c from f",
            (lambda fn=fn: fn(c.S("f", (1000, 1000), 176), axis=0, dtype=DT["c"])),
        )
        yield f"red {fn_name} dtype python err", (lambda fn=fn: fn(small["f"], dtype=float))
    for fn_name in ("var", "std", "nanvar", "nanstd"):
        fn = getattr(A, fn_name)
        for corr in (0, 1, 1.5, 6, 100, -1, 0.0):
            yield (
                f"red {fn_name} correction={corr!r}",
                (lambda fn=fn, corr=corr: fn(small["f"], axis=2, correction=corr)),
            )
        yield f"red {fn_name} correction=1 none", (lambda fn=fn: fn(small["c"], correction=1))
        yield f"red {fn_name} correction str err", (lambda fn=fn: fn(small["f"], correction="1"))
        yield f"red {fn_name} correction 1j err", (lambda fn=fn: fn(small["f"], correction=1j))
        yield f"red {fn_name} large correction=1", (lambda fn=fn: fn(c.S("f", (1000, 1000), 177), axis=0, correction=1))
    yield "red prod float overflow order", lambda: A.prod(c.lit([1e200, 1e-200] * 8))
    yield "red prod large near 1", lambda: A.prod(c.S("f", (1_000_000,), 178) * 1e-4 + 1.0)
    yield "red prod large unit complex", lambda: A.prod(_unit(c.S("c", (200_000,), 179)))
    yield "red sum large exact", lambda: A.sum(c.S("f", (3_000_001,), 180))
    yield "red sum large complex 2M", lambda: A.sum(c.S("c", (2_000_000,), 181))
    yield "red mean bool large", lambda: A.mean(c.S("b", (1000, 1000), 182))
    yield "red any nan is true", lambda: (A.any(c.lit([0.0, NAN])), A.all(c.lit([1.0, NAN])), A.any(c.lit([-0.0, 0.0])))


def _unit(x):
    return x / x.__abs__()


def cumulative(c):
    A, DT = c.A, c.DT
    for fn_name in CUMULATIVE + NAN_CUMULATIVE:
        fn = getattr(A, fn_name)
        for k in KINDS + "n":
            x3 = c.S(k, (4, 5, 6), 190)
            x1 = c.S(k, (30,), 191)
            yield f"cum {fn_name} {k} 1d", (lambda fn=fn, x=x1: fn(x))
            yield f"cum {fn_name} {k} 1d include_initial", (lambda fn=fn, x=x1: fn(x, include_initial=True))
            for ax in (0, 1, -1):
                yield f"cum {fn_name} {k} 3d axis={ax}", (lambda fn=fn, x=x3, ax=ax: fn(x, axis=ax))
            yield (
                f"cum {fn_name} {k} 3d axis=1 include_initial",
                (lambda fn=fn, x=x3: fn(x, axis=1, include_initial=True)),
            )
            for d in KINDS:
                yield f"cum {fn_name} {k} dtype={d}", (lambda fn=fn, x=x1, d=d: fn(x, dtype=DT[d]))
        yield f"cum {fn_name} 3d axis None err", (lambda fn=fn: fn(c.S("f", (2, 3), 192)))
        yield f"cum {fn_name} axis tuple err", (lambda fn=fn: fn(c.S("f", (2, 3), 192), axis=(0,)))
        yield f"cum {fn_name} axis oob err", (lambda fn=fn: fn(c.S("f", (2, 3), 192), axis=2))
        yield f"cum {fn_name} keepdims err", (lambda fn=fn: fn(c.S("f", (3,), 192), keepdims=True))
        yield f"cum {fn_name} empty", (lambda fn=fn: fn(c.S("f", (0,), 193)))
        yield f"cum {fn_name} empty include_initial", (lambda fn=fn: fn(c.S("i", (0,), 193), include_initial=True))
        yield f"cum {fn_name} empty 2d axis=1", (lambda fn=fn: fn(c.S("f", (3, 0), 193), axis=1, include_initial=True))
        yield f"cum {fn_name} 1elem", (lambda fn=fn: fn(c.lit([3])))
        yield f"cum {fn_name} F-order", (lambda fn=fn: fn(c.S("f", (6, 5), 194).T, axis=0))
        yield f"cum {fn_name} strided", (lambda fn=fn: fn(c.S("f", (40,), 195)[::-3]))
        yield f"cum {fn_name} int wrap", (lambda fn=fn: fn(c.lit([INT_MAX, 1, 2**62, 2**62, 3])))
        yield f"cum {fn_name} special", (lambda fn=fn: fn(c.lit(F_SP, "f")))
        yield f"cum {fn_name} complex special", (lambda fn=fn: fn(c.lit(C_SP[:20], "c")))
        yield f"cum {fn_name} large f", (lambda fn=fn: fn(c.S("f", (1_000_000,), 196) * 1e-4 + 1.0))
        yield f"cum {fn_name} large f nan strided", (lambda fn=fn: fn(c.S("n", (1_000_000,), 197)[::-2]))
        yield f"cum {fn_name} large int", (lambda fn=fn: fn(c.S("I", (100_003,), 198)))
        yield f"cum {fn_name} large complex", (lambda fn=fn: fn(_unit(c.S("c", (300_000,), 199))))
        yield f"cum {fn_name} large bool", (lambda fn=fn: fn(c.S("b", (1_000_000,), 200)))
        yield f"cum {fn_name} large tall axis=0", (lambda fn=fn: fn(c.S("f", (30_000, 3), 201), axis=0))
        yield f"cum {fn_name} large 2d axis=1", (lambda fn=fn: fn(c.S("f", (1000, 1000), 202) * 1e-3 + 1.0, axis=1))
        yield f"cum {fn_name} large 2d F axis=0", (lambda fn=fn: fn(c.S("f", (1000, 1000), 203).T, axis=0))
        yield (
            f"cum {fn_name} large include_initial",
            (lambda fn=fn: fn(c.S("f", (500_000,), 204), include_initial=True)),
        )
    yield "cum dtype=bool err", lambda: A.cumulative_sum(c.lit([True, False]), dtype=DT["b"])
    for fn_name in NAN_CUMULATIVE:
        fn = getattr(A, fn_name)
        yield f"cum {fn_name} leading nan", (lambda fn=fn: fn(c.lit([NAN, NAN, 2.0, NAN, 3.0])))
        yield f"cum {fn_name} all nan", (lambda fn=fn: fn(c.lit([NAN, NAN]), include_initial=True))
        yield f"cum {fn_name} complex nan", (lambda fn=fn: fn(c.lit([1 + 1j, complex(NAN, 0), complex(0, NAN), 2j])))


# ---------------------------------------------------------------- sorting


def sorting(c):
    A = c.A
    samples = {
        "b": c.S("b", (4, 37), 210),
        "i": c.S("i", (4, 37), 211),
        "t": c.S("t", (4, 37), 212),
        "n": c.S("n", (4, 37), 213),
        "f special": c.lit(F_SP * 2, "f").reshape((4, 11)),
        "i special": c.lit(I_SP * 2, "i").reshape((4, 7)),
    }
    for k, x in samples.items():
        for ax in (-1, 0, 1):
            for desc in (False, True):
                for stable in (True, False):
                    tag = f"axis={ax} desc={desc} stable={stable}"
                    yield (
                        f"sort {k} {tag}",
                        (lambda x=x, ax=ax, d=desc, s=stable: A.sort(x, axis=ax, descending=d, stable=s)),
                    )
                    yield (
                        f"argsort {k} {tag}",
                        (lambda x=x, ax=ax, d=desc, s=stable: A.argsort(x, axis=ax, descending=d, stable=s)),
                    )
        yield f"argsort {k} unstable gives sorted", (lambda x=x: _take_sorted(A, x))
    yield "sort complex err", lambda: A.sort(c.S("c", (4, 37), 214))
    yield "argsort complex err", lambda: A.argsort(c.S("c", (4, 37), 214), stable=False)
    for fn in ("sort", "argsort"):
        f = getattr(A, fn)
        yield f"{fn} F-order axis=0", (lambda f=f: f(c.S("t", (37, 4), 215).T, axis=0))
        yield f"{fn} F-order axis=-1", (lambda f=f: f(c.S("t", (37, 4), 215).T))
        yield f"{fn} strided", (lambda f=f: f(c.S("t", (8, 40), 216)[::-2, ::3]))
        yield f"{fn} 5d axis=2", (lambda f=f: f(c.S("t", (2, 3, 5, 1, 2), 217), axis=2))
        yield f"{fn} empty", (lambda f=f: f(c.S("f", (3, 0), 218)))
        yield f"{fn} empty axis0", (lambda f=f: f(c.S("f", (0, 3), 218), axis=0))
        yield f"{fn} 1elem", (lambda f=f: f(c.lit([1.0])))
        yield f"{fn} bcast view", (lambda f=f: f(A.broadcast_to(c.S("f", (5,), 219), (3, 5)), axis=0))
        yield f"{fn} axis None err", (lambda f=f: f(c.S("f", (3, 4), 220), axis=None))
        yield f"{fn} axis oob err", (lambda f=f: f(c.S("f", (3, 4), 220), axis=2))
        yield f"{fn} axis tuple err", (lambda f=f: f(c.S("f", (3, 4), 220), axis=(0,)))
        yield f"{fn} list err", (lambda f=f: f([3, 1, 2]))
        yield f"{fn} descending positional err", (lambda f=f: f(c.S("f", (3,), 220), -1, True))
        yield f"{fn} stable None err", (lambda f=f: f(c.S("f", (3,), 220), stable=None))
        # adversarial patterns
        for pat in ("sorted", "reversed", "equal", "organ", "sawtooth"):
            yield f"{fn} pattern {pat}", (lambda f=f, pat=pat: f(_pattern(c, pat)))
            yield (
                f"{fn} pattern {pat} unstable values",
                (
                    lambda f=f, pat=pat: (
                        A.sort(_pattern(c, pat), stable=False)
                        if f is A.sort
                        else A.argsort(_pattern(c, pat), stable=True)
                    )
                ),
            )
        # large
        for k, shape in (
            ("f", (1_000_000,)),
            ("n", (1_000_000,)),
            ("I", (1_000_000,)),
            ("b", (1_000_000,)),
            ("t", (1_000_000,)),
        ):
            big = c.S(k, shape, 221)
            yield f"{fn} large {k}", (lambda f=f, x=big: f(x))
            yield f"{fn} large {k} desc", (lambda f=f, x=big: f(x, descending=True))
            if k != "c":
                yield f"{fn} large {k} unstable", (lambda f=f, x=big: f(x, stable=False))
        yield f"{fn} large strided", (lambda f=f: f(c.S("f", (2_000_000,), 222)[::-2]))
        M = c.S("t", (1000, 1000), 223)
        yield f"{fn} large 2d axis=-1", (lambda f=f: f(M))
        yield f"{fn} large 2d axis=0", (lambda f=f: f(M, axis=0))
        yield f"{fn} large 2d F axis=0", (lambda f=f: f(M.T, axis=0))
        yield f"{fn} large many short lanes", (lambda f=f: f(c.S("f", (100_000, 5), 224)))
    for n in (0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4097):
        yield f"sort size {n}", (lambda n=n: A.sort(c.S("f", (n,), 225)))
        yield f"argsort size {n} ties", (lambda n=n: A.argsort(c.S("i", (n,), 226) % 7))


def _take_sorted(A, x):
    """argsort(stable=False) may order ties freely; record only the values it selects."""
    idx = A.argsort(x, stable=False)
    return A.sort(x) if x.ndim != 2 else A.stack([x[i][idx[i]] for i in range(x.shape[0])])


def _pattern(c, pat):
    A = c.A
    n = 5000
    if pat == "sorted":
        return A.arange(n)
    if pat == "reversed":
        return A.arange(n)[::-1]
    if pat == "equal":
        return A.full(n, 1.5)
    if pat == "organ":
        return A.concat([A.arange(n // 2), A.arange(n // 2)[::-1]])
    return A.arange(n) % 17


# ---------------------------------------------------------------- unique


def unique(c):
    A = c.A
    ins = {
        "b": c.S("b", (4, 9), 230),
        "i": c.S("i", (5, 7), 231),
        "t": c.S("t", (6, 5), 232),
        "n": c.S("n", (40,), 233),
        "c": c.lit(
            [1 + 1j, 1 - 1j, 1 + 1j, -1j, 0j, -0.0 + 0j, complex(NAN, 0), complex(NAN, 0), complex(0, NAN), 1 + 1j]
        ),
        "f special": c.lit(F_SP + F_SP[::-1], "f"),
        "i special": c.lit(I_SP + I_SP, "i"),
        "signed zeros": c.lit([0.0, -0.0, 0.0, -0.0]),
        "nans": c.lit([NAN, 1.0, NAN, 1.0]),
        "empty": c.S("f", (0, 3), 234),
        "empty i": c.S("i", (0,), 234),
        "1elem": c.lit([[5]]),
        "F-order": c.S("t", (7, 4), 235).T,
        "strided": c.S("t", (20, 10), 236)[::-3, ::2],
        "5d": c.S("i", (2, 3, 1, 2, 3), 237) % 4,
        "large I mod": c.S("I", (1_000_000,), 238) % 5000,
        "large f": c.S("f", (1_000_000,), 239),
        "large n": c.S("n", (1_000_000,), 240),
        "large b": c.S("b", (1_000_000,), 241),
        "large c": c.S("c", (200_000,), 242),
        "large t F": c.S("t", (1000, 1000), 243).T,
    }
    for fn_name in ("unique_values", "unique_counts", "unique_inverse", "unique_all"):
        fn = getattr(A, fn_name)
        for nm, x in ins.items():
            yield f"{fn_name} {nm}", (lambda fn=fn, x=x: fn(x))
        yield f"{fn_name} list err", (lambda fn=fn: fn([1, 1, 2]))
        yield f"{fn_name} pyscalar err", (lambda fn=fn: fn(1))
        yield f"{fn_name} axis err", (lambda fn=fn: fn(ins["i"], axis=0))
    yield "unique_all field names", lambda: A.unique_all(ins["i"])._fields
    yield "unique_counts type name", lambda: type(A.unique_counts(ins["i"])).__name__


# ---------------------------------------------------------------- matmul


def matmul(c):
    A = c.A
    shapes = [
        ((4,), (4,)),
        ((3, 4), (4,)),
        ((4,), (4, 5)),
        ((3, 4), (4, 5)),
        ((2, 3, 4), (4, 5)),
        ((3, 4), (2, 4, 5)),
        ((2, 1, 3, 4), (5, 4, 2)),
        ((1, 4), (4, 1)),
        ((3, 0), (0, 4)),
        ((0, 4), (4, 3)),
        ((2, 0, 3), (3, 4)),
        ((0,), (0,)),
        ((1, 1, 1, 1, 3), (3, 2)),
    ]
    for sa, sb in shapes:
        for ka, kb in (("f", "f"), ("i", "i"), ("b", "b"), ("c", "c"), ("i", "f"), ("b", "c"), ("f", "c"), ("b", "i")):
            yield (
                f"matmul {sa}@{sb} {ka}{kb}",
                (lambda ka=ka, kb=kb, sa=sa, sb=sb: A.matmul(c.S(ka, sa, 250), c.S(kb, sb, 251))),
            )
    layouts = {
        "C": lambda s: c.S("f", s, 252),
        "F": lambda s: c.S("f", s[::-1], 253).T,
        "strided": lambda s: c.S("f", (2 * s[0], 2 * s[1]), 254)[::-2, 1::2],
        "bcast": lambda s: A.broadcast_to(c.S("f", (s[1],), 255), s),
    }
    for la, fa in layouts.items():
        for lb, fb in layouts.items():
            yield f"matmul layout {la}@{lb}", (lambda fa=fa, fb=fb: A.matmul(fa((7, 9)), fb((9, 5))))
            yield f"matmul large layout {la}@{lb}", (lambda fa=fa, fb=fb: A.matmul(fa((300, 300)), fb((300, 300))))
    yield (
        "matmul special values",
        lambda: A.matmul(c.lit(F_SP[:20], "f").reshape((4, 5)), c.lit(F_SP[2:22], "f").reshape((5, 4))),
    )
    yield "matmul nan inf", lambda: A.matmul(c.lit([[NAN, 1.0], [INF, 0.0]]), c.lit([[0.0, 1.0], [1.0, -INF]]))
    yield "matmul int overflow", lambda: A.matmul(c.lit([[INT_MAX, 2]]), c.lit([[2], [INT_MAX]]))
    yield (
        "matmul complex special",
        lambda: A.matmul(c.lit(C_SP[:16], "c").reshape((4, 4)), c.lit(C_SP[40:56], "c").reshape((4, 4))),
    )
    yield "matmul large f", lambda: A.matmul(c.S("f", (300, 400), 256), c.S("f", (400, 350), 257))
    yield "matmul large f MMT", lambda: (lambda M: M @ M.T)(c.S("f", (300, 500), 258))
    yield "matmul large i", lambda: A.matmul(c.S("I", (300, 400), 259) % 1000, c.S("I", (400, 350), 260) % 1000)
    yield "matmul large c", lambda: A.matmul(c.S("c", (200, 250), 261), c.S("c", (250, 220), 262))
    yield "matmul large b", lambda: A.matmul(c.S("b", (300, 300), 263), c.S("b", (300, 300), 264))
    yield "matmul large i@f", lambda: A.matmul(c.S("i", (300, 300), 265), c.S("f", (300, 300), 266))
    yield "matmul large matvec", lambda: A.matmul(c.S("f", (2000, 1000), 267), c.S("f", (1000,), 268))
    yield "matmul large vecmat", lambda: A.matmul(c.S("f", (1000,), 269), c.S("f", (1000, 2000), 270))
    yield "matmul large vecvec", lambda: A.matmul(c.S("f", (1_000_000,), 271), c.S("f", (1_000_000,), 272))
    yield "matmul batched small 64", lambda: A.matmul(c.S("f", (10, 64, 64), 273), c.S("f", (10, 64, 64), 274))
    yield "matmul batched small 8 bcast", lambda: A.matmul(c.S("f", (1000, 8, 8), 275), c.S("f", (8, 8), 276))
    yield "matmul batched small complex", lambda: A.matmul(c.S("c", (500, 8, 8), 277), c.S("c", (500, 8, 8), 278))
    yield "matmul batched small int", lambda: A.matmul(c.S("i", (2000, 4, 4), 279), c.S("i", (2000, 4, 4), 280))
    yield (
        "matmul batched transposed",
        lambda: A.matmul(A.permute_dims(c.S("f", (16, 16, 500), 281), (2, 0, 1)), c.S("f", (500, 16, 16), 282).mT),
    )
    yield "matmul batched bcast 4d", lambda: A.matmul(c.S("f", (20, 1, 6, 7), 283), c.S("f", (1, 30, 7, 5), 284))
    yield "matmul inner mismatch err", lambda: A.matmul(c.S("f", (3, 4), 285), c.S("f", (5, 3), 286))
    yield "matmul batch mismatch err", lambda: A.matmul(c.S("f", (2, 3, 4), 285), c.S("f", (3, 4, 5), 286))
    yield "matmul vec mismatch err", lambda: A.matmul(c.S("f", (3,), 285), c.S("f", (4,), 286))
    yield "matmul pyscalar err", lambda: A.matmul(c.S("f", (3,), 285), 2.0)
    yield "matmul list err", lambda: A.matmul(c.S("f", (3,), 285), [1.0, 2.0, 3.0])
    yield "matmul keyword err", lambda: A.matmul(x1=c.S("f", (3,), 285), x2=c.S("f", (3,), 285))


# ---------------------------------------------------------------- data type functions


def dtype_functions(c):
    A, DT = c.A, c.DT
    names = {"b": "bool", "i": "int64", "f": "float64", "c": "complex128"}
    xs = {k: c.S(k, (3,), 300) for k in KINDS}
    for a in KINDS:
        for b in KINDS:
            yield f"can_cast {a} {b}", (lambda a=a, b=b: A.can_cast(DT[a], DT[b]))
            yield f"can_cast {a} {b} names", (lambda a=a, b=b: A.can_cast(names[a], names[b]))
            yield f"can_cast {a} array {b}", (lambda a=a, b=b: A.can_cast(xs[a], DT[b]))
            yield f"result_type {a} {b}", (lambda a=a, b=b: A.result_type(DT[a], DT[b]))
            yield f"result_type {a} array {b} array", (lambda a=a, b=b: A.result_type(xs[a], xs[b]))
            yield f"result_type {a} pyscalar {b}", (lambda a=a, b=b: A.result_type(xs[a], PYSCALAR[b]))
    yield "result_type three", lambda: A.result_type(DT["b"], xs["i"], 2.5)
    yield "result_type pyscalars only", lambda: A.result_type(True, 3)
    yield "result_type one", lambda: A.result_type(DT["c"])
    yield "result_type none err", lambda: A.result_type()
    yield "result_type string err", lambda: A.result_type(A.string, DT["f"])
    yield "result_type list err", lambda: A.result_type([1.0])
    yield "can_cast string err", lambda: A.can_cast(A.string, DT["f"])
    yield "can_cast unknown name err", lambda: A.can_cast("float32", DT["f"])
    yield "can_cast pyscalar err", lambda: A.can_cast(1.0, DT["f"])
    kinds = ["bool", "signed integer", "unsigned integer", "integral", "real floating", "complex floating", "numeric"]
    for a in KINDS:
        for kind in kinds:
            yield f"isdtype {a} {kind}", (lambda a=a, kind=kind: A.isdtype(DT[a], kind))
        for b in KINDS:
            yield f"isdtype {a} dtype {b}", (lambda a=a, b=b: A.isdtype(DT[a], DT[b]))
        yield f"isdtype {a} tuple", (lambda a=a: A.isdtype(DT[a], ("bool", "real floating", DT["c"])))
        yield f"isdtype {a} name", (lambda a=a: A.isdtype(names[a], "numeric"))
    yield "isdtype string", lambda: A.isdtype(A.string, "numeric")
    yield "isdtype unknown kind err", lambda: A.isdtype(DT["f"], "floating")
    yield "isdtype array err", lambda: A.isdtype(xs["f"], "numeric")
    yield "isdtype kind int err", lambda: A.isdtype(DT["f"], 3)
    for nm, arg in (("float64", DT["f"]), ("complex128", DT["c"]), ("name", "float64"), ("array", xs["f"])):
        yield f"finfo {nm}", (lambda arg=arg: A.finfo(arg))
    yield "finfo fields", lambda: A.finfo(DT["f"])._fields
    for nm, arg in (("int64", DT["i"]), ("name", "int64"), ("array", xs["i"])):
        yield f"iinfo {nm}", (lambda arg=arg: A.iinfo(arg))
    yield "iinfo fields", lambda: A.iinfo(DT["i"])._fields
    for nm, arg in (("bool", DT["b"]), ("int64", DT["i"]), ("string", A.string), ("pyfloat", 1.0)):
        yield f"finfo {nm} err", (lambda arg=arg: A.finfo(arg))
    for nm, arg in (("bool", DT["b"]), ("float64", DT["f"]), ("complex128", DT["c"]), ("pyint", 1)):
        yield f"iinfo {nm} err", (lambda arg=arg: A.iinfo(arg))


# ---------------------------------------------------------------- meshgrid, tril, triu, from_dlpack


def creation_more(c):
    A, np = c.A, c.np
    v = {k: c.S(k, (n,), 310) for k, n in (("b", 2), ("i", 3), ("f", 4), ("c", 5))}
    for indexing in ("xy", "ij"):
        yield f"meshgrid none {indexing}", (lambda ix=indexing: A.meshgrid(indexing=ix))
        yield f"meshgrid one {indexing}", (lambda ix=indexing: A.meshgrid(v["f"], indexing=ix))
        yield f"meshgrid two {indexing}", (lambda ix=indexing: A.meshgrid(v["i"], v["f"], indexing=ix))
        yield (
            f"meshgrid four kinds {indexing}",
            (lambda ix=indexing: A.meshgrid(v["b"], v["i"], v["f"], v["c"], indexing=ix)),
        )
        yield (
            f"meshgrid strided {indexing}",
            (lambda ix=indexing: A.meshgrid(c.S("f", (9,), 311)[::-2], v["i"], indexing=ix)),
        )
        yield f"meshgrid empty {indexing}", (lambda ix=indexing: A.meshgrid(c.S("f", (0,), 312), v["f"], indexing=ix))
        yield (
            f"meshgrid large {indexing}",
            (lambda ix=indexing: A.meshgrid(c.S("f", (1000,), 313), c.S("f", (700,), 314), indexing=ix)),
        )
    yield "meshgrid default is xy", lambda: A.meshgrid(v["i"], v["f"])[0].shape
    yield "meshgrid indexing err", lambda: A.meshgrid(v["f"], indexing="xyz")
    yield "meshgrid 2d err", lambda: A.meshgrid(c.S("f", (2, 2), 315))
    yield "meshgrid list err", lambda: A.meshgrid([1.0, 2.0])
    yield "meshgrid positional indexing err", lambda: A.meshgrid(v["f"], "ij")
    for name in ("tril", "triu"):
        fn = getattr(A, name)
        for k in KINDS:
            x = c.S(k, (5, 7), 316)
            for kk in (0, 1, -1, 3, -4, 10, -10):
                yield f"{name} {k} k={kk}", (lambda fn=fn, x=x, kk=kk: fn(x, k=kk))
        yield f"{name} default k", (lambda fn=fn: fn(c.S("f", (4, 4), 317)))
        yield f"{name} batch", (lambda fn=fn: fn(c.S("f", (3, 4, 5), 318), k=1))
        yield f"{name} batch 4d", (lambda fn=fn: fn(c.S("i", (2, 3, 4, 2), 319), k=-1))
        yield f"{name} F-order", (lambda fn=fn: fn(c.S("f", (6, 4), 320).T))
        yield f"{name} strided", (lambda fn=fn: fn(c.S("f", (10, 12), 321)[::-2, 1::3]))
        yield f"{name} empty", (lambda fn=fn: fn(c.S("f", (0, 3), 322)))
        yield f"{name} special", (lambda fn=fn: fn(c.lit(F_SP[:20], "f").reshape((4, 5))))
        yield f"{name} large", (lambda fn=fn: fn(c.S("f", (1000, 1000), 323), k=3))
        yield f"{name} large batch", (lambda fn=fn: fn(c.S("c", (40, 100, 100), 324), k=-2))
        yield f"{name} large F-order", (lambda fn=fn: fn(c.S("f", (700, 900), 325).T))
        yield f"{name} 1d err", (lambda fn=fn: fn(c.S("f", (4,), 326)))
        yield f"{name} k float err", (lambda fn=fn: fn(c.S("f", (3, 3), 326), k=1.0))
        yield f"{name} k positional err", (lambda fn=fn: fn(c.S("f", (3, 3), 326), 1))
        yield f"{name} list err", (lambda fn=fn: fn([[1.0, 2.0], [3.0, 4.0]]))
    for k in KINDS:
        x = c.S(k, (3, 4), 327)
        yield f"from_dlpack sd {k}", (lambda x=x: A.from_dlpack(x))
        yield f"from_dlpack numpy {k}", (lambda k=k: A.from_dlpack(c.gen(k, (3, 4), 328).copy()))
    yield "from_dlpack sd strided", lambda: A.from_dlpack(c.S("f", (6, 8), 329)[::-2, 1::3])
    yield "from_dlpack numpy F", lambda: A.from_dlpack(np.asfortranarray(c.gen("f", (5, 6), 330)))
    yield "from_dlpack shares memory", lambda: (lambda x: _shares(A, x, A.from_dlpack(x)))(c.S("f", (4,), 331).copy())
    yield (
        "from_dlpack copy True",
        lambda: (lambda x: _shares(A, x, A.from_dlpack(x, copy=True)))(c.S("f", (4,), 331).copy()),
    )
    yield (
        "from_dlpack copy False",
        lambda: (lambda x: _shares(A, x, A.from_dlpack(x, copy=False)))(c.S("f", (4,), 331).copy()),
    )
    yield "from_dlpack readonly", lambda: A.from_dlpack(A.broadcast_to(c.S("f", (3,), 332), (2, 3)))
    yield "from_dlpack numpy readonly", lambda: A.from_dlpack(c.gen("f", (3,), 333))
    yield "from_dlpack large", lambda: A.from_dlpack(c.gen("f", (1_000_000,), 334).copy())
    yield "from_dlpack empty", lambda: A.from_dlpack(c.S("f", (0, 3), 335))
    yield "from_dlpack device cpu", lambda: A.from_dlpack(c.S("f", (2,), 336), device=(1, 0))
    yield "from_dlpack device err", lambda: A.from_dlpack(c.S("f", (2,), 336), device=(2, 0))
    yield "from_dlpack list err", lambda: A.from_dlpack([1.0, 2.0])
    yield "from_dlpack numpy float32 err", lambda: A.from_dlpack(np.ones(3, dtype=np.float32))


def _shares(A, x, y):
    """y's values, and whether writing to x shows in y."""
    before = y.copy()
    x[0] = 99.5
    return before, y[0] == 99.5


# ---------------------------------------------------------------- flip, roll, repeat, tile, unstack, diff


def shapes_more(c):
    A = c.A
    x3 = {k: c.S(k, (3, 4, 5), 340) for k in KINDS}
    for k, x in x3.items():
        for ax in (None, 0, 1, -1, (0, 2), (2, 0, 1)):
            yield f"flip {k} axis={ax}", (lambda x=x, ax=ax: _view(A.flip(x, axis=ax), x))
    yield "flip default", lambda: A.flip(c.S("i", (2, 3), 341))
    yield "flip F-order", lambda: _view(A.flip(c.S("f", (6, 4), 342).T, axis=0), None)
    yield "flip strided", lambda: A.flip(c.S("f", (8, 9), 343)[::-2, 1::3], axis=1)
    yield "flip large", lambda: A.flip(c.S("f", (1000, 1000), 344), axis=1)
    yield "flip empty", lambda: A.flip(c.S("f", (0, 3), 345))
    yield "flip axis oob err", lambda: A.flip(x3["f"], axis=3)
    yield "flip axis dup err", lambda: A.flip(x3["f"], axis=(0, 0))
    yield "flip axis positional err", lambda: A.flip(x3["f"], 0)
    for k, x in x3.items():
        for shift, ax in (
            (1, None),
            (-7, None),
            (2, 0),
            (3, -1),
            ((1, 2), (0, 2)),
            ([1, -1], [1, 2]),
            (2, (0, 1)),
            ((1, 2), 0),
        ):
            yield f"roll {k} shift={shift} axis={ax}", (lambda x=x, s=shift, ax=ax: A.roll(x, s, axis=ax))
    yield "roll base", lambda: A.roll(c.S("f", (5,), 346), 2).base is None
    yield "roll zero shift", lambda: A.roll(c.S("f", (5,), 346), 0)
    yield "roll big shift", lambda: A.roll(c.S("i", (7,), 347), 10**12)
    yield "roll F-order", lambda: A.roll(c.S("f", (6, 4), 348).T, 1, axis=1)
    yield "roll large", lambda: A.roll(c.S("f", (1_000_000,), 349), 12345)
    yield "roll large 2d", lambda: A.roll(c.S("f", (1000, 1000), 350), (3, -5), axis=(0, 1))
    yield "roll empty", lambda: A.roll(c.S("f", (0,), 351), 3)
    yield "roll tuple shift no axis err", lambda: A.roll(x3["f"], (1, 2))
    yield "roll length mismatch err", lambda: A.roll(x3["f"], (1, 2), axis=(0, 1, 2))
    yield "roll axis oob err", lambda: A.roll(x3["f"], 1, axis=3)
    yield "roll float shift err", lambda: A.roll(x3["f"], 1.0)
    for k, x in x3.items():
        yield f"repeat {k} 2", (lambda x=x: A.repeat(x, 2))
        yield f"repeat {k} 3 axis=1", (lambda x=x: A.repeat(x, 3, axis=1))
        yield f"repeat {k} array axis=2", (lambda x=x: A.repeat(x, c.lit([0, 1, 2, 3, 1], "i"), axis=2))
    yield "repeat 0", lambda: A.repeat(x3["f"], 0, axis=0)
    yield "repeat length-1 array", lambda: A.repeat(x3["i"], c.lit([2], "i"), axis=-1)
    yield "repeat array no axis", lambda: A.repeat(c.S("f", (2, 2), 352), c.lit([1, 0, 2, 1], "i"))
    yield "repeat F-order", lambda: A.repeat(c.S("f", (6, 4), 353).T, 2, axis=0)
    yield "repeat large", lambda: A.repeat(c.S("f", (300_000,), 354), 3)
    yield "repeat large array", lambda: A.repeat(c.S("f", (1000, 300), 355), c.S("I", (1000,), 356) % 4, axis=0)
    yield "repeat negative err", lambda: A.repeat(x3["f"], -1)
    yield "repeat negative array err", lambda: A.repeat(c.lit([1.0, 2.0]), c.lit([1, -1], "i"))
    yield "repeat array length err", lambda: A.repeat(c.lit([1.0, 2.0]), c.lit([1, 2, 3], "i"))
    yield "repeat float err", lambda: A.repeat(x3["f"], 2.0)
    yield "repeat float array err", lambda: A.repeat(c.lit([1.0, 2.0]), c.lit([1.0, 2.0]))
    for k, x in x3.items():
        for reps in (2, (2,), (1, 2, 3), (2, 1, 1, 2), (0, 1, 1)):
            yield f"tile {k} {reps}", (lambda x=x, r=reps: A.tile(x, r))
    yield "tile 1d to 2d", lambda: A.tile(c.S("f", (3,), 357), (2, 2))
    yield "tile large", lambda: A.tile(c.S("f", (500, 400), 358), (3, 2))
    yield "tile list err", lambda: A.tile(x3["f"], [2, 2])
    yield "tile negative err", lambda: A.tile(x3["f"], (2, -1))
    yield "tile float err", lambda: A.tile(x3["f"], 2.0)
    for k, x in x3.items():
        for ax in (0, 1, -1):
            yield f"unstack {k} axis={ax}", (lambda x=x, ax=ax: tuple(_view(v, x) for v in A.unstack(x, axis=ax)))
    for k in KINDS:
        yield f"unstack {k} 1d scalars", (lambda k=k: A.unstack(c.S(k, (4,), 359)))
    yield "unstack empty axis", lambda: A.unstack(c.S("f", (0, 3), 360))
    yield "unstack large", lambda: [v.shape for v in A.unstack(c.S("f", (4, 500_000), 361), axis=0)]
    yield "unstack axis oob err", lambda: A.unstack(x3["f"], axis=3)
    yield "unstack 0d err", lambda: A.unstack(A.asarray(1.0))
    for k, x in x3.items():
        for kw in (dict(), dict(axis=0), dict(axis=1, n=2), dict(n=4), dict(n=5), dict(n=0)):
            tag = ",".join(f"{a}={b}" for a, b in kw.items()) or "default"
            yield f"diff {k} {tag}", (lambda x=x, kw=kw: A.diff(x, **kw))
        yield f"diff {k} prepend scalar", (lambda x=x, k=k: A.diff(x, prepend=PYSCALAR[k]))
        yield f"diff {k} append array", (lambda x=x, k=k: A.diff(x, axis=1, append=c.S(k, (3, 2, 5), 362)))
    yield "diff prepend and append", lambda: A.diff(c.lit([1.0, 4.0, 9.0]), prepend=0.0, append=c.lit([20, 30], "i"))
    yield "diff n=0 with prepend", lambda: A.diff(c.lit([1.0, 4.0]), n=0, prepend=7.0)
    yield "diff special", lambda: A.diff(c.lit(F_SP, "f"))
    yield "diff int wraps", lambda: A.diff(c.lit([INT_MIN, INT_MAX, INT_MIN]))
    yield "diff F-order", lambda: A.diff(c.S("f", (6, 4), 363).T, axis=0)
    yield "diff large", lambda: A.diff(c.S("f", (1_000_000,), 364))
    yield "diff large 2d axis=0 n=3", lambda: A.diff(c.S("f", (1000, 1000), 365), axis=0, n=3)
    yield "diff large nan", lambda: A.diff(c.S("n", (1_000_000,), 366)[::-1])
    yield "diff empty", lambda: A.diff(c.S("f", (0,), 367))
    yield "diff negative n err", lambda: A.diff(x3["f"], n=-1)
    yield "diff axis oob err", lambda: A.diff(x3["f"], axis=3)
    yield "diff append shape err", lambda: A.diff(x3["f"], axis=1, append=c.S("f", (2, 1, 5), 368))
    yield "diff append ndim err", lambda: A.diff(x3["f"], append=c.S("f", (5,), 368))
    yield "diff prepend list err", lambda: A.diff(x3["f"], prepend=[1.0])
    yield "diff 0d err", lambda: A.diff(A.asarray(1.0))


# ---------------------------------------------------------------- quantiles


METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


def quantiles(c):
    A = c.A
    ins = {
        "f": c.S("f", (4, 5, 6), 370),
        "t": c.S("t", (4, 5, 6), 371),
        "n": c.S("n", (4, 5, 6), 372),
        "i": c.S("i", (4, 5, 6), 373),
        "b": c.S("b", (4, 5, 6), 374),
    }
    qs = {"0.5": 0.5, "0": 0, "1": 1, "0.3": 0.3, "array": c.lit([0.0, 0.1, 0.5, 0.9, 1.0]), "int array": c.lit([0, 1])}
    for name in ("quantile", "nanquantile"):
        fn = getattr(A, name)
        for k, x in ins.items():
            for qn, q in qs.items():
                yield f"{name} {k} q={qn}", (lambda fn=fn, x=x, q=q: fn(x, q))
            for ax in (0, 1, -1, (0, 2)):
                yield f"{name} {k} axis={ax}", (lambda fn=fn, x=x, ax=ax: fn(x, qs["array"], axis=ax))
            yield f"{name} {k} keepdims", (lambda fn=fn, x=x: fn(x, 0.25, axis=1, keepdims=True))
        for m in METHODS:
            yield f"{name} method={m}", (lambda fn=fn, m=m: fn(ins["t"], qs["array"], axis=1, method=m))
            yield f"{name} method={m} nan", (lambda fn=fn, m=m: fn(ins["n"], 0.37, axis=2, method=m))
            yield f"{name} method={m} large", (lambda fn=fn, m=m: fn(c.S("f", (100_003,), 375), 0.37, method=m))
        yield f"{name} special", (lambda fn=fn: fn(c.lit(F_SP, "f"), qs["array"]))
        yield f"{name} signed zeros", (lambda fn=fn: fn(c.lit([0.0, -0.0, -0.0, 0.0, 0.0]), qs["array"]))
        yield f"{name} all nan lane", (lambda fn=fn: fn(c.lit([[NAN, NAN], [1.0, NAN]]), 0.5, axis=1))
        yield f"{name} empty lane", (lambda fn=fn: fn(c.S("f", (3, 0), 376), 0.5, axis=1))
        yield f"{name} empty", (lambda fn=fn: fn(c.S("f", (0,), 376), 0.5))
        yield f"{name} F-order", (lambda fn=fn: fn(c.S("f", (6, 5), 377).T, 0.5, axis=0))
        yield f"{name} strided", (lambda fn=fn: fn(c.S("f", (20, 9), 378)[::-3, 1::2], 0.75, axis=0))
        yield f"{name} large", (lambda fn=fn: fn(c.S("f", (1_000_000,), 379), qs["array"]))
        yield f"{name} large nan", (lambda fn=fn: fn(c.S("n", (1_000_000,), 380), 0.5))
        yield f"{name} large 2d axis=0", (lambda fn=fn: fn(c.S("f", (1000, 1000), 381), 0.9, axis=0))
        yield f"{name} large 2d axis=1", (lambda fn=fn: fn(c.S("n", (1000, 1000), 382), qs["array"], axis=1))
        yield f"{name} large many short lanes", (lambda fn=fn: fn(c.S("t", (5, 200_000), 383), 0.5, axis=0))
        yield f"{name} q out of range err", (lambda fn=fn: fn(ins["f"], 1.5))
        yield f"{name} q nan err", (lambda fn=fn: fn(ins["f"], NAN))
        yield f"{name} q bool err", (lambda fn=fn: fn(ins["f"], True))
        yield f"{name} q bool array err", (lambda fn=fn: fn(ins["f"], c.lit([True])))
        yield f"{name} q list err", (lambda fn=fn: fn(ins["f"], [0.5]))
        yield f"{name} q 2d err", (lambda fn=fn: fn(ins["f"], c.lit([[0.5]])))
        yield f"{name} method err", (lambda fn=fn: fn(ins["f"], 0.5, method="cubic"))
        yield f"{name} complex err", (lambda fn=fn: fn(c.S("c", (5,), 384), 0.5))
        yield f"{name} axis oob err", (lambda fn=fn: fn(ins["f"], 0.5, axis=3))
        yield f"{name} list err", (lambda fn=fn: fn([1.0, 2.0], 0.5))
    for name in ("median", "nanmedian"):
        fn = getattr(A, name)
        for k, x in ins.items():
            for ax in (None, 0, 1, -1, (0, 2)):
                yield f"{name} {k} axis={ax}", (lambda fn=fn, x=x, ax=ax: fn(x, axis=ax))
            yield f"{name} {k} keepdims", (lambda fn=fn, x=x: fn(x, axis=2, keepdims=True))
        yield f"{name} even and odd", (lambda fn=fn: (fn(c.lit([3.0, 1.0, 2.0])), fn(c.lit([4.0, 1.0, 3.0, 2.0]))))
        yield f"{name} special", (lambda fn=fn: fn(c.lit(F_SP, "f")))
        yield f"{name} signed zeros", (lambda fn=fn: fn(c.lit([-0.0, 0.0, -0.0, 0.0])))
        yield f"{name} all nan lane", (lambda fn=fn: fn(c.lit([[NAN, NAN], [1.0, NAN]]), axis=1))
        yield f"{name} empty", (lambda fn=fn: fn(c.S("f", (0, 4), 385), axis=0))
        yield f"{name} large", (lambda fn=fn: fn(c.S("f", (1_000_001,), 386)))
        yield f"{name} large nan", (lambda fn=fn: fn(c.S("n", (1_000_000,), 387)))
        yield f"{name} large 2d axis=0", (lambda fn=fn: fn(c.S("f", (1000, 1000), 388), axis=0))
        yield f"{name} large 2d F axis=1", (lambda fn=fn: fn(c.S("t", (1000, 1000), 389).T, axis=1))
        yield f"{name} complex err", (lambda fn=fn: fn(c.S("c", (5,), 390)))
        yield f"{name} q argument err", (lambda fn=fn: fn(ins["f"], 0.5))
        yield f"{name} list err", (lambda fn=fn: fn([1.0, 2.0]))


# ---------------------------------------------------------------- cov, corrcoef


def covariance(c):
    A = c.A
    ins = {
        "f 2d": c.S("f", (4, 30), 400),
        "i 2d": c.S("i", (3, 20), 401),
        "b 2d": c.S("b", (3, 20), 402),
        "c 2d": c.S("c", (3, 15), 403),
        "f 1d": c.S("f", (25,), 404),
        "c 1d": c.S("c", (25,), 405),
        "n 2d": c.S("n", (3, 30), 406),
        "special": c.lit(F_SP[:20], "f").reshape((4, 5)),
        "one obs": c.S("f", (3, 1), 407),
        "no obs": c.S("f", (3, 0), 408),
        "no vars": c.S("f", (0, 3), 409),
        "F-order": c.S("f", (30, 4), 410).T,
        "strided": c.S("f", (8, 60), 411)[::2, ::-3],
        "constant row": c.lit([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]),
    }
    for nm, x in ins.items():
        for rv in (True, False):
            yield f"cov {nm} rowvar={rv}", (lambda x=x, rv=rv: A.cov(x, rowvar=rv))
            yield f"corrcoef {nm} rowvar={rv}", (lambda x=x, rv=rv: A.corrcoef(x, rowvar=rv))
    for corr in (0, 1, 1.5, 29, 30, 31, -1, 0.0):
        yield f"cov correction={corr!r}", (lambda corr=corr: A.cov(ins["f 2d"], correction=corr))
    yield "cov correction complex", lambda: A.cov(ins["c 2d"], correction=1)
    yield "cov symmetric", lambda: (lambda C: bool(A.all(C == C.mT)))(A.cov(c.S("f", (40, 200), 412)))
    yield "cov large", lambda: A.cov(c.S("f", (64, 50_000), 413))
    yield "cov large rowvar=False", lambda: A.cov(c.S("f", (50_000, 64), 414), rowvar=False)
    yield "cov large many vars", lambda: A.cov(c.S("f", (500, 2000), 415), correction=1)
    yield "cov large complex", lambda: A.cov(c.S("c", (16, 20_000), 416))
    yield "cov large int", lambda: A.cov(c.S("I", (32, 40_000), 417))
    yield "corrcoef large", lambda: A.corrcoef(c.S("f", (64, 50_000), 418))
    yield "corrcoef large complex", lambda: A.corrcoef(c.S("c", (16, 20_000), 419), rowvar=True)
    for fn_name in ("cov", "corrcoef"):
        fn = getattr(A, fn_name)
        yield f"{fn_name} 3d err", (lambda fn=fn: fn(c.S("f", (2, 3, 4), 420)))
        yield f"{fn_name} 0d err", (lambda fn=fn: fn(A.asarray(1.0)))
        yield f"{fn_name} list err", (lambda fn=fn: fn([[1.0, 2.0], [3.0, 4.0]]))
        yield f"{fn_name} rowvar positional err", (lambda fn=fn: fn(ins["f 2d"], False))
    yield "corrcoef correction err", lambda: A.corrcoef(ins["f 2d"], correction=1)
    yield "cov correction str err", lambda: A.cov(ins["f 2d"], correction="1")


# ---------------------------------------------------------------- bincount, histogram


def counting(c):
    A = c.A
    small = c.S("i", (40,), 430) % 7 + 7
    for nm, kw in (
        ("plain", dict()),
        ("minlength", dict(minlength=20)),
        ("minlength small", dict(minlength=3)),
        ("weights f", dict(weights=c.S("f", (40,), 431))),
        ("weights i", dict(weights=c.S("i", (40,), 432))),
        ("weights b", dict(weights=c.S("b", (40,), 433))),
        ("weights special", dict(weights=c.lit((F_SP * 2)[:40], "f"))),
    ):
        yield f"bincount {nm}", (lambda kw=kw: A.bincount(small, **kw))
    yield "bincount bool", lambda: A.bincount(c.S("b", (30,), 434))
    yield "bincount empty", lambda: A.bincount(c.S("i", (0,), 435))
    yield "bincount empty minlength", lambda: A.bincount(c.S("i", (0,), 435), minlength=4)
    yield "bincount strided", lambda: A.bincount(small[::-3])
    yield "bincount large", lambda: A.bincount(c.S("I", (1_000_000,), 436) % 1000 + 1000)
    yield (
        "bincount large weights parallel",
        lambda: A.bincount(_mod(c, 100_000, 10_000, 437), weights=c.S("f", (100_000,), 438)),
    )
    yield (
        "bincount large weights sequential",
        lambda: A.bincount(_mod(c, 1_000_000, 50, 439), weights=c.S("f", (1_000_000,), 440)),
    )
    yield (
        "bincount large weights int",
        lambda: A.bincount(_mod(c, 100_000, 10_000, 441), weights=c.S("I", (100_000,), 442)),
    )
    yield "bincount negative err", lambda: A.bincount(c.lit([1, -1, 2]))
    yield "bincount float err", lambda: A.bincount(c.lit([1.0, 2.0]))
    yield "bincount 2d err", lambda: A.bincount(c.S("i", (2, 3), 443) % 3 + 3)
    yield "bincount weights length err", lambda: A.bincount(small, weights=c.S("f", (39,), 444))
    yield "bincount weights complex err", lambda: A.bincount(small, weights=c.S("c", (40,), 445))
    yield "bincount minlength negative err", lambda: A.bincount(small, minlength=-1)
    yield "bincount list err", lambda: A.bincount([1, 2])
    x = c.S("f", (5, 40), 446)
    for nm, kw in (
        ("default", dict()),
        ("bins=1", dict(bins=1)),
        ("bins=7", dict(bins=7)),
        ("range", dict(bins=5, range=(-1.0, 1.0))),
        ("range int", dict(bins=4, range=(-2, 2))),
        ("edges f", dict(bins=c.lit([-3.0, -1.0, 0.0, 0.5, 3.0]))),
        ("edges i", dict(bins=c.lit([-3, -1, 0, 1, 3]))),
        ("edges one", dict(bins=c.lit([0.0]))),
        ("edges none", dict(bins=c.S("f", (0,), 447))),
        ("weights f", dict(weights=c.S("f", (5, 40), 448))),
        ("weights i", dict(weights=c.S("i", (5, 40), 449))),
        ("weights b", dict(weights=c.S("b", (5, 40), 450))),
        ("edges weights", dict(bins=c.lit([-3.0, 0.0, 3.0]), weights=c.S("f", (5, 40), 451))),
        ("density", dict(density=True)),
        ("density edges weights", dict(bins=c.lit([-2.0, 0.0, 1.0]), weights=c.S("i", (5, 40), 452), density=True)),
    ):
        yield f"histogram {nm}", (lambda kw=kw: A.histogram(x, **kw))
    yield "histogram int data", lambda: A.histogram(c.S("i", (50,), 453))
    yield "histogram bool data", lambda: A.histogram(c.S("b", (50,), 454), bins=2)
    yield "histogram nan with range", lambda: A.histogram(c.S("n", (60,), 455), bins=4, range=(-2.0, 2.0))
    yield "histogram one value", lambda: A.histogram(c.lit([2.5, 2.5, 2.5]))
    yield "histogram empty", lambda: A.histogram(c.S("f", (0,), 456))
    yield "histogram fields", lambda: A.histogram(x)._fields
    yield (
        "histogram edges shared",
        lambda: (lambda e: A.histogram(x, bins=e).bin_edges.base is e)(c.lit([-1.0, 0.0, 1.0])),
    )
    yield "histogram large", lambda: A.histogram(c.S("f", (1_000_000,), 457), bins=50)
    yield (
        "histogram large weights",
        lambda: A.histogram(c.S("f", (1_000_000,), 458), bins=64, weights=c.S("f", (1_000_000,), 459)),
    )
    yield (
        "histogram large edges weights",
        lambda: A.histogram(
            c.S("f", (300_000,), 460),
            bins=c.lit([-4.0, -1.0, -0.1, 0.0, 0.3, 2.0, 5.0]),
            weights=c.S("f", (300_000,), 461),
        ),
    )
    yield (
        "histogram large int weights",
        lambda: A.histogram(c.S("f", (300_000,), 462), bins=10, weights=c.S("I", (300_000,), 463)),
    )
    yield "histogram nan without range err", lambda: A.histogram(c.S("n", (60,), 464))
    yield "histogram inf without range err", lambda: A.histogram(c.lit([1.0, INF]))
    yield "histogram complex err", lambda: A.histogram(c.S("c", (5,), 465))
    yield "histogram edges decreasing err", lambda: A.histogram(x, bins=c.lit([1.0, 0.0]))
    yield "histogram bins zero err", lambda: A.histogram(x, bins=0)
    yield "histogram bins str err", lambda: A.histogram(x, bins="auto")
    yield "histogram range reversed err", lambda: A.histogram(x, range=(1.0, -1.0))
    yield "histogram weights shape err", lambda: A.histogram(x, weights=c.S("f", (40,), 466))
    yield "histogram list err", lambda: A.histogram([1.0, 2.0])


def _mod(c, n, m, seed):
    return c.S("I", (n,), seed) % m + m


# ---------------------------------------------------------------- take_along_axis, nonzero, count_nonzero, searchsorted


def index_more(c):
    A = c.A
    x = {k: c.S(k, (4, 5, 6), 470) for k in KINDS}
    t = c.S("t", (4, 5, 6), 471)
    for ax in (0, 1, 2, -1):
        idx = A.argsort(t, axis=ax)
        for k in KINDS:
            yield (
                f"take_along_axis {k} argsort axis={ax}",
                (lambda k=k, idx=idx, ax=ax: A.take_along_axis(x[k], idx, axis=ax)),
            )
    for k in KINDS:
        yield f"take {k} axis=1", (lambda k=k: A.take(x[k], c.lit([0, 4, -1, 2, 2], "i"), axis=1))
        yield f"take {k} 1d", (lambda k=k: A.take(c.S(k, (9,), 469), c.lit([8, 0, -9, 3], "i")))
    yield "take int index", lambda: A.take(x["f"], 2, axis=2)
    yield "take 2d indices", lambda: A.take(x["f"], c.lit([[0, 1], [-1, 3]], "i"), axis=0)
    yield "take empty indices", lambda: A.take(x["f"], c.S("i", (0,), 468), axis=1)
    yield "take F-order", lambda: A.take(c.S("f", (6, 4), 467).T, c.lit([5, 0], "i"), axis=1)
    yield "take large", lambda: A.take(c.S("f", (1_000_000,), 466), c.S("I", (300_000,), 465) % 1_000_000)
    yield "take large axis=0", lambda: A.take(c.S("f", (1000, 1000), 464), c.S("I", (2000,), 463) % 1000, axis=0)
    yield "take oob err", lambda: A.take(x["f"], c.lit([5], "i"), axis=0)
    yield "take bool err", lambda: A.take(x["f"], c.lit([True, False]), axis=0)
    yield "take float err", lambda: A.take(x["f"], c.lit([1.0]), axis=0)
    yield "take no axis 2d err", lambda: A.take(x["f"], c.lit([0], "i"))
    yield "take list err", lambda: A.take(x["f"], [0, 1], axis=0)
    yield "take_along_axis default axis", lambda: A.take_along_axis(x["f"], A.argsort(t))
    yield "take_along_axis bcast", lambda: A.take_along_axis(x["f"], c.lit([[[0], [5], [-1], [2], [3]]], "i"), axis=2)
    yield "take_along_axis negative", lambda: A.take_along_axis(x["i"], c.lit([[[-1, -6]]], "i"), axis=2)
    yield (
        "take_along_axis F-order",
        lambda: A.take_along_axis(c.S("f", (6, 4), 472).T, c.lit([[0, 5], [1, 1], [2, 0], [5, 3]], "i"), axis=1),
    )
    yield (
        "take_along_axis large",
        lambda: (lambda M: A.take_along_axis(M, A.argsort(M, axis=0), axis=0))(c.S("f", (1000, 1000), 473)),
    )
    yield "take_along_axis oob err", lambda: A.take_along_axis(x["f"], c.lit([[[6]]], "i"), axis=2)
    yield "take_along_axis bool err", lambda: A.take_along_axis(x["f"], c.lit([[[True]]]), axis=2)
    yield "take_along_axis float err", lambda: A.take_along_axis(x["f"], c.lit([[[1.0]]]), axis=2)
    yield "take_along_axis ndim err", lambda: A.take_along_axis(x["f"], c.lit([0, 1], "i"), axis=2)
    yield "take_along_axis bcast err", lambda: A.take_along_axis(x["f"], c.S("i", (3, 5, 2), 474) % 6, axis=2)
    yield "take_along_axis axis oob err", lambda: A.take_along_axis(x["f"], A.argsort(t), axis=3)
    yield "take_along_axis list err", lambda: A.take_along_axis(x["f"], [[[0]]], axis=2)
    nz = {
        "b": c.S("b", (4, 5), 475),
        "i": c.S("i", (4, 5), 476) % 3,
        "f": c.lit([[0.0, -0.0, NAN, 1.0], [0.0, INF, -2.0, 0.0]]),
        "c": c.lit([0j, complex(0, 1), complex(1, 0), complex(-0.0, 0), complex(NAN, 0)]),
        "3d": c.S("t", (2, 3, 4), 477),
        "1d": c.S("t", (9,), 478),
        "empty": c.S("f", (0, 3), 479),
        "all zero": A.zeros((3, 3)),
        "F-order": c.S("t", (5, 4), 480).T,
        "strided": c.S("t", (10, 10), 481)[::-3, 1::2],
        "large": c.S("b", (1_000_000,), 482),
        "large 2d": c.S("t", (1000, 1000), 483),
        "large sparse": c.S("I", (1_000_000,), 484) % 1000 == 0,
    }
    for nm, v in nz.items():
        yield f"nonzero {nm}", (lambda v=v: A.nonzero(v))
        yield f"count_nonzero {nm}", (lambda v=v: A.count_nonzero(v))
    for ax in (0, 1, -1, (0, 1), ()):
        yield f"count_nonzero axis={ax}", (lambda ax=ax: A.count_nonzero(nz["3d"], axis=ax))
        yield f"count_nonzero axis={ax} keepdims", (lambda ax=ax: A.count_nonzero(nz["3d"], axis=ax, keepdims=True))
    yield "count_nonzero large axis=0", lambda: A.count_nonzero(nz["large 2d"], axis=0)
    yield "count_nonzero large F axis=1", lambda: A.count_nonzero(nz["large 2d"].T, axis=1)
    yield "nonzero 0d err", lambda: A.nonzero(A.asarray(1.0))
    yield "nonzero list err", lambda: A.nonzero([0, 1])
    yield "count_nonzero axis oob err", lambda: A.count_nonzero(nz["3d"], axis=3)
    yield "count_nonzero list err", lambda: A.count_nonzero([0, 1])
    srt = {
        "f": A.sort(c.S("t", (50,), 485)),
        "i": A.sort(c.S("i", (50,), 486)),
        "b": A.sort(c.S("b", (20,), 487)),
        "n": A.sort(c.S("n", (50,), 488)),
    }
    probes = {
        "f": c.lit([-3.0, -2.5, 0.0, -0.0, 0.25, 2.5, 3.0, NAN, INF, -INF]),
        "i": c.lit([-60, -50, 0, 7, 49, 50, 60]),
        "b": c.lit([False, True]),
        "n": c.lit([NAN, INF, -INF, 0.0, -0.0, 1.0]),
    }
    for k, a in srt.items():
        for side in ("left", "right"):
            yield f"searchsorted {k} side={side}", (lambda a=a, k=k, s=side: A.searchsorted(a, probes[k], side=s))
            yield f"searchsorted {k} scalar side={side}", (lambda a=a, s=side: A.searchsorted(a, 0, side=s))
    yield "searchsorted mixed dtypes", lambda: A.searchsorted(srt["i"], probes["f"])
    yield "searchsorted 2d probes", lambda: A.searchsorted(srt["f"], c.S("t", (3, 4), 489))
    yield (
        "searchsorted sorter",
        lambda: (lambda u: A.searchsorted(u, probes["f"], sorter=A.argsort(u)))(c.S("t", (40,), 490)),
    )
    yield (
        "searchsorted sorter right",
        lambda: (lambda u: A.searchsorted(u, probes["f"], side="right", sorter=A.argsort(u)))(c.S("t", (40,), 490)),
    )
    yield "searchsorted empty", lambda: A.searchsorted(c.S("f", (0,), 491), probes["f"])
    yield "searchsorted strided", lambda: A.searchsorted(A.sort(c.S("f", (60,), 492))[::2], probes["f"])
    yield "searchsorted large", lambda: A.searchsorted(A.sort(c.S("f", (1_000_000,), 493)), c.S("f", (1_000_000,), 494))
    yield (
        "searchsorted large right",
        lambda: A.searchsorted(A.sort(c.S("t", (1_000_000,), 495)), c.S("t", (100_000,), 496), side="right"),
    )
    yield (
        "searchsorted large sorter",
        lambda: (lambda u: A.searchsorted(u, c.S("f", (200_000,), 497), sorter=A.argsort(u)))(
            c.S("f", (300_000,), 498)
        ),
    )
    yield "searchsorted side err", lambda: A.searchsorted(srt["f"], probes["f"], side="middle")
    yield "searchsorted 2d err", lambda: A.searchsorted(c.S("f", (2, 3), 499), probes["f"])
    yield "searchsorted complex err", lambda: A.searchsorted(c.lit([1j, 2j]), c.lit([1j]))
    yield "searchsorted sorter length err", lambda: A.searchsorted(srt["f"], probes["f"], sorter=c.lit([0, 1], "i"))
    yield (
        "searchsorted sorter oob err",
        lambda: A.searchsorted(c.lit([1.0, 2.0]), probes["f"], sorter=c.lit([0, 5], "i")),
    )
    yield "searchsorted list err", lambda: A.searchsorted([1.0, 2.0], 1.5)


# ---------------------------------------------------------------- tensordot, vecdot


def tensordot_vecdot(c):
    A = c.A
    for ka, kb in (("f", "f"), ("i", "i"), ("b", "b"), ("c", "c"), ("i", "f"), ("b", "c")):
        a, b = c.S(ka, (3, 4, 5), 500), c.S(kb, (4, 5, 2), 501)
        for nm, axes in (("2", 2), ("0", 0), ("pairs", ([1, 2], [0, 1])), ("pairs reversed", ([2, 1], [1, 0]))):
            yield f"tensordot {ka}{kb} axes={nm}", (lambda a=a, b=b, axes=axes: A.tensordot(a, b, axes=axes))
        yield f"tensordot {ka}{kb} axes=1", (lambda a=a, kb=kb: A.tensordot(a, c.S(kb, (5, 3), 502), axes=1))
    yield "tensordot default axes", lambda: A.tensordot(c.S("f", (3, 4, 5), 503), c.S("f", (4, 5), 504))
    yield "tensordot full to scalar", lambda: A.tensordot(c.S("f", (3, 4), 505), c.S("f", (3, 4), 506), axes=2)
    yield (
        "tensordot tuple pairs",
        lambda: A.tensordot(c.S("f", (3, 4), 505), c.S("f", (4, 3), 506), axes=((0, 1), (1, 0))),
    )
    yield "tensordot int pairs err", lambda: A.tensordot(c.S("f", (3, 4), 505), c.S("f", (4, 3), 506), axes=(1, 0))
    yield "tensordot F-order", lambda: A.tensordot(c.S("f", (5, 4), 507).T, c.S("f", (5, 3), 508), axes=1)
    yield "tensordot strided", lambda: A.tensordot(c.S("f", (8, 10), 509)[::2, ::-2], c.S("f", (5, 6), 510), axes=1)
    yield "tensordot empty", lambda: A.tensordot(c.S("f", (3, 0), 511), c.S("f", (0, 4), 512), axes=1)
    yield "tensordot int overflow", lambda: A.tensordot(c.lit([[INT_MAX, 2]]), c.lit([[2], [INT_MAX]]), axes=1)
    yield "tensordot large", lambda: A.tensordot(c.S("f", (30, 40, 50), 513), c.S("f", (40, 50, 60), 514), axes=2)
    yield "tensordot large outer", lambda: A.tensordot(c.S("f", (500,), 515), c.S("f", (600,), 516), axes=0)
    yield "tensordot large complex", lambda: A.tensordot(c.S("c", (100, 200), 517), c.S("c", (200, 150), 518), axes=1)
    yield "tensordot length err", lambda: A.tensordot(c.S("f", (3, 4), 519), c.S("f", (5, 3), 520), axes=1)
    yield (
        "tensordot dup axis err",
        lambda: A.tensordot(c.S("f", (3, 3), 519), c.S("f", (3, 3), 520), axes=([0, 0], [0, 1])),
    )
    yield "tensordot too many axes err", lambda: A.tensordot(c.S("f", (3,), 519), c.S("f", (3,), 520), axes=2)
    yield "tensordot negative axes err", lambda: A.tensordot(c.S("f", (3,), 519), c.S("f", (3,), 520), axes=-1)
    yield "tensordot pyscalar err", lambda: A.tensordot(c.S("f", (3,), 519), 2.0, axes=0)
    yield "tensordot list err", lambda: A.tensordot([1.0], c.S("f", (1,), 520), axes=1)
    for ka, kb in (("f", "f"), ("i", "i"), ("b", "b"), ("c", "c"), ("c", "f"), ("i", "c")):
        a, b = c.S(ka, (3, 4, 5), 521), c.S(kb, (3, 4, 5), 522)
        for ax in (-1, 0, 1):
            yield f"vecdot {ka}{kb} axis={ax}", (lambda a=a, b=b, ax=ax: A.vecdot(a, b, axis=ax))
    yield "vecdot 1d scalar", lambda: A.vecdot(c.lit([1, 2]), c.lit([3, 4]))
    yield "vecdot complex conj", lambda: A.vecdot(c.lit([1 + 2j, 3 - 1j]), c.lit([2 - 1j, 1j]))
    yield "vecdot bcast", lambda: A.vecdot(c.S("f", (4, 1, 5), 523), c.S("f", (3, 5), 524))
    yield (
        "vecdot special",
        lambda: A.vecdot(c.lit(F_SP[:20], "f").reshape((4, 5)), c.lit(F_SP[2:22], "f").reshape((4, 5))),
    )
    yield "vecdot F-order", lambda: A.vecdot(c.S("f", (5, 4), 525).T, c.S("f", (4, 5), 526), axis=0)
    yield "vecdot empty", lambda: A.vecdot(c.S("f", (3, 0), 527), c.S("f", (3, 0), 528))
    yield "vecdot large", lambda: A.vecdot(c.S("f", (1_000_000,), 529), c.S("f", (1_000_000,), 530))
    yield "vecdot large 2d", lambda: A.vecdot(c.S("f", (1000, 1000), 531), c.S("f", (1000, 1000), 532), axis=0)
    yield "vecdot large complex", lambda: A.vecdot(c.S("c", (300, 1000), 533), c.S("c", (300, 1000), 534))
    yield "vecdot large int", lambda: A.vecdot(c.S("I", (1000, 1000), 535), c.S("I", (1000, 1000), 536))
    yield "vecdot length err", lambda: A.vecdot(c.S("f", (3, 4), 537), c.S("f", (3, 5), 538))
    yield "vecdot axis bcast err", lambda: A.vecdot(c.S("f", (3, 4), 537), c.S("f", (3, 1), 538), axis=1)
    yield "vecdot axis oob err", lambda: A.vecdot(c.S("f", (3, 4), 537), c.S("f", (3, 4), 538), axis=2)
    yield "vecdot pyscalar err", lambda: A.vecdot(c.S("f", (3,), 537), 2.0)
    yield "vecdot list err", lambda: A.vecdot([1.0, 2.0], c.S("f", (2,), 538))


# ---------------------------------------------------------------- threads


def threads(c):
    A = c.A

    def roundtrip():
        before = A.get_num_threads()
        try:
            got = []
            for n in (1, 4, 7, 10):
                A.set_num_threads(n)
                got.append(A.get_num_threads())
            return got
        finally:
            A.set_num_threads(before)

    yield "threads set/get roundtrip", roundtrip
    yield "threads get returns int", lambda: type(A.get_num_threads()).__name__
    for bad in (0, -1, True, 2.0, 2000, None, "2"):
        yield f"threads set {bad!r} err", (lambda bad=bad: _set_threads_bad(A, bad))
    yield "threads set keyword err", lambda: _set_threads_bad(A, None, kw=True)
    yield "threads get args err", lambda: A.get_num_threads(1)


def _set_threads_bad(A, bad, kw=False):
    before = A.get_num_threads()
    try:
        return A.set_num_threads(n=2) if kw else A.set_num_threads(bad)
    finally:
        A.set_num_threads(before)


# ---------------------------------------------------------------- repr / str


def printing(c):
    A, np = c.A, c.np
    items = {
        "f": c.lit([1.5, -0.0, NAN, INF, -INF, 1e300, 5e-324, 0.1]),
        "i": c.lit([1, -22, 333, INT_MIN]),
        "b": c.lit([True, False]),
        "c": c.lit([1 + 2j, complex(NAN, -INF), -0.0j]),
        "2d": c.lit([[1.0, 2.0], [3.0, 4.5]]),
        "3d": c.lit(np.arange(12).reshape(2, 3, 2)),
        "4d": c.lit(np.arange(16.0).reshape(2, 2, 2, 2)),
        "empty": c.S("f", (0,), 290),
        "empty 2d": c.S("i", (2, 0), 290),
        "empty 3d": c.S("f", (2, 0, 3), 290),
        "1elem": c.lit([7]),
        "1elem 3d": c.lit([[[True]]]),
        "long row": c.lit(np.arange(40.0) * 1.1),
        "wrap ints": A.arange(100) * 1000,
        "summarized": A.arange(2000),
        "summarized 2d": A.reshape(A.arange(3000.0), (30, 100)),
        "summarized 3d": A.reshape(A.arange(2400), (12, 10, 20)),
        "exactly 1000": A.arange(1000),
        "1001": A.arange(1001),
        "F-order": c.lit(np.arange(6.0).reshape(2, 3)).T,
        "strided": A.arange(20)[::-3],
        "bcast": A.broadcast_to(c.lit([1, 2]), (3, 2)),
        "random": c.S("f", (3, 4), 291),
        "random c": c.S("c", (2, 3), 292),
        "big summarized random": c.S("f", (100, 100), 293),
    }
    for nm, x in items.items():
        yield f"repr {nm}", (lambda x=x: repr(x))
        yield f"str {nm}", (lambda x=x: str(x))
    yield "format", lambda: format(c.lit([1.0, 2.0]))
    yield "fstring", lambda: f"{c.lit([1, 2])}"
