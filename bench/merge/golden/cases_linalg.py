"""Golden cases for the linalg area (see golden.py for the contract).

Covers every name in stardust_linalg.__all__ (stardust-linalg/API.md, the frozen repository): all four
input dtypes, the input layouts the packing code tells apart (C, F, strided,
negative strides, stride-0 batch), tiny / medium / large (>= 2^17 elements:
the C-order transpose on the pool) matrices, batches big enough to run on the
pool, long vectors for the parallel pairwise sums, empty inputs, every keyword
option, and the documented errors (LinAlgError with the batch index, NaN/inf,
shapes, types).

Inputs are built with numpy from fixed seeds and handed over with L.arr
(zero copy, so the numpy strides are what the library sees).
"""

DTYPES = ("bool", "int64", "float64", "complex128")
FC = ("float64", "complex128")
LAYOUTS = ("C", "F", "strided", "reversed", "bcast")
INF = float("inf")
MATRIX_ORDS = ("fro", "nuc", 1, -1, 2, -2, INF, -INF)


class Raw:
    """An argument passed through as is (not converted to an L.A array)."""

    def __init__(self, v):
        self.v = v


def cases(L):
    np, A, LA = L.np, L.A, L.LA

    # ---------------------------------------------------------------- helpers
    def conv(a):
        if isinstance(a, Raw):
            return a.v
        if isinstance(a, np.ndarray):
            return L.arr(a)
        return a

    def C(name, f, *args, **kw):
        """A case calling f(*args, **kw); numpy arrays become L.A arrays."""
        args = tuple(conv(a) for a in args)
        kw = {k: conv(v) for k, v in kw.items()}
        return name, (lambda: f(*args, **kw))

    def vals(seed, shape, dt):
        r = L.rng(seed)
        if dt == "bool":
            return r.integers(0, 2, shape).astype(np.bool_)
        if dt == "int64":
            return r.integers(-3, 4, shape).astype(np.int64)
        if dt == "float64":
            return r.standard_normal(shape)
        return r.standard_normal(shape) + 1j * r.standard_normal(shape)

    def wc(seed, shape, dt):
        """Square matrices (..., n, n), safely nonsingular in every dtype."""
        n = shape[-1]
        if dt == "bool":  # unit lower triangular
            return np.tril(vals(seed, shape, "bool"), -1) | np.eye(n, dtype=np.bool_)
        return vals(seed, shape, dt) + (3 * n + 1) * np.eye(n, dtype=np.int64)

    def herm(seed, shape, dt):
        a = vals(seed, shape, dt)
        if dt == "bool":
            return a | np.swapaxes(a, -1, -2)
        return a + np.conj(np.swapaxes(a, -1, -2))

    def spd(seed, shape, dt):
        """Hermitian positive definite (bool: the identity)."""
        n = shape[-1]
        if dt == "bool":
            return np.ascontiguousarray(np.broadcast_to(np.eye(n, dtype=np.bool_), shape))
        b = vals(seed, shape, dt)
        return b @ np.conj(np.swapaxes(b, -1, -2)) + n * np.eye(n, dtype=np.int64)

    def lay(x, layout):
        """The same values as x (ndim >= 2) in another memory layout."""
        x = np.asarray(x)
        if layout == "C":
            return np.ascontiguousarray(x)
        if layout == "F":
            return np.swapaxes(np.ascontiguousarray(np.swapaxes(x, -1, -2)), -1, -2)
        if layout == "strided":
            big = np.zeros((*x.shape[:-2], 2 * x.shape[-2] + 1, 2 * x.shape[-1] + 1), x.dtype)
            big[..., 1::2, 1::2] = x
            return big[..., 1::2, 1::2]
        if layout == "reversed":
            return np.ascontiguousarray(x[..., ::-1, ::-1])[..., ::-1, ::-1]
        if layout == "bcast":  # batch axis 0 has stride 0
            return np.broadcast_to(x[:1], x.shape)
        raise ValueError(layout)

    def vlay(x, layout):
        """Vector layouts along the last axis."""
        x = np.asarray(x)
        if layout == "C":
            return np.ascontiguousarray(x)
        if layout == "strided":
            big = np.zeros((*x.shape[:-1], 3 * x.shape[-1] + 1), x.dtype)
            big[..., 1::3] = x
            return big[..., 1::3]
        return np.ascontiguousarray(x[..., ::-1])[..., ::-1]  # reversed

    def with_nan(x, idx, v=np.nan):
        y = np.array(x, dtype=np.result_type(x, np.float64))
        y[idx] = v
        return y

    def singular_batch(seed, batch, n, dt="float64", bad=(7,)):
        a = wc(seed, (batch, n, n), dt).astype(np.result_type(dt, np.float64))
        for i in bad:
            a[i] = 0
            a[i][:, 0] = 1  # rank 1
        return a

    # ================================================================ 1. basic
    # vecdot
    for dt in DTYPES:
        yield C(f"vecdot/{dt}", LA.vecdot, vals(1, (4, 7), dt), vals(2, (4, 7), dt))
    for d1, d2 in (("int64", "complex128"), ("bool", "float64"), ("complex128", "bool"), ("complex128", "complex128")):
        yield C(f"vecdot/mixed {d1} {d2} vec", LA.vecdot, vals(3, (9,), d1), vals(4, (9,), d2))
    x, y = vals(5, (5, 3), "complex128"), vals(6, (5, 3), "complex128")
    yield C("vecdot/axis=0", LA.vecdot, x, y, axis=0)
    yield C("vecdot/axis=-2", LA.vecdot, x, y, axis=-2)
    yield C("vecdot/broadcast", LA.vecdot, vals(7, (3, 1, 5), "float64"), vals(8, (4, 5), "float64"))
    yield C("vecdot/broadcast axis=0", LA.vecdot, vals(7, (5, 1), "float64"), vals(8, (5, 4), "float64"), axis=0)
    for n in (32767, 32768, 100_000):
        for dt in FC:
            yield C(f"vecdot/long {n} {dt}", LA.vecdot, vals(10 + n, (n,), dt), vals(11 + n, (n,), dt))
    yield C("vecdot/long int64", LA.vecdot, vals(12, (70_000,), "int64"), vals(13, (70_000,), "int64"))
    yield C("vecdot/long batch", LA.vecdot, vals(14, (3, 40_000), "float64"), vals(15, (3, 40_000), "float64"))
    yield C("vecdot/long axis=0", LA.vecdot, vals(14, (40_000, 3), "float64"), vals(15, (40_000, 3), "float64"), axis=0)
    for lo in ("strided", "reversed"):
        yield C(f"vecdot/long {lo}", LA.vecdot, vlay(vals(16, (50_000,), "complex128"), lo),
                vlay(vals(17, (50_000,), "complex128"), lo))
    yield C("vecdot/many short", LA.vecdot, vals(18, (20_000, 4), "float64"), vals(19, (20_000, 4), "float64"))
    yield C("vecdot/empty len 0", LA.vecdot, np.zeros(0), np.zeros(0))
    yield C("vecdot/empty (3,0)", LA.vecdot, np.zeros((3, 0)), np.zeros((3, 0)))
    yield C("vecdot/empty batch", LA.vecdot, np.zeros((0, 5)), np.ones((0, 5)))
    yield C("vecdot/nan", LA.vecdot, with_nan(vals(20, (6,), "float64"), 2), vals(21, (6,), "float64"))
    yield C("vecdot/inf", LA.vecdot, with_nan(vals(20, (6,), "float64"), 2, np.inf), vals(21, (6,), "float64"))
    yield C("vecdot/err size mismatch", LA.vecdot, np.ones(5), np.ones(4))
    yield C("vecdot/err axis out of bounds", LA.vecdot, np.ones((2, 3)), np.ones((2, 3)), axis=2)
    yield C("vecdot/err axis not int", LA.vecdot, np.ones((2, 3)), np.ones((2, 3)), axis=1.0)
    yield C("vecdot/err scalar arg", LA.vecdot, Raw(1.0), np.ones(3))
    yield C("vecdot/err numpy arg", LA.vecdot, Raw(np.ones(3)), np.ones(3))
    yield C("vecdot/err list arg 2", LA.vecdot, np.ones(3), Raw([1.0, 2.0, 3.0]))

    # outer
    for dt in DTYPES:
        yield C(f"outer/{dt}", LA.outer, vals(30, (4,), dt), vals(31, (5,), dt))
    yield C("outer/mixed bool complex", LA.outer, vals(32, (3,), "bool"), vals(33, (6,), "complex128"))
    yield C("outer/batch broadcast", LA.outer, vals(34, (2, 1, 3), "float64"), vals(35, (4, 5), "float64"))
    yield C("outer/strided reversed", LA.outer, vlay(vals(36, (7,), "float64"), "strided"),
            vlay(vals(37, (6,), "float64"), "reversed"))
    yield C("outer/large", LA.outer, vals(38, (600,), "float64"), vals(39, (500,), "float64"))
    yield C("outer/empty", LA.outer, np.zeros(0), np.ones(3))
    yield C("outer/nan", LA.outer, with_nan(np.ones(3), 1), np.ones(2))
    yield C("outer/err scalar arg", LA.outer, Raw(2.0), np.ones(3))
    yield C("outer/err batch mismatch", LA.outer, np.ones((2, 3)), np.ones((3, 3)))
    yield C("outer/err numpy arg", LA.outer, np.ones(3), Raw(np.ones(3)))

    # trace
    for dt in DTYPES:
        yield C(f"trace/{dt}", LA.trace, vals(40, (5, 5), dt))
    m = vals(41, (5, 6), "complex128")
    for k in (1, -2, 5, 6, -5, -6, 100):
        yield C(f"trace/k={k}", LA.trace, m, k=k)
    yield C("trace/dtype float64 from int", LA.trace, vals(42, (4, 4), "int64"), dtype=A.float64)
    yield C("trace/dtype complex128 from float", LA.trace, vals(42, (4, 4), "float64"), dtype=A.complex128)
    yield C("trace/dtype int64 from float", LA.trace, 3 * vals(42, (4, 4), "float64"), dtype=A.int64)
    yield C("trace/dtype int64 from bool", LA.trace, vals(42, (4, 4), "bool"), dtype=A.int64)
    yield C("trace/err dtype float64 from complex", LA.trace, vals(42, (4, 4), "complex128"), dtype=A.float64)
    yield C("trace/dtype None", LA.trace, vals(42, (4, 4), "int64"), dtype=None)
    for lo in LAYOUTS:
        yield C(f"trace/layout {lo}", LA.trace, lay(vals(43, (6, 4, 4), "float64"), lo))
    yield C("trace/batch 1000x3x3", LA.trace, vals(44, (1000, 3, 3), "float64"))
    yield C("trace/batch 100x32x32 k=3", LA.trace, vals(45, (100, 32, 32), "complex128"), k=3)
    yield C("trace/large 512 C", LA.trace, vals(46, (512, 512), "float64"))
    yield C("trace/large 512 F", LA.trace, lay(vals(46, (512, 512), "float64"), "F"))
    base = vals(47, (80_000,), "float64")
    long_view = np.lib.stride_tricks.as_strided(base, shape=(40_000, 40_000), strides=(8, 8))
    yield C("trace/long diagonal 40000 float", LA.trace, long_view)
    cbase = vals(48, (80_000,), "complex128")
    yield C("trace/long diagonal 40000 complex", LA.trace,
            np.lib.stride_tricks.as_strided(cbase, shape=(40_000, 40_000), strides=(16, 16)))
    yield C("trace/long diagonal 40000 k=7", LA.trace, long_view, k=7)
    yield C("trace/empty 0x0", LA.trace, np.zeros((0, 0)))
    yield C("trace/empty batch", LA.trace, np.zeros((0, 3, 3)))
    yield C("trace/nan", LA.trace, with_nan(np.eye(3), (1, 1)))
    yield C("trace/err ndim 1", LA.trace, np.ones(3))
    yield C("trace/err numpy dtype", LA.trace, np.eye(3), dtype=Raw(np.float64))
    yield C("trace/err k float", LA.trace, np.eye(3), k=Raw(1.0))
    yield C("trace/err numpy arg", LA.trace, Raw(np.eye(3)))

    # diagonal
    for dt in DTYPES:
        yield C(f"diagonal/{dt}", LA.diagonal, vals(50, (4, 5), dt))
    for k in (1, -1, 4, -4, 10):
        yield C(f"diagonal/k={k}", LA.diagonal, vals(51, (4, 5), "complex128"), k=k)
    for lo in LAYOUTS:
        yield C(f"diagonal/layout {lo}", LA.diagonal, lay(vals(52, (3, 5, 5), "int64"), lo), k=1)
    yield C("diagonal/batch 1000x3x3", LA.diagonal, vals(53, (1000, 3, 3), "float64"))
    yield C("diagonal/large 512 F", LA.diagonal, lay(vals(54, (512, 512), "float64"), "F"))
    yield C("diagonal/empty", LA.diagonal, np.zeros((0, 0)))
    yield C("diagonal/empty batch", LA.diagonal, np.zeros((0, 3, 4)))
    yield C("diagonal/err ndim 1", LA.diagonal, np.ones(3))
    yield C("diagonal/err numpy arg", LA.diagonal, Raw(np.eye(3)))

    # matrix_power
    m = wc(60, (4, 4), "float64") / 13
    for n in (0, 1, 2, 3, 7, 16, -1, -3):
        yield C(f"matrix_power/float n={n}", LA.matrix_power, m, n)
    for dt in DTYPES:
        yield C(f"matrix_power/{dt} n=3", LA.matrix_power, wc(61, (3, 3), dt), 3)
        yield C(f"matrix_power/{dt} n=-2", LA.matrix_power, wc(61, (3, 3), dt), -2)
        yield C(f"matrix_power/{dt} n=0", LA.matrix_power, wc(61, (3, 3), dt), 0)
    for lo in LAYOUTS:
        yield C(f"matrix_power/layout {lo} n=5", LA.matrix_power, lay(wc(62, (3, 4, 4), "float64") / 13, lo), 5)
        yield C(f"matrix_power/layout {lo} n=-1", LA.matrix_power, lay(wc(62, (3, 4, 4), "complex128"), lo), -1)
    yield C("matrix_power/batch 1000x3x3 n=5", LA.matrix_power, wc(63, (1000, 3, 3), "float64") / 10, 5)
    yield C("matrix_power/batch 1000x3x3 n=-2", LA.matrix_power, wc(63, (1000, 3, 3), "float64"), -2)
    yield C("matrix_power/batch 100x32x32 n=4", LA.matrix_power, wc(64, (100, 32, 32), "complex128") / 97, 4)
    yield C("matrix_power/large 300 n=2", LA.matrix_power, wc(65, (300, 300), "float64") / 901, 2)
    yield C("matrix_power/large 512 F n=-1", LA.matrix_power, lay(wc(65, (512, 512), "float64"), "F"), -1)
    yield C("matrix_power/singular n=2", LA.matrix_power, np.ones((3, 3)), 2)
    yield C("matrix_power/err singular n=-1", LA.matrix_power, np.ones((3, 3)), -1)
    yield C("matrix_power/err singular batch n=-1", LA.matrix_power, singular_batch(66, 1000, 3, bad=(7, 900)), -1)
    yield C("matrix_power/nan n=2", LA.matrix_power, with_nan(np.eye(3), (0, 1)), 2)
    yield C("matrix_power/err nan n=-1", LA.matrix_power, with_nan(np.eye(3), (0, 1)), -1)
    yield C("matrix_power/empty n=-1", LA.matrix_power, np.zeros((0, 0)), -1)
    yield C("matrix_power/empty n=0", LA.matrix_power, np.zeros((0, 0)), 0)
    yield C("matrix_power/empty batch n=2", LA.matrix_power, np.zeros((0, 3, 3)), 2)
    yield C("matrix_power/err nonsquare", LA.matrix_power, np.ones((2, 3)), 2)
    yield C("matrix_power/err ndim 1", LA.matrix_power, np.ones(3), 2)
    yield C("matrix_power/err n float", LA.matrix_power, np.eye(3), 2.0)
    yield C("matrix_power/err n bool", LA.matrix_power, np.eye(3), True)
    yield C("matrix_power/err numpy arg", LA.matrix_power, Raw(np.eye(3)), 2)

    # kron
    for dt in DTYPES:
        yield C(f"kron/{dt}", LA.kron, vals(70, (2, 3), dt), vals(71, (4, 5), dt))
    yield C("kron/mixed bool complex", LA.kron, vals(72, (2, 2), "bool"), vals(73, (3, 2), "complex128"))
    yield C("kron/batch broadcast", LA.kron, vals(74, (3, 2, 2), "float64"), vals(75, (2, 2), "float64"))
    yield C("kron/batch broadcast 4d", LA.kron, vals(76, (2, 1, 2, 3), "float64"), vals(77, (4, 2, 2), "float64"))
    for lo in ("F", "strided", "reversed"):
        yield C(f"kron/layout {lo}", LA.kron, lay(vals(78, (3, 4), "complex128"), lo), lay(vals(79, (2, 3), "float64"), lo))
    yield C("kron/large", LA.kron, vals(80, (30, 40), "float64"), vals(81, (20, 15), "float64"))
    yield C("kron/empty", LA.kron, np.zeros((0, 3)), np.ones((2, 2)))
    yield C("kron/nan", LA.kron, with_nan(np.ones((2, 2)), (0, 1)), np.ones((2, 2)))
    yield C("kron/err ndim 1", LA.kron, np.ones(3), np.ones((2, 2)))
    yield C("kron/err numpy arg", LA.kron, Raw(np.ones((2, 2))), np.ones((2, 2)))

    # cross
    for dt in DTYPES:
        yield C(f"cross/{dt}", LA.cross, vals(85, (5, 3), dt), vals(86, (5, 3), dt))
    yield C("cross/mixed int complex", LA.cross, vals(87, (3,), "int64"), vals(88, (3,), "complex128"))
    yield C("cross/axis=0", LA.cross, vals(89, (3, 4), "float64"), vals(90, (3, 4), "float64"), axis=0)
    yield C("cross/axis=-2 3d", LA.cross, vals(89, (2, 3, 4), "float64"), vals(90, (2, 3, 4), "float64"), axis=-2)
    yield C("cross/broadcast", LA.cross, vals(91, (4, 1, 3), "float64"), vals(92, (5, 3), "float64"))
    yield C("cross/strided reversed", LA.cross, lay(vals(93, (6, 3), "float64"), "strided"),
            lay(vals(94, (6, 3), "float64"), "reversed"))
    yield C("cross/large batch", LA.cross, vals(95, (100_000, 3), "float64"), vals(96, (100_000, 3), "float64"))
    yield C("cross/empty batch", LA.cross, np.zeros((0, 3)), np.zeros((0, 3)))
    yield C("cross/nan", LA.cross, with_nan(np.ones(3), 0), np.arange(3.0))
    yield C("cross/err axis length 2", LA.cross, np.ones(2), np.ones(2))
    yield C("cross/err batch mismatch", LA.cross, np.ones((3, 3)), np.ones((3, 4)), axis=0)
    yield C("cross/err axis lengths 3 and 4", LA.cross, np.ones(3), np.ones(4))
    yield C("cross/err axis out of bounds", LA.cross, np.ones(3), np.ones(3), axis=1)
    yield C("cross/err numpy arg", LA.cross, Raw(np.ones(3)), np.ones(3))

    # ======================================================= 2. norms, metrics
    # vector_norm
    v = vals(100, (4, 6), "float64")
    for o in (2, 1, INF, -INF, 0, 3, 0.5, -1, -2, 2.5):
        yield C(f"vector_norm/ord={o}", LA.vector_norm, v, ord=o)
    for dt in DTYPES:
        for o in (2, 1, INF, 0, 3):
            yield C(f"vector_norm/{dt} ord={o}", LA.vector_norm, vals(101, (4, 6), dt), ord=o)
    v3 = vals(102, (3, 4, 5), "complex128")
    for ax in (0, 1, -1, (0, 1), (0, 2), (2, 0), (0, 1, 2), None):
        for kd in (False, True):
            yield C(f"vector_norm/axis={ax} keepdims={kd}", LA.vector_norm, v3, axis=ax, keepdims=kd)
    yield C("vector_norm/axis=1 ord=inf keepdims", LA.vector_norm, v3, axis=1, ord=INF, keepdims=True)
    yield C("vector_norm/axis=(0,2) ord=1", LA.vector_norm, v3, axis=(0, 2), ord=1)
    for n in (32767, 32768, 100_000):
        for dt in FC:
            for o in (2, 1, INF, 3, 0, -INF):
                yield C(f"vector_norm/long {n} {dt} ord={o}", LA.vector_norm, vals(103 + n, (n,), dt), ord=o)
    yield C("vector_norm/long int64", LA.vector_norm, vals(104, (70_000,), "int64"))
    yield C("vector_norm/long bool ord=1", LA.vector_norm, vals(104, (70_000,), "bool"), ord=1)
    for lo in ("strided", "reversed"):
        yield C(f"vector_norm/long {lo}", LA.vector_norm, vlay(vals(105, (60_000,), "float64"), lo))
    yield C("vector_norm/long rows axis=-1", LA.vector_norm, vals(106, (3, 50_000), "float64"), axis=-1)
    yield C("vector_norm/long cols axis=0", LA.vector_norm, vals(106, (50_000, 3), "float64"), axis=0)
    yield C("vector_norm/many short axis=-1", LA.vector_norm, vals(107, (20_000, 5), "complex128"), axis=-1)
    yield C("vector_norm/large 2d axis None", LA.vector_norm, vals(108, (300, 400), "float64"))
    yield C("vector_norm/large 2d F axis None", LA.vector_norm, lay(vals(108, (300, 400), "float64"), "F"))
    for o in (2, 1, 3):
        yield C(f"vector_norm/overflow ord={o}", LA.vector_norm, np.full(5, 1e200), ord=o)
        yield C(f"vector_norm/underflow ord={o}", LA.vector_norm, np.full(5, 1e-200), ord=o)
        yield C(f"vector_norm/overflow long ord={o}", LA.vector_norm, 1e300 * vals(109, (40_000,), "float64"), ord=o)
    yield C("vector_norm/overflow complex", LA.vector_norm, np.full(4, 1e200 + 1e200j))
    yield C("vector_norm/max finite", LA.vector_norm, np.array([1.7e308, 1.7e308]))
    for o in (2, 1, INF, -INF, 0, 3):
        yield C(f"vector_norm/nan ord={o}", LA.vector_norm, with_nan(np.arange(5.0), 2), ord=o)
        yield C(f"vector_norm/inf ord={o}", LA.vector_norm, with_nan(np.arange(5.0), 2, np.inf), ord=o)
        yield C(f"vector_norm/nan and inf ord={o}", LA.vector_norm, np.array([1.0, np.inf, np.nan]), ord=o)
    yield C("vector_norm/nan long", LA.vector_norm, with_nan(vals(110, (40_000,), "float64"), 39_000))
    yield C("vector_norm/zeros ord=-1", LA.vector_norm, np.zeros(4), ord=-1)
    yield C("vector_norm/zeros ord=2", LA.vector_norm, np.zeros(4))
    for o in (2, 1, 0, INF, -INF):
        yield C(f"vector_norm/empty ord={o}", LA.vector_norm, np.zeros(0), ord=o)
    yield C("vector_norm/empty axis=0", LA.vector_norm, np.zeros((0, 3)), axis=0)
    yield C("vector_norm/empty axis=1", LA.vector_norm, np.zeros((0, 3)), axis=1)
    yield C("vector_norm/err ord fro", LA.vector_norm, np.ones(3), ord="fro")
    yield C("vector_norm/err ord nan", LA.vector_norm, np.ones(3), ord=float("nan"))
    yield C("vector_norm/err repeated axis", LA.vector_norm, np.ones((2, 3)), axis=(1, -1))
    yield C("vector_norm/err axis out of bounds", LA.vector_norm, np.ones((2, 3)), axis=2)
    yield C("vector_norm/err numpy arg", LA.vector_norm, Raw(np.ones(3)))

    # matrix_norm
    for dt in FC:
        for o in MATRIX_ORDS:
            yield C(f"matrix_norm/{dt} (5,4) ord={o}", LA.matrix_norm, vals(120, (5, 4), dt), ord=o)
            yield C(f"matrix_norm/{dt} (3,6) ord={o}", LA.matrix_norm, vals(121, (3, 6), dt), ord=o)
    for dt in ("bool", "int64"):
        for o in MATRIX_ORDS:
            yield C(f"matrix_norm/{dt} ord={o}", LA.matrix_norm, vals(122, (3, 3), dt), ord=o)
    for o in MATRIX_ORDS:
        yield C(f"matrix_norm/keepdims ord={o}", LA.matrix_norm, vals(123, (2, 3, 4), "float64"), ord=o, keepdims=True)
        yield C(f"matrix_norm/batch 1000x3x3 ord={o}", LA.matrix_norm, vals(124, (1000, 3, 3), "float64"), ord=o)
    for o in ("fro", 2, "nuc", 1):
        yield C(f"matrix_norm/batch 100x32x32 ord={o}", LA.matrix_norm, vals(125, (100, 32, 32), "complex128"), ord=o)
    for lo in LAYOUTS:
        for o in ("fro", 1, INF, 2):
            yield C(f"matrix_norm/layout {lo} ord={o}", LA.matrix_norm, lay(vals(126, (3, 5, 4), "float64"), lo), ord=o)
    for lo in ("C", "F"):
        for o in ("fro", 1, -1, INF, -INF, 2):
            yield C(f"matrix_norm/large 512 {lo} ord={o}", LA.matrix_norm, lay(vals(127, (512, 512), "float64"), lo), ord=o)
    yield C("matrix_norm/large 300x700 complex fro", LA.matrix_norm, vals(128, (300, 700), "complex128"))
    yield C("matrix_norm/overflow fro", LA.matrix_norm, np.full((3, 3), 1e200))
    yield C("matrix_norm/overflow 1", LA.matrix_norm, np.full((3, 3), 1e308), ord=1)
    yield C("matrix_norm/underflow fro", LA.matrix_norm, np.full((3, 3), 1e-200))
    for o in MATRIX_ORDS:
        nan_tag = "err nan" if o in (2, -2, "nuc") else "nan"  # the SVD ords reject NaN (API.md §2)
        yield C(f"matrix_norm/{nan_tag} ord={o}", LA.matrix_norm, with_nan(np.eye(3), (1, 2)), ord=o)
        yield C(f"matrix_norm/empty 0x0 ord={o}", LA.matrix_norm, np.zeros((0, 0)), ord=o)
        yield C(f"matrix_norm/empty 0x3 ord={o}", LA.matrix_norm, np.zeros((0, 3)), ord=o)
        yield C(f"matrix_norm/empty batch ord={o}", LA.matrix_norm, np.zeros((0, 3, 3)), ord=o)
    yield C("matrix_norm/inf fro", LA.matrix_norm, with_nan(np.eye(3), (1, 2), np.inf))
    yield C("matrix_norm/err ord=3", LA.matrix_norm, np.eye(3), ord=3)
    yield C("matrix_norm/err ord=foo", LA.matrix_norm, np.eye(3), ord="foo")
    yield C("matrix_norm/err ndim 1", LA.matrix_norm, np.ones(3))
    yield C("matrix_norm/err numpy arg", LA.matrix_norm, Raw(np.eye(3)))

    # cond
    for dt in FC:
        for o in MATRIX_ORDS:
            yield C(f"cond/{dt} ord={o}", LA.cond, wc(130, (4, 4), dt) / 7, ord=o)
    for dt in ("bool", "int64"):
        for o in (2, 1, "fro"):
            yield C(f"cond/{dt} ord={o}", LA.cond, wc(131, (3, 3), dt), ord=o)
    for o in (2, -2):
        yield C(f"cond/nonsquare (5,3) ord={o}", LA.cond, vals(132, (5, 3), "float64"), ord=o)
        yield C(f"cond/nonsquare (3,5) ord={o}", LA.cond, vals(132, (3, 5), "complex128"), ord=o)
    for o in MATRIX_ORDS:
        yield C(f"cond/singular ord={o}", LA.cond, np.ones((3, 3)), ord=o)
        yield C(f"cond/singular zeros ord={o}", LA.cond, np.zeros((3, 3)), ord=o)
        yield C(f"cond/batch 1000x3x3 ord={o}", LA.cond, wc(133, (1000, 3, 3), "float64"), ord=o)
    yield C("cond/singular batch ord=1", LA.cond, singular_batch(134, 1000, 3, bad=(7, 900)), ord=1)
    yield C("cond/batch 100x32x32", LA.cond, wc(135, (100, 32, 32), "complex128"))
    for lo in LAYOUTS:
        yield C(f"cond/layout {lo}", LA.cond, lay(wc(136, (3, 4, 4), "float64"), lo), ord=1)
    yield C("cond/large 300", LA.cond, wc(137, (300, 300), "float64"))
    yield C("cond/large 512 C ord=1", LA.cond, wc(137, (512, 512), "float64"), ord=1)
    yield C("cond/err empty", LA.cond, np.zeros((0, 0)))
    yield C("cond/empty batch", LA.cond, np.zeros((0, 3, 3)))
    yield C("cond/err nonsquare ord=1", LA.cond, np.ones((3, 5)), ord=1)
    yield C("cond/err nonsquare ord=fro", LA.cond, np.ones((3, 5)), ord="fro")
    yield C("cond/err nan ord=2", LA.cond, with_nan(np.eye(3), (0, 0)))
    yield C("cond/err nan ord=1", LA.cond, with_nan(np.eye(3), (0, 0)), ord=1)
    yield C("cond/err ord=3", LA.cond, np.eye(3), ord=3)
    yield C("cond/err ndim 1", LA.cond, np.ones(3))
    yield C("cond/err numpy arg", LA.cond, Raw(np.eye(3)))

    # det, slogdet
    for fn in ("det", "slogdet"):
        f = getattr(LA, fn)
        for dt in DTYPES:
            for n in (1, 2, 3, 32):
                yield C(f"{fn}/{dt} n={n}", f, wc(140 + n, (n, n), dt))
            yield C(f"{fn}/{dt} random 4x4", f, vals(145, (4, 4), dt))
        for lo in LAYOUTS:
            yield C(f"{fn}/layout {lo}", f, lay(wc(146, (5, 4, 4), "complex128"), lo))
        yield C(f"{fn}/batch 1000x3x3", f, vals(147, (1000, 3, 3), "float64"))
        yield C(f"{fn}/batch 1000x3x3 complex", f, vals(147, (1000, 3, 3), "complex128"))
        yield C(f"{fn}/batch 100x32x32", f, vals(148, (100, 32, 32), "float64"))
        yield C(f"{fn}/batch broadcast", f, lay(vals(149, (7, 3, 3), "float64"), "bcast"))
        yield C(f"{fn}/large 300", f, vals(150, (300, 300), "float64") / 10)
        yield C(f"{fn}/large 300 complex", f, vals(150, (300, 300), "complex128") / 10)
        yield C(f"{fn}/large 512 C", f, vals(151, (512, 512), "float64") / 20)
        yield C(f"{fn}/large 512 F", f, lay(vals(151, (512, 512), "float64") / 20, "F"))
        yield C(f"{fn}/overflow", f, 1e10 * np.eye(200))
        yield C(f"{fn}/negative", f, np.diag([-1.0, 2.0, 3.0]))
        yield C(f"{fn}/singular zeros", f, np.zeros((3, 3)))
        yield C(f"{fn}/singular int rank 1", f, np.array([[1, 2], [2, 4]]))
        yield C(f"{fn}/singular complex", f, np.ones((3, 3), complex))
        yield C(f"{fn}/singular batch", f, singular_batch(152, 1000, 3, bad=(7, 900)))
        yield C(f"{fn}/empty 0x0", f, np.zeros((0, 0)))
        yield C(f"{fn}/empty 0x0 complex", f, np.zeros((0, 0), complex))
        yield C(f"{fn}/empty batch (0,3,3)", f, np.zeros((0, 3, 3)))
        yield C(f"{fn}/empty batch (3,0,0)", f, np.zeros((3, 0, 0)))
        yield C(f"{fn}/err nan", f, with_nan(np.eye(3), (2, 1)))
        yield C(f"{fn}/err inf", f, with_nan(np.eye(3), (2, 1), -np.inf))
        nb = wc(153, (1000, 3, 3), "float64")
        nb[600, 1, 1] = np.nan
        nb[800, 0, 0] = np.inf
        yield C(f"{fn}/err nan batch", f, nb)
        yield C(f"{fn}/err nonsquare", f, np.ones((2, 3)))
        yield C(f"{fn}/err ndim 1", f, np.ones(3))
        yield C(f"{fn}/err numpy arg", f, Raw(np.eye(3)))
        yield C(f"{fn}/err none arg", f, Raw(None))

    # matrix_rank
    for dt in DTYPES:
        yield C(f"matrix_rank/{dt} full", LA.matrix_rank, wc(160, (5, 5), dt))
        yield C(f"matrix_rank/{dt} random (4,6)", LA.matrix_rank, vals(161, (4, 6), dt))
    low = vals(162, (6, 3), "float64") @ vals(163, (3, 6), "float64")
    for rt in (None, 0, 0.0, 1e-20, 0.1, 0.999, 1, 1.0, 2.5):
        yield C(f"matrix_rank/rank 3 rtol={rt}", LA.matrix_rank, low, rtol=rt)
    yield C("matrix_rank/complex rank 2", LA.matrix_rank,
            vals(164, (5, 2), "complex128") @ vals(165, (2, 7), "complex128"))
    yield C("matrix_rank/zeros", LA.matrix_rank, np.zeros((3, 4)))
    bm = vals(166, (1000, 3, 3), "float64")
    bm[::3, 2] = bm[::3, 0]
    bm[::7] = 0
    yield C("matrix_rank/batch 1000x3x3 mixed", LA.matrix_rank, bm)
    yield C("matrix_rank/batch 100x32x32", LA.matrix_rank, vals(167, (100, 32, 32), "complex128"))
    for lo in LAYOUTS:
        yield C(f"matrix_rank/layout {lo}", LA.matrix_rank, lay(vals(168, (3, 5, 4), "float64"), lo))
    yield C("matrix_rank/large 300", LA.matrix_rank, vals(169, (300, 300), "float64"))
    yield C("matrix_rank/large 512 C rank-deficient", LA.matrix_rank,
            vals(170, (512, 100), "float64") @ vals(171, (100, 512), "float64"))
    yield C("matrix_rank/empty 0x0", LA.matrix_rank, np.zeros((0, 0)))
    yield C("matrix_rank/empty 0x3", LA.matrix_rank, np.zeros((0, 3)))
    yield C("matrix_rank/empty batch", LA.matrix_rank, np.zeros((0, 3, 3)))
    yield C("matrix_rank/err rtol negative", LA.matrix_rank, np.eye(3), rtol=-1.0)
    yield C("matrix_rank/err rtol nan", LA.matrix_rank, np.eye(3), rtol=float("nan"))
    yield C("matrix_rank/err rtol inf", LA.matrix_rank, np.eye(3), rtol=INF)
    yield C("matrix_rank/err rtol str", LA.matrix_rank, np.eye(3), rtol="a")
    yield C("matrix_rank/err rtol bool", LA.matrix_rank, np.eye(3), rtol=True)
    yield C("matrix_rank/err nan", LA.matrix_rank, with_nan(np.eye(3), (0, 2)))
    yield C("matrix_rank/err ndim 1", LA.matrix_rank, np.ones(3))
    yield C("matrix_rank/err numpy arg", LA.matrix_rank, Raw(np.eye(3)))

    # ===================================================== 3. linear systems
    # solve
    for dt in DTYPES:
        yield C(f"solve/{dt} vec", LA.solve, wc(200, (4, 4), dt), vals(201, (4,), dt))
        yield C(f"solve/{dt} mat", LA.solve, wc(200, (4, 4), dt), vals(202, (4, 3), dt))
    for da, db in (("bool", "complex128"), ("complex128", "int64"), ("int64", "float64"), ("float64", "complex128")):
        yield C(f"solve/mixed {da} {db}", LA.solve, wc(203, (3, 3), da), vals(204, (3, 2), db))
    for n in (1, 2, 3, 32):
        for dt in FC:
            yield C(f"solve/n={n} {dt}", LA.solve, wc(205 + n, (n, n), dt), vals(206 + n, (n, 3), dt))
    for lo in LAYOUTS:
        yield C(f"solve/layout {lo}", LA.solve, lay(wc(210, (6, 4, 4), "float64"), lo), lay(vals(211, (6, 4, 2), "float64"), lo))
        yield C(f"solve/layout {lo} complex", LA.solve, lay(wc(210, (6, 4, 4), "complex128"), lo),
                lay(vals(211, (6, 4, 2), "complex128"), lo))
    yield C("solve/broadcast a (3,1,4,4) b (5,4,2)", LA.solve, wc(212, (3, 1, 4, 4), "float64"), vals(213, (5, 4, 2), "float64"))
    yield C("solve/broadcast a (4,4) b (7,4,2)", LA.solve, wc(212, (4, 4), "float64"), vals(213, (7, 4, 2), "float64"))
    yield C("solve/broadcast a (6,4,4) b vec", LA.solve, wc(212, (6, 4, 4), "float64"), vals(213, (4,), "float64"))
    yield C("solve/b (6,4) as matrix", LA.solve, wc(212, (6, 6), "float64"), vals(213, (6, 4), "float64"))
    yield C("solve/batch 1000x3x3", LA.solve, wc(214, (1000, 3, 3), "float64"), vals(215, (1000, 3, 1), "float64"))
    yield C("solve/batch 1000x3x3 complex vec", LA.solve, wc(214, (1000, 3, 3), "complex128"), vals(215, (3,), "complex128"))
    yield C("solve/batch 100x32x32", LA.solve, wc(216, (100, 32, 32), "float64"), vals(217, (100, 32, 4), "float64"))
    for lo in ("C", "F", "reversed"):
        yield C(f"solve/large 300 {lo}", LA.solve, lay(vals(218, (300, 300), "float64"), lo), vals(219, (300, 5), "float64"))
        yield C(f"solve/large 512 {lo}", LA.solve, lay(vals(220, (512, 512), "float64"), lo), lay(vals(221, (512, 512), "float64"), lo))
    yield C("solve/large 300 complex", LA.solve, vals(222, (300, 300), "complex128"), vals(223, (300,), "complex128"))
    yield C("solve/large 512 C int", LA.solve, vals(224, (512, 512), "int64"), vals(225, (512, 2), "int64"))
    yield C("solve/err singular", LA.solve, np.ones((3, 3)), np.ones(3))
    yield C("solve/err singular batch", LA.solve, singular_batch(226, 1000, 3, bad=(7, 900)), np.ones(3))
    yield C("solve/err singular broadcast batch", LA.solve, singular_batch(227, 4, 3, bad=(2,)).reshape(4, 1, 3, 3),
            np.ones((5, 3, 1)))
    yield C("solve/err nan a", LA.solve, with_nan(np.eye(3), (1, 0)), np.ones(3))
    yield C("solve/err nan b", LA.solve, np.eye(3), with_nan(np.ones(3), 1))
    yield C("solve/err inf b batch", LA.solve, wc(228, (50, 3, 3), "float64"), with_nan(np.ones((50, 3, 1)), (33, 0, 0), np.inf))
    yield C("solve/empty 0x0 vec", LA.solve, np.zeros((0, 0)), np.zeros(0))
    yield C("solve/empty batch", LA.solve, np.zeros((0, 3, 3)), np.zeros((0, 3, 2)))
    yield C("solve/empty rhs", LA.solve, np.eye(3), np.zeros((3, 0)))
    yield C("solve/err shape mismatch", LA.solve, np.eye(3), np.ones(4))
    yield C("solve/err shape mismatch mat", LA.solve, np.eye(3), np.ones((4, 2)))
    yield C("solve/err batch mismatch", LA.solve, np.ones((2, 3, 3)), np.ones((3, 3, 1)))
    yield C("solve/err nonsquare", LA.solve, np.ones((2, 3)), np.ones(2))
    yield C("solve/err ndim 1", LA.solve, np.ones(3), np.ones(3))
    yield C("solve/err b scalar", LA.solve, np.eye(3), Raw(1.0))
    yield C("solve/err numpy a", LA.solve, Raw(np.eye(3)), np.ones(3))
    yield C("solve/err list b", LA.solve, np.eye(3), Raw([1.0, 2.0, 3.0]))

    # solve_triangular
    for dt in DTYPES:
        for up in (False, True):
            t = wc(230, (4, 4), dt)
            yield C(f"solve_triangular/{dt} upper={up}", LA.solve_triangular, t, vals(231, (4, 2), dt), upper=up)
            yield C(f"solve_triangular/{dt} upper={up} unit", LA.solve_triangular, t, vals(231, (4,), dt), upper=up,
                    unit_diagonal=True)
    for up in (False, True):
        g = wc(232, (5, 5), "complex128")
        idx = np.triu_indices(5, 1) if not up else np.tril_indices(5, -1)
        g[idx] = np.nan  # the unread triangle
        yield C(f"solve_triangular/nan in unread triangle upper={up}", LA.solve_triangular, g, vals(233, (5, 2), "complex128"), upper=up)
        z = wc(234, (4, 4), "float64")
        z[2, 2] = 0
        yield C(f"solve_triangular/err zero diagonal upper={up}", LA.solve_triangular, z, np.ones(4), upper=up)
        yield C(f"solve_triangular/zero diagonal unit upper={up}", LA.solve_triangular, z, np.ones(4), upper=up, unit_diagonal=True)
        nd = with_nan(np.eye(4), (2, 2))
        yield C(f"solve_triangular/nan diagonal unit upper={up}", LA.solve_triangular, nd, np.ones(4), upper=up, unit_diagonal=True)
        yield C(f"solve_triangular/err nan diagonal upper={up}", LA.solve_triangular, nd, np.ones(4), upper=up)
        rd = with_nan(np.eye(4), (3, 0) if not up else (0, 3))
        yield C(f"solve_triangular/err nan read triangle upper={up}", LA.solve_triangular, rd, np.ones(4), upper=up)
        for lo in LAYOUTS:
            yield C(f"solve_triangular/layout {lo} upper={up}", LA.solve_triangular, lay(wc(235, (3, 5, 5), "float64"), lo),
                    lay(vals(236, (3, 5, 3), "float64"), lo), upper=up)
        for lo in ("C", "F", "reversed"):
            yield C(f"solve_triangular/large 512 {lo} upper={up}", LA.solve_triangular, lay(wc(237, (512, 512), "float64"), lo),
                    vals(238, (512, 4), "float64"), upper=up)
        yield C(f"solve_triangular/large 300 upper={up}", LA.solve_triangular, wc(237, (300, 300), "complex128"),
                vals(238, (300,), "complex128"), upper=up)
        yield C(f"solve_triangular/batch 1000x3x3 upper={up}", LA.solve_triangular, wc(239, (1000, 3, 3), "float64"),
                vals(240, (1000, 3, 2), "float64"), upper=up)
        yield C(f"solve_triangular/batch 100x32x32 upper={up}", LA.solve_triangular, wc(241, (100, 32, 32), "float64"),
                vals(242, (32,), "float64"), upper=up)
    zb = wc(243, (1000, 3, 3), "float64")
    zb[7, 1, 1] = 0
    zb[900, 0, 0] = 0
    yield C("solve_triangular/err zero diagonal batch", LA.solve_triangular, zb, np.ones(3), upper=True)
    yield C("solve_triangular/mixed int complex", LA.solve_triangular, wc(244, (3, 3), "int64"), vals(245, (3,), "complex128"), upper=False)
    yield C("solve_triangular/empty", LA.solve_triangular, np.zeros((0, 0)), np.zeros((0, 2)), upper=True)
    yield C("solve_triangular/empty batch", LA.solve_triangular, np.zeros((0, 2, 2)), np.zeros(2), upper=False)
    yield C("solve_triangular/err nan b", LA.solve_triangular, np.eye(3), with_nan(np.ones(3), 0), upper=True)
    yield C("solve_triangular/err upper missing", LA.solve_triangular, np.eye(3), np.ones(3))
    yield C("solve_triangular/err shape mismatch", LA.solve_triangular, np.eye(3), np.ones(2), upper=True)
    yield C("solve_triangular/err nonsquare", LA.solve_triangular, np.ones((3, 2)), np.ones(3), upper=True)
    yield C("solve_triangular/err numpy arg", LA.solve_triangular, np.eye(3), Raw(np.ones(3)), upper=True)

    # cholesky
    for dt in DTYPES:
        for up in (False, True):
            for n in (1, 2, 3, 32):
                yield C(f"cholesky/{dt} n={n} upper={up}", LA.cholesky, spd(250 + n, (n, n), dt), upper=up)
    for up in (False, True):
        for lo in LAYOUTS:
            yield C(f"cholesky/layout {lo} upper={up}", LA.cholesky, lay(spd(255, (3, 5, 5), "complex128"), lo), upper=up)
        for lo in ("C", "F", "reversed"):
            yield C(f"cholesky/large 512 {lo} upper={up}", LA.cholesky, lay(spd(256, (512, 512), "float64"), lo), upper=up)
        yield C(f"cholesky/large 300 complex upper={up}", LA.cholesky, spd(257, (300, 300), "complex128"), upper=up)
        yield C(f"cholesky/batch 1000x3x3 upper={up}", LA.cholesky, spd(258, (1000, 3, 3), "float64"), upper=up)
        yield C(f"cholesky/batch 100x32x32 upper={up}", LA.cholesky, spd(259, (100, 32, 32), "complex128"), upper=up)
    g = spd(260, (5, 5), "float64")
    g[np.triu_indices(5, 1)] = np.nan
    yield C("cholesky/nan in upper triangle", LA.cholesky, g)
    yield C("cholesky/nan in upper triangle upper=True", LA.cholesky, g, upper=True)
    yield C("cholesky/err not positive definite", LA.cholesky, -np.eye(3))
    yield C("cholesky/err indefinite", LA.cholesky, np.array([[1.0, 2.0], [2.0, 1.0]]))
    yield C("cholesky/err bool psd", LA.cholesky, np.array([[True, False], [True, True]]))
    bb = spd(261, (1000, 3, 3), "float64")
    bb[7] = -np.eye(3)
    bb[900] = -np.eye(3)
    yield C("cholesky/err not pd batch", LA.cholesky, bb)
    yield C("cholesky/err nan lower", LA.cholesky, with_nan(np.eye(3), (2, 0)))
    yield C("cholesky/empty 0x0", LA.cholesky, np.zeros((0, 0)))
    yield C("cholesky/empty batch", LA.cholesky, np.zeros((0, 3, 3)), upper=True)
    yield C("cholesky/err nonsquare", LA.cholesky, np.ones((2, 3)))
    yield C("cholesky/err ndim 1", LA.cholesky, np.ones(3))
    yield C("cholesky/err upper not bool", LA.cholesky, np.eye(3), upper=Raw("yes"))
    yield C("cholesky/err numpy arg", LA.cholesky, Raw(np.eye(3)))

    # cholesky_solve
    for dt in DTYPES:
        for up in (False, True):
            t = wc(270, (4, 4), dt)
            t = np.triu(t) if up else np.tril(t)
            yield C(f"cholesky_solve/{dt} upper={up}", LA.cholesky_solve, t, vals(271, (4, 2), dt), upper=up)
    for up in (False, True):
        for lo in LAYOUTS:
            t = wc(272, (3, 5, 5), "complex128")
            t = np.triu(t) if up else np.tril(t)
            yield C(f"cholesky_solve/layout {lo} upper={up}", LA.cholesky_solve, lay(t, lo), lay(vals(273, (3, 5, 2), "complex128"), lo), upper=up)
        g = wc(274, (4, 4), "float64")
        g[np.tril_indices(4, -1) if up else np.triu_indices(4, 1)] = np.nan
        yield C(f"cholesky_solve/nan unread triangle upper={up}", LA.cholesky_solve, g, np.ones(4), upper=up)
        z = np.eye(4)
        z[1, 1] = 0
        yield C(f"cholesky_solve/err zero diagonal upper={up}", LA.cholesky_solve, z, np.ones(4), upper=up)
        for lo in ("C", "F"):
            t = wc(275, (512, 512), "float64")
            t = np.triu(t) if up else np.tril(t)
            yield C(f"cholesky_solve/large 512 {lo} upper={up}", LA.cholesky_solve, lay(t, lo), vals(276, (512, 3), "float64"), upper=up)
        yield C(f"cholesky_solve/batch 1000x3x3 upper={up}", LA.cholesky_solve, wc(277, (1000, 3, 3), "float64"),
                vals(278, (3,), "float64"), upper=up)
        yield C(f"cholesky_solve/batch 100x32x32 upper={up}", LA.cholesky_solve, wc(279, (100, 32, 32), "complex128"),
                vals(280, (100, 32, 2), "complex128"), upper=up)
    yield C("cholesky_solve/from cholesky", lambda: LA.cholesky_solve(LA.cholesky(L.arr(spd(281, (6, 6), "float64"))),
                                                                        L.arr(vals(282, (6,), "float64"))))
    yield ("cholesky_solve/from cholesky upper",
           lambda: LA.cholesky_solve(LA.cholesky(L.arr(spd(281, (6, 6), "complex128")), upper=True),
                                     L.arr(vals(282, (6, 2), "complex128")), upper=True))
    yield C("cholesky_solve/err nan read", LA.cholesky_solve, with_nan(np.eye(3), (2, 1)), np.ones(3))
    yield C("cholesky_solve/err nan b", LA.cholesky_solve, np.eye(3), with_nan(np.ones(3), 2))
    yield C("cholesky_solve/empty", LA.cholesky_solve, np.zeros((0, 0)), np.zeros(0))
    yield C("cholesky_solve/err shape mismatch", LA.cholesky_solve, np.eye(3), np.ones(4))
    yield C("cholesky_solve/err nonsquare", LA.cholesky_solve, np.ones((3, 2)), np.ones(3))
    yield C("cholesky_solve/err numpy arg", LA.cholesky_solve, Raw(np.eye(3)), np.ones(3))

    # lu
    for dt in DTYPES:
        for shp in ((4, 4), (6, 4), (4, 6), (1, 1)):
            yield C(f"lu/{dt} {shp}", LA.lu, vals(290, shp, dt))
    for lo in LAYOUTS:
        yield C(f"lu/layout {lo}", LA.lu, lay(vals(291, (3, 5, 4), "complex128"), lo))
    yield C("lu/singular", LA.lu, np.ones((3, 3)))
    yield C("lu/singular zeros", LA.lu, np.zeros((4, 4)))
    yield C("lu/batch 1000x3x3", LA.lu, vals(292, (1000, 3, 3), "float64"))
    yield C("lu/batch 100x32x32", LA.lu, vals(293, (100, 32, 32), "complex128"))
    yield C("lu/batch 100x6x3", LA.lu, vals(294, (100, 6, 3), "float64"))
    for lo in ("C", "F", "reversed"):
        yield C(f"lu/large 512 {lo}", LA.lu, lay(vals(295, (512, 512), "float64"), lo))
    yield C("lu/large 300x500", LA.lu, vals(296, (300, 500), "float64"))
    yield C("lu/large 700x200 complex", LA.lu, vals(297, (700, 200), "complex128"))
    for shp in ((0, 0), (0, 3), (3, 0), (0, 3, 3), (2, 0, 4)):
        yield C(f"lu/empty {shp}", LA.lu, np.zeros(shp))
    yield C("lu/err nan", LA.lu, with_nan(np.eye(3), (1, 1)))
    yield C("lu/err ndim 1", LA.lu, np.ones(3))
    yield C("lu/err numpy arg", LA.lu, Raw(np.eye(3)))

    # lu_solve
    def lu_then_solve(a, b):
        a, b = L.arr(a), L.arr(b)
        return lambda: LA.lu_solve(*LA.lu(a), b)

    for dt in DTYPES:
        yield f"lu_solve/{dt} via lu", lu_then_solve(wc(300, (4, 4), dt), vals(301, (4, 2), dt))
        yield f"lu_solve/{dt} via lu vec", lu_then_solve(wc(300, (4, 4), dt), vals(301, (4,), dt))
    yield "lu_solve/batch 1000x3x3", lu_then_solve(wc(302, (1000, 3, 3), "float64"), vals(303, (1000, 3, 2), "float64"))
    yield "lu_solve/batch 100x32x32 complex", lu_then_solve(wc(304, (100, 32, 32), "complex128"), vals(305, (32,), "complex128"))
    yield "lu_solve/large 512", lu_then_solve(vals(306, (512, 512), "float64"), vals(307, (512, 3), "float64"))
    yield "lu_solve/large 300 complex", lu_then_solve(vals(308, (300, 300), "complex128"), vals(309, (300,), "complex128"))
    yield "lu_solve/err singular via lu", lu_then_solve(np.ones((3, 3)), np.ones(3))
    yield "lu_solve/err singular batch via lu", lu_then_solve(singular_batch(310, 1000, 3, bad=(7, 900)), np.ones(3))
    lmat = np.tril(vals(311, (4, 4), "float64"), -1) + np.diag([np.nan, 5.0, -2.0, 0.0])  # diagonal not read
    umat = np.triu(wc(312, (4, 4), "float64"))
    perm = np.array([2, 0, 3, 1], dtype=np.int64)
    yield C("lu_solve/explicit with garbage l diagonal", LA.lu_solve, lmat, umat, perm, vals(313, (4, 2), "float64"))
    for lo in LAYOUTS:
        yield C(f"lu_solve/layout {lo}", LA.lu_solve, lay(np.broadcast_to(lmat, (3, 4, 4)), lo),
                lay(np.broadcast_to(umat, (3, 4, 4)), lo), perm, lay(vals(314, (3, 4, 2), "float64"), lo))
    yield C("lu_solve/perm reversed", LA.lu_solve, lmat, umat, perm[::-1], vals(313, (4,), "float64"))
    yield C("lu_solve/perm batch", LA.lu_solve, lmat, umat, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64),
            vals(315, (2, 4, 1), "float64"))
    yield C("lu_solve/mixed complex b", LA.lu_solve, lmat, umat, perm, vals(316, (4,), "complex128"))
    yield C("lu_solve/mixed int l u", LA.lu_solve, np.tril(vals(317, (3, 3), "int64")), np.triu(wc(318, (3, 3), "int64")),
            np.array([1, 2, 0], dtype=np.int64), vals(319, (3,), "float64"))
    yield C("lu_solve/err nan u", LA.lu_solve, lmat, with_nan(umat, (0, 3)), perm, np.ones(4))
    yield C("lu_solve/nan unread u", LA.lu_solve, lmat, with_nan(umat, (3, 0)), perm, np.ones(4))
    uz = umat.copy()
    uz[2, 2] = 0
    yield C("lu_solve/err zero u diagonal", LA.lu_solve, lmat, uz, perm, np.ones(4))
    yield C("lu_solve/err perm not permutation", LA.lu_solve, lmat, umat, np.array([0, 0, 1, 2], dtype=np.int64), np.ones(4))
    yield C("lu_solve/err perm out of range", LA.lu_solve, lmat, umat, np.array([0, 1, 2, 4], dtype=np.int64), np.ones(4))
    yield C("lu_solve/err perm negative", LA.lu_solve, lmat, umat, np.array([0, 1, 2, -1], dtype=np.int64), np.ones(4))
    yield C("lu_solve/err perm float", LA.lu_solve, lmat, umat, perm.astype(np.float64), np.ones(4))
    yield C("lu_solve/err perm wrong length", LA.lu_solve, lmat, umat, perm[:3], np.ones(4))
    yield C("lu_solve/err b shape", LA.lu_solve, lmat, umat, perm, np.ones(3))
    yield C("lu_solve/err l u shapes", LA.lu_solve, lmat, umat[:3, :3], perm, np.ones(4))
    yield C("lu_solve/empty", LA.lu_solve, np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=np.int64), np.zeros(0))
    yield C("lu_solve/err numpy perm", LA.lu_solve, lmat, umat, Raw(perm), np.ones(4))

    # lstsq
    for dt in DTYPES:
        yield C(f"lstsq/{dt} over vec", LA.lstsq, vals(320, (8, 3), dt), vals(321, (8,), dt))
        yield C(f"lstsq/{dt} under mat", LA.lstsq, vals(322, (3, 8), dt), vals(323, (3, 2), dt))
        yield C(f"lstsq/{dt} square", LA.lstsq, wc(324, (4, 4), dt), vals(325, (4, 2), dt))
    rd = vals(326, (6, 4), "float64")
    rd[:, 3] = rd[:, 0] + rd[:, 1]
    for rt in (None, 0, 1e-12, 0.5, 1.0, 2.0):
        yield C(f"lstsq/rank-deficient rtol={rt}", LA.lstsq, rd, vals(327, (6, 2), "float64"), rtol=rt)
    yield C("lstsq/complex rank-deficient", LA.lstsq, vals(328, (6, 2), "complex128") @ vals(329, (2, 5), "complex128"),
            vals(330, (6,), "complex128"))
    yield C("lstsq/mixed int complex", LA.lstsq, vals(331, (5, 3), "int64"), vals(332, (5,), "complex128"))
    for lo in LAYOUTS:
        yield C(f"lstsq/layout {lo}", LA.lstsq, lay(vals(333, (3, 7, 4), "float64"), lo), lay(vals(334, (3, 7, 2), "float64"), lo))
    yield C("lstsq/batch 100x6x3", LA.lstsq, vals(335, (100, 6, 3), "float64"), vals(336, (100, 6, 2), "float64"))
    yield C("lstsq/batch 1000x3x3", LA.lstsq, vals(337, (1000, 3, 3), "float64"), vals(338, (3,), "float64"))
    yield C("lstsq/batch broadcast", LA.lstsq, vals(339, (4, 1, 5, 3), "complex128"), vals(340, (2, 5, 1), "complex128"))
    yield C("lstsq/large 300x200", LA.lstsq, vals(341, (300, 200), "float64"), vals(342, (300, 3), "float64"))
    yield C("lstsq/large 512 C", LA.lstsq, vals(343, (512, 512), "float64"), vals(344, (512,), "float64"))
    yield C("lstsq/empty 0x3", LA.lstsq, np.zeros((0, 3)), np.zeros(0))
    yield C("lstsq/empty 3x0", LA.lstsq, np.zeros((3, 0)), np.ones(3))
    yield C("lstsq/empty batch", LA.lstsq, np.zeros((0, 4, 3)), np.zeros((0, 4, 1)))
    yield C("lstsq/zeros", LA.lstsq, np.zeros((4, 3)), np.ones(4))
    yield C("lstsq/err nan a", LA.lstsq, with_nan(np.ones((4, 3)), (0, 0)), np.ones(4))
    yield C("lstsq/err nan b", LA.lstsq, np.ones((4, 3)), with_nan(np.ones(4), 3))
    yield C("lstsq/err rtol negative", LA.lstsq, np.ones((4, 3)), np.ones(4), rtol=-0.5)
    yield C("lstsq/err rtol inf", LA.lstsq, np.ones((4, 3)), np.ones(4), rtol=INF)
    yield C("lstsq/err shape mismatch", LA.lstsq, np.ones((4, 3)), np.ones(3))
    yield C("lstsq/err ndim 1", LA.lstsq, np.ones(4), np.ones(4))
    yield C("lstsq/err numpy arg", LA.lstsq, Raw(np.ones((4, 3))), np.ones(4))

    # inv
    for dt in DTYPES:
        for n in (1, 2, 3, 32):
            yield C(f"inv/{dt} n={n}", LA.inv, wc(350 + n, (n, n), dt))
    for lo in LAYOUTS:
        yield C(f"inv/layout {lo}", LA.inv, lay(wc(355, (4, 5, 5), "float64"), lo))
        yield C(f"inv/layout {lo} complex", LA.inv, lay(wc(355, (4, 5, 5), "complex128"), lo))
    for lo in ("C", "F", "reversed", "strided"):
        yield C(f"inv/large 512 {lo}", LA.inv, lay(vals(356, (512, 512), "float64"), lo))
    yield C("inv/large 300", LA.inv, vals(357, (300, 300), "float64"))
    yield C("inv/large 362 C", LA.inv, vals(358, (362, 362), "float64"))
    yield C("inv/large 363 C", LA.inv, vals(358, (363, 363), "float64"))
    yield C("inv/large 300 complex", LA.inv, vals(359, (300, 300), "complex128"))
    yield C("inv/large 512 bool", LA.inv, wc(360, (512, 512), "bool"))
    yield C("inv/batch 1000x3x3", LA.inv, vals(361, (1000, 3, 3), "float64"))
    yield C("inv/batch 100x32x32", LA.inv, vals(362, (100, 32, 32), "complex128"))
    yield C("inv/batch 4d", LA.inv, wc(363, (5, 6, 4, 4), "int64"))
    yield C("inv/err singular", LA.inv, np.ones((3, 3)))
    yield C("inv/err singular batch", LA.inv, singular_batch(364, 1000, 3, bad=(7, 900)))
    yield C("inv/err singular batch 4d", LA.inv, singular_batch(365, 12, 3, bad=(5,)).reshape(3, 4, 3, 3))
    yield C("inv/err nan", LA.inv, with_nan(np.eye(3), (0, 1)))
    yield C("inv/err nan batch", LA.inv, with_nan(wc(366, (100, 3, 3), "float64"), (42, 0, 1)))
    yield C("inv/empty 0x0", LA.inv, np.zeros((0, 0)))
    yield C("inv/empty batch", LA.inv, np.zeros((0, 3, 3), complex))
    yield C("inv/err nonsquare", LA.inv, np.ones((2, 3)))
    yield C("inv/err ndim 1", LA.inv, np.ones(3))
    yield C("inv/err numpy arg", LA.inv, Raw(np.eye(3)))

    # pinv
    for dt in DTYPES:
        yield C(f"pinv/{dt} (5,3)", LA.pinv, vals(370, (5, 3), dt))
        yield C(f"pinv/{dt} (3,5)", LA.pinv, vals(371, (3, 5), dt))
    lowr = vals(372, (6, 2), "float64") @ vals(373, (2, 5), "float64")
    for rt in (None, 0, 1e-10, 0.3, 1.0, 5):
        yield C(f"pinv/rank-deficient rtol={rt}", LA.pinv, lowr, rtol=rt)
    for lo in LAYOUTS:
        yield C(f"pinv/layout {lo}", LA.pinv, lay(vals(374, (3, 6, 4), "complex128"), lo))
    yield C("pinv/batch 100x6x4", LA.pinv, vals(375, (100, 6, 4), "float64"))
    yield C("pinv/batch 1000x3x3", LA.pinv, vals(376, (1000, 3, 3), "float64"))
    yield C("pinv/large 300x200", LA.pinv, vals(377, (300, 200), "float64"))
    yield C("pinv/large 512 C", LA.pinv, vals(378, (512, 512), "float64"))
    yield C("pinv/zeros", LA.pinv, np.zeros((3, 4)))
    yield C("pinv/empty 0x3", LA.pinv, np.zeros((0, 3)))
    yield C("pinv/empty batch", LA.pinv, np.zeros((0, 3, 2)))
    yield C("pinv/err nan", LA.pinv, with_nan(np.ones((3, 2)), (1, 1)))
    yield C("pinv/err rtol negative", LA.pinv, np.eye(3), rtol=-1)
    yield C("pinv/err rtol str", LA.pinv, np.eye(3), rtol="x")
    yield C("pinv/err ndim 1", LA.pinv, np.ones(3))
    yield C("pinv/err numpy arg", LA.pinv, Raw(np.eye(3)))

    # ========================================================= 4. factorizations
    # qr, qr_pivoted
    for fn in ("qr", "qr_pivoted"):
        f = getattr(LA, fn)
        for mode in ("reduced", "complete"):
            for dt in FC:
                for shp in ((6, 4), (4, 6), (5, 5), (1, 1)):
                    yield C(f"{fn}/{dt} {shp} {mode}", f, vals(400, shp, dt), mode=mode)
            for dt in ("bool", "int64"):
                yield C(f"{fn}/{dt} {mode}", f, vals(401, (5, 3), dt), mode=mode)
            for lo in LAYOUTS:
                yield C(f"{fn}/layout {lo} {mode}", f, lay(vals(402, (3, 6, 4), "float64"), lo), mode=mode)
            yield C(f"{fn}/batch 1000x3x3 {mode}", f, vals(403, (1000, 3, 3), "float64"), mode=mode)
            yield C(f"{fn}/batch 100x32x32 {mode}", f, vals(404, (100, 32, 32), "complex128"), mode=mode)
            yield C(f"{fn}/batch 100x8x3 {mode}", f, vals(405, (100, 8, 3), "float64"), mode=mode)
            yield C(f"{fn}/large 512 C {mode}", f, vals(406, (512, 512), "float64"), mode=mode)
            yield C(f"{fn}/large 512 F {mode}", f, lay(vals(406, (512, 512), "float64"), "F"), mode=mode)
            yield C(f"{fn}/large 400x300 {mode}", f, vals(407, (400, 300), "float64"), mode=mode)
            yield C(f"{fn}/rank-deficient {mode}", f, vals(408, (6, 2), "float64") @ vals(409, (2, 4), "float64"), mode=mode)
            yield C(f"{fn}/zeros {mode}", f, np.zeros((4, 3)), mode=mode)
            for shp in ((0, 0), (0, 3), (3, 0), (0, 4, 2)):
                yield C(f"{fn}/empty {shp} {mode}", f, np.zeros(shp), mode=mode)
        yield C(f"{fn}/default mode", f, vals(410, (5, 3), "float64"))
        yield C(f"{fn}/err mode r", f, np.eye(3), mode="r")
        yield C(f"{fn}/err mode economic", f, np.eye(3), mode="economic")
        yield C(f"{fn}/err mode not str", f, np.eye(3), mode=Raw(1))
        yield C(f"{fn}/err nan", f, with_nan(np.eye(3), (2, 2)))
        yield C(f"{fn}/err ndim 1", f, np.ones(3))
        yield C(f"{fn}/err numpy arg", f, Raw(np.eye(3)))

    # ldl
    for dt in DTYPES:
        for n in (1, 2, 5):
            yield C(f"ldl/{dt} n={n}", LA.ldl, herm(420 + n, (n, n), dt))
    for lo in LAYOUTS:
        yield C(f"ldl/layout {lo}", LA.ldl, lay(herm(426, (3, 6, 6), "complex128"), lo))
    g = herm(427, (5, 5), "float64")
    g[np.triu_indices(5, 1)] = np.nan
    yield C("ldl/nan in upper triangle", LA.ldl, g)
    yield C("ldl/2x2 pivot", LA.ldl, np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]))
    yield C("ldl/singular zeros", LA.ldl, np.zeros((4, 4)))
    yield C("ldl/singular ones", LA.ldl, np.ones((4, 4)))
    yield C("ldl/batch 1000x3x3", LA.ldl, herm(428, (1000, 3, 3), "float64"))
    yield C("ldl/batch 100x32x32", LA.ldl, herm(429, (100, 32, 32), "complex128"))
    yield C("ldl/large 300", LA.ldl, herm(430, (300, 300), "float64"))
    yield C("ldl/large 512 C", LA.ldl, herm(431, (512, 512), "float64"))
    yield C("ldl/large 512 F", LA.ldl, lay(herm(431, (512, 512), "float64"), "F"))
    yield C("ldl/empty 0x0", LA.ldl, np.zeros((0, 0)))
    yield C("ldl/empty batch", LA.ldl, np.zeros((0, 3, 3)))
    yield C("ldl/err nan", LA.ldl, with_nan(np.eye(3), (2, 0)))
    yield C("ldl/err nonsquare", LA.ldl, np.ones((2, 3)))
    yield C("ldl/err numpy arg", LA.ldl, Raw(np.eye(3)))

    # eigh, eigvalsh
    for fn in ("eigh", "eigvalsh"):
        f = getattr(LA, fn)
        for dt in DTYPES:
            for n in (1, 2, 3, 32):
                yield C(f"{fn}/{dt} n={n}", f, herm(440 + n, (n, n), dt))
        for lo in LAYOUTS:
            yield C(f"{fn}/layout {lo}", f, lay(herm(445, (3, 6, 6), "complex128"), lo))
        g = herm(446, (5, 5), "complex128")
        g[np.triu_indices(5, 1)] = np.nan
        yield C(f"{fn}/nan in upper triangle", f, g)
        yield C(f"{fn}/repeated eigenvalues", f, np.diag([2.0, 1.0, 2.0, 1.0]))
        yield C(f"{fn}/batch 1000x3x3", f, herm(447, (1000, 3, 3), "float64"))
        yield C(f"{fn}/batch 100x32x32", f, herm(448, (100, 32, 32), "complex128"))
        yield C(f"{fn}/large 300", f, herm(449, (300, 300), "float64"))
        for lo in ("C", "F", "reversed"):
            yield C(f"{fn}/large 512 {lo}", f, lay(herm(450, (512, 512), "float64"), lo))
        yield C(f"{fn}/large 300 complex", f, herm(451, (300, 300), "complex128"))
        # generalized a v = lambda b v
        for dt in DTYPES:
            yield C(f"{fn}/gen {dt}", f, herm(452, (4, 4), dt), spd(453, (4, 4), dt))
        yield C(f"{fn}/gen mixed float complex", f, herm(454, (4, 4), "float64"), spd(455, (4, 4), "complex128"))
        yield C(f"{fn}/gen mixed complex int", f, herm(454, (4, 4), "complex128"), spd(455, (4, 4), "int64"))
        for lo in LAYOUTS:
            yield C(f"{fn}/gen layout {lo}", f, lay(herm(456, (3, 5, 5), "float64"), lo), lay(spd(457, (3, 5, 5), "float64"), lo))
        gb = spd(458, (5, 5), "float64")
        gb[np.triu_indices(5, 1)] = np.nan
        yield C(f"{fn}/gen nan in b upper triangle", f, herm(459, (5, 5), "float64"), gb)
        yield C(f"{fn}/gen broadcast a batch", f, herm(460, (10, 4, 4), "float64"), spd(461, (4, 4), "float64"))
        yield C(f"{fn}/gen broadcast b batch", f, herm(460, (4, 4), "complex128"), spd(461, (2, 3, 4, 4), "complex128"))
        yield C(f"{fn}/gen batch 1000x3x3", f, herm(462, (1000, 3, 3), "float64"), spd(463, (1000, 3, 3), "float64"))
        yield C(f"{fn}/gen batch 100x32x32", f, herm(464, (100, 32, 32), "complex128"), spd(465, (100, 32, 32), "complex128"))
        yield C(f"{fn}/gen large 300", f, herm(466, (300, 300), "float64"), spd(467, (300, 300), "float64"))
        yield C(f"{fn}/gen large 512", f, herm(468, (512, 512), "float64"), spd(469, (512, 512), "float64"))
        yield C(f"{fn}/gen b=None", f, herm(470, (3, 3), "float64"), None)
        yield C(f"{fn}/err gen b not pd", f, herm(471, (3, 3), "float64"), -np.eye(3))
        nb = spd(472, (1000, 3, 3), "float64")
        nb[7] = -np.eye(3)
        nb[900] = -np.eye(3)
        yield C(f"{fn}/err gen b not pd batch", f, herm(473, (1000, 3, 3), "float64"), nb)
        yield C(f"{fn}/err gen nan b", f, np.eye(3), with_nan(np.eye(3), (1, 0)))
        yield C(f"{fn}/err gen shape mismatch", f, np.eye(3), np.eye(4))
        yield C(f"{fn}/err gen batch mismatch", f, np.ones((2, 3, 3)), np.ones((3, 3, 3)))
        yield C(f"{fn}/gen empty", f, np.zeros((0, 0)), np.zeros((0, 0)))
        yield C(f"{fn}/empty 0x0", f, np.zeros((0, 0)))
        yield C(f"{fn}/empty batch", f, np.zeros((0, 3, 3), complex))
        yield C(f"{fn}/err nan", f, with_nan(np.eye(3), (1, 0)))
        yield C(f"{fn}/err inf diagonal", f, with_nan(np.eye(3), (1, 1), np.inf))
        yield C(f"{fn}/err nonsquare", f, np.ones((2, 3)))
        yield C(f"{fn}/err ndim 1", f, np.ones(3))
        yield C(f"{fn}/err numpy arg", f, Raw(np.eye(3)))
        yield C(f"{fn}/err numpy b", f, np.eye(3), Raw(np.eye(3)))

    # eig, eigvals, schur, hessenberg
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    for fn in ("eig", "eigvals", "schur", "hessenberg"):
        f = getattr(LA, fn)
        for dt in DTYPES:
            for n in (1, 2, 3, 5, 32):
                yield C(f"{fn}/{dt} n={n}", f, vals(480 + n, (n, n), dt))
        for lo in LAYOUTS:
            yield C(f"{fn}/layout {lo}", f, lay(vals(486, (3, 6, 6), "float64"), lo))
            yield C(f"{fn}/layout {lo} complex", f, lay(vals(486, (3, 6, 6), "complex128"), lo))
        yield C(f"{fn}/rotation", f, rot)
        yield C(f"{fn}/defective", f, np.array([[1.0, 1.0], [0.0, 1.0]]))
        yield C(f"{fn}/repeated diagonal", f, np.diag([2.0, -1.0, 2.0, -1.0, 0.0]))
        yield C(f"{fn}/zeros", f, np.zeros((3, 3)))
        yield C(f"{fn}/triangular", f, np.triu(vals(487, (5, 5), "float64")))
        yield C(f"{fn}/batch 1000x3x3", f, vals(488, (1000, 3, 3), "float64"))
        yield C(f"{fn}/batch 100x32x32", f, vals(489, (100, 32, 32), "complex128"))
        yield C(f"{fn}/batch broadcast", f, lay(vals(490, (5, 4, 4), "float64"), "bcast"))
        yield C(f"{fn}/large 300", f, vals(491, (300, 300), "float64"))
        yield C(f"{fn}/large 512 C", f, vals(492, (512, 512), "float64"))
        yield C(f"{fn}/large 512 F", f, lay(vals(492, (512, 512), "float64"), "F"))
        yield C(f"{fn}/large 300 complex", f, vals(493, (300, 300), "complex128"))
        yield C(f"{fn}/empty 0x0", f, np.zeros((0, 0)))
        yield C(f"{fn}/empty batch", f, np.zeros((0, 3, 3)))
        yield C(f"{fn}/err nan", f, with_nan(np.eye(3), (0, 2)))
        nb = vals(494, (100, 3, 3), "float64")
        nb[40, 1, 2] = np.inf
        nb[70, 0, 0] = np.nan
        yield C(f"{fn}/err nan batch", f, nb)
        yield C(f"{fn}/err nonsquare", f, np.ones((2, 3)))
        yield C(f"{fn}/err ndim 1", f, np.ones(3))
        yield C(f"{fn}/err numpy arg", f, Raw(np.eye(3)))

    # svd, svdvals
    for fn, modes in (("svd", ("reduced", "complete")), ("svdvals", (None,))):
        f = getattr(LA, fn)
        for mode in modes:
            kw = {} if mode is None else {"mode": mode}
            sfx = "" if mode is None else f" {mode}"
            for dt in DTYPES:
                for shp in ((6, 4), (4, 6), (5, 5), (1, 1), (1, 4)):
                    yield C(f"{fn}/{dt} {shp}{sfx}", f, vals(500, shp, dt), **kw)
            for lo in LAYOUTS:
                yield C(f"{fn}/layout {lo}{sfx}", f, lay(vals(501, (3, 6, 4), "complex128"), lo), **kw)
            yield C(f"{fn}/batch 1000x3x3{sfx}", f, vals(502, (1000, 3, 3), "float64"), **kw)
            yield C(f"{fn}/batch 100x32x32{sfx}", f, vals(503, (100, 32, 32), "complex128"), **kw)
            yield C(f"{fn}/batch 100x6x4{sfx}", f, vals(504, (100, 6, 4), "float64"), **kw)
            yield C(f"{fn}/large 300{sfx}", f, vals(505, (300, 300), "float64"), **kw)
            yield C(f"{fn}/large 512 C{sfx}", f, vals(506, (512, 512), "float64"), **kw)
            yield C(f"{fn}/large 512 F{sfx}", f, lay(vals(506, (512, 512), "float64"), "F"), **kw)
            yield C(f"{fn}/large 400x200 complex{sfx}", f, vals(507, (400, 200), "complex128"), **kw)
            yield C(f"{fn}/rank-deficient{sfx}", f, vals(508, (6, 2), "float64") @ vals(509, (2, 5), "float64"), **kw)
            yield C(f"{fn}/zeros{sfx}", f, np.zeros((3, 4)), **kw)
            for shp in ((0, 0), (0, 3), (3, 0), (0, 4, 2)):
                yield C(f"{fn}/empty {shp}{sfx}", f, np.zeros(shp), **kw)
        yield C(f"{fn}/err nan", f, with_nan(np.ones((3, 2)), (2, 1)))
        yield C(f"{fn}/err ndim 1", f, np.ones(3))
        yield C(f"{fn}/err numpy arg", f, Raw(np.eye(3)))
    yield C("svd/default mode", LA.svd, vals(510, (5, 3), "float64"))
    yield C("svd/err mode r", LA.svd, np.eye(3), mode="r")

    # ===================================================== 5. special structure
    # solve_tridiagonal
    def tri(seed, batch, n, dt):
        dl = vals(seed, (*batch, max(n - 1, 0)), dt)
        du = vals(seed + 1, (*batch, max(n - 1, 0)), dt)
        if dt == "bool":
            return dl, np.ones((*batch, n), np.bool_), np.zeros_like(du)  # unit lower bidiagonal
        d = vals(seed + 2, (*batch, n), dt) + 7
        return dl, d, du

    for dt in DTYPES:
        dl, d, du = tri(520, (), 6, dt)
        yield C(f"solve_tridiagonal/{dt} vec", LA.solve_tridiagonal, dl, d, du, vals(523, (6,), dt))
        yield C(f"solve_tridiagonal/{dt} mat", LA.solve_tridiagonal, dl, d, du, vals(523, (6, 3), dt))
    dl, d, du = tri(524, (), 5, "float64")
    yield C("solve_tridiagonal/mixed complex b", LA.solve_tridiagonal, dl, d, du, vals(527, (5,), "complex128"))
    yield C("solve_tridiagonal/mixed complex dl", LA.solve_tridiagonal, dl.astype(complex) * 1j, d, du, vals(527, (5,), "float64"))
    for lo in ("strided", "reversed"):
        yield C(f"solve_tridiagonal/layout {lo}", LA.solve_tridiagonal, vlay(dl, lo), vlay(d, lo), vlay(du, lo),
                vlay(vals(528, (5,), "float64"), lo))
        yield C(f"solve_tridiagonal/b layout {lo}", LA.solve_tridiagonal, dl, d, du, lay(vals(528, (5, 3), "float64"), lo))
    yield C("solve_tridiagonal/b F", LA.solve_tridiagonal, dl, d, du, lay(vals(528, (5, 3), "float64"), "F"))
    yield C("solve_tridiagonal/n=1", LA.solve_tridiagonal, np.zeros(0), np.array([4.0]), np.zeros(0), np.array([2.0]))
    yield C("solve_tridiagonal/pivoting", LA.solve_tridiagonal, np.array([5.0, 5.0]), np.array([1e-3, 1e-3, 1e-3]),
            np.array([1.0, 1.0]), np.ones(3))
    bdl, bd, bdu = tri(529, (4,), 5, "float64")
    yield C("solve_tridiagonal/broadcast", LA.solve_tridiagonal, bdl, d, du, vals(532, (3, 1, 5, 2), "float64"))
    yield C("solve_tridiagonal/broadcast b vec", LA.solve_tridiagonal, bdl, bd, bdu, vals(532, (5,), "float64"))
    yield C("solve_tridiagonal/broadcast b batch", LA.solve_tridiagonal, bdl, bd, bdu, vals(533, (4, 5, 1), "float64"))
    yl, yd, yu = tri(534, (1000,), 3, "float64")
    yield C("solve_tridiagonal/batch 1000x3", LA.solve_tridiagonal, yl, yd, yu, vals(537, (1000, 3, 2), "float64"))
    yl, yd, yu = tri(538, (100,), 32, "complex128")
    yield C("solve_tridiagonal/batch 100x32", LA.solve_tridiagonal, yl, yd, yu, vals(541, (100, 32, 1), "complex128"))
    yl, yd, yu = tri(542, (), 100_000, "float64")
    yield C("solve_tridiagonal/long 100000", LA.solve_tridiagonal, yl, yd, yu, vals(545, (100_000,), "float64"))
    yield C("solve_tridiagonal/long 100000 mat", LA.solve_tridiagonal, yl, yd, yu, vals(545, (100_000, 3), "float64"))
    yield C("solve_tridiagonal/err singular", LA.solve_tridiagonal, np.zeros(2), np.zeros(3), np.zeros(2), np.ones(3))
    sl_, sd_, su_ = tri(546, (1000,), 3, "float64")
    sl_[[7, 900]] = 0
    sd_[[7, 900]] = 0
    su_[[7, 900]] = 0
    yield C("solve_tridiagonal/err singular batch", LA.solve_tridiagonal, sl_, sd_, su_, np.ones(3))
    yield C("solve_tridiagonal/err nan d", LA.solve_tridiagonal, dl, with_nan(d, 2), du, np.ones(5))
    yield C("solve_tridiagonal/err nan b", LA.solve_tridiagonal, dl, d, du, with_nan(np.ones(5), 4))
    yield C("solve_tridiagonal/empty n=0", LA.solve_tridiagonal, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    yield C("solve_tridiagonal/empty batch", LA.solve_tridiagonal, np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(3))
    yield C("solve_tridiagonal/err dl length", LA.solve_tridiagonal, np.ones(5), d, du, np.ones(5))
    yield C("solve_tridiagonal/err b length", LA.solve_tridiagonal, dl, d, du, np.ones(4))
    yield C("solve_tridiagonal/err numpy arg", LA.solve_tridiagonal, dl, Raw(d), du, np.ones(5))

    # solve_banded
    def band(a, kl, ku, corner=np.nan):
        """a (..., n, n) dense -> ab (..., kl + ku + 1, n); corners get `corner`."""
        n = a.shape[-1]
        ab = np.full((*a.shape[:-2], kl + ku + 1, n), corner, dtype=np.result_type(a, np.float64))
        if a.dtype in (np.bool_, np.int64):
            ab = np.zeros((*a.shape[:-2], kl + ku + 1, n), dtype=a.dtype)
        for j in range(n):
            for i in range(max(0, j - ku), min(n, j + kl + 1)):
                ab[..., ku + i - j, j] = a[..., i, j]
        return ab

    def banded_dense(seed, shape, kl, ku, dt):
        a = wc(seed, shape, dt)
        n = shape[-1]
        i, j = np.indices((n, n))
        return np.where((i - j <= kl) & (j - i <= ku), a, np.zeros((), a.dtype))

    for kl, ku in ((1, 1), (0, 0), (2, 1), (0, 3), (3, 0), (2, 2), (7, 7), (5, 0)):
        for dt in FC:
            ab = band(banded_dense(550, (6, 6), kl, ku, dt), kl, ku)
            yield C(f"solve_banded/{dt} kl={kl} ku={ku}", LA.solve_banded, ab, vals(551, (6, 2), dt),
                    lower_bandwidth=kl, upper_bandwidth=ku)
    for dt in ("bool", "int64"):
        ab = band(banded_dense(552, (5, 5), 1, 2, dt), 1, 2)
        yield C(f"solve_banded/{dt}", LA.solve_banded, ab, vals(553, (5,), dt), lower_bandwidth=1, upper_bandwidth=2)
    # same as solve_tridiagonal
    tdl, td, tdu = tri(554, (), 6, "float64")
    tab = np.stack([np.concatenate([[np.nan], tdu]), td, np.concatenate([tdl, [np.nan]])])
    yield C("solve_banded/tridiagonal 1 1", LA.solve_banded, tab, vals(557, (6,), "float64"), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/tridiagonal reference", LA.solve_tridiagonal, tdl, td, tdu, vals(557, (6,), "float64"))
    for lo in ("F", "strided", "reversed"):
        ab = band(banded_dense(558, (3, 7, 7), 2, 1, "float64"), 2, 1)
        yield C(f"solve_banded/layout {lo}", LA.solve_banded, lay(ab, lo), lay(vals(559, (3, 7, 2), "float64"), lo),
                lower_bandwidth=2, upper_bandwidth=1)
    yield C("solve_banded/batch 1000 n=3", LA.solve_banded, band(banded_dense(560, (1000, 3, 3), 1, 1, "float64"), 1, 1),
            vals(561, (1000, 3, 1), "float64"), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/batch 100 n=32", LA.solve_banded, band(banded_dense(562, (100, 32, 32), 3, 2, "complex128"), 3, 2),
            vals(563, (100, 32, 2), "complex128"), lower_bandwidth=3, upper_bandwidth=2)
    yield C("solve_banded/broadcast b", LA.solve_banded, band(banded_dense(564, (6, 6), 2, 2, "float64"), 2, 2),
            vals(565, (4, 6, 1), "float64"), lower_bandwidth=2, upper_bandwidth=2)
    lab = vals(566, (4, 50_000), "float64")
    lab[1] += 10
    yield C("solve_banded/long n=50000 kl=2 ku=1", LA.solve_banded, lab, vals(567, (50_000,), "float64"),
            lower_bandwidth=2, upper_bandwidth=1)
    tl = vals(568, (3, 50_000), "float64")
    tl[1] += 10
    yield C("solve_banded/long n=50000 tridiagonal", LA.solve_banded, tl, vals(569, (50_000, 2), "float64"),
            lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/err singular", LA.solve_banded, np.zeros((3, 4)), np.ones(4), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/err singular kl=2", LA.solve_banded, np.zeros((4, 4)), np.ones(4), lower_bandwidth=2, upper_bandwidth=1)
    sb = band(banded_dense(570, (1000, 3, 3), 1, 1, "float64"), 1, 1)
    sb[[7, 900]] = 0
    yield C("solve_banded/err singular batch", LA.solve_banded, sb, np.ones(3), lower_bandwidth=1, upper_bandwidth=1)
    nab = band(banded_dense(571, (5, 5), 1, 1, "float64"), 1, 1)
    nab[1, 2] = np.nan
    yield C("solve_banded/err nan in band", LA.solve_banded, nab, np.ones(5), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/err nan b", LA.solve_banded, band(banded_dense(571, (5, 5), 1, 1, "float64"), 1, 1),
            with_nan(np.ones(5), 0), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/empty n=0", LA.solve_banded, np.zeros((3, 0)), np.zeros(0), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/err rows mismatch", LA.solve_banded, np.ones((3, 4)), np.ones(4), lower_bandwidth=1, upper_bandwidth=2)
    yield C("solve_banded/err b length", LA.solve_banded, np.ones((3, 4)), np.ones(5), lower_bandwidth=1, upper_bandwidth=1)
    yield C("solve_banded/err negative bandwidth", LA.solve_banded, np.ones((1, 4)), np.ones(4), lower_bandwidth=-1, upper_bandwidth=1)
    yield C("solve_banded/err bandwidth missing", LA.solve_banded, np.ones((3, 4)), np.ones(4), lower_bandwidth=1)
    yield C("solve_banded/err bandwidth float", LA.solve_banded, np.ones((3, 4)), np.ones(4), lower_bandwidth=1.0, upper_bandwidth=1)
    yield C("solve_banded/err ndim 1", LA.solve_banded, np.ones(4), np.ones(4), lower_bandwidth=0, upper_bandwidth=0)
    yield C("solve_banded/err numpy arg", LA.solve_banded, Raw(np.ones((3, 4))), np.ones(4), lower_bandwidth=1, upper_bandwidth=1)

    # solve_sylvester
    for dt in DTYPES:
        yield C(f"solve_sylvester/{dt}", LA.solve_sylvester, wc(580, (4, 4), dt), wc(581, (3, 3), dt), vals(582, (4, 3), dt))
    yield C("solve_sylvester/mixed", LA.solve_sylvester, wc(583, (3, 3), "int64"), wc(584, (2, 2), "complex128"), vals(585, (3, 2), "float64"))
    for lo in LAYOUTS:
        yield C(f"solve_sylvester/layout {lo}", LA.solve_sylvester, lay(wc(586, (3, 4, 4), "float64"), lo),
                lay(wc(587, (3, 5, 5), "float64"), lo), lay(vals(588, (3, 4, 5), "float64"), lo))
    yield C("solve_sylvester/broadcast", LA.solve_sylvester, wc(589, (5, 4, 4), "float64"), wc(590, (3, 3), "float64"),
            vals(591, (4, 3), "float64"))
    yield C("solve_sylvester/batch 1000x3x3", LA.solve_sylvester, wc(592, (1000, 3, 3), "float64"), wc(593, (1000, 3, 3), "float64"),
            vals(594, (1000, 3, 3), "float64"))
    yield C("solve_sylvester/batch 100x32x32", LA.solve_sylvester, wc(595, (100, 32, 32), "complex128"),
            wc(596, (32, 32), "complex128"), vals(597, (100, 32, 32), "complex128"))
    yield C("solve_sylvester/n=1", LA.solve_sylvester, np.array([[2.0]]), np.array([[3.0]]), np.array([[10.0]]))
    yield C("solve_sylvester/large 300", LA.solve_sylvester, vals(598, (300, 300), "float64"), vals(599, (300, 300), "float64") + 50 * np.eye(300),
            vals(600, (300, 300), "float64"))
    yield C("solve_sylvester/large 512x40", LA.solve_sylvester, wc(601, (512, 512), "float64"), wc(602, (40, 40), "float64"),
            vals(603, (512, 40), "float64"))
    yield C("solve_sylvester/err common eigenvalue", LA.solve_sylvester, np.eye(2), -np.eye(2), np.ones((2, 2)))
    sa = wc(604, (1000, 3, 3), "float64")
    sb2 = wc(605, (1000, 3, 3), "float64")
    sa[7], sb2[7] = np.eye(3), -np.eye(3)
    sa[900], sb2[900] = np.eye(3), -np.eye(3)
    yield C("solve_sylvester/err common eigenvalue batch", LA.solve_sylvester, sa, sb2, np.ones((3, 3)))
    yield C("solve_sylvester/err nan", LA.solve_sylvester, with_nan(np.eye(2), (0, 1)), np.eye(2), np.ones((2, 2)))
    yield C("solve_sylvester/err nan c", LA.solve_sylvester, np.eye(2), np.eye(2), with_nan(np.ones((2, 2)), (1, 1)))
    yield C("solve_sylvester/empty", LA.solve_sylvester, np.zeros((0, 0)), np.eye(2), np.zeros((0, 2)))
    yield C("solve_sylvester/empty batch", LA.solve_sylvester, np.zeros((0, 2, 2)), np.eye(3), np.zeros((2, 3)))
    yield C("solve_sylvester/err c shape", LA.solve_sylvester, np.eye(2), np.eye(3), np.ones((3, 2)))
    yield C("solve_sylvester/err c vector", LA.solve_sylvester, np.eye(2), np.eye(1), np.ones(2))
    yield C("solve_sylvester/err nonsquare", LA.solve_sylvester, np.ones((2, 3)), np.eye(3), np.ones((2, 3)))
    yield C("solve_sylvester/err numpy arg", LA.solve_sylvester, np.eye(2), np.eye(2), Raw(np.ones((2, 2))))

    # ============================================================ 6. extension
    for dt in DTYPES:
        for n in (1, 2, 3, 32):
            yield C(f"expm/{dt} n={n}", LA.expm, vals(610 + n, (n, n), dt))
    for s in (1e-8, 0.01, 0.1, 1, 5, 30, 200):
        yield C(f"expm/scale {s}", LA.expm, s * vals(615, (6, 6), "float64"))
        yield C(f"expm/scale {s} complex", LA.expm, s * vals(616, (6, 6), "complex128"))
    yield C("expm/zeros", LA.expm, np.zeros((4, 4)))
    yield C("expm/diagonal", LA.expm, np.diag([-1.0, 0.0, 1.0, 700.0]))
    yield C("expm/diagonal complex", LA.expm, np.diag([1j * np.pi, 2.0, -3 + 1j]))
    yield C("expm/upper triangular", LA.expm, np.triu(vals(617, (5, 5), "float64")))
    yield C("expm/lower triangular", LA.expm, np.tril(vals(618, (5, 5), "complex128")))
    yield C("expm/nilpotent", LA.expm, np.triu(np.ones((4, 4)), 1))
    yield C("expm/overflow", LA.expm, np.diag([800.0, 1.0]))
    yield C("expm/overflow full", LA.expm, 500 * np.ones((3, 3)))
    for lo in LAYOUTS:
        yield C(f"expm/layout {lo}", LA.expm, lay(vals(619, (3, 5, 5), "float64"), lo))
    yield C("expm/batch 1000x3x3", LA.expm, vals(620, (1000, 3, 3), "float64"))
    yield C("expm/batch 100x32x32", LA.expm, vals(621, (100, 32, 32), "complex128") / 5)
    yield C("expm/batch mixed diagonal", LA.expm, np.concatenate([vals(622, (5, 3, 3), "float64"),
                                                                  np.broadcast_to(np.diag([1.0, 2.0, 3.0]), (5, 3, 3))]))
    yield C("expm/large 300", LA.expm, vals(623, (300, 300), "float64") / 17)
    yield C("expm/large 512 C", LA.expm, vals(624, (512, 512), "float64") / 23)
    yield C("expm/large 512 F", LA.expm, lay(vals(624, (512, 512), "float64") / 23, "F"))
    yield C("expm/empty 0x0", LA.expm, np.zeros((0, 0)))
    yield C("expm/empty batch", LA.expm, np.zeros((0, 3, 3)))
    yield C("expm/err nan", LA.expm, with_nan(np.ones((3, 3)), (1, 2)))
    yield C("expm/err nan diagonal", LA.expm, np.diag([1.0, np.nan]))
    yield C("expm/err inf batch", LA.expm, with_nan(vals(625, (100, 3, 3), "float64"), (55, 0, 0), np.inf))
    yield C("expm/err nonsquare", LA.expm, np.ones((2, 3)))
    yield C("expm/err ndim 1", LA.expm, np.ones(3))
    yield C("expm/err numpy arg", LA.expm, Raw(np.eye(3)))
