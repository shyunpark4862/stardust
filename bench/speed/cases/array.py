"""The whole array area against numpy: every public function at 3, 10^3 .. 10^7
elements, in several layouts (contiguous, strided, transposed, broadcast) and dtypes.
From bench/array/full_scan.py (its case set, without its result check).

numpy runs as it is (one thread, except its BLAS). Each case is a statement written
once with `M` for the library (M = sd on stardust's side, M = np on numpy's), or two
statements where the calls differ; the statement is compiled into a function whose
globals hold that side's arrays, so the timed call is the statement itself.

The arrays (names as in the old script):
    f, g     10^6 float64 in [0.5, 1.5)          fs   10^6 standard normal
    u        10^6 in (-0.99, 0.99)               fh   10^6 in [1.5, 2.5)
    fn       fs with 10% NaN                     zh   half ±0.0, half normal
    i        10^6 int64 in [-1000, 1000)         j    10^6 int64 in [1, 1000)
    sh       shifts in [0, 63)                   b, b2  10^6 bool
    c, cc    10^6 complex128                     iv   10^6 int64 in [0, 1000)
    ivw      10^6 int64 in ±10^12                srt  sorted 10^6 float64
    f2, g2   (1000, 1000) float64                i2   (1000, 1000) int64
    ft       f2.T (transposed view)              fst  big2[::2] (stride 2, 10^6)
    v        1000 float64                        t3   (100, 100, 100)
    fm, gm   10^4 float64                        s3, t3s, i3, sm  tiny (3 or 3 x 4)
    x, fw, gz  written in place (copies of f, zeros)
    np_*     numpy arrays given to both sides (conversions)

Every array is drawn from its own seeded generator and only the arrays a case uses are
built, so the values differ from the old script's (one generator for everything); the
distributions, sizes, dtypes and layouts are the same."""

from __future__ import annotations

import ast
import math
import zlib

from ..registry import Case, Ref

N = 1_000_000
MID = 10_000
SIZES = {"1e3": 1_000, "1e5": 100_000, "1e7": 10_000_000}


# ----------------------------------------------------------------- data


def _builders(np):
    """key -> build(rng, get): the numpy data, each from its own generator."""
    nan = np.nan
    B = {
        "f": lambda r, get: r.random(N) + 0.5,  # positive: log, sqrt
        "g": lambda r, get: r.random(N) + 0.5,
        "fs": lambda r, get: r.standard_normal(N),  # both signs
        "u": lambda r, get: r.random(N) * 1.98 - 0.99,  # (-1, 1): asin, acos, atanh
        "fh": lambda r, get: r.random(N) + 1.5,  # > 1: acosh
        "fn": lambda r, get: np.where(r.random(N) < 0.1, nan, get("fs")),  # 10% NaN
        "i": lambda r, get: r.integers(-1000, 1000, N),
        "j": lambda r, get: r.integers(1, 1000, N),
        "sh": lambda r, get: r.integers(0, 63, N),
        "b": lambda r, get: r.random(N) > 0.5,
        "b2": lambda r, get: r.random(N) > 0.5,
        "c": lambda r, get: get("f") + 1j * r.random(N),
        "cc": lambda r, get: r.random(N) + 1j * r.random(N),
        "f2": lambda r, get: r.random((1000, 1000)),
        "g2": lambda r, get: r.random((1000, 1000)),
        "i2": lambda r, get: r.integers(-1000, 1000, (1000, 1000)),
        "v": lambda r, get: r.random(1000),
        "t3": lambda r, get: r.random((100, 100, 100)),
        "big2": lambda r, get: r.random(2 * N),  # strided views come from here
        "m512": lambda r, get: r.random((512, 512)),
        "m1k": lambda r, get: r.random((1000, 1000)),
        "im": lambda r, get: r.integers(-100, 100, (200, 200)),
        "bm": lambda r, get: r.random((1000, 8, 8)),
        "obs": lambda r, get: r.random((20, 100_000)),
        "iv": lambda r, get: r.integers(0, 1000, N),
        "ivw": lambda r, get: r.integers(-(10**12), 10**12, N),
        "idx": lambda r, get: r.integers(0, N, 100_000),
        "idxn": lambda r, get: r.integers(0, N, N),
        "rows": lambda r, get: r.integers(0, 1000, 1000),
        "idx2": lambda r, get: r.integers(0, 1000, (1000, 1000)),
        "srt": lambda r, get: np.sort(r.random(N)),
        "keys": lambda r, get: r.random(100_000),
        "q5": lambda r, get: np.array([0.1, 0.25, 0.5, 0.75, 0.9]),
        "zh": lambda r, get: (np.where(r.random(N) < 0.5, 0.0, r.standard_normal(N))
                              * np.where(r.random(N) < 0.5, 1, -1)),
        "fm": lambda r, get: r.random(MID) + 0.5,
        "gm": lambda r, get: r.random(MID) + 0.5,
        "s3": lambda r, get: np.array([1.0, 2.0, 3.0]),
        "t3s": lambda r, get: np.array([4.0, 5.0, 6.0]),
        "i3": lambda r, get: np.array([1, 2, 3]),
        "sm": lambda r, get: np.arange(12.0).reshape(3, 4),
        # only for the conversions
        "m2k": lambda r, get: r.random((2000, 2000)),
        "be2k": lambda r, get: r.random((2000, 1000)),
    }
    for nm, k in SIZES.items():
        B["a" + nm] = lambda r, get, k=k: r.random(k) + 0.5
        B["b" + nm] = lambda r, get, k=k: r.random(k) + 0.5
        B["i" + nm] = lambda r, get, k=k: r.integers(-1000, 1000, k)
        B["j" + nm] = lambda r, get, k=k: r.integers(-1000, 1000, k)
    return B


# numpy arrays (or a list) handed to both sides as they are: the conversions
_CONV = {
    "np_f": lambda np, get: get("f"),
    "np_fst": lambda np, get: get("big2")[::2],
    "np_ft": lambda np, get: get("f2").T,
    "np_f32": lambda np, get: get("f").astype(np.float32),
    "np_f32t": lambda np, get: get("m2k").astype(np.float32)[:1000, :1000].T,
    "np_f32r": lambda np, get: get("m2k").astype(np.float32)[:1000, 1:1001],
    "np_be": lambda np, get: get("f").astype(">f8"),
    "np_be2": lambda np, get: get("be2k").astype(">f8")[::2],
    "np_i32": lambda np, get: get("i").astype(np.int32),
    "np_list3": lambda np, get: [1.0, 2.0, 3.0],
    "np_list": lambda np, get: get("f")[:10_000].tolist(),
}

# per side, from that side's arrays (lib is np or sd; side(key) is that side's array)
_DERIVED = {
    "fst": lambda lib, get, side: side("big2")[::2],  # strided, 10^6
    "ft": lambda lib, get, side: side("f2").T,  # transposed
    "x": lambda lib, get, side: lib.asarray(get("f"), copy=True),  # written in place
    "fw": lambda lib, get, side: lib.asarray(get("f"), copy=True),
    "gz": lambda lib, get, side: lib.zeros(N),
}


def _names(*stmts) -> set[str]:
    out = set()
    for s in stmts:
        if s:
            out |= {n.id for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Name)}
    return out


def _make_setup(names, alias):
    """A setup building only the arrays `names` refer to (through `alias`: name -> key)."""

    def setup():
        import numpy as np

        import stardust as sd

        builders = _builders(np)
        raw: dict = {}

        def get(key):
            if key not in raw:
                rng = np.random.default_rng([0, zlib.crc32(key.encode())])
                raw[key] = builders[key](rng, get)
            return raw[key]

        npn = {"M": np, "np": np, "sd": sd, "N": N}
        sdn = {"M": sd, "np": np, "sd": sd, "N": N}
        per = {"np": {}, "sd": {}}

        def side(lib_name):
            def at(key):
                if key not in per[lib_name]:
                    v = get(key)
                    per[lib_name][key] = v.copy() if lib_name == "np" else sd.asarray(v, copy=True)
                return per[lib_name][key]
            return at

        for name in sorted(names):
            key = alias.get(name, name)
            if key in _CONV:
                npn[name] = sdn[name] = _CONV[key](np, get)
            elif key in _DERIVED:
                npn[name] = _DERIVED[key](np, get, side("np"))
                sdn[name] = _DERIVED[key](sd, get, side("sd"))
            elif key in builders:
                npn[name] = side("np")(key)
                sdn[name] = side("sd")(key)
            # other names (builtins, functions) resolve as usual
        return {"np": npn, "sd": sdn}

    return setup


def _compile(stmt: str, ns: dict):
    """A zero-argument function running stmt with ns as its globals."""
    try:
        ast.parse(stmt, mode="eval")
        src = f"def _f():\n    return {stmt}\n"
    except SyntaxError:
        src = f"def _f():\n    {stmt}\n"
    loc: dict = {}
    exec(compile(src, f"<array case: {stmt}>", "exec"), ns, loc)  # noqa: S102
    return loc["_f"]


def _n(size: str) -> int | None:
    if size.startswith("10^"):
        return 10 ** int(size[3:])
    if size.startswith("("):
        return math.prod(int(p) for p in size.strip("()").split(",") if p.strip())
    if size.isdigit():
        return int(size)
    return None


def _strip(stmt: str) -> str:
    return stmt.replace("M.", "").replace("sd.", "")


# ---------------------------------------------------------------- cases
# _add(group, sd statement, size, np=numpy statement (default: the same text, M = np),
#      name=..., note=..., quick=..., alias={name in the statement: data key},
#      extra=((label, numpy statement), ...) further numpy comparators)

G_CREATE = "생성"
G_CONVERT = "변환"
G_MANIP = "모양 바꾸기"
G_BIN_F = "이항 f64"
G_BIN_B = "이항 bool"
G_BIN_I = "이항 i64"
G_BIN_C = "이항 c128"
G_BIN_MIX = "이항 dtype 섞기·스칼라"
G_BIN_LAYOUT = "이항 메모리 배치"
G_BIN_MID = "이항 10^4"
G_UN_F = "단항 f64"
G_UN_C = "단항 c128"
G_UN_I = "단항 i64"
G_UN_B = "단항 bool"
G_SELECT = "골라 넣기"
G_RED_F = "리덕션 f64"
G_RED_I = "리덕션 i64"
G_RED_C = "리덕션 c128"
G_RED_B = "리덕션 bool"
G_RED_AX = "축 리덕션"
G_RED_MID = "리덕션 10^4"
G_SCAN = "누적"
G_NAN = "NaN 건너뛰기"
G_COUNT = "개수 세기"
G_STATS = "통계"
G_SORT = "정렬"
G_UNIQUE = "고유값"
G_SEARCH = "찾기·뽑기"
G_INDEX = "인덱싱"
G_ASSIGN = "대입"
G_MATMUL = "행렬곱"
G_SIZE = "크기별"
G_TINY = "작은 배열"

M6 = "10^6"
SQ = "(1000, 1000)"
T3 = "(100, 100, 100)"


def _specs() -> list[dict]:
    S: list[dict] = []

    def add(group, stmt, size, np=None, name=None, note="", quick=False, alias=None, extra=()):
        S.append({"group": group, "stmt": stmt, "size": size, "np": np, "name": name or _strip(stmt),
                  "note": note, "quick": quick, "alias": alias or {}, "extra": extra})

    # creation
    for s in ["zeros(N)", "ones(N)", "empty(N)", "full(N, 2.5)", "zeros_like(f)", "ones_like(f)", "empty_like(f)",
              "full_like(f, 2.5)", "arange(N)"]:
        add(G_CREATE, "M." + s, M6)
    add(G_CREATE, "M.linspace(0.0, 1.0, N)", M6, quick=True)
    add(G_CREATE, "M.eye(1000)", SQ)
    add(G_CREATE, "M.meshgrid(v, v)", SQ, note="1000개짜리 둘에서")
    add(G_CREATE, "M.tril(f2)", SQ)
    add(G_CREATE, "M.triu(f2)", SQ)
    add(G_CREATE, "M.zeros((1000, 1000))", SQ)
    add(G_CREATE, "M.arange(N, dtype=M.float64)", M6)
    add(G_CREATE, "M.asarray(f, copy=True)", M6)

    # conversion from numpy (sd.asarray(x) against np.asarray(x, dtype=float64): numpy keeps
    # the memory order, stardust makes C order)
    def conv(stmt, np, size, note, **kw):
        add(G_CONVERT, stmt, size, np=np, name=stmt, note=note, **kw)

    conv("sd.asarray(np_f)", "np.asarray(np_f)", M6, "f64, 복사 없음")
    conv("sd.asarray(np_fst)", "np.asarray(np_fst)", M6, "f64 stride 2")
    conv("sd.asarray(np_ft)", "np.asarray(np_ft)", SQ, "f64 전치")
    conv("sd.asarray(np_f32)", "np.asarray(np_f32, dtype=np.float64)", M6, "f32 → f64", quick=True)
    conv("sd.asarray(np_f32t)", "np.asarray(np_f32t, dtype=np.float64)", SQ,
         "f32 전치 → f64; numpy는 F 순서를 지키고, 'numpy C 순서'는 C 순서로 만든다",
         extra=(("numpy C 순서", "np.ascontiguousarray(np_f32t, dtype=np.float64)"),))
    conv("sd.asarray(np_f32r)", "np.asarray(np_f32r, dtype=np.float64)", SQ, "f32 행의 일부 m[:, 1:1001] → f64")
    conv("sd.asarray(np_be)", "np.asarray(np_be, dtype=np.float64)", M6, ">f8 → f64")
    conv("sd.asarray(np_be2)", "np.asarray(np_be2, dtype=np.float64)", SQ, ">f8 2-D 한 줄 걸러 → f64")
    conv("sd.asarray(np_i32)", "np.asarray(np_i32, dtype=np.int64)", M6, "i32 → i64")
    conv("sd.asarray(np_list3)", "np.asarray(np_list3)", "3", "파이썬 리스트")
    conv("sd.asarray(np_list)", "np.asarray(np_list)", "10^4", "파이썬 리스트")
    conv("sd.from_dlpack(np_f)", "np.from_dlpack(np_f)", M6, "f64")
    conv("np.asarray(f)", "np.asarray(f)", M6, "sd 배열 → numpy (비교: numpy 배열)")

    # manipulation
    for s, size, note in [
        ("reshape(f, (1000, 1000))", M6, "뷰"), ("reshape(ft, (N,))", SQ, "전치라 복사"),
        ("permute_dims(t3, (2, 0, 1))", T3, ""), ("matrix_transpose(f2)", SQ, ""), ("moveaxis(t3, 0, -1)", T3, ""),
        ("squeeze(M.reshape(f, (1, N)), axis=0)", M6, ""), ("expand_dims(f, axis=0)", M6, ""),
        ("broadcast_to(v, (1000, 1000))", SQ, ""), ("broadcast_arrays(f2, v)", SQ, ""),
        ("concat([f, g])", M6, ""), ("concat([f2, g2], axis=1)", SQ, ""), ("stack([f, g])", M6, ""),
        ("flip(f)", M6, ""), ("roll(f, 1)", M6, ""), ("roll(f2, 1, axis=1)", SQ, ""), ("repeat(f, 2)", M6, ""),
        ("tile(v, (1000,))", "1000", "결과 10^6"), ("unstack(f2)", SQ, ""), ("astype(f, M.int64)", M6, ""),
        ("astype(i, M.float64)", M6, ""), ("astype(f, M.complex128)", M6, ""), ("astype(b, M.float64)", M6, ""),
        ("astype(ft, M.float64)", SQ, "전치"),
    ]:
        add(G_MANIP, "M." + s, size, note=note, quick=(s == "concat([f2, g2], axis=1)"))
    add(G_MANIP, "M.reshape(ft, (-1,))", SQ, note="ravel, 전치라 복사")
    add(G_MANIP, "M.diff(f)", M6)

    # elementwise: binary
    for op in ["add", "subtract", "multiply", "divide", "floor_divide", "remainder", "pow", "atan2", "maximum", "minimum",
               "copysign", "hypot", "logaddexp", "nextafter", "equal", "less", "greater_equal"]:
        add(G_BIN_F, f"M.{op}(f, g)", M6, quick=(op == "add"))
    for op in ["logical_and", "logical_or", "logical_xor"]:
        add(G_BIN_B, f"M.{op}(b, b2)", M6)
    for op in ["add", "multiply", "floor_divide", "remainder", "maximum", "equal", "bitwise_and", "bitwise_or",
               "bitwise_xor"]:
        add(G_BIN_I, f"M.{op}(i, j)", M6)
    add(G_BIN_I, "M.bitwise_left_shift(i, sh)", M6)
    add(G_BIN_I, "M.bitwise_right_shift(i, sh)", M6)
    add(G_BIN_I, "M.pow(i, 2)", M6)
    for op in ["add", "multiply", "divide", "equal", "pow"]:
        add(G_BIN_C, f"M.{op}(c, cc)", M6)
    add(G_BIN_MIX, "f * i", M6, note="f64 × i64 (형 변환)")
    add(G_BIN_MIX, "f * i + g", M6)
    add(G_BIN_MIX, "f + 1.0", M6, note="스칼라")
    add(G_BIN_MIX, "i + 1", M6, note="스칼라")
    add(G_BIN_MIX, "c * f", M6, note="c128 × f64")
    add(G_BIN_LAYOUT, "f2 + v", SQ, note="행 브로드캐스트")
    add(G_BIN_LAYOUT, "f2 + M.reshape(v, (1000, 1))", SQ, note="열 브로드캐스트")
    add(G_BIN_LAYOUT, "ft + 1.0", SQ, note="전치")
    add(G_BIN_LAYOUT, "ft + f2", SQ, note="전치 + C 순서", quick=True)
    add(G_BIN_LAYOUT, "fst + 1.0", M6, note="stride 2")
    add(G_BIN_LAYOUT, "fst + f", M6, note="stride 2 + 연속")
    add(G_BIN_MID, "fm + gm", "10^4")
    add(G_BIN_MID, "fm * 2.0", "10^4")

    # elementwise: unary
    for op in ["negative", "positive", "abs", "square", "sqrt", "sign", "exp", "expm1", "log", "log1p", "log2", "log10",
               "sin", "cos", "tan", "atan", "sinh", "cosh", "tanh", "asinh", "floor", "ceil", "trunc", "round",
               "isnan", "isinf", "isfinite", "signbit", "reciprocal"]:
        add(G_UN_F, f"M.{op}(f)", M6, quick=(op == "exp"))
    for op in ["asin", "acos", "atanh"]:
        add(G_UN_F, f"M.{op}(u)", M6, note="(-1, 1)")
    add(G_UN_F, "M.acosh(fh)", M6, note="> 1")
    add(G_UN_F, "M.sign(fs)", M6, note="양수·음수 섞임")
    add(G_UN_F, "M.exp(fst)", M6, note="stride 2")
    add(G_UN_F, "M.sqrt(ft)", SQ, note="전치")
    add(G_UN_F, "M.exp(fm)", "10^4")
    for op in ["abs", "exp", "sqrt", "log", "sin", "conj", "real", "imag", "square", "isnan"]:
        add(G_UN_C, f"M.{op}(c)", M6)
    for op in ["abs", "negative", "square", "sign", "bitwise_invert"]:
        add(G_UN_I, f"M.{op}(i)", M6)
    add(G_UN_B, "M.logical_not(b)", M6)
    add(G_SELECT, "M.clip(f, min=0.6, max=1.2)", M6, np="np.clip(f, 0.6, 1.2)")
    add(G_SELECT, "M.clip(i, min=-10, max=10)", M6, np="np.clip(i, -10, 10)")
    add(G_SELECT, "M.where(b, f, g)", M6, quick=True)
    add(G_SELECT, "M.where(b, f, 0.0)", M6)

    # reductions
    for op in ["sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax"]:
        add(G_RED_F, f"M.{op}(f)", M6, quick=(op == "sum"))
    for op in ["sum", "min", "max", "mean", "var", "argmax"]:
        add(G_RED_I, f"M.{op}(i)", M6)
    for op in ["sum", "prod", "mean", "var"]:
        add(G_RED_C, f"M.{op}(c)", M6)
    add(G_RED_B, "M.any(b)", M6)
    add(G_RED_B, "M.all(b)", M6)
    add(G_RED_B, "M.sum(b)", M6)
    for op in ["sum", "mean", "max", "var", "argmax"]:
        add(G_RED_AX, f"M.{op}(f2, axis=0)", SQ, quick=(op == "sum"))
        add(G_RED_AX, f"M.{op}(f2, axis=1)", SQ)
    add(G_RED_AX, "M.sum(t3, axis=(0, 2))", T3)
    add(G_RED_AX, "M.sum(ft)", SQ, note="전치")
    add(G_RED_AX, "M.sum(ft, axis=0)", SQ, note="전치")
    add(G_RED_AX, "M.sum(fst)", M6, note="stride 2")
    add(G_RED_AX, "M.sum(i2, axis=0)", SQ)
    add(G_RED_MID, "M.sum(fm)", "10^4")
    add(G_RED_MID, "M.var(fm)", "10^4")
    add(G_SCAN, "M.cumulative_sum(f)", M6, quick=True)
    add(G_SCAN, "M.cumulative_prod(g)", M6)
    add(G_SCAN, "M.cumulative_sum(i)", M6)
    add(G_SCAN, "M.cumulative_sum(f2, axis=0)", SQ)
    add(G_SCAN, "M.cumulative_sum(f2, axis=1)", SQ)
    add(G_SCAN, "M.cumulative_sum(c)", M6)
    for op in ["nansum", "nanprod", "nanmin", "nanmax", "nanmean", "nanvar", "nanstd", "nanargmin", "nanargmax"]:
        add(G_NAN, f"M.{op}(fn)", M6, note="NaN 10%", quick=(op == "nanmean"))
    add(G_NAN, "sd.nancumulative_sum(fn)", M6, np="np.nancumsum(fn)", note="NaN 10%")
    add(G_NAN, "sd.nancumulative_prod(fn)", M6, np="np.nancumprod(fn)", note="NaN 10%")
    add(G_NAN, "M.nansum(M.reshape(fn, (1000, 1000)), axis=0)", SQ, note="NaN 10%")
    add(G_COUNT, "M.count_nonzero(b)", M6)
    add(G_COUNT, "M.count_nonzero(fs)", M6)
    add(G_COUNT, "M.count_nonzero(f2, axis=0)", SQ)

    # statistics
    add(G_STATS, "M.quantile(fs, q5)", M6, note="확률 5개")
    add(G_STATS, "M.quantile(fs, 0.5)", M6)
    add(G_STATS, "M.median(fs)", M6, quick=True)
    add(G_STATS, "M.median(f2, axis=0)", SQ)
    add(G_STATS, "M.median(f2, axis=1)", SQ)
    add(G_STATS, "M.nanquantile(fn, q5)", M6, note="NaN 10%, 확률 5개")
    add(G_STATS, "M.nanmedian(fn)", M6, note="NaN 10%")
    add(G_STATS, "sd.cov(m1k, correction=1)", SQ, np="np.cov(m1k)")
    add(G_STATS, "sd.cov(obs, correction=1)", "(20, 100000)", np="np.cov(obs)")
    add(G_STATS, "M.corrcoef(m1k)", SQ)
    add(G_STATS, "M.bincount(iv)", M6, note="값 0..999")
    add(G_STATS, "M.bincount(iv, weights=f)", M6, note="값 0..999")
    add(G_STATS, "M.histogram(fs, bins=100)", M6)
    add(G_STATS, "M.histogram(fs, bins=100, range=(-3.0, 3.0))", M6)

    # sorting and searching
    add(G_SORT, "M.sort(fs)", M6, quick=True)
    add(G_SORT, "M.sort(fs, stable=False)", M6)
    add(G_SORT, "M.sort(zh)", M6, note="절반이 ±0")
    add(G_SORT, "M.sort(i)", M6)
    add(G_SORT, "M.sort(ivw)", M6, note="값 ±10^12")
    add(G_SORT, "M.sort(f2, axis=1)", SQ)
    add(G_SORT, "M.sort(f2, axis=0)", SQ)
    add(G_SORT, "sd.sort(fs, descending=True)", M6, np="np.sort(fs)[::-1].copy()")
    add(G_SORT, "M.argsort(fs)", M6)
    add(G_SORT, "M.argsort(fs, stable=False)", M6)
    add(G_SORT, "M.argsort(i)", M6)
    add(G_SORT, "M.argsort(i, stable=False)", M6)
    add(G_SORT, "M.argsort(f2, axis=1)", SQ)
    add(G_SORT, "M.sort(fm)", "10^4")
    add(G_SORT, "M.argsort(fm)", "10^4")
    add(G_UNIQUE, "M.unique_values(iv)", M6, note="값 1000가지", quick=True)
    add(G_UNIQUE, "M.unique_counts(iv)", M6, note="값 1000가지")
    add(G_UNIQUE, "M.unique_inverse(iv)", M6, note="값 1000가지")
    add(G_UNIQUE, "M.unique_all(iv)", M6, note="값 1000가지")
    add(G_UNIQUE, "M.unique_values(fs)", M6, note="모두 다름")
    add(G_SEARCH, "M.searchsorted(srt, keys)", "10^5", note="정렬된 10^6에서")
    add(G_SEARCH, "M.searchsorted(srt, f)", M6, note="정렬된 10^6에서")
    add(G_SEARCH, "M.nonzero(b)", M6)
    add(G_SEARCH, "M.nonzero(f2 > 0.5)", SQ)
    add(G_SEARCH, "M.take(f, idx)", "10^5", note="10^6에서", quick=True)
    add(G_SEARCH, "M.take(f2, rows, axis=0)", SQ, note="행 1000개")
    add(G_SEARCH, "M.take_along_axis(f2, idx2, axis=1)", SQ)

    # indexing (operators)
    add(G_INDEX, "f[1000:2000]", M6, note="뷰")
    add(G_INDEX, "f[::2]", M6, note="뷰")
    add(G_INDEX, "f2[5]", SQ, note="행 뷰")
    add(G_INDEX, "f2[:, 5]", SQ, note="열")
    add(G_INDEX, "f2[10, 20]", SQ, note="스칼라")
    add(G_INDEX, "f[idx]", "10^5", note="10^6에서 모으기")
    add(G_INDEX, "f[idxn]", M6, note="모으기")
    add(G_INDEX, "f[b]", M6, note="마스크", quick=True)
    add(G_INDEX, "f2[rows]", SQ, note="행 1000개")
    add(G_INDEX, "f2[f2 > 0.5]", SQ, note="2-D 마스크")
    add(G_INDEX, "s3[1]", "3", note="스칼라")
    add(G_ASSIGN, "gz[idx] = 1.0", "10^5", note="10^6에 흩뿌리기")
    add(G_ASSIGN, "gz[idxn] = 1.0", M6, note="흩뿌리기")
    add(G_ASSIGN, "fw[b] = 0.0", M6, note="마스크", quick=True)
    add(G_ASSIGN, "fw[:] = 1.0", M6, note="채우기")
    add(G_ASSIGN, "fw[:] = g", M6, note="복사")
    add(G_ASSIGN, "fw[::2] = 1.0", M6, note="stride 2 채우기")
    # x += g would make x a local name of the compiled function: call the method
    add(G_ASSIGN, "x.__iadd__(g)", M6, name="x += g", note="제자리")
    add(G_ASSIGN, "x[1:] += x[:-1]", M6, note="겹침")
    add(G_ASSIGN, "x.__imul__(1.0000001)", M6, name="x *= 1.0000001", note="제자리")

    # linear algebra in the array area
    add(G_MATMUL, "M.matmul(m512, m512)", "(512, 512)")
    add(G_MATMUL, "M.matmul(m1k, m1k)", SQ, quick=True)
    add(G_MATMUL, "M.matmul(f2, v)", SQ, note="gemv")
    add(G_MATMUL, "M.matmul(v, v)", "1000", note="dot")
    add(G_MATMUL, "M.matmul(bm, bm)", "(1000, 8, 8)", note="8×8 1000개")
    add(G_MATMUL, "M.matmul(im, im)", "(200, 200)", note="정수")
    add(G_MATMUL, "M.matmul(ft, f2)", SQ, note="전치")
    add(G_MATMUL, "M.vecdot(f2, g2)", SQ)
    add(G_MATMUL, "M.tensordot(t3, t3, axes=2)", T3)

    # the same operations across sizes (cache-resident to memory-bound; 10^7 is 80 MB),
    # a, b (float64 in [0.5, 1.5)) and i, j (int64 in [-1000, 1000)) standing for that
    # size's arrays
    for s in ["a + b", "a * 2.0", "M.exp(a)", "M.sin(a)", "M.sum(a)", "M.max(a)", "M.var(a)", "M.cumulative_sum(a)",
              "a < b", "a[a > 1.0]", "M.where(a > 1.0, a, b)", "M.sort(a)", "M.argsort(a)", "M.astype(a, M.int64)",
              "M.asarray(a, copy=True)", "i + j", "M.sum(i)", "M.sort(i)", "M.median(a)", "M.unique_values(i)"]:
        for nm in SIZES:
            alias = {k: k + nm for k in ("a", "b", "i", "j")}
            add(G_SIZE, s, "10^" + nm[2:], alias=alias, quick=(s == "a + b" and nm == "1e7"))

    # per-call overhead (tiny arrays)
    for s, size in [
        ("M.zeros(3)", "3"), ("M.empty(3)", "3"), ("M.asarray(s3)", "3"), ("s3 + t3s", "3"), ("s3 * 2.0", "3"),
        ("i3 + 1", "3"), ("M.sqrt(s3)", "3"), ("M.exp(s3)", "3"), ("s3 < t3s", "3"), ("M.sum(s3)", "3"),
        ("M.max(s3)", "3"), ("M.mean(s3)", "3"), ("M.sum(sm, axis=0)", "(3, 4)"), ("M.sort(s3)", "3"),
        ("M.reshape(sm, (4, 3))", "(3, 4)"), ("sm.T", "(3, 4)"), ("sm[1]", "(3, 4)"), ("sm[1:, ::2]", "(3, 4)"),
        ("M.concat([s3, t3s])", "3"), ("M.where(s3 > 1.5, s3, t3s)", "3"), ("M.matmul(sm.T, sm)", "(3, 4)"),
        ("M.result_type(f, i)", ""), ("M.can_cast(M.int64, M.float64)", ""), ("M.finfo(M.float64)", ""),
        ("s3.shape", "3"), ("len(s3)", "3"), ("float(s3[0])", "3"),
    ]:
        add(G_TINY, s, size, quick=(s == "s3 + t3s"), note={
            "M.asarray(s3)": "이미 배열(복사 없음)", "M.result_type(f, i)": "dtype만 봄"}.get(s, ""))
    return S


def _sd_factory(stmt):
    return lambda d: _compile(stmt, d["sd"])


def _np_factory(stmt):
    return lambda d: _compile(stmt, d["np"])


def cases() -> list[Case]:
    out = []
    for s in _specs():
        np_stmt = s["np"] if s["np"] is not None else s["stmt"]
        extra_stmts = [e[1] for e in s["extra"]]
        names = _names(s["stmt"], np_stmt, *extra_stmts)
        refs = [Ref("numpy", _np_factory(np_stmt))]
        refs += [Ref(label, _np_factory(st), lib="numpy") for label, st in s["extra"]]
        out.append(Case(
            area="array", group=s["group"], name=s["name"], size=s["size"], n=_n(s["size"]),
            setup=_make_setup(frozenset(names), dict(s["alias"])),
            sd=_sd_factory(s["stmt"]),
            refs=tuple(refs),
            quick=s["quick"], note=s["note"],
        ))
    return out
