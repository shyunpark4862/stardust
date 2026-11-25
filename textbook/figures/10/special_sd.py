"""따로 빌드한 stardust의 특수값·변환·반올림 동작과 서브노멀 연산 시간."""
import math, time, numpy as np, stardust as sd
A0 = sd.asarray
def A(x):
    return A0(x) if isinstance(x, (list, np.ndarray)) else A0([x])
def s(x): return repr(float(x)) if np.ndim(np.array(x.tolist() if hasattr(x,'tolist') else x)) == 0 else x
nz, pz, inf, nan = -0.0, 0.0, math.inf, math.nan
rows = [
 ("add(-0, +0)", sd.add(A(nz), A(pz))), ("add(-0, -0)", sd.add(A(nz), A(nz))), ("subtract(1, 1)", sd.subtract(A(1.0), A(1.0))),
 ("sqrt(-0)", sd.sqrt(A(nz))), ("divide(1, -0)", sd.divide(A(1.0), A(nz))),
 ("maximum(-0, +0)", sd.maximum(A(nz), A(pz))), ("minimum(+0, -0)", sd.minimum(A(pz), A(nz))),
 ("maximum(nan, 1)", sd.maximum(A(nan), A(1.0))), 
 ("max([1, nan, 3])", sd.max(A([1.0, nan, 3.0]))), ("nanmax([1, nan, 3])", sd.nanmax(A([1.0, nan, 3.0]))),
 ("sum([-0.0])", sd.sum(A([nz]))), ("nextafter(0, -1)", sd.nextafter(A(0.0), A(-1.0))), ("nextafter(max, inf)", sd.nextafter(A(1.7976931348623157e308), A(inf))),
 ("round(2.5)", sd.round(A(2.5))), ("round(-2.5)", sd.round(A(-2.5))), ("round(0.5)", sd.round(A(0.5))), ("round(1.5)", sd.round(A(1.5))),
]
for name, v in rows:
    vv = v.tolist() if hasattr(v, "tolist") else v
    vv = vv[0] if isinstance(vv, list) else vv
    print(f"{name:24s} {float(vv)!r}")
x = A([1e19, -1e19, nan, 2.9, -2.9, 9.3e18, inf])
print("astype int64:", x.astype(sd.int64).tolist())
big = A(np.array([2**53 + 1], dtype=np.int64))
print("int64 2^53+1 -> float64:", repr(float(big.astype(sd.float64)[0])))
ints = A(np.array([2**53 + 1] * 3 + [0], dtype=np.int64))
print("mean of [2^53+1]*3+[0] =", repr(float(sd.mean(ints))), " exact", (3 * (2**53 + 1)) / 4)
# 서브노멀 연산 시간
n = 10**7
xn = A0(np.full(n, 1.5)); xs = A0(np.full(n, 3e-310))
for name, arr in (("normal", xn), ("subnormal", xs)):
    best = 1e9
    for _ in range(5):
        t = time.perf_counter(); sd.multiply(arr, 0.75); best = min(best, time.perf_counter() - t)
    print(f"multiply {name}: {best*1e3:.2f} ms for 1e7")
print("subnormal kept:", sd.multiply(A(3e-310), A(0.5)).tolist(), " 2^-1074 * 0.5 ->", sd.multiply(A(5e-324), A(0.5)).tolist())
