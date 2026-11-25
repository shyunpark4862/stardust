"""유한 정밀도의 한계: f(x) = (x-1)^2 + 1의 평평한 구간과 stardust의 최소화 정확도(복제본)."""
import math, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
f = lambda x: (x - 1.0) ** 2 + 1.0
# 1. f(1+h) == 1인 가장 긴 구간(binary64): 1+h를 1 근처의 이웃 double로 훑는다
x = 1.0
hi = 0
y = 1.0
while True:
    y2 = math.nextafter(y, 2.0)
    if f(y2) != 1.0: break
    y = y2; hi += 1
lo = 0; y = 1.0
while True:
    y2 = math.nextafter(y, 0.0)
    if f(y2) != 1.0: break
    y = y2; lo += 1
print("flat ulps up", hi, "down", lo, "h_up", (math.nextafter(1.0,2.0)-1.0)*hi, "h_down", (1-math.nextafter(1.0,0.0))*lo)
# 2. 그림 자료: f(1+h) - 1 (h를 2^-30 간격으로)
with open(f"{OUT}/flat.dat", "w") as o:
    o.write("h fm1\n")
    for k in range(-400, 401):
        h = k * 2.0 ** -32
        o.write(f"{h:.6e} {f(1.0 + h) - 1.0:.6e}\n")
# 3. stardust (복제본)
try:
    import stardust as sd
except ImportError:
    sys.exit(0)
for meth, kw in (("brent", {}), ("golden", {}), ("bounded", {"bounds": (0.0, 3.0)})):
    r = sd.optimize.minimize_scalar(f, method=meth, **kw)
    print("quadratic", meth, "x-1 =", float(r.x) - 1.0, "nfev", int(r.nfev))
e = lambda t: math.exp(t) - 2.0 * t
ln2_ = math.log(2.0)
for meth, kw in (("brent", {}), ("golden", {}), ("bounded", {"bounds": (0.0, 3.0)}),
                 ("bounded", {"bounds": (0.0, 3.0), "options": {"xatol": 1e-15}})):
    r = sd.optimize.minimize_scalar(e, method=meth, **kw)
    print("exp(x)-2x", meth, kw.get("options", ""), "rel err", (float(r.x) - ln2_) / ln2_, "nfev", int(r.nfev))
# exp(x)-2x의 평평한 구간(값이 최솟값의 반올림과 같은 x의 폭)
fm = e(ln2_); y = ln2_; k = 0
while e(math.nextafter(y, 9.0)) <= fm: y = math.nextafter(y, 9.0); k += 1
print("exp flat up from fl(ln2):", y - ln2_, "ulps", k)
# 기울기를 쓰는 방법: g(x) = exp(x) - 2x (최소점 ln 2)
g = lambda v: float(sd.exp(v[0]) - 2.0 * v[0]) if hasattr(v, "shape") else math.exp(v) - 2 * v
dg = lambda v: sd.asarray([float(sd.exp(v[0])) - 2.0])
ln2 = math.log(2.0)
r = sd.optimize.minimize(lambda v: math.exp(float(v[0])) - 2.0 * float(v[0]), sd.asarray([0.0]), method="bfgs",
                         jac=lambda v: sd.asarray([math.exp(float(v[0])) - 2.0]), options={"gtol": 1e-15})
print("bfgs exp(x)-2x: rel err", (float(r.x[0]) - ln2) / ln2, r.status)
r = sd.optimize.minimize_scalar(lambda t: math.exp(t) - 2.0 * t, method="brent")
print("brent exp(x)-2x: rel err", (float(r.x) - ln2) / ln2)
