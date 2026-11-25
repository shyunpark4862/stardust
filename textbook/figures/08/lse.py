"""special.logsumexp against Decimal (60 digits): absolute and relative error; the naive m + log(sum)
without the TwoSum correction; the case of one dominant term (value near 0)."""
import numpy as np, stardust as sd, math
from decimal import Decimal, getcontext
getcontext().prec = 60
def exact(x):
    return float(sum((Decimal(float(v)).exp() for v in x), Decimal(0)).ln())
def naive(x):
    m = max(x); return m + math.log(sum(math.exp(v - m) for v in x))
rng = np.random.default_rng(2)
cases = {
  "one dominant [0,-40]": np.array([0.0, -40.0]),
  "[0,-1e-3 .. ] value~1": np.array([0.0, -36.0, -37.0]),
  "near zero sum: [-0.69314718, -0.69314718]": np.array([-0.6931471805599453, -0.6931471805599453]),
  "spread 700": np.concatenate([[0.0], -700 + 1e-3 * rng.random(2000)]),
}
for k in range(3):
    cases[f"random n=1000 scale 300 #{k}"] = rng.normal(size=1000) * 300
for name, x in cases.items():
    e = exact(x); s = float(sd.special.logsumexp(sd.asarray(x))); nv = naive(list(x))
    print(f"{name:45s} exact {e:.17g} sd abs {abs(s-e):.1e} rel {abs(s-e)/abs(e) if e else float('nan'):.1e} | naive abs {abs(nv-e):.1e}")
# value near zero: log(2) - log(2)
x = np.array([-math.log(2.0), -math.log(2.0)])
e = exact(x); s = float(sd.special.logsumexp(sd.asarray(x)))
print("lse(-ln2,-ln2): exact", e, "sd", s)
x = np.array([-1e-20, -50.0])
e = exact(x); s = float(sd.special.logsumexp(sd.asarray(x)))
print("lse(-1e-20,-50): exact", e, "sd", s, "rel", abs(s-e)/abs(e))
