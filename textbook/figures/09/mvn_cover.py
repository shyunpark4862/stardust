"""Coverage of the reported 3-sigma error of MultivariateNormal.cdf (Genz QMC, 12 shifts).
Exact value for equicorrelated rho: P = int Phi((b + sqrt(rho) z)/sqrt(1-rho))^d phi(z) dz
(trapezoid on [-12, 12], 24001 nodes: spectrally accurate)."""
import math, numpy as np, stardust as sd
from math import erfc, sqrt
z = np.linspace(-12, 12, 24001); h = z[1] - z[0]; phi = np.exp(-z*z/2)/sqrt(2*math.pi)
Phi = np.vectorize(lambda x: 0.5*erfc(-x/sqrt(2)))
def exact(b, rho, d):
    return float(np.sum(Phi((b + sqrt(rho)*z)/sqrt(1-rho))**d * phi) * h)
rng = np.random.default_rng(7)
rows = []
for trial in range(300):
    d = int(rng.integers(4, 11)); rho = float(rng.uniform(0.1, 0.9)); b = float(rng.uniform(-1.0, 1.5))
    atol = float(10.0 ** rng.uniform(-6.5, -4))
    S = rho*np.ones((d, d)) + (1-rho)*np.eye(d)
    M = sd.stats.MultivariateNormal(mu=sd.asarray(np.zeros(d)), cov=sd.asarray(S))
    v, e = M.cdf(sd.asarray(np.full(d, b)), atol=atol, return_error=True)
    ex = exact(b, rho, d)
    rows.append((abs(float(v) - ex), float(e), atol))
r = np.array(rows)
print("trials", len(r), " |err| <= reported:", np.mean(r[:,0] <= r[:,1]), " |err| <= atol:", np.mean(r[:,0] <= r[:,2]))
print("median |err|/reported", np.median(r[:,0]/r[:,1]), " max", np.max(r[:,0]/r[:,1]))
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
with open(f"{OUT}/mvn_cover.dat", "w") as o:
    o.write("reported actual atol\n")
    for a, e, t in rows: o.write(f"{e:.6e} {max(a,1e-18):.6e} {t:.6e}\n")
