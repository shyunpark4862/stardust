"""permutation_test p-values: super-uniformity under H0 (Monte Carlo, (b+1)/(n+1)) and exact."""
import numpy as np, stardust as sd
rng = sd.random.default_rng(11); nrng = np.random.default_rng(3)
def stat(x, y, axis=-1):
    return sd.mean(x, axis=axis) - sd.mean(y, axis=axis)
ps = []
for t in range(400):
    x = sd.asarray(nrng.normal(size=15)); y = sd.asarray(nrng.normal(size=15))
    r = sd.htest.permutation_test((x, y), stat, n_resamples=99, alternative="greater", rng=rng)
    ps.append(float(r.pvalue))
ps = np.array(ps)
print("MC (n=99): min p", ps.min(), " P(p<=0.05)", np.mean(ps <= 0.05), " P(p<=0.01)", np.mean(ps <= 0.01), " distinct grid:", np.allclose(ps*100, np.round(ps*100)))
x = sd.asarray(np.array([5., 6, 7])); y = sd.asarray(np.array([1., 2, 3]))
r = sd.htest.permutation_test((x, y), stat, n_resamples=10**4, alternative="greater", rng=rng)
print("exact (20 arrangements):", float(r.pvalue), "= 1/20 ?", float(r.pvalue) == 1/20)
import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
srt = np.sort(ps)
with open(f"{OUT}/perm_p.dat", "w") as o:
    o.write("p ecdf\n")
    for i, v in enumerate(srt, 1): o.write(f"{v:.4f} {i/len(srt):.5f}\n")
