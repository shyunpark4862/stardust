import time, math, numpy as np, stardust as sd
A=sd.asarray; rng=np.random.default_rng(8)
def ok(f):
    try: f(); return True
    except Exception: return False
def bis(lo,hi,mk):
    while hi-lo>1:
        m=(lo+hi)//2
        if ok(mk(m)): lo=m
        else: hi=m
    return lo
mw=lambda n: (lambda x=rng.normal(size=n),y=rng.normal(size=n): (lambda: sd.htest.mannwhitneyu(A(x),A(y),method='exact')))()
bis_mw=bis(100,149,mw); print("mwu c=d max", bis_mw)
mw8=lambda N: (lambda x=rng.normal(size=8),y=rng.normal(size=N-8): (lambda: sd.htest.mannwhitneyu(A(x),A(y),method='exact')))()
print("mwu c=8 max N", bis(1000,400000,mw8))
an=lambda N: (lambda x=rng.normal(size=N//2),y=rng.normal(size=N-N//2): (lambda: sd.htest.ansari(A(x),A(y),method='exact')))()
print("ansari max N", bis(100,500,an))
print("2^124 boundary C(N,N/2):", max(N for N in range(2,400) if math.comb(N,N//2)<2**124))
kt=lambda n: (lambda x=rng.normal(size=n),y=rng.normal(size=n): (lambda: sd.htest.kendalltau(A(x),A(y),method='exact')))()
print("kendall max n (indep)", bis(1000,2000,kt))
for n in (1000,1500):
    x=rng.normal(size=n); y=rng.normal(size=n); t=time.perf_counter(); r=sd.htest.kendalltau(A(x),A(y),method='exact'); print("kendall",n,"%.2fs"%(time.perf_counter()-t))
for n in (bis_mw,):
    x=rng.normal(size=n); y=rng.normal(size=n); t=time.perf_counter(); r=sd.htest.mannwhitneyu(A(x),A(y),method='exact'); print("mwu",n,"%.2fs"%(time.perf_counter()-t))
d=rng.normal(size=1250); t=time.perf_counter(); sd.htest.wilcoxon(A(d),method='exact'); print("wil 1250 %.3fs"%(time.perf_counter()-t))
