"""T0: quantile / median, the nan* reductions, cov and corrcoef against numpy.

Time per call (best of 5), stardust on 1 and 10 threads, numpy (one thread
for these functions). Each line is printed as soon as it is measured.

Run:  .venv/bin/python bench/array/t0_statistics.py
"""

import timeit
import warnings

import numpy as np

import stardust as sd


def best_ms(stmt, ns):
    number = 1
    while True:  # aim at ~20 ms per repeat
        t = timeit.timeit(stmt, globals=ns, number=number)
        if t > 0.02 or number >= 1000:
            break
        number *= 4
    return min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number * 1e3


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    v = rng.standard_normal(1_000_000)
    m = rng.standard_normal((1000, 1000))
    vn = v.copy()
    vn[rng.random(v.size) < 0.1] = np.nan
    mn = m.copy()
    mn[rng.random(m.shape) < 0.1] = np.nan
    big = rng.standard_normal(10_000_000)
    bign = big.copy()
    bign[rng.random(big.size) < 0.1] = np.nan
    small = rng.standard_normal(100)
    obs = rng.standard_normal((20, 100_000))
    wide = rng.standard_normal((300, 3000))
    arrays = dict(v=v, m=m, vn=vn, mn=mn, bign=bign, small=small, obs=obs, wide=wide)
    sd_arrays = {k: sd.asarray(a) for k, a in arrays.items()}
    qs = [0.1, 0.5, 0.9]
    cases = [
        ("quantile(v 1e6, 0.5)", "M.quantile(v, 0.5)", None),
        ("quantile(v 1e6, [.1,.5,.9])", "M.quantile(v, Q)", None),
        ("quantile(small 100, 0.3)", "M.quantile(small, 0.3)", None),
        ("quantile(m, 0.5, axis=0)", "M.quantile(m, 0.5, axis=0)", None),
        ("quantile(m, [.1,.5,.9], axis=1)", "M.quantile(m, Q, axis=1)", None),
        ("quantile(v, 0.5, nearest)", "M.quantile(v, 0.5, method='nearest')", None),
        ("median(v 1e6)", "M.median(v)", None),
        ("median(m, axis=1)", "M.median(m, axis=1)", None),
        ("nanmedian(vn 1e6)", "M.nanmedian(vn)", None),
        ("nanmedian(mn, axis=1)", "M.nanmedian(mn, axis=1)", None),
        ("nanquantile(mn, [.1,.5,.9], axis=0)", "M.nanquantile(mn, Q, axis=0)", None),
        ("nansum(bign 1e7)", "M.nansum(bign)", None),
        ("nanprod(mn, axis=0)", "M.nanprod(mn, axis=0)", None),
        ("nanmean(bign 1e7)", "M.nanmean(bign)", None),
        ("nanmean(mn, axis=1)", "M.nanmean(mn, axis=1)", None),
        ("nanvar(bign 1e7)", "M.nanvar(bign)", None),
        ("nanstd(mn, axis=0)", "M.nanstd(mn, axis=0)", None),
        ("nanmin(bign 1e7)", "M.nanmin(bign)", None),
        ("nanmax(mn, axis=1)", "M.nanmax(mn, axis=1)", None),
        ("nanargmin(bign 1e7)", "M.nanargmin(bign)", None),
        ("nancumulative_sum(vn 1e6)", "CS(vn)", None),
        ("cov(obs 20x1e5)", "COV(obs)", None),
        ("cov(wide 300x3000)", "COV(wide)", None),
        ("corrcoef(wide 300x3000)", "M.corrcoef(wide)", None),
    ]
    sd_ns = sd_arrays | {"M": sd, "Q": sd.asarray(qs), "CS": sd.nancumulative_sum, "COV": sd.cov}
    np_ns = dict(arrays) | {"M": np, "Q": qs, "CS": np.nancumsum, "COV": lambda x: np.cov(x, ddof=0)}
    print(f"{'':38} {'sd 1T':>9} {'sd 10T':>9} {'numpy':>9} {'np/sd1':>7} {'np/sd10':>8}", flush=True)
    before = sd.get_num_threads()
    try:
        for label, stmt, _ in cases:
            sd.set_num_threads(1)
            t1 = best_ms(stmt, sd_ns)
            sd.set_num_threads(10)
            t10 = best_ms(stmt, sd_ns)
            tn = best_ms(stmt, np_ns)
            print(f"{label:38} {t1:7.3f}ms {t10:7.3f}ms {tn:7.3f}ms {tn / t1:6.2f}x {tn / t10:7.2f}x", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
