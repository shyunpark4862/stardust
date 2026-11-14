"""T2: speed of the noncentral distribution functions against scipy (1e5 points, 1 and 10 threads).

scipy has no upper tails for these: chndtrc, nctdtrc and ncfdtrc are timed against
scipy.stats' sf (Boost underneath, as scipy.special's forward functions), and
chndtr / chndtrix against scipy's x-first argument order.

    .venv/bin/python bench/special/t2/speed.py > bench/special/t2/speed.md
"""

import timeit
import warnings

import numpy as np
import scipy.special as sc
import scipy.stats as st

import stardust as sd


def best(f, n=1):
    return min(timeit.repeat(f, number=n, repeat=3)) / n * 1e3


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    N = 100_000
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), N))
    p = rng.uniform(0, 1, N)
    df, nc = lu(0.5, 50), lu(0.1, 50)
    x = (df + nc) * lu(0.3, 3)
    tdf, tnc = lu(1, 50), rng.normal(0, 3, N)
    t = tnc + rng.normal(0, 2, N)
    d1, d2, fnc = lu(0.5, 50), lu(2.5, 50), lu(0.1, 50)
    fx = (d1 + fnc) / d1 * lu(0.3, 3)
    cases = [
        ("chndtr", (df, nc, x), lambda: sc.chndtr(x, df, nc)),
        ("chndtrc", (df, nc, x), lambda: st.ncx2.sf(x, df, nc)),
        ("chndtrix", (df, nc, p), lambda: sc.chndtrix(p, df, nc)),
        ("nctdtr", (tdf, tnc, t), lambda: sc.nctdtr(tdf, tnc, t)),
        ("nctdtrc", (tdf, tnc, t), lambda: st.nct.sf(t, tdf, tnc)),
        ("nctdtrit", (tdf, tnc, p), lambda: sc.nctdtrit(tdf, tnc, p)),
        ("ncfdtr", (d1, d2, fnc, fx), lambda: sc.ncfdtr(d1, d2, fnc, fx)),
        ("ncfdtrc", (d1, d2, fnc, fx), lambda: st.ncf.sf(fx, d1, d2, fnc)),
        ("ncfdtri", (d1, d2, fnc, p), lambda: sc.ncfdtri(d1, d2, fnc, p)),
    ]
    print("| function | sd 1T (ms) | sd 10T (ms) | scipy (ms) | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---:|---:|---:|---:|---:|")
    before = sd.get_num_threads()
    try:
        for name, args, ref in cases:
            sargs = [sd.asarray(a) for a in args]
            f = getattr(sd.special, name)
            sd.set_num_threads(1)
            t1 = best(lambda: f(*sargs))
            sd.set_num_threads(10)
            t10 = best(lambda: f(*sargs))
            ts = best(ref)
            print(f"| `{name}` | {t1:.1f} | {t10:.1f} | {ts:.1f} | {ts / t1:.2f} | {ts / t10:.1f} |", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
