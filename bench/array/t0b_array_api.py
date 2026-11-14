"""T0b: the Array API additions, bincount and histogram against numpy.

Time per call (best of 5), stardust on 1 and 10 threads, numpy. Each line
is printed as soon as it is measured.

Run:  .venv/bin/python bench/array/t0b_array_api.py
"""

import timeit
import warnings

import numpy as np

import stardust as sd


def best_ms(stmt, ns):
    number = 1
    while True:
        t = timeit.timeit(stmt, globals=ns, number=number)
        if t > 0.02 or number >= 10000:
            break
        number *= 4
    return min(timeit.repeat(stmt, globals=ns, number=number, repeat=5)) / number * 1e3


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    arrays = dict(
        v=rng.standard_normal(1_000_000), w=rng.standard_normal(1_000_000),
        z=rng.standard_normal(1_000_000) + 1j * rng.standard_normal(1_000_000),
        m=rng.standard_normal((1000, 1000)), s=np.sort(rng.standard_normal(100_000)),
        idx=rng.integers(0, 1_000_000, 1_000_000), am=np.argsort(rng.standard_normal((1000, 1000)), axis=1),
        nz=(rng.random(1_000_000) < 0.3).astype(np.float64), big=rng.standard_normal(10_000_000),
        ints=rng.integers(0, 1000, 1_000_000), a3=rng.standard_normal((100, 100, 100)),
        b2=rng.standard_normal((100, 100, 50)), x1=rng.standard_normal(1000), y1=rng.standard_normal(1000),
        small=rng.standard_normal(10),
    )
    cases = [
        ("copysign(v, w)", "M.copysign(v, w)"),
        ("hypot(v, w)", "M.hypot(v, w)"),
        ("logaddexp(v, w)", "M.logaddexp(v, w)"),
        ("nextafter(v, w)", "M.nextafter(v, w)"),
        ("reciprocal(v)", "M.reciprocal(v)"),
        ("reciprocal(z complex)", "M.reciprocal(z)"),
        ("signbit(v)", "M.signbit(v)"),
        ("roll(v, 12345)", "M.roll(v, 12345)"),
        ("roll(m, 7, axis=(0,1))", "M.roll(m, 7, axis=(0, 1))"),
        ("repeat(m, 2, axis=1)", "M.repeat(m, 2, axis=1)"),
        ("tile(m, (2, 2))", "M.tile(m, (2, 2))"),
        ("tril(m)", "M.tril(m)"),
        ("diff(v)", "M.diff(v)"),
        ("diff(m, axis=0)", "M.diff(m, axis=0)"),
        ("meshgrid(x1, y1)", "M.meshgrid(x1, y1)"),
        ("take(v, idx)", "M.take(v, idx)"),
        ("take_along_axis(m, am, axis=1)", "M.take_along_axis(m, am, axis=1)"),
        ("nonzero(nz 1e6)", "M.nonzero(nz)"),
        ("count_nonzero(big 1e7)", "M.count_nonzero(big)"),
        ("searchsorted(s 1e5, v 1e6)", "M.searchsorted(s, v)"),
        ("bincount(ints 1e6)", "M.bincount(ints)"),
        ("histogram(v, 100)", "M.histogram(v, bins=100)"),
        ("histogram(v, 100, weights)", "M.histogram(v, bins=100, weights=w)"),
        ("tensordot(a3, b2, axes=2)", "M.tensordot(a3, b2, axes=([1, 2], [0, 1]))"),
        ("flip(small) (call cost)", "M.flip(small)"),
        ("hypot(1.0, 2.0) (scalar call)", "M.hypot(1.0, 2.0)"),
    ]
    sd_ns = {k: sd.asarray(a) for k, a in arrays.items()} | {"M": sd}
    np_ns = dict(arrays) | {"M": np}
    print(f"{'':34} {'sd 1T':>10} {'sd 10T':>10} {'numpy':>10} {'np/sd1':>7} {'np/sd10':>8}", flush=True)
    before = sd.get_num_threads()
    try:
        for label, stmt in cases:
            sd.set_num_threads(1)
            t1 = best_ms(stmt, sd_ns)
            sd.set_num_threads(10)
            t10 = best_ms(stmt, sd_ns)
            tn = best_ms(stmt, np_ns)
            print(f"{label:34} {t1:8.4f}ms {t10:8.4f}ms {tn:8.4f}ms {tn / t1:6.2f}x {tn / t10:7.2f}x", flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
