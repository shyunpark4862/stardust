"""M7: kernels measured where memory is not the bottleneck.

At 1M elements an elementwise op moves 24-48 MB and runs at memory speed,
which hides how fast the arithmetic itself is. Here the operands are 4096
elements (64 KB of complex128), small enough to stay in the core's caches,
and out= is reused so no allocation is timed.

Run:  .venv/bin/python bench/m7_incache.py
"""

import timeit

import numpy as np

import stardust_numpy as sdnp

N = 4096


def main():
    rng = np.random.default_rng(0)
    c, d = (rng.random(N) + 1j * rng.random(N) for _ in range(2))
    f, g = rng.random(N), rng.random(N)
    cases = [
        ("multiply complex128", "multiply", c, d),
        ("maximum float64", "maximum", f, g),
        ("add float64", "add", f, g),
    ]
    print(f"{'':22} {'stardust':>12} {'numpy':>12}   (per element)")
    for label, fn, a, b in cases:
        times = []
        for mod, conv in ((sdnp, sdnp.asarray), (np, np.asarray)):
            ns = {"F": getattr(mod, fn), "a": conv(a), "b": conv(b), "o": conv(np.zeros_like(a))}
            times.append(min(timeit.repeat("F(a, b, out=o)", globals=ns, number=20000, repeat=5)) / 20000 / N)
        print(f"{label:22} {times[0] * 1e9:9.3f} ns {times[1] * 1e9:9.3f} ns")


if __name__ == "__main__":
    main()
