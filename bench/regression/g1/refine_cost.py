"""G1: the time of the fit against the number of refinement steps (docs/design/regression.md
§3.1), next to the QR alone (sd.linalg.qr, which also forms Q: about twice geqrf).

Run: uv run --no-sync python bench/regression/g1/refine_cost.py
"""

import time

import numpy as np

import stardust as sd

SIZES = ((1000, 10), (100_000, 10), (10_000, 200))


def best(f, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return min(t)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for threads in (1, 10):
        sd.set_num_threads(threads)
        print(f"threads {threads}")
        for n, k in SIZES:
            X = sd.asarray(rng.standard_normal((n, k)))
            y = sd.asarray(rng.standard_normal(n))
            reps = 30 if n * k <= 1e4 else 7
            tq = best(lambda: sd.linalg.qr(X), reps)
            ts = [
                best(lambda s=s: sd._core.regression.fit("ols", y, X, None, None, True, -1.0, 0, s), reps)
                for s in range(4)
            ]
            per = [(ts[s] - ts[0]) / s for s in (1, 2, 3)]
            print(
                f"  ({n}, {k}): qr {tq * 1e3:8.3f} ms | fit steps 0..3 "
                + " ".join(f"{t * 1e3:8.3f}" for t in ts)
                + " ms | per step "
                + " ".join(f"{p * 1e3:7.3f}" for p in per)
                + " ms"
            )
