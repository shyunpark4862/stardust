"""S4: thread splitting when the cost per element varies (ROADMAP.md S4).

The incomplete functions iterate a data-dependent number of times: a continued
fraction near the mean of large a, b takes ~sqrt(a) terms, a series far in a tail a
handful. The pool hands out tasks from one atomic counter (docs/design/array.md §10.1),
so a slow task delays only the thread that took it. This measures how much of the ideal
speed-up survives when the expensive elements sit together (sorted by cost) against the
same elements shuffled.

    .venv/bin/python bench/special/s4_threads.py > bench/special/s4/threads.md
"""

import common
import numpy as np


def inputs(n, rng):
    # half cheap (a tail: few terms), half expensive (near the mean of large a, b)
    a = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n))
    b = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n))
    mean = a / (a + b)
    sdv = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    near = np.clip(mean + rng.normal(0, 1, n) * sdv, 1e-300, 1 - 1e-16)
    far = np.clip(mean * 0.5, 1e-300, 1 - 1e-16)
    expensive = rng.random(n) < 0.5
    x = np.where(expensive, near, far)
    return a, b, x, expensive


def main():
    import stardust as sd

    n = 1_000_000
    rng = np.random.default_rng(0)
    a, b, x, expensive = inputs(n, rng)
    order = np.argsort(~expensive, kind="stable")  # expensive first, then cheap
    cases = {"shuffled": (a, b, x), "sorted": (a[order], b[order], x[order])}
    gammas = {
        "shuffled": (a, a * (1 + np.where(expensive, rng.normal(0, 3, n) / np.sqrt(a), 0.5))),
    }
    gammas["sorted"] = tuple(v[order] for v in gammas["shuffled"])
    print(f"machine: {common.machine()}\n")
    print("| function | inputs | 1 thread (ms) | 10 threads (ms) | speed-up |")
    print("|---|---|---:|---:|---:|")
    for name, data in (("betainc", cases), ("gammainc", gammas)):
        f = getattr(sd.special, name)
        for label, xs in data.items():
            arrs = [sd.asarray(np.ascontiguousarray(v)) for v in xs]
            times = []
            for threads in (1, 10):
                sd.set_num_threads(threads)
                times.append(common.best_ns(lambda: f(*arrs), repeat=5) / 1e6)
            print(f"| `{name}` | {label} | {times[0]:.1f} | {times[1]:.1f} | {times[0] / times[1]:.2f} |")
    sd.set_num_threads(10)


if __name__ == "__main__":
    main()
