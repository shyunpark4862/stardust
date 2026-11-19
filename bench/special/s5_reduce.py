"""S5: logsumexp, softmax, log_softmax against scipy (docs/design/special.md §3.5).

.venv/bin/python bench/special/s5_reduce.py > bench/special/s5/reduce.md
"""

import common
import numpy as np
import scipy.special as sp


def main():
    import stardust as sd

    rng = np.random.default_rng(0)
    shapes = [((10**6,), None), ((1000, 1000), 1), ((1000, 1000), 0), ((10**7,), None)]
    print(f"machine: {common.machine()}\n")
    print("| function | shape | axis | sd 1 thread (ms) | sd 10 threads (ms) | scipy (ms) | scipy / sd 10 |")
    print("|---|---|---|---:|---:|---:|---:|")
    for name in ["logsumexp", "softmax", "log_softmax"]:
        f, g = getattr(sd.special, name), getattr(sp, name)
        for shape, axis in shapes:
            x = rng.normal(0, 10, shape)
            a = sd.asarray(x)
            t = []
            for threads in (1, 10):
                sd.set_num_threads(threads)
                t.append(common.best_ns(lambda: f(a, axis=axis), repeat=5) / 1e6)
            ts = common.best_ns(lambda: g(x, axis=axis), repeat=5) / 1e6
            print(f"| `{name}` | {shape} | {axis} | {t[0]:.2f} | {t[1]:.2f} | {ts:.2f} | {ts / t[1]:.1f} |")
    sd.set_num_threads(10)


if __name__ == "__main__":
    main()
