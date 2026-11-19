"""Per-element cost of each special function, for csrc/special/cost.hpp
(docs/design/special.md §3.3): serial ns per element on 10^6 contiguous elements, and
the weight it gives (cost / the cost of add). Compare the last two columns with the
table in cost.hpp; a factor of two is fine (the weight moves where the split starts,
never a result).

    .venv/bin/python bench/special/cost.py [NAME ...]
"""

import sys

import common
import numpy as np

N = 10**6

# Engine test functions without a reference file: normal inputs.
EXTRA = {"_fma3": 3, "_sum4": 4, "_powers4": 1}


def main(argv):
    import stardust as sd

    sd.set_num_threads(1)
    rng = np.random.default_rng(0)
    x, y = sd.asarray(rng.normal(size=N)), sd.asarray(rng.normal(size=N))
    add_ns = common.best_ns(lambda: sd.add(x, y), repeat=15) / N
    print(f"add: {add_ns:.3f} ns per element (cost.hpp kAddNs)")
    print()
    print("| function | ns per element | weight from it | weight in cost.hpp |")
    print("|---|---:|---:|---:|")
    names = argv or list(EXTRA) + common.ref_names()
    for name in names:
        if name in EXTRA:
            xs = tuple(rng.normal(size=N) for _ in range(EXTRA[name]))
        else:
            xs = common.sample_inputs(name, N)
        f = common.sd_function(name)
        sxs = tuple(sd.asarray(v) for v in xs)
        ns = common.best_ns(lambda: f(*sxs), repeat=7) / N
        print(
            f"| `{name}` | {ns:.2f} | {max(1, round(ns / add_ns))} | {sd.special._core._weight(common.PART.get(name, (name,))[0])} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
