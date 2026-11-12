"""The accuracy table of sd.special (docs/design/special.md §6): for every function with
a reference file, the largest and the 99.9th-percentile error in ULP, in the guaranteed
region and outside it, next to scipy's on the same inputs.

    .venv/bin/python bench/special/accuracy.py [NAME ...] > table.md
"""

import sys

import numpy as np

import common


def stats(err):
    if err.size == 0:
        return "-", "-"
    return f"{err.max():.3g}", f"{np.quantile(err, 0.999):.3g}"


def main(argv):
    sys.path.insert(0, str(common.ROOT))
    import stardust as sd
    from tests.special import refdata

    names = argv or refdata.names()
    print("| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |")
    print("|---|---:|---|---:|---:|---|---:|---:|")
    for name in names:
        xs, hi, lo, g = refdata.load(name)
        got = np.asarray(common.sd_function(name)(*(sd.asarray(x) for x in xs)))
        err = refdata.ulp_error(got, hi, lo)
        ref = common.scipy_function(name)
        serr = refdata.ulp_error(ref(*xs), hi, lo) if ref is not None else None
        for label, m in (("guaranteed", g), ("loose", ~g)):
            if not m.any():
                continue
            s = stats(serr[m]) if serr is not None else ("-", "-")
            print(f"| `{name}` | {m.sum()} | {label} | {' | '.join(stats(err[m]))} | `{common.counterpart(name)}` | "
                  f"{' | '.join(s)} |")


if __name__ == "__main__":
    main(sys.argv[1:])
