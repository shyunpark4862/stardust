"""The accuracy table of sd.special (docs/design/special.md §6): for every function with
a reference file, the largest and the 99.9th-percentile error in ULP, in the guaranteed
region and outside it, next to scipy's on the same inputs. Outside the guaranteed region
the functions in refdata.MIXED report error / max(1, |value|) in units of 2^-52 ("mixed").

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
            e, se = err, serr
            if label == "loose" and name in refdata.MIXED:
                # the mixed bound: error / max(1, |value|), in units of 2^-52 (45 = the 1e-14 bound)
                label = "mixed"
                e = refdata.mixed_error(got, hi, lo) / 2.0**-52
                se = refdata.mixed_error(ref(*xs), hi, lo) / 2.0**-52 if ref is not None else None
            s = stats(se[m]) if se is not None else ("-", "-")
            print(f"| `{name}` | {m.sum()} | {label} | {' | '.join(stats(e[m]))} | `{common.counterpart(name)}` | "
                  f"{' | '.join(s)} |")


if __name__ == "__main__":
    main(sys.argv[1:])
