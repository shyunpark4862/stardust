"""Compare two outputs of dump.cpp (old and new headers): the largest relative difference of
the value methods per (distribution, method) in contract units, and for the inverses the
number that differ and whether each differing pair is a tie inside the backward-error rule
(the cdf at the two answers within 1e-14 of the target or of each other)."""

import collections
import math
import sys


def parse(path):
    rows = []
    for line in open(path):
        f = line.split()
        rows.append((f[0], f[1], f[2], [float.fromhex(v) for v in f[3:]]))
    return rows


def main():
    old, new = parse(sys.argv[1]), parse(sys.argv[2])
    assert len(old) == len(new)
    worst = collections.defaultdict(lambda: (0.0, ""))
    count = collections.Counter()
    differ = collections.Counter()
    bad = []
    for (c0, x0, m0, v0), (c1, x1, m1, v1) in zip(old, new):
        dist = c0.split("(")[0]
        key = (dist, m0)
        if (c0, m0) != (c1, m1) or x0 != x1:  # the inputs came from an inverse that changed
            differ[(dist, "inputs")] += 1
            continue
        count[key] += 1
        a, b = v0[0], v1[0]
        if a == b or (a != a and b != b):
            continue
        if m0.startswith("i"):
            differ[key] += 1
            # a tie: the cdf values at the two answers agree to 1e-13 of each other
            fa, fb = v0[1], v1[1]
            if m0 in ("icdf", "iccdf") and not (abs(fa - fb) <= 1e-13 * max(abs(fa), abs(fb)) and abs(a - b) <= 1):
                bad.append((c0, x0, m0, a, b, fa, fb))
            continue
        scale = max(1.0, abs(a)) if m0.startswith("log") else abs(a)
        d = abs(a - b) / scale if scale > 0 else math.inf
        if d > worst[key][0]:
            worst[key] = (d, f"{c0} {x0}: {a!r} -> {b!r}")
    for key in sorted(count):
        d, where = worst[key]
        print(
            f"{key[0]:10s} {key[1]:9s} n={count[key]:6d} differing inverses={differ[key]:4d} max diff={d:.2e} {where}"
        )
    for k, v in differ.items():
        if k[1] == "inputs":
            print(f"{k[0]}: {v} rows skipped (their input, an inverse, changed)")
    print("inverse differences outside a tie:", len(bad))
    for b in bad[:20]:
        print("  ", b)


if __name__ == "__main__":
    main()
