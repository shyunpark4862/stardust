"""Worst error (relative to the largest moment) of the 13 even-degree cos moments (degrees 0..24) over p in (2, 60]:
binary64 forward recurrence against the boundary-value solve (p <= 24, QUADPACK's branch). Output: moments_scan.dat."""
import numpy as np
from decimal import Decimal as D
from moments import forward, bvp

rows = []
for p in np.round(np.arange(2.25, 60.01, 0.5), 2):
    p = float(p)
    ref = forward(p, 16, dec=True); fw = forward(p, 16)
    sc = max(abs(ref[j]) for j in range(1, 14))  # the moments' scale (a moment near a zero has no relative accuracy to lose)
    ef = max(float(abs(D(repr(fw[j])) - ref[j]) / sc) for j in range(1, 14))
    eb = max(float(abs(D(repr(bvp(p)[j])) - ref[j]) / sc) for j in range(1, 14)) if p <= 24 else float('nan')
    rows.append((p, ef, eb))
with open("moments_scan.dat", "w") as f:
    f.write("p fwd bvp\n")
    for r in rows: f.write("%.2f %.4e %s\n" % (r[0], max(r[1], 1e-17), ("%.4e" % max(r[2], 1e-17)) if r[2] == r[2] else "nan"))
a = np.array(rows)
print("bvp worst over p<=24: %.2e" % np.nanmax(a[:, 2]), "forward worst p<=24: %.2e" % a[a[:, 0] <= 24, 1].max(), "forward worst p>24: %.2e" % a[a[:, 0] > 24, 1].max())
