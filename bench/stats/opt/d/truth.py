"""Independent values of the noncentral cdfs and densities by Arb, for looking at single points
where the fast and the precise paths differ: tools/stats_ref.py's textbook definitions (the
Poisson mixtures with the terms left out bounded, the Bessel I and 1F1 densities, the
noncentral t's integrals over S), as balls settled to 2^-110 relative (tools/arbref.py).

    .venv/bin/python bench/stats/opt/d/truth.py ncf_cdf 0.129 0.166 0.1495 0.93

The arguments are the parameters, then the point: ncx2_* df nc x, ncf_* dfn dfd nc x,
nct_* df nc t; the methods are cdf, ccdf and pdf.
"""

import math
import sys
from pathlib import Path

from flint import arb

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from arbref import rigorous  # noqa: E402
from stats_ref import DISTRIBUTIONS, method_value  # noqa: E402

DIST = {"ncx2": "noncentralchisquared", "ncf": "noncentralf", "nct": "noncentralt"}


def truth(name, args):
    """The settled ball (or an exact float: 0, inf, NaN) of name ("ncf_cdf", ...) at args."""
    dist, method = name.split("_", 1)
    D = DISTRIBUTIONS[DIST[dist]]
    return rigorous(lambda: method_value(D, method, [arb(float(a)) for a in args]))


if __name__ == "__main__":
    v = truth(sys.argv[1], sys.argv[2:])
    print(v if isinstance(v, float) and not math.isfinite(v) else arb(v).str(25, radius=False))
