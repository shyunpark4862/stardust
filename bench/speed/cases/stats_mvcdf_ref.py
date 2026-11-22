"""Reference values for the mvn / mvt cdf speed cases at d >= 4 (bench/speed/cases/stats.py,
group mvn-mvt-cdf): every library's result there is held against them, and the error it
reached is recorded next to its time (registry.Case.abs_err).

No Arb reference exists for these integrals (random correlation matrices at d = 5 and 10),
so the reference is a quasi-Monte Carlo run of sd far below the cases' tolerances: atol =
1e-8 (three standard errors) with a cap of 4 x 10^8 evaluations, one point a call (sd runs
the levels of all the points of a call together; at this tolerance 40 points at d = 10 hold
more than 8 GB, and one hard point at a cap of 10^9 holds 8 GB, at 4 x 10^8 under 3 GB).
Each point's estimated error is stored with its value. Every point at d = 5 reaches 1e-8;
at d = 10, 19 of the 60 points stop at the cap, with estimates up to 3.6e-7 (2026-11-22),
so at d = 10 errors below about 4e-7 are not resolved (the old R comparator's 1.7e-5 at
atol = 1e-6 is). Checked against mvtnorm (GenzBretz, abseps = 1e-8, maxpts = 10^9, 432 s):
the hardest normal point, P = 0.0668, differs by 6.7e-8 with estimates 1.9e-7 (sd) and
2.1e-7 (R). At d <= 3 sd's
cdf is an exact formula (to about 1e-14 against Arb, bench/stats/accuracy_multivariate.md):
the cases use its values directly and need no file.

The file is keyed by the SHA-256 of each point set and matrix, so a changed setup cannot
meet an old reference: reference() raises, and tests/bench checks the keys.

    .venv/bin/python -m bench.speed.cases.stats_mvcdf_ref     # rewrite stats_mvcdf_ref.json
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

PATH = pathlib.Path(__file__).with_name("stats_mvcdf_ref.json")
ATOL = 1e-8
MAX_POINTS = 4 * 10**8
SETS = ((5, math.inf), (5, 4.0), (10, math.inf), (10, 4.0))


def key(d: int, df: float) -> str:
    return f"d={d} df={'inf' if math.isinf(df) else repr(df)}"


def sha256(a) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def reference(d: int, df: float, X, S):
    """The stored values for the points X (n, d) of matrix S, or ValueError when the file
    does not hold this point set."""
    import numpy as np

    sets = json.loads(PATH.read_text())["sets"] if PATH.exists() else {}
    got = sets.get(key(d, df))
    if got is None or got["x_sha256"] != sha256(X) or got["S_sha256"] != sha256(S):
        raise ValueError(
            f"{PATH.name} has no reference for this {key(d, df)} point set: python -m bench.speed.cases.stats_mvcdf_ref"
        )
    return np.array(got["values"])


def main() -> None:
    import numpy as np

    import stardust as sd

    from .. import harness, results
    from .stats import _mvcdf_setup

    git = harness.git_info(str(results.ROOT))
    out = {
        "_provenance": {
            "tool": "bench/speed/cases/stats_mvcdf_ref.py",
            "method": "sd quasi-Monte Carlo, one point a call",
            "atol": ATOL,
            "max_points": MAX_POINTS,
            "commit": git["commit"] + ("+" if git["dirty"] else ""),
            "made": datetime.date.today().isoformat(),
        },
        "sets": {},
    }
    for d, df in SETS:
        data = _mvcdf_setup(d, df, truth=False)()
        X, D = data["xn"], data["D"]
        vals, errs = [], []
        t0 = time.perf_counter()
        for i in range(len(X)):
            r = D.cdf(sd.asarray(X[i : i + 1]), atol=ATOL, return_error=True, max_points=MAX_POINTS)
            vals.append(float(np.asarray(r.value)[0]))
            errs.append(float(np.asarray(r.error)[0]))
            print(f"{key(d, df)} point {i}: {vals[-1]!r} +- {errs[-1]:.1e}", flush=True)
        out["sets"][key(d, df)] = {
            "x_sha256": sha256(X),
            "S_sha256": sha256(data["S"]),
            "values": vals,
            "errors": errs,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
