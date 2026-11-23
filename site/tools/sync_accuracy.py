"""Copy a compact summary of the library's accuracy baselines into the site.

    ../stardust/.venv/bin/python tools/sync_accuracy.py [--lib PATH]

reads PATH/bench/accuracy/results/<area>.json (PATH defaults to ../stardust next to this
repository) for the areas that exist and writes src/data/accuracy.json, which
src/components/fig/AccuracySummary.astro renders. Only the measures are kept: the run
records (commit, date, machine), the comparator libraries and the worst inputs are dropped.
Numbers are rounded to three significant digits; the within-contract counts and the worst
contract ratios are computed from the unrounded values. The stats area's multivariate group
(sweeps named mv/...) is part of the stats area and also gets a summary of its own. Standard
library only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "src" / "data" / "accuracy.json"
# The areas the library measures, in its order (bench/areas.py without frame and random).
AREAS = ("array", "linalg", "special", "stats", "regression", "optimize", "htest", "sparse", "integrate", "differentiate", "spatial", "fft", "interpolate", "graph")
GROUPS = {"stats": {"mv": "mv/"}}  # area -> {group: sweep name prefix}

# The output: {"fields": FIELDS, "areas": {area: {"summary": {...}, "groups": {group: summary},
# "regions": [names], "sweeps": [[name, points, [row, ...]], ...]}}}, the areas in AREAS order,
# the sweeps sorted by name.
#
# The columns of a region row, in this order; trailing missing values are dropped. "region" is
# an index into the area's "regions" list (the region names: guaranteed, loose, mixed for special
# and stats, and for the multivariate group also tiny, "exact d<=3" and "atol=..."; for array the
# input kind and shape, for linalg the condition number kappa=..., for optimize the problem set
# or stopping test; for regression the result's field names). The measures follow the column
# order of the library's tables (bench/accuracy/tables.py).
FIELDS = (
    "region",
    "points",
    "ulp_max",
    "ulp_p999",
    "ulp_scale_max",
    "ulp_scale_p999",
    "mixed_max",
    "mixed_p999",
    "rel_max",
    "norm_rel",
    "norm_inf_rel_max",
    "numpy_dist_max",
    "nonfinite",
    "abs_max",
    "rel_cond_max",
    "backward_max",
    "orth_max",
    "grad_max",
    "excess_max",
    "tol_ratio_max",
    "fnorm_max",
    "kkt_max",
    "unconverged",
    "off_target",
    "est_ratio_max",
    "contract",
    "bound",
)


def num(v):
    """A JSON-safe number: rounded to 3 significant digits; inf and NaN as strings."""
    if v is None:
        return None
    if isinstance(v, bool) or isinstance(v, int):
        return v
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return 0
    r = float(f"{v:.3g}")
    return int(r) if r.is_integer() and abs(r) < 1e15 else r


def row(index, r):
    vals = [index] + [num(r.get(k)) for k in FIELDS[1:]]
    while vals and vals[-1] is None:
        vals.pop()
    return vals


def summarize(area, data):
    sweeps = sorted((s for s in data["sweeps"] if s["area"] == area), key=lambda s: s["name"])
    out = []
    names = []  # the region names, in the order first seen
    for s in sweeps:
        regions = []
        for label, r in s["regions"].items():
            if r.get("unmapped"):
                continue
            if label not in names:
                names.append(label)
            regions.append(row(names.index(label), r))
        out.append([s["name"], s["points"], regions])
    groups = {g: counts([s for s in sweeps if s["name"].startswith(p)]) for g, p in GROUPS.get(area, {}).items()}
    result = {"summary": counts(sweeps), "regions": names, "sweeps": out}
    if groups:
        result["groups"] = groups
    return result


def counts(sweeps):
    """Sweeps, points, (function, region) rows, the rows with a numerical contract and those
    within it, and the largest contract ratio."""
    judged = within = rows = 0
    worst = None
    for s in sweeps:
        for label, r in s["regions"].items():
            if r.get("unmapped"):
                continue
            rows += 1
            c = r.get("contract")
            if c is None:
                continue
            judged += 1
            if c <= 1.0:
                within += 1
            if worst is None or not c <= worst[0]:
                worst = (c, s["name"], label)
    summary = {
        "sweeps": len(sweeps),
        "points": sum(s["points"] for s in sweeps),
        "regions": rows,
        "judged": judged,
        "within": within,
    }
    if worst is not None:
        summary["worst"] = {"contract": num(worst[0]), "name": worst[1], "region": worst[2]}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", default=str(SITE.parent / "stardust"), help="the library repository")
    a = ap.parse_args(argv)
    results = pathlib.Path(a.lib) / "bench" / "accuracy" / "results"
    areas = {}
    for area in AREAS:
        p = results / f"{area}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        if data.get("schema") != 1:
            raise SystemExit(f"{p}: accuracy schema {data.get('schema')}, expected 1")
        areas[area] = summarize(area, data)
    if not areas:
        raise SystemExit(f"no accuracy results in {results}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"fields": FIELDS, "areas": areas}, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(text + "\n")
    for area, d in areas.items():
        for label, s in [(area, d["summary"])] + [(f"{area}/{g}", gs) for g, gs in d.get("groups", {}).items()]:
            print(
                f"{label}: {s['sweeps']} sweeps, {s['points']} points, {s['regions']} rows,"
                f" {s['within']}/{s['judged']} within the contract"
            )
    print(f"wrote {OUT.relative_to(SITE)} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
