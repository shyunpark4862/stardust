"""Copy a compact summary of the library's accuracy baselines into the site.

    ../stardust/.venv/bin/python tools/sync_accuracy.py [--lib PATH]

reads PATH/bench/accuracy/results/<area>.json (PATH defaults to ../stardust next to this
repository) for the areas that exist and writes src/data/accuracy.json, which
src/components/fig/AccuracySummary.astro renders. Only the measures are kept: the run
records (commit, date, machine) and the worst inputs are dropped. Numbers are rounded to
three significant digits; the within-contract counts and the worst contract ratios are
computed from the unrounded values. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "src" / "data" / "accuracy.json"
AREAS = ("special", "stats", "regression")

# The output: {"fields": FIELDS, "areas": {area: {"summary": {...}, "regions": [names],
# "sweeps": [[name, points, [row, ...]], ...]}}}, the sweeps sorted by name.
#
# The columns of a region row, in this order; trailing missing values are dropped, so the
# measures most rows have come first. "region" is an index into the area's "regions" list
# (the region names: guaranteed, loose, mixed; for regression the result's field names).
FIELDS = (
    "region",
    "points",
    "ulp_max",
    "ulp_p999",
    "rel_max",
    "norm_rel",
    "contract",
    "bound",
    "ulp_cond_max",
    "ulp_cond_p999",
    "mixed_max",
    "mixed_p999",
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
    judged = within = 0
    worst = None
    by_label = {}
    names = []  # the region names, in the order first seen
    for s in sweeps:
        regions = []
        for label, r in s["regions"].items():
            if r.get("unmapped"):
                continue
            if label not in by_label:
                names.append(label)
            regions.append(row(names.index(label), r))
            b = by_label.setdefault(label, {"regions": 0, "points": 0})
            b["regions"] += 1
            b["points"] += r["points"]
            c = r.get("contract")
            if c is None:
                continue
            judged += 1
            if c <= 1.0:
                within += 1
            if worst is None or not c <= worst[0]:
                worst = (c, s["name"], label)
        out.append([s["name"], s["points"], regions])
    summary = {
        "sweeps": len(sweeps),
        "points": sum(s["points"] for s in sweeps),
        "regions": sum(len(s[2]) for s in out),
        "judged": judged,
        "within": within,
        "by_region": by_label,
    }
    if worst is not None:
        summary["worst"] = {"contract": num(worst[0]), "name": worst[1], "region": worst[2]}
    return {"summary": summary, "regions": names, "sweeps": out}


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
        s = d["summary"]
        print(f"{area}: {s['sweeps']} sweeps, {s['points']} points, {s['within']}/{s['judged']} within the contract")
    print(f"wrote {OUT.relative_to(SITE)} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
