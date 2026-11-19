"""Results files: one JSON per run, and one baseline JSON per area.

A run file (build/speed/<stamp>.json, ignored by git):

    {"schema": 1, "run": "<stamp>-<commit>", "env": {...}, "options": {...},
     "cases": [{"id", "area", "group", "name", "size", "n", "cost", "note",
                "timings": [{"label", "lib", "threads", "min", "median", "noise",
                             "batches", "calls", "error"}]}]}

`threads` is the thread count the call ran at: stardust's pool for "sd", the BLAS limit
for a "follow" comparator, and null for a "fixed" comparator (its library's defaults).
Times are seconds per call.

A baseline (bench/speed/results/<area>.json, in the repository) has the same cases
list, each case with the "run" it came from, and "runs": {run: env}. Promoting a run
replaces the cases it measured and keeps the others, so an area's baseline can be
refreshed one function at a time.
"""

from __future__ import annotations

import json
import pathlib

SCHEMA = 1
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "bench" / "speed" / "results"
RUN_DIR = ROOT / "build" / "speed"


def load(path) -> dict:
    data = json.loads(pathlib.Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: results schema {data.get('schema')}, expected {SCHEMA}")
    return data


def save(data: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def baseline_path(area: str) -> pathlib.Path:
    return BASELINE_DIR / f"{area}.json"


def load_baseline(area: str) -> dict | None:
    p = baseline_path(area)
    return load(p) if p.exists() else None


def baseline_cases(areas=None) -> dict[str, dict]:
    """The baseline cases of the given areas (all baselines by default), by id, each with
    its run's env under "env"."""
    out = {}
    paths = [baseline_path(a) for a in areas] if areas else sorted(BASELINE_DIR.glob("*.json"))
    for p in paths:
        if not p.exists():
            continue
        b = load(p)
        for c in b["cases"]:
            out[c["id"]] = dict(c, env=b["runs"].get(c.get("run"), {}))
    return out


def promote(run: dict, *, known_ids: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Merge a run's cases into the baselines of their areas. Cases whose every timing
    failed are skipped. known_ids (the registry's ids, in its order): drop baseline cases
    no longer in the registry, and keep the registry's order.
    Returns {area: (cases replaced or added, cases dropped)}."""
    by_area: dict[str, list[dict]] = {}
    for c in run["cases"]:
        if not any(t.get("min") is not None for t in c["timings"]):
            continue
        by_area.setdefault(c["area"], []).append(c)
    report = {}
    for area, new_cases in by_area.items():
        base = load_baseline(area) or {"schema": SCHEMA, "area": area, "runs": {}, "cases": []}
        new_ids = {c["id"] for c in new_cases}
        kept = [c for c in base["cases"] if c["id"] not in new_ids]
        order = {cid: i for i, cid in enumerate(known_ids or ())}
        dropped = 0
        if known_ids is not None:
            dropped = sum(c["id"] not in order for c in kept)
            kept = [c for c in kept if c["id"] in order]
        merged = kept + [dict(c, run=run["run"]) for c in new_cases]
        merged.sort(key=lambda c: (order.get(c["id"], len(order)), c["id"]))
        used = {c["run"] for c in merged}
        runs = {k: v for k, v in base["runs"].items() if k in used}
        runs[run["run"]] = run["env"]
        base.update(runs=runs, cases=merged)
        save(base, baseline_path(area))
        report[area] = (len(new_cases), dropped)
    return report


def timing(case: dict, label: str, threads) -> dict | None:
    for t in case["timings"]:
        if t["label"] == label and t["threads"] == threads:
            return t
    return None
