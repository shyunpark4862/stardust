"""Run files and baselines of the accuracy measurement (the same layout as bench/speed).

A run file (build/accuracy/<stamp>.json, ignored by git):

    {"schema": 1, "run": "<stamp>-<commit>", "env": {...},
     "sweeps": [{"id", "area", "name", "points", "regions": {region: {"points", "ulp_max",
                 "ulp_p999", "rel_max", "contract", "bound", "worst", ["ulp_cond_max",
                 "ulp_cond_p999"], ["mixed_max", "mixed_p999"], ["norm_rel", "nonfinite"]}}}]}

region is "guaranteed", "loose" or "mixed" (for regression, a result field); "contract" is the
largest error over the bound (at most 1 passes; null where no bound is documented); "worst" is
the input where it is; "ulp_cond_*" is ULP / the bound's scale (sources.py); "nonfinite" (regression)
counts the points where the reference is finite and stardust's value is not.

A baseline (bench/accuracy/results/<area>.json, in the repository) has the same sweeps, each
with the "run" it came from, and "runs": {run: env}. Promoting a run replaces the sweeps it
measured and keeps the others.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform
import subprocess

SCHEMA = 1
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "bench" / "accuracy" / "results"
RUN_DIR = ROOT / "build" / "accuracy"


def env() -> dict:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()

    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", "src", "csrc")),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def new_run(sweeps: list[dict], compare: bool = False) -> dict:
    e = env()
    if compare:
        from . import compare as C

        e["comparators"] = C.versions()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return {"schema": SCHEMA, "run": f"{stamp}-{e['commit']}", "env": e, "sweeps": sweeps}


def load(path) -> dict:
    data = json.loads(pathlib.Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: accuracy schema {data.get('schema')}, expected {SCHEMA}")
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


def baseline_sweeps(areas) -> dict[str, dict]:
    out = {}
    for a in areas:
        b = load_baseline(a)
        if b:
            out.update({s["id"]: s for s in b["sweeps"]})
    return out


def promote(run: dict) -> list[str]:
    """Merge a run's sweeps into the baselines of their areas; the areas written."""
    by_area: dict[str, list] = {}
    for s in run["sweeps"]:
        by_area.setdefault(s["area"], []).append(dict(s, run=run["run"]))
    for area, new in by_area.items():
        base = load_baseline(area) or {"schema": SCHEMA, "area": area, "runs": {}, "sweeps": []}
        ids = {s["id"] for s in new}
        old = {s["id"]: s for s in base["sweeps"]}
        for s in new:  # a run without --compare keeps the comparators measured before
            if "comparators" not in s and "comparators" in old.get(s["id"], {}):
                s["comparators"] = old[s["id"]]["comparators"]
                s["comparators_run"] = old[s["id"]].get("comparators_run", old[s["id"]].get("run"))
            elif "comparators" in s:
                s["comparators_run"] = run["run"]
        sweeps = [s for s in base["sweeps"] if s["id"] not in ids] + new
        sweeps.sort(key=lambda s: s["id"])
        used = {s["run"] for s in sweeps} | {s["comparators_run"] for s in sweeps if "comparators_run" in s}
        runs = {k: v for k, v in base["runs"].items() if k in used}
        runs[run["run"]] = run["env"]
        save({"schema": SCHEMA, "area": area, "runs": runs, "sweeps": sweeps}, baseline_path(area))
    return sorted(by_area)
