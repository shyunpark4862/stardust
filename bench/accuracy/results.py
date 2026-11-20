"""Run files and baselines of the accuracy measurement (the same layout as bench/speed).

A run file (build/accuracy/<stamp>-<commit>.json, ignored by git):

    {"schema": 1, "run": "<stamp>-<commit>[+]", "env": {...},
     "sweeps": [{"id": "<area>/<name>", "area", "name", "points", "regions": {region: entry},
                 ["comparators": {library: {region: entry} | "not run" | {"error": message}}]}]}

The run id ends in "+" when the working tree differs from the commit; env holds the commit,
dirty, the changed files (git status), the date, the machine, Python and per area the SHA-256
of its reference folders (references). A region is named by what it holds: the guaranteed
region and the rest (guaranteed, loose, mixed: special, stats), the result's range (array:
normal, subnormal), a condition number (linalg: kappa=...), a probability range (stats/mv
cdfs: tiny), a setting (optimize: default, xtol=0, ...) or a result field (regression). An
entry has "points", "contract" (the largest error over the bound: at most 1 passes; null where
no bound is documented), "bound", "worst" (the input where the contract error, else the
relative error, is largest) and the measures it has, among: ulp_max, ulp_p999 (ulp_distance
to the reference rounded to a double), ulp_scale_max, ulp_scale_p999 (ULP over the bound's
scale), mixed_max, mixed_p999, rel_max, norm_rel (2-norm, regression), norm_inf_rel_max
(infinity norm: linalg, optimize), nonfinite (points where the reference is finite and sd's
value is not), abs_max, rel_cond_max, backward_max, orth_max, grad_max, excess_max,
tol_ratio_max, fnorm_max, kkt_max, est_ratio_max, numpy_dist_max, unconverged, off_target,
excluded (a comparator's points left out), unmapped (a field without a value).

The files are JSON as Python writes it: an infinite or undefined measure is the token Infinity
or NaN, which a strict JSON parser outside Python does not read.

A baseline (bench/accuracy/results/<area>.json, in the repository) has the same sweeps, each
with the "run" it came from, and "runs": {run: env}. Promoting a run replaces the sweeps it
measured and keeps the others.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import platform

SCHEMA = 1
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "bench" / "accuracy" / "results"
RUN_DIR = ROOT / "build" / "accuracy"


# the reference folders of each area (their SHA-256 is part of a run's environment)
REFERENCES = {
    "array": ("tests/array/ref",),
    "linalg": ("tests/linalg/ref",),
    "special": ("tests/special/ref",),
    "stats": ("tests/stats/ref",),
    "regression": ("tests/regression/ref",),
    "optimize": ("tests/optimize/ref",),
}


def references_sha256(area: str) -> str:
    """One SHA-256 over an area's reference files (their paths and bytes, in path order)."""
    h = hashlib.sha256()
    for folder in REFERENCES.get(area, ()):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(ROOT)).encode() + b"\0")
                h.update(p.read_bytes())
    return h.hexdigest()


def env() -> dict:
    from bench.speed.harness import git_info

    g = git_info(str(ROOT))
    return {
        "commit": g["commit"] or "unknown",
        "dirty": g["dirty"],
        "changed": g["changed"],
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "references": {a: references_sha256(a) for a in REFERENCES},
    }


def new_run(sweeps: list[dict], compare: bool = False, e: dict | None = None, stamp: str | None = None) -> dict:
    """A run file's content; e and stamp: the environment and time stamp taken when it started."""
    e = dict(e or env())
    if compare and "comparators" not in e:
        from . import compare as C

        e["comparators"] = C.versions()
    stamp = stamp or new_stamp()
    return {
        "schema": SCHEMA,
        "run": f"{stamp}-{e['commit']}{'+' if e.get('dirty') else ''}",
        "env": e,
        "sweeps": sweeps,
    }


def new_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


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
                if "comparator_seconds" in old[s["id"]]:
                    s["comparator_seconds"] = old[s["id"]]["comparator_seconds"]
            elif "comparators" in s:
                s["comparators_run"] = run["run"]
        sweeps = [s for s in base["sweeps"] if s["id"] not in ids] + new
        sweeps.sort(key=lambda s: s["id"])
        used = {s["run"] for s in sweeps} | {s["comparators_run"] for s in sweeps if "comparators_run" in s}
        runs = {k: v for k, v in base["runs"].items() if k in used}
        runs[run["run"]] = run["env"]
        save({"schema": SCHEMA, "area": area, "runs": runs, "sweeps": sweeps}, baseline_path(area))
    return sorted(by_area)
