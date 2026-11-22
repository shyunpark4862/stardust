"""Results files: one JSON Lines file per run, and one baseline JSON per area.

A run file (build/speed/<stamp>[-label].jsonl, ignored by git) grows by one line per event,
so a long run never rewrites what it has written:

    {"schema": 2, "kind": "run", "run": "<stamp>-<commit>", "env": {...}, "options": {...}}
    {"case": {"id", "area", "group", "name", "size", "n", "cost", "note", "rounds", ...,
              "timings": [{"label", "lib", "threads", "n", "min", "median", "noise",
                           "batches", "calls", "times", "nan", "abs_err", "check", "error"}]}}
    {"env": {...}}          the whole env again, when the workers added to it
    {"options": {...}}      changed options (progress: next, complete, elapsed_s)

A later case line replaces the earlier one of the same id (it holds every round so far,
with the raw batch times, so `run --resume` continues from it). load() assembles the lines
into one dict {"schema", "run", "env", "options", "cases"}; a last line cut short by an
interruption is ignored.

`threads` is the thread count the call ran at: stardust's pool for "sd", the BLAS limit
for a "follow" comparator, and null for a "fixed" comparator (its library's defaults).
Times are seconds per call. `nan` is the NaN share of the call's result; `abs_err` (cases
with a reference, registry.Case.abs_err) the largest error of the result against it;
`check` (smoke runs) is the results check against sd: {"ok", "text"}.

A baseline (bench/speed/results/<area>.json, in the repository) is one JSON:
{"schema", "area", "runs": {run: env}, "cases": [...]}. Its timings keep min, median,
noise, batches, calls, n, nan and abs_err (not the batch times), the run each came from and that
run's rounds.
Promoting a run merges it timing by timing: a timing the run measured replaces the
baseline's (label, threads) timing, every other timing stays, so a run of one thread
count, without R, or of one function changes only what it measured. A timing measured in
fewer rounds than the baseline's is refused unless forced.
"""

from __future__ import annotations

import json
import pathlib

SCHEMA = 2
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "bench" / "speed" / "results"
RUN_DIR = ROOT / "build" / "speed"

# what a baseline keeps of a run's timing (with the run and its rounds)
BASELINE_TIMING = (
    "label", "lib", "threads", "n", "min", "median", "noise", "batches", "calls", "nan", "abs_err", "error"
)  # fmt: skip


def load(path) -> dict:
    """A run file (JSON Lines) or a baseline (JSON) as one dict."""
    text = pathlib.Path(path).read_text()
    first = text.split("\n", 1)[0]
    try:
        head = json.loads(first)
    except json.JSONDecodeError:
        head = None
    if isinstance(head, dict) and head.get("kind") == "run":
        data = _assemble(text.splitlines())
    else:
        data = json.loads(text)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: results schema {data.get('schema')}, expected {SCHEMA}")
    return data


def _assemble(lines: list[str]) -> dict:
    head = json.loads(lines[0])
    data = {"schema": head["schema"], "run": head["run"], "env": head["env"], "options": head["options"]}
    cases: dict[str, dict] = {}
    for k, line in enumerate(lines[1:], 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                break  # the last line, cut short
            raise
        if "case" in obj:
            cases[obj["case"]["id"]] = obj["case"]
        elif "env" in obj:
            data["env"] = obj["env"]
        elif "options" in obj:
            data["options"].update(obj["options"])
    order = {cid: i for i, cid in enumerate(data["options"].get("case_ids") or [])}
    data["cases"] = sorted(cases.values(), key=lambda c: order.get(c["id"], len(order)))
    return data


class RunFile:
    """Appends the lines of a run file (start() truncates it, or resume=True appends)."""

    def __init__(self, path, *, resume: bool = False):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists():  # drop a last line an interruption cut short
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                with open(self.path, "r+b") as fh:
                    fh.truncate(raw.rfind(b"\n") + 1)
        self.fh = open(self.path, "a" if resume else "w", buffering=1)  # noqa: SIM115

    def _line(self, obj) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()

    def start(self, run: str, env: dict, options: dict) -> None:
        self._line({"schema": SCHEMA, "kind": "run", "run": run, "env": env, "options": options})

    def case(self, rec: dict) -> None:
        self._line({"case": rec})

    def env(self, env: dict) -> None:
        self._line({"env": env})

    def options(self, changed: dict) -> None:
        self._line({"options": changed})

    def close(self) -> None:
        self.fh.close()


def save(data: dict, path) -> None:
    """Write a baseline (one JSON)."""
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
    the env of its sd timings' run under "env"."""
    out = {}
    paths = [baseline_path(a) for a in areas] if areas else sorted(BASELINE_DIR.glob("*.json"))
    for p in paths:
        if not p.exists():
            continue
        b = load(p)
        for c in b["cases"]:
            out[c["id"]] = dict(c, env=b["runs"].get(case_run(c), {}))
    return out


def case_run(c: dict) -> str | None:
    """The run a case's sd timings came from (its first sd timing's; a run file's cases
    have none)."""
    for t in c["timings"]:
        if t["label"] == "sd" and t.get("run"):
            return t["run"]
    return c.get("run")


class PromoteRefused(ValueError):
    """The run has fewer rounds than the baseline for some cases (pass force=True)."""


def promote(run: dict, *, known_ids: list[str] | None = None, force: bool = False) -> dict[str, tuple[int, int]]:
    """Merge a run's cases into the baselines of their areas, timing by timing. Cases whose
    every timing failed are skipped. known_ids (the registry's ids, in its order): drop
    baseline cases no longer in the registry, and keep the registry's order. Raises
    PromoteRefused (before writing anything) when a case has fewer rounds than its baseline
    case, unless force. Returns {area: (cases merged, cases dropped)}."""
    by_area: dict[str, list[dict]] = {}
    for c in run["cases"]:
        if not any(t.get("min") is not None for t in c["timings"]):
            continue
        by_area.setdefault(c["area"], []).append(c)
    bases = {area: load_baseline(area) for area in by_area}
    if not force:
        fewer = []
        for area, new_cases in by_area.items():
            old = {c["id"]: c for c in (bases[area] or {}).get("cases", [])}
            for c in new_cases:
                b = old.get(c["id"])
                if b is None:
                    continue
                rounds = c.get("rounds") or 0
                for t in c["timings"]:
                    bt = timing(b, t["label"], t["threads"]) if t.get("min") is not None else None
                    if bt is not None and bt.get("min") is not None and (bt.get("rounds") or 0) > rounds:
                        fewer.append(f"{c['id']} {t['label']} ({rounds} < {bt['rounds']} rounds)")
        if fewer:
            raise PromoteRefused(
                f"{len(fewer)} cases have fewer rounds than the baseline, e.g. {fewer[0]}; force to promote anyway"
            )
    report = {}
    for area, new_cases in by_area.items():
        base = bases[area] or {"schema": SCHEMA, "area": area, "runs": {}, "cases": []}
        old = {c["id"]: c for c in base["cases"]}
        for c in new_cases:
            old[c["id"]] = merge_case(old.get(c["id"]), c, run["run"])
        order = {cid: i for i, cid in enumerate(known_ids or ())}
        dropped = 0
        cases = list(old.values())
        if known_ids is not None:
            dropped = sum(c["id"] not in order for c in cases)
            cases = [c for c in cases if c["id"] in order]
        cases.sort(key=lambda c: (order.get(c["id"], len(order)), c["id"]))
        used = {t.get("run") for c in cases for t in c["timings"]} - {None}
        runs = {k: v for k, v in base["runs"].items() if k in used}
        runs[run["run"]] = run["env"]
        base.update(schema=SCHEMA, runs=runs, cases=cases)
        save(base, baseline_path(area))
        report[area] = (len(new_cases), dropped)
    return report


def merge_case(old: dict | None, new: dict, run_id: str) -> dict:
    """A baseline case: the new case's facts, its measured timings, and the old case's
    timings the new run did not measure (or failed to)."""
    out = {k: v for k, v in new.items() if k != "timings"}
    out.pop("run", None)
    timings: dict[tuple, dict] = {}
    if old is not None:
        for t in old["timings"]:
            timings[(t["label"], t["threads"])] = dict(t)
    for t in new["timings"]:
        key = (t["label"], t["threads"])
        slim = {k: t[k] for k in BASELINE_TIMING if k in t}
        slim["run"] = run_id
        slim["rounds"] = new.get("rounds") or 0
        if new.get("busy"):
            slim["busy"] = True
        if t.get("min") is not None or key not in timings or timings[key].get("min") is None:
            timings[key] = slim
    out["timings"] = list(timings.values())
    out["busy"] = any(t.get("busy") for t in out["timings"])
    return out


def timing(case: dict, label: str, threads) -> dict | None:
    for t in case["timings"]:
        if t["label"] == label and t["threads"] == threads:
            return t
    return None
