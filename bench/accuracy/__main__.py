"""Command line of the accuracy measurement (bench/README.md; options as bench/cli.py).

    .venv/bin/python -m bench.accuracy list [AREA ...] [--match P]
    .venv/bin/python -m bench.accuracy run [AREA ...] [--match P] [--jobs N] [--out FILE] [--compare] [--promote]
    .venv/bin/python -m bench.accuracy compare RUN.json         # against the baselines
    .venv/bin/python -m bench.accuracy promote RUN.json         # into bench/accuracy/results + tables
    .venv/bin/python -m bench.accuracy table [AREA ...] [--from RUN.json]

AREA: the areas measured (default all of them); --match keeps the sweeps whose id (area/name)
matches (a glob, or a substring; repeatable). Exit status: 0 all good; 1 a sweep beyond its
contract (run, compare), a measure of sd worse than the baseline, a region or sweep gone,
or no baseline to compare against (compare); 2 a usage or tool error.
The console output goes to build/logs/accuracy/<stamp>.log too.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

from bench import cli

from . import results, sources, tables

MEASURES = (
    "contract",
    "ulp_max",
    "ulp_p999",
    "ulp_scale_max",
    "ulp_scale_p999",
    "mixed_max",
    "mixed_p999",
    "rel_max",
    "norm_rel",
    "norm_inf_rel_max",
    "nonfinite",
    "abs_max",
    "rel_cond_max",
    "est_ratio_max",
    "backward_max",
    "orth_max",
    "grad_max",
    "excess_max",
    "tol_ratio_max",
    "fnorm_max",
    "kkt_max",
    "unconverged",
    "off_target",
    "numpy_dist_max",
    "short_max",
    "error_below_true",
    "e2e_rel_max",
    "missed_max",
    "extra_max",
    "solve_max",
    "growth_max",
    "residual_max",
    "exact_data_max",
    "nonfinite_max",
    "rejected_max",
)

# Counts where fewer is worse: the converged runs of integrate, the converged entries of differentiate.
MEASURES_DOWN = (
    "converged_max",
    "entries",
)


def select(a):
    sws = sources.sweeps(a.chosen)
    return [s for s in sws if cli.matches(a.match, s.id)]


def failures(run):
    """'id [region]: N x the bound at <worst>' for every region beyond its contract."""
    out = []
    for s in run["sweeps"]:
        for label, r in s["regions"].items():
            if r.get("contract") is not None and not r["contract"] <= 1.0:
                out.append(f"{s['id']} [{label}]: {r['contract']:.3g} x the bound at {r.get('worst')}")
    return out


def cmd_list(a):
    sws = select(a)
    for s in sws:
        print(s.id)
    print(f"{len(sws)} sweeps")
    return 0


def cmd_run(a):
    from bench import console as C

    from . import compare as CMP
    from . import parallel, report

    sws = select(a)
    t0 = time.perf_counter()
    stamp, env = results.new_stamp(), results.env()
    versions = CMP.versions() if a.compare else {}
    if versions:
        env["comparators"] = versions
    jobs = a.jobs
    plus = "+" if env.get("dirty") else ""
    path = pathlib.Path(a.out) if a.out else results.RUN_DIR / f"{stamp}-{env['commit']}{plus}.json"
    log_path = cli.log_file("accuracy")
    with open(log_path, "w") as log:
        con = C.Console(log=log, show=not a.quiet)
        con.lines(report.header(stamp, env, sws, jobs, a.compare, versions, path, log_path))
        con.start(len(sws), "accuracy")

        def progress(i, n, r):
            el = time.perf_counter() - t0
            left = el / i * (n - i) if i else None
            what = f"{r['area']} · {r['name']}"
            eta = f", ~{C.fmt_duration(left)} left" if left else ""
            con.step(i, "accuracy", what, el, left, f"[{i}/{n}] {what}  ({C.fmt_duration(el)} elapsed{eta})")
            con.lines(report.sweep_lines(r, con))

        out = parallel.measure_all(sws, compare=a.compare, jobs=jobs, progress=progress)
        con.close()
        run = results.new_run(out, compare=a.compare, e=env, stamp=stamp)
        results.save(run, path)
        bad = failures(run)
        con.lines(report.summary(run, time.perf_counter() - t0, path, log_path, bad))
        if a.promote:
            promote(run, con)
    return 1 if bad else 0


def compare(run, base, partial=False) -> dict:
    """The findings of a run against the baselines: beyond (the contract), worse (id, region,
    moves, worst point), new, gone, changed (comparators: id, what, moves), better (a count).
    A measure is worse when it grew (a measure of MEASURES_DOWN: when it fell), or became None or
    NaN where the baseline had a value; a
    region is worse when it is gone, has fewer points than the baseline, or is unmapped (its
    field has no value from the measurement). partial: the run measured only some sweeps of
    its areas, so missing ones are not gone."""
    f = {"beyond": failures(run), "worse": [], "new": [], "gone": [], "changed": [], "better": 0}
    for s in run["sweeps"]:
        b = base.get(s["id"])
        if b is None:
            f["new"].append(s["id"])
            continue
        f["changed"] += comparator_changes(s, b)
        for label in b["regions"]:
            if label not in s["regions"]:
                f["worse"].append((s["id"], label, ["region gone"], None))
        for label, r in s["regions"].items():
            rb = b["regions"].get(label)
            if rb is None:
                f["new"].append(f"{s['id']} [{label}]")
                continue
            got_worse = []
            if r.get("unmapped"):
                got_worse.append("unmapped: no value for this field")
            pn, po = r.get("points", 0), rb.get("points", 0)
            if pn < po:
                got_worse.append(f"points {po} -> {pn}")
            elif pn > po:
                f["better"] += 1
            for m in MEASURES + MEASURES_DOWN:
                new, old = r.get(m), rb.get(m)
                if old is None or new == old or (new != new and old != old):  # NaN = NaN
                    continue
                if new is None or new != new:  # a value lost, or turned NaN
                    got_worse.append(f"{m} {tables.num(old)} -> {'none' if new is None else 'NaN'}")
                elif old != old or (new > old if m in MEASURES_DOWN else new < old):
                    f["better"] += 1
                else:
                    got_worse.append(f"{m} {tables.num(old)} -> {tables.num(new)}")
            if got_worse:
                f["worse"].append((s["id"], label, got_worse, r.get("worst")))
    ids = {s["id"] for s in run["sweeps"]}
    areas = {s["area"] for s in run["sweeps"]}
    f["gone"] = [i for i, b in base.items() if i not in ids and b["area"] in areas and not partial]
    return f


def comparator_changes(s, b) -> list:
    """(id, what, moves) for the comparator libraries whose results moved (information only: the
    exit status follows sd)."""
    out = []
    new, old = s.get("comparators"), b.get("comparators")
    if not new or not old:
        return out
    for lib, regs in new.items():
        prev = old.get(lib)
        if not isinstance(regs, dict) or not isinstance(prev, dict) or "error" in regs or "error" in prev:
            if regs != prev:
                out.append((s["id"], lib, [f"{_state(prev)} -> {_state(regs)}"]))
            continue
        for label, r in regs.items():
            moved = [f"{m} {tables.num(prev.get(label, {}).get(m))} -> {tables.num(r.get(m))}"
                     for m in MEASURES + MEASURES_DOWN if r.get(m) != prev.get(label, {}).get(m)]  # fmt: skip
            if moved:
                out.append((s["id"], f"{lib} {label}", moved))
    return out


def _state(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, str):
        return v
    return "error" if "error" in v else "measured"


def cmd_compare(a):
    from bench import console as C

    con = C.Console()
    run = results.load(a.run)
    areas = sorted({s["area"] for s in run["sweeps"]})
    base = results.baseline_sweeps(areas)
    if not base:
        con.lines(C.notes([f"No baseline yet for {', '.join(areas)}: promote a run first."]))
        return 1
    partial = len(run["sweeps"]) < sum(1 for b in base.values() if b["area"] in areas)
    f = compare(run, base, partial)
    con.lines(
        C.block(
            "stardust accuracy compare",
            [
                ("Run", run["run"]),
                ("Sweeps", f"{len(run['sweeps']):,}"),
                ("Baseline sweeps", f"{sum(1 for b in base.values() if b['area'] in areas):,}"),
                ("Partial run", "yes" if partial else "no"),
            ],
            [("Areas", ", ".join(areas))],
        )  # fmt: skip
    )
    w = con.width()

    def section(title, rows):
        if rows:
            con.print(title)
            for head, detail in rows:
                con.lines(C.split_label(head, 4, w))
                if detail:
                    con.lines(C.wrap([(d, d) for d in detail], 6, w, sep=", "))

    section("Beyond the contract:", [(f"  {x}", []) for x in f["beyond"]])
    section(
        "Worse than the baseline:",
        [(f"  {i} [{lab}]", moves + ([f"worst at {wst}"] if wst else [])) for i, lab, moves, wst in f["worse"]],
    )
    section("New:", [(f"  {x}", []) for x in f["new"]])
    section("Gone:", [(f"  {x}", []) for x in f["gone"]])
    shown = f["changed"][:40]
    section("Comparators that changed (information only):", [(f"  {i} [{what}]", moves) for i, what, moves in shown])
    nbad = len(f["worse"]) + len(f["beyond"]) + len(f["gone"])
    con.lines(
        [""]
        + C.block(
            "Compare summary",
            [
                ("Beyond the contract", str(len(f["beyond"]))),
                ("Worse", str(len(f["worse"]))),
                ("New", str(len(f["new"]))),
                ("Gone", str(len(f["gone"]))),
                ("Measures better", str(f["better"])),
                ("Comparator changes", str(len(f["changed"]))),
            ],
        )  # fmt: skip
        + C.notes(
            ([f"{len(f['changed']) - 40} more comparator changes are not shown."] if len(f["changed"]) > 40 else [])
            + (
                [
                    f"Nothing is worse: make the run the baseline with\n.venv/bin/python -m bench.accuracy promote {a.run}"
                ]
                if not nbad
                else [
                    "The exit status is 1: a measure of sd is worse than the baseline,\nbeyond its contract, "
                    "or a sweep is gone."
                ]
            )
            + ["Comparator changes never fail the comparison\n(a new library version, a fixed library)."]
        )
    )
    return 1 if nbad else 0


def promote(run, con=None):
    from bench import console as C

    con = con or C.Console()
    areas = results.promote(run)
    tables.write(areas)
    rows = [[a, str(results.baseline_path(a).relative_to(results.ROOT)), ", ".join(t[0] for t in tables.parts(a))]
            for a in areas]  # fmt: skip
    con.lines(
        [""]
        + C.block("stardust accuracy promote", [("Run", run["run"]), ("Areas", str(len(areas)))])
        + C.table(["Area", "Baseline", "Table"], rows, "lll")
        + [C.rule()]
    )


def cmd_promote(a):
    promote(results.load(a.run))
    return 0


def cmd_table(a):
    from bench import console as C

    data = results.load(a.from_) if a.from_ else None
    if a.areas:
        areas = a.chosen
    elif data:
        areas = sorted({s["area"] for s in data["sweeps"]})
    else:
        areas = list(sources.AREAS)
    written = tables.write(areas, data)
    C.Console().lines(
        C.block("stardust accuracy table", [("From", a.from_ or "the baselines"), ("Tables", str(len(written)))])
        + C.table(["Table"], [[str(p.relative_to(results.ROOT))] for p in written], "l")
        + [C.rule()]
    )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.accuracy", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    parsers = {}
    for name in ("list", "run"):
        p = parsers[name] = sub.add_parser(name)
        cli.add_areas(p, sources.AREAS)
        cli.add_match(p)
        if name == "run":
            cli.add_jobs(p)
            cli.add_out(p, "the run (default build/accuracy/<stamp>-<commit>.json)")
            p.add_argument("--promote", action="store_true", help="make the run the baseline of its areas")
            p.add_argument("-q", "--quiet", action="store_true", help="only the log file, nothing on the terminal")
            p.add_argument("--compare", action="store_true", help="also the comparator libraries (compare.py)")
    p = sub.add_parser("compare")
    p.add_argument("run", metavar="RUN.json")
    p = sub.add_parser("promote")
    p.add_argument("run", metavar="RUN.json")
    p = parsers["table"] = sub.add_parser("table")
    cli.add_areas(p, sources.AREAS)
    p.add_argument("--from", dest="from_", metavar="RUN.json")
    a = ap.parse_args(argv)
    if a.cmd in parsers:
        a.chosen = cli.areas(parsers[a.cmd], a, sources.AREAS)  # an unknown area: exit status 2
    sys.path.insert(0, str(results.ROOT))
    try:
        return {"list": cmd_list, "run": cmd_run, "compare": cmd_compare, "promote": cmd_promote, "table": cmd_table}[
            a.cmd
        ](a)
    except (OSError, ValueError) as e:  # a missing or malformed run file
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
