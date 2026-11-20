"""Command line of the speed benchmarks (bench/README.md).

    .venv/bin/python -m bench.speed list [--area A] [--match P] [--quick]
    .venv/bin/python -m bench.speed estimate [selection] [--rounds R]
    .venv/bin/python -m bench.speed run [selection] [--rounds R] [--threads 1,10] [--no-r] [--out F]
                                        [--promote] [--timeout MIN] [--max-rss GB] [--no-notify]
                                        [--no-wait | --strict] [--quiet-cores C] [--smoke [--jobs N]]
    .venv/bin/python -m bench.speed run --resume build/speed/<run>.json
    .venv/bin/python -m bench.speed run --rerun-busy build/speed/<run>.json   # cases measured while busy
    .venv/bin/python -m bench.speed compare RUN.json [--base FILE] [--threshold 0.10] [--faster]
    .venv/bin/python -m bench.speed promote RUN.json      # into bench/speed/results/<area>.json + tables
    .venv/bin/python -m bench.speed table [AREA ...] [--from RUN.json]
    .venv/bin/python -m bench.speed cost [--check]

selection: --area A (repeatable), --match PATTERN (regex, or glob with * ?; repeatable),
--group G (repeatable), --quick (the quick subset).
"""

from __future__ import annotations

import argparse
import sys

from . import registry, results


def _selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--area", action="append", default=[], choices=registry.AREAS)
    p.add_argument("--match", action="append", default=[])
    p.add_argument("--group", action="append", default=[])
    p.add_argument("--quick", action="store_true")


def _select(a) -> list[registry.Case]:
    cases = registry.load(a.area or None)
    return registry.select(cases, match=a.match, quick=a.quick, groups=a.group)


def cmd_list(a) -> int:
    cases = _select(a)
    for c in cases:
        refs = ", ".join(r.label for r in c.refs)
        flags = ("Q " if c.quick else "  ") + ("C " if c.cost else "  ")
        print(f"{flags}{c.id}  [{', '.join(map(str, c.threads))}T; {refs or '-'}]")
    print(
        f"{len(cases):,} cases (Q: {sum(c.quick for c in cases)} in the quick subset, "
        f"C: {sum(bool(c.cost) for c in cases)} feed a cost table)",
        file=sys.stderr,
    )
    return 0


def cmd_estimate(a) -> int:
    from .console import fmt_duration, rule, table
    from .run import estimate

    cases = _select(a)
    by_area: dict[str, list] = {}
    for c in cases:
        by_area.setdefault(c.area, []).append(c)
    rows, total, total_unknown = [], 0.0, 0
    for area, cs in by_area.items():
        s, unknown = estimate(cs, a.rounds)
        total += s
        total_unknown += unknown
        rows.append([area, f"{len(cs):,}", f"{len(cs) - unknown:,}", fmt_duration(s) if unknown < len(cs) else "-"])
    rows.append(["Total", f"{len(cases):,}", f"{len(cases) - total_unknown:,}", fmt_duration(total)])
    lines = table(["Area", "Cases", "In a baseline", f"Expected ({a.rounds} rounds)"], rows, "lrrr")
    lines.insert(-1, rule("-"))
    print("\n".join(lines))
    if total_unknown:
        print(f"{total_unknown:,} cases have no baseline yet: their time is not in the total.")
    return 0


def cmd_run(a) -> int:
    from .run import Limits, run

    limits = Limits(
        case_timeout_s=a.timeout * 60,
        rss_bytes=int(a.max_rss * 2**30),
        quiet_cores=a.quiet_cores,
        busy_cores=a.quiet_cores,
    )
    if a.rerun_busy:
        prev = results.load(a.rerun_busy)
        busy = [c["id"] for c in prev["cases"] if c.get("busy")]
        if not busy:
            print("No case of that run was measured while busy.", file=sys.stderr)
            return 0
        by_id = {c.id: c for c in registry.load(sorted({cid.split("/", 1)[0] for cid in busy}))}
        cases = [by_id[cid] for cid in busy if cid in by_id]
        if a.rounds_given is None:
            a.rounds = prev["options"]["rounds"]
        a.label = a.label or "rerun"
        print(f"Measuring again the {len(cases)} cases measured while busy ({a.rounds} rounds).", file=sys.stderr)
    elif a.resume:
        prev = results.load(a.resume)
        areas = sorted({cid.split("/", 1)[0] for cid in prev["options"]["case_ids"]})
        cases = registry.load(areas)
    else:
        cases = _select(a)
        if not cases:
            print("No case matches the selection.", file=sys.stderr)
            return 2
    threads = tuple(int(t) for t in a.threads.split(",")) if a.threads else None
    data = run(
        cases,
        rounds=a.rounds,
        threads=threads,
        use_r=not a.no_r,
        out=a.out,
        label=a.label,
        limits=limits,
        notify=not a.no_notify,
        resume=a.resume,
        smoke=a.smoke,
        jobs=a.jobs,
        quiet="strict" if a.strict else ("no" if a.no_wait else "wait"),
    )
    if data.get("stopped_busy"):
        return 3
    if data.get("interrupted"):
        return 130
    if a.promote and not a.smoke:
        _promote(data)
    return 0


def _promote(data) -> None:
    from . import tables

    known = [c.id for c in registry.load()]
    rep = results.promote(data, known_ids=known)
    cases = registry.load()
    for area, (n, dropped) in rep.items():
        base = results.load_baseline(area)
        path = tables.write(area, base, cases, f"bench/speed/results/{area}.json")
        extra = f", {dropped} no longer registered dropped" if dropped else ""
        print(f"{area}: {n} cases into the baseline{extra}; wrote {path.relative_to(results.ROOT)}")
    if any(c.get("cost") for c in data["cases"]):
        print("Cost-table cases changed: run `.venv/bin/python -m bench.speed cost` to rewrite the C++ cost tables.")


def cmd_promote(a) -> int:
    data = results.load(a.run)
    if data["options"].get("smoke"):
        print("A --smoke run cannot go into a baseline.", file=sys.stderr)
        return 2
    if not data["options"].get("complete", True) and not a.force:
        print("The run is not complete (finish it with --resume, or pass --force).", file=sys.stderr)
        return 2
    _promote(data)
    return 0


def cmd_compare(a) -> int:
    from .compare import compare, report

    new = results.load(a.run)
    if a.base:
        b = results.load(a.base)
        base = {c["id"]: dict(c, run=c.get("run", b.get("run"))) for c in b["cases"]}
    else:
        base = results.baseline_cases(sorted({c["area"] for c in new["cases"]}))
    res = compare(new, base, threshold=a.threshold)
    text = report(res, threshold=a.threshold, show_faster=a.faster)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    return 1 if any(r["beyond_noise"] for r in res["slower"]) else 0


def cmd_table(a) -> int:
    from . import tables

    cases = registry.load()
    if a.from_:
        data = results.load(a.from_)
        areas = a.areas or sorted({c["area"] for c in data["cases"]})
        for area in areas:
            path = tables.write(area, data, cases, a.from_, path=a.out)
            print(path)
        return 0
    for area in a.areas or registry.AREAS:
        base = results.load_baseline(area)
        if base is None:
            if a.areas:
                print(f"{area}: no baseline", file=sys.stderr)
            continue
        print(tables.write(area, base, cases, f"bench/speed/results/{area}.json").relative_to(results.ROOT))
    return 0


def cmd_cost(a) -> int:
    from . import cost

    changed_any = False
    for path, changed, n in cost.generate(check=a.check):
        changed_any |= changed
        state = ("differs" if changed else "up to date") if a.check else ("rewritten" if changed else "unchanged")
        print(f"{path.relative_to(results.ROOT)}: {n} rows measured in the baselines, {state}")
    if changed_any and not a.check:
        print("A C++ header changed: the next import rebuilds the core.")
    return 1 if (a.check and changed_any) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bench.speed", description="stardust speed benchmarks")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="list cases")
    _selection(s)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("estimate", help="expected run time, from the baselines")
    _selection(s)
    s.add_argument("--rounds", type=int, default=None, help="default: 1 with --quick, else 3")
    s.set_defaults(fn=cmd_estimate)

    s = sub.add_parser("run", help="measure")
    _selection(s)
    s.add_argument("--rounds", type=int, default=None, help="default: 1 with --quick, else 3")
    s.add_argument("--threads", help="comma-separated sd thread counts, e.g. 1 or 1,10")
    s.add_argument("--no-r", action="store_true", help="skip the R comparators")
    s.add_argument("--out", help="the run file (default build/speed/<time>[-label].json)")
    s.add_argument("--label", default="")
    s.add_argument("--promote", action="store_true", help="merge into the baselines and rewrite the tables")
    s.add_argument("--resume", help="continue an interrupted run file")
    s.add_argument("--timeout", type=float, default=15.0, help="minutes per case and round (default 15)")
    s.add_argument("--max-rss", type=float, default=16.0, help="GB, the worker and R together (default 16)")
    s.add_argument("--no-notify", action="store_true")
    s.add_argument("--smoke", action="store_true", help="call every case once: does it run? (not promotable)")
    s.add_argument(
        "--jobs",
        type=int,
        default=10,
        help="processes a --smoke run may use at once, R sessions included (default 10); measuring runs "
        "always go one case at a time",
    )
    s.add_argument("--rerun-busy", metavar="RUN.json", help="measure again the cases a run marked busy")
    s.add_argument("--no-wait", action="store_true", help="do not wait for a quiet machine")
    s.add_argument("--strict", action="store_true", help="stop when the machine does not get quiet")
    s.add_argument(
        "--quiet-cores",
        type=float,
        default=1.0,
        help="cores other processes may use: before the run, and per case (default 1)",
    )
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("compare", help="flag slowdowns against a baseline")
    s.add_argument("run")
    s.add_argument("--base", help="a results file (default: the baselines of the run's areas)")
    s.add_argument("--threshold", type=float, default=0.10)
    s.add_argument("--faster", action="store_true", help="also list what got faster")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("promote", help="merge a run into the baselines")
    s.add_argument("run")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("table", help="write bench/<area>/speed.md")
    s.add_argument("areas", nargs="*")
    s.add_argument("--from", dest="from_", help="a run file instead of the baselines")
    s.add_argument("--out", help="with --from and one area: write here instead")
    s.set_defaults(fn=cmd_table)

    s = sub.add_parser("cost", help="rewrite csrc/stats/cost_table.hpp and csrc/special/cost_table.hpp")
    s.add_argument("--check", action="store_true", help="only report whether they would change")
    s.set_defaults(fn=cmd_cost)

    a = p.parse_args(argv)
    a.rounds_given = getattr(a, "rounds", None)
    if getattr(a, "rounds", 0) is None:
        a.rounds = 1 if (a.quick or getattr(a, "smoke", False)) else 3
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
