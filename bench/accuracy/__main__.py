"""Command line of the accuracy measurement (tests/README.md).

    .venv/bin/python -m bench.accuracy list [--area A] [--match P]
    .venv/bin/python -m bench.accuracy run [--area A] [--match P] [--out F] [--promote]
    .venv/bin/python -m bench.accuracy compare RUN.json         # against the baselines
    .venv/bin/python -m bench.accuracy promote RUN.json         # into bench/accuracy/results + tables
    .venv/bin/python -m bench.accuracy table [AREA ...] [--from RUN.json]

--area is repeatable (default: every area), --match is a regex on the sweep id (area/name).
run exits 1 when a sweep exceeds its contract; compare exits 1 when a sweep exceeds it or got
worse than the baseline in any measure (even within the contract), and lists every such sweep.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from . import results, sources, tables

MEASURES = ("contract", "ulp_max", "ulp_p999", "ulp_cond_max", "ulp_cond_p999", "mixed_max", "mixed_p999", "rel_max")


def select(a):
    sws = sources.sweeps(a.area or None)
    if a.match:
        rx = re.compile(a.match)
        sws = [s for s in sws if rx.search(s.id)]
    return sws


def failures(run):
    out = []
    for s in run["sweeps"]:
        for label, r in s["regions"].items():
            if r.get("contract") is not None and not r["contract"] <= 1.0:
                out.append(f"FAIL     {s['id']} [{label}]: {r['contract']:.3g} x the bound at {r['worst']}")
    return out


def cmd_list(a):
    sws = select(a)
    for s in sws:
        print(s.id)
    print(f"{len(sws)} sweeps")


def cmd_run(a):
    sws = select(a)
    t0 = time.perf_counter()
    out = []
    for i, s in enumerate(sws, 1):
        out.append(sources.measure(s))
        if a.verbose:
            print(f"[{i}/{len(sws)}] {s.id}", file=sys.stderr, flush=True)
    run = results.new_run(out)
    path = a.out or results.RUN_DIR / f"{run['run']}.json"
    results.save(run, path)
    bad = failures(run)
    for line in bad:
        print(line)
    print(f"{len(out)} sweeps in {time.perf_counter() - t0:.1f} s -> {path}; {len(bad)} beyond the contract")
    if a.promote:
        promote(run)
    return 1 if bad else 0


def compare(run, base, partial=False):
    """(lines, sweeps worse or beyond the contract, measures better); partial: the run measured
    only some sweeps of its areas, so missing ones are not GONE."""
    lines, worse, better = [], 0, 0
    for s in run["sweeps"]:
        b = base.get(s["id"])
        if b is None:
            lines.append(f"NEW      {s['id']}")
            continue
        for label, r in s["regions"].items():
            rb = b["regions"].get(label)
            if rb is None:
                lines.append(f"NEW      {s['id']} [{label}]")
                continue
            got_worse = []
            for m in MEASURES:
                new, old = r.get(m), rb.get(m)
                if new is None or old is None or new == old:
                    continue
                if new > old:
                    got_worse.append(f"{m} {tables.num(old)} -> {tables.num(new)}")
                else:
                    better += 1
            if got_worse:
                worse += 1
                lines.append(f"WORSE    {s['id']} [{label}]: " + ", ".join(got_worse) + f" (worst at {r['worst']})")
    ids = {s["id"] for s in run["sweeps"]}
    areas = {s["area"] for s in run["sweeps"]}
    for i, b in base.items():
        if i not in ids and b["area"] in areas and not partial:
            lines.append(f"GONE     {i}")
    bad = failures(run)
    return bad + lines, worse + len(bad), better


def cmd_compare(a):
    run = results.load(a.run)
    areas = sorted({s["area"] for s in run["sweeps"]})
    base = results.baseline_sweeps(areas)
    if not base:
        print("no baseline yet for", ", ".join(areas))
        return 1
    partial = len(run["sweeps"]) < sum(1 for b in base.values() if b["area"] in areas)
    lines, nbad, better = compare(run, base, partial)
    for line in lines:
        print(line)
    print(f"{len(run['sweeps'])} sweeps: {nbad} worse or beyond the contract, {better} measures better")
    return 1 if nbad else 0


def promote(run):
    areas = results.promote(run)
    for p in tables.write(areas):
        print(f"wrote {p.relative_to(results.ROOT)}")


def cmd_promote(a):
    promote(results.load(a.run))


def cmd_table(a):
    data = results.load(a.from_) if a.from_ else None
    if a.areas:
        areas = a.areas
    elif data:
        areas = sorted({s["area"] for s in data["sweeps"]})
    else:
        areas = list(sources.AREAS)
    for p in tables.write(areas, data):
        print(f"wrote {p.relative_to(results.ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench.accuracy", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "run"):
        p = sub.add_parser(name)
        p.add_argument("--area", action="append", default=[], choices=sources.AREAS)
        p.add_argument("--match")
        if name == "run":
            p.add_argument("--out")
            p.add_argument("--promote", action="store_true")
            p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("run")
    p = sub.add_parser("promote")
    p.add_argument("run")
    p = sub.add_parser("table")
    p.add_argument("areas", nargs="*")
    p.add_argument("--from", dest="from_")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(results.ROOT))
    return {"list": cmd_list, "run": cmd_run, "compare": cmd_compare, "promote": cmd_promote, "table": cmd_table}[
        a.cmd
    ](a)


if __name__ == "__main__":
    sys.exit(main())
