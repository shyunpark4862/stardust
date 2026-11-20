"""Compare a run with a baseline: which sd timings got slower.

A timing is flagged when it is more than `threshold` (10%, the ROADMAP's baseline rule)
slower than the baseline. Each line shows the noise band, the larger of the two
measurements' (median / min - 1): a slowdown inside it may be the machine, not the code,
and is marked so. The comparators' own change (their median new / old ratio) is shown
too: when numpy or scipy also moved, the machine changed between the runs. A case
measured while other processes kept the CPU busy (in either run) is marked "busy"
and does not count as a slowdown (rerun it: `run --rerun-busy`).
"""

from __future__ import annotations

import statistics

from . import results
from .harness import fmt_ratio, fmt_time


def _pairs(new: dict, base: dict[str, dict]):
    for c in new["cases"]:
        b = base.get(c["id"])
        if b is None:
            continue
        for t in c["timings"]:
            if t.get("min") is None:
                continue
            bt = results.timing(b, t["label"], t["threads"])
            if bt is None or bt.get("min") is None:
                continue
            yield c, t, bt


def compare(new: dict, base: dict[str, dict], *, threshold: float = 0.10) -> dict:
    """{"slower": [...], "faster": [...], "same": n, "refs_drift": ratio | None, "missing": n}."""
    slower, faster, same = [], [], 0
    ref_ratios = []
    compared = set()
    for c, t, bt in _pairs(new, base):
        ratio = t["min"] / bt["min"]
        compared.add(c["id"])
        if t["label"] != "sd":
            ref_ratios.append(ratio)
            continue
        band = max(t.get("noise") or 0.0, bt.get("noise") or 0.0)
        busy = bool(c.get("busy") or base[c["id"]].get("busy"))
        row = {
            "id": c["id"],
            "threads": t["threads"],
            "new": t["min"],
            "base": bt["min"],
            "ratio": ratio,
            "band": band,
            "beyond_noise": ratio - 1 > band and not busy,
            "busy": busy,
            "base_run": bt.get("run") or base[c["id"]].get("run"),
        }
        if ratio > 1 + threshold:
            slower.append(row)
        elif ratio < 1 / (1 + threshold):
            faster.append(row)
        else:
            same += 1
    slower.sort(key=lambda r: -r["ratio"])
    faster.sort(key=lambda r: r["ratio"])
    return {
        "slower": slower,
        "faster": faster,
        "same": same,
        "refs_drift": statistics.median(ref_ratios) if ref_ratios else None,
        "missing": sum(1 for c in new["cases"] if c["id"] not in compared),
    }


def report(res: dict, *, threshold: float = 0.10, show_faster: bool = False) -> str:
    lines = []
    n_slow = len(res["slower"])
    real = sum(r["beyond_noise"] for r in res["slower"])
    lines.append(
        f"sd timings more than {threshold * 100:.0f}% slower than the baseline: {n_slow} "
        f"({real} beyond the noise band); faster: {len(res['faster'])}; about the same: {res['same']}."
    )
    if res["refs_drift"] is not None:
        d = res["refs_drift"]
        lines.append(
            f"Comparator libraries, new / baseline (median): {fmt_ratio(d)}x"
            + (": the machine may have changed between the runs." if abs(d - 1) > threshold else ".")
        )
    if res["missing"]:
        lines.append(f"{res['missing']} cases have no baseline and were not compared.")

    def table(rows, title):
        out = [
            "",
            f"## {title}",
            "",
            "| Case | Threads | Baseline | New | New / baseline | Noise band | Verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            if r.get("busy"):
                verdict = "busy"
            else:
                verdict = ("slower" if r["beyond_noise"] else "within noise") if r["ratio"] > 1 else "faster"
            out.append(
                f"| `{r['id']}` | {r['threads']} | {fmt_time(r['base'])} | {fmt_time(r['new'])} | "
                f"{fmt_ratio(r['ratio'])} | {r['band'] * 100:.0f}% | {verdict} |"
            )
        return out

    if res["slower"]:
        lines += table(res["slower"], "Slower")
    if show_faster and res["faster"]:
        lines += table(res["faster"], "Faster")
    return "\n".join(lines) + "\n"


def _label(cid: str) -> str:
    """'area · name [size]' from a case id 'area/group/name [size]'."""
    area, _, rest = cid.split("/", 2)
    return f"{area} · {rest}"


def _threads(t) -> str:
    return "default threads" if t is None else f"{t} thread" + ("" if t == 1 else "s")


def console_report(res: dict, new: dict, base_label: str, threshold: float = 0.10, show_faster: bool = False) -> list:
    """The comparison for the console (bench/console.py): a summary block, then each slower
    sd timing as a few lines (and the faster ones with show_faster), then notes."""
    from ..console import block, notes, rule, split_label

    n_slow = len(res["slower"])
    real = sum(r["beyond_noise"] for r in res["slower"])
    busy = sum(1 for r in res["slower"] + res["faster"] if r.get("busy"))
    drift = res["refs_drift"]
    git = (new.get("env") or {}).get("git") or {}
    lines = [""] + block(
        "bench.speed compare",
        [
            ("New run", new.get("run", "?")),
            ("Threshold", f"{threshold * 100:.0f}%"),
            ("Slower", f"{n_slow:,}"),
            ("Beyond the noise", f"{real:,}"),
            ("Faster", f"{len(res['faster']):,}"),
            ("About the same", f"{res['same']:,}"),
            ("Measured while busy", f"{busy:,}"),
            ("Comparators moved", f"{drift:.2f}x" if drift is not None else "-"),
            ("Commit", f"{git.get('commit') or '?'}{' (dirty)' if git.get('dirty') else ''}"),
            ("Not in the baseline", f"{res['missing']:,}"),
        ],
        [("Baseline", base_label)],
    )

    def section(rows, title):
        out = ["", title, rule("-")]
        for r in rows:
            out += ["  " + h for h in split_label(_label(r["id"]), 4, 98)]
            if r.get("busy"):
                verdict = "measured while busy: run it again"
            elif r["ratio"] > 1:
                verdict = "slower" if r["beyond_noise"] else "within the noise"
            else:
                verdict = "faster"
            change = f"{r['ratio']:.2f}x slower" if r["ratio"] > 1 else f"{1 / r['ratio']:.2f}x faster"
            out.append(
                f"      {_threads(r['threads'])}: {fmt_time(r['base'])} -> {fmt_time(r['new'])}   "
                f"{change}   noise {r['band'] * 100:.0f}%   {verdict}"
            )
        return out

    if res["slower"]:
        lines += section(res["slower"], f"Slower than the baseline by more than {threshold * 100:.0f}%")
    if show_faster and res["faster"]:
        lines += section(res["faster"], f"Faster than the baseline by more than {threshold * 100:.0f}%")
    lines += ["", rule()]
    items = [
        "A timing is 'slower' when it is past the threshold and past the noise band (the larger\n"
        "of the two measurements' median / min - 1); inside the band it may be the machine.",
    ]
    if drift is not None and abs(drift - 1) > threshold:
        items.append(
            f"The comparator libraries moved too ({drift:.2f}x, new / baseline): the machine may\n"
            "have changed between the runs."
        )
    if busy:
        items.append(
            "Timings measured while other processes kept the CPU busy do not count; measure\nthem again with run --rerun-busy."
        )
    if res["faster"] and not show_faster:
        items.append("--faster lists the faster timings too.")
    return lines + notes(items)
