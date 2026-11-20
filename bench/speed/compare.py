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
