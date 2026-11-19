"""Compare a run with a baseline: which sd timings got slower.

A timing is flagged when it is more than `threshold` (10%, the ROADMAP's baseline rule)
slower than the baseline. Each line shows the noise band, the larger of the two
measurements' (median / min - 1): a slowdown inside it may be the machine, not the code,
and is marked so. The comparators' own change (their median new / old ratio) is shown
too: when numpy or scipy also moved, the machine changed between the runs. A case
measured while other processes kept the CPU busy (in either run) is marked "바쁠 때 잼"
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
        row = {"id": c["id"], "threads": t["threads"], "new": t["min"], "base": bt["min"], "ratio": ratio,
               "band": band, "beyond_noise": ratio - 1 > band and not busy, "busy": busy,
               "base_run": bt.get("run") or base[c["id"]].get("run")}
        if ratio > 1 + threshold:
            slower.append(row)
        elif ratio < 1 / (1 + threshold):
            faster.append(row)
        else:
            same += 1
    slower.sort(key=lambda r: -r["ratio"])
    faster.sort(key=lambda r: r["ratio"])
    return {"slower": slower, "faster": faster, "same": same,
            "refs_drift": statistics.median(ref_ratios) if ref_ratios else None,
            "missing": sum(1 for c in new["cases"] if c["id"] not in compared)}


def report(res: dict, *, threshold: float = 0.10, show_faster: bool = False) -> str:
    lines = []
    n_slow = len(res["slower"])
    real = sum(r["beyond_noise"] for r in res["slower"])
    lines.append(f"기준선보다 {threshold * 100:.0f}% 넘게 느린 sd 측정: {n_slow}개 (잡음 폭 밖 {real}개), "
                 f"빠른 것 {len(res['faster'])}개, 비슷한 것 {res['same']}개.")
    if res["refs_drift"] is not None:
        d = res["refs_drift"]
        lines.append(f"비교 대상 라이브러리 시간의 변화(새 ÷ 기준선, 중앙값): {fmt_ratio(d)}배"
                     + (" — 기계 상태가 달라졌을 수 있다." if abs(d - 1) > threshold else "."))
    if res["missing"]:
        lines.append(f"기준선에 없는 사례 {res['missing']}개는 비교하지 않았다.")

    def table(rows, title):
        out = ["", f"## {title}", "", "| 사례 | 스레드 | 기준선 | 새 측정 | 새 ÷ 기준선 | 잡음 폭 | 판단 |",
               "|---|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            if r.get("busy"):
                verdict = "바쁠 때 잼"
            else:
                verdict = ("느려짐" if r["beyond_noise"] else "잡음 안") if r["ratio"] > 1 else "빨라짐"
            out.append(f"| `{r['id']}` | {r['threads']} | {fmt_time(r['base'])} | {fmt_time(r['new'])} | "
                       f"{fmt_ratio(r['ratio'])} | {r['band'] * 100:.0f}% | {verdict} |")
        return out

    if res["slower"]:
        lines += table(res["slower"], "느린 것")
    if show_faster and res["faster"]:
        lines += table(res["faster"], "빠른 것")
    return "\n".join(lines) + "\n"
