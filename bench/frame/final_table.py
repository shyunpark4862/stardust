"""M12: bench/results/<dir>/*.json -> bench/results/<dir>/table.md.

Usage: python bench/final_table.py [final|final-fresh]
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _harness import env_info, fmt_ns  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "final"
DIR = pathlib.Path(__file__).parent / "results" / NAME
ALL = ["M9", "M10", "M11-1", "M11", "now-1", "now", "pandas", "polars-1", "polars"]
COLS = [c for c in ALL if (DIR / f"{c}.json").exists()]
R = {c: json.loads((DIR / f"{c}.json").read_text()) for c in COLS}
T = {c: R[c]["threads"] for c in COLS}
HEAD = {
    "M9": "M9",
    "M10": "M10",
    "M11-1": "M11 @1",
    "M11": f"M11 @{T.get('M11')}",
    "now-1": "now @1",
    "now": f"now @{T.get('now')}",
    "pandas": "pandas",
    "polars-1": "Polars @1",
    "polars": f"Polars @{T.get('polars')}",
}
OURS1, OURS = ("now-1", "now") if "now" in COLS else ("M11-1", "M11")
keys = list(R[OURS]["rows"])
sizes = sorted({int(k.split(" | ")[1]) for k in keys})


def t(c, k):
    return R[c]["rows"][k]


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def wtl(a, b, ks):
    """(rows where column a is >10% faster than b, within 10%, >10% slower)."""
    r = [t(b, k) / t(a, k) for k in ks]
    return sum(x > 1.1 for x in r), sum(1 / 1.1 <= x <= 1.1 for x in r), sum(x < 1 / 1.1 for x in r)


mode = "a different input each call (--fresh)" if NAME.endswith("fresh") else "one input per case, called over and over"
out = [
    f"# M12: final comparison ({NAME})",
    "",
    "Environment: " + ", ".join(f"{k} {v}" for k, v in env_info().items()),
    "",
    f"Inputs: {mode}. Rounds: {R[OURS]['rounds']} (each row: best over rounds). "
    "Builds: M9 64ca0c1, M10 d6aadd5, M11 e2195f7, now = the working tree.",
    "",
]

pairs = [
    (c, b)
    for c, b in [
        ("pandas", OURS1),
        ("polars-1", OURS1),
        ("pandas", OURS),
        ("polars", OURS),
        ("M9", OURS1),
        ("M11-1", OURS1),
        ("M9", OURS),
    ]
    if c in COLS and b in COLS and c != b
]
out += [
    f"## Summary: geometric mean of time ratios (>1: the second is faster); ours = {OURS1} / {OURS}",
    "",
    "| n | rows | " + " | ".join(f"{HEAD[a]} / {HEAD[b]}" for a, b in pairs) + " |",
    "|---:|---:|" + "---:|" * len(pairs),
]
for n in sizes:
    ks = [k for k in keys if k.endswith(f"| {n}")]
    out.append(
        f"| {n:,} | {len(ks)} | " + " | ".join(f"{gmean([t(a, k) / t(b, k) for k in ks]):.2f}" for a, b in pairs) + " |"
    )
vs = [(a, b) for a, b in [(OURS1, "pandas"), (OURS1, "polars-1"), (OURS, "pandas"), (OURS, "polars")] if b in COLS]
out += [
    "",
    "## Faster / within 10% / slower (rows)",
    "",
    "| n | " + " | ".join(f"{HEAD[a]} vs {HEAD[b]}" for a, b in vs) + " |",
    "|---:|" + "---|" * len(vs),
]
for n in sizes:
    ks = [k for k in keys if k.endswith(f"| {n}")]
    out.append(f"| {n:,} | " + " | ".join("%d / %d / %d" % wtl(a, b, ks) for a, b in vs) + " |")

for n in sizes:
    out += [
        "",
        f"## n = {n:,}",
        "",
        "| case | " + " | ".join(HEAD[c] for c in COLS) + " |",
        "|---|" + "---:|" * len(COLS),
    ]
    for k in keys:
        if k.endswith(f"| {n}"):
            out.append(f"| {k.split(' | ')[0]} | " + " | ".join(fmt_ns(t(c, k)) for c in COLS) + " |")
(DIR / "table.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
