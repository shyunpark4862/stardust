"""The console output of python -m bench.accuracy, in bench.speed's style (bench/console.py): a
header block, one small block per sweep above the progress bar (sd's measure, then each
comparator's, the contract in words), and at the end a summary block, per-area and
per-comparator tables, the failures and numbered notes."""

from __future__ import annotations

from bench import console as C

from . import tables

# the measure a region's line leads with, in this order: (key, words)
HEADLINE = [
    ("ulp_max", "{} ulp max"),
    ("mixed_max", "{} mixed max"),
    ("backward_max", "residual {} n·eps"),
    ("rel_max", "rel {}"),
    ("norm_rel", "norm rel {}"),
    ("grad_max", "|grad| {}"),
    ("fnorm_max", "|F| {}"),
    ("kkt_max", "KKT {} eps"),
    ("abs_max", "abs {}"),
]


def headline(e: dict) -> str:
    """A region entry in a few words: its leading measure, then unconverged and excluded counts."""
    parts = []
    for key, words in HEADLINE:
        if e.get(key) is not None:
            parts.append(words.format(tables.num(e[key])))
            break
    if e.get("rel_cond_max") is not None:
        parts.append(f"{tables.num(e['rel_cond_max'])} u·cond")
    if e.get("unconverged"):
        parts.append(f"{e['unconverged']} unconverged")
    if e.get("excluded"):
        parts.append(f"{e['excluded']} excluded")
    if not parts:
        return "no points" if not e.get("points") else ""
    return ", ".join(parts)


def contract_words(e: dict, con) -> tuple[str, str] | None:
    """('within the contract (0.03)', shown) or the same 'beyond' in red; None without a bound."""
    c = e.get("contract")
    if c is None:
        return None
    if c <= 1:
        text = f"within the contract ({tables.num(c, 2)})"
        return text, text
    text = f"beyond the contract ({tables.num(c)}x)"
    return text, con.color(text, "31")


def comparator_words(lib: str, v, label: str, con=None) -> tuple[str, str] | None:
    """A comparator's part of a region line (plain, shown), or None when it has no such
    function. Past the bound sd is held to, it says so (in red on a terminal)."""

    def same(t):
        return t, t

    if v is None:
        return None
    if isinstance(v, str):
        return same(f"{lib} {v}")
    if "error" in v:
        e = v["error"]
        if "Guarded" in e:
            return same(f"{lib} stopped ({e.split('Guarded: ')[-1][:30]})")
        return same(f"{lib} needs starting values" if "starting values" in e else f"{lib} error")
    r = v.get(label)
    if r is None:
        return None
    text = " ".join(t for t in (lib, headline(r)) if t)
    c = r.get("contract")
    if c is not None and c > 1:
        tail = f"beyond sd's bound ({tables.num(c)}x)"
        return f"{text}, {tail}", f"{text}, " + (con.color(tail, "31") if con else tail)
    if c is not None and text == lib:
        return same(f"{lib} within sd's bound ({tables.num(c, 2)})")
    return same(text)


def sweep_lines(s: dict, con) -> list[str]:
    """One sweep's block: the sweep, then per region its points, sd's measure and contract, and
    each comparator's measure, wrapped within the console width."""
    out = [f"  {s['area']} · {s['name']}"]
    w = con.width()
    for label, e in s["regions"].items():
        items = [(f"{label} ({e.get('points', 0):,})", f"{label} ({e.get('points', 0):,})")]
        h = headline(e)
        cw = contract_words(e, con)
        if h:
            items.append((f"sd {h}", f"sd {h}"))
            if cw:
                items.append(cw)
        elif cw:
            items.append(("sd " + cw[0], "sd " + cw[1]))
        for lib, v in (s.get("comparators") or {}).items():
            t = comparator_words(lib, v, label, con)
            if t:
                items.append(t)
        out += C.wrap(items, 6, w)
    return out


def header(stamp, env, sweeps, jobs, compare, versions, results_path, log_path) -> list[str]:
    areas = sorted({s.area for s in sweeps})
    commit = env.get("commit", "?") + (" (dirty)" if env.get("dirty") else "")
    wide = [("Areas", ", ".join(areas))]
    if compare:
        py = [f"{k} {v}" for k, v in versions.items() if v and not k.startswith(("R", "Boost"))]
        r = [f"{k[2:]} {v}" for k, v in versions.items() if v and k.startswith("R ")]
        wide.append(("Comparators", ", ".join(py) or "none installed"))
        if versions.get("R"):
            wide.append(("", f"R {versions['R']}" + (f" ({', '.join(r)})" if r else "")))
        if versions.get("Boost"):
            wide.append(("", f"Boost.Math {versions['Boost'].replace('boostorg/math ', '')}"))
    else:
        wide.append(("Comparators", "none (add --compare)"))
    wide += [("Results", str(results_path)), ("Log", str(log_path))]
    lines = C.block(
        "stardust accuracy run",
        [("Run", stamp), ("Commit", commit), ("Sweeps", f"{len(sweeps):,}"), ("Processes", str(jobs))],
        wide,
    )
    return [line.replace(":", " ", 1) if line.startswith(":") else line for line in lines]


def _worst(regions) -> float | None:
    cs = [r.get("contract") for r in regions if r.get("contract") is not None]
    return max(cs) if cs else None


def summary(run: dict, elapsed: float, results_path, log_path, failures: list[str]) -> list[str]:
    sws = run["sweeps"]
    regions = [r for s in sws for r in s["regions"].values()]
    compare = any("comparators" in s for s in sws)
    out = [""] + C.block(
        "Accuracy run summary",
        [
            ("Sweeps", f"{len(sws):,}"),
            ("Elapsed", C.fmt_duration(elapsed)),
            ("Regions", f"{len(regions):,}"),
            ("Beyond the contract", str(len(failures))),
            ("Areas", str(len({s["area"] for s in sws}))),
            ("Comparators", "measured" if compare else "not measured"),
        ],  # fmt: skip
    )
    rows = []
    for area in sorted({s["area"] for s in sws}):
        ss = [s for s in sws if s["area"] == area]
        rr = [r for s in ss for r in s["regions"].values()]
        beyond = sum(1 for r in rr if (r.get("contract") or 0) > 1)
        w = _worst(rr)
        rows.append([area, str(len(ss)), str(len(rr)), str(beyond), "-" if w is None else tables.num(w, 2)])
    out += C.table(["Area", "Sweeps", "Regions", "Beyond", "Worst / bound"], rows, "lrrrr") + [C.rule()]
    if compare:
        crows = []
        for area in sorted({s["area"] for s in sws}):
            libs: dict[str, list] = {}
            for s in (s for s in sws if s["area"] == area):
                for lib, v in (s.get("comparators") or {}).items():
                    libs.setdefault(lib, []).append(v)
            for lib, vs in libs.items():
                stopped = sum(1 for v in vs if isinstance(v, dict) and "Guarded" in v.get("error", ""))
                errors = sum(1 for v in vs if isinstance(v, dict) and "error" in v) - stopped
                notrun = sum(1 for v in vs if isinstance(v, str))
                beyond = sum(
                    1 for v in vs if isinstance(v, dict) and "error" not in v
                    for r in v.values() if (r.get("contract") or 0) > 1
                )  # fmt: skip
                crows.append([area, lib, str(len(vs)), str(stopped), str(errors + notrun), str(beyond)])
        out += C.table(["Area", "Compared with", "Sweeps", "Stopped", "Errors", "Beyond sd's bound"], crows,
                       "llrrrr") + [C.rule()]  # fmt: skip
    if failures:
        out += ["Beyond the contract:"] + [f"  {f}" for f in failures]
    out += C.notes(
        [
            f"Compare with the baselines (worse measures, new and gone sweeps):\n"
            f".venv/bin/python -m bench.accuracy compare {results_path}",
            f"Make this run the baseline and write the tables:\n.venv/bin/python -m bench.accuracy promote {results_path}",
            "'Worst / bound' is the largest contract error over its bound (at most 1 passes);\n"
            "'Beyond sd's bound' counts a comparator's regions past the bound sd is held to.",
            f"Results: {results_path}\nLog:     {log_path}",
        ]
    )
    return out
