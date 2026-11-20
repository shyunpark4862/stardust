"""Regenerate the C++ cost tables from the baselines.

The parallel split weighs each element by its measured one-thread cost
(csrc/stats/engine.hpp cost_of; csrc/special/cost.hpp weight_of). A case feeds a table
through its `cost` tag (table, key): its 1-thread sd time divided by its element count,
in ns. Several cases with one key (a distribution's parameter sets of different cost,
as the discrete distributions' small and wide supports) give their geometric mean.

A key no baseline case has keeps its row from the current header, so measuring one
area or one function and regenerating changes only what was measured. A stats "@memo"
key no baseline measures gets MEMO_PRESET_NS's value (the inverses) or, without a row yet,
MEMO_DEFAULT_NS, so the engine's narrow-support weight (stats/engine.hpp memo_cost_ns) is on
before the first measurement; the measurement replaces both. `--check` also lists the rows no registry case feeds (they keep their value
forever) and the registry keys without a row.

    .venv/bin/python -m bench.speed cost [--check]
"""

from __future__ import annotations

import math
import pathlib
import re

from . import results

ROOT = results.ROOT

# ns per element of an @memo row before it is measured. The memo lookups of the discrete
# distributions' value methods (pmf, logpmf, cdf, ccdf and their logs) took 1.0-2.2 ns per
# element (median 1.7 ns over 60 keys, one call each in a smoke run; binomial.cdf@memo 1.3 ns
# in timed runs): this is their middle.
MEMO_DEFAULT_NS = 1.5

# ns per element of the inverses' @memo rows before they are measured (9-1080 ns, far from
# MEMO_DEFAULT_NS): the 1-thread sd time per element of their cases (10^5 elements), the least of
# three smoke runs (`python -m bench.speed run stats --smoke`).
MEMO_PRESET_NS = {
    "bernoulli.icdf@memo": 20.6, "bernoulli.iccdf@memo": 20.9,
    "betabinomial.icdf@memo": 1080, "betabinomial.iccdf@memo": 935,
    "binomial.icdf@memo": 165, "binomial.iccdf@memo": 165, "binomial.ilogcdf@memo": 543, "binomial.ilogccdf@memo": 481,
    "discreteuniform.icdf@memo": 9.29, "discreteuniform.iccdf@memo": 9.42,
    "geometric.icdf@memo": 25.7, "geometric.iccdf@memo": 9.32,
    "hypergeometric.icdf@memo": 459, "hypergeometric.iccdf@memo": 458,
    "logseries.icdf@memo": 66, "logseries.iccdf@memo": 66.3,
    "negativebinomial.icdf@memo": 143, "negativebinomial.iccdf@memo": 144,
    "poisson.icdf@memo": 57.3, "poisson.iccdf@memo": 56.9, "poisson.ilogcdf@memo": 142, "poisson.ilogccdf@memo": 152,
    "zipf.icdf@memo": 19.4, "zipf.iccdf@memo": 33.1,
}  # fmt: skip


def measured(table: str) -> dict[str, float]:
    """key -> ns per element from the baselines' 1-thread sd timings."""
    seen: dict[str, list[float]] = {}
    for c in results.baseline_cases().values():
        cost = c.get("cost")
        if not cost or cost[0] != table or not c.get("n"):
            continue
        t = results.timing(c, "sd", 1)
        if t is None or t.get("min") is None:
            continue
        seen.setdefault(cost[1], []).append(t["min"] / c["n"] * 1e9)
    return {k: math.exp(sum(map(math.log, v)) / len(v)) for k, v in seen.items()}


# ---------------------------------------------------------------- stats

STATS_OUT = ROOT / "csrc" / "stats" / "cost_table.hpp"
# {"dist", "method", ns, memo}: memo rows are the cost of a call the per-thread memo of
# values at integers answers (scalar parameters, a narrow discrete support; cost tag
# "<dist>.<method>@memo").
_STATS_ROW = re.compile(r'\{"(\w+)", "(\w+)", ([0-9.e+-]+), (true|false)\}')


def _old_stats_rows(path: pathlib.Path) -> dict[tuple[str, str, bool], float]:
    if not path.exists():
        return {}
    return {(d, m, memo == "true"): float(ns) for d, m, ns, memo in _STATS_ROW.findall(path.read_text())}


def _stats_key(key: str) -> tuple[str, str, bool]:
    dist, method = key.split(".", 1)
    return dist, method.removesuffix("@memo"), method.endswith("@memo")


def registry_keys(table: str) -> set[str]:
    """The cost keys the registry's cases feed into `table` ("stats" or "special")."""
    from . import registry

    return {c.cost[1] for c in registry.load([table]) if c.cost and c.cost[0] == table}


def _source_line(table: str, n_measured: int, n_rows: int) -> str:
    if not results.baseline_path(table).exists():
        return "// No baseline yet; all rows kept from the previous file."
    if n_measured >= n_rows:
        return f"// All {n_rows} rows measured in bench/speed/results/{table}.json."
    return (
        f"// {n_measured} of {n_rows} rows measured in bench/speed/results/{table}.json; "
        "the rest kept from the previous file."
    )


def _old_rows(path: pathlib.Path, pattern) -> dict:
    if not path.exists():
        return {}
    return {m.groups()[:-1]: float(m.groups()[-1]) for m in pattern.finditer(path.read_text())}


def stats_header(rows: dict[tuple[str, str, bool], float], n_measured: int) -> str:
    items = sorted(rows.items())
    lines = [
        "// Generated by `python -m bench.speed cost` -- do not edit.",
        "// The measured per-element cost (ns, one thread) of each distribution method, for the",
        "// parallel split (stats/engine.hpp, cost_of; special/cost.hpp). An @memo row not",
        f"// measured yet holds its MEMO_PRESET_NS value (bench/speed/cost.py) or {MEMO_DEFAULT_NS:g} ns.",
        _source_line("stats", n_measured, len(items)),
        "#pragma once",
        "",
        "#include <string_view>",
        "",
        "namespace sd::stats::cost_table {",
        "",
        "// memo: the cost when the per-thread memo of values at integers answers the call",
        "// (scalar parameters, a narrow support); the other rows: one evaluation per element.",
        "struct Row {",
        "    std::string_view dist, method;",
        "    double ns;",
        "    bool memo;",
        "};",
        "",
        f"inline constexpr Row kRows[{len(items)}] = {{",
    ]
    lines += [f'    {{"{d}", "{m}", {ns:.3g}, {"true" if memo else "false"}}},' for (d, m, memo), ns in items]
    lines += [
        "};",
        "",
        "// The row's cost, or a negative number when the tables have no such row.",
        "constexpr double find(std::string_view dist, std::string_view method, bool memo = false) {",
        "    for (const Row& r : kRows)",
        "        if (r.dist == dist && r.method == method && r.memo == memo) return r.ns;",
        "    return -1.0;",
        "}",
        "",
        "}  // namespace sd::stats::cost_table",
        "",
    ]
    return "\n".join(lines)


def stats_rows(keys: set[str] | None = None) -> tuple[dict, int]:
    """The rows: the header's, MEMO_PRESET_NS over them, the measured ones over both, and
    MEMO_DEFAULT_NS for an @memo key of the registry (or `keys`) that still has no row."""
    rows = _old_stats_rows(STATS_OUT)
    new = measured("stats")
    for key, ns in MEMO_PRESET_NS.items():
        rows[_stats_key(key)] = ns
    for key, ns in new.items():
        rows[_stats_key(key)] = max(ns, 0.5)
    for key in registry_keys("stats") if keys is None else keys:
        k = _stats_key(key)
        if k[2] and k not in rows:
            rows[k] = MEMO_DEFAULT_NS
    return rows, len(new)


# ---------------------------------------------------------------- special

SPECIAL_OUT = ROOT / "csrc" / "special" / "cost_table.hpp"
_SPECIAL_ROW = re.compile(r'\{"(\w+)", ([0-9.e+-]+)\}')


def special_header(rows: dict[str, float], n_measured: int) -> str:
    items = sorted(rows.items())
    lines = [
        "// Generated by `python -m bench.speed cost` -- do not edit.",
        "// The measured per-element cost (ns, one thread, 10^6 contiguous elements over the function's",
        "// reference inputs) of each special function, for the parallel split (special/cost.hpp).",
        _source_line("special", n_measured, len(items)),
        "#pragma once",
        "",
        "#include <string_view>",
        "",
        "namespace sd::special::cost_table {",
        "",
        "struct Row {",
        "    std::string_view name;",
        "    double ns;",
        "};",
        "",
        f"inline constexpr Row kRows[{len(items)}] = {{",
    ]
    lines += [f'    {{"{name}", {ns:.3g}}},' for name, ns in items]
    lines += [
        "};",
        "",
        "// The row's cost, or a negative number when the table has no such row.",
        "constexpr double find(std::string_view name) {",
        "    for (const Row& r : kRows)",
        "        if (r.name == name) return r.ns;",
        "    return -1.0;",
        "}",
        "",
        "// The row's cost; a name without a row does not compile.",
        "consteval double ns(std::string_view name) {",
        "    const double v = find(name);",
        '    if (v < 0.0) throw "special/cost_table.hpp has no row for this name";',
        "    return v;",
        "}",
        "",
        "}  // namespace sd::special::cost_table",
        "",
    ]
    return "\n".join(lines)


def special_rows() -> tuple[dict, int]:
    rows = {k[0]: v for k, v in _old_rows(SPECIAL_OUT, _SPECIAL_ROW).items()}
    new = measured("special")
    rows.update(new)
    return rows, len(new)


# ---------------------------------------------------------------- both


def generate(check: bool = False, stats_keys: set[str] | None = None) -> list[tuple[pathlib.Path, bool, int]]:
    """Write (or with check=True only compare) both headers. A header counts as changed
    when its rows differ, not its comments (a comment-only difference is written but not
    reported). Returns (path, changed, rows measured). stats_keys: the stats cost keys
    (default: the registry's)."""
    out = []
    for path, pattern, (rows, n) in (
        (STATS_OUT, _STATS_ROW, stats_rows(stats_keys)),
        (SPECIAL_OUT, _SPECIAL_ROW, special_rows()),
    ):
        text = stats_header(rows, n) if path is STATS_OUT else special_header(rows, n)
        old = path.read_text() if path.exists() else ""
        changed = _rows_of(old, pattern) != _rows_of(text, pattern)
        if text != old and not check:
            path.write_text(text)
        out.append((path, changed, n))
    return out


def _rows_of(text: str, pattern) -> list[tuple]:
    return [m.groups() for m in pattern.finditer(text)]


def coverage(table: str) -> dict:
    """{"baseline": the area's baseline exists, "unfed": rows no registry case feeds,
    "missing": registry keys without a row} for "stats" or "special"."""
    keys = registry_keys(table)
    if table == "stats":
        rows = {f"{d}.{m}{'@memo' if memo else ''}" for d, m, memo in _old_stats_rows(STATS_OUT)}
    else:
        rows = {k[0] for k in _old_rows(SPECIAL_OUT, _SPECIAL_ROW)}
    return {
        "baseline": results.baseline_path(table).exists(),
        "unfed": sorted(rows - keys),
        "missing": sorted(keys - rows),
    }


def coverage_notes() -> list[str]:
    """The notes `cost --check` prints about the rows the measurement cannot refresh."""
    out = []
    for table in ("stats", "special"):
        cov = coverage(table)
        if not cov["baseline"]:
            out.append(f"{table}: no baseline yet (bench/speed/results/{table}.json); every row is kept.")
        if cov["unfed"]:
            out.append(f"{table}: {len(cov['unfed'])} rows no case feeds keep their value: {_some(cov['unfed'])}.")
        if cov["missing"]:
            out.append(f"{table}: {len(cov['missing'])} case keys have no row: {_some(cov['missing'])}.")
    return out


def _some(names: list[str], k: int = 6) -> str:
    return ", ".join(names[:k]) + (f" and {len(names) - k} more" if len(names) > k else "")
