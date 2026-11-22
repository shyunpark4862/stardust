"""The speed benchmark system (bench/speed, bench/README.md): timing, selection, case ids,
results, tables, comparison, cost headers, the results check, the command line, and the
supervisor's limits, rounds and resume."""

import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from bench.speed import check, compare, cost, harness, proc, registry, results, tables
from bench.speed import run as runmod
from bench.speed.registry import Case, Ref
from bench.speed.run import Eta, Limits, run

FAKE = "tests.bench.fake_cases"


# ---------------------------------------------------------------- timing


def test_measure_batches_reach_target():
    calls = [0]

    def f():
        calls[0] += 1
        time.sleep(0.0002)

    s = harness.measure(f, target_s=0.005, batches=4, budget_s=1.0)
    assert len(s.times) == 4
    assert s.calls >= 16  # 0.2 ms calls: a 5 ms batch needs >= 16 of them (rounded up to a power of 2)
    assert all(t >= 0.0002 for t in s.times)
    assert s.min <= s.median and s.noise >= 0


def test_measure_cuts_batches_to_budget_but_keeps_min_batches():
    s = harness.measure(lambda: time.sleep(0.02), target_s=0.01, batches=7, min_batches=3, budget_s=0.05)
    assert len(s.times) == 3 and s.calls == 1


def test_measure_smoke_calls_once():
    calls = []
    s = harness.measure(lambda: calls.append(1), smoke=True)
    assert len(calls) == 1 and len(s.times) == 1


def test_measure_keeps_the_warm_up_result():
    got = []
    harness.measure(lambda: [1.0, float("nan")], target_s=0.001, batches=2, budget_s=0.01, keep=got)
    assert got == [[1.0, float("nan")]] or (len(got) == 1 and got[0][0] == 1.0)
    assert check.nan_share(got[0]) == 0.5


def test_sample_merge_and_noise():
    a = harness.Sample([1.0, 1.2], 4).merged(harness.Sample([1.1], 8))
    assert a.min == 1.0 and a.median == 1.1 and a.calls == 8
    assert a.noise == pytest.approx(0.1)
    assert a.abs_err is None
    e = harness.Sample([1.0], 1, abs_err=2e-7).merged(harness.Sample([1.0], 1, abs_err=1e-7))
    assert e.abs_err == 2e-7 and harness.Sample([1.0], 1).merged(e).abs_err == 2e-7


def test_sd_threads_restores():
    import stardust as sd

    before = sd.get_num_threads()
    with harness.sd_threads(1):
        assert sd.get_num_threads() == 1
    assert sd.get_num_threads() == before


def test_formats():
    assert harness.fmt_time(1.234e-7) == "123 ns"
    assert harness.fmt_time(0.0123) == "12.3 ms"
    assert harness.fmt_time(0.9996) == "1.00 s"
    assert harness.fmt_time(None) == "-"
    assert harness.fmt_ratio(0.678) == "0.68" and harness.fmt_ratio(123.4) == "123"


# ---------------------------------------------------------------- registry


def _case(name, group="g", quick=False, **kw):
    return Case(area="array", group=group, name=name, sd=lambda d: lambda: None, quick=quick, **kw)


def test_select_by_glob_substring_quick():
    cs = [_case("sum", quick=True, size="10^6"), _case("sort", size="3"), _case("mean", group="h")]
    assert [c.name for c in registry.select(cs, match=["array/g/s*"])] == ["sum", "sort"]
    assert [c.name for c in registry.select(cs, match=["sort"])] == ["sort"]  # plain text: a substring
    assert [c.name for c in registry.select(cs, match=["sum [10^6]"])] == ["sum"]  # [ ] are plain characters
    assert [c.name for c in registry.select(cs, match=["*sum [10^?]"])] == ["sum"]  # ? a wildcard, [ ] literal
    assert [c.name for c in registry.select(cs, match=["*sum*10^6*"])] == ["sum"]
    assert [c.name for c in registry.select(cs, match=["*/h/*"])] == ["mean"]  # a group
    assert [c.name for c in registry.select(cs, match=["sort", "mean"])] == ["sort", "mean"]
    assert [c.name for c in registry.select(cs, quick=True)] == ["sum"]


def test_case_checks():
    with pytest.raises(ValueError):
        Case(area="nope", group="g", name="x", sd=lambda d: None)
    with pytest.raises(ValueError):
        Ref("numpy", lambda d: None, threads="some")
    assert _case("x", size="3").id == "array/g/x [3]"
    c = _case("x", note="a", refs=(Ref("numpy", None), Ref("R", None, differs="rounds k")))
    assert c.notes == "a; R: rounds k"


def test_registry_areas_from_the_shared_list():
    from bench import areas

    assert registry.AREAS == areas.AREAS
    assert [a for a in areas.AREAS if a in {c.area for c in registry.load()}] == list(
        dict.fromkeys(c.area for c in registry.load())
    )


def test_registry_ids_ascii_and_groups_titled():
    cases = registry.load()
    assert not [c.id for c in cases if not c.id.isascii()]
    assert not sorted({r.label for c in cases for r in c.refs if not r.label.isascii()})
    # every group has a section title for bench/<area>/speed.md
    assert not sorted({(c.area, c.group) for c in cases if c.title == c.group})


def test_registry_sizes_follow_the_id_notation():
    """bench/README.md, case ids: 10^k / 2^k counts, (m, n) shapes, B x (m, n) batches."""
    bad = [(c.id, registry.size_problem(c.size)) for c in registry.load() if registry.size_problem(c.size)]
    assert not bad
    for size in ("10^6", "2^18 rows", "20000", "(1000, 1000)", "(1000000,)", "10^3 x (3, 3)", "n=40", "10^5 points"):
        assert registry.size_problem(size) is None, size
    for size in ("1,000", "(3,3)", "1000", "10,000 \u00d7 (50, 3)", "Osborne 2", "(10^6,)", "10^3 \u00d7 (3, 3)"):
        assert registry.size_problem(size), size
    assert registry.count(1000) == "10^3" and registry.count(500_000) == "500000" and registry.count(100) == "100"


def test_registry_ids_unique_and_areas_match():
    cases = registry.load()
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    assert all(c.id.startswith(c.area + "/") for c in cases)


# ---------------------------------------------------------------- results, tables, compare, cost


def _run_data(run_id, cases):
    return {
        "schema": results.SCHEMA,
        "run": run_id,
        "env": {
            "git": {"commit": "abc1234", "dirty": False},
            "machine": {"chip": "M", "cpus": 14},
            "versions": {"numpy": "2"},
        },
        "options": {"complete": True},
        "cases": cases,
    }


def _rec(name, sd1, sd10=None, ref=None, *, noise=0.01, n=1000, cost_key=None, group="g", area="array"):
    ts = [{"label": "sd", "lib": "sd", "threads": 1, "n": n, "min": sd1, "noise": noise}]
    if sd10 is not None:
        ts.append({"label": "sd", "lib": "sd", "threads": 10, "n": n, "min": sd10, "noise": noise})
    if ref is not None:
        ts.append({"label": "numpy", "lib": "numpy", "threads": None, "n": n, "min": ref, "noise": noise})
    return {
        "id": f"{area}/{group}/{name}",
        "area": area,
        "group": group,
        "name": name,
        "size": "",
        "n": n,
        "cost": list(cost_key) if cost_key else None,
        "note": "",
        "timings": ts,
    }


def test_promote_replaces_measured_cases_keeps_others(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path)
    results.promote(_run_data("r1", [_rec("a", 1.0), _rec("b", 2.0)]), known_ids=["array/g/a", "array/g/b"])
    results.promote(_run_data("r2", [_rec("b", 3.0)]), known_ids=["array/g/a", "array/g/b"])
    base = results.load_baseline("array")
    assert [c["id"] for c in base["cases"]] == ["array/g/a", "array/g/b"]
    assert {c["id"]: results.case_run(c) for c in base["cases"]} == {"array/g/a": "r1", "array/g/b": "r2"}
    assert results.timing(base["cases"][1], "sd", 1)["min"] == 3.0
    assert set(base["runs"]) == {"r1", "r2"}
    # a case dropped from the registry leaves the baseline
    rep = results.promote(_run_data("r3", [_rec("b", 4.0)]), known_ids=["array/g/b"])
    assert rep["array"] == (1, 1)
    base = results.load_baseline("array")
    assert [c["id"] for c in base["cases"]] == ["array/g/b"] and set(base["runs"]) == {"r3"}


def test_promote_merges_timing_by_timing(tmp_path, monkeypatch):
    """A run of one thread count, with R timed out, in fewer rounds: it changes only what it
    measured, and only when forced."""
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path)
    full = _rec("a", 1.0, 0.2, ref=2.0)
    full["rounds"] = 3
    full["timings"].append({"label": "R", "lib": "R", "threads": None, "n": 1000, "min": 5.0, "noise": 0.0})
    for t in full["timings"]:
        t["times"] = [t["min"]] * 3
    results.promote(_run_data("r3", [full]), known_ids=["array/g/a"])
    base = results.load_baseline("array")
    assert all("times" not in t for t in base["cases"][0]["timings"])  # batch times stay in the run file
    quick = _rec("a", 1.1)
    quick["rounds"] = 1
    quick["timings"].append({"label": "R", "lib": "R", "threads": None, "n": 1000, "min": None, "error": "timeout"})
    with pytest.raises(results.PromoteRefused):
        results.promote(_run_data("q1", [quick]), known_ids=["array/g/a"])
    assert results.load_baseline("array") == base  # refused: nothing written
    results.promote(_run_data("q1", [quick]), known_ids=["array/g/a"], force=True)
    c = results.load_baseline("array")["cases"][0]
    by = {(t["label"], t["threads"]): t for t in c["timings"]}
    assert by[("sd", 1)]["min"] == 1.1 and by[("sd", 1)]["run"] == "q1" and by[("sd", 1)]["rounds"] == 1
    assert by[("sd", 10)]["min"] == 0.2 and by[("sd", 10)]["run"] == "r3"  # not measured: kept
    assert by[("R", None)]["min"] == 5.0 and by[("R", None)]["run"] == "r3"  # failed: kept
    assert by[("numpy", None)]["min"] == 2.0
    assert set(results.load_baseline("array")["runs"]) == {"r3", "q1"}
    # more rounds are always fine
    again = _rec("a", 0.9)
    again["rounds"] = 3
    results.promote(_run_data("r4", [again]), known_ids=["array/g/a"])


def test_table_compact_and_ratios():
    data = _run_data("r1", [_rec("fast", 1e-3, 2e-4, ref=2e-3), _rec("slow", 2e-3, 1e-3, ref=1e-3)])
    cases = [_case("fast"), _case("slow")]
    md = tables.render("array", data, cases, "x.json")
    assert "| `fast` | 1.00 ms | 200 µs | 2.00 ms | 2.00 | 10.0 | 1% |" in md
    assert "**0.50**" in md  # sd slower: bold
    assert "sd가 느린 줄 1/2" in md


def test_table_notes_nan_shares():
    rec = _rec("f", 1e-3, ref=2e-3)
    rec["timings"][0]["nan"] = 0.0
    rec["timings"][1]["nan"] = 0.78
    rec["note"] = "R: rounds k"
    md = tables.render("array", _run_data("r1", [rec]), [_case("f")], "x.json")
    assert "R: rounds k; numpy: NaN 78%" in md
    assert check.nan_note("scipy", 0.005, 0.0) == ""  # within a point of sd's


def test_table_follow_rows_per_thread_and_scaled_sample():
    rec = _rec("fit", 1.0, 0.5)
    rec["timings"] += [
        {"label": "statsmodels", "lib": "statsmodels", "threads": 1, "n": 1000, "min": 2.0, "noise": 0},
        {"label": "statsmodels", "lib": "statsmodels", "threads": 10, "n": 1000, "min": 1.5, "noise": 0},
        {"label": "R", "lib": "R", "threads": None, "n": 100, "min": 0.3, "noise": 0},
    ]
    md = tables.render("array", _run_data("r1", [rec]), [_case("fit")], "x.json")
    assert "| `fit` | 1 | 1.00 s | 2.00 s | 2.00 | 3.00 s\\* | 3.00 | 1% |" in md
    assert "| `fit` | 10 | 500 ms | 1.50 s | 3.00 | 3.00 s\\* | 6.00 | 1% |" in md
    assert "환산" in md


def test_table_error_columns():
    """A group whose cases record abs_err gets an error column after each library's time
    (both layouts); a group without stays as it was."""
    rec = _rec("q", 1e-3, 2e-4, ref=2e-3)
    for t, e in zip(rec["timings"], (3e-7, 3e-7, 1.234e-5), strict=True):
        t["abs_err"] = e
    md = tables.render("array", _run_data("r1", [rec]), [_case("q")], "x.json")
    assert "| 사례 | sd 1스레드 | sd 10스레드 | sd 오차 | numpy | 오차 | ÷ sd 1 | ÷ sd 10 | 잡음 |" in md
    assert "| `q` | 1.00 ms | 200 µs | 3.0e-07 | 2.00 ms | 1.2e-05 | 2.00 | 10.0 | 1% |" in md
    assert "오차: 그 라이브러리의 결과와 참값의 차이" in md
    plain = tables.render("array", _run_data("r1", [_rec("p", 1e-3, ref=2e-3)]), [_case("p")], "x.json")
    assert "오차" not in plain
    fit = _rec("fit", 1.0)
    fit["timings"][0]["abs_err"] = 1e-9
    fit["timings"].append(
        {"label": "statsmodels", "lib": "statsmodels", "threads": 1, "n": 1000, "min": 2.0, "noise": 0}
    )
    md = tables.render("array", _run_data("r1", [fit]), [_case("fit")], "x.json")
    assert "| `fit` | 1 | 1.00 s | 1.0e-09 | 2.00 s | 2.00 | 1% |" in md  # statsmodels has none: no column


def test_abs_err_kept_by_state_and_baseline(tmp_path, monkeypatch):
    case = _case("a", threads=(1,), refs=(Ref("numpy", lambda d: lambda: None),))
    st = runmod.State([case], None, False)
    st.add_reply({"id": case.id, "samples": [["sd", "sd", 1, [1e-3], 1, None, 2.5e-7],
                                             ["numpy", "numpy", None, [2e-3], 1, None]], "errors": []})  # fmt: skip
    st.rounds_done[case.id] = 1
    rec = st.record(case)
    by = {t["label"]: t for t in rec["timings"]}
    assert by["sd"]["abs_err"] == 2.5e-7 and "abs_err" not in by["numpy"]
    again = runmod.State.from_file({"options": {}, "cases": [rec]}, [case]).record(case)
    assert {t["label"]: t.get("abs_err") for t in again["timings"]} == {"sd": 2.5e-7, "numpy": None}
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path)
    results.promote(_run_data("r1", [rec]), known_ids=[case.id])
    assert results.timing(results.load_baseline("array")["cases"][0], "sd", 1)["abs_err"] == 2.5e-7


def test_compare_without_a_baseline_exits_1(tmp_path, monkeypatch):
    from bench.speed import __main__ as cli_main

    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path / "none")
    run_file = tmp_path / "run.jsonl"
    rf = results.RunFile(run_file)
    rf.start("r1", {}, {"case_ids": ["array/g/a"]})
    rf.case(_rec("a", 1.0))
    rf.close()
    assert cli_main.main(["compare", str(run_file)]) == 1


def test_compare_flags_beyond_threshold_and_noise():
    base = {c["id"]: c for c in [_rec("a", 1.0, ref=1.0), _rec("b", 1.0, noise=0.3), _rec("c", 1.0)]}
    new = _run_data("r2", [_rec("a", 1.2, ref=1.0), _rec("b", 1.15, noise=0.3), _rec("c", 0.5), _rec("d", 1.0)])
    res = compare.compare(new, base, threshold=0.10)
    slow = {r["id"]: r for r in res["slower"]}
    assert set(slow) == {"array/g/a", "array/g/b"}
    assert slow["array/g/a"]["beyond_noise"] and not slow["array/g/b"]["beyond_noise"]
    assert [r["id"] for r in res["faster"]] == ["array/g/c"]
    assert res["refs_drift"] == pytest.approx(1.0) and res["missing"] == 1
    text = compare.report(res, show_faster=True)
    assert "| slower |" in text and "| within noise |" in text and "| faster |" in text


def test_cost_tables_merge_measured_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path / "results")
    stats_out, special_out = tmp_path / "stats.hpp", tmp_path / "special.hpp"
    stats_out.write_text('    {"normal", "cdf", 9.0, false},\n    {"normal", "pdf", 3.0, false},\n')
    special_out.write_text('    {"erf", 5.4},\n    {"_add", 0.19},\n')
    monkeypatch.setattr(cost, "STATS_OUT", stats_out)
    monkeypatch.setattr(cost, "SPECIAL_OUT", special_out)
    monkeypatch.setattr(cost, "MEMO_PRESET_NS", {"binomial.cdf@memo": 7.0, "poisson.icdf@memo": 50.0})
    run_cases = [
        _rec("Normal.cdf a", 100e-9 * 1000, n=1000, cost_key=("stats", "normal.cdf"), area="stats"),
        _rec("Normal.cdf b", 400e-9 * 1000, n=1000, cost_key=("stats", "normal.cdf"), area="stats"),
        _rec("Binomial.cdf narrow", 2e-9 * 1000, n=1000, cost_key=("stats", "binomial.cdf@memo"), area="stats"),
        _rec("erf", 6e-9 * 10**6, n=10**6, cost_key=("special", "erf"), area="special"),
    ]
    results.promote(_run_data("r1", run_cases), known_ids=[c["id"] for c in run_cases])
    keys = {"normal.cdf", "normal.pdf", "binomial.cdf@memo", "poisson.pmf@memo", "poisson.icdf@memo"}
    out = cost.generate(stats_keys=keys)
    assert [(p.name, changed, n) for p, changed, n in out] == [("stats.hpp", True, 2), ("special.hpp", True, 1)]
    st = stats_out.read_text()
    assert '{"normal", "cdf", 200, false},' in st  # geometric mean of 100 and 400
    assert '{"normal", "pdf", 3, false},' in st  # kept
    assert '{"binomial", "cdf", 2, true},' in st  # a memo row of its own: the measurement beats the preset
    assert '{"poisson", "icdf", 50, true},' in st  # a preset @memo row not measured yet
    assert f'{{"poisson", "pmf", {cost.MEMO_DEFAULT_NS:g}, true}},' in st  # an @memo key not measured yet
    assert "kRows[5]" in st and "bool memo = false" in st
    assert "2 of 5 rows measured in bench/speed/results/stats.json" in st
    sp = special_out.read_text()
    assert '{"erf", 6},' in sp and '{"_add", 0.19},' in sp and "consteval double ns" in sp
    assert "1 of 2 rows measured in bench/speed/results/special.json" in sp
    # nothing new measured: no change
    assert all(not changed for _, changed, _ in cost.generate(check=True, stats_keys=keys))


def test_cost_header_without_a_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path / "none")
    text = cost.special_header({"erf": 5.0, "_add": 0.19}, 0)
    assert "// No baseline yet; all rows kept from the previous file." in text
    assert "4151d2a" not in text and "commit" not in text


def test_cost_coverage_reports_rows_no_case_feeds():
    cov = cost.coverage("stats")
    assert "bernoulli.cdf" in cov["unfed"]  # Bernoulli has only narrow supports: no wide case feeds it
    assert not [k for k in cov["missing"] if k.endswith("@memo")]  # every @memo key has a row
    assert any("rows no case feeds" in n for n in cost.coverage_notes())


def test_real_cost_headers_parse():
    """The checked-in headers are readable by the generator (rows are kept on regeneration)."""
    assert len(cost._old_stats_rows(cost.STATS_OUT)) > 200
    sp = cost._old_rows(cost.SPECIAL_OUT, cost._SPECIAL_ROW)
    assert ("_add",) in sp and ("erf",) in sp


def test_eta_scales_by_progress():
    pairs = [(0, "a"), (0, "b"), (0, "c")]
    eta = Eta(pairs, {"a": 10.0, "b": 10.0, "c": None})
    assert eta.remaining(pairs) is None  # c has no estimate and nothing unknown ran yet
    eta.done("a", 20.0)  # twice as slow as the baseline
    eta.done("c", 3.0)
    assert eta.remaining([(0, "b"), (0, "c")]) == pytest.approx(20.0 + 3.0)


# ---------------------------------------------------------------- supervisor (real worker processes)


def _limits(**kw):
    """Limits for runs of the fake cases. The run's time limit is long: only the fake "sleep"
    case (which never ends, with a limit of its own) and "hog" (which grows until stopped)
    reach a limit, whatever the load of the machine."""
    base = dict(
        case_timeout_s=120, rss_bytes=8 * 2**30, poll_s=0.2, notify_after_s=1e9, load_warn=1e9, idle_window_s=0.2
    )
    base.update(kw)
    return Limits(**base)


@pytest.mark.serial
def test_supervisor_limits_and_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.load(["array"])
    out = tmp_path / "run.jsonl"
    data = run(
        cases,
        out=out,
        limits=_limits(rss_bytes=600 * 2**20),
        notify=False,
        progress=False,
        keep_awake=False,
        quiet="no",
    )
    by = {c["name"]: c for c in data["cases"]}
    assert by["sum"]["timings"][0]["min"] > 0 and by["sum"]["rounds"] == 1
    assert {(t["label"], t["threads"]) for t in by["sum"]["timings"]} == {("sd", 1), ("sd", 10), ("numpy", None)}
    assert by["sleep"]["failure"] == "timeout: over 1s"  # the case's own limit
    assert by["hog"]["failure"].startswith("memory") and by["hog"]["peak_rss_mb"] >= 500
    boom = {t["label"]: t for t in by["boom"]["timings"]}
    assert "boom" in boom["sd"]["error"] and boom["numpy"]["min"] is not None
    assert by["sum2"]["timings"][0]["min"] > 0  # the run goes on after the kills
    assert not any("check" in t for c in data["cases"] for t in c["timings"])  # only smoke runs check
    assert data["options"]["complete"] and results.load(out)["run"] == data["run"]
    assert data["failed"] == sum(1 for c in data["cases"] if runmod.case_failed(c)) >= 3
    assert os.path.exists(data["log"]) and "build/logs/speed/" in data["log"]


@pytest.mark.serial
def test_parallel_smoke_lanes_memory_retry_and_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.load(["array"])
    out = tmp_path / "run.jsonl"
    data = run(
        cases,
        out=out,
        limits=_limits(rss_bytes=600 * 2**20),
        notify=False,
        progress=False,
        keep_awake=False,
        smoke=True,
        jobs=4,
        use_r=False,
    )
    by = {c["name"]: c for c in data["cases"]}
    with open(data["log"]) as fh:
        log = fh.read()
    assert "Lanes:" in log and "4 (at most 4 processes)" in log
    # the hog takes the lanes past the shared memory limit: the stopped cases run again one at
    # a time, and the hog, stopped again on its own, keeps its memory failure
    assert "again one at a time" in log
    assert by["hog"]["failure"].startswith("memory")
    assert by["sleep"]["failure"] == "timeout: over 1s"
    assert all(c.get("failure") or any(t["min"] is not None for t in c["timings"]) for c in data["cases"])
    for name in ("sum", "sum2"):
        assert "failure" not in by[name] and by[name]["timings"][0]["min"] > 0
    assert {(t["label"], t["threads"]) for t in by["sum"]["timings"]} == {("sd", 1), ("sd", 10), ("numpy", None)}
    boom = {t["label"]: t for t in by["boom"]["timings"]}
    assert "boom" in boom["sd"]["error"] and boom["numpy"]["min"] is not None
    assert data["options"]["complete"] and data["options"]["smoke"]
    assert all(c["other_cpu"] is None for c in data["cases"])  # no busy marks in a parallel smoke run
    # the results check: the first Python comparator, never one that differs by design
    ck = {t["label"]: t.get("check") for t in by["check"]["timings"]}
    assert ck["numpy"]["ok"] and "NaN 33.3% (sd) vs 0.0%" in ck["numpy"]["text"] and ck["other"] is None
    assert by["check"]["timings"][0]["nan"] == pytest.approx(1 / 3, abs=1e-4)
    wrong = {t["label"]: t.get("check") for t in by["wrong"]["timings"]}
    assert not wrong["numpy"]["ok"] and wrong["numpy swapped"] is None
    assert by["wrong"]["note"] == "numpy swapped: takes the arguments reversed"
    assert runmod.case_failed(by["wrong"]) and not runmod.case_failed(by["check"])
    assert not any(t.get("check") for t in by["skipped"]["timings"]) and not runmod.case_failed(by["skipped"])


@pytest.mark.serial
@pytest.mark.parametrize("smoke", [True, False])
def test_abs_err_measured_for_every_library(tmp_path, monkeypatch, smoke):
    """A case with abs_err: each measurement records its warm-up result's error (a
    comparator timed on a sample against the first reference values)."""
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.select(registry.load(["array"]), match=["array/fake/err"])
    data = run(cases, out=tmp_path / "r.jsonl", limits=_limits(), notify=False, progress=False, keep_awake=False,
               smoke=smoke, use_r=False, quiet="no")  # fmt: skip
    errs = {t["label"]: t.get("abs_err") for t in data["cases"][0]["timings"]}
    assert errs == {"sd": pytest.approx(1e-12, rel=1e-3), "numpy": pytest.approx(1e-7, rel=1e-3), "sampled": 0.5}


@pytest.mark.serial
def test_worker_is_ended_after_a_stopped_request(monkeypatch):
    """A request stopped at a limit ends its worker: the reply may still come, and a next
    request must not read it as its own."""
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    with open(os.devnull, "w") as log:
        w = runmod.WorkerHandle(["array"], log, _limits())
        try:
            req = {"cmd": "case", "id": "array/fake/sleep", "threads": [1], "use_r": False}
            reply, status, _ = w.request(req, 0.5)
            assert (reply, status) == (None, "timeout")
            assert w.p.poll() is not None and w.p.pid not in runmod.WorkerHandle.live
            req = {"cmd": "case", "id": "array/fake/sum2", "threads": [1], "use_r": False}
            assert w.request(req, 30)[1] == "died"
        finally:
            w.kill()


@pytest.mark.serial
def test_stopped_env_request_does_not_shift_the_replies(tmp_path, monkeypatch):
    """The env request of a lane's new worker stopped at the shared memory limit: the
    worker is ended and the case runs again, instead of the case request reading the env
    reply (which lost the case from the run without a failure)."""
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.select(registry.load(["array"]), match=["sum", "check"])
    orig_request, orig_read = runmod.WorkerHandle.request, runmod.WorkerHandle._read
    stopped = []

    def request(self, obj, timeout_s, starting=False):
        self.last_cmd = obj["cmd"]
        return orig_request(self, obj, timeout_s, starting)

    def read(self, timeout_s, starting=False):
        if getattr(self, "last_cmd", None) == "env" and not stopped:
            stopped.append(self.p.pid)
            return None, "memory", 0  # the env reply is still on its way
        return orig_read(self, timeout_s, starting)

    monkeypatch.setattr(runmod.WorkerHandle, "request", request)
    monkeypatch.setattr(runmod.WorkerHandle, "_read", read)
    data = run(cases, out=tmp_path / "r.jsonl", limits=_limits(), notify=False, progress=False, keep_awake=False,
               smoke=True, jobs=2, use_r=False)  # fmt: skip
    assert stopped and "versions" in data["env"]  # a later worker gave the env
    for c in data["cases"]:
        assert "failure" not in c and all(t["min"] is not None for t in c["timings"]), c["name"]
    assert not runmod.WorkerHandle.live


@pytest.mark.serial
def test_a_lane_error_stops_the_run(tmp_path, monkeypatch):
    """A supervisor error in a lane is raised, not lost with the lane's cases."""
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.select(registry.load(["array"]), match=["sum", "check"])
    orig = runmod.State.add_reply

    def add_reply(self, reply):
        if reply["id"].endswith("/sum2"):
            raise RuntimeError("supervisor bug")
        orig(self, reply)

    monkeypatch.setattr(runmod.State, "add_reply", add_reply)
    with pytest.raises(RuntimeError, match="supervisor bug"):
        run(cases, out=tmp_path / "r.jsonl", limits=_limits(), notify=False, progress=False, keep_awake=False,
            smoke=True, jobs=2, use_r=False)  # fmt: skip
    assert not runmod.WorkerHandle.live


@pytest.mark.serial
def test_supervisor_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.select(registry.load(["array"]), quick=True)
    out = tmp_path / "run.jsonl"
    run(cases, rounds=2, threads=(1,), out=out, limits=_limits(), notify=False, progress=False, keep_awake=False,
        quiet="no")  # fmt: skip
    full = results.load(out)
    sums = {c["name"]: results.timing(c, "sd", 1)["batches"] for c in full["cases"]}
    # cut the file where round 2's second case would have been written (and half a line more)
    lines = out.read_text().splitlines(keepends=True)
    case_lines = [i for i, ln in enumerate(lines) if ln.startswith('{"case"')]
    assert len(case_lines) == 4
    kept = lines[: case_lines[3]]
    out.write_text("".join(kept) + lines[case_lines[3]][:40])
    part = results.load(out)
    assert not part["options"]["complete"] and part["options"]["next"] == [1, 1]
    data = run(
        cases, out=None, limits=_limits(), notify=False, progress=False, keep_awake=False, resume=out, quiet="no"
    )
    assert data["options"]["complete"] and data["run"] == full["run"]
    again = {c["name"]: results.timing(c, "sd", 1)["batches"] for c in data["cases"]}
    assert again["sum"] == sums["sum"]  # not re-run
    assert again["sum2"] > sums["sum2"] // 2  # round 1 kept + round 2 again
    assert {c["name"]: c["rounds"] for c in data["cases"]} == {"sum": 2, "sum2": 2}
    assert "resumed" in data["env"]
    assert results.load(out)["options"]["complete"]


@pytest.mark.serial
def test_rounds_run_in_new_workers(tmp_path, monkeypatch):
    """With several rounds, each round's case is measured in a worker of its own."""
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    pids = tmp_path / "pids"
    monkeypatch.setenv("SDBENCH_PID_LOG", str(pids))
    cases = registry.select(registry.load(["array"]), match=["array/fake/pid"])
    run(cases, rounds=3, out=tmp_path / "r.jsonl", limits=_limits(), notify=False, progress=False,
        keep_awake=False, quiet="no")  # fmt: skip
    seen = pids.read_text().split()
    assert len(seen) == 3 and len(set(seen)) == 3


@pytest.mark.serial
def test_smoke_failure_exits_1(tmp_path):
    env = dict(os.environ, SDBENCH_CASES_MODULE=FAKE)
    cmd = [sys.executable, "-m", "bench.speed", "run", "array", "--smoke", "--no-r", "--no-notify", "--jobs", "1"]
    for match, code in (("boom", 1), ("wrong", 1), ("sum2", 0)):
        r = subprocess.run(
            [*cmd, "--match", match, "--out", str(tmp_path / f"{match}.jsonl")],
            env=env, cwd=str(results.ROOT), capture_output=True, text=True, timeout=300,
        )  # fmt: skip
        assert r.returncode == code, (match, r.stdout[-2000:], r.stderr[-2000:])
        assert (tmp_path / f"{match}.jsonl").exists()  # the results stay


def test_rerun_busy_keeps_the_runs_options():
    from bench.speed.__main__ import rerun_options

    class A:
        rounds, rounds_given, threads, no_r = 1, None, None, False

    prev = {"rounds": 3, "threads": [1], "r": False}
    assert rerun_options(A, prev) == (3, (1,), False)
    A.rounds_given, A.rounds, A.threads = 2, 2, "1,10"
    assert rerun_options(A, prev) == (2, (1, 10), False)
    A.no_r = True
    assert rerun_options(A, {"rounds": 3, "threads": None, "r": True}) == (2, (1, 10), False)


def test_cli_conventions():
    from bench.speed import __main__ as m

    with pytest.raises(SystemExit) as e:
        m.main(["list", "nope"])  # an unknown area: a usage error
    assert e.value.code == 2
    for old in (["list", "--area", "array"], ["run", "--group", "g"], ["run", "--jobs", "11"]):
        with pytest.raises(SystemExit) as e:
            m.main(old)
        assert e.value.code == 2


def test_list_builds_nothing(monkeypatch, capsys):
    """Listing the cases needs no Boost build, no R and no subprocess."""
    from bench.speed import __main__ as m
    from bench.speed import boost

    def refuse(*a, **k):
        raise AssertionError("subprocess.run while listing")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(boost, "_lib", lambda: refuse())
    assert m.main(["list", "stats", "special", "--match", "Normal.cdf"]) == 0
    assert "Normal.cdf" in capsys.readouterr().out


def test_check_compare():
    ok = check.compare(np.array([1.0, np.nan, 3.0, np.inf]), np.array([1.0 + 1e-9, 2.0, 3.0, np.inf]))
    assert ok["ok"] and ok["agree"] == 1.0 and ok["nan_a"] == 0.25 and ok["nan_b"] == 0.0
    bad = check.compare(np.arange(10.0), np.arange(10.0)[::-1])
    assert not bad["ok"] and "first [0]" in bad["text"]
    assert not check.compare(np.zeros(3), np.zeros(4))["ok"]
    assert not check.compare(("opaque", "x"), np.zeros(3))["ok"]
    assert check.compare(np.array([1 + 1j]), np.array([1 + 1j]))["ok"]
    assert check.flat(check.normalise({"x": [1, 2]})).tolist() == [1.0, 2.0]


# ---------------------------------------------------------------- a quiet machine


@pytest.mark.serial
def test_cpu_meter_excludes_own_process():
    m = proc.CpuMeter().start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.5:
        pass  # one core busy, in this process
    other = m.stop()
    if other is None:
        pytest.skip("no CPU counters on this platform")
    # measured for another (sleeping) process tree, this process's spin counts as other
    import subprocess

    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        whole = proc.CpuMeter(own_pid=sleeper.pid).start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            pass
        with_spin = whole.stop()
    finally:
        sleeper.kill()
        sleeper.wait()
    assert with_spin >= other + 0.5 or with_spin >= 0.8  # the spin's core shows up


def test_wait_for_quiet(monkeypatch):
    seq = iter([(3.0, ["x (pid 1, 3.0 cores)"]), (2.5, ["x"]), (0.2, [])])
    monkeypatch.setattr(proc, "idle_check", lambda *a, **k: next(seq))
    monkeypatch.setattr(runmod.time, "sleep", lambda s: None)
    said = []

    class Con:
        print = staticmethod(said.append)

    lim = _limits(quiet_cores=1.0, wait_quiet_s=1e9, wait_poll_s=0)
    res = runmod.wait_for_quiet(Con, lim, "wait")
    assert res["quiet"] and not res["stopped"] and res["at_first"] == 3.0
    assert any("keep 3.0 cores busy" in s for s in said) and any(s.startswith("Quiet after") for s in said)
    # strict: still busy after the wait -> stop
    monkeypatch.setattr(proc, "idle_check", lambda *a, **k: (2.0, ["y"]))
    res = runmod.wait_for_quiet(Con, _limits(wait_quiet_s=0.0, wait_poll_s=0), "strict")
    assert res["stopped"] and not res["quiet"]
    res = runmod.wait_for_quiet(Con, _limits(), "no")
    assert not res["quiet"] and not res["stopped"] and res["waited_s"] == 0


def test_busy_cases_marked_in_table_and_compare():
    rec = _rec("a", 1.2)
    rec["busy"] = True
    md = tables.render("array", _run_data("r1", [rec]), [_case("a")], "x.json")
    assert "`a` †" in md and "1코어 넘게" in md
    res = compare.compare(_run_data("r2", [rec]), {"array/g/a": _rec("a", 1.0)})
    assert res["slower"][0]["busy"] and not res["slower"][0]["beyond_noise"]
    assert "| busy |" in compare.report(res)


def test_boost_speed_driver_matches_the_accuracy_comparator():
    """The speed driver knows the same Boost functions and distributions as the accuracy
    comparator (and as the static tables BOOST_SPECIAL and BOOST_DISTS), and returns the same
    values: the same calls in another loop, built by the same function with the same flags
    (bench.accuracy.boost.build, FLAGS: -O3 -mcpu=native -ffp-contract=off -std=c++17), so the
    same bits."""
    from bench.accuracy import boost as acc
    from bench.accuracy.compare import BOOST_DISTS, BOOST_SPECIAL
    from bench.speed import boost

    if not boost.available():
        pytest.skip("no Boost.Math checkout")
    mine = boost.names()
    assert {n for n in mine if not n.endswith(":")} == BOOST_SPECIAL
    assert {n for n in mine if n.endswith(":")} == {n for n in acc.names() if n.endswith(":")}
    assert {n for n in mine if n.endswith(":")} == {d + ":" for d in BOOST_DISTS}
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3.0, 64)
    for name, args in (("erf", (x,)), ("gammainc", (x, x[::-1].copy())), ("jv", (x, x))):
        f = boost.special(name, args)
        out = np.array(f())  # the call returns its output buffer
        assert out is not None and f.out is not None
        np.testing.assert_allclose(out, acc.evaluate(name, *args), rtol=0, atol=0)
    k = np.arange(0.0, 20.0)
    for dname, method, params, pts in (("normal", "cdf", (1.5, 2.0), x), ("binomial", "logpmf", (40.0, 0.3), k)):
        f = boost.dist(dname, method, params, pts)
        out = np.array(f())
        ref = acc.evaluate(f"{dname}:{method}", *(np.full(pts.size, p) for p in params), pts)
        np.testing.assert_allclose(out, ref, rtol=0, atol=0)


def test_boost_build_key_includes_the_source(tmp_path, monkeypatch):
    from bench.accuracy import boost as acc
    from bench.speed import boost

    monkeypatch.setattr(acc, "head", lambda h: "1" * 40)
    a, b = tmp_path / "a.cpp", tmp_path / "b.cpp"
    a.write_text("a")
    b.write_text("b")
    assert acc.build_key(tmp_path, a) != acc.build_key(tmp_path, b)  # the speed driver's own name
    assert acc.build_key(tmp_path, boost.SRC) != acc.build_key(tmp_path)  # and the comparator's
    monkeypatch.undo()
    assert acc.head(tmp_path) == ""  # not a git checkout


def test_reordered_scipy_counterparts_are_the_accuracy_measurements():
    from bench.accuracy.compare import SCIPY_REORDERED, scipy_reordered
    from bench.speed.cases import _special_common as common

    for name, (dotted, _) in SCIPY_REORDERED.items():
        assert common.counterpart(name).startswith(dotted), name  # the table's note names the same callable
    df, nc, x = np.array([3.0, 10.0]), np.array([2.0, 50.0]), np.array([4.0, 70.0])
    for name in ("chndtr", "chndtrc", "nctdtrc"):
        assert np.array_equal(common.scipy_function(name)(df, nc, x), scipy_reordered(name)(df, nc, x)), name


def test_boost_chndtr_takes_sds_argument_order():
    """sd's order (df, nc, x): Boost agrees with sd.special on the reference inputs."""
    import stardust as sd
    from bench.speed import boost
    from bench.speed.cases import _special_common as common

    if not boost.available():
        pytest.skip("no Boost.Math checkout")
    for name in ("chndtr", "chndtrc", "chndtrix"):
        xs = common.sample_inputs(name, 1000, seed=3)
        mine = np.asarray(getattr(sd.special, name)(*[sd.asarray(x) for x in xs]))
        res = check.compare(mine, np.array(boost.special(name, xs)()))
        assert res["agree"] > 0.9, (name, res["text"])


def test_rsession_one_thread_rotation_and_values():
    """R runs with its BLAS on one thread, a rotated case goes through its input sets in
    turn, and the check reads R's values (lists by component)."""
    import shutil

    from bench.speed.rsession import RSession

    if shutil.which("R") is None:
        pytest.skip("no R")
    with RSession() as r:
        info = r.info()
        assert info["threads"]["seen"] == ["1", "1"] and info["version"]
        sets = [{"x": np.array([1.0, 4.0])}, {"x": np.array([9.0, 16.0])}, {"x": np.array([25.0])}]
        assert r.value("sqrt(x)", {}, sets=sets).tolist() == [1.0, 2.0]  # the first set
        s = r.time(".sd_k", {}, sets=sets, target_s=0.0, batches=1, min_batches=1)
        assert s.nan == 0.0
        assert r.value(".sd_k", {}, sets=sets).tolist() == [1.0]  # each call moves the turn on
        got = r.value("eigen(m)", {"m": np.array([[2.0, 0.0], [0.0, 1.0]])})
        assert sorted(got) == ["values", "vectors"] and got["values"].tolist() == [2.0, 1.0]
        unnamed = r.value("list(1, 2:3)", {})
        assert sorted(unnamed) == ["1", "2"] and unnamed["2"].tolist() == [2.0, 3.0]
        assert r.value("lm(y ~ 1)", {"y": np.array([1.0, 3.0])}, value="coef(.v)").tolist() == [2.0]


def test_mvcdf_references_fit_the_case_points():
    """The stored mvn / mvt cdf references (bench/speed/cases/stats_mvcdf_ref.json) belong to
    the point sets the cases build now, and every case records the error against them."""
    from bench.speed.cases import stats as st_cases
    from bench.speed.cases import stats_mvcdf_ref as mvref

    for d, df in mvref.SETS:
        data = st_cases._mvcdf_setup(d, df)()  # raises when the file holds another point set
        assert data["truth"].shape == (len(data["xn"]),)
        assert st_cases._mvcdf_abs_err(data, data["truth"][:3]) == 0.0
    with pytest.raises(ValueError, match="no reference"):
        mvref.reference(5, math.inf, np.zeros((2, 5)), np.eye(5))
    group = [c for c in registry.load(["stats"]) if c.group == "mvn-mvt-cdf"]
    assert group and all(c.abs_err is not None and "max_points=" in c.name for c in group)
