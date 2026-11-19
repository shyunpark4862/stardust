"""The speed benchmark system (bench/speed, bench/README.md): timing, selection, results,
tables, comparison, cost headers, and the supervisor's limits and resume."""

import json
import time

import pytest

from bench.speed import compare, cost, harness, proc, registry, results, tables
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


def test_sample_merge_and_noise():
    a = harness.Sample([1.0, 1.2], 4).merged(harness.Sample([1.1], 8))
    assert a.min == 1.0 and a.median == 1.1 and a.calls == 8
    assert a.noise == pytest.approx(0.1)


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


def test_select_by_regex_glob_group_quick():
    cs = [_case("sum", quick=True, size="10^6"), _case("sort", size="3"), _case("mean", group="h")]
    assert [c.name for c in registry.select(cs, match=["^array/g/s"])] == ["sum", "sort"]
    assert [c.name for c in registry.select(cs, match=["*sort*"])] == ["sort"]
    assert [c.name for c in registry.select(cs, match=["sum [10^6]"])] == []  # a regex: [ ] is a class
    assert [c.name for c in registry.select(cs, match=[r"sum \[10\^6\]"])] == ["sum"]
    assert [c.name for c in registry.select(cs, groups=["h"])] == ["mean"]
    assert [c.name for c in registry.select(cs, quick=True)] == ["sum"]


def test_case_checks():
    with pytest.raises(ValueError):
        Case(area="nope", group="g", name="x", sd=lambda d: None)
    with pytest.raises(ValueError):
        Ref("numpy", lambda d: None, threads="some")
    assert _case("x", size="3").id == "array/g/x [3]"


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
    assert {c["id"]: c["run"] for c in base["cases"]} == {"array/g/a": "r1", "array/g/b": "r2"}
    assert results.timing(base["cases"][1], "sd", 1)["min"] == 3.0
    assert set(base["runs"]) == {"r1", "r2"}
    # a case dropped from the registry leaves the baseline
    rep = results.promote(_run_data("r3", [_rec("b", 4.0)]), known_ids=["array/g/b"])
    assert rep["array"] == (1, 1)
    base = results.load_baseline("array")
    assert [c["id"] for c in base["cases"]] == ["array/g/b"] and set(base["runs"]) == {"r3"}


def test_table_compact_and_ratios():
    data = _run_data("r1", [_rec("fast", 1e-3, 2e-4, ref=2e-3), _rec("slow", 2e-3, 1e-3, ref=1e-3)])
    cases = [_case("fast"), _case("slow")]
    md = tables.render("array", data, cases, "x.json")
    assert "| `fast` | 1.00 ms | 200 µs | 2.00 ms | 2.00 | 10.0 | 1% |" in md
    assert "**0.50**" in md  # sd slower: bold
    assert "sd가 느린 줄 1/2" in md


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
    assert "느려짐" in text and "잡음 안" in text and "빨라짐" in text


def test_cost_tables_merge_measured_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "BASELINE_DIR", tmp_path / "results")
    stats_out, special_out = tmp_path / "stats.hpp", tmp_path / "special.hpp"
    stats_out.write_text('    {"normal", "cdf", 9.0},\n    {"normal", "pdf", 3.0},\n')
    special_out.write_text('    {"erf", 5.4},\n    {"_add", 0.19},\n')
    monkeypatch.setattr(cost, "STATS_OUT", stats_out)
    monkeypatch.setattr(cost, "SPECIAL_OUT", special_out)
    run_cases = [
        _rec("Normal.cdf a", 100e-9 * 1000, n=1000, cost_key=("stats", "normal.cdf"), area="stats"),
        _rec("Normal.cdf b", 400e-9 * 1000, n=1000, cost_key=("stats", "normal.cdf"), area="stats"),
        _rec("erf", 6e-9 * 10**6, n=10**6, cost_key=("special", "erf"), area="special"),
    ]
    results.promote(_run_data("r1", run_cases), known_ids=[c["id"] for c in run_cases])
    out = cost.generate()
    assert [(p.name, changed, n) for p, changed, n in out] == [("stats.hpp", True, 1), ("special.hpp", True, 1)]
    st = stats_out.read_text()
    assert '{"normal", "cdf", 200},' in st  # geometric mean of 100 and 400
    assert '{"normal", "pdf", 3},' in st  # kept
    assert "kRows[2]" in st
    sp = special_out.read_text()
    assert '{"erf", 6},' in sp and '{"_add", 0.19},' in sp and "consteval double ns" in sp
    # nothing new measured: no change
    assert all(not changed for _, changed, _ in cost.generate(check=True))


def test_real_cost_headers_parse():
    """The checked-in headers are readable by the generator (rows are kept on regeneration)."""
    assert len(cost._old_rows(cost.STATS_OUT, cost._STATS_ROW)) > 200
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
    base = dict(case_timeout_s=60, rss_bytes=8 * 2**30, poll_s=0.2, notify_after_s=1e9, load_warn=1e9)
    base.update(kw)
    return Limits(**base)


def test_supervisor_limits_and_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.load(["array"])
    out = tmp_path / "run.json"
    data = run(
        cases,
        out=out,
        limits=_limits(case_timeout_s=8, rss_bytes=600 * 2**20),
        notify=False,
        progress=False,
        keep_awake=False,
        quiet="no",
    )
    by = {c["name"]: c for c in data["cases"]}
    assert by["sum"]["timings"][0]["min"] > 0 and by["sum"]["rounds"] == 1
    assert {(t["label"], t["threads"]) for t in by["sum"]["timings"]} == {("sd", 1), ("sd", 10), ("numpy", None)}
    assert "제한 시간" in by["sleep"]["failure"]
    assert "메모리" in by["hog"]["failure"] and by["hog"]["peak_rss_mb"] >= 500
    boom = {t["label"]: t for t in by["boom"]["timings"]}
    assert "boom" in boom["sd"]["error"] and boom["numpy"]["min"] is not None
    assert by["sum2"]["timings"][0]["min"] > 0  # the run goes on after the kills
    assert data["options"]["complete"] and json.loads(out.read_text())["run"] == data["run"]
    assert out.with_suffix(".log").exists()


def test_supervisor_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("SDBENCH_CASES_MODULE", FAKE)
    cases = registry.select(registry.load(["array"]), quick=True)
    out = tmp_path / "run.json"
    run(cases, rounds=2, out=out, limits=_limits(), notify=False, progress=False, keep_awake=False, quiet="no")
    full = results.load(out)
    sums = {c["name"]: results.timing(c, "sd", 1)["batches"] for c in full["cases"]}
    # pretend the run stopped before round 2's second case: drop that round's batches
    partial = json.loads(out.read_text())
    partial["options"].update(next=[1, 1], complete=False)
    for c in partial["cases"]:
        if c["name"] == "sum2":
            for t in c["timings"]:
                t["times"] = t["times"][: len(t["times"]) // 2]
            c["rounds"] = 1
    out.write_text(json.dumps(partial))
    data = run(
        cases, out=None, limits=_limits(), notify=False, progress=False, keep_awake=False, resume=out, quiet="no"
    )
    assert data["options"]["complete"] and data["run"] == full["run"]
    again = {c["name"]: results.timing(c, "sd", 1)["batches"] for c in data["cases"]}
    assert again["sum"] == sums["sum"]  # not re-run
    assert again["sum2"] == sums["sum2"]  # half kept + one more round
    assert {c["name"]: c["rounds"] for c in data["cases"]} == {"sum": 2, "sum2": 2}
    assert "resumed" in data["env"]


# ---------------------------------------------------------------- a quiet machine


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
    seq = iter([(3.0, ["x (pid 1, 3.0코어)"]), (2.5, ["x"]), (0.2, [])])
    monkeypatch.setattr(proc, "idle_check", lambda *a, **k: next(seq))
    monkeypatch.setattr(runmod.time, "sleep", lambda s: None)
    said = []
    lim = _limits(quiet_cores=1.0, wait_quiet_s=1e9, wait_poll_s=0)
    res = runmod.wait_for_quiet(said.append, lim, "wait")
    assert res["quiet"] and not res["stopped"] and res["at_first"] == 3.0
    assert any("경고" in s for s in said) and any("조용해졌다" in s for s in said)
    # strict: still busy after the wait -> stop
    monkeypatch.setattr(proc, "idle_check", lambda *a, **k: (2.0, ["y"]))
    res = runmod.wait_for_quiet(said.append, _limits(wait_quiet_s=0.0, wait_poll_s=0), "strict")
    assert res["stopped"] and not res["quiet"]
    res = runmod.wait_for_quiet(said.append, _limits(), "no")
    assert not res["quiet"] and not res["stopped"] and res["waited_s"] == 0


def test_busy_cases_marked_in_table_and_compare():
    rec = _rec("a", 1.2)
    rec["busy"] = True
    md = tables.render("array", _run_data("r1", [rec]), [_case("a")], "x.json")
    assert "`a` †" in md and "1코어 넘게" in md
    res = compare.compare(_run_data("r2", [rec]), {"array/g/a": _rec("a", 1.0)})
    assert res["slower"][0]["busy"] and not res["slower"][0]["beyond_noise"]
    assert "바쁠 때 잼" in compare.report(res)
