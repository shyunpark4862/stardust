"""The generators around the measurements: the special cost header's provenance line, the
accuracy tables' legend and columns, the regression sweeps' NaN count and contract, the
comparison's rules, the one ULP distance, the reference file layout and its provenance, the
double-double rounding, the Boost build key and the reference tools' command lines."""

import json
import subprocess
import sys

import numpy as np
import pytest

from bench.accuracy import __main__ as acc
from bench.accuracy import boost, common, compare, results, sources, tables
from bench.speed import cost
from tests import reffile
from tests.special import refdata


def test_special_cost_header_keeps_where_unmeasured_rows_come_from(tmp_path, monkeypatch):
    from bench.speed import results as speed_results

    rows = {"erf": 5.0, "_add": 0.19}
    monkeypatch.setattr(speed_results, "BASELINE_DIR", tmp_path)
    assert "// No baseline yet; all rows kept from the previous file." in cost.special_header(rows, 0)
    (tmp_path / "special.json").write_text("{}")
    text = cost.special_header(rows, 1)
    assert "// 1 of 2 rows measured in bench/speed/results/special.json; the rest kept from the previous file." in text
    assert "// All 2 rows measured in bench/speed/results/special.json." in cost.special_header(rows, 2)
    monkeypatch.undo()
    # the checked-in header is what the generator writes from its own rows
    rows, n = cost.special_rows()
    assert cost.special_header(rows, n) == cost.SPECIAL_OUT.read_text()


def _data(area, regions):
    env = {"date": "2026-11-20", "commit": "abc", "dirty": False}
    return {"runs": {"r": env}, "sweeps": [{"area": area, "name": "f", "regions": regions}]}


def test_table_legend_follows_its_columns():
    sp = tables.render(
        "special",
        _data(
            "special",
            {
                "guaranteed": {
                    "points": 3,
                    "ulp_max": 9e6,
                    "ulp_p999": 1.0,
                    "ulp_scale_max": 2.0,
                    "rel_max": 1e-9,
                    "contract": 0.05,
                }
            },
        ),
    )
    assert "| ULP/배율 |" in sp and "ULP/max(1, κ)" not in sp
    assert "- ULP/배율:" in sp and "Bessel·Airy" in sp and "- 계약 대비:" in sp and "- 영역:" in sp
    assert "- mixed:" not in sp and "- NaN·inf 점:" not in sp and "- 노름 상대오차:" not in sp
    reg = tables.render(
        "regression",
        _data(
            "regression",
            {
                "cooks_distance": {
                    "points": 21,
                    "ulp_max": float("inf"),
                    "ulp_p999": float("inf"),
                    "rel_max": float("inf"),
                    "norm_rel": float("nan"),
                    "nonfinite": 21,
                    "contract": None,
                }
            },
        ),
    )
    assert "| NaN·inf 점 |" in reg and reg.rstrip().endswith("| 21 |")
    assert "- NaN·inf 점:" in reg and "- ULP/배율:" not in reg and "- 영역:" not in reg and "- 계약 대비:" not in reg


def test_regression_sweep_counts_nonfinite_points(monkeypatch):
    problem = {"ref": {"a": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], "b": [[4.0, 0.0]]}}
    sw = sources.RegSweep("regression", "linear/fake", "linear", problem)
    values = {"a": np.array([1.0, np.nan, np.inf]), "b": np.array([4.0])}
    monkeypatch.setattr(sources, "_reg_fit", lambda sw, **kw: None)
    monkeypatch.setattr(sources, "_reg_value", lambda sw, r, field, cache: values[field])
    out = sources.measure(sw)["regions"]
    assert out["a"]["nonfinite"] == 2 and out["a"]["ulp_max"] == float("inf")
    assert out["b"]["nonfinite"] == 0 and out["b"]["ulp_max"] == 0


def test_reference_file_round_trip(tmp_path):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    mats = [np.eye(2), np.ones((3, 1)), np.zeros(0)]  # ragged: several shapes and an empty item
    reffile.write(
        tmp_path / "f.npz", arb="python-flint 0.9.0", target_bits=110, tool="tools/special_ref.py", n_requested=5,
        inputs={"x0": x, "m": mats + [np.full((1, 1), 7.0)]}, group=[0, 0, 1, 1],
        group_inputs={"df": np.array([2.0, 3.0])},
        methods={"f": {"over": "point", "index": np.arange(4), "hi": x * 2, "lo": np.zeros(4), "dropped": [4],
                       "guaranteed": np.ones(4, bool)},
                 "g": {"over": "group", "index": np.array([1]), "hi": [np.arange(3.0)], "lo": [np.zeros(3)]}},
    )  # fmt: skip
    r = reffile.read(tmp_path / "f.npz")
    assert r.meta == {"arb": "python-flint 0.9.0", "target_bits": 110, "tool": "tools/special_ref.py", "n_requested": 5}
    assert list(r.inputs) == ["x0", "m"] and r.points == 4 and r.ngroups == 2
    assert all(np.array_equal(a, b) and a.shape == b.shape for a, b in zip(r.inputs["m"][:3], mats))
    assert r.inputs["m"][3].shape == (1, 1) and float(r.inputs["m"][3][0, 0]) == 7.0
    f = r.methods["f"]
    assert f.over == "point" and f.index.tolist() == [0, 1, 2, 3] and f.dropped.tolist() == [4]
    assert f.fields["guaranteed"].all() and np.array_equal(f.hi, x * 2)
    assert r.methods["g"].index.tolist() == [1] and r.methods["g"].hi.tolist() == [[0.0, 1.0, 2.0]]
    assert np.array_equal(r.args("f")[0], x)
    with np.load(tmp_path / "f.npz") as z:  # every point in order: no index stored
        assert "method/f/index" not in z.files and "method/g/index" in z.files
    assert reffile.provenance(tmp_path / "f.npz")["tool"] == "tools/special_ref.py"
    with pytest.raises(ValueError):
        reffile.write(tmp_path / "bad.npz", arb="a", target_bits=1, tool="t", inputs={"x0": x},
                      methods={"f": {"over": "point", "index": [0], "hi": [1.0, 2.0], "lo": [0.0, 0.0]}})  # fmt: skip


def test_every_reference_file_has_the_layout_and_its_provenance():
    root = results.ROOT / "tests"
    files = [p for d in ("special", "stats", "array", "linalg", "optimize") for p in (root / d / "ref").rglob("*.npz")]
    assert len(files) > 200
    for p in files:
        prov = reffile.provenance(p)
        assert prov["arb"].startswith("python-flint ") and prov["target_bits"] >= 36, p
        assert prov["tool"].startswith("tools/") and prov["tool"].endswith("_ref.py"), p
    for p in [*(root / "regression" / "ref").glob("*.json"), *(root / "optimize" / "ref").glob("*.json")]:
        prov = json.loads(p.read_text())["_provenance"]
        assert prov["arb"].startswith("python-flint ") and prov["tool"].endswith("_ref.py"), p
    assert refdata.provenance(refdata.names()[0])["target_bits"] == 110


def test_ulp_distance_counts_the_doubles_between():
    nan, inf, tiny = np.nan, np.inf, 5e-324
    a = np.array([0.0, -0.0, 1.0, tiny, nan, nan, 1.0, -inf])
    b = np.array([-0.0, 0.0, np.nextafter(1.0, 2.0), -tiny, nan, 1.0, 1.0, inf])
    d = common.ulp_distance(a, b)
    assert d.dtype == np.uint64
    assert d.tolist()[:6] == [0, 0, 1, 2, 0, int(common.NAN_DISTANCE)]
    assert d[6] == 0 and d[7] == 0xFFE0000000000000
    assert common.ulp_distance(2.0, np.nextafter(2.0, 0.0)) == 1  # across a power of two: one double
    u = common.ulps(np.array([1.0, nan, inf, 2.0]), np.array([np.nextafter(1.0, 0.0), 1.0, 1.0, 2.0]))
    assert u.tolist() == [1.0, np.inf, np.inf, 0.0]


def _run(regions, **sweep):
    return {"run": "r", "env": {}, "sweeps": [{"id": "a/f", "area": "a", "name": "f", "regions": regions, **sweep}]}


def test_compare_counts_lost_regions_points_and_values_as_worse():
    base = {"a/f": {"id": "a/f", "area": "a", "regions": {
        "g": {"points": 30, "rel_max": 1e-16, "norm_rel": 1e-16, "contract": 0.1},
        "h": {"points": 3, "rel_max": 1.0}}}}  # fmt: skip

    def worse(regions):
        return [(lab, moves) for _, lab, moves, _ in acc.compare(_run(regions), base)["worse"]]

    assert worse({"g": {"points": 30, "rel_max": 1e-16, "norm_rel": 1e-16, "contract": 0.1}}) == [
        ("h", ["region gone"])
    ]
    w = worse({"g": {"points": 3, "rel_max": 1e-16, "norm_rel": None, "contract": float("nan")},
               "h": {"points": 3, "unmapped": True}})  # fmt: skip
    assert w[0][0] == "g" and any("points 30 -> 3" in m for m in w[0][1])
    assert any("norm_rel" in m and "none" in m for m in w[0][1]) and any(
        "contract" in m and "NaN" in m for m in w[0][1]
    )
    assert w[1][0] == "h" and w[1][1][0].startswith("unmapped")
    f = acc.compare(_run({"g": {"points": 31, "rel_max": 1e-17, "norm_rel": 1e-16, "contract": 0.1},
                          "h": {"points": 3, "rel_max": 1.0}}), base)  # fmt: skip
    assert f["worse"] == [] and f["better"] == 2


def test_compare_judges_every_recorded_measure_in_its_direction():
    """Every measure a result file records is compared: errors and failure counts (roots missed, short
    runs, estimates below the true error) are worse when they grow, the converged counts when they fall."""
    base = {"a/f": {"id": "a/f", "area": "a", "regions": {"g": {
        "points": 4, "missed_max": 0.0, "e2e_rel_max": 1e-16, "short_max": 0.0, "error_below_true": 0,
        "converged_max": 10.0, "entries": 40}}}}  # fmt: skip

    def moves(**r):
        g = {**base["a/f"]["regions"]["g"], **r}
        w = acc.compare(_run({"g": g}), base)["worse"]
        return [m.split()[0] for m in w[0][2]] if w else []

    assert moves() == []
    assert moves(missed_max=1.0) == ["missed_max"]
    assert moves(e2e_rel_max=1e-15, short_max=1.0, error_below_true=2) == [
        "short_max",
        "error_below_true",
        "e2e_rel_max",
    ]
    assert moves(converged_max=9.0, entries=39) == ["converged_max", "entries"]
    assert moves(converged_max=11.0, entries=41) == []
    measures = set(acc.MEASURES + acc.MEASURES_DOWN)
    for area in ("integrate", "differentiate", "interpolate"):
        for s in results.load_baseline(area)["sweeps"]:
            for r in s["regions"].values():
                assert set(r) - {"points", "worst", "bound", "unmapped", "excluded"} <= measures, s["id"]


def test_regression_linear_params_carry_the_documented_contract(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    problem = {"X": X.tolist(), "y": (X @ [1.0, 2.0]).tolist(), "weights": None, "intercept": True,
               "ref": {"params": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}}  # fmt: skip
    assert sources.design_kappa(problem) == pytest.approx(np.linalg.cond(np.c_[np.ones(20), X]))
    sw = sources.RegSweep("regression", "linear/fake", "linear", problem)
    monkeypatch.setattr(sources, "_reg_fit", lambda sw, **kw: None)
    monkeypatch.setattr(sources, "_reg_value", lambda sw, r, field, cache: np.array([0.0, 1.0, 2.0 + 4e-16]))
    e = sources.measure(sw)["regions"]["params"]
    assert e["bound"] == 1e-14 and 0.0 < e["contract"] < 0.1
    problem["X"] = (X * [1.0, 1e-13]).tolist()  # kappa beyond 1e11: no documented bound
    assert sources.measure(sw)["regions"]["params"]["contract"] is None


def test_to_dd_rounds_ties_to_even_in_the_subnormals():
    sys.path.insert(0, str(results.ROOT / "tools"))
    from arbref import precision, to_dd
    from flint import arb

    tiny = 2.0**-1074
    with precision(3000):
        for k, want in ((0.5, 0.0), (1.5, 2.0), (2.5, 2.0), (3.5, 4.0)):
            assert to_dd(arb(k) * arb(tiny))[0] == want * tiny, k
        assert to_dd(arb(1) + arb(2) ** -53) == (1.0, 2.0**-53)  # the tie at 1 + u: 1 is even


def test_boost_build_key_follows_the_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(boost, "head", lambda h: "a" * 40)
    k1 = boost.build_key(tmp_path)
    monkeypatch.setattr(boost, "head", lambda h: "b" * 40)
    assert boost.build_key(tmp_path) != k1
    assert boost.vendored()  # third_party/boost_math/README.md names its commit


def test_scipy_counterparts_take_sds_argument_order():
    import stardust as sd

    df, nc, x = np.array([3.0, 10.0]), np.array([2.0, 50.0]), np.array([4.0, 70.0])
    for name in ("chndtr", "chndtrc"):
        got = compare.scipy_reordered(name)(df, nc, x)
        want = np.asarray(getattr(sd.special, name)(sd.asarray(df), sd.asarray(nc), sd.asarray(x)))
        np.testing.assert_allclose(got, want, rtol=1e-12)
    p = np.asarray(sd.special.chndtr(sd.asarray(df), sd.asarray(nc), sd.asarray(x)))
    np.testing.assert_allclose(compare.scipy_reordered("chndtrix")(df, nc, p), x, rtol=1e-10)


def test_stats_table_splits_the_multivariate_group():
    regions = {"guaranteed": {"points": 1, "rel_max": 0.0, "contract": 0.0}}
    data = {"runs": {"r": {"date": "d", "commit": "c", "references": {"stats": "f" * 64}}},
            "sweeps": [{"area": "stats", "name": "normal_cdf", "regions": regions, "run": "r"},
                       {"area": "stats", "name": "mv/cdf.bvn", "regions": regions, "run": "r"}]}  # fmt: skip
    main, mv = tables.render("stats", data), tables.render("stats", data, "mv")
    assert "`normal_cdf`" in main and "cdf.bvn" not in main and "- 영역:" in main
    assert "`cdf.bvn`" in mv and "normal_cdf" not in mv and "참값 ffffffffffff" in mv
    assert [t[0] for t in tables.parts("stats")] == ["bench/stats/accuracy.md", "bench/stats/accuracy_multivariate.md"]


@pytest.mark.parametrize(
    "tool", ["special_ref", "stats_ref", "array_ref", "linalg_ref", "optimize_ref", "regression_ref"]
)
def test_reference_tools_reject_an_unknown_name(tool):
    r = subprocess.run([sys.executable, str(results.ROOT / "tools" / f"{tool}.py"), "--help"], capture_output=True,
                       text=True)  # fmt: skip
    assert r.returncode == 0 and "NAME" in r.stdout or "FAMILY" in r.stdout or "KIND" in r.stdout
    r = subprocess.run([sys.executable, str(results.ROOT / "tools" / f"{tool}.py"), "nosuch"], capture_output=True,
                       text=True)  # fmt: skip
    assert r.returncode == 2, r.stderr


def test_accuracy_cli_exit_codes(tmp_path):
    assert acc.main(["list", "linalg", "--match", "linalg/solve*"]) == 0
    with pytest.raises(SystemExit) as e:
        acc.main(["list", "nosuch"])
    assert e.value.code == 2
    assert acc.main(["compare", str(tmp_path / "missing.json")]) == 2


def test_stats_generator_shares_one_input_set_per_kind(monkeypatch, tmp_path):
    sys.path.insert(0, str(results.ROOT / "tools"))
    import stats_ref

    monkeypatch.setattr(stats_ref, "OUT", tmp_path)
    xs = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([-1.0, 0.5, 2.0]))
    g = np.ones(3, bool)
    assert stats_ref._save("normal", "cdf", xs, g, {0: (0.1, 0.0, 1.0), 1: None, 2: (0.9, 0.0, 1.0)}) == (2, 1)
    stats_ref._save("normal", "ccdf", xs, g, {0: (0.9, 0.0, 1.0), 1: (0.3, 0.0, 1.0), 2: (0.1, 0.0, 1.0)})
    r = reffile.read(tmp_path / "normal_x.npz")
    assert list(r.methods) == ["cdf", "ccdf"] and r.meta["n_requested"] == 3
    assert r.methods["cdf"].index.tolist() == [0, 2] and r.methods["cdf"].dropped.tolist() == [1]
    assert r.methods["ccdf"].index.tolist() == [0, 1, 2] and r.methods["ccdf"].dropped.size == 0
    again, _ = stats_ref.file_inputs("normal", "cdf")  # the dropped point is in the inputs: tried again
    assert np.array_equal(again[2], xs[2])
    other = (xs[0], xs[1], np.array([5.0, 6.0, 7.0]))  # new inputs: the methods not rebuilt drop out
    stats_ref._save("normal", "cdf", other, g, {0: (0.1, 0.0, 1.0), 1: (0.2, 0.0, 1.0), 2: (0.3, 0.0, 1.0)})
    assert list(reffile.read(tmp_path / "normal_x.npz").methods) == ["cdf"]
