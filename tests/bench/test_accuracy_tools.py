"""The generators around the measurements: the special cost header's provenance line, the
accuracy tables' legend and columns, the regression sweeps' NaN count and the reference
files' provenance fields."""

import numpy as np

from bench.accuracy import sources, tables
from bench.speed import cost
from tests.special import refdata


def test_special_cost_header_keeps_where_unmeasured_rows_come_from():
    text = cost.special_header({"erf": 5.0, "_add": 0.19}, 1)
    assert "// 1 of 2 rows measured in the baseline; the rest kept from the previous file (first written" in text
    assert (
        "// from the hand-measured constants of special/cost.hpp, bench/special/s*/cost.md at commit 4151d2a)." in text
    )
    assert "// All 2 rows measured in the baseline." in cost.special_header({"erf": 5.0, "_add": 0.19}, 2)
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
                    "ulp_cond_max": 2.0,
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


def test_reference_provenance_reads_both_spellings(tmp_path):
    base = dict(nin=1, x0=np.zeros(1), hi=np.zeros(1), lo=np.zeros(1), guaranteed=np.ones(1, bool))
    np.savez(tmp_path / "old.npz", **base, arb="0.9.0", target_bits=110)
    np.savez(tmp_path / "older.npz", **base, arb="python-flint 0.9.0", bits=110)
    np.savez(tmp_path / "none.npz", **base)
    want = {"arb": "python-flint 0.9.0", "target_bits": 110}
    assert refdata.provenance("old", tmp_path) == want
    assert refdata.provenance("older", tmp_path) == want
    assert refdata.provenance("none", tmp_path) == {"arb": None, "target_bits": None}
    assert refdata.provenance(refdata.names()[0])["target_bits"] == 110
