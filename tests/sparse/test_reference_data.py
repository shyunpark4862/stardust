"""The reference and comparison data of sd.sparse (docs/design/sparse.md §7): every file belongs to
the test matrices as they are (digests), the references pass checks of their own (exact
residuals in rationals on small cases), and the recorded scipy 1.18.1 and R Matrix results agree
with the references, with the rules the design takes from them (SpMV's order, duplicates). The
recorded files are compared, not the installed libraries, so the tests do not change with them."""

import hashlib
import importlib.util
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np
import pytest

from .._bits import bits_equal
from . import matrices as M
from . import refdata

ROOT = pathlib.Path(__file__).parents[2]
SCIPY = refdata.data("scipy")
R = refdata.data("r")
POINTS = {f: refdata.points(f) for f in refdata.FAMILIES}


def by_matrix(family):
    out = {}
    for p in POINTS[family]:
        out.setdefault(p["matrix"], []).append(p)
    return out


def tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("sparse_ref", ROOT / "tools" / "sparse_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fr(v) -> Fraction:
    return Fraction(float(v))


# ---- the files belong to the matrices ------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == list(refdata.FAMILIES)
    for f in refdata.FAMILIES:
        prov = refdata.provenance(f)
        assert prov["arb"].startswith("python-flint 0.9.0"), f
        assert prov["tool"] == "tools/sparse_ref.py" and prov["target_bits"] == 110, f
        r = refdata.load(f)
        assert prov["n_requested"] == r.points, f
        for m in r.methods.values():
            assert m.dropped.size == 0, f  # every point settled when the files were made


def test_digests_match_the_matrices():
    for f, pts in POINTS.items():
        for p in pts:
            assert p["digest"] == M.digest(M.get(p["matrix"])), (f, p["matrix"])
    for k, rec in list(SCIPY.items()) + list(R.items()):
        if "digest" in rec:
            mid = k if k in M.IDS else next(i for i in M.IDS if f"/{i}/" in f"/{k}/")
            assert rec["digest"] == M.digest(M.get(mid)), k


def test_versions():
    assert refdata.data_provenance("scipy")["scipy"] == "1.18.1"
    prov = refdata.data_provenance("r")
    assert prov["R"].startswith("R version 4.6.2") and prov["Matrix"] == "1.7.6"


def test_references_are_double_doubles():
    for f, pts in POINTS.items():
        for p in pts:
            for k, v in p.items():
                if not k.endswith("_hi"):
                    continue
                hi, lo = np.asarray(v), np.asarray(p[k[:-3] + "_lo"])
                for part in (np.real, np.imag):
                    h, lw = part(hi), part(lo)
                    ok = ~np.isfinite(h) | (np.abs(lw) <= np.abs(h) * 2.0**-52)
                    assert ok.all(), (f, p["matrix"], k)


def test_the_families_cover_their_kinds():
    assert {"cancel/50", "long_rows/9000", "coo/cancel", "coo/complex", "coo/int"} <= set(by_matrix("spmv"))
    assert {"graded/400x12", "near_singular/20", "random_complex/800", "random_herm/800"} <= set(by_matrix("solve"))
    assert {p["kind"] for p in POINTS["eig"]} == {"closed", "arb", "svd"}
    assert all("2_hi" in p for p in POINTS["norms"] if min(M.get(p["matrix"]).shape) <= 220)


# ---- the references check themselves -------------------------------------------------------


def exact_matvec(a, x):
    """A x in rationals (real matrices)."""
    r, c, v = a.canonical()
    y = [Fraction(0)] * a.shape[0]
    for i, j, aij in zip(r.tolist(), c.tolist(), v.tolist()):
        y[i] += fr(aij) * fr(x[j])
    return y


def dd_of(q):
    hi = float(q)
    return hi, float(q - Fraction(hi))


@pytest.mark.parametrize("mid", ["lap1d/10", "coo/cancel", "cancel/50", "coo/dups", "coo/int", "empty/rows"])
def test_spmv_references_are_the_exact_products(mid):
    for p in by_matrix("spmv")[mid]:
        a = M.get(mid)
        y = exact_matvec(a, p["x"])
        hi = [dd_of(q)[0] for q in y]
        lo = [dd_of(q)[1] for q in y]
        assert bits_equal(np.asarray(p["matvec_hi"], dtype=float), np.array(hi, dtype=float)), mid
        assert bits_equal(np.asarray(p["matvec_lo"], dtype=float), np.array(lo, dtype=float)), mid


def test_coo_cancel_keeps_the_left_to_right_sum():
    # [1, 1e16, 1, -1e16] at (0, 1) is 0 in input order (2 exactly): the stored value is 0
    (p,) = by_matrix("sums")["coo/cancel"]
    assert p["row_sums_hi"].tolist() == [0.0, 3.0, 0.0] and p["total_hi"].tolist() == [3.0]


@pytest.mark.parametrize("mid", ["lap1d/10", "lap2d/4", "random_unsym/50x5", "graded/100x8", "saddle/40x10"])
def test_solve_references_have_tiny_exact_residuals(mid):
    (p,) = by_matrix("solve")[mid]
    a = M.get(mid)
    x = [fr(h) + fr(lw) for h, lw in zip(p["solve_hi"], p["solve_lo"])]
    r, c, v = a.canonical()
    ax = [Fraction(0)] * a.shape[0]
    for i, j, aij in zip(r.tolist(), c.tolist(), v.tolist()):
        ax[i] += fr(aij) * x[j]
    res = max(abs(fr(b) - y) for b, y in zip(p["b"], ax))
    scale = float(np.abs(a.dense()).sum(axis=1).max()) * max(abs(float(q)) for q in x)
    assert float(res) <= scale * 2.0**-100, mid


def test_closed_form_eigenvalues_agree_with_arb_on_a_small_grid():
    from flint import acb_mat

    (p,) = [q for q in POINTS["eig"] if q["matrix"] == "lap2d/4"]
    w = sorted(float(z.real.mid()) for z in acb_mat(M.get("lap2d/4").dense().tolist()).eig(multiple=True))
    assert np.allclose(p["smallest_hi"], w[:6], rtol=1e-14, atol=1e-14)
    assert np.allclose(p["largest_hi"], w[::-1][:6], rtol=1e-14, atol=1e-14)


@pytest.mark.deep
def test_the_tool_rebuilds_a_few_points():
    T = tool()
    for fam in ("sums", "spmv", "solve", "norms"):
        p = POINTS[fam][0]
        q = {k: v for k, v in p.items() if not k.endswith(("_hi", "_lo", "_cond"))}
        got = T.FAMILIES[fam][1](q)
        for name, val in got.items():
            if name in T.EXTRAS:
                continue
            assert bits_equal(
                np.asarray(val[0]).astype(np.asarray(p[f"{name}_hi"]).dtype), np.asarray(p[f"{name}_hi"])
            ), (fam, name)


# ---- the recorded scipy ----------------------------------------------------------------------


def fma_csr_matvec(a, x):
    """y_i = sum_j a_ij x_j over the row's stored entries in ascending j, one fma per term from 0:
    the summation order of docs/design/sparse.md §2.6."""
    indptr, indices, data = a.csr()
    y = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        acc = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            acc = math.fma(float(data[k]), float(x[indices[k]]), acc)
        y[i] = acc
    return y


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


@pytest.mark.parametrize(
    "mid", ["lap2d/10", "random_unsym/50x5", "cancel/50", "coo/dups", "banded/200x2x3", "long_rows/300"]
)
def test_scipy_spmv_is_the_sequential_fma_sum_in_every_format(mid):
    a = M.get(mid)
    for p in by_matrix("spmv")[mid]:
        rec = SCIPY[f"spmv/{mid}/{p['kind']}"]
        y = fma_csr_matvec(a, p["x"])
        assert rec["csr_matvec"] == rec["csc_matvec"] == rec["coo_matvec"] == sha(y), (mid, p["kind"])


def test_scipy_operations_keep_or_drop_zeros_as_the_design_records():
    k = "ops/random_unsym/50x5/"
    assert SCIPY[k + "A-A"]["nnz"] == 0 and SCIPY[k + "A*0"]["nnz"] == SCIPY[k + "A*2"]["nnz"] > 0
    assert SCIPY[k + "A/0"]["error"].startswith("ZeroDivisionError")
    assert SCIPY[k + "A+1"]["error"].startswith("NotImplementedError")
    assert SCIPY[k + "A.multiply(nan)"]["format"] == "coo"
    assert SCIPY[k + "A.multiply(nan)"]["stored_nan"] == SCIPY[k + "A*2"]["nnz"]  # implicit zeros stay 0
    assert SCIPY[k + "A.sum(axis=0)"]["type"] == "ndarray" and SCIPY[k + "A.max(axis=0)"]["format"] == "coo"
    assert SCIPY[k + "A.T"]["format"] == "csc" and SCIPY[k + "A/B"]["type"] == "ndarray"
    assert SCIPY["construct/eye_array"]["format"] == "dia" and SCIPY["construct/kron"]["format"] == "coo"
    assert SCIPY["construct/diags_array"]["format"] == "dia"


def test_scipy_duplicate_summation():
    # tocsr sums duplicates in input order for short rows; coo.sum_duplicates is the first
    # entry plus numpy's reduction of the rest (np.add.reduceat), not the input order
    short, long = SCIPY["dups/random_one_row/short"], SCIPY["dups/random_one_row/long"]
    assert short["tocsr_equals_input_order"] == short["rows"]
    assert short["coo_sum_duplicates_equals_first_plus_reduce_rest"] == short["rows"]
    assert long["coo_sum_duplicates_equals_first_plus_reduce_rest"] == long["rows"]
    assert short["coo_sum_duplicates_equals_input_order"] < short["rows"]
    assert long["tocsr_equals_input_order"] < long["rows"]  # rows past 16 entries are reordered
    assert SCIPY["dups/coo/cancel"]["value_at_0_1"] == 0.0


def test_scipy_agrees_with_the_references():
    closed = {(p["matrix"]): p for p in POINTS["eig"]}
    rec = SCIPY["eigen/eigsh/lap2d/30/SA"]["eigenvalues"]
    assert np.allclose(rec, closed["lap2d/30"]["smallest_hi"], rtol=1e-12)
    rec = SCIPY["eigen/eigsh/lap2d/30/LA"]["eigenvalues"]
    assert np.allclose(sorted(rec, reverse=True), closed["lap2d/30"]["largest_hi"], rtol=1e-12)
    sv = closed["rect/60x20"]["svdvals_hi"]
    assert np.allclose(SCIPY["eigen/svds/rect/60x20/LM"]["singular_values"], sv[:6], rtol=1e-12)
    norms = {p["matrix"]: p for p in POINTS["norms"]}
    for mid in ("lap2d/30", "convdiff2d/40", "random_unsym/1000x8"):
        m = SCIPY[f"matfuncs/{mid}"]
        for o, key in (("fro", "fro"), ("1", "1"), ("-1", "-1"), ("inf", "inf"), ("-inf", "-inf")):
            assert math.isclose(m[f"norm_{o}"], norms[mid][f"{key}_hi"][0], rel_tol=1e-13), (mid, o)
    for k, m in SCIPY.items():
        if k.startswith("matfuncs/expm/"):
            assert m["max_rel_error"] < 1e-13, k
        if k.startswith("direct/") and k.endswith("/spsolve") and "backward" in m:
            assert m["backward"] < 1e-15, k


def test_scipy_singular_solves_are_recorded():
    assert SCIPY["direct/struct_singular/40/splu"]["error"] == "RuntimeError: Factor is exactly singular"
    rec = SCIPY["direct/struct_singular/40/spsolve"]
    assert rec["warnings"] == ["MatrixRankWarning"] and not rec["finite"]
    # exactly singular in value: no warning, a finite solution of enormous size
    rec = SCIPY["direct/dupcol/40/spsolve"]
    assert rec["finite"] and "warnings" not in rec and rec["max_abs_x"] > 1e15


# ---- the recorded R -----------------------------------------------------------------------


def test_r_agrees_with_the_references():
    conds = {p["matrix"]: p["solve_cond"] for p in POINTS["solve"]}
    for mid, rec in R.items():
        for k, v in rec.items():
            if not k.startswith("max_rel_"):
                continue
            name = k[len("max_rel_") :]
            bound = 1e-13
            if name.startswith(("solve", "lstsq")):
                bound = max(1e-13, 1e-15 * conds.get(mid, 1e3))
            if name.startswith("expm"):
                bound = 1e-13
            assert v <= bound, (mid, k, v)
    for mid in ("struct_singular/40", "dupcol/40", "lap2d/10n"):
        assert "error_solve_lu" in R[mid], mid
