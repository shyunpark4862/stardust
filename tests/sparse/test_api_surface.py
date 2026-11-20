"""docs/api/sparse.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented
one, and the documented signatures follow the area's rules (decisions SP1-SP18 in
docs/design/sparse.md §1.1).

A block line is a module name (`name(...)`, or `name` for a class used without a call), a
method (`Class.method(...)`) or an attribute (`Class.attr`)."""

import inspect
import pathlib
import re

import pytest

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "sparse.md"

# Documented, not implemented yet (docs/design/sparse.md §8): every name until SP1. A milestone
# moves its names from here to sd.sparse.__all__.
PLANNED = {
    # SP1
    "sparray", "coo_array", "csr_array", "csc_array", "dok_array", "issparse", "matrix_transpose",
    # SP3
    "eye_array", "diags_array", "random_array", "kron", "kronsum", "hstack", "vstack", "block_array", "block_diag",
    "tril", "triu", "find",
    # SP4
    "spsolve", "splu", "factorized", "inv", "spsolve_triangular", "is_sptriangular", "spbandwidth", "LinearOperator",
    "aslinearoperator",
    # SP5
    "cg", "gmres", "minres", "bicgstab", "lsqr", "lsmr",
    # SP6
    "eigsh", "eigs", "svds",
    # SP7
    "expm", "expm_multiply", "onenormest", "norm",
}  # fmt: skip
CLASSES = {"sparray", "coo_array", "csr_array", "csc_array", "dok_array", "SparseLU", "LinearOperator"}
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")


def entries():
    """[(class or None, name, signature text or None)] of every block line, in order."""
    out = []
    for block in signature_blocks(API.read_text()):
        for line in block:
            text = line.split("#")[0].strip()
            if not text:
                continue
            m = LINE.match(text)
            assert m, f"not a signature line: {line!r}"
            out.append((m.group("cls"), m.group("name"), m.group("sig")))
    return out


def module_names():
    return {name: sig for cls, name, sig in entries() if cls is None}


def params(sig):
    """{name: default text or None} of a documented signature, in order (`/` and `*` left out)."""
    out = {}
    for p in re.split(r",\s*(?![^()]*\))", sig):
        p = p.strip()
        if p in ("/", "*") or not p:
            continue
        name, _, default = p.partition("=")
        out[name.strip()] = default.strip() or None
    return out


def test_exports_match_api_md():
    doc = set(module_names())
    exported = set(sd.sparse.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.sparse, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.sparse.__all__:
        if doc[name] is not None:
            assert str(inspect.signature(getattr(sd.sparse, name)))[1:-1] == doc[name], name


def test_lines_are_unique_and_methods_belong_to_documented_classes():
    seen = [(c, n) for c, n, _ in entries()]
    assert len(seen) == len(set(seen))
    for cls, _, _ in entries():
        assert cls is None or cls in CLASSES, cls


def test_no_spmatrix_lil_bsr_dia_or_linalg_submodule():
    # SP2, SP3, SP6, SP7
    doc = module_names()
    for name in ("csr_matrix", "coo_matrix", "spmatrix", "isspmatrix", "lil_array", "bsr_array", "dia_array", "eye",
                 "identity", "diags", "spdiags", "random", "rand", "bmat", "linalg", "csgraph", "get_index_dtype",
                 "SuperLU", "ArpackNoConvergence", "MatrixRankWarning", "use_solver"):  # fmt: skip
        assert name not in doc, name
        assert not hasattr(sd.sparse, name), name


def test_no_aliases_or_canonical_form_methods():
    # §1.1 no aliases; SP10: CSR and CSC are always canonical
    methods = {(c, n) for c, n, _ in entries() if c is not None}
    for name in ("todense", "conjugate", "dot", "getrow", "getcol", "tolil", "tobsr", "todia"):
        assert not any(n == name for _, n in methods), name
    for cls in ("csr_array", "csc_array", "sparray"):
        for name in ("sort_indices", "sorted_indices", "has_sorted_indices", "check_format", "prune",
                     "has_canonical_format", "sum_duplicates"):  # fmt: skip
            assert (cls, name) not in methods, (cls, name)
    assert ("coo_array", "sum_duplicates") in methods and ("coo_array", "has_canonical_format") in methods


def test_data_positional_options_keyword():
    # the data before `/`, every option after `*` (a class's constructor included)
    for cls, name, sig in entries():
        if sig is None or not sig:
            continue
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig)]
        if "/" in parts:
            after = parts[parts.index("/") + 1 :]
            assert not after or after[0] == "*", (cls, name)
        for p in parts[parts.index("*") + 1 :] if "*" in parts else []:
            assert "=" in p, (cls, name, p)


@pytest.mark.parametrize("name", ["cg", "gmres", "minres", "bicgstab", "lsqr", "lsmr", "eigsh", "eigs", "svds",
                                  "onenormest"])  # fmt: skip
def test_iteration_caps_are_max_iter(name):
    # optimize O14: not scipy's maxiter, iter_lim, itmax
    p = params(module_names()[name])
    assert "max_iter" in p and not {"maxiter", "iter_lim", "itmax"} & set(p)


def test_no_callbacks_or_scipy_solver_options():
    for name, sig in module_names().items():
        p = params(sig) if sig else {}
        assert not {"callback", "callback_type", "show", "check", "calc_var", "options", "permc_spec",
                    "diag_pivot_thresh", "use_umfpack", "overwrite_A", "overwrite_b", "solver", "random_state",
                    "maxprint"} & set(p), name  # fmt: skip


def test_rng_is_keyword_and_defaults_to_none():
    for name in ("random_array", "eigsh", "eigs", "svds"):
        assert params(module_names()[name])["rng"] == "None"


def test_formats_have_no_dia_default():
    # SP6: format=None means CSR where scipy gives DIA or BSR
    for name in ("eye_array", "diags_array", "kron"):
        assert params(module_names()[name])["format"] == "None"


def test_module_is_wired():
    assert sd.sparse.__all__ == []
    assert not hasattr(sd.sparse, "linalg")
    assert callable(sd.sparse._core._lu_solve)
