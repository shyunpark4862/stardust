"""docs/api/sparse.md and the module agree: every module-level name its signature blocks define is
planned or exported, nothing else is exported, an exported name's signature is the documented
one, an exported class has exactly the documented methods and attributes (a `sparray.` line
belongs to every format), and the documented signatures follow the area's rules (decisions
SP1-SP34 in docs/design/sparse.md §1.1).

A block line is a module name (`name(...)`, or `name` for a class used without a call), a
method (`Class.method(...)`) or an attribute (`Class.attr`)."""

import inspect
import pathlib
import re

import pytest

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "sparse.md"

# Documented, not implemented yet (docs/design/sparse.md §8). A milestone moves its names from
# here to sd.sparse.__all__.
PLANNED = {
    # SP7
    "expm", "expm_multiply", "onenormest", "norm",
}  # fmt: skip
# Options without a default (a keyword the caller must give), as sd.linalg's (SP68).
REQUIRED_OPTIONS = {("spsolve_triangular", "upper")}
CLASSES = {"sparray", "coo_array", "csr_array", "csc_array", "dok_array", "SparseLU", "LinearOperator", "CgResult",
           "BicgstabResult", "GmresResult", "MinresResult", "LsqrResult", "LsmrResult", "EigshResult", "EigsResult",
           "SvdsResult"}  # fmt: skip
FORMATS = ("coo_array", "csr_array", "csc_array", "dok_array")
LINE = re.compile(r"^(?:(?P<cls>[A-Za-z_]\w*)\.)?(?P<name>[A-Za-z_]\w*)(?:\((?P<sig>.*)\))?$")


def entries(text=None):
    """[(class or None, name, signature text or None)] of every block line, in order."""
    out = []
    for block in signature_blocks(API.read_text() if text is None else text):
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
            assert "=" in p or (name, p) in REQUIRED_OPTIONS, (cls, name, p)


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


def members(cls):
    """{name: signature text or None} the document's §2 to §5 (the public milestones SP1 to SP4)
    give class `cls`: its own lines and, for a format, the sparray lines. The methods of §6 on
    come with their milestones."""
    owners = {cls, "sparray"} if cls in FORMATS else {cls}
    public = API.read_text().split("\n## §6.")[0]
    return {name: sig for c, name, sig in entries(public) if c in owners}


def _class(cls):
    # SparseLU is not exported: splu's result is its only instance (decision SP60)
    return sd.sparse._direct.SparseLU if cls == "SparseLU" else getattr(sd.sparse, cls)


@pytest.mark.parametrize("cls", FORMATS + ("SparseLU", "LinearOperator"))
def test_exported_classes_have_the_documented_members(cls):
    doc = members(cls)
    klass = _class(cls)
    public = {n for n in dir(klass) if not n.startswith("_")}
    assert public == set(doc), (cls, public ^ set(doc))
    for name, sig in doc.items():
        attr = getattr(klass, name)
        if sig is None:
            assert not callable(attr) or isinstance(attr, property), (cls, name)
        else:
            assert callable(attr), (cls, name)
            got = str(inspect.signature(attr)).removeprefix("(self, ").removeprefix("(self")
            assert "(" + got == "(" + sig + ")", (cls, name, got)


def test_module_is_wired():
    assert set(sd.sparse.__all__) == {
        "sparray", "coo_array", "csr_array", "csc_array", "dok_array", "issparse", "matrix_transpose",
        "eye_array", "diags_array", "random_array", "kron", "kronsum", "hstack", "vstack", "block_array",
        "block_diag", "tril", "triu", "find", "spsolve", "splu", "factorized", "inv", "spsolve_triangular",
        "is_sptriangular", "spbandwidth", "LinearOperator", "aslinearoperator",
        "cg", "gmres", "minres", "bicgstab", "lsqr", "lsmr", "eigsh", "eigs", "svds",
    }  # fmt: skip
    assert not hasattr(sd.sparse, "linalg") and not hasattr(sd.sparse, "SparseLU")
    assert callable(sd.sparse._core._lu_solve) and callable(sd.sparse._core._lu_factor)
    for name in ("_cg", "_bicgstab", "_gmres", "_minres", "_lsqr", "_lsmr", "_eigsh", "_eigs", "_svds"):
        assert callable(getattr(sd.sparse._core, name))
    assert not any(hasattr(sd.sparse, f"{n}Result") for n in ("Cg", "Gmres", "Minres", "Bicgstab", "Lsqr", "Lsmr",
                                                                "Eigsh", "Eigs", "Svds"))  # fmt: skip


RESULTS = ("CgResult", "BicgstabResult", "GmresResult", "MinresResult", "LsqrResult", "LsmrResult", "EigshResult",
           "EigsResult", "SvdsResult")  # fmt: skip


@pytest.mark.parametrize("cls", RESULTS)
def test_result_classes_have_the_documented_fields(cls):
    """§6's `<Solver>Result.<field>` lines are the result's fields in order (SP72; not exported, as
    sd.htest's); the classes have no other public names but summary()."""
    from stardust.sparse import _result

    doc = [name for c, name, sig in entries() if c == cls]
    klass = getattr(_result, cls)
    assert doc and list(klass._FIELDS) == [n for n in doc if n != "summary"]
    assert {n for n in dir(klass) if not n.startswith("_")} == set(doc)
