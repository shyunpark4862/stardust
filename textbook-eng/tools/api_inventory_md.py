"""Print the per-namespace public API of the shared stardust build as Markdown.

Run with the shared build's interpreter (BUILD.md) from outside the working repository:
    .sdbuild/stardust/.venv/bin/python -I tools/api_inventory_md.py > /tmp/inventory.md
Groups: the top level by the comments of stardust/__init__.py's __all__; linalg and special by
hand-written groups below; the Python areas by the private module that defines each name.
Classes list their public members (methods, `name*` = property or field). Result classes that
functions return but that are not in __all__ are listed after each area.
"""

import importlib
import inspect
import pkgutil
import re
import sys

import stardust as sd

AREAS = ["linalg", "frame", "special", "stats", "random", "regression", "optimize", "htest", "sparse",
         "integrate", "differentiate", "spatial", "graph", "interpolate", "fft", "smooth", "plot"]

HAND = {
    "linalg": [
        ("products and norms", "vecdot outer cross kron kronsum trace diagonal vector_norm matrix_norm matrix_power block_diag bandwidth"),
        ("LU and determinants", "lu lu_solve solve det slogdet inv"),
        ("structured systems", "solve_triangular solve_banded solve_tridiagonal"),
        ("symmetric factorizations", "cholesky cholesky_solve ldl"),
        ("QR, least squares, rank", "qr qr_pivoted lstsq pinv matrix_rank cond"),
        ("eigenvalues and SVD", "eigh eigvalsh eig eigvals schur hessenberg svd svdvals"),
        ("matrix functions and equations", "expm solve_sylvester"),
        ("exceptions", "LinAlgError"),
    ],
    "special": [
        ("logistic and information", "expit logit log_expit xlogy xlog1py entr rel_entr logsumexp softmax log_softmax"),
        ("error functions", "erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri ndtri_exp wofz"),
        ("gamma, beta, zeta", "gamma gammaln rgamma loggamma digamma polygamma beta betaln binom zeta"),
        ("incomplete gamma and beta", "gammainc gammaincc gammaincinv gammainccinv betainc betaincc betaincinv betainccinv"),
        ("exponential integrals, elliptic", "expi expn ellipk ellipe"),
        ("Bessel (integer order)", "j0 j1 y0 y1 i0 i0e i1 i1e k0 k0e k1 k1e"),
        ("Bessel (real order) and Airy", "jv yv iv kv jve yve ive kve airy"),
        ("hypergeometric, Lambert W", "hyp1f1 hyp2f1 lambertw"),
        ("distribution functions", "stdtr stdtrit chdtr chdtrc chdtri fdtr fdtrc fdtri pdtr pdtrc bdtr bdtrc nbdtr nbdtrc"),
        ("noncentral distribution functions", "chndtr chndtrc chndtrix nctdtr nctdtrc nctdtrit ncfdtr ncfdtrc ncfdtri"),
    ],
}


def members(cls):
    out = []
    for name in sorted(dir(cls)):
        if name.startswith("_") or name in ("count", "index") and issubclass(cls, tuple):
            continue
        owner = next((b for b in cls.__mro__ if name in b.__dict__), None)
        if owner is None or owner.__module__ == "builtins":
            continue
        attr = inspect.getattr_static(cls, name)
        is_method = callable(attr) and not isinstance(attr, property) and type(attr).__name__ not in (
            "getset_descriptor", "nb_property", "member_descriptor", "_tuplegetter")
        out.append(name if is_method else name + "*")
    fields = getattr(cls, "_fields", None)
    if isinstance(fields, tuple):
        out = [f + "*" for f in fields] + [m for m in out if m.rstrip("*") not in fields]
    return out


def fmt_class(name, cls):
    mem = members(cls)
    bases = [b.__name__ for b in cls.__mro__[1:-1] if not b.__name__.startswith("_")]
    head = f"`{name}`" + (f" ({', '.join(bases)})" if bases else "")
    if not mem:
        doc = (cls.__doc__ or "").strip().split("\n")[0]
        return f"- {head}: (공개 멤버 없음{'; ' + doc[:160] if doc else ''})"
    line = f"- {head} [{len(mem)}]: " + " ".join(mem)
    dynamic = any("__getattr__" in b.__dict__ and b.__module__.startswith("stardust") for b in cls.__mro__)
    if dynamic:  # fields are looked up at run time: quote the docstring's field list
        doc = " ".join((cls.__doc__ or "").split())
        line += f" — 동적 필드(docstring): {doc[:300]}"
    return line


def top_level_groups():
    src = inspect.getsource(sd)
    body = src[src.index("__all__ = ["):]
    body = body[: body.index("]\n")]
    groups, cur = [], None
    for line in body.splitlines()[1:]:
        line = line.strip()
        if line.startswith("#"):
            cur = (line[1:].strip(), [])
            groups.append(cur)
        elif line.startswith('"'):
            cur[1].append(line.strip('",'))
    return groups


def area_section(area, mod):
    names = list(mod.__all__)
    lines = [f"### `sd.{area}` — {len(names)}개", ""]
    classes = [n for n in names if inspect.isclass(getattr(mod, n))]
    funcs = [n for n in names if n not in classes]
    if area in HAND:
        for title, ns in HAND[area]:
            ns = [n for n in ns.split() if n in names]
            lines.append(f"- {title} ({len(ns)}): " + " ".join(f"`{n}`" for n in ns))
        listed = {n for _, ns in HAND[area] for n in ns.split()}
        rest = [n for n in names if n not in listed]
        if rest:
            lines.append("- (분류 밖): " + " ".join(rest))
        classes = [c for c in classes if c != "LinAlgError"]
    else:
        groups = {}
        values = [n for n in funcs if not callable(getattr(mod, n))]
        if values:
            lines.append("- 값 (" + str(len(values)) + "): " + " ".join(f"`{n}`" for n in values))
        funcs = [n for n in funcs if n not in values]
        for n in funcs:
            m = getattr(getattr(mod, n), "__module__", "") or ""
            groups.setdefault(m.split(".")[-1].lstrip("_") or area, []).append(n)
        for g, ns in groups.items():
            lines.append(f"- 함수 · {g} ({len(ns)}): " + " ".join(f"`{n}`" for n in ns))
    if classes:
        lines += ["", f"클래스 {len(classes)}개 (`이름*` = 속성·필드, 나머지는 메서드):", ""]
        lines += [fmt_class(n, getattr(mod, n)) for n in classes]
    return lines


def hidden_results(area_mods):
    exported = {id(getattr(m, n)) for m in area_mods.values() for n in m.__all__}
    exported |= {id(getattr(sd, n)) for n in sd.__all__}
    found = {}
    for mi in pkgutil.walk_packages(sd.__path__, "stardust."):
        if mi.name.startswith("stardust.plot._"):
            continue
        m = importlib.import_module(mi.name)
        for n, o in vars(m).items():
            if (inspect.isclass(o) and not n.startswith("_") and re.search(r"(Result|SparseLU|GroupBy|NAType)$", n)
                    and o.__module__.startswith("stardust") and id(o) not in exported):
                found[n] = o
    return found


# which area returns each hidden result class (by name; checked by hand against docs/api)
OWNER = {
    "sd": "UniqueAllResult UniqueCountsResult UniqueInverseResult HistogramResult FinfoResult IinfoResult",
    "linalg": "LUResult LDLResult QRResult QRPivotedResult LstsqResult EighResult EigResult SchurResult HessenbergResult SVDResult SlogdetResult BandwidthResult",
    "frame": "GroupBy NAType",
    "special": "AiryResult",
    "stats": "CdfResult",
    "sparse": "SparseLU CgResult GmresResult MinresResult BicgstabResult LsqrResult LsmrResult EigshResult EigsResult SvdsResult OnenormestResult",
    "spatial": "QueryResult RadiusResult",
    "graph": "ConnectedComponentsResult TraversalResult ShortestPathResult MinOnlyResult YenResult MaximumFlowResult AssignmentResult LaplacianResult BiconnectedComponentsResult",
    "smooth": "SmootherResult",
}


def main():
    mods = {a: importlib.import_module("stardust." + a) for a in AREAS}
    hidden = hidden_results(mods)
    owner = {n: a for a, ns in OWNER.items() for n in ns.split()}
    for n, o in hidden.items():
        if n not in owner:
            owner[n] = o.__module__.split(".")[1]
    out = [f"### `sd` (최상위, 배열) — {len(sd.__all__)}개", ""]
    for title, ns in top_level_groups():
        out.append(f"- {title} ({len(ns)}): " + " ".join(f"`{n}`" for n in ns))
    out += ["", "클래스:", "", fmt_class("ndarray", sd.ndarray)]
    extra = sorted(n for n, a in owner.items() if a == "sd" and n in hidden)
    if extra:
        out += ["", "`__all__` 밖의 결과 클래스 (함수가 돌려준다):", ""] + [fmt_class(n, hidden[n]) for n in extra]
    for a in AREAS:
        out += [""] + area_section(a, mods[a])
        extra = sorted(n for n, ar in owner.items() if ar == a and n in hidden)
        if extra:
            out += ["", "`__all__` 밖의 결과 클래스 (함수가 돌려준다):", ""] + [fmt_class(n, hidden[n]) for n in extra]
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
