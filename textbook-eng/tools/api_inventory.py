"""Dump the public API of the shared stardust build as JSON.

Run with the shared build's interpreter (see BUILD.md), from any directory outside the
working repository, e.g.
    .sdbuild/stardust/.venv/bin/python -I tools/api_inventory.py > /path/api.json
For every namespace: __all__ (or the public names), and for each class its public members
(methods, properties, class attributes) defined on it or inherited from stardust classes.
"""

import importlib
import inspect
import json
import sys

import stardust as sd

AREAS = [
    "", "linalg", "frame", "special", "stats", "random", "regression", "optimize", "htest",
    "sparse", "integrate", "differentiate", "spatial", "graph", "interpolate", "fft", "smooth",
    "plot", "cluster", "embed", "classify",
]


def kind_of(obj):
    if inspect.isclass(obj):
        return "class"
    if inspect.ismodule(obj):
        return "module"
    if callable(obj):
        return "function"
    return "value"


def members(cls):
    out = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        # skip members that come only from builtins (object, tuple, Exception, ...)
        owner = None
        for base in cls.__mro__:
            if name in base.__dict__:
                owner = base
                break
        if owner is None:
            continue
        mod = getattr(owner, "__module__", "") or ""
        if not (mod.startswith("stardust") or mod == "" or owner is cls):
            if owner.__module__ in ("builtins",) and owner is not cls:
                continue
        try:
            attr = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        if isinstance(attr, property) or type(attr).__name__ in ("getset_descriptor", "nb_property", "property"):
            k = "property"
        elif isinstance(attr, (staticmethod, classmethod)):
            k = "method"
        elif callable(attr) or type(attr).__name__ in ("nb_method", "method_descriptor", "nb_func"):
            k = "method"
        else:
            k = "attribute"
        out[name] = k
    return out


result = {"stardust_file": sd.__file__, "python": sys.version, "areas": {}}
for area in AREAS:
    modname = "stardust" + ("." + area if area else "")
    try:
        mod = importlib.import_module(modname)
    except ImportError as exc:
        result["areas"][area or "sd"] = {"missing": str(exc)}
        continue
    has_all = hasattr(mod, "__all__")
    names = list(mod.__all__) if has_all else [n for n in dir(mod) if not n.startswith("_")]
    entries = {}
    for n in names:
        obj = getattr(mod, n, None)
        k = kind_of(obj)
        e = {"kind": k}
        if k == "class":
            e["bases"] = [b.__name__ for b in obj.__mro__[1:-1]]
            e["members"] = members(obj)
        entries[n] = e
    result["areas"][area or "sd"] = {"has_all": has_all, "names": entries}

json.dump(result, sys.stdout, indent=1, ensure_ascii=False)
