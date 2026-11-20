"""docs/api/random.md and the module agree: the signature block names default_rng and the
Generator's public methods and attributes (`rng.name(...)`, `rng.name`), nothing else is
public, and the signatures are the documented ones."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "random.md"


def documented():
    """({function: signature}, {method: signature}, {attribute}) from the signature blocks."""
    functions, methods, attributes = {}, {}, set()
    for block in signature_blocks(API.read_text()):
        for line in block:
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = re.fullmatch(r"(rng\.)?([A-Za-z_0-9]+)(?:\((.*)\))?", line)
            assert m, line
            on_rng, name, sig = m.groups()
            if sig is None:
                assert on_rng, line
                attributes.add(name)
            elif on_rng:
                methods[name] = sig
            else:
                functions[name] = sig
    return functions, methods, attributes


def public_members():
    """The Generator's public methods and properties."""
    methods, attributes = {}, set()
    for name, v in vars(sd.random.Generator).items():
        if name.startswith("_"):
            continue
        if isinstance(v, property):
            attributes.add(name)
        elif callable(v):
            methods[name] = v
    return methods, attributes


def test_exports_match_api_md():
    functions, methods, attributes = documented()
    # Generator is exported for isinstance checks; its methods are the rng.* lines
    assert set(sd.random.__all__) == set(functions) | {"Generator"}
    for name in sd.random.__all__:
        assert hasattr(sd.random, name)
    have_methods, have_attributes = public_members()
    assert set(have_methods) == set(methods)
    assert have_attributes == attributes


def test_signatures_match_api_md():
    functions, methods, _ = documented()
    for name, sig in functions.items():
        assert str(inspect.signature(getattr(sd.random, name)))[1:-1] == sig, name
    have, _ = public_members()
    for name, sig in methods.items():
        params = list(inspect.signature(have[name]).parameters.values())[1:]  # without self
        assert str(inspect.Signature(params))[1:-1] == sig, name
