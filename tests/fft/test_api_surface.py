"""docs/api/fft.md and the module agree: every name its signature blocks define is planned or exported,
nothing else is exported, an exported name's signature is the documented one, and the documented
signatures follow the area's rules (decisions FT1-FT68 in docs/design/fft.md §1.1)."""

import inspect
import pathlib
import re

import numpy
import scipy.fft

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "fft.md"

# Documented, not implemented yet: empty, every documented name is in sd.fft.__all__ (docs/design/fft.md §8).
PLANNED: set[str] = set()
LINE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\((?P<sig>.*)\)$")
# scipy.fft's public names that sd.fft leaves out on purpose (FT2, FT8, FT10)
NOT_IN_SD = {"set_workers", "get_workers", "set_backend", "skip_backend", "set_global_backend", "register_backend"}


def module_names():
    out = {}
    for block in signature_blocks(API.read_text()):
        for line in block:
            text = line.strip()
            if not text:
                continue
            m = LINE.match(text)
            assert m, f"not a signature line: {line!r}"
            assert m.group("name") not in out, m.group("name")
            out[m.group("name")] = m.group("sig")
    return out


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
    exported = set(sd.fft.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.fft, name)


def test_exported_signatures_match_api_md():
    doc = module_names()
    for name in sd.fft.__all__:
        assert str(inspect.signature(getattr(sd.fft, name)))[1:-1] == doc[name], name


def test_names_are_scipy_fft_names():
    # FT2, FT3: scipy.fft's names (numpy.fft's are among them), nothing of our own
    doc = set(module_names())
    public = {n for n in scipy.fft.__all__ if not n.startswith("_")}
    assert doc == public - NOT_IN_SD
    assert {n for n in numpy.fft.__all__ if not n.startswith("_")} <= doc


def test_data_positional_options_keyword():
    # FT6: the data (and a transform's required arguments) before `/`, every option after `*` with
    # a default; FT32: fht's dln and mu are positional
    for name, sig in module_names().items():
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig)]
        assert "/" in parts, name
        after = parts[parts.index("/") + 1 :]
        assert after and after[0] == "*", name
        for p in after[1:]:
            assert "=" in p, (name, p)
        before = [q.split("=")[0] for q in parts[: parts.index("/")]]
        for key in ("n", "axis", "s", "axes", "norm", "type", "d", "real", "orthogonalize"):
            assert key not in before or (key == "n" and name in ("fftfreq", "rfftfreq")), (name, key)


def test_no_left_out_options():
    # FT8-FT11: no workers, overwrite_x, plan, out; no device (numpy's helpers)
    for name, sig in module_names().items():
        assert not {"workers", "overwrite_x", "plan", "out", "device"} & set(params(sig)), name


def test_shared_options_have_one_spelling():
    doc = module_names()
    for name, sig in doc.items():
        p = params(sig)
        if "norm" in p:
            assert p["norm"] == "'backward'", name  # FT7
        if "axis" in p:
            assert p["axis"] == "-1", name
        if "n" in p and name not in ("fftfreq", "rfftfreq"):
            assert p["n"] == "None", name
        if "s" in p:
            assert p["s"] == "None", name
        if "axes" in p:
            want = "(-2, -1)" if name.endswith("2") else "None"
            assert p["axes"] == want, name
        if "type" in p:
            assert p["type"] == "2" and p["orthogonalize"] == "None", name
    for one, many in (("fft", "fftn"), ("dct", "dctn"), ("rfft", "rfftn"), ("hfft", "hfftn")):
        a, b = list(params(doc[one])), list(params(doc[many]))
        assert [q.replace("n", "s", 1) if q == "n" else q.replace("axis", "axes") for q in a] == b, (one, many)


def test_module_is_wired():
    assert sd.fft.__all__ == ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fft2", "ifft2", "fftn", "ifftn",
                              "rfft2", "irfft2", "rfftn", "irfftn", "hfft2", "ihfft2", "hfftn", "ihfftn", "dct", "idct",
                              "dst", "idst", "dctn", "idctn", "dstn", "idstn", "fht", "ifht", "fhtoffset", "fftfreq",
                              "rfftfreq", "fftshift", "ifftshift", "next_fast_len", "prev_fast_len"]  # fmt: skip
    assert sd.fft.__name__ == "stardust.fft"
    assert not hasattr(sd.fft, "set_workers")
