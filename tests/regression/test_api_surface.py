"""docs/api/regression.md and the module agree: every name its signature blocks define is
either exported or planned for a later milestone, nothing else is exported, and the exported
functions' signatures are the documented ones."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "regression.md"

# Documented, not implemented yet (docs/design/regression.md §7): none.
PLANNED = set()
RESULTS = {
    "OLSResult",
    "WLSResult",
    "GLSResult",
    "GLMResult",
    "LinregressResult",
    "JarqueBeraResult",
    "BreuschPaganResult",
    "RidgeResult",
    "LassoResult",
    "ElasticNetResult",
    "RidgeCVResult",
    "LassoCVResult",
    "ElasticNetCVResult",
    "PenalizedGLMResult",
    "RLMResult",
    "TheilslopesResult",
    "SiegelslopesResult",
    "QuantRegResult",
    "NLSResult",
    "TTestResult",
    "WaldTestResult",
    "CompareFTestResult",
    "CompareLRTestResult",
    "AnovaResult",
    "WhiteTestResult",
    "GoldfeldQuandtResult",
    "ResetTestResult",
    "StepwiseResult",
    "BestSubsetResult",
}


def documented():
    """{name: signature text} for every `name(...)` line in the signature blocks."""
    names = {}
    for block in signature_blocks(API.read_text()):
        joined = " ".join(line.split("#")[0].strip() for line in block)  # a signature may span lines
        for m in re.finditer(r"([A-Za-z_0-9]+)\((.*?)\)(?=\s*([A-Za-z_0-9]+\(|$))", joined):
            names[m.group(1)] = m.group(2)
    return names


def test_exports_match_api_md():
    doc = set(documented())
    functions = set(sd.regression.__all__) - RESULTS
    assert functions.isdisjoint(PLANNED)
    assert functions | PLANNED == doc
    for name in sd.regression.__all__:
        assert hasattr(sd.regression, name)


def test_signatures_match_api_md():
    doc = documented()
    for name in set(sd.regression.__all__) - RESULTS:
        sig = str(inspect.signature(getattr(sd.regression, name)))[1:-1]
        assert sig == doc[name], name
