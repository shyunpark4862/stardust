"""docs/api/htest.md and the module agree: every name its signature blocks define is planned or
exported, nothing else is exported, an exported function's signature is the documented one, and
the documented signatures follow the area's rules (decisions H3, H6, H9, H10 in
docs/design/htest.md §1.1)."""

import inspect
import pathlib
import re

import stardust as sd

from .._api_doc import signature_blocks

API = pathlib.Path(__file__).parents[2] / "docs" / "api" / "htest.md"

# Documented, not implemented yet (docs/design/htest.md §8). A milestone moves its names from here to
# sd.htest.__all__; H9 was the last one with new functions.
PLANNED = set()
# Taking tables (the last axes) or summaries, not samples along an axis (H9)
NO_AXIS = {
    "ttest_ind_from_stats", "page_trend_test", "anderson_ksamp", "somersd", "chi2_contingency",
    "fisher_exact", "barnard_exact", "boschloo_exact", "mcnemar", "cochrans_q", "odds_ratio", "relative_risk",
    "binomtest", "poisson_means_test", "proportions_ztest", "proportions_chisquare", "proportion_confint",
    "test_proportions_2indep", "confint_proportions_2indep", "tukey_hsd", "dunnett", "PermutationMethod",
    "MonteCarloMethod", "BootstrapMethod", "multiscale_graphcorr", "goodness_of_fit",
}  # fmt: skip


def documented():
    """{name: signature text} for every `name(...)` line in the signature blocks."""
    names = {}
    for block in signature_blocks(API.read_text()):
        joined = " ".join(line.split("#")[0].strip() for line in block)
        for m in re.finditer(r"([A-Za-z_0-9]+)\((.*?)\)(?=\s*([A-Za-z_0-9]+\(|$))", joined):
            names[m.group(1)] = m.group(2)
    return names


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
    doc = set(documented())
    exported = set(sd.htest.__all__)
    assert exported.isdisjoint(PLANNED)
    assert exported | PLANNED == doc
    for name in exported:
        assert hasattr(sd.htest, name)
    for name in PLANNED:
        assert not hasattr(sd.htest, name), name


def test_signatures_match_api_md():
    doc = documented()
    for name in sd.htest.__all__:
        sig = str(inspect.signature(getattr(sd.htest, name)))[1:-1]
        assert sig == doc[name], name


def test_result_classes_are_not_exported():
    # H6: the result classes stay private (read through their fields); kstest is not there (H34)
    assert not [n for n in sd.htest.__all__ if n.endswith("Result")]
    assert not hasattr(sd.htest, "ConfidenceInterval") and not hasattr(sd.htest, "JarqueBeraResult")
    assert not hasattr(sd.htest, "kstest")


def test_data_positional_options_keyword():
    # H3: inputs positional-only, options keyword-only
    for name, sig in documented().items():
        if name in ("PermutationMethod", "MonteCarloMethod", "BootstrapMethod"):
            continue  # scipy's configuration objects take their settings positionally (H9)
        parts = [p.strip() for p in re.split(r",\s*(?![^()]*\))", sig)]
        if parts[0].startswith("*"):  # *samples: every later parameter is keyword-only
            continue
        assert "/" in parts, name
        slash = parts.index("/")
        # an optional input (f_exp, wilcoxon's y, spearmanr's b, somersd's y) defaults to None
        assert all(p.endswith("=None") for p in parts[:slash] if "=" in p), name
        rest = parts[slash + 1 :]
        if rest:
            assert rest[0] == "*", name


def test_axis_default_is_none():
    # H9: axis=None as sd.stats' descriptive functions; tables take no axis
    for name, sig in documented().items():
        p = params(sig)
        if name in NO_AXIS:
            assert "axis" not in p or name in ("multipletests",), name
        elif "axis" in p:
            assert p["axis"] == "None", name
    assert params(documented()["multipletests"])["axis"] == "None"
    for name in set(documented()) - NO_AXIS:
        assert "axis" in params(documented()[name]), name


def test_common_option_defaults():
    for name, sig in documented().items():
        p = params(sig)
        if "nan_policy" in p:
            assert p["nan_policy"] == "'propagate'", name
        if "alternative" in p:
            assert p["alternative"] == "'two-sided'", name
        if "keepdims" in p:
            assert p["keepdims"] == "False", name
            assert "axis" in p, name
        # the sd.stats order: axis first, then the test's options, nan_policy and keepdims last
        keys = list(p)
        if "axis" in keys and name not in ("permutation_test", "monte_carlo_test", "bootstrap", "multipletests"):
            first = keys.index("axis")
            assert all(k.startswith("*") or k in ("x", "y", "a", "b") or keys.index(k) < first for k in keys[:first])
        if "keepdims" in keys:
            assert keys[-1] == "keepdims", name
            assert keys[-2] == "nan_policy", name


def test_no_scipy_or_statsmodels_leftovers():
    # H3, H10, O14: no random_state or seed (rng only), no dof, no maxiter, no return flags, no aliases
    for name, sig in documented().items():
        p = params(sig)
        assert not {"random_state", "seed", "dof", "maxiter", "return_results", "return_object"} & set(p), name
        assert not {"is_sorted", "returnsorted", "args", "N", "bootstrap_result", "midrank"} & set(p), name
        assert "confidence_level" not in p, name  # H16: regression's alpha, not scipy's confidence_level


def test_random_tests_take_rng():
    doc = {n: params(s) for n, s in documented().items()}
    always = ("monte_carlo_test", "bootstrap", "MonteCarloMethod", "BootstrapMethod", "multiscale_graphcorr")
    for name in always + ("goodness_of_fit",):
        assert doc[name]["rng"] is None, name  # required, no default
    for name in ("permutation_test", "PermutationMethod"):
        assert doc[name]["rng"] == "None", name  # enumerated permutations need none (H9)
    assert "rng" not in doc["dunnett"]


def test_interval_level_is_alpha():
    # H16: regression's conf_int(*, alpha=0.05) (G99) wherever a level is an argument; multipletests'
    # alpha is statsmodels' (and the same meaning: a significance level)
    doc = {n: params(s) for n, s in documented().items()}
    with_alpha = {n for n, p in doc.items() if "alpha" in p}
    assert with_alpha == {"proportion_confint", "confint_proportions_2indep", "multipletests"}
    for name in with_alpha:
        assert doc[name]["alpha"] == "0.05", name


def test_auto_methods():
    # H12: the tests whose method has 'auto' (the result records the method used)
    auto = {n for n, s in documented().items() if params(s).get("method") == "'auto'"}
    assert auto == {
        "ks_1samp",
        "ks_2samp",
        "mannwhitneyu",
        "wilcoxon",
        "ansari",
        "kendalltau",
        "cramervonmises_2samp",
        "page_trend_test",
    }
