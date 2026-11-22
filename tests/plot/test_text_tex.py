"""The text engine against TeX Live itself (deep tier; docs/design/plot.md §3.2, §8): numbers
laid out by hand are TeX's layout glyph for glyph, the two paths agree on Latin widths, the
cache saves the TeX run, errors name the string, and the recorded cache of the quick tests is
what this TeX makes."""

import pathlib

import pytest

from stardust.plot import _text

from . import scenes

pytestmark = pytest.mark.tex
FAMILIES = _text.FAMILIES


def same(a, b):
    return (a.width, a.height, a.depth, a.rules) == (b.width, b.height, b.depth, b.rules) and [
        (a.fonts[f], c, x, y) for f, c, x, y in a.glyphs
    ] == [(b.fonts[f], c, x, y) for f, c, x, y in b.glyphs]


@pytest.mark.parametrize("family", FAMILIES)
def test_numbers_by_hand_are_tex_glyph_for_glyph(family, tmp_path):
    e = _text.engine(family)
    numbers = [
        f"${n}$"
        for n in (
            "0",
            "7",
            "-1.5",
            "0.25",
            "12.75",
            "-0.001",
            "1000",
            "3.14159",
            "-273.15",
            "0.4217",
            "11.71",
            "47",
            "74",
            "-11",
        )
    ]
    powers = ["$10^{-3}$", "$10^{12}$", "$2^{-10}$", "$10^{0}$", "$-1.5^{4}$", "$7^{11}$"]
    mine = e.layout(numbers + powers)
    tex = e._pdflatex(numbers + powers, tmp_path / "direct")
    for s, a, (b, _) in zip(numbers + powers, mine, tex):
        assert same(a, b), s


@pytest.mark.parametrize("family", FAMILIES)
def test_the_two_paths_agree_on_latin_widths(family, tmp_path):
    e = _text.engine(family)
    strings = ["Residuals", "Fitted values", "Group A", "$x$", "$-1.5$", r"$p < 0.001$", r"$\hat\beta_1 = 0.4217$"]
    pdf = e._pdflatex(strings, tmp_path / "pdf")
    xe = e._xelatex(strings, tmp_path / "xe")
    for s, (a, _), (b, _) in zip(strings, pdf, xe):
        assert abs(a.width - b.width) <= 1, s  # scaled points
        assert abs(a.height - b.height) <= 0.2 * 65536 and abs(a.depth - b.depth) <= 0.2 * 65536, s


def test_cache_saves_the_run_and_survives_the_process(tmp_path):
    with _text.cache(tmp_path / "c"):
        e = _text.engine("serif")
        first = e.layout(["Residuals", "$-1.5$", "잔차"])
        runs = e.tex_runs
        assert runs == 2  # pdfLaTeX and XeLaTeX, once each (the glyph pages are not counted)
        assert e.layout(["Residuals", "$-1.5$", "잔차", "$3.25$"])[:3] == first
        assert e.tex_runs == runs
        again = _text.Engine("serif", tmp_path / "c", frozen=True)  # as a new process reads it
        got = again.layout(["Residuals", "$2.5$", "잔차"])
        assert same(got[0], first[0]) and same(got[2], first[2])
        assert again.outline(*[(first[0].fonts[f][0], c) for f, c, _, _ in first[0].glyphs][0])[0]


def test_errors_name_the_string_and_tex_reason():
    e = _text.engine("serif")
    with pytest.raises(_text.TypesetError) as err:
        e.layout(["fine", "$a^b^c$", "also fine"])
    assert err.value.string == "$a^b^c$" and err.value.reason == "! Double superscript."
    assert str(err.value) == "LaTeX could not typeset '$a^b^c$': ! Double superscript."
    with pytest.raises(_text.TypesetError, match="Undefined control sequence") as err:
        e.layout(["fine", "한글", r"한글 \nosuchmacro"])
    assert err.value.string == r"한글 \nosuchmacro"  # the XeLaTeX path names its string too
    assert "fine" not in e.strings  # nothing of a failed batch is kept
    assert e.layout(["fine"])[0].glyphs


def test_strings_that_would_break_the_batch_are_refused_before_tex():
    e = _text.engine("serif")
    for s, why in (
        (r"$\frac{1}{2$", "unbalanced braces"),
        ("a}b{", "unbalanced braces"),
        ("50% of", r"an unescaped % (write \%)"),
    ):
        with pytest.raises(_text.TypesetError) as err:
            e.layout(["fine", s])
        assert (err.value.string, err.value.reason) == (s, why)
    assert e.tex_runs == 0
    assert e.layout([r"50\% of \{a\}"])[0].glyphs


@pytest.mark.parametrize("family", FAMILIES)
def test_recorded_cache_is_what_this_tex_makes(family):
    """Fails after a TeX Live update that moves a glyph: run tools/plot_text.py --fresh, then
    record the golden outputs again."""
    fresh = _text.engine(family)
    recorded = _text.Engine(family, scenes.TEXT, frozen=True)
    strings = [s for s in recorded.strings]
    for s, a in zip(strings, fresh.layout(strings)):
        b = recorded.strings[s]
        assert same(a, b), s
        for f, c, _, _ in a.glyphs:
            assert fresh.outline(a.fonts[f][0], c) == recorded.outline(a.fonts[f][0], c), (s, a.fonts[f][0], c)


def test_missing_tool_message(monkeypatch):
    monkeypatch.setattr(_text, "_tools", {})
    monkeypatch.setattr(_text, "_TEX_DIRS", ())
    monkeypatch.setenv("PATH", str(pathlib.Path("/nonexistent")))
    with pytest.raises(
        _text.MissingTool,
        match=r"sd.plot needs 'latex', 'xelatex', 'tex', 'xetex', 'dvisvgm' and 'kpsewhich' \(TeX Live\) on PATH; "
        "'latex' was not found",
    ):
        _text.tool("latex")
