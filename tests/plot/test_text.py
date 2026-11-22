"""The text engine on the recorded cache (docs/design/plot.md §3.2): layouts, numbers laid out
from their characters, outlines, the frozen cache. test_text_tex.py compares with TeX itself."""

import pytest

from stardust.plot import _dvi, _svgpath, _text

from . import scenes

FAMILIES = _text.FAMILIES


@pytest.mark.parametrize("family", FAMILIES)
def test_recorded_strings_lay_out_without_tex(family):
    e = _text.engine(family)
    lays = e.layout(scenes.STRINGS)
    assert e.tex_runs == 0 and e.frozen
    for s, lay in zip(scenes.STRINGS, lays):
        w, h, d = lay.box(10.0)
        assert w > 0 and h > 0 and d >= 0, s
        assert lay.glyphs, s
        for fi, _code, x, _y in lay.glyphs:
            assert 0 <= fi < len(lay.fonts) and -lay.width <= x <= 2 * lay.width


@pytest.mark.parametrize("family", FAMILIES)
def test_box_scales_with_the_size(family):
    lay = _text.engine(family).layout(["Residuals"])[0]
    w10, h10, d10 = lay.box(10.0)
    assert lay.box(7.0) == pytest.approx((0.7 * w10, 0.7 * h10, 0.7 * d10), rel=1e-15)
    # a 10 bp size is TeX's 10pt layout scaled by 72.27/72
    assert w10 == pytest.approx(lay.width / 65536 * 72 / 72.27 * 10 / _text.UNIT_BP)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_number_is_its_characters_side_by_side(family):
    e = _text.engine(family)
    one = {c: e.layout([f"${c}$"])[0] for c in "0123456789-."}
    for text in ("-273.15", "0.4217", "1000", "7", "-0.001"):
        lay = e.layout([f"${text}$"])[0]
        assert lay.width == sum(one[c].width for c in text)
        assert lay.height == max(one[c].height for c in text)
        assert [c for _, c, _, _ in lay.glyphs] == [one[c].glyphs[0][1] for c in text]
        xs = [x for _, _, x, _ in lay.glyphs]
        assert xs == sorted(xs) and len(set(xs)) == len(xs)
    assert e.tex_runs == 0  # any number, none of them recorded as a string


@pytest.mark.parametrize("family", FAMILIES)
def test_a_power_raises_the_small_exponent(family):
    e = _text.engine(family)
    base, lay = e.layout(["$10$", "$10^{-3}$"])
    small = [e.layout([rf"$\scriptstyle {c}$"])[0] for c in "-3"]
    assert lay.width == base.width + sum(s.width for s in small) + _text.SCRIPTSPACE
    assert lay.height > base.height and lay.depth == base.depth
    sizes = [lay.fonts[fi][1] for fi, _, _, _ in lay.glyphs]
    assert sizes[0] == sizes[1] > sizes[2]  # the exponent's fonts are the script size
    ys = [y for _, _, _, y in lay.glyphs]
    assert ys[2] < ys[0]  # and stand higher (y is down)


def test_unrecorded_string_is_a_frozen_miss_naming_it():
    with pytest.raises(_text.FrozenMiss, match=r"the frozen text cache does not hold 'not recorded' \(serif\)"):
        _text.engine("serif").layout(["not recorded"])


def test_an_empty_string_is_an_empty_box_without_tex():
    with _text.cache("/nonexistent/cache", frozen=True):
        e = _text.engine("serif")
        empty, blank = e.layout(["", "  "])
        assert empty.box(10.0) == (0.0, 0.0, 0.0) and not blank.glyphs and e.path(empty, 10.0) == ("", [])


def test_unknown_family():
    with pytest.raises(ValueError, match="'font' must be 'serif' or 'sans', not 'mono'"):
        _text.engine("mono")


@pytest.mark.parametrize("family", FAMILIES)
def test_path_is_closed_outlines_inside_the_box(family):
    e = _text.engine(family)
    for s in (r"$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt\pi}{2}$", "표준화 잔차", "$-1.5$"):
        lay = e.layout([s])[0]
        ops, xy = e.path(lay, 10.0, 100.0, 50.0)
        w, h, d = lay.box(10.0)
        assert ops.count("M") == ops.count("Z") > 0
        assert sum({"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}[o] for o in ops) == len(xy)
        # ink stays near the box (glyphs may overhang their boxes a little)
        assert 100 - 1 <= min(xy[0::2]) and max(xy[0::2]) <= 100 + w + 1
        assert 50 - h - 1 <= min(xy[1::2]) and max(xy[1::2]) <= 50 + d + 1


def test_hangul_takes_the_xelatex_path_and_system_fonts():
    serif, sans = (_text.engine(f).layout(["잔차"])[0] for f in FAMILIES)
    assert [k for k, _ in serif.fonts] == ["KoPubWorld Batang_Pro Light.otf#0"]
    assert [k for k, _ in sans.fonts] == ["AppleSDGothicNeo.ttc#0"]
    latin = _text.engine("serif").layout(["Residuals"])[0]
    assert "#" not in latin.fonts[0][0]  # a TFM font: the pdfLaTeX path


# ---- the readers


def test_tex_scaling_of_fix_words():
    # 1.0 as a fix_word is 2**20: the size itself; TeX truncates, it does not round
    assert _dvi.scaled(1 << 20, 655360) == 655360
    assert _dvi.scaled(1 << 19, 655360) == 327680
    assert _dvi.scaled((-(1 << 19)) & 0xFFFFFFFF, 655360) == -327680
    assert _dvi.scaled(349525, 655360) == 218453  # a third: 218453.3 truncated
    assert _dvi.scaled(1 << 20, 458752) == 458752


def test_svg_path_commands_become_absolute():
    ops, xy = _svgpath.parse("M5.5-2.25V-2.75H.5v.5h5Z")
    assert ops == "MLLLLZ" and xy == [5.5, -2.25, 5.5, -2.75, 0.5, -2.75, 0.5, -2.25, 5.5, -2.25]
    ops, xy = _svgpath.parse("m1 1l1 0 0 1c1 0 1 1 2 2s1 1 2 2q1 0 1 1t1 1z")
    assert ops == "MLLCCQQZ"
    assert xy[:6] == [1, 1, 2, 1, 2, 2]
    assert xy[12:18] == [5, 5, 5, 5, 6, 6]  # S mirrors the last control point (4, 4) about (4, 4)
    assert xy[22:26] == [7, 8, 8, 8]  # T mirrors (7, 6) about (7, 7)


def test_svg_matrices_compose():
    m = _svgpath.matrix("translate(10,20) scale(2)")
    assert _svgpath.apply(m, [1.0, 1.0]) == [12.0, 22.0]
    assert _svgpath.matrix(None) == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="unsupported transform 'rotate'"):
        _svgpath.matrix("rotate(30)")
