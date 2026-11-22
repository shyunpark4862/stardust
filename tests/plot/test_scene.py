"""The scene and its SVG output (docs/design/plot.md §3.1, §3.6) on the recorded text."""

import math
import xml.etree.ElementTree as ET

import pytest

from stardust.plot import _scene as S
from stardust.plot import _svg, _text

from . import scenes

SVG = "{http://www.w3.org/2000/svg}"


@pytest.mark.parametrize("name", scenes.SCENES)
@pytest.mark.parametrize("family", _text.FAMILIES)
def test_svg_is_well_formed_and_the_same_every_time(name, family):
    a = _svg.write(scenes.SCENES[name](family))
    b = _svg.write(scenes.SCENES[name](family))
    assert a == b
    root = ET.fromstring(a)
    sc = scenes.SCENES[name](family)
    assert (
        root.get("width") == _svg.num(sc.width) + "pt"
        and root.get("viewBox") == f"0 0 {_svg.num(sc.width)} {_svg.num(sc.height)}"
    )
    ids = [e.get("id") for e in root.iter() if e.get("id")]
    assert len(ids) == len(set(ids))
    for use in root.iter(SVG + "use"):
        assert use.get("{http://www.w3.org/1999/xlink}href")[1:] in ids
    # one group per text, one path per path item
    assert len(root.findall(SVG + "g")) == len(sc.strings())
    assert len(root.findall(SVG + "path")) == sum(isinstance(i, S.Path) for i in sc.items)


def test_a_glyph_is_defined_once_and_reused():
    root = ET.fromstring(_svg.write(scenes.curve("serif")))
    defs = root.find(SVG + "defs").findall(SVG + "path")
    uses = list(root.iter(SVG + "use"))
    assert len(defs) < len(uses)  # the digits of the ticks repeat


def test_scene_dict_keeps_text_as_strings():
    d = scenes.curve("sans").to_dict()
    assert d["font"] == "sans" and d["width"] == 89 * scenes.MM
    kinds = [i["type"] for i in d["items"]]
    assert kinds.count("text") == 15 and kinds[0] == "path"
    text = next(i for i in d["items"] if i["type"] == "text")
    assert set(text) == {"type", "s", "x", "y", "size", "ha", "va", "rotation", "color"}
    path = d["items"][1]
    assert path["join"] == "round" and "fill" not in path and len(path["clip"]) == 4


def test_scene_does_not_depend_on_the_text_cache():
    # the dict is built without laying out any text
    with _text.cache("/nonexistent/cache", frozen=True):
        assert scenes.anchors("serif").to_dict()["items"]


@pytest.mark.parametrize(
    "ha,va", [(h, v) for h in ("left", "center", "right") for v in ("baseline", "bottom", "center", "top")]
)
def test_anchor_places_the_box(ha, va):
    lay = _text.engine("serif").layout([r"$y = \frac{1}{2}gt^2$"])[0]
    w, h, d = lay.box(8.0)
    g = S.place(S.Text(r"$y = \frac{1}{2}gt^2$", 100.0, 50.0, size=8, ha=ha, va=va), lay)
    a, b, c, dd, e, f = g.matrix
    assert (a, b, c, dd) == (1.0, 0.0, 0.0, 1.0)
    assert e == pytest.approx(100.0 - {"left": 0, "center": w / 2, "right": w}[ha])
    # f is the baseline's y: the box spans [f - h, f + d]
    assert f == pytest.approx(50.0 + {"baseline": 0.0, "bottom": -d, "center": (h - d) / 2, "top": h}[va])


def test_rotation_is_counter_clockwise_about_the_anchor():
    lay = _text.engine("serif").layout(["Group A"])[0]
    w, _, _ = lay.box(6.0)
    g = S.place(S.Text("Group A", 10.0, 20.0, size=6, rotation=90.0), lay)
    a, b, c, d, e, f = g.matrix
    # the baseline's right end goes straight up the page (y decreases)
    assert (a * w + e, b * w + f) == pytest.approx((10.0, 20.0 - w))
    g = S.place(S.Text("Group A", 10.0, 20.0, size=6, rotation=45.0), lay)
    a, b, c, d, e, f = g.matrix
    assert (a * w + e, b * w + f) == pytest.approx((10.0 + w * math.sqrt(0.5), 20.0 - w * math.sqrt(0.5)))


def test_bad_anchor():
    with pytest.raises(ValueError, match="bad text anchor 'middle'"):
        S.Text("x", 0, 0, ha="middle")


def test_builders():
    assert S.polyline([(0, 0), (1, 0), (1, 1)], closed=True) == ("MLLZ", [0, 0, 1, 0, 1, 1])
    assert S.rect(1, 2, 3, 4) == ("MLLLZ", [1, 2, 4, 2, 4, 6, 1, 6])
    ops, xy = S.circle(0.0, 0.0, 1.0)
    assert ops == "MCCCCZ" and len(xy) == 26
    # the cubic's midpoint is on the circle to 3e-4
    mid = 0.125 * (xy[0] + 3 * xy[2] + 3 * xy[4] + xy[6]), 0.125 * (xy[1] + 3 * xy[3] + 3 * xy[5] + xy[7])
    assert math.hypot(*mid) == pytest.approx(1.0, abs=3e-4)


def test_svg_numbers_and_paint():
    assert _svg.num(1.0) == "1" and _svg.num(-0.00001) == "0" and _svg.num(2.50004) == "2.5"
    sc = S.Scene(10, 10)
    sc.add(S.Path(*S.rect(0, 0, 5, 5), fill=(1, 0, 0, 0.5), stroke=(0, 0, 1, 1), width=0.25, dash=(1, 2), cap="round"))
    out = _svg.write(sc)
    assert (
        'fill="#ff0000" fill-opacity="0.5" stroke="#0000ff" stroke-width="0.25" stroke-linecap="round" '
        'stroke-miterlimit="10" stroke-dasharray="1 2"' in out  # corners: the PNG's and the PDF's miter limit
    )
    hole = S.Scene(10, 10)
    hole.add(S.Path(*S.rect(0, 0, 5, 5), fill=(0, 0, 0, 1), rule="evenodd"))
    assert 'fill-rule="evenodd"' in _svg.write(hole) and "fill-rule" not in out
    assert "<defs>" not in out
