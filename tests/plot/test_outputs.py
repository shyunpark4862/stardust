"""The PNG and PDF files of a scene (docs/design/plot.md §3.5, §3.6) and their agreement with
the SVG: the same scene is the same bytes, the files are well formed, and (deep tier, with
Ghostscript and rsvg-convert) the three outputs are the same picture."""

import math
import re
import shutil
import struct
import subprocess
import zlib

import pytest

from stardust.plot import _pdf, _png, _svg, _text

from . import scenes

CASES = [(n, f) for n in scenes.SCENES for f in _text.FAMILIES]


def decode_png(data):
    """(width, height, channels, rows of bytes) of an 8-bit PNG."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i, idat, head = 8, b"", None
    while i < len(data):
        n, tag = struct.unpack(">I4s", data[i : i + 8])
        body = data[i + 8 : i + 8 + n]
        assert zlib.crc32(tag + body) == struct.unpack(">I", data[i + 8 + n : i + 12 + n])[0]
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        i += 12 + n
    w, h, depth, kind = head[:4]
    assert depth == 8 and kind in (2, 6)
    ch = 3 if kind == 2 else 4
    raw, rows, prev = zlib.decompress(idat), [], bytearray(w * ch)
    for y in range(h):
        f, line = raw[y * (w * ch + 1)], bytearray(raw[y * (w * ch + 1) + 1 : (y + 1) * (w * ch + 1)])
        for x in range(len(line)):
            a = line[x - ch] if x >= ch else 0
            b, c = prev[x], prev[x - ch] if x >= ch else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(bytes(line))
        prev = line
    return w, h, ch, rows


@pytest.mark.parametrize("name,family", CASES)
def test_png_holds_the_pixels(name, family):
    sc = scenes.SCENES[name](family)
    w, h, px = _png.pixels(sc, 100)
    assert (w, h) == _png.size(sc, 100) == (round(sc.width * 100 / 72), round(sc.height * 100 / 72))
    data = _png.write(sc, 100)
    assert data == _png.write(sc, 100)
    dw, dh, ch, rows = decode_png(data)
    assert (dw, dh, ch) == (w, h, 4) and b"".join(rows) == px
    assert struct.pack(">II", 3937, 3937) in data  # pHYs: 100 dpi in pixels per metre
    assert _png.encode(w, h, px, level=1) != data and decode_png(_png.encode(w, h, px, level=1))[3] == rows


def test_png_of_the_curve_looks_like_a_figure():
    sc = scenes.curve("serif")
    w, h, px = _png.pixels(sc, 150)
    assert px[:4] == b"\xff\xff\xff\xff"  # the white background
    colours = {px[i : i + 3] for i in range(0, len(px), 4)}
    assert bytes([0, 114, 178]) in colours and bytes([213, 94, 0]) in colours and b"\x00\x00\x00" in colours


def test_png_draws_each_line_join_as_its_own():
    """A right angle stroked with each join: the PNG's corner holds the miter's square, the round
    join's quarter disc and the bevel's triangle (areas 1, pi/4 and 1/2 of the corner square)."""
    from stardust.plot import _scene

    def corner(join):
        sc = _scene.Scene(20.0, 20.0)
        sc.add(
            _scene.Path(
                *_scene.polyline([(4.0, 10.0), (10.0, 10.0), (10.0, 16.0)]), stroke=scenes.BLACK, width=4.0, join=join
            )
        )
        w, h, px = _png.pixels(sc, 72)
        assert (w, h) == (20, 20)
        return sum(px[4 * (y * w + x) + 3] for y in (8, 9) for x in (10, 11)) / (4 * 255)

    assert corner("miter") == pytest.approx(1.0, abs=0.01)
    assert corner("round") == pytest.approx(math.pi / 4, abs=0.03)
    assert corner("bevel") == pytest.approx(0.5, abs=0.01)


def test_png_too_large():
    with pytest.raises(ValueError, match="pixels: too large"):
        _png.pixels(scenes.anchors("serif"), 1e6)


@pytest.mark.parametrize("name,family", CASES)
def test_pdf_is_well_formed_and_the_same_every_time(name, family):
    sc = scenes.SCENES[name](family)
    data = _pdf.write(sc)
    assert data == _pdf.write(scenes.SCENES[name](family))
    assert data.startswith(b"%PDF-1.4\n") and data.endswith(b"%%EOF\n")
    xref = int(re.search(rb"startxref\n(\d+)\n", data).group(1))
    assert data[xref : xref + 5] == b"xref\n"
    n = int(re.match(rb"xref\n0 (\d+)\n", data[xref:]).group(1))
    entries = re.findall(rb"(\d{10}) 00000 n \n", data[xref:])
    assert len(entries) == n - 1
    for i, off in enumerate(entries, 1):
        assert data[int(off) :].startswith(f"{i} 0 obj\n".encode())
    box = re.search(rb"/MediaBox \[0 0 (\S+) (\S+)\]", data)
    assert float(box.group(1)) == pytest.approx(sc.width, abs=1e-4) and float(box.group(2)) == pytest.approx(
        sc.height, abs=1e-4
    )
    assert b"/Font" not in data and b"/CreationDate" not in data and b"/ID" not in data
    # one form per glyph used, each placed by a Do
    forms = set(re.findall(rb"/(G\d+) \d+ 0 R", data))
    content = zlib.decompress(re.search(rb"stream\n(.*?)\nendstream", data, re.S).group(1)).decode()
    used = set(re.findall(r"/(G\d+) Do", content))
    assert {f.decode() for f in forms} == used and len(used) > 5
    assert content.startswith(f"1 0 0 -1 0 {_pdf._n(sc.height)} cm")
    assert content.count("\nq") + content.count("q ") >= content.count("Q") > 0


def test_pdf_paint_operators():
    from stardust.plot import _scene as S

    sc = S.Scene(20, 10)
    sc.add(
        S.Path(
            *S.rect(1, 1, 5, 5),
            fill=(1, 0, 0, 0.5),
            stroke=(0, 0, 1, 1),
            width=0.5,
            dash=(2, 1),
            cap="round",
            clip=(0, 0, 4, 4),
        )
    )
    sc.add(S.Path("MQ", [0, 0, 3, 6, 6, 0], stroke=(0, 0, 0, 0.25)))
    data = _pdf.write(sc)
    content = zlib.decompress(re.search(rb"stream\n(.*?)\nendstream", data, re.S).group(1)).decode()
    assert "0 0 4 4 re W n" in content and "1 0 0 rg" in content and "0 0 1 RG 0.5 w 1 J 0 j" in content
    assert "[2 1] 0 d" in content and "\nB\n" in content and "\nS\n" in content
    assert "0 0 m\n2 4 4 4 6 0 c" in content  # the quadratic as its cubic
    assert b"/ExtGState << /A0 << /ca 0.5 >> /A1 << /CA 0.25 >> >>" in data


# ---- the three outputs are one picture (deep; needs Ghostscript and rsvg-convert)


def _external(cmd, tmp_path, name, data):
    tool = shutil.which(cmd[0])
    if tool is None:
        pytest.skip(f"{cmd[0]} is not installed")
    src = tmp_path / name
    src.write_bytes(data if isinstance(data, bytes) else data.encode())
    out = tmp_path / (name + ".png")
    subprocess.run([tool, *[a.format(src=src, out=out) for a in cmd[1:]]], check=True, capture_output=True)
    return decode_png(out.read_bytes())


def _difference(mine, other, w, h, ch):
    """(mean absolute difference, share of pixels more than 80 levels apart) on white."""
    total = far = 0
    for y in range(h):
        a, b = mine[y], other[y]
        for x in range(w):
            al = a[4 * x + 3] / 255.0
            d = 0
            for k in range(3):
                va = a[4 * x + k] * al + 255.0 * (1.0 - al)
                if ch == 4:
                    bl = b[4 * x + 3] / 255.0
                    vb = b[4 * x + k] * bl + 255.0 * (1.0 - bl)
                else:
                    vb = b[3 * x + k]
                d = max(d, abs(va - vb))
            total += d
            far += d > 80
    return total / (w * h), far / (w * h)


@pytest.mark.deep
@pytest.mark.parametrize("name,family", CASES)
def test_pdf_and_svg_render_as_the_png(name, family, tmp_path):
    """Measured 2026-11-22 at 144 dpi: against rsvg-convert (cairo) the mean difference is 0.45 to
    0.5 levels of 255; Ghostscript paints text and hairlines darker (2.4 to 2.8). The sizes may
    differ by a pixel (each program rounds the page its own way): the common part is compared."""
    sc = scenes.SCENES[name](family)
    w, h, px = _png.pixels(sc, 144)
    mine = [px[y * 4 * w : (y + 1) * 4 * w] for y in range(h)]
    rw, rh, rch, rrows = _external(
        ["rsvg-convert", "-z", "1.5", "-o", "{out}", "{src}"], tmp_path, "a.svg", _svg.write(sc)
    )
    assert abs(rw - w) <= 1 and abs(rh - h) <= 1
    mean, far = _difference(mine, rrows, min(w, rw), min(h, rh), rch)
    assert mean < 1.0 and far < 0.002, ("svg", mean, far)
    gw, gh, gch, grows = _external(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=png16m",
            "-r144",
            "-dGraphicsAlphaBits=4",
            "-dTextAlphaBits=4",
            "-sOutputFile={out}",
            "{src}",
        ],
        tmp_path,
        "a.pdf",
        _pdf.write(sc),
    )
    assert abs(gw - w) <= 1 and abs(gh - h) <= 1
    mean, far = _difference(mine, grows, min(w, gw), min(h, gh), gch)
    assert mean < 4.0 and far < 0.01, ("pdf", mean, far)
