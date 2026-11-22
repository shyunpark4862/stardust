"""The 2D rasterizer (csrc/plot/raster.hpp; docs/design/plot.md §3.5): exact coverage of fills,
strokes that keep one width, joins, caps, dashes, clips, and the same pixels at any thread count."""

from array import array

import pytest

import stardust as sd
from stardust._core import plot as core

ID = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
RED = (1.0, 0.0, 0.0, 1.0)
BUTT, ROUND, SQUARE = 0, 1, 2
MITER, RJOIN = 0, 1


def pack(*xy):
    return array("d", xy).tobytes()


def rect(x0, y0, x1, y1):
    return b"MLLLZ", pack(x0, y0, x1, y0, x1, y1, x0, y1)


def alpha(c):
    px = c.rgba8()
    return [[px[4 * (y * c.width + x) + 3] for x in range(c.width)] for y in range(c.height)]


def line(c, pts, width, cap=BUTT, join=MITER, dash=(), rgba=RED, clip=()):
    c.stroke(b"M" + b"L" * (len(pts) // 2 - 1), pack(*pts), ID, rgba, clip, width, cap, join, list(dash))


def test_fill_coverage_is_the_exact_area():
    c = core.Canvas(6, 4)
    c.fill(*rect(1.25, 1.0, 3.5, 2.5), ID, RED, ())
    a = alpha(c)
    assert a[0] == [0] * 6 and a[3] == [0] * 6
    assert a[1] == [0, 191, 255, 128, 0, 0]  # 0.75, 1, 0.5 of a pixel
    assert a[2] == [0, 96, 128, 64, 0, 0]  # the half row: 0.375, 0.5, 0.25
    px = c.rgba8()
    assert px[4 * (1 * 6 + 2) : 4 * (1 * 6 + 2) + 4] == bytes([255, 0, 0, 255])
    assert px[4 * (1 * 6 + 1) : 4 * (1 * 6 + 1) + 3] == bytes([255, 0, 0])  # straight colour under partial alpha


def test_fill_triangle_areas():
    c = core.Canvas(4, 4)
    c.fill(b"MLLZ", pack(0, 0, 4, 0, 0, 4), ID, RED, ())
    a = alpha(c)
    assert [a[i][3 - i] for i in range(4)] == [128] * 4  # the diagonal halves its pixels
    assert a[0][0] == 255 and a[3][3] == 0
    assert sum(map(sum, a)) == pytest.approx(8 * 255, abs=4)


def test_a_reversed_inner_contour_is_a_hole_and_same_way_overlap_counts_once():
    c = core.Canvas(10, 10)
    c.fill(b"MLLLZMLLLZ", pack(1, 1, 9, 1, 9, 9, 1, 9) + pack(3, 3, 3, 7, 7, 7, 7, 3), ID, RED, ())
    a = alpha(c)
    assert a[5][5] == 0 and a[2][2] == 255 and a[5][2] == 255
    c = core.Canvas(10, 10)
    c.fill(b"MLLLZMLLLZ", pack(1, 1, 6, 1, 6, 6, 1, 6) + pack(4, 4, 9, 4, 9, 9, 4, 9), ID, (1.0, 0.0, 0.0, 0.5), ())
    a = alpha(c)
    assert a[5][5] == a[2][2] == a[7][7] == 128  # the overlap is not painted twice


def test_shapes_cut_by_the_canvas_and_by_a_clip():
    c = core.Canvas(5, 3)
    c.fill(*rect(-10, -10, 20, 20), ID, RED, ())
    assert alpha(c) == [[255] * 5] * 3
    c = core.Canvas(5, 3)
    c.fill(*rect(2, -5, 30, 9), ID, RED, ())  # the right edge is outside: covered to the end
    assert alpha(c) == [[0, 0, 255, 255, 255]] * 3
    c = core.Canvas(6, 4)
    c.fill(*rect(0, 0, 6, 4), ID, RED, (1.5, 1.0, 4.0, 2.25))
    assert alpha(c) == [[0] * 6, [0, 128, 255, 255, 0, 0], [0, 32, 64, 64, 0, 0], [0] * 6]


def test_source_over():
    c = core.Canvas(2, 1)
    c.fill(*rect(0, 0, 2, 1), ID, (1.0, 1.0, 1.0, 1.0), ())
    c.fill(*rect(0, 0, 1, 1), ID, (0.0, 0.0, 1.0, 0.25), ())
    assert c.rgba8() == bytes([191, 191, 255, 255, 255, 255, 255, 255])


def test_hairlines_have_their_width_as_coverage():
    c = core.Canvas(8, 5)
    line(c, [1, 2.5, 7, 2.5], 0.5)
    a = alpha(c)
    assert a[2][1:7] == [128] * 6 and a[1] == [0] * 8 and a[3] == [0] * 8
    c = core.Canvas(8, 5)
    line(c, [1, 2.0, 7, 2.0], 1.0)  # on the boundary of two rows: half in each
    a = alpha(c)
    assert a[1][1:7] == [128] * 6 and a[2][1:7] == [128] * 6
    c = core.Canvas(5, 8)
    line(c, [2.5, 1, 2.5, 7], 3.0)
    assert [row[1:4] for row in alpha(c)[1:7]] == [[255] * 3] * 6


def test_butt_square_and_round_caps():
    for cap, ends in ((BUTT, (0, 0)), (SQUARE, (255, 255)), (ROUND, None)):
        c = core.Canvas(12, 5)
        line(c, [4, 2.5, 8, 2.5], 3.0, cap=cap)
        a = alpha(c)
        assert a[2][4:8] == [255] * 4
        if ends is not None:
            assert (a[2][3], a[2][8]) == ends and (a[1][3], a[1][8]) == ends
        else:
            assert a[2][3] == 255 and a[2][8] == 255 and 0 < a[1][2] < 255  # the disc rounds the corner


def test_a_densely_sampled_line_is_the_line():
    """The stroke is one outline: many collinear vertices leave the pixels of the plain two-point
    line, and a translucent stroke is nowhere painted twice."""
    for join in (MITER, RJOIN):
        plain = core.Canvas(60, 30)
        line(plain, [3.3, 4.1, 55.7, 25.9], 1.7, join=join)
        dense = core.Canvas(60, 30)
        pts = []
        for i in range(201):
            t = i / 200
            pts += [3.3 + t * 52.4, 4.1 + t * 21.8]
        line(dense, pts, 1.7, join=join)
        assert alpha(plain) == alpha(dense), join
    c = core.Canvas(60, 30)
    line(c, pts, 4.0, join=RJOIN, rgba=(0.0, 0.0, 0.0, 0.4))
    assert max(map(max, alpha(c))) == 102  # 0.4, nowhere painted twice


def test_miter_join_fills_the_corner_and_nothing_beyond():
    c = core.Canvas(20, 20)
    line(c, [4, 10, 10, 10, 10, 16], 4.0, join=MITER)
    a = alpha(c)
    assert a[8][11] == 255 and a[8][10] == 255  # the outer corner square [10, 12] x [8, 10]
    assert a[7][11] == 0 and a[8][12] == 0 and a[7][12] == 0
    c = core.Canvas(20, 20)
    line(c, [4, 10, 10, 10, 10, 16], 4.0, join=RJOIN)
    a = alpha(c)
    assert 0 < a[8][11] < 255  # rounded


def test_closed_path_joins_at_its_start():
    c = core.Canvas(12, 12)
    c.stroke(b"MLLLZ", pack(3, 3, 9, 3, 9, 9, 3, 9), ID, RED, (), 2.0, BUTT, MITER, [])
    a = alpha(c)
    for y, x in ((2, 2), (2, 9), (9, 2), (9, 9)):
        assert a[y][x] == 255  # all four corners, the first one too
    assert a[6][6] == 0 and a[1][1] == 0


def test_dashes():
    c = core.Canvas(20, 3)
    line(c, [0, 1.5, 20, 1.5], 1.0, dash=(3, 2))
    assert alpha(c)[1] == ([255] * 3 + [0] * 2) * 4
    c = core.Canvas(20, 3)
    line(c, [0, 1.5, 20, 1.5], 1.0, dash=(2.5, 2.5))
    assert alpha(c)[1][:5] == [255, 255, 128, 0, 0]


def test_curves_are_flattened_within_the_tolerance():
    k = 0.5522847498307936
    c = core.Canvas(64, 64)
    r, o = 30.0, 32.0
    xy = pack(
        o + r,
        o,
        o + r,
        o + k * r,
        o + k * r,
        o + r,
        o,
        o + r,
        o - k * r,
        o + r,
        o - r,
        o + k * r,
        o - r,
        o,
        o - r,
        o - k * r,
        o - k * r,
        o - r,
        o,
        o - r,
        o + k * r,
        o - r,
        o + r,
        o - k * r,
        o + r,
        o,
    )
    c.fill(b"MCCCCZ", xy, ID, RED, ())
    area = sum(map(sum, alpha(c))) / 255.0
    # the flattened outline is inscribed: up to kFlatness (0.05 pixel) inside the curve
    assert 0.9975 * 3.141592653589793 * r * r < area < 3.141592653589793 * r * r
    q = core.Canvas(8, 8)
    q.fill(b"MQZ", pack(0, 8, 4, 0, 8, 8), ID, RED, ())  # a parabola's segment: 2/3 of base 8 x height 4
    assert 0.98 * 8 * 4 * 2 / 3 < sum(map(sum, alpha(q))) / 255.0 <= 8 * 4 * 2 / 3 + 0.02


def test_matrix_maps_the_path():
    a, b = core.Canvas(20, 20), core.Canvas(20, 20)
    a.fill(*rect(4, 6, 10, 12), ID, RED, ())
    b.fill(*rect(0, 0, 3, 3), (2.0, 0.0, 0.0, 2.0, 4.0, 6.0), RED, ())
    assert a.rgba8() == b.rgba8()


def test_pixels_do_not_depend_on_the_thread_count():
    def draw():
        c = core.Canvas(200, 120)
        c.fill(*rect(0, 0, 200, 120), ID, (1.0, 1.0, 1.0, 1.0), ())
        pts = []
        for i in range(400):
            pts += [i * 0.5, 60 + 40 * ((i * 37 % 17) / 17 - 0.5)]
        line(c, pts, 1.3, join=RJOIN, rgba=(0.0, 0.3, 0.7, 0.8))
        return c.rgba8()

    old = sd.get_num_threads()
    try:
        out = []
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            out.append(draw())
    finally:
        sd.set_num_threads(old)
    assert out[0] == out[1] == out[2]


def test_malformed_calls_are_value_errors():
    c = core.Canvas(4, 4)
    with pytest.raises(ValueError, match="ops must be over MLQCZ"):
        c.fill(b"MX", pack(0, 0), ID, RED, ())
    with pytest.raises(ValueError, match="ops and xy do not agree"):
        c.fill(b"ML", pack(0, 0), ID, RED, ())
    with pytest.raises(ValueError, match="matrix needs 6 numbers, rgba 4"):
        c.fill(b"M", pack(0, 0), (1.0, 0.0), RED, ())
    with pytest.raises(ValueError, match="canvas size out of range"):
        core.Canvas(0, 4)
    with pytest.raises(ValueError, match="canvas size out of range"):
        core.Canvas(65536, 65536)
    c.fill(b"MLLZ", pack(0, 0, float("nan"), 1, 2, 2), ID, RED, ())  # a NaN path paints nothing
    line(c, [0, 0, float("inf"), 1], 1.0)
    assert set(c.rgba8()) == {0}


def test_markers_overlap_one_after_another_and_do_not_depend_on_the_thread_count():
    circle = (b"MLLLZ", pack(-2, -2, 2, -2, 2, 2, -2, 2))
    c = core.Canvas(20, 20)
    c.markers(*circle, ID, pack(8, 8, 10, 10), (1.0, 0.0, 0.0, 0.5), (), 0.0, ())
    a = alpha(c)
    assert a[7][7] == 128 and a[9][9] == 191 and a[11][11] == 128  # the overlap is painted twice: 0.5 over 0.5
    one = core.Canvas(20, 20)
    one.fill(b"MLLLZ", pack(6, 6, 10, 6, 10, 10, 6, 10), ID, (1.0, 0.0, 0.0, 0.5), ())
    one.fill(b"MLLLZ", pack(8, 8, 12, 8, 12, 12, 8, 12), ID, (1.0, 0.0, 0.0, 0.5), ())
    assert one.rgba8() == c.rgba8()  # a stamp is the shape filled at the point
    edged = core.Canvas(20, 20)
    edged.markers(*circle, ID, pack(10, 10), (), (0.0, 0.0, 0.0, 1.0), 1.0, ())
    e = alpha(edged)
    assert e[10][10] == 0 and e[10][7] == 128 and e[10][8] == 128  # the outline alone, half in each pixel it straddles
    at = array("d")
    for i in range(6000):  # across many bands of rows, overlapping, in a fixed scramble
        at.extend((3.0 + (i * 37 % 293) * 1.0, 3.0 + (i * 101 % 389) * 1.0))
    old, outs = sd.get_num_threads(), []
    try:
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            cv = core.Canvas(300, 400)
            m, fill, edge, clip = (
                (1.3, 0.0, 0.0, 1.3, 0.0, 0.0),
                (0.1, 0.4, 0.8, 0.3),
                (0.0, 0.0, 0.0, 0.6),
                (5.5, 5.5, 280.0, 390.0),
            )
            cv.markers(*circle, m, at.tobytes(), fill, edge, 0.7, clip)
            outs.append(cv.rgba8())
    finally:
        sd.set_num_threads(old)
    assert outs[0] == outs[1] == outs[2] and len(set(outs[0])) > 50
    skipped = core.Canvas(4, 4)
    skipped.markers(*circle, ID, pack(float("nan"), 1, 500, 500), RED, (), 0.0, ())  # not finite, or far outside
    assert set(skipped.rgba8()) == {0}


def test_decimate_keeps_what_a_column_can_show():
    xy = pack(0.0, 5.0, 0.2, 9.0, 0.4, 1.0, 0.6, 4.0, 0.9, 6.0, 1.1, 2.0, 2.5, 3.0, 2.6, 3.5)
    out = array("d")
    out.frombytes(core.decimate(xy, 1.0))
    # column [0, 1): first, highest, lowest, last (in their order); [1, 2): its one point; [2, 3): two points
    assert list(out) == [0.0, 5.0, 0.2, 9.0, 0.4, 1.0, 0.9, 6.0, 1.1, 2.0, 2.5, 3.0, 2.6, 3.5]
    assert core.decimate(b"", 1.0) == b"" and core.decimate(pack(1.0, 2.0), 0.5) == pack(1.0, 2.0)
    back = pack(0.5, 1.0, 1.5, 2.0, 0.6, 3.0)  # a run is consecutive points: returning to a column starts a new one
    assert core.decimate(back, 1.0) == back
    with pytest.raises(ValueError, match="xy must be packed pairs, column positive"):
        core.decimate(xy, 0.0)
