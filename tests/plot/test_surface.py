"""3D plots (docs/api/plot.md §4.5; docs/design/plot.md §6): the depth buffer keeps the nearest
surface, lines and points are tested against it, the box's axes are vector shapes around one
picture, and the picture is the same at any thread count."""

import math
import struct
from array import array

import pytest

import stardust as sd
from stardust._core import plot as core
from stardust.plot import _scene, _surface

from . import figures

P = sd.plot
TOP = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # looking down the z axis: x right, y up
LIGHT = [0.0, 0.0, 1.0]
# a plain shading for these tests: ambient and the key light, one small highlight, no fill or rim light
SHADE = [0.34, 0.56, 0.0, 0.0, 0.12, 0.0, 1.0, 0.30, 32.0, 0.0]


def view(scale=10.0, cx=10.0, cy=10.0, distance=0.0, zlo=-9.0, zhi=9.0, rot=TOP):
    return [*rot, distance, scale, cx, cy, zlo, zhi, *LIGHT, *SHADE]


def square(z, rgb, half=0.5, alpha=1.0):
    """A square at height z facing up: (vertices, triangles) bytes."""
    v = []
    for x, y in ((-half, -half), (half, -half), (half, half), (-half, half)):
        v += [x, y, z, 0.0, 0.0, 1.0, *rgb, alpha]
    return array("d", v).tobytes(), array("q", [0, 1, 2, 0, 2, 3]).tobytes()


def pixel(px, w, x, y):
    return tuple(px[4 * (y * w + x) : 4 * (y * w + x) + 4])


def render(vert, tri, lines=b"", counts=(), discs=b"", ss=2, **kw):
    return core.render3(20, 20, ss, view(**kw), vert, tri, lines, list(counts), [1.0] * len(counts), discs)


def test_the_nearest_surface_wins_whatever_the_order():
    (va, ta), (vb, _) = square(0.0, (1.0, 0.0, 0.0)), square(1.0, (0.0, 0.0, 1.0), half=0.2)
    tb = array("q", [4, 5, 6, 4, 6, 7]).tobytes()
    for vert, tri in ((va + vb, ta + tb), (va + vb, tb + ta)):
        px = render(vert, tri)
        assert pixel(px, 20, 10, 10)[2] > 200 and pixel(px, 20, 10, 10)[0] < 80  # the higher, smaller square on top
        assert (
            pixel(px, 20, 6, 10)[0] > 200 and pixel(px, 20, 6, 10)[2] < 80
        )  # the red one around it (the highlight adds a little)
        assert pixel(px, 20, 1, 1) == (0, 0, 0, 0)  # nothing: transparent


def test_translucent_surfaces_show_through_each_other_in_depth_order():
    """Red (opaque, lowest), green and blue sheets (half clear) above it: the picture is the same
    whatever order they are given in, the nearest sheet weighs most, and alone on the page a
    clear sheet keeps its alpha."""
    r, g, b = (
        square(0.0, (1.0, 0.0, 0.0)),
        square(1.0, (0.0, 1.0, 0.0), alpha=0.5),
        square(2.0, (0.0, 0.0, 1.0), alpha=0.5),
    )
    idx = [array("q", [k * 4 + i for i in (0, 1, 2, 0, 2, 3)]).tobytes() for k in range(3)]
    vert = r[0] + g[0] + b[0]
    first = render(vert, idx[0] + idx[1] + idx[2])
    assert all(render(vert, idx[a] + idx[c] + idx[d]) == first for a, c, d in ((2, 1, 0), (1, 2, 0), (1, 0, 2)))
    red, green, blue, alpha = pixel(first, 20, 10, 10)
    assert alpha == 255 and blue > green > 40 and red > 40  # blue over green over red: a half, a quarter, a quarter
    assert blue > green + 30 and abs(green - red) < 40  # the highlight adds to each layer
    alone = render(g[0], idx[0])
    assert 120 < pixel(alone, 20, 10, 10)[3] < 135 and pixel(alone, 20, 10, 10)[1] > 200  # half clear, straight green
    # a line on a clear sheet is hidden behind an opaque one, seen through a clear one
    line = array("d", [-0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]).tobytes()
    under_clear = render(g[0], idx[0], lines=line, counts=[2])
    assert sum(pixel(under_clear, 20, 10, 10)[:3]) < sum(pixel(alone, 20, 10, 10)[:3]) - 60
    top = square(1.0, (0.0, 1.0, 0.0))
    under_solid = render(top[0] + b[0], idx[0] + idx[1], lines=line, counts=[2])
    assert under_solid == render(top[0] + b[0], idx[0] + idx[1])


def test_crossing_triangles_split_along_their_line():
    # two tilted squares crossing along x = 0: each is nearer on one side
    def tilted(slope, rgb, base):
        v = []
        for x, y in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            v += [x, y, slope * x, -slope, 0.0, 1.0, *rgb, 1.0]
        return array("d", v).tobytes(), array("q", [base, base + 1, base + 2, base, base + 2, base + 3]).tobytes()

    (va, ta), (vb, tb) = tilted(1.0, (1.0, 0.0, 0.0), 0), tilted(-1.0, (0.0, 0.0, 1.0), 4)
    px = render(va + vb, ta + tb)
    left, right = pixel(px, 20, 7, 10), pixel(px, 20, 13, 10)
    assert left[2] > left[0] and right[0] > right[2]  # blue is higher where x < 0, red where x > 0


def test_shading_is_brightest_facing_the_light_and_two_sided():
    v, t = square(0.0, (0.5, 0.5, 0.5))
    facing = pixel(render(v, t), 20, 10, 10)
    assert facing[0] == facing[1] == facing[2] and facing[0] > 160  # ambient + diffuse + highlight on mid grey
    away = array("d", v)
    for i in range(4):
        away[10 * i + 5] = -1.0  # the normal turned away: the other side is lit the same
    assert pixel(render(away.tobytes(), t), 20, 10, 10) == facing
    tilted = array("d", v)
    for i in range(4):
        tilted[10 * i + 3], tilted[10 * i + 5] = 1.0, 0.3
    assert pixel(render(tilted.tobytes(), t), 20, 10, 10)[0] < facing[0] - 40


def test_lines_and_discs_are_tested_against_the_depth():
    v, t = square(0.0, (1.0, 1.0, 1.0))
    black = [0.0, 0.0, 0.0, 1.0]
    y = -0.05  # the middle of pixel row 10
    above = array("d", [-0.4, y, 0.5, *black, 0.4, y, 0.5, *black]).tobytes()
    below = array("d", [-0.4, y, -0.5, *black, 0.4, y, -0.5, *black]).tobytes()
    on = array("d", [-0.4, y, 0.0, *black, 0.4, y, 0.0, *black]).tobytes()
    assert pixel(render(v, t, above, [2]), 20, 10, 10)[:3] == (0, 0, 0)
    assert pixel(render(v, t, below, [2]), 20, 10, 10)[0] > 200  # hidden behind the square
    assert pixel(render(v, t, on, [2]), 20, 10, 10)[:3] == (0, 0, 0)  # a line on its surface shows
    disc = array("d", [0.0, 0.0, 0.5, 3.0, 1.0, 0.0, 0.0, 1.0]).tobytes()
    px = render(v, t, discs=disc)
    assert pixel(px, 20, 10, 10)[:3] == (255, 0, 0) and pixel(px, 20, 10, 6)[1] > 200  # a disc of radius 3 pixels
    assert (
        pixel(render(v, t, discs=array("d", [0.0, 0.0, -0.5, 3.0, 1.0, 0.0, 0.0, 1.0]).tobytes()), 20, 10, 10)[1] > 200
    )


def test_heights_outside_the_range_are_cut_and_edges_are_smoothed():
    v, t = square(0.0, (1.0, 0.0, 0.0))
    assert set(render(v, t, zlo=0.5, zhi=2.0)) == {0}
    px = render(v, t, cx=10.25, ss=4)  # the edge falls inside a pixel: partial alpha
    assert pixel(px, 20, 10, 10)[3] == 255 and 0 < pixel(px, 20, 5, 10)[3] < 255
    far = render(v, t, distance=4.0)
    near = render(*square(2.0, (1.0, 0.0, 0.0)), distance=4.0)
    assert sum(near[3::4]) > 2 * sum(far[3::4])  # perspective: nearer is larger


def test_render3_checks_its_input():
    v, t = square(0.0, (1.0, 0.0, 0.0))
    for args, message in (
        ((0, 20, 2, view(), v, t, b"", [], [], b""), "picture size out of range"),
        ((20, 20, 2, view()[:5], v, t, b"", [], [], b""), "view needs 18 or 28 numbers"),
        ((20, 20, 2, view(), v[:-8], t, b"", [], [], b""), "vertices, triangles, lines or discs have the wrong size"),
        ((20, 20, 2, view(), v, t, b"\0" * 56, [2], [1.0], b""), "line_counts do not add up to the points"),
        ((20, 20, 2, view(), v, t, b"\0" * 56, [1], [], b""), "a width for every strip"),
    ):
        with pytest.raises(ValueError, match=message):
            core.render3(*args)
    bad = array("q", [0, 1, 99]).tobytes()  # an index out of range: the triangle is left out
    assert set(core.render3(20, 20, 2, view(), v, bad, b"", [], [], b"")) == {0}


def space_of(fig):
    return fig._cell.layers[0]


def test_surface_builds_one_space_with_the_first_camera():
    fig = P.surface(lambda x, y: x * y, -1, 2, -3, 4, azimuth=10, elevation=20, perspective=False)
    sp = space_of(fig)
    assert isinstance(sp, _surface.Space) and (sp.az, sp.el, sp.perspective, sp.equal) == (10.0, 20.0, False, False)
    X, Y, Z, spec, mesh, alpha, fine, contours = sp.surfaces[0]
    assert (
        tuple(X.shape) == (_surface.N + 1, _surface.N + 1)
        and isinstance(spec, _surface.Auto)
        and mesh == _surface.MESH
        and alpha == 1.0
    )
    assert sp.limits() == ([-1.0, -3.0, -6.0], [2.0, 4.0, 8.0]) and fig._cell.frame is False
    fig.surface(lambda x, y: x + 10, -1, 2, -3, 4, color="red", azimuth=99)
    assert len(fig._cell.layers) == 1 and len(sp.surfaces) == 2 and sp.az == 10.0  # the camera of the first
    assert sp.surfaces[1][3] == (1.0, 0.0, 0.0, 1.0) and sp.limits()[1][2] == 12.0
    held = space_of(P.surface(lambda x, y: x * y, -1, 1, -1, 1, zlim=(-0.5, 0.25)))
    assert held.zlim == (-0.5, 0.25) and held.limits()[0][2] == -0.5
    pole = space_of(P.surface(lambda x, y: 1 / (x * x + y * y), -1, 1, -1, 1)).limits()
    assert pole[1][2] < 100  # the bulk of the heights, not the pole


def test_the_scene_of_a_3d_cell():
    fig = figures.hill("serif")
    sc = fig._root._build("t")
    kinds = [type(i).__name__ for i in sc.items]
    im = kinds.index("Image")
    assert kinds.count("Image") == 1 and kinds[1:im].count("Path") == 6  # three faces behind: grid and outline
    assert (
        all(k in ("Path", "Text") for k in kinds[im + 1 :]) and kinds[im + 1 :].count("Path") == 6
    )  # three axes: edge and ticks
    image = sc.items[im]
    assert (image.iw, image.ih) == (round(image.w * 300 / 72), round(image.h * 300 / 72)) and image.smooth
    alpha = image.rgba[3::4]
    assert 0.1 < sum(1 for a in alpha if a) / len(alpha) < 0.6 and alpha[0] == 0  # the surface on a transparent ground
    strings = [i.s for i in sc.items if isinstance(i, _scene.Text)]
    assert {"$x$", "$y$", "$z$"} <= set(strings) and len(strings) >= 12  # the labels and the ticks of three axes
    low = fig._root._build("t", 100.0).items[im]
    assert (low.iw, low.ih) == (
        round(image.w * 100 / 72),
        round(image.h * 100 / 72),
    )  # the picture follows the resolution
    assert (
        fig.to_png(dpi=72)[:4] == b"\x89PNG" and b"/Subtype /Image" in fig.to_pdf() and "data:image/png" in fig.to_svg()
    )


def test_parametric_surfaces_curves_and_points_keep_equal_units():
    fig = figures.solids("serif")
    torus, helix = (c.layers[0] for c in fig._root._cells)
    assert torus.equal and helix.equal and len(helix.curves) == 1 and len(helix.points) == 1
    lo, hi = torus.limits()
    assert (lo[0], hi[0]) == pytest.approx((-3.0, 3.0)) and (lo[2], hi[2]) == pytest.approx((-1.0, 1.0), abs=1e-3)
    x, y, z, rgba, width = helix.curves[0]
    assert x.shape[0] > 257 and width == 1.0 and helix.points[0][4] == 5.0 and not helix.perspective
    grouped = P.scatter3d([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], group=["a", "b", "a"])
    assert [p[0].tolist() for p in space_of(grouped).points] == [[0.0, 2.0], [1.0]]


def test_the_picture_does_not_depend_on_the_thread_count():
    old, outs = sd.get_num_threads(), []
    try:
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            fig = figures.crossing("serif")
            outs.append(next(i.rgba for i in fig._root._build("t").items if isinstance(i, _scene.Image)))
    finally:
        sd.set_num_threads(old)
    assert outs[0] == outs[1] == outs[2] and len(struct.unpack(f"{len(outs[0])}B", outs[0])) > 100000


def test_3d_errors():
    f = lambda x, y: x * y  # noqa: E731
    cases = [
        (lambda: P.surface(3, 0, 1, 0, 1), TypeError, "surface(): 'f' must be callable, not int"),
        (lambda: P.surface(f, 1, 0, 0, 1), ValueError, "surface(): 'x0' must be less than 'x1'"),
        (lambda: P.surface(f, 0, 1, 0, 1, zlim=3), TypeError, "surface(): 'zlim' must be a pair (low, high), not int"),
        (lambda: P.surface(f, 0, 1, 0, 1, elevation=100), ValueError, "surface(): 'elevation' must be in [-90, 90]"),
        (
            lambda: P.surface(f, 0, 1, 0, 1, cmap="jet"),
            ValueError,
            "surface(): 'cmap' must be 'cool', 'viridis', 'magma', 'cividis', 'rdbu', 'spectral', 'coolwarm', "
            "'sunrise' or 'lagoon', not 'jet'",
        ),
        (
            lambda: P.surface(f, 0, 1, 0, 1, cmap="magma", color="red"),
            ValueError,
            "surface(): 'color' and 'cmap' cannot both be given",
        ),
        (
            lambda: P.surface(f, 0, 1, 0, 1, mesh="x"),
            TypeError,
            "surface(): 'mesh' must be a bool or an int, not str",
        ),
        (
            lambda: P.line([0.0, 1.0], [0.0, 1.0]).surface(f, 0, 1, 0, 1),
            ValueError,
            "surface(): the cell already holds a 2D plot",
        ),
        (
            lambda: P.parametric_surface(f, f, 3, 0, 1, 0, 1),
            TypeError,
            "parametric_surface(): 'fz' must be callable, not int",
        ),
        (lambda: P.curve3d(sd.sin, sd.cos, sd.sin, 2, 1), ValueError, "curve3d(): 'a' must be less than 'b'"),
        (
            lambda: P.scatter3d([1.0], [1.0, 2.0], [1.0]),
            ValueError,
            "scatter3d(): 'x' and 'y' must have the same length: 1 and 2",
        ),
        (lambda: P.Figure().zlabel(3), TypeError, "zlabel(): 'text' must be a str, not int"),
    ]
    for call, kind, message in cases:
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message, message
    assert math.isfinite(_surface.DISTANCE)


def test_a_3d_cell_is_laid_out_as_a_2d_cell():
    """The box's projection fills the plot rectangle, the labels stand in the margins, and beside
    a 2D plot the two share their top and bottom."""
    solo = figures.hill("serif")
    sc = solo._root._build("t")
    image = next(i for i in sc.items if isinstance(i, _scene.Image))
    sp = solo._cell.layers[0]
    assert image.h / image.w == pytest.approx(sp.aspect(), rel=1e-9)  # a figure of one 3D cell takes the box's shape
    edges = [i for i in sc.items if isinstance(i, _scene.Path) and i.stroke == _surface.EDGE]
    xs = [v for e in edges for v in e.xy[0::2]]
    ys = [v for e in edges for v in e.xy[1::2]]
    assert min(xs) == pytest.approx(image.x, abs=0.6) and max(xs) == pytest.approx(
        image.x + image.w, abs=0.6
    )  # no unused room
    assert min(ys) >= image.y - 0.6 and max(ys) <= image.y + image.h + 0.6
    assert image.x > 10 and sc.height - (image.y + image.h) > 10  # the z ticks at the left, the x and y ticks below
    both = P.grid([P.line([0.0, 1.0], [0.0, 1.0]), figures.hill("serif")], ncols=2, width=183)
    sc = both._root._build("t")
    axes = next(i for i in sc.items if isinstance(i, _scene.Path) and i.ops == "MLL" and i.cap == "square")
    image = next(i for i in sc.items if isinstance(i, _scene.Image))
    top2d, bottom2d = axes.xy[1], axes.xy[3]
    assert image.y >= top2d - 0.01 and image.y + image.h <= bottom2d + 0.01  # inside the row's plot band
    assert (image.y + image.h / 2) == pytest.approx((top2d + bottom2d) / 2, abs=0.01)
    one = P.surface(lambda x, y: x * y, -1, 1, -1, 1)
    two = P.surface(lambda x, y: x * y, -1, 1, -1, 1).surface(lambda x, y: x + y, -1, 1, -1, 1, alpha=0.5)
    assert all(isinstance(s[3], _surface.Auto) for s in one._cell.layers[0].surfaces + two._cell.layers[0].surfaces)
    assert two._cell.layers[0].surfaces[1][5] == 0.5  # alone by height, among several a palette colour each
    with pytest.raises(ValueError, match=r"surface\(\): 'alpha' must be in \(0, 1\]"):
        P.surface(lambda x, y: x * y, -1, 1, -1, 1, alpha=0.0)


def test_a_surface_is_sampled_again_only_where_it_needs_it():
    calls = []

    def counted(f):
        def g(x, y):
            calls.append(x.size)
            return f(x, y)

        return g

    smooth = P.surface(counted(lambda x, y: sd.sin(x) * sd.cos(y)), -3, 3, -3, 3)
    assert space_of(smooth).surfaces[0][6] is None and calls == [(_surface.N + 1) ** 2]  # one call, as before
    calls.clear()
    pole = P.surface(counted(lambda x, y: 1.0 / (x * x + y * y)), -1, 1, -1, 1)
    keep, parts, (line, counts) = space_of(pole).surfaces[0][6]
    # the grid, two finer levels, the few points at the pole itself (no value there), the mesh lines
    assert len(parts) == 2 and calls[0] == (_surface.N + 1) ** 2 and len(calls) < 30
    assert sum(calls[1:3]) < 2 * calls[0]  # around the pole only
    (X1, Y1, Z1), keep1 = parts[0]
    assert tuple(X1.shape)[1:] == (5, 5) and tuple(keep1.shape) == (X1.shape[0], 4, 4)
    assert 0 < int(sd.sum(~keep)) == X1.shape[0] < 0.1 * keep.size  # the cells that were replaced
    assert sum(counts) == line[0].shape[0] and max(counts) > _surface.N + 1  # denser lines by the pole
    # no crack: where a finer cell lies beside a cell of the first grid, its edge is that cell's straight edge
    i, j = (v.tolist() for v in sd.nonzero(~keep))
    taken = set(zip(i, j))
    k = next(n for n, (a, b) in enumerate(zip(i, j)) if (a, b + 1) not in taken)  # a coarser cell on the right
    edge = Z1[k, :, 4].tolist()
    assert edge[2] == pytest.approx(0.5 * (edge[0] + edge[4]), rel=1e-12)
    assert edge[0] == pytest.approx(
        1.0 / (X1[k, 0, 4] ** 2 + Y1[k, 0, 4] ** 2), rel=1e-12
    )  # corners are the function's
    assert pole._root._build("t") is not None


def test_the_edge_of_a_domain_is_followed():
    dome = P.surface(lambda x, y: sd.sqrt(1.0 - x * x - y * y), -1.2, 1.2, -1.2, 1.2)
    keep, parts, _ = space_of(dome).surfaces[0][6]
    (X, Y, Z), _ = parts[-1]
    r = sd.sqrt(X * X + Y * Y)
    ok = sd.isfinite(Z)
    assert float(sd.max(r[ok])) <= 1.0 and float(sd.max(r[ok])) > 1.0 - 1e-5  # points moved onto the rim
    low = Z[ok & (r > 1.0 - 1e-6)]
    assert low.shape[0] > 100 and float(sd.max(low)) < 2e-3  # the rim reaches the floor
    twisted = P.parametric_surface(lambda u, v: u, lambda u, v: v, lambda u, v: sd.abs(u) + 0.0 * v, -1, 1, -1, 1)
    keep, parts, _ = space_of(twisted).surfaces[0][6]
    cols = sd.nonzero(~keep)[1].tolist()
    assert set(cols) <= {58, 59, 60, 61} and len(parts) == 1  # the fold, and nothing finer inside it (it is flat there)


def test_a_space_curve_is_sampled_where_it_turns_and_cut_at_a_pole():
    import math

    line = P.curve3d(lambda t: t, lambda t: 2 * t, lambda t: -t, 0, 1)
    assert space_of(line).curves[0][0].shape[0] == 257  # straight: the first samples are enough
    coil = P.curve3d(lambda t: sd.cos(t), lambda t: sd.sin(t), lambda t: t / 400, 0, 400 * math.pi)
    assert space_of(coil).curves[0][0].shape[0] > 12000  # 200 turns: sixty samples a turn or more
    wild = P.curve3d(lambda t: t, lambda t: 0.0 * t, lambda t: sd.tan(t), -3, 3)
    sp = space_of(wild)
    x, y, z = sp.curves[0][:3]
    assert int(sd.sum(~sd.isfinite(z))) >= 2  # the line stops at the two poles
    lo, hi = sp.limits()
    assert (
        -40 < lo[2] < -3 and 3 < hi[2] < 40 and float(sd.max(sd.abs(z[sd.isfinite(z)]))) > 100
    )  # the bulk, not the poles
    cx, cy, cz = _surface._clip(x, y, z, lo, hi)
    ok = sd.isfinite(cz)
    assert float(sd.max(cz[ok])) == hi[2] and float(sd.min(cz[ok])) == lo[2]  # ends on the box's faces
    assert wild._root._build("t") is not None


def test_a_surface_rough_everywhere_is_sampled_finer_as_a_whole():
    rough = P.surface(lambda x, y: sd.sin(30 * x) * sd.sin(30 * y), -1, 1, -1, 1)
    X, _, _, _, _, _, fine, _ = space_of(rough).surfaces[0]
    assert tuple(X.shape) == (3 * _surface.N + 1, 3 * _surface.N + 1) and fine is None


def test_a_triangle_is_cut_at_the_height_range_before_it_is_projected():
    """A long triangle seen in perspective: nothing of it shows beyond the cut (a straight
    interpolation on the screen would let a spike through)."""
    v = array(
        "d",
        [-0.5, -0.5, 0.0, 0, 0, 1, 1, 0, 0, 1, 0.5, -0.5, 0.0, 0, 0, 1, 1, 0, 0, 1, 0.0, 0.0, 5.0, 0, 0, 1, 1, 0, 0, 1],
    )
    side = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0]  # looking along y: x right, z up
    px = render(v.tobytes(), array("q", [0, 1, 2]).tobytes(), rot=side, distance=7.0, zlo=-1.0, zhi=0.5, cy=15.0)
    rows = [y for y in range(20) if any(pixel(px, 20, x, y)[3] for x in range(20))]
    top = 15.0 - 10.0 * 0.5 * 7.0 / 7.0  # the row of height 0.5 on the axis (the triangle is in the plane y = 0)
    assert rows and min(rows) >= top - 1.5 and max(rows) <= 15  # from the base up to the cut, no further


def test_mesh_count_and_floor_contours():
    f = lambda x, y: x * x - y * y  # noqa: E731
    plain = space_of(P.surface(f, -1, 1, -1, 1, mesh=False))
    lo, hi = plain.limits()
    th, box = {"marker_size": 3.0}, _surface.BOX_GRAPH
    assert plain._buffers(lo, hi, th, 1.0, box)[3] == []  # no lines at all
    for mesh, lines in ((True, 2 * (_surface.MESH + 1)), (10, 22), (40, 82)):
        counts = space_of(P.surface(f, -1, 1, -1, 1, mesh=mesh))._buffers(lo, hi, th, 1.0, box)[3]
        assert len(counts) == lines, mesh
    sp = space_of(P.surface(f, -1, 1, -1, 1, mesh=False, contours=True))
    _, _, lines, counts, widths, _ = sp._buffers(lo, hi, th, 2.0, box)
    assert 8 <= len(counts) <= 20 and set(widths) == {_surface.CONTOUR_WIDTH * 2.0}
    pts = array("d", lines)
    assert set(pts[2::7]) <= {-box[2], math.nan} or all(z == -box[2] or z != z for z in pts[2::7])  # on the floor
    assert len(space_of(P.surface(f, -1, 1, -1, 1, mesh=False, contours=5))._buffers(lo, hi, th, 1.0, box)[3]) < len(
        counts
    )
    torus = P.parametric_surface(lambda u, v: sd.cos(u), lambda u, v: sd.sin(u), lambda u, v: v, 0, 6, 0, 1)
    assert space_of(torus).surfaces[0][7] == 0  # a parametric surface has no floor contours
    for bad, kind, text in (
        (dict(mesh=0), ValueError, "'mesh' must be between 1 and 400"),
        (dict(mesh=1.5), TypeError, "'mesh' must be a bool or an int, not float"),
        (dict(contours=101), ValueError, "'contours' must be between 1 and 100"),
        (dict(contours="a"), TypeError, "'contours' must be a bool or an int, not str"),
    ):
        with pytest.raises(kind, match=text):
            P.surface(f, -1, 1, -1, 1, **bad)


def test_the_camera_is_chosen_when_it_is_not_given():
    wave = lambda x, y: sd.sin(x) * sd.cos(y)  # noqa: E731
    assert space_of(P.surface(wave, -3, 3, -3, 3)).camera() == (-60.0, 30.0)  # nothing to gain: the usual view
    assert space_of(P.surface(wave, -3, 3, -3, 3, azimuth=10, elevation=20)).camera() == (10.0, 20.0)
    # a step facing the far side is looked at from there
    step = space_of(P.surface(lambda x, y: sd.tanh(6 * (x + y)) + 0.2 * sd.sin(4 * x), -1, 1, -1, 1))
    az, el = step.camera()
    assert az in _surface.VIEW_AZIMUTHS and el in _surface.VIEW_ELEVATIONS and az != -60.0
    right, up, eye = step.rotation()
    assert eye[0] + eye[1] < 0  # from the side of the lower half: the face of the step is in view
    # one angle given: only the other is chosen
    half = space_of(P.surface(lambda x, y: sd.tanh(6 * (x + y)), -1, 1, -1, 1, elevation=55))
    assert half.camera()[1] == 55.0 and half.camera()[0] in _surface.VIEW_AZIMUTHS
    assert space_of(P.surface(wave, -3, 3, -3, 3, azimuth=10)).camera()[0] == 10.0
    # the choice follows the scene: a second surface is taken into account
    two = P.surface(wave, -3, 3, -3, 3)
    first = space_of(two).camera()
    two.surface(lambda x, y: sd.tanh(6 * (x + y)) * 3, -3, 3, -3, 3)
    assert space_of(two)._chosen[0] != (2, 0, 0, None, False) and space_of(two).camera() in [
        (a, e) for a in _surface.VIEW_AZIMUTHS for e in _surface.VIEW_ELEVATIONS
    ]
    assert first == (-60.0, 30.0)
    # points alone: the view that spreads them (a flat cloud is not looked at edge on)
    xs = [math.cos(i) * (1 + i % 7) / 4 for i in range(200)]
    zs = [math.sin(i * 1.7) * (1 + i % 5) / 3 for i in range(200)]
    cloud = space_of(P.scatter3d(xs, [0.01 * math.sin(3 * i) for i in range(200)], zs))
    az, el = cloud.camera()
    assert abs(math.cos(math.radians(az))) < 0.6  # not along the plane of the points (y = 0)
    assert cloud.cell is not None and P.scatter3d(xs, xs, zs)._root._build("t") is not None


def test_view_scores_in_the_core():
    v = array("d", [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]).tobytes()  # a square in the plane z = 0
    t = array("q", [0, 1, 2, 0, 2, 3]).tobytes()
    top, slant, edge = (
        list(r) for r in (_surface._rot(-90.0, 90.0), _surface._rot(-60.0, 30.0), _surface._rot(-60.0, 0.0))
    )
    flat = [x for rot in (top, slant, edge) for row in rot for x in row]
    s = core.view_scores(v, t, b"", flat, 0.0, 1.5, 160)
    assert s[0] == pytest.approx(1.0, abs=0.02) and 0.7 < s[2] < 0.9 and s[4] < 0.05  # face on, slanted, edge on
    assert s[1] == s[3] == s[5] == 0.0  # no loose points
    # a second square just above hides the first: half of the area is seen from the top
    w = array("d", [-1, -1, 0.5, 1, -1, 0.5, 1, 1, 0.5, -1, 1, 0.5]).tobytes()
    both = core.view_scores(v + w, t + array("q", [4, 5, 6, 4, 6, 7]).tobytes(), b"", flat[:9], 0.0, 1.5, 160)
    assert both[0] == pytest.approx(0.5, abs=0.03)
    pts = array("d", [c for i in range(50) for c in (math.cos(i), math.sin(i), 0.0)]).tobytes()
    p = core.view_scores(b"", b"", pts, flat, 0.0, 1.5, 160)
    assert p[1] > p[3] > p[5] == pytest.approx(0.0, abs=1e-9) and p[0] == 0.0  # a ring: widest from the top
    with pytest.raises(ValueError, match="view_scores"):
        core.view_scores(v, t, b"", flat[:5], 0.0, 1.5, 160)
