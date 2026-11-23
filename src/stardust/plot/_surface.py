"""3D plots (docs/api/plot.md §4.5; docs/design/plot.md §6): surface, parametric_surface,
curve3d, scatter3d.

A 3D cell holds one Space: the surfaces, curves and points added to it and the camera of the
first of them. When the figure is drawn, the Space scales everything into a box, hands the
triangles, lines and discs to the depth-buffer renderer (csrc/plot/zbuffer.hpp) for one picture,
and draws the box's axes as vector shapes: the three faces behind the scene with their grid
below the picture, the three axis edges with ticks and labels above it.
"""

import math
from array import array

from .. import _core as sd
from .._core import _type_name
from .._core import plot as _core_plot
from ..optimize._common import _check_bool
from . import _adapt, _basic, _color, _field, _layers, _sample, _scene, _ticks
from ._basic import _opt_real, numbers_, same_length
from ._figure import INK, real
from ._function import _callable, _interval, call

N = 120  # cells of a surface grid along each direction
BOX_GRAPH = (1.0, 1.0, 0.72)  # half sizes of the box a graph z = f(x, y) is scaled into
DISTANCE = 7.0  # of the eye from the box's centre, in those units (perspective)
SS = 3  # samples per pixel along each direction
LIGHT = (-0.35, 0.45, 0.82)
GRID = (0.88, 0.88, 0.89, 1.0)  # the grid on the faces behind the scene
EDGE = (0.72, 0.72, 0.74, 1.0)  # those faces' outlines
MESH_DARK, MESH_ALPHA = 0.36, 0.62  # mesh lines: the surface's own colour darkened
# The views an angle that was not given is chosen from; the first of each is the usual one.
# (no view along an axis of the box: its ticks would fall on one another)
VIEW_AZIMUTHS = (-60.0, -30.0, 30.0, 60.0, 120.0, 150.0, -150.0, -120.0)
VIEW_ELEVATIONS = (30.0, 18.0, 42.0)
VIEW_USUAL = (0.03, 0.07)  # what the usual azimuth and elevation are ahead by (of a score of at most 1)
VIEW_GRID, VIEW_SIZE = 32, 160  # cells a side of the coarse copy of a surface; samples a side of its depth buffer
LONE = _color.CMAP  # the colour map of a cell's only surface
MESH, MESH_WIDTH = 24, 0.42  # cells of the mesh a side; the lines' width (bp)
CONTOURS, CONTOUR_WIDTH = 14, 0.6  # levels of the floor contours aimed at; their width (bp)
# The shading (csrc/plot/zbuffer.hpp, View3::shade): ambient; key, fill and rim light; the darkening
# of faces turned away; the key light's broad gloss (strength, exponent) and sharp highlight
# (strength, exponent); the fill light's gloss.
SHADE = (0.52, 0.36, 0.20, 0.10, 0.03, 0.18, 5.0, 0.30, 70.0, 0.12)


class Space(_layers.Layer):
    """The 3D content of a cell and its camera."""

    def __init__(self, azimuth, elevation, perspective):
        self.az, self.el, self.perspective = azimuth, elevation, perspective  # an angle None: chosen (camera)
        self._chosen = None  # (what the scene held, (azimuth, elevation)) of the last choice
        self._picture = None  # (what was rendered, the pixels) of the last picture
        # (X, Y, Z 2-D sd arrays, colour spec, mesh cells, alpha, what _adapt added or None, floor contour levels)
        self.surfaces = []
        self.curves = []  # (x, y, z 1-D arrays, rgba, width, label)
        self.points = []  # (x, y, z arrays, rgba, size, label)
        self.zlim = None  # a user's z range
        self.equal = True  # equal data units along the three axes (until a graph z = f(x, y) is added)

    def fork(self, cell):
        """Turns a shallow copy into a space of its own, in the cell `cell`."""
        self.surfaces, self.curves, self.points = list(self.surfaces), list(self.curves), list(self.points)
        self._chosen = self._picture = None
        self.cell = cell

    def entries(self):
        """The legend's entries: named curves and points (scatter3d's groups)."""
        out = []
        for _x, _y, _z, rgba, width, label in self.curves:
            if label is not None:
                out.append((label, _layers._line_handle(rgba, 1.0 if width is None else width, ())))
        for _x, _y, _z, rgba, size, label in self.points:
            if label is not None:
                out.append((label, _layers._marker_handle("o", 3.8 if size is None else size, rgba, None, 0.0)))
        return out

    # -- ranges

    def limits(self):
        lo, hi = [math.inf] * 3, [-math.inf] * 3
        for group in (self.surfaces, self.curves, self.points):
            for item in group:
                for k in range(3):
                    r = _layers.finite_range(sd.reshape(item[k], (-1,)))
                    if r is not None and group is self.curves:
                        # a curve that runs away (a pole) does not take the axis: the bulk of its samples
                        r = _sample.robust_range(item[k])[:2]
                    if r is not None:
                        lo[k], hi[k] = min(lo[k], r[0]), max(hi[k], r[1])
        for k in range(3):
            if not lo[k] <= hi[k]:
                lo[k], hi[k] = 0.0, 1.0
            if lo[k] == hi[k]:
                d = 0.5 if lo[k] == 0.0 else abs(lo[k]) * 0.05
                lo[k], hi[k] = lo[k] - d, hi[k] + d
        if self.zlim is not None:
            lo[2], hi[2] = self.zlim
        elif self.surfaces:
            # a surface that runs away (a pole) does not take the axis: the bulk of the heights
            zs = sd.concat([sd.reshape(s[2], (-1,)) for s in self.surfaces])
            r = _sample.robust_range(zs)
            if r[2] or r[3]:
                lo[2], hi[2] = max(lo[2], r[0]), min(hi[2], r[1])
        if not self.surfaces and self.points:
            # points alone: a margin as a 2D plot keeps, so that the outermost do not sit on the axes
            for k in range(3):
                d = 0.04 * (hi[k] - lo[k])
                lo[k], hi[k] = lo[k] - d, hi[k] + d
        return lo, hi

    # -- the camera

    def camera(self):
        """(azimuth, elevation): the angles given, and for one that was not given the best of
        VIEW_AZIMUTHS or VIEW_ELEVATIONS. A view is good when it shows much of the surfaces'
        area, little of it hidden behind other parts or seen edge on, and spreads the curves
        and points over the screen (csrc/plot/zbuffer.hpp, view_scores, on a coarse copy of the
        scene); the usual view (-60, 30) wins a near tie, so plain surfaces look as ever."""
        if self.az is not None and self.el is not None:
            return self.az, self.el
        key = (len(self.surfaces), len(self.curves), len(self.points), self.zlim, self.equal)
        if self._chosen is not None and self._chosen[0] == key:
            return self._chosen[1]
        lo, hi = self.limits()
        BOX = self.box_of(lo, hi)

        def box(k, a):
            return sd.clip((a - lo[k]) * (2.0 * BOX[k] / (hi[k] - lo[k])) - BOX[k], min=-BOX[k], max=BOX[k])

        vert, tri, loose, base = [], [], [], 0
        for X, Y, Z, *_ in self.surfaces:
            sy, sx = (max(1, -(-(n - 1) // VIEW_GRID)) for n in X.shape)
            cols = [sd.reshape(box(k, c[::sy, ::sx]), (-1,)) for k, c in enumerate((X, Y, Z))]
            ny, nx = X[::sy, ::sx].shape
            vert.append(bytes(memoryview(sd.reshape(sd.stack(cols, axis=1), (-1,)))))
            idx = sd.reshape(sd.arange(0, ny * nx), (ny, nx)) + base
            p00, p01, p10, p11 = (
                sd.reshape(q, (-1,)) for q in (idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:])
            )
            tri.append(bytes(memoryview(sd.reshape(sd.stack([p00, p01, p11, p00, p11, p10], axis=1), (-1,)))))
            base += ny * nx
        for group, most in ((self.curves, 300), (self.points, 600)):
            for x, y, z, *_ in group:
                step = max(1, -(-x.shape[0] // most))
                cols = [box(k, c[::step]) for k, c in enumerate((x, y, z))]
                loose.append(bytes(memoryview(sd.reshape(sd.stack(cols, axis=1), (-1,)))))
        views = [
            (a, e)
            for a in (VIEW_AZIMUTHS if self.az is None else (self.az,))
            for e in (VIEW_ELEVATIONS if self.el is None else (self.el,))
        ]
        flat = [v for a, e in views for row in _rot(a, e) for v in row]
        scores = _core_plot.view_scores(
            b"".join(vert),
            b"".join(tri),
            b"".join(loose),
            flat,
            DISTANCE if self.perspective else 0.0,
            math.sqrt(sum(b * b for b in BOX)),
            VIEW_SIZE,
        )
        spread = max(scores[1::2]) or 1.0
        best, top = views[0], -1.0
        for i, (a, e) in enumerate(views):
            total = scores[2 * i] + (0.15 if tri else 1.0) * scores[2 * i + 1] / spread
            total += VIEW_USUAL[0] * (a == VIEW_AZIMUTHS[0]) + VIEW_USUAL[1] * (e == VIEW_ELEVATIONS[0])
            if total > top:
                best, top = (a, e), total
        self._chosen = (key, best)
        return best

    def rotation(self, usual=False):
        """(right, up, eye) of the camera; `usual`: of the usual view where an angle is to be chosen."""
        az, el = (
            (VIEW_AZIMUTHS[0] if self.az is None else self.az, VIEW_ELEVATIONS[0] if self.el is None else self.el)
            if usual
            else self.camera()
        )
        return _rot(az, el)

    def project(self, p, rot, scale=1.0, cx=0.0, cy=0.0):
        """The page position of a box point (and its depth)."""
        right, up, eye = rot
        x = right[0] * p[0] + right[1] * p[1] + right[2] * p[2]
        y = up[0] * p[0] + up[1] * p[1] + up[2] * p[2]
        z = eye[0] * p[0] + eye[1] * p[1] + eye[2] * p[2]
        f = DISTANCE / (DISTANCE - z) if self.perspective else 1.0
        return cx + scale * f * x, cy - scale * f * y, z

    def box_of(self, lo, hi):
        """The box's half sizes."""
        if self.equal:  # the box has the data's proportions
            big = max(hi[k] - lo[k] for k in range(3))
            return tuple(max((hi[k] - lo[k]) / big, 0.05) for k in range(3))
        return BOX_GRAPH

    def outline(self, lo, hi):
        """The box's half sizes and the bounds (x0, x1, y0, y1) of its projection at scale 1
        (y down, as on the page)."""
        box = self.box_of(lo, hi)
        rot = self.rotation()
        pr = [
            self.project((sx * box[0], sy * box[1], sz * box[2]), rot)
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        return box, (min(p[0] for p in pr), max(p[0] for p in pr), min(p[1] for p in pr), max(p[1] for p in pr))

    def aspect(self):
        """Height over width of the box's projection: the shape of the cell's plot rectangle."""
        _, (x0, x1, y0, y1) = self.outline(*self.limits())
        return (y1 - y0) / (x1 - x0)

    def margins(self, th, engine):
        """(left, right, top, bottom): the room the ticks and labels around the box take."""
        ts, tl = th["tick_size"], th["tick_len"]
        lo, hi = self.limits()
        wide = []
        for k in range(3):
            text, _ = _ticks.labels(_ticks.linear(lo[k], hi[k], 4))
            wide.append(max((engine.layout([s])[0].box(ts)[0] for s in text), default=0.0))
        names = self.labels()
        extra = [engine.layout([n])[0].box(th["size"]) if n else None for n in names]
        left = tl + 3.0 + wide[2] + (extra[2][0] + 4.0 if extra[2] else 0.0)  # the z name stands level: its width
        below = tl + 3.0 + ts + (th["size"] + 4.0 if (extra[0] or extra[1]) else 0.0)
        return left, 0.5 * max(wide[0], wide[1]) + 2.0, 2.0, below

    # -- drawing

    def draw(self, fr):
        th = fr.theme
        lo, hi = self.limits()
        rot = self.rotation()
        BOX, (bx0, bx1, by0, by1) = self.outline(lo, hi)
        eye = rot[2]
        # The box's projection fills the plot rectangle, as a 2D plot's axes do; ticks and labels
        # stand in the margins the layout kept (margins).
        scale = min(fr.w / (bx1 - bx0), fr.h / (by1 - by0))
        cx = fr.x + (fr.w - scale * (bx1 - bx0)) / 2 - scale * bx0
        cy = fr.y + (fr.h - scale * (by1 - by0)) / 2 - scale * by0

        def P(p):
            x, y, _ = self.project(p, rot, scale, cx, cy)
            return x, y

        def box(k, v):
            return (v - lo[k]) / (hi[k] - lo[k]) * 2.0 * BOX[k] - BOX[k]

        # the faces behind the scene: for each axis, the side turned away from the eye
        back = [-BOX[k] if eye[k] > 0 else BOX[k] for k in range(3)]
        front = [-b for b in back]

        # the axes: the bottom edges nearest the eye for x and y, the upright edge at the side for z
        def edge(k, fixed):
            a, b = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            for m in range(3):
                if m == k:
                    a[m], b[m] = -BOX[m], BOX[m]
                else:
                    a[m] = b[m] = fixed[m]
            return a, b

        zc = min(
            ((back[0], front[1]), (front[0], back[1])), key=lambda c: P((c[0], c[1], 0.0))[0]
        )  # the left of the two
        axes = [
            (0, edge(0, (0.0, front[1], -BOX[2])), (0.0, front[1] / BOX[1], -0.35)),
            (1, edge(1, (front[0], 0.0, -BOX[2])), (front[0] / BOX[0], 0.0, -0.35)),
            (2, edge(2, (zc[0], zc[1], 0.0)), (zc[0] / BOX[0], zc[1] / BOX[1], 0.0)),
        ]
        # as many ticks as the drawn length of each axis has room for (an axis seen end on is short)
        drawn = {k_: math.dist(P(a), P(b)) for k_, (a, b), _ in axes}
        want = [max(2, min(6, round(drawn[k] * 0.8 / (24.0 if k < 2 else 18.0)) + 1)) for k in range(3)]
        ticks = [[t for t in _ticks.linear(lo[k], hi[k], want[k]) if lo[k] <= t <= hi[k]] for k in range(3)]
        scene = fr.scene
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            quad = []
            for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[k], p[i], p[j] = back[k], si * BOX[i], sj * BOX[j]
                quad.append(P(p))
            ops, xy = [], array("d")
            for a, b in ((i, j), (j, i)):  # grid lines at the ticks of axis a, running along b
                for t in ticks[a]:
                    p, q = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                    p[k] = q[k] = back[k]
                    p[a] = q[a] = box(a, t)
                    p[b], q[b] = -BOX[b], BOX[b]
                    ops.append("ML")
                    xy.extend(P(p) + P(q))
            if ops:
                scene.add(_scene.Path("".join(ops), xy, stroke=GRID, width=0.4))
            scene.add(_scene.Path(*_scene.polyline(quad, closed=True), stroke=EDGE, width=0.4))

        # the picture
        dpi = getattr(fr, "dpi", 300.0)
        k = dpi / 72.0
        iw, ih = max(1, round(fr.w * k)), max(1, round(fr.h * k))
        right, up, _ = rot
        zr = (-BOX[2] - 1e-9, BOX[2] + 1e-9)
        view = [
            *right,
            *up,
            *eye,
            DISTANCE if self.perspective else 0.0,
            scale * k,
            (cx - fr.x) * k,
            (cy - fr.y) * k,
            zr[0],
            zr[1],
            *_unit(LIGHT),
            *SHADE,
        ]
        vert, tri, lines, counts, widths, discs = self._buffers(lo, hi, th, k, BOX)
        if vert or lines or discs:
            # A notebook asks for the figure twice (SVG and PNG): the picture of an unchanged scene
            # at the same size is rendered once.
            key = (
                iw,
                ih,
                tuple(view),
                len(vert),
                len(tri),
                len(lines),
                tuple(counts),
                tuple(widths),
                len(discs),
                hash(discs),
            )
            if self._picture is None or self._picture[0] != key:
                self._picture = (key, _core_plot.render3(iw, ih, SS, view, vert, tri, lines, counts, widths, discs))
            rgba = self._picture[1]
            scene.add(_scene.Image(fr.x, fr.y, fr.w, fr.h, iw, ih, rgba, smooth=True))

        ts, tl = th["tick_size"], th["tick_len"]
        labels = self.labels()
        placed = []  # (axis, box) of the tick labels set so far
        for k_, (a, b), out in axes:
            pa, pb = P(a), P(b)
            scene.add(_scene.Path(*_scene.polyline([pa, pb]), stroke=INK, width=th["axis_width"], cap="round"))
            # ticks point away from the box: the page direction of `out` from the edge's middle
            mid = [(a[m] + b[m]) / 2 for m in range(3)]
            m0, m1 = P(mid), P([mid[m] + 0.3 * out[m] for m in range(3)])
            dx, dy = m1[0] - m0[0], m1[1] - m0[1]
            dl = math.hypot(dx, dy) or 1.0
            ux, uy = dx / dl, dy / dl
            # an axis seen almost end on (a thin box) has no room for ticks
            shown = ticks[k_] if math.hypot(pb[0] - pa[0], pb[1] - pa[1]) >= 14.0 else []
            text, mult = _ticks.labels(shown) if shown else ([], None)
            ops, xy, far = [], array("d"), 0.0
            for t, s in zip(shown, text):
                p = list(a)
                p[k_] = box(k_, t)
                px, py = P(p)
                ops.append("ML")
                xy.extend((px, py, px + ux * tl, py + uy * tl))
                w, h, d = fr.engine.layout([s])[0].box(ts)
                # the label's centre lies beyond the tick by half its extent along the direction
                r = abs(ux) * w / 2 + abs(uy) * (h + d) / 2
                far = max(far, tl + 2.0 + 2 * r)
                lx, ly = px + ux * (tl + 2.0 + r), py + uy * (tl + 2.0 + r)
                rect = (lx - w / 2, ly - (h + d) / 2, lx + w / 2, ly + (h + d) / 2)
                # A label needs clear room around it: a little from its own axis's labels, and from
                # another axis's (at a corner where two axes meet) enough that the two numbers are
                # not read as one or as each other's.
                if any(
                    rect[0] - q[2] < gx and q[0] - rect[2] < gx and rect[1] - q[3] < gy and q[1] - rect[3] < gy
                    for q, (gx, gy) in ((q, (3.0, 2.0) if n == k_ else (ts, 0.6 * ts)) for n, q in placed)
                ):
                    continue  # left out
                placed.append((k_, rect))
                scene.add(_scene.Text(s, lx, ly, size=ts, ha="center", va="center"))
            if ops:
                scene.add(_scene.Path("".join(ops), xy, stroke=INK, width=th["axis_width"]))
            name = labels[k_]
            if mult:
                name = f"{name} {mult}" if name else mult
            if name:
                w, h, d = fr.engine.layout([name])[0].box(th["size"])
                r = abs(ux) * w / 2 + abs(uy) * (h + d) / 2
                scene.add(
                    _scene.Text(
                        name,
                        m0[0] + ux * (far + 3.0 + r),
                        m0[1] + uy * (far + 3.0 + r),
                        size=th["size"],
                        ha="center",
                        va="center",
                    )
                )

    def labels(self):
        c = self.cell
        return (c.xlabel or "", c.ylabel or "", c.zlabel or "")

    def _buffers(self, lo, hi, th, k, BOX):
        """The renderer's inputs in box coordinates: vertices, triangles, lines, strip lengths, discs."""

        def box(k, a):
            return (a - lo[k]) * (2.0 * BOX[k] / (hi[k] - lo[k])) - BOX[k]

        vert, tri, lines, counts, widths, discs = [], [], [], [], [], []
        base = 0
        several = len(self.surfaces) > 1
        for X, Y, Z, spec, mesh, alpha, fine, contours in self.surfaces:
            if isinstance(spec, Auto):  # no colour given: by height alone, a palette colour among several
                spec = _color.fill_of(spec.rgba, 0.18) if several else LONE

            def colours(Zs, spec=spec):
                if isinstance(spec, (str, list)):  # a colour map by height: a named one, or anchors
                    return _shade(spec, sd.clip((Zs - lo[2]) / (hi[2] - lo[2]), min=0.0, max=1.0)), alpha
                return [sd.full(tuple(Zs.shape), c) for c in spec[:3]], spec[3] * alpha

            # the first grid, and the grids of the cells that were sampled again (_adapt)
            sheets = [((X, Y, Z), None)] if fine is None else [((X, Y, Z), fine[0]), *fine[1]]
            for (Xs, Ys, Zs), keep in sheets:
                x, y, z = box(0, Xs), box(1, Ys), box(2, Zs)
                nrm = _adapt.normals(x, y, z)
                (r, g, b), a = colours(Zs)
                cols = [sd.reshape(c, (-1,)) for c in (x, y, z, *nrm, r, g, b, sd.full(tuple(Zs.shape), a))]
                vert.append(bytes(memoryview(sd.reshape(sd.stack(cols, axis=1), (-1,)))))
                count = cols[0].shape[0]
                idx = sd.reshape(sd.arange(0, count), tuple(Zs.shape)) + base
                p00, p01, p10, p11 = (
                    sd.reshape(q, (-1,))
                    for q in (idx[..., :-1, :-1], idx[..., :-1, 1:], idx[..., 1:, :-1], idx[..., 1:, 1:])
                )
                cells = sd.stack([p00, p01, p11, p00, p11, p10], axis=1)
                if keep is not None:
                    cells = cells[sd.reshape(keep, (-1,))]
                tri.append(bytes(memoryview(sd.reshape(cells, (-1,)))))
                base += count
            if contours:
                # the level curves of the first grid on the floor of the box, coloured as the surface is
                grid = _field.Grid(X[0, :].tolist(), Y[:, 0].tolist(), Z)
                for v in _ticks.linear(lo[2], hi[2], contours):
                    if not lo[2] < v < hi[2]:
                        continue
                    cx, cy = _field.Cells(None, None, None, None, v, "surface", polish=False, grid=grid).arrays()
                    m = cx.shape[0]
                    if m == 0:
                        continue
                    (r, g, b), _ = colours(sd.full((m,), v))
                    full = sd.stack(
                        [box(0, cx), box(1, cy), sd.full((m,), -BOX[2]), r, g, b, sd.full((m,), 1.0)], axis=1
                    )
                    lines.append(bytes(memoryview(sd.reshape(full, (-1,)))))
                    counts.append(m)
                    widths.append(CONTOUR_WIDTH * k)
            if not mesh:
                continue
            mesh_alpha = MESH_ALPHA * (0.4 + 0.6 * alpha)
            if fine is not None:
                (lx, ly, lz), cnts = fine[2]
                (r, g, b), _ = colours(lz)
                full = sd.stack(
                    [box(0, lx), box(1, ly), box(2, lz), r * MESH_DARK, g * MESH_DARK, b * MESH_DARK]
                    + [sd.full((lz.shape[0],), mesh_alpha)],
                    axis=1,
                )
                lines.append(bytes(memoryview(sd.reshape(full, (-1,)))))
                counts += cnts
                widths += [MESH_WIDTH * k] * len(cnts)
                continue
            ny, nx = X.shape
            x, y, z = box(0, X), box(1, Y), box(2, Z)
            (r, g, b), _ = colours(Z)
            sy, sx = mesh_steps(ny, nx, mesh)
            for rows, n_each in (
                (sd.stack([x[::sy, :], y[::sy, :], z[::sy, :]], axis=2), nx),
                (sd.stack([x[:, ::sx].T, y[:, ::sx].T, z[:, ::sx].T], axis=2), ny),
            ):
                m = rows.shape[0]
                pts = sd.reshape(rows, (-1, 3))
                colour = [
                    sd.reshape(q, (-1,)) * MESH_DARK
                    for q in ((c[::sy, :] if n_each == nx else c[:, ::sx].T) for c in (r, g, b))
                ]
                colour.append(sd.full((m * n_each,), mesh_alpha))
                full = sd.stack([pts[:, 0], pts[:, 1], pts[:, 2], *colour], axis=1)
                lines.append(bytes(memoryview(sd.reshape(full, (-1,)))))
                counts += [n_each] * m
                widths += [MESH_WIDTH * k] * m
        for x, y, z, rgba, _width, _label in self.curves:
            x, y, z = _clip(x, y, z, lo, hi)
            n = x.shape[0]
            colour = [sd.full((n,), c) for c in rgba]
            lines.append(
                bytes(memoryview(sd.reshape(sd.stack([box(0, x), box(1, y), box(2, z), *colour], axis=1), (-1,))))
            )
            counts.append(n)
            widths.append((1.0 if _width is None else _width) * k)
        for x, y, z, rgba, size, _label in self.points:
            n = x.shape[0]
            rad = (th["marker_size"] if size is None else size) / 2 * k
            cols = [box(0, x), box(1, y), box(2, z), sd.full((n,), rad), *[sd.full((n,), c) for c in rgba]]
            discs.append(bytes(memoryview(sd.reshape(sd.stack(cols, axis=1), (-1,)))))
        return b"".join(vert), b"".join(tri), b"".join(lines), counts, widths, b"".join(discs)


def _clip(x, y, z, lo, hi):
    """A curve cut at the box [lo, hi]: the samples outside become NaN (the line stops) and
    where the curve leaves or enters, the point on the box's face is put in."""
    P = (x, y, z)
    inside = sd.isfinite(x) & sd.isfinite(y) & sd.isfinite(z)
    for k in range(3):
        inside = inside & (P[k] >= lo[k]) & (P[k] <= hi[k])
    if bool(sd.all(inside)):
        return x, y, z
    flag = bytes(memoryview(inside))
    X, Y, Z = (c.tolist() for c in P)
    out = ([], [], [])
    nan = math.nan

    def face(i, j):
        """The point where the segment from the inside sample i to the outside sample j leaves the box."""
        a, b = (X[i], Y[i], Z[i]), (X[j], Y[j], Z[j])
        if not all(math.isfinite(v) for v in b):
            return None
        s = 1.0
        for k in range(3):
            d = b[k] - a[k]
            if d > 0.0 and b[k] > hi[k]:
                s = min(s, (hi[k] - a[k]) / d)
            elif d < 0.0 and b[k] < lo[k]:
                s = min(s, (lo[k] - a[k]) / d)
        return tuple(min(max(a[k] + s * (b[k] - a[k]), lo[k]), hi[k]) for k in range(3))

    n = len(flag)
    for i in range(n):
        if flag[i]:
            if i > 0 and not flag[i - 1]:
                p = face(i, i - 1)
                if p is not None:
                    for k in range(3):
                        out[k].append(p[k])
            out[0].append(X[i])
            out[1].append(Y[i])
            out[2].append(Z[i])
            if i + 1 < n and not flag[i + 1]:
                p = face(i, i + 1)
                if p is not None:
                    for k in range(3):
                        out[k].append(p[k])
        elif i > 0 and flag[i - 1]:
            for k in range(3):
                out[k].append(nan)
    return tuple(sd.asarray(c) if c else sd.zeros(0) for c in out)


def mesh_steps(ny, nx, count):
    """Every how many rows and columns of the grid a mesh line runs, for `count` cells of the mesh a side."""
    return max(1, round((ny - 1) / count)), max(1, round((nx - 1) / count))


def _mesh(who, mesh):
    """The cells of the mesh a side: True the default, False (0) none, or an int."""
    if isinstance(mesh, bool):
        return MESH if mesh else 0
    if not isinstance(mesh, int):
        raise TypeError(f"{who}(): 'mesh' must be a bool or an int, not {_type_name(mesh)}")
    if not 1 <= mesh <= 400:
        raise ValueError(f"{who}(): 'mesh' must be between 1 and 400")
    return mesh


def _contours(who, contours):
    """The levels of the floor contours aimed at: True the default, False (0) none, or an int."""
    if isinstance(contours, bool):
        return CONTOURS if contours else 0
    if not isinstance(contours, int):
        raise TypeError(f"{who}(): 'contours' must be a bool or an int, not {_type_name(contours)}")
    if not 1 <= contours <= 100:
        raise ValueError(f"{who}(): 'contours' must be between 1 and 100")
    return contours


def _fine(fn, X, Y, P, zlim, mesh, graph, levels):
    """What adaptive sampling adds to a surface's first grid (None: nothing): the cells that
    stay, the finer grids, and the mesh lines."""
    lo, hi = [], []
    for c in P:
        r = _layers.finite_range(sd.reshape(c, (-1,)))
        lo.append(r[0] if r else 0.0)
        hi.append(r[1] if r else 1.0)
    zcut = None
    if graph:
        r = _sample.robust_range(sd.reshape(P[2], (-1,)))
        zcut = zlim if zlim is not None else (r[0], r[1]) if (r[2] or r[3]) else None
        if zcut is not None:
            lo[2], hi[2] = zcut
        scale = [2.0 * b / max(hi[k] - lo[k], 1e-300) for k, b in enumerate(BOX_GRAPH)]
    else:
        scale = [2.0 / max(max(hi[k] - lo[k] for k in range(3)), 1e-300)] * 3
    out = _adapt.refine(fn, X, Y, P, scale, zcut, levels)
    if out is None or out is _adapt.ROUGH:
        return out
    keep, parts, asked = out
    return keep, parts, _adapt.mesh_lines(fn, X, Y, asked, *mesh_steps(*X.shape, mesh)) if mesh else None


def _rot(azimuth, elevation):
    """(right, up, eye): the rows of the rotation from the scene to the view."""
    a, e = math.radians(azimuth), math.radians(elevation)
    eye = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    right = (-math.sin(a), math.cos(a), 0.0)
    up = (
        eye[1] * right[2] - eye[2] * right[1],
        eye[2] * right[0] - eye[0] * right[2],
        eye[0] * right[1] - eye[1] * right[0],
    )
    return right, up, eye


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _shade(name, t):
    """(r, g, b) arrays of a colour map at t in [0, 1] (the anchors, interpolated)."""
    anchors = name if isinstance(name, list) else [_color._hex(h) for h in _color.CMAPS[name]]
    n = len(anchors)
    u = sd.where(sd.isfinite(t), t, 0.0) * (n - 1)
    i = sd.clip(sd.floor(u), min=0.0, max=n - 2.0)
    f = u - i
    ii = i.astype(sd.int64)
    out = []
    for c in range(3):
        col = sd.asarray([a[c] for a in anchors])
        lo, hi = (
            sd.take(sd.reshape(col, (-1,)), sd.reshape(ii, (-1,))),
            sd.take(sd.reshape(col, (-1,)), sd.reshape(ii + 1, (-1,))),
        )
        out.append(
            sd.reshape(lo, tuple(t.shape)) + (sd.reshape(hi, tuple(t.shape)) - sd.reshape(lo, tuple(t.shape))) * f
        )
    return out


def _space(self, who, azimuth, elevation, perspective):
    """The cell's Space (made with this call's camera when it is the first 3D layer)."""
    cell = self._cell
    az = None if azimuth is None else real(azimuth, who, "azimuth")
    el = None if elevation is None else real(elevation, who, "elevation")
    if el is not None and not -90.0 <= el <= 90.0:
        raise ValueError(f"{who}(): 'elevation' must be in [-90, 90]")
    _check_bool(perspective, who, "perspective")
    for layer in cell.layers:
        if isinstance(layer, Space):
            return layer
    if cell.layers:
        raise ValueError(f"{who}(): the cell already holds a 2D plot")
    sp = Space(az, el, perspective)
    cell.layers.append(sp)
    cell.frame = False
    sp.cell = cell
    return sp


class Auto:
    """The colour of a surface that was given none: decided when the cell is drawn, by how many
    surfaces it holds. Alone it is coloured by height (LONE);
    among several each takes its palette colour whole, a little lightened (the light does the
    modelling), so that surfaces drawn together stay apart."""

    def __init__(self, rgba):
        self.rgba = rgba


def _colour(self, who, color, cmap):
    """A surface's colours: one colour, a map by height, or Auto."""
    if color is not None and cmap is not None:
        raise ValueError(f"{who}(): 'color' and 'cmap' cannot both be given")
    palette = self._root._palette
    if color is not None:
        return _color.parse(color, who, "color", palette)
    if cmap is not None:
        return _field._cmap(who, cmap)
    return Auto(self._cell.next_color(palette))


def _alpha(who, alpha):
    alpha = real(alpha, who, "alpha")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"{who}(): 'alpha' must be in (0, 1]")
    return alpha


def surface(
    self,
    f,
    x0,
    x1,
    y0,
    y1,
    /,
    *,
    zlim=None,
    mesh=True,
    contours=False,
    color=None,
    cmap=None,
    alpha=1.0,
    samples=None,
    refine=None,
    azimuth=None,
    elevation=None,
    perspective=True,
):
    who, graph = "surface", True
    given = _field.given(f, who, samples, refine)
    x0, x1, y0, y1 = _field._box(who, x0, x1, y0, y1)
    mesh = _mesh(who, mesh)
    spec, alpha = _colour(self, who, color, cmap), _alpha(who, alpha)
    if zlim is not None:
        if not isinstance(zlim, (tuple, list)) or len(zlim) != 2:
            raise TypeError(f"{who}(): 'zlim' must be a pair (low, high), not {_type_name(zlim)}")
        zlim = _interval(who, zlim[0], zlim[1], "zlim[0]", "zlim[1]")
    if given is not None:  # the heights given on an even grid over the box
        ny, nx = given.shape
        X, Y = sd.meshgrid(sd.linspace(x0, x1, nx), sd.linspace(y0, y1, ny), indexing="xy")
        sp = _space(self, who, azimuth, elevation, perspective)
        sp.surfaces.append((X, Y, given, spec, mesh, alpha, None, _contours(who, contours)))
        sp.equal = False
        if zlim is not None:
            sp.zlim = zlim
        return self._touch()
    (nx, ny), levels = _sample.knobs(who, samples, refine, (N + 1, N + 1), 2, pair=True, most=2000)
    X, Y = sd.meshgrid(sd.linspace(x0, x1, nx), sd.linspace(y0, y1, ny), indexing="xy")

    def fn(U, V):
        return U, V, _field.call2(f, U, V, who)

    Z = fn(X, Y)[2]
    fine = _fine(fn, X, Y, (X, Y, Z), zlim, mesh, True, levels)
    if fine is _adapt.ROUGH:  # rough everywhere: the whole grid finer, once
        mx, my = (nx - 1) * _adapt.FINER + 1, (ny - 1) * _adapt.FINER + 1
        X, Y = sd.meshgrid(sd.linspace(x0, x1, mx), sd.linspace(y0, y1, my), indexing="xy")
        Z, fine = fn(X, Y)[2], None
    sp = _space(self, who, azimuth, elevation, perspective)
    sp.surfaces.append((X, Y, Z, spec, mesh, alpha, fine, _contours(who, contours) if graph else 0))
    sp.equal = False  # a graph: the height has its own scale
    if zlim is not None:
        sp.zlim = zlim
    return self._touch()


def parametric_surface(
    self,
    fx,
    fy,
    fz,
    u0,
    u1,
    v0,
    v1,
    /,
    *,
    mesh=True,
    color=None,
    cmap=None,
    alpha=1.0,
    samples=None,
    refine=None,
    azimuth=None,
    elevation=None,
    perspective=True,
):
    who, graph, contours = "parametric_surface", False, False
    for g, name in ((fx, "fx"), (fy, "fy"), (fz, "fz")):
        _callable(g, who, name)
    u0, u1 = _interval(who, u0, u1, "u0", "u1")
    v0, v1 = _interval(who, v0, v1, "v0", "v1")
    mesh = _mesh(who, mesh)
    spec, alpha = _colour(self, who, color, cmap), _alpha(who, alpha)
    (nu, nv), levels = _sample.knobs(who, samples, refine, (N + 1, N + 1), 2, pair=True, most=2000)
    U, V = sd.meshgrid(sd.linspace(u0, u1, nu), sd.linspace(v0, v1, nv), indexing="xy")

    def fn(U, V):
        return tuple(_field.call2(g, U, V, who, name) for g, name in ((fx, "fx"), (fy, "fy"), (fz, "fz")))

    X, Y, Z = fn(U, V)
    fine = _fine(fn, U, V, (X, Y, Z), None, mesh, False, levels)
    if fine is _adapt.ROUGH:  # rough everywhere: the whole grid finer, once
        mu, mv = (nu - 1) * _adapt.FINER + 1, (nv - 1) * _adapt.FINER + 1
        U, V = sd.meshgrid(sd.linspace(u0, u1, mu), sd.linspace(v0, v1, mv), indexing="xy")
        (X, Y, Z), fine = fn(U, V), None
    sp = _space(self, who, azimuth, elevation, perspective)
    sp.surfaces.append((X, Y, Z, spec, mesh, alpha, fine, _contours(who, contours) if graph else 0))
    return self._touch()


def curve3d(
    self,
    fx,
    fy,
    fz,
    a,
    b,
    /,
    *,
    samples=None,
    refine=None,
    color=None,
    width=None,
    label=None,
    azimuth=None,
    elevation=None,
    perspective=True,
):
    who = "curve3d"
    for g, name in ((fx, "fx"), (fy, "fy"), (fz, "fz")):
        _callable(g, who, name)
    a, b = _interval(who, a, b)
    width = _opt_real(who, "width", width)
    c = _basic._color_of(self, who, color)
    sp = _space(self, who, azimuth, elevation, perspective)

    def at(t):
        return tuple(call(g, t, who, name) for g, name in ((fx, "fx"), (fy, "fy"), (fz, "fz")))

    # Sampled as a plane curve is (_sample): on the screen of this cell's camera, with the bulk of
    # the first samples as the view, so the samples close in on the turns and the curve stops at a pole.
    n, depth = _sample.knobs(who, samples, refine, _sample.N0 + 1, _sample.DEPTH, most=200000)
    t0 = _sample.first_grid(a, b, n - 1)
    first = at(t0)
    rs = [_sample.robust_range(v) for v in first]
    mid = [0.5 * (r[0] + r[1]) for r in rs]
    span = [max(r[1] - r[0], 1e-300) for r in rs]
    if sp.equal:
        span = [max(span)] * 3
    right, up, _ = sp.rotation(usual=True)

    def screen(P):
        q = [(P[k] - mid[k]) / span[k] for k in range(3)]
        return (
            right[0] * q[0] + right[1] * q[1] + right[2] * q[2],
            up[0] * q[0] + up[1] * q[1] + up[2] * q[2],
        )

    view = (-0.9, 0.9)
    found = _sample.refine(lambda t: screen(at(t)), t0, *screen(first), view, view, depth)
    x, y, z = _sample.cut(found, at(found.t))
    sp.curves.append((x, y, z, c, width, _basic._label(who, label)))
    return self._touch()


def scatter3d(
    self,
    x,
    y,
    z,
    /,
    *,
    data=None,
    group=None,
    color=None,
    size=None,
    label=None,
    azimuth=None,
    elevation=None,
    perspective=True,
):
    who = "scatter3d"
    _basic._check_data(who, data)
    X, Y, Z = numbers_(who, "x", x, data), numbers_(who, "y", y, data), numbers_(who, "z", z, data)
    n = same_length(who, x=X, y=Y, z=Z)
    size = _opt_real(who, "size", size)
    parts = _basic._each(self, who, n, group, data, color, _basic._label(who, label))
    sp = _space(self, who, azimuth, elevation, perspective)
    for idx, col, lab in parts:
        sp.points.append(
            (
                _basic._take(X, idx),
                _basic._take(Y, idx),
                _basic._take(Z, idx),
                _basic._color_of(self, who, col),
                size,
                lab,
            )
        )
    return self._touch()


METHODS = {"surface": surface, "parametric_surface": parametric_surface, "curve3d": curve3d, "scatter3d": scatter3d}
