"""Adaptive sampling of a surface (docs/design/plot.md §6).

A surface is first sampled on a regular grid. The cells of that grid where the surface is not
well told by four corners are sampled again, K x K cells each, and the cells of those that still
are not, once more: the cells where the surface bends (its normals at the corners differ by more
than BEND), the cells the cut of the height range passes through (the flank of a pole), and the
cells at the edge of the function's domain (some corners finite, some not). Every level is one
call of the function with all its new points, so a smooth surface costs what it did and a hard
one a few calls more.

A finer cell beside a coarser one would leave a crack along their edge: the finer cell's points
on that edge are moved onto the coarser cell's straight edge.
"""

import math

from .. import _core as sd

K = 4  # a cell that is sampled again becomes K x K cells
BEND = math.cos(math.radians(10.0))  # corner normals further apart than this ask for more samples
MAX_FIRST = 0.4  # a surface with more than this share of its cells asking is rough everywhere
ROUGH = "rough"  # ... and is sampled FINER times finer as a whole instead
FINER = 3
MAX_SECOND = 24000  # cells of the second level at most
SHORE = 12  # bisections towards the edge of the domain
DENSE = K * K  # points of a mesh line per cell where the surface was sampled again


def _d(a, axis):
    """Differences along one of the last two axes: central, one-sided at the ends and beside a
    point without a value."""
    if axis == -1:
        f = a[..., 1:] - a[..., :-1]
        fwd, bwd = sd.concat([f, f[..., -1:]], axis=a.ndim - 1), sd.concat([f[..., :1], f], axis=a.ndim - 1)
    else:
        f = a[..., 1:, :] - a[..., :-1, :]
        fwd, bwd = sd.concat([f, f[..., -1:, :]], axis=a.ndim - 2), sd.concat([f[..., :1, :], f], axis=a.ndim - 2)
    c = 0.5 * (fwd + bwd)
    return sd.where(sd.isfinite(c), c, sd.where(sd.isfinite(fwd), fwd, bwd))


def normals(x, y, z):
    """Vertex normals (not unit) of grids whose last two axes run along the surface."""
    du = [_d(c, -1) for c in (x, y, z)]
    dv = [_d(c, -2) for c in (x, y, z)]
    return du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]


def _corners(a):
    return a[..., :-1, :-1], a[..., :-1, 1:], a[..., 1:, :-1], a[..., 1:, 1:]


def asks(P, scale, zcut):
    """The cells (bool, one per cell of the grids P = (X, Y, Z)) that need more samples. `scale`
    brings the three coordinates to the box's units; `zcut` is the height range the picture
    keeps (low, high) or None."""
    x, y, z = (c * s for c, s in zip(P, scale))
    fin = sd.isfinite(x) & sd.isfinite(y) & sd.isfinite(z)
    f = _corners(fin)
    every, some = f[0] & f[1] & f[2] & f[3], f[0] | f[1] | f[2] | f[3]
    nx, ny, nz = normals(x, y, z)
    ln = sd.sqrt(nx * nx + ny * ny + nz * nz)
    n = [_corners(c / ln) for c in (nx, ny, nz)]

    def dot(i, j):
        return n[0][i] * n[0][j] + n[1][i] * n[1][j] + n[2][i] * n[2][j]

    # the sign of a normal is the grid's, the same at the four corners: no absolute value
    bend = (dot(0, 3) < BEND) | (dot(1, 2) < BEND) | (dot(0, 1) < BEND) | (dot(0, 2) < BEND)
    out = some & ~every | every & bend
    if zcut is not None:
        zc = _corners(P[2])
        low = sd.minimum(sd.minimum(zc[0], zc[1]), sd.minimum(zc[2], zc[3]))
        high = sd.maximum(sd.maximum(zc[0], zc[1]), sd.maximum(zc[2], zc[3]))
        cut = (low < zcut[0]) & (high > zcut[0]) | (low < zcut[1]) & (high > zcut[1])
        out = (out | every & cut) & ~(every & ((low > zcut[1]) | (high < zcut[0])))  # nothing to see outside
    return out


def _children(U, V, pick):
    """The parameters of the K x K cells of the picked cells: two arrays (m, K + 1, K + 1)."""
    t = sd.linspace(0.0, 1.0, K + 1)
    u = _corners(U)
    v = _corners(V)
    u00, u01, v00, v10 = u[0][pick], u[1][pick], v[0][pick], v[2][pick]
    m = u00.shape[0]
    uu = sd.reshape(u00, (m, 1, 1)) * (1.0 - t)[None, None, :] + sd.reshape(u01, (m, 1, 1)) * t[None, None, :]
    vv = sd.reshape(v00, (m, 1, 1)) * (1.0 - t)[None, :, None] + sd.reshape(v10, (m, 1, 1)) * t[None, :, None]
    one = sd.full((m, K + 1, K + 1), 1.0)
    return uu * one, vv * one


def _call(fn, U, V):
    m = U.shape[0]
    flat = fn(sd.reshape(U, (m * (K + 1), K + 1)), sd.reshape(V, (m * (K + 1), K + 1)))
    return [sd.reshape(c, (m, K + 1, K + 1)) for c in flat]


def _straighten(P, up, down, left, right):
    """Moves the points of the edges named by the bool arrays (one per patch) onto the straight
    line between the edge's corners: the edge of a coarser neighbour."""
    t = sd.linspace(0.0, 1.0, K + 1)[None, :]
    out = []
    for c in P:
        c = c.copy()
        for side, row in ((up, (slice(None), 0, slice(None))), (down, (slice(None), K, slice(None)))):
            line = c[:, row[1], 0:1] * (1.0 - t) + c[:, row[1], K : K + 1] * t
            c[row] = sd.where(side[:, None], line, c[row])
        for side, col in ((left, 0), (right, K)):
            line = c[:, 0:1, col] * (1.0 - t) + c[:, K : K + 1, col] * t
            c[:, :, col] = sd.where(side[:, None], line, c[:, :, col])
        out.append(c)
    return out


def _finite(P):
    return sd.isfinite(P[0]) & sd.isfinite(P[1]) & sd.isfinite(P[2])


def _shore(fn, U, V, P, bounds):
    """Moves the points where the function has no value onto the edge of its domain: each such
    point with a neighbour (one step along u or v) that has a value goes to the last point with
    a value on the way from that neighbour, found by SHORE bisections. The neighbours are asked
    of the function itself, in a fixed order, so two patches that share a point agree on it."""
    bad = ~_finite(P)
    n = int(sd.sum(bad))
    if n == 0:
        return P
    u, v = sd.reshape(U[bad], (n, 1)), sd.reshape(V[bad], (n, 1))
    hu = sd.reshape((U[:, :, 1:2] - U[:, :, 0:1] + 0.0 * U)[bad], (n, 1))
    hv = sd.reshape((V[:, 1:2, :] - V[:, 0:1, :] + 0.0 * V)[bad], (n, 1))
    au, av = u.copy(), v.copy()
    found = sd.full((n, 1), False)
    for du, dv in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
        cu = sd.clip(u + du * hu, min=bounds[0], max=bounds[1])
        cv = sd.clip(v + dv * hv, min=bounds[2], max=bounds[3])
        take = _finite(fn(cu, cv)) & ~found
        au, av, found = sd.where(take, cu, au), sd.where(take, cv, av), found | take
    if not bool(sd.any(found)):
        return P
    bu, bv = u, v
    for _ in range(SHORE):
        mu, mv = 0.5 * (au + bu), 0.5 * (av + bv)
        ok = _finite(fn(mu, mv))
        au, av = sd.where(ok, mu, au), sd.where(ok, mv, av)
        bu, bv = sd.where(ok, bu, mu), sd.where(ok, bv, mv)
    Q = fn(au, av)
    out = []
    for c, q in zip(P, Q):
        c = c.copy()
        c[bad] = sd.reshape(sd.where(found, q, math.nan), (n,))
        out.append(c)
    return out


def _padded(mask):
    """The mask with a border of True: outside the grid nothing is coarser."""
    ny, nx = mask.shape
    out = sd.full((ny + 2, nx + 2), True)
    out[1:-1, 1:-1] = mask
    return out


def refine(fn, U, V, P, scale, zcut, levels=2):
    """Samples again where the first grid asks for it. fn(U, V) -> (X, Y, Z) takes and returns 2-D
    arrays. Returns None when nothing asks, ROUGH when more than MAX_FIRST of the cells do (the
    caller samples the whole grid finer instead), else (keep, parts, asked):
    `keep` the cells of the first grid that stay, `parts` a list of ((X, Y, Z), keep) of grids
    (m, K + 1, K + 1) with the cells of each that are drawn (None: all), and `asked` the cells
    of the first grid that were sampled again."""
    if levels == 0:
        return None
    ask = asks(P, scale, zcut)
    count = int(sd.sum(ask))
    ny, nx = ask.shape
    if count == 0:
        return None
    if count > MAX_FIRST * ny * nx:
        return ROUGH
    i1, j1 = sd.nonzero(ask)
    U1, V1 = _children(U, V, ask)
    P1 = _call(fn, U1, V1)
    pad = _padded(ask)
    P1 = _straighten(P1, ~pad[i1, j1 + 1], ~pad[i1 + 2, j1 + 1], ~pad[i1 + 1, j1], ~pad[i1 + 1, j1 + 2])
    bounds = (float(sd.min(U)), float(sd.max(U)), float(sd.min(V)), float(sd.max(V)))
    ask2 = asks(P1, scale, zcut)
    count2 = int(sd.sum(ask2))
    if levels > 1 and 0 < count2 <= MAX_SECOND:
        p, a, b = sd.nonzero(ask2)
        U2, V2 = _children(U1, V1, ask2)
        P2 = _call(fn, U2, V2)
        for c2, c1 in zip(P2, P1):  # the corners are the parent's (an edge of it may have been moved)
            c2[:, 0, 0], c2[:, 0, K] = c1[p, a, b], c1[p, a, b + 1]
            c2[:, K, 0], c2[:, K, K] = c1[p, a + 1, b], c1[p, a + 1, b + 1]
        fine = sd.full((ny * K, nx * K), False)
        gi, gj = i1[p] * K + a, j1[p] * K + b
        fine[gi, gj] = True
        pad = _padded(fine)
        P2 = _straighten(P2, ~pad[gi, gj + 1], ~pad[gi + 2, gj + 1], ~pad[gi + 1, gj], ~pad[gi + 1, gj + 2])
        return ~ask, [(P1, ~ask2), (_shore(fn, U2, V2, P2, bounds), None)], ask
    return ~ask, [(_shore(fn, U1, V1, P1, bounds), None)], ask


def mesh_lines(fn, U, V, asked, step_rows, step_cols):
    """The mesh lines of a surface that was sampled again: along the rows and columns of the
    first grid, with DENSE points per cell where a cell beside the line was sampled again.
    Returns ((X, Y, Z) of the points, one after another), and the points of each line."""
    ny, nx = asked.shape
    xs, counts = [], []
    for rows in (True, False):
        A = asked if rows else asked.T
        Uc, Vc = (U, V) if rows else (U.T, V.T)  # the line runs along the last axis
        n_lines, n_cells = A.shape[0] + 1, A.shape[1]
        which = list(range(0, n_lines, step_rows if rows else step_cols))
        idx = sd.asarray(which)
        pad = sd.concat([sd.full((1, n_cells), False), A, sd.full((1, n_cells), False)], axis=0)
        beside = pad[idx] | pad[idx + 1]  # (lines, cells): a cell above or below was sampled again
        t = sd.linspace(0.0, 1.0, DENSE + 1)[:-1]
        u0, u1 = Uc[idx][:, :-1], Uc[idx][:, 1:]
        v0, v1 = Vc[idx][:, :-1], Vc[idx][:, 1:]
        uu = sd.reshape(u0[:, :, None] * (1.0 - t) + u1[:, :, None] * t, (len(which), n_cells * DENSE))
        vv = sd.reshape(v0[:, :, None] * (1.0 - t) + v1[:, :, None] * t, (len(which), n_cells * DENSE))
        uu = sd.concat([uu, Uc[idx][:, -1:]], axis=1)
        vv = sd.concat([vv, Vc[idx][:, -1:]], axis=1)
        first = sd.full((DENSE,), False)
        first[0] = True
        take = sd.reshape(beside[:, :, None] | first[None, None, :], (len(which), n_cells * DENSE))
        take = sd.concat([take, sd.full((len(which), 1), True)], axis=1)
        P = fn(uu, vv)
        xs.append([c[take] for c in P])
        counts += [int(c) for c in sd.sum(take, axis=1).tolist()]
    return [sd.concat([a[k] for a in xs]) for k in range(3)], counts
