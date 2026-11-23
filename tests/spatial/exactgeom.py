"""Exact computational geometry for the tests of sd.spatial's Delaunay / ConvexHull / Voronoi
(docs/design/spatial_geometry.md §7.3): every float64 is read as a rational (`fractions.Fraction`) and
the tie rule's infinitesimal is a polynomial variable (python-flint's `fmpq_poly`). Nothing here uses
floating point, so the functions are the definitions the C++ predicates and structures are tested
against. Slow by design: points in the thousands, exhaustive references in the tens.

- predicates: `orient`, `in_sphere` (by the circumcentre: the geometric definition), `lifted` (the
  determinant form), `in_sphere_eps` (the tie rule of §3.2 as a polynomial in eps), `sos_rule` (the
  closed form predicates.hpp implements);
- references: `delaunay_reference` (every (d + 1)-subset whose perturbed sphere is empty),
  `extreme_points`, `voronoi_ridge_exists`, `voronoi_reference`;
- checkers of §4: `check_delaunay` (1), `check_perturbed` (2), `check_find_simplex` (4), `check_hull` (5),
  `check_voronoi` (6); `check_delaunay3_large` (1 and 2 for thousands of 3-D points: a float filter whose open
  rows go to the same exact predicates; numpy).
"""

import itertools
from fractions import Fraction

import flint
import numpy as np


def frac(p):
    """A point as a tuple of Fractions (float64 values convert exactly)."""
    return tuple(Fraction(float(v)) if not isinstance(v, Fraction) else v for v in p)


def fpoints(points):
    return [frac(p) for p in points]


def sign(v) -> int:
    return (v > 0) - (v < 0)


def det(m):
    """Determinant of a small square matrix over any ring (Laplace expansion along the first row)."""
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    total = 0
    for j in range(n):
        minor = [row[:j] + row[j + 1 :] for row in m[1:]]
        term = m[0][j] * det(minor)
        total = total + term if j % 2 == 0 else total - term
    return total


def orient_value(*p):
    """det[p1 - p0, ..., pd - p0] of d + 1 points in d dimensions (§2.2: positive for a positive simplex)."""
    return det([[b - a for a, b in zip(p[0], q)] for q in p[1:]])


def orient(*p) -> int:
    return sign(orient_value(*p))


def circumcenter(*p):
    """The centre of the sphere through d + 1 affinely independent points."""
    d = len(p[0])
    rows = [[2 * (q[k] - p[0][k]) for k in range(d)] for q in p[1:]]
    rhs = [sum(q[k] ** 2 - p[0][k] ** 2 for k in range(d)) for q in p[1:]]
    den = det(rows)
    if den == 0:
        raise ValueError("degenerate simplex")
    out = []
    for k in range(d):
        m = [row[:k] + [rhs[i]] + row[k + 1 :] for i, row in enumerate(rows)]
        out.append(det(m) / den)
    return tuple(out)


def dist2(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def in_sphere(*p) -> int:
    """The last point against the sphere through the first d + 1, by the circumcentre: +1 strictly
    inside, 0 on it, -1 outside, times the orientation of the simplex (the sign convention of
    predicates.hpp's incircle / insphere)."""
    simplex, q = p[:-1], p[-1]
    o = orient(*simplex)
    if o == 0:
        raise ValueError("degenerate simplex")
    c = circumcenter(*simplex)
    return o * sign(dist2(simplex[0], c) - dist2(q, c))


def _lifted_sign(d: int) -> int:
    # in_sphere = +lifted for d = 2 and -lifted for d = 3 (tested in test_exactgeom.py)
    return 1 if d == 2 else -1


def lifted(*p, heights=None):
    """sigma_d * det of the rows [coordinates, height, 1] (height |p|^2 by default): in_sphere's value form."""
    d = len(p[0])
    if heights is None:
        heights = [sum(v * v for v in q) for q in p]
    return _lifted_sign(d) * det([list(q) + [h, 1] for q, h in zip(p, heights)])


def lex_ranks(points):
    """rank[i] of point i among the DISTINCT coordinates: 1 for the lexicographically largest (§3.2)."""
    distinct = sorted(set(points), reverse=True)
    pos = {q: k + 1 for k, q in enumerate(distinct)}
    return [pos[q] for q in points]


def in_sphere_eps(*p, ranks=None) -> int:
    """in_sphere with the lift height of the point of rank k raised by eps^k (0 < eps, infinitesimal):
    the sign of the lowest-degree non-zero coefficient of the lifted determinant as a polynomial in eps.
    `ranks` are the points' ranks in the whole set (default: among the given points; only their order
    matters)."""
    if ranks is None:
        ranks = lex_ranks(list(p))
    eps = flint.fmpq_poly([0, 1])

    def poly(v):
        return flint.fmpq_poly([flint.fmpq(v.numerator, v.denominator)])

    d = len(p[0])
    rows = []
    for q, r in zip(p, ranks):
        h = poly(sum(v * v for v in q)) + eps**r
        rows.append([poly(v) for v in q] + [h, flint.fmpq_poly([1])])
    value = det(rows)
    for c in value.coeffs():
        if c != 0:
            return _lifted_sign(d) * (1 if c > 0 else -1)
    return 0


def sos_rule(*p) -> int:
    """The closed form of in_sphere_eps (predicates.hpp): in_sphere when it is not 0; otherwise, taking the
    points from the lexicographically largest down, the first non-zero (-1)^(i + d) * orient(the others),
    i the argument position."""
    s = in_sphere(*p)
    if s != 0:
        return s
    d = len(p[0])
    order = sorted(range(len(p)), key=lambda i: (p[i], -i), reverse=True)
    for i in order:
        o = orient(*(q for j, q in enumerate(p) if j != i))
        if o != 0:
            return o if (i + d) % 2 == 0 else -o
    return 0


# ---------------------------------------------------------------- references


def representatives(points):
    """rep[i]: the smallest index with point i's coordinates."""
    first = {}
    return [first.setdefault(q, i) for i, q in enumerate(points)]


def positive(simplex, points):
    """The canonical row of §2.2 for a set of d + 1 indices: positive orientation, smallest index first
    (d = 3: then the smallest of the other three second)."""
    s = sorted(simplex)
    if orient(*(points[i] for i in s)) < 0:
        s[-1], s[-2] = s[-2], s[-1]
    if len(s) == 4:
        a, b, c, d = s  # even permutations fixing a: (b, c, d), (c, d, b), (d, b, c)
        s = min([a, b, c, d], [a, c, d, b], [a, d, b, c], key=lambda r: r[1])
    else:
        a, b, c = s
        s = [a, b, c]
    return tuple(s)


def delaunay_reference(points, perturbed=True):
    """The Delaunay triangulation by definition: the canonical rows of every (d + 1)-subset of the
    distinct points that is not degenerate and has no other point inside its sphere, with the tie rule
    (perturbed=True: the unique triangulation of §3.2) or without (every simplex of every Delaunay
    triangulation: a superset when points are cospherical). O(n^(d+2)): n <= 40 (d = 2), 16 (d = 3)."""
    pts = fpoints(points)
    d = len(pts[0])
    rep = representatives(pts)
    idx = sorted(set(rep))
    ranks = lex_ranks(pts)
    out = set()
    for comb in itertools.combinations(idx, d + 1):
        sp = [pts[i] for i in comb]
        o = orient(*sp)
        if o == 0:
            continue
        c = circumcenter(*sp)
        r2 = dist2(sp[0], c)
        empty = True
        for k in idx:
            if k in comb:
                continue
            s = sign(r2 - dist2(pts[k], c))
            if s == 0 and perturbed:
                s = o * in_sphere_eps(*sp, pts[k], ranks=[ranks[i] for i in comb] + [ranks[k]])
            if s > 0:
                empty = False
                break
        if empty:
            out.add(positive(comb, pts))
    return sorted(out)


def _hull2(pts, idx):
    """Extreme points of 2-D points (indices `idx`, distinct coordinates): Andrew's monotone chain with
    strict turns, counterclockwise from the lexicographically smallest."""
    order = sorted(idx, key=lambda i: pts[i])
    if len(order) < 3:
        return order

    def chain(seq):
        out = []
        for i in seq:
            while len(out) >= 2 and orient(pts[out[-2]], pts[out[-1]], pts[i]) <= 0:
                out.pop()
            out.append(i)
        return out

    lower, upper = chain(order), chain(order[::-1])
    return lower[:-1] + upper[:-1]


def extreme_points(points):
    """The indices (representatives) of the vertices of the convex hull, sorted. d = 2: monotone chain.
    d = 3: by definition (every plane through three points with all points on one side supports the hull;
    the extreme points are the corners of the polygons on those planes): O(n^4), n <= 30."""
    pts = fpoints(points)
    d = len(pts[0])
    idx = sorted(set(representatives(pts)))
    if d == 2:
        return sorted(_hull2(pts, idx))
    out = set()
    for a, b, c in itertools.combinations(idx, 3):
        n = _cross(_sub(pts[b], pts[a]), _sub(pts[c], pts[a]))
        if not any(n):
            continue
        sides = [sign(_dot(n, _sub(pts[k], pts[a]))) for k in idx]
        if any(s > 0 for s in sides) and any(s < 0 for s in sides):
            continue
        on = [k for k, s in zip(idx, sides) if s == 0]
        drop = max(range(3), key=lambda k: abs(n[k]))  # project the plane on two axes
        keep = [k for k in range(3) if k != drop]
        flat = {k: (pts[k][keep[0]], pts[k][keep[1]]) for k in on}
        out.update(_hull2(flat, on))
    return sorted(out)


def _sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def _dot(p, q):
    return sum(a * b for a, b in zip(p, q))


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _strictly_feasible(rows):
    """Is {x : a.x > b for (a, b) in rows} non-empty? Fourier-Motzkin elimination (exact)."""
    rows = [(list(a), b) for a, b in rows]
    n = len(rows[0][0]) if rows else 0
    for k in range(n):
        pos, neg, rest = [], [], []
        for a, b in rows:
            (pos if a[k] > 0 else neg if a[k] < 0 else rest).append((a, b))
        for ap, bp in pos:
            for an, bn in neg:
                # x_k > (bp - ap'.x') / ap_k  and  x_k < (bn - an'.x') / an_k
                wp, wn = -an[k], ap[k]
                rest.append(([wp * u + wn * v for u, v in zip(ap, an)], wp * bp + wn * bn))
        rows = rest
    return all(b < 0 for a, b in rows)  # every remaining row reads 0 > b


def voronoi_ridge_exists(points, i, j) -> bool:
    """Do the Voronoi regions of the distinct points i and j share a (d - 1)-dimensional face? By
    definition: some x has |x - pi| = |x - pj| < |x - pk| for every other distinct point k."""
    pts = fpoints(points)
    d = len(pts[0])
    pi, pj = pts[i], pts[j]
    # the bisector 2 (pi - pj).x = |pi|^2 - |pj|^2: solve for the coordinate with the largest coefficient
    n = [2 * (a - b) for a, b in zip(pi, pj)]
    c = _dot(pi, pi) - _dot(pj, pj)
    e = max(range(d), key=lambda k: abs(n[k]))
    free = [k for k in range(d) if k != e]
    rows = []
    for pk in pts:
        if pk == pi or pk == pj:
            continue
        a = [2 * (u - v) for u, v in zip(pi, pk)]  # 2 (pi - pk).x > |pi|^2 - |pk|^2
        b = _dot(pi, pi) - _dot(pk, pk)
        # x_e = (c - sum n_f x_f) / n_e
        rows.append(([a[f] - a[e] * n[f] / n[e] for f in free], b - a[e] * c / n[e]))
    return _strictly_feasible(rows)


def _facets(simplex):
    """[(k, facet)]: the facet opposite vertex k, oriented so the simplex is on its negative side... as a
    sorted tuple key for matching."""
    return [(k, tuple(sorted(simplex[:k] + simplex[k + 1 :]))) for k in range(len(simplex))]


def adjacency(simplices):
    """{sorted facet: [(simplex, k)]} of a list of simplices (k: the opposite vertex's position)."""
    out = {}
    for t, s in enumerate(simplices):
        for k, f in _facets(tuple(int(v) for v in s)):
            out.setdefault(f, []).append((t, k))
    return out


def voronoi_reference(points, simplices):
    """The Voronoi diagram as the dual of a Delaunay triangulation (§3.7), exactly: clusters of simplices
    with one circumsphere, their centres, and the ridges.
    Returns (cluster of each simplex, [centre of each cluster], {(i, j): set of clusters, -1 if unbounded})
    with the zero-size ridges dropped (closed ones by the number of distinct vertices; unbounded ones in
    3-D by voronoi_ridge_exists)."""
    pts = fpoints(points)
    d = len(pts[0])
    simplices = [tuple(int(v) for v in s) for s in simplices]
    centres = [circumcenter(*(pts[i] for i in s)) for s in simplices]
    key = {}
    cluster = [key.setdefault(c, len(key)) for c in centres]
    cc = [None] * len(key)
    for t, c in enumerate(centres):
        cc[cluster[t]] = c
    adj = adjacency(simplices)
    ridges = {}
    for t, s in enumerate(simplices):
        for i, j in itertools.combinations(sorted(s), 2):
            ridges.setdefault((i, j), set()).add(cluster[t])
    for f, owners in adj.items():
        if len(owners) == 1:  # a boundary facet: every edge of it has an unbounded ridge
            for i, j in itertools.combinations(f, 2):
                ridges[(i, j)].add(-1)
    kept = {}
    for e, c in ridges.items():
        if -1 not in c:
            ok = len(c) >= d  # a closed ridge with fewer than d distinct vertices has no size
        elif d == 2:
            ok = True  # a ray or a line
        else:
            # d = 3: an unbounded ridge can still be a ray (the diagonal of a flat hull face: the two hull
            # triangles at the edge are coplanar), so the definition decides
            ok = voronoi_ridge_exists(points, *e)
        if ok:
            kept[e] = c
    return cluster, cc, kept


# ---------------------------------------------------------------- the checkers of §4


def _contains(sp, x):
    """Closed containment of x in a positive simplex: every facet has orient >= 0 with x in the vertex's place."""
    return all(orient(*(sp[:k] + [x] + sp[k + 1 :])) >= 0 for k in range(len(sp)))


def check_delaunay(points, simplices, neighbors=None, coplanar=None, exhaustive=60):
    """§4 item 1 (and the definition of `neighbors` when given). Raises AssertionError.
    - every simplex is positive; the vertices are exactly the representatives of the distinct points;
    - every facet belongs to one or two simplices; a facet of one has every point on its inner side (it is
      on the hull); the centroid of each simplex lies in no other simplex (no double cover; all pairs up
      to 400 simplices, a spread sample beyond); d = 2: the count is 2m - 2 - b;
    - empty sphere: across every inner facet the opposite vertex is not strictly inside (the local
      condition, which with the cover gives the global one), and for <= `exhaustive` distinct points every
      (simplex, point) pair."""
    pts = fpoints(points)
    d = len(pts[0])
    simplices = [tuple(int(v) for v in s) for s in simplices]
    rep = representatives(pts)
    distinct = sorted(set(rep))
    used = sorted({i for s in simplices for i in s})
    assert used == distinct, "the vertices are not the representatives of the distinct points"
    for s in simplices:
        assert len(set(s)) == d + 1 and orient(*(pts[i] for i in s)) > 0, f"simplex {s} is not positive"
    adj = adjacency(simplices)
    boundary_vertices = set()
    for f, owners in adj.items():
        assert len(owners) in (1, 2), f"facet {f} belongs to {len(owners)} simplices"
        if len(owners) == 1:
            t, k = owners[0]
            s = simplices[t]
            for q in distinct:
                trial = [pts[i] for i in s]
                trial[k] = pts[q]
                assert orient(*trial) >= 0, f"point {q} is outside the boundary facet {f}"
            boundary_vertices.update(f)
        else:
            (t, k), (t2, k2) = owners
            s, s2 = simplices[t], simplices[t2]
            sp = [pts[i] for i in s]
            assert in_sphere(*sp, pts[s2[k2]]) <= 0, f"simplices {s} and {s2} are not locally Delaunay"
            if neighbors is not None:
                assert int(neighbors[t][k]) == t2 and int(neighbors[t2][k2]) == t, f"neighbors of facet {f}"
    if neighbors is not None:
        for f, owners in adj.items():
            if len(owners) == 1:
                t, k = owners[0]
                assert int(neighbors[t][k]) == -1, f"boundary facet {f} has a neighbor"
    if d == 2:
        on_boundary = set(boundary_vertices)
        assert len(simplices) == 2 * len(distinct) - 2 - len(on_boundary), "wrong number of triangles"
    nt = len(simplices)
    sample = range(nt) if nt <= 400 else range(0, nt, max(1, nt // 200))
    for t in sample:
        sp = [pts[i] for i in simplices[t]]
        g = tuple(sum(c) / (d + 1) for c in zip(*sp))
        for t2, s2 in enumerate(simplices):
            if t2 != t:
                assert not _contains([pts[i] for i in s2], g), f"simplices {t} and {t2} overlap"
    if len(distinct) <= exhaustive:
        for s in simplices:
            sp = [pts[i] for i in s]
            for q in distinct:
                if q not in s:
                    assert in_sphere(*sp, pts[q]) <= 0, f"point {q} is inside the sphere of {s}"
    if coplanar is not None:
        rows = [tuple(int(v) for v in r) for r in coplanar]
        assert [r[0] for r in rows] == [i for i in range(len(pts)) if rep[i] != i], "coplanar's points"
        for i, t, v in rows:
            assert v == rep[i] and v in simplices[t], f"coplanar row {(i, t, v)}"


def check_perturbed(points, simplices):
    """§4 item 2 by the local condition: across every inner facet the opposite vertex is strictly outside
    the PERTURBED sphere (in_sphere_eps < 0). With check_delaunay this makes the triangulation the one of §3.2."""
    pts = fpoints(points)
    ranks = lex_ranks(pts)
    simplices = [tuple(int(v) for v in s) for s in simplices]
    for owners in adjacency(simplices).values():
        if len(owners) == 2:
            (t, k), (t2, k2) = owners
            s, q = simplices[t], simplices[t2][k2]
            v = in_sphere_eps(*(pts[i] for i in s), pts[q], ranks=[ranks[i] for i in s] + [ranks[q]])
            assert v < 0, f"simplex {s} against point {q}: perturbed in_sphere is {v}"


def find_simplex_reference(points, simplices, x):
    """The smallest simplex number whose closed simplex holds x, or -1 (§4 item 4)."""
    pts = fpoints(points)
    x = frac(x)
    for t, s in enumerate(simplices):
        if _contains([pts[int(i)] for i in s], x):
            return t
    return -1


def check_find_simplex(points, simplices, xi, found):
    for x, t in zip(xi, found):
        want = find_simplex_reference(points, simplices, x)
        assert int(t) == want, f"find_simplex({tuple(x)}) is {int(t)}, the reference is {want}"


def check_hull(points, vertices, simplices, reference=30):
    """§4 item 5. `vertices`: the extreme points; `simplices`: the boundary facets, outward (d = 2: edges
    (i, j) with the inside on the left; d = 3: counterclockwise seen from outside).
    - every point is on the inner side of every facet (or on it); the facets close up (d = 2: one cycle;
      d = 3: every edge once in each direction); the facets' vertices are exactly `vertices`;
    - for <= `reference` distinct points (any number for d = 2), `vertices` equals extreme_points()."""
    pts = fpoints(points)
    d = len(pts[0])
    simplices = [tuple(int(v) for v in s) for s in simplices]
    vertices = [int(v) for v in vertices]
    distinct = sorted(set(representatives(pts)))
    for f in simplices:
        fp = [pts[i] for i in f]
        assert len(set(fp)) == d, f"facet {f} is degenerate"
        for q in distinct:
            assert orient(*fp, pts[q]) * (1 if d == 2 else -1) >= 0, f"point {q} is outside facet {f}"
    if d == 2:
        nxt = dict(simplices)
        assert len(nxt) == len(simplices) and sorted(nxt.values()) == sorted(nxt), "the edges do not form cycles"
        seen, i = [], simplices[0][0]
        while i not in seen:
            seen.append(i)
            i = nxt[i]
        assert len(seen) == len(simplices), "the edges form more than one cycle"
    else:
        edges = [(f[k], f[(k + 1) % 3]) for f in simplices for k in range(3)]
        assert len(set(edges)) == len(edges), "an edge is used twice in one direction"
        assert all((j, i) in set(edges) for i, j in edges), "the surface is not closed"
        for f in simplices:
            assert any(_cross(_sub(pts[f[1]], pts[f[0]]), _sub(pts[f[2]], pts[f[0]]))), f"facet {f} is flat"
    assert sorted(set(vertices)) == sorted({i for f in simplices for i in f}), "vertices differ from the facets'"
    assert len(set(vertices)) == len(vertices), "a vertex is repeated"
    if d == 2 or len(distinct) <= reference:
        assert sorted(vertices) == extreme_points(points), "vertices are not the extreme points"


def check_voronoi(points, simplices, ridge_points, ridge_vertex_sets, simplex_vertex, definition=20):
    """§4 item 6 against voronoi_reference(points, simplices) (simplices: the Delaunay triangulation the
    diagram came from). `simplex_vertex[t]` is the Voronoi vertex number of simplex t, `ridge_points` the
    pairs (i, j), `ridge_vertex_sets[k]` the vertex numbers of ridge k (-1 for unbounded).
    For <= `definition` distinct points the ridges are also checked against voronoi_ridge_exists."""
    cluster, _, ridges = voronoi_reference(points, simplices)
    sv = [int(v) for v in simplex_vertex]
    pairs = {}
    for t, (c, v) in enumerate(zip(cluster, sv)):
        assert pairs.setdefault(c, v) == v, f"simplex {t}: cospherical simplices have different vertices"
    assert len(set(pairs.values())) == len(pairs), "simplices with different circumspheres share a vertex"
    got = {tuple(int(v) for v in e): {int(v) for v in vs} for e, vs in zip(ridge_points, ridge_vertex_sets)}
    want = {e: {(-1 if c == -1 else pairs[c]) for c in cs} for e, cs in ridges.items()}
    assert got == want, "the ridges differ from the reference"
    pts = fpoints(points)
    distinct = sorted(set(representatives(pts)))
    if len(distinct) <= definition:
        for i, j in itertools.combinations(distinct, 2):
            assert ((i, j) in got) == voronoi_ridge_exists(points, i, j), f"ridge {(i, j)} against the definition"


# ---------------------------------------------------------------- bulk signs for the predicate tests


def scaled_ints(row):
    """The points of one row as integer tuples: every coordinate times one power of two (the signs of the
    predicates do not change, and Python ints are much faster than Fractions)."""
    fr = [[Fraction(float(v)) for v in p] for p in row]
    den = max(v.denominator for p in fr for v in p)
    return [tuple(int(v * den) for v in p) for p in fr]


def predicate_sign(name: str, row) -> int:
    """The exact sign of predicates.hpp's `name` ('orient2d', 'orient3d', 'incircle', 'insphere') on a row."""
    p = scaled_ints(row)
    return orient(*p) if name.startswith("orient") else sign(lifted(*p))


# ---------------------------------------------------------------- a filtered exact checker for large 3-D sets


def _det3(A):
    """det and permanent of (k, 3, 3) float arrays, by the first row's cofactors."""
    m0 = A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1]
    m1 = A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0]
    m2 = A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0]
    B = np.abs(A)
    p0 = B[:, 1, 1] * B[:, 2, 2] + B[:, 1, 2] * B[:, 2, 1]
    p1 = B[:, 1, 0] * B[:, 2, 2] + B[:, 1, 2] * B[:, 2, 0]
    p2 = B[:, 1, 0] * B[:, 2, 1] + B[:, 1, 1] * B[:, 2, 0]
    return A[:, 0, 0] * m0 - A[:, 0, 1] * m1 + A[:, 0, 2] * m2, B[:, 0, 0] * p0 + B[:, 0, 1] * p1 + B[:, 0, 2] * p2


def _orient_signs(x, pts, R):
    """The exact orient3d signs of the rows R (k, 4) of point indices: float with a bound far above its rounding
    (1e-12 of the permanent; the true one is below 10u), Fractions for the rows it leaves open."""
    val, perm = _det3(x[R[:, 1:]] - x[R[:, :1]])
    out = np.where(val > 1e-12 * perm, 1, np.where(val < -1e-12 * perm, -1, 0))
    for r in np.nonzero(np.abs(val) <= 1e-12 * perm)[0]:
        out[r] = orient(*(pts[int(i)] for i in R[r]))
    return out


def _insphere_open(x, S, E):
    """Float in_sphere of the simplices S (k, 4) against the points E (k,): +1 / -1 where certain (1e-10 of the
    permanent), 0 where open. in_sphere = -det[q_i, |q_i|^2] with q_i = p_i - e (exactgeom's sign for d = 3)."""
    Q = x[S] - x[E][:, None, :]
    L = (Q * Q).sum(axis=2)
    val = np.zeros(len(S))
    perm = np.zeros(len(S))
    for i in range(4):
        rows = [j for j in range(4) if j != i]
        m, p = _det3(Q[:, rows, :])
        val += (-1) ** (i + 3) * L[:, i] * m
        perm += L[:, i] * p
    val = -val
    return np.where(val > 1e-10 * perm, 1, np.where(val < -1e-10 * perm, -1, 0))


def _facets_at(S, t, k):
    """The facets (len(t), 3) of the simplices t opposite their positions k, in position order."""
    return S[t][np.arange(4)[None, :] != np.asarray(k)[:, None]].reshape(-1, 3)


def check_delaunay3_large(x, S, NB, perturbed=True):
    """§4 items 1 and 2 for d = 3 without O(t^2) work: the vertices are the distinct points' representatives;
    every simplex is positive; neighbours match the facets (each facet in one or two simplices); across every
    inner facet the opposite vertices lie on opposite sides and each is strictly outside the other's perturbed
    sphere (locally Delaunay); every boundary vertex is on the inner side of every boundary facet. With
    positive simplices and opposite sides the covering degree is the same on both sides of every inner facet,
    and the boundary facets lie on the hull (the extreme points are boundary vertices), so the simplices tile
    the hull once; locally Delaunay then gives the empty spheres. perturbed=False leaves out the tie rule (§4
    item 1 alone: another library's triangulation). The float filter assumes no overflow or underflow in the
    determinants (the extreme-range sets go to check_delaunay)."""
    pts = fpoints(x)
    rep = representatives(pts)
    assert sorted(set(S.ravel().tolist())) == sorted(set(rep)), "the vertices are not the distinct points"
    t = len(S)
    assert (_orient_signs(x, pts, S) > 0).all(), "a simplex is not positive"
    tt, kk = np.nonzero(NB >= 0)
    other = NB[tt, kk]
    back = NB[other] == tt[:, None]
    assert (back.sum(axis=1) == 1).all(), "neighbours are not symmetric"
    kb = np.argmax(back, axis=1)
    assert (np.sort(_facets_at(S, tt, kk), axis=1) == np.sort(_facets_at(S, other, kb), axis=1)).all(), "shared facet"
    allt = np.repeat(np.arange(t), 4)
    keys = np.sort(_facets_at(S, allt, np.tile(np.arange(4), t)), axis=1)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    assert counts.max() <= 2, "a facet in more than two simplices"
    assert (counts == 2).sum() == len(tt) // 2, "a shared facet without neighbours"
    # Opposite sides: simplex t with vertex k replaced by the other simplex's opposite vertex is negative.
    opp = S[other, kb]
    R = S[tt].copy()
    R[np.arange(len(tt)), kk] = opp
    assert (_orient_signs(x, pts, R) < 0).all(), "two neighbours on the same side of their facet"
    # Locally Delaunay with the tie rule: certain negatives pass; open rows go to the exact predicates.
    s = _insphere_open(x, S[tt], opp)
    assert (s <= 0).all(), "a neighbour's vertex strictly inside a sphere"
    ranks = lex_ranks(pts)
    for r in np.nonzero(s == 0)[0]:
        sp = [int(i) for i in S[tt[r]]]
        q = int(opp[r])
        assert in_sphere(*(pts[i] for i in sp), pts[q]) <= 0, "a neighbour's vertex strictly inside a sphere"
        if perturbed:
            assert in_sphere_eps(*(pts[i] for i in sp), pts[q], ranks=[ranks[i] for i in sp] + [ranks[q]]) < 0
    # Boundary facets against the boundary vertices.
    bt, bk = np.nonzero(NB < 0)
    verts = np.unique(_facets_at(S, bt, bk))
    for c in range(0, len(bt), max(1, 200_000 // len(verts))):
        a, k = bt[c : c + 200_000 // len(verts) + 1], bk[c : c + 200_000 // len(verts) + 1]
        R = np.repeat(S[a], len(verts), axis=0)
        R[np.arange(len(R)), np.repeat(k, len(verts))] = np.tile(verts, len(a))
        assert (_orient_signs(x, pts, R) >= 0).all(), "a point outside a boundary facet"
