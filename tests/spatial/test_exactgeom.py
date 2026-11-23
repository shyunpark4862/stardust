"""The exact checker checks itself (tests/spatial/exactgeom.py; docs/design/spatial_geometry.md §7.3):
its two definitions of the in-sphere predicate agree, the tie rule's closed form equals the polynomial in
eps, the perturbed triangulation is a triangulation (and scipy's on general position), the checkers accept
right structures and reject wrong ones."""

import itertools

import numpy as np
import pytest
import scipy.spatial

from . import exactgeom as eg
from . import geompoints as gp
from .conftest import LIVE

SMALL2 = [i for i in gp.ids("d2", "small") if "invalid" not in gp.get(i).tags]
SMALL3 = [i for i in gp.ids("d3", "small") if "invalid" not in gp.get(i).tags]


def distinct_rows(rows):
    return [r for r in rows if len({tuple(p) for p in r.tolist()}) == len(r)]


@pytest.mark.parametrize("name", ["incircle", "insphere"])
@pytest.mark.parametrize("kind", ["uniform", "grid", "zero"])
def test_lifted_determinant_is_the_geometric_predicate(name, kind):
    k, d = gp.PREDICATES[name]
    done = 0
    for row in gp.predicate_rows(name, kind, 150):
        p = eg.fpoints(row)
        if eg.orient(*p[:-1]) == 0:
            continue
        assert eg.sign(eg.lifted(*p)) == eg.in_sphere(*p)
        done += 1
    assert done > 20


@pytest.mark.parametrize("name", ["incircle", "insphere"])
@pytest.mark.parametrize("kind", ["grid", "zero", "uniform"])
def test_tie_rule_closed_form_is_the_eps_polynomial(name, kind):
    """sos_rule (what predicates.hpp implements) against the lifted determinant with eps^rank added to the
    heights, in every argument order of a row."""
    rows = distinct_rows(gp.predicate_rows(name, kind, 400 if kind == "grid" else 60))[:80]
    g = np.random.default_rng(5)
    zeros = 0
    for row in rows:
        p = eg.fpoints(row)
        base = eg.in_sphere_eps(*p)
        if eg.sign(eg.lifted(*p)) != 0:
            assert base == eg.sign(eg.lifted(*p))
        else:
            zeros += 1
        for _ in range(4):
            perm = g.permutation(len(p))
            q = [p[i] for i in perm]
            parity = np.linalg.det(np.eye(len(p))[perm])
            got = eg.in_sphere_eps(*q)
            assert got == round(parity) * base  # a determinant: odd permutations flip it
            if eg.orient(*q[:-1]) != 0:
                assert eg.sos_rule(*q) == got
                assert got != 0
    assert kind == "uniform" or zeros >= 1


def test_eps_ranks_only_matter_through_their_order():
    rows = distinct_rows(gp.predicate_rows("incircle", "zero", 30))
    for row in rows:
        p = eg.fpoints(row)
        local = eg.lex_ranks(p)
        assert eg.in_sphere_eps(*p) == eg.in_sphere_eps(*p, ranks=[3 * r + 2 for r in local])


@pytest.mark.parametrize("id_", SMALL2 + SMALL3)
def test_perturbed_reference_is_a_delaunay_triangulation(id_):
    """§3.2's rule gives ONE triangulation of every point set: the simplices whose perturbed sphere is
    empty cover the hull without overlap (check_delaunay), satisfy the local perturbed condition, and do
    not depend on the order of the rows."""
    x = gp.get(id_).x
    ref = eg.delaunay_reference(x)
    assert ref
    eg.check_delaunay(x, ref)
    eg.check_perturbed(x, ref)
    free = set(eg.delaunay_reference(x, perturbed=False))
    assert set(ref) <= free
    perm = np.random.default_rng(7).permutation(len(x))
    shuffled = eg.delaunay_reference(x[perm])
    as_points = lambda pts, tris: {frozenset(tuple(pts[i]) for i in t) for t in tris}  # noqa: E731
    assert as_points(x.tolist(), ref) == as_points(x[perm].tolist(), shuffled)


@LIVE
@pytest.mark.parametrize("id_", ["uniform2/12", "uniform3/12"])
def test_reference_equals_scipy_in_general_position(id_):
    x = gp.get(id_).x
    ref = eg.delaunay_reference(x)
    assert ref == eg.delaunay_reference(x, perturbed=False)
    tri = scipy.spatial.Delaunay(x)
    assert {frozenset(s) for s in tri.simplices.tolist()} == {frozenset(s) for s in ref}


def test_check_delaunay_rejects_wrong_triangulations():
    x = gp.get("uniform2/12").x
    ref = eg.delaunay_reference(x)
    eg.check_delaunay(x, ref)
    with pytest.raises(AssertionError, match="not positive"):
        eg.check_delaunay(x, [ref[0][::-1]] + ref[1:])
    with pytest.raises(AssertionError):
        eg.check_delaunay(x, ref[1:])
    adj = eg.adjacency(ref)
    f, ((t, k), (t2, k2)) = next((f, o) for f, o in adj.items() if len(o) == 2)
    a, b = f
    c, e = ref[t][k], ref[t2][k2]
    flipped = [s for i, s in enumerate(ref) if i not in (t, t2)]
    pts = eg.fpoints(x)
    if eg.orient(pts[c], pts[e], pts[a]) * eg.orient(pts[c], pts[e], pts[b]) < 0:  # a convex quadrilateral
        flipped += [eg.positive((c, e, a), pts), eg.positive((c, e, b), pts)]
        with pytest.raises(AssertionError, match="Delaunay|inside"):
            eg.check_delaunay(x, flipped)
    with pytest.raises(AssertionError, match="perturbed"):
        grid = gp.get("grid2/5").x
        g = eg.delaunay_reference(grid)
        cell = [i for i in range(len(grid)) if tuple(grid[i]) in {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}]
        mine = [s for s in g if set(s) <= set(cell)]
        assert len(mine) == 2
        other = sorted(set(itertools.combinations(cell, 3)) - {tuple(sorted(s)) for s in mine})
        swapped = [s for s in g if s not in mine] + [eg.positive(s, eg.fpoints(grid)) for s in other]
        eg.check_delaunay(grid, swapped)  # the other diagonal is Delaunay too
        eg.check_perturbed(grid, swapped)  # but not the rule's


def test_find_simplex_reference():
    x = gp.get("grid2/5").x
    ref = eg.delaunay_reference(x)
    assert eg.find_simplex_reference(x, ref, (10.0, 10.0)) == -1
    assert eg.find_simplex_reference(x, ref, (-1e-300, 0.0)) == -1
    v = x[ref[3][0]]
    assert eg.find_simplex_reference(x, ref, v) == min(t for t, s in enumerate(ref) if ref[3][0] in s)
    eg.check_find_simplex(
        x, ref, [(0.5, 0.5), (4.0, 4.0)], [eg.find_simplex_reference(x, ref, q) for q in [(0.5, 0.5), (4.0, 4.0)]]
    )
    with pytest.raises(AssertionError):
        eg.check_find_simplex(x, ref, [(10.0, 10.0)], [0])


def outward(x, facets):
    """scipy's hull facets turned outward (§2.2)."""
    pts = eg.fpoints(x)
    inner = tuple(sum(c) / len(pts) for c in zip(*pts))
    out = []
    for f in facets.tolist():
        s = eg.orient(*(pts[i] for i in f), inner)
        want = 1 if len(f) == 2 else -1
        out.append(tuple(f) if s == want else (f[1], f[0], *f[2:]))
    return out


@LIVE
@pytest.mark.parametrize("id_", ["uniform2/12", "uniform3/12", "uniform2/1000", "normal3/1000"])
def test_hull_checker_accepts_scipy_in_general_position(id_):
    x = gp.get(id_).x
    hull = scipy.spatial.ConvexHull(x)
    facets = outward(x, hull.simplices)
    eg.check_hull(x, hull.vertices, facets)
    if len(x) <= 30:
        assert eg.extreme_points(x) == sorted(hull.vertices.tolist())
    with pytest.raises(AssertionError):
        eg.check_hull(x, hull.vertices, facets[1:])
    with pytest.raises(AssertionError):
        eg.check_hull(x, hull.vertices, [facets[0][::-1]] + facets[1:])


def test_extreme_points_of_degenerate_sets():
    x = gp.get("boundary2").x
    assert {tuple(x[i]) for i in eg.extreme_points(x)} == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    cube = gp.get("grid3/2").x
    assert eg.extreme_points(cube) == list(range(8))
    x = np.concatenate([cube, [[0.5, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]])
    assert eg.extreme_points(x) == list(range(8))  # face, edge and inner points and a duplicate are not extreme
    assert len(eg.extreme_points(gp.get("parabola2/int").x)) == 200


@LIVE
@pytest.mark.parametrize("id_", ["uniform2/12", "uniform3/12"])
def test_voronoi_checker_accepts_scipy_in_general_position(id_):
    x = gp.get(id_).x
    ref = eg.delaunay_reference(x)
    cluster, centres, ridges = eg.voronoi_reference(x, ref)
    vor = scipy.spatial.Voronoi(x)
    assert len(centres) == len(vor.vertices) == len(ref)
    number = {}
    for c, centre in enumerate(centres):
        d = np.abs(vor.vertices - np.array([float(v) for v in centre])).max(axis=1)
        number[c] = int(d.argmin())
        assert d.min() <= 1e-6 * (1.0 + np.abs(vor.vertices[number[c]]).max())
    assert len(set(number.values())) == len(number)
    rp = [tuple(sorted(e)) for e in vor.ridge_points.tolist()]
    rv = [list(v) for v in vor.ridge_vertices]
    # scipy 1.18.1 (Qhull) leaves out the unbounded 3-D ridges with ONE finite vertex (a Delaunay edge of a
    # single tetrahedron), though the two regions do share a two-dimensional face: add them back
    missing = {e: c for e, c in ridges.items() if e not in set(rp)}
    assert all(len(c) == 2 and -1 in c for c in missing.values()) and (not missing or x.shape[1] == 3)
    for e, c in missing.items():
        rp.append(e)
        rv.append([-1 if v == -1 else number[v] for v in c])
    eg.check_voronoi(x, ref, rp, rv, [number[c] for c in cluster], definition=12)
    with pytest.raises(AssertionError):
        eg.check_voronoi(x, ref, rp[1:], rv[1:], [number[c] for c in cluster], definition=0)


def test_voronoi_reference_merges_cospherical_simplices():
    x = np.array(list(itertools.product(range(3), repeat=2)), dtype=np.float64)
    ref = eg.delaunay_reference(x)
    cluster, centres, ridges = eg.voronoi_reference(x, ref)
    assert len(centres) == 4 and len(ridges) == 12  # scipy's counts on the 3 x 3 grid (§2.5)
    assert all(abs(a - c) + abs(b - d) == 1 for (i, j) in ridges for (a, b), (c, d) in [(x[i], x[j])])
    for i, j in itertools.combinations(range(9), 2):
        assert ((i, j) in ridges) == eg.voronoi_ridge_exists(x, i, j)
    cube = gp.get("grid3/2").x
    cluster, centres, ridges = eg.voronoi_reference(cube, eg.delaunay_reference(cube))
    assert len(centres) == 1 and len(ridges) == 12  # the cube's edges; no ridge across a face diagonal
    assert all(np.abs(cube[i] - cube[j]).sum() == 1 for i, j in ridges)
