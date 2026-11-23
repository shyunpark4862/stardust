"""The recorded scipy 1.18.1 results for the computational geometry
(tests/spatial/data/scipy_geometry.json, written by tools/spatial_geometry_probe.py;
docs/design/spatial_geometry.md §2.7, §7.4): the records belong to the test points as they are, and they
show the behaviours the design departs from. The file is compared, not the installed scipy."""

import json
import pathlib

from . import geompoints as gp
from .points import digest

DATA = json.loads((pathlib.Path(__file__).parent / "data" / "scipy_geometry.json").read_text())


def test_provenance_and_digests():
    assert DATA["_provenance"]["scipy"] == "1.18.1"
    assert DATA["_provenance"]["tool"] == "tools/spatial_geometry_probe.py"
    for key, rec in DATA.items():
        if key.startswith(("exact/", "shape/uniform")):
            assert rec["digest"] == digest(gp.get(key.split("/", 1)[1]).x), key


def test_scipy_is_exact_on_integer_and_general_data():
    for id_ in ("grid2/30", "circle2/int", "boundary2", "uniform2/1000", "uniform3/1000", "sphere3/int"):
        r = DATA[f"exact/{id_}"]
        assert r["not_locally_delaunay"] == 0 and r["dropped"] == 0 and not r["inverted"], id_
    assert DATA["exact/grid2/30"]["simplices"] == 1682


def test_scipy_departs_from_the_definition_on_near_degenerate_data():
    assert DATA["exact/grid2/30+1e-13"]["inverted"] > 0 and DATA["exact/grid2/30+1e-13"]["not_locally_delaunay"] > 100
    assert DATA["exact/far2/1e8grid"]["dropped"] == 896 and DATA["exact/far2/1e8grid"]["simplices"] == 2
    assert DATA["exact/far2/1e6uniform"]["dropped"] > 400
    assert DATA["exact/parabola2/1e-14"]["dropped"] > 200
    for id_ in ("polygon2/200", "parabola2/1e-9", "ulp2/200", "ulp3/200"):
        assert DATA[f"exact/{id_}"]["not_locally_delaunay"] > 50, id_
    assert DATA["exact/grid3/6"]["flat"] > 0 and DATA["exact/cube/5"]["flat"] > 0  # zero-volume tetrahedra


def test_scipy_shapes_and_orders():
    for d in (2, 3):
        r = DATA[f"shape/uniform{d}/1000"]
        assert r["Delaunay.simplices"][0] == "int32" and r["ConvexHull.vertices"][0] == "int32"
        assert r["vertex_neighbor_vertices"][2] is False  # rows are not sorted
        assert 0 < r["ConvexHull.outward"][0] < r["ConvexHull.outward"][1]  # mixed facet orientations
        assert r["Voronoi.types"] == ["list", "list", "dict"] and r["Voronoi.empty_regions"] == 1
    assert DATA["shape/uniform2/1000"]["Delaunay.orientation"] == [1982, 1982]
    pos, total = DATA["shape/uniform3/1000"]["Delaunay.orientation"]
    assert 0 < pos < total
    assert DATA["shape/uniform2/1000"]["Voronoi.ridges_left_out"] == [0, []]
    assert DATA["shape/uniform3/1000"]["Voronoi.ridges_left_out"] == [23, [1]]
    assert DATA["shape/grid3x3"] == {"vertices": 4, "ridges": 12}


def test_scipy_edge_behaviour():
    e = {k.split("/", 1)[1]: v for k, v in DATA.items() if k.startswith("edge/")}
    assert e["nan"]["error"] == "ValueError" and e["inf"]["error"] == "QhullError"
    assert e["complex"]["warnings"] == ["ComplexWarning"]
    assert e["collinear"]["error"] == e["coplanar3"]["error"] == e["two_points"]["error"] == "QhullError"
    assert e["four_dimensions"]["value"] == 5 and e["shares_points"]["value"] is True
    assert e["duplicates"]["value"] == [4, [[5, 1, 0]]]
    assert e["find_simplex_1e-15_outside"]["value"] >= 0 and e["find_simplex_1e-14_outside"]["value"] == -1
    assert e["find_simplex_nan"]["value"] == -1 and e["find_simplex_dtype"]["value"] == ["ndarray", "int32"]
    assert e["hull_edge_point_is_vertex"]["value"] is False and e["hull_area_volume_2d"]["value"] == [4.0, 1.0]
