"""Probes of scipy 1.18.1's Delaunay / ConvexHull / Voronoi (Qhull) for sd.spatial's computational
geometry (docs/design/spatial_geometry.md §2.7, §5, §7.4; SPT8). The records the tests read go to
tests/spatial/data/scipy_geometry.json; the timings go to the terminal and to
build/logs/spatial_geometry_probe/<stamp>.log, and the design document quotes them.

    .venv/bin/python tools/spatial_geometry_probe.py              # every section
    .venv/bin/python tools/spatial_geometry_probe.py exact time   # only these sections

Sections (positional; default all):
  exact    scipy's triangulations of the degenerate and near-degenerate point sets of
           tests/spatial/geompoints.py, checked in rational arithmetic (tests/spatial/exactgeom.py):
           inverted and flat simplices, facets across which the empty-sphere condition fails, dropped points
  shape    dtypes, shapes and orientations of the attributes; the 3-D Voronoi ridges scipy leaves out
  edge     what scipy does with NaN, inf, complex, collinear, duplicate and 4-D input, find_simplex's
           tolerance, the points it shares with the caller
  time     the timings of §5 up to 1e5 points (best of three, one thread; every case stays under 1 GB)
  predicates   (only when named) sd's own exact predicates through the private sd._core.spatial._predicate
           and _det: ns per call by stage, how many rows each stage decides; terminal and log only, the
           records file is not touched (design §3.1 quotes the numbers)

A module of its own beside tools/spatial_probe.py (the distance and neighbour probes), like the test points.
No R section: no comparison package is installed (design §0).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.spatial import exactgeom as eg  # noqa: E402
from tests.spatial import geompoints as gp  # noqa: E402
from tests.spatial.points import digest  # noqa: E402

DATA = ROOT / "tests" / "spatial" / "data"
LOGS = ROOT / "build" / "logs" / "spatial_geometry_probe"
SECTIONS = ("exact", "shape", "edge", "time")
EXTRA = ("predicates",)  # run only when named
EXACT_SETS = (
    "grid2/30", "grid2/30+1e-13", "far2/1e8grid", "polygon2/200", "parabola2/1e-9", "parabola2/1e-14",
    "far2/1e6uniform", "ulp2/200", "circle2/int", "boundary2", "dup2", "uniform2/1000",
    "grid3/6", "cube/5", "far3/1e8grid", "ulp3/200", "sphere3/int", "boundary3", "dup3", "uniform3/1000",
)  # fmt: skip


class Log:
    def __init__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.path = LOGS / (time.strftime("%Y%m%d-%H%M%S") + ".log")
        self.file = self.path.open("w")

    def __call__(self, text=""):
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


def outcome(fn):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = fn()
    except Exception as e:  # noqa: BLE001 -- recording what the library raises
        return {"error": type(e).__name__, "msg": str(e).strip().splitlines()[0][:200]}
    return {"value": v, "warnings": sorted({x.category.__name__ for x in w})}


def exact_record(x):
    """scipy's Delaunay of x against the definition, exactly."""
    import scipy.spatial

    try:
        tri = scipy.spatial.Delaunay(x)
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__, "msg": str(e).strip().splitlines()[0][:200]}
    pts = eg.fpoints(x)
    simplices = tri.simplices.tolist()
    orient = [eg.orient(*(pts[i] for i in s)) for s in simplices]
    d = x.shape[1]
    inverted = sum(o < 0 for o in orient) if d == 2 else None  # 3-D orientations are mixed by design
    violations = 0
    for owners in eg.adjacency(simplices).values():
        if len(owners) == 2:
            (t, k), (t2, k2) = owners
            if orient[t] != 0 and orient[t] * eg.in_sphere(*(pts[i] for i in simplices[t]), pts[simplices[t2][k2]]) > 0:
                violations += 1
    distinct = len(set(pts))
    used = len({i for s in simplices for i in s})
    return {"points": len(x), "distinct": distinct, "simplices": len(simplices),
            "flat": sum(o == 0 for o in orient), "inverted": inverted, "positive": sum(o > 0 for o in orient),
            "negative": sum(o < 0 for o in orient), "not_locally_delaunay": violations, "dropped": distinct - used,
            "coplanar": int(len(tri.coplanar))}  # fmt: skip


def exact_section(log, records):
    log("== exact: scipy's Delaunay checked in rational arithmetic")
    log(f"{'set':22s} {'pts':>5s} {'simpl':>6s} {'flat':>5s} {'inv':>5s} {'viol':>5s} {'dropped':>7s}")
    for id_ in EXACT_SETS:
        x = gp.get(id_).x
        r = exact_record(x)
        r["digest"] = digest(x)
        records[f"exact/{id_}"] = r
        if "error" in r:
            log(f"{id_:22s} {r['error']}: {r['msg'][:60]}")
        else:
            log(f"{id_:22s} {r['points']:5d} {r['simplices']:6d} {r['flat']:5d} {str(r['inverted']):>5s} "
                f"{r['not_locally_delaunay']:5d} {r['dropped']:7d}")  # fmt: skip


def shape_section(log, records):
    import scipy.spatial

    log("== shape: attributes")
    for id_ in ("uniform2/1000", "uniform3/1000"):
        x = gp.get(id_).x
        d = x.shape[1]
        pts = eg.fpoints(x)
        tri, hull, vor = scipy.spatial.Delaunay(x), scipy.spatial.ConvexHull(x), scipy.spatial.Voronoi(x)
        rec = {"digest": digest(x)}
        for obj, names in ((tri, ("simplices", "neighbors", "convex_hull", "vertex_to_simplex", "coplanar", "transform",
                                  "equations", "points")),
                           (hull, ("vertices", "simplices", "neighbors", "equations", "points")),
                           (vor, ("vertices", "ridge_points", "point_region"))):  # fmt: skip
            for n in names:
                a = getattr(obj, n)
                rec[f"{type(obj).__name__}.{n}"] = [str(a.dtype), list(a.shape)]
        indptr, indices = tri.vertex_neighbor_vertices
        rows = [indices[indptr[i] : indptr[i + 1]].tolist() for i in range(len(x))]
        rec["vertex_neighbor_vertices"] = [str(indptr.dtype), str(indices.dtype), all(r == sorted(r) for r in rows)]
        positive = sum(eg.orient(*(pts[i] for i in s)) > 0 for s in tri.simplices.tolist())
        rec["Delaunay.orientation"] = [positive, len(tri.simplices)]
        inner = tuple(sum(c) / len(pts) for c in zip(*pts))
        out = sum(eg.orient(*(pts[i] for i in f), inner) * (1 if d == 2 else -1) > 0 for f in hull.simplices.tolist())
        rec["ConvexHull.outward"] = [out, len(hull.simplices)]
        rec["ConvexHull.area_volume"] = [float(hull.area).hex(), float(hull.volume).hex()]
        rec["ConvexHull.vertices_sorted"] = hull.vertices.tolist() == sorted(hull.vertices.tolist())
        rec["Voronoi.types"] = [type(v).__name__ for v in (vor.regions, vor.ridge_vertices, vor.ridge_dict)]
        rec["Voronoi.empty_regions"] = sum(len(r) == 0 for r in vor.regions)
        rec["Voronoi.counts"] = [len(vor.vertices), len(vor.ridge_points), len(vor.regions)]
        edges = {}
        for t, s in enumerate(tri.simplices.tolist()):
            for i, j in itertools.combinations(sorted(s), 2):
                edges.setdefault((i, j), set()).add(t)
        have = {tuple(sorted(e)) for e in vor.ridge_points.tolist()}
        missing = [e for e in edges if e not in have]
        rec["Voronoi.ridges_left_out"] = [len(missing), sorted({len(edges[e]) for e in missing})]
        records[f"shape/{id_}"] = rec
        log(f"{id_}: " + json.dumps({k: v for k, v in rec.items() if k != "digest"}))
    grid = np.array(list(itertools.product(range(3), repeat=2)), dtype=np.float64)
    vor = scipy.spatial.Voronoi(grid)
    records["shape/grid3x3"] = {"vertices": len(vor.vertices), "ridges": len(vor.ridge_points)}
    log(f"3 x 3 grid Voronoi: {records['shape/grid3x3']}")


def edge_section(log, records):
    import scipy.spatial

    log("== edge: degenerate and wrong input")
    S = scipy.spatial
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.25]])
    cases = {
        "nan": lambda: S.Delaunay(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, np.nan]])).nsimplex,
        "inf": lambda: S.Delaunay(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, np.inf], [1.0, 1.0]])).nsimplex,
        "complex": lambda: S.Delaunay(sq.astype(np.complex128) + 1j).nsimplex,
        "collinear": lambda: S.Delaunay(gp.get("line2").x).nsimplex,
        "coplanar3": lambda: S.Delaunay(gp.get("plane3").x).nsimplex,
        "two_points": lambda: S.Delaunay(sq[:2]).nsimplex,
        "four_dimensions": lambda: S.Delaunay(np.random.default_rng(0).random((20, 4))).simplices.shape[1],
        "duplicates": lambda: (lambda t: [int(t.nsimplex), t.coplanar.tolist()])(
            S.Delaunay(np.concatenate([sq, sq[:1]]))),
        "shares_points": lambda: bool(np.shares_memory(S.Delaunay(sq).points, sq)),
        "find_simplex_dtype": lambda: [type(S.Delaunay(sq).find_simplex(sq[4])).__name__,
                                       str(S.Delaunay(sq).find_simplex(sq).dtype)],
        "find_simplex_outside": lambda: int(S.Delaunay(sq).find_simplex([2.0, 2.0])),
        "find_simplex_nan": lambda: int(S.Delaunay(sq).find_simplex([np.nan, 0.5])),
        "find_simplex_1e-14_outside": lambda: int(S.Delaunay(sq).find_simplex([0.5, -1e-14])),
        "find_simplex_1e-10_outside": lambda: int(S.Delaunay(sq).find_simplex([0.5, -1e-10])),
        "find_simplex_1e-15_outside": lambda: int(S.Delaunay(sq).find_simplex([0.5, -1e-15])),
        "find_simplex_1e-16_outside": lambda: int(S.Delaunay(sq).find_simplex([0.5, -1e-16])),
        "find_simplex_on_shared_edge": lambda: (lambda t: [int(t.find_simplex([0.5, 0.5])), t.simplices.tolist()])(
            S.Delaunay(sq[:4])),
        "hull_edge_point_is_vertex": lambda: 5 in S.ConvexHull(np.concatenate([sq, [[0.5, 0.0]]])).vertices.tolist(),
        "hull_collinear": lambda: S.ConvexHull(gp.get("line2").x).vertices.tolist(),
        "hull_area_volume_2d": lambda: [float(S.ConvexHull(sq).area), float(S.ConvexHull(sq).volume)],
        "voronoi_duplicates_region": lambda: (lambda v: int(v.point_region[0]) == int(v.point_region[5]))(
            S.Voronoi(np.concatenate([sq, sq[:1]]))),
        "parabola_1e-14": lambda: S.Delaunay(gp.get("parabola2/1e-14").x).nsimplex,
    }  # fmt: skip
    for name, fn in cases.items():
        records[f"edge/{name}"] = outcome(fn)
        log(f"{name:28s} {json.dumps(records[f'edge/{name}'])[:150]}")


def best(fn, reps=3):
    b = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b * 1e3


def time_section(log, records):
    import scipy.spatial

    S = scipy.spatial
    log(f"== time: ms, best of three (load {os.getloadavg()[0]:.1f})")
    g = np.random.default_rng(20261123)
    for d, sizes in ((2, (10**3, 10**4, 10**5)), (3, (10**3, 10**4, 10**5))):
        for n in sizes:
            x, q = g.random((n, d)), g.random((n, d))
            row = {"Delaunay": best(lambda: S.Delaunay(x), 3 if n < 10**5 else 1)}
            tri = S.Delaunay(x)
            row["find_simplex"] = best(lambda: tri.find_simplex(q), 3 if n < 10**5 else 1)
            row["ConvexHull"] = best(lambda: S.ConvexHull(x))
            if n <= 10**4 or d == 2:
                row["Voronoi"] = best(lambda: S.Voronoi(x), 3 if n < 10**5 else 1)
            records[f"time/uniform{d}/{n}"] = {k: round(v, 3) for k, v in row.items()}
            log(f"d={d} n={n:6d} " + "  ".join(f"{k} {v:9.2f}" for k, v in row.items()))
    a = 2 * np.pi * g.random(10**4)
    circle = np.stack([np.cos(a), np.sin(a)], axis=1)
    records["time/circle2/10000"] = {"Delaunay": round(best(lambda: S.Delaunay(circle), 1), 3)}
    log(f"unit circle, 1e4 points: Delaunay {records['time/circle2/10000']['Delaunay']:.1f}")


def predicates_section(log, records):
    """sd's predicates, one thread, best of seven (three for the slow paths). Every array stays under 0.5 GB."""
    import stardust as sd

    C = sd._core.spatial
    sd.set_num_threads(1)
    log(f"== predicates: ns per call, one thread (load {os.getloadavg()[0]:.1f})")
    g = np.random.default_rng(20261123)

    def per_call(name, rows, mode, reps=7):
        a = sd.asarray(np.ascontiguousarray(rows))
        return best(lambda: C._predicate(name, a, mode), reps) * 1e6 / len(rows)

    for name, (k, d) in gp.PREDICATES.items():
        uniform = g.random((10**6, k, d))
        pts = g.random((200_000, d))
        _, idx = sd.spatial.KDTree(pts).query(pts, k=k)
        neighbours = pts[np.asarray(idx)]  # the tuples a triangulation asks about: a point and its neighbours
        grid, zero = gp.predicate_rows(name, "grid", 200_000), gp.predicate_rows(name, "zero", 200_000)
        near, extreme = gp.predicate_rows(name, "near", 200_000), gp.predicate_rows(name, "extreme", 20_000)
        row = {
            "filter uniform": per_call(name, uniform, "fast"),
            "auto uniform": per_call(name, uniform, "auto"),
            "filter neighbours": per_call(name, neighbours, "fast"),
            "grid": per_call(name, grid, "fast", 3),
            "zero (expansion)": per_call(name, zero, "fast", 3),
            "near": per_call(name, near, "fast", 3),
            "wide uniform": per_call(name, uniform[:200_000], "wide", 3),
            "wide extreme": per_call(name, extreme, "wide", 3),
        }
        if name.startswith("in"):
            row["sos zero"] = per_call(name + "_sos", zero, "fast", 3)
        log(f"{name:9s} " + "  ".join(f"{key} {v:.1f}" for key, v in row.items()))
        for label, rows in (("neighbours", neighbours), ("grid", grid), ("near", near)):
            log(f"{'':9s} stages (filter, expansion, wide, outside) on {label}: "
                f"{C._predicate_stages(name, sd.asarray(np.ascontiguousarray(rows)))}")  # fmt: skip
        del uniform
        big = sd.asarray(g.random((2 * 10**6, k, d)))
        log(f"{'':9s} stages on 2e6 uniform rows: {C._predicate_stages(name, big)}")
        del big
    for name, pred in (("det2", "orient2d"), ("det3", "orient3d")):
        k, d = gp.PREDICATES[pred]
        uniform = sd.asarray(g.random((10**6, k, d)))
        near = sd.asarray(gp.predicate_rows(pred, "near", 200_000))
        fast = np.asarray(C._det(name, uniform, "fast"))
        declined = float(np.isnan(np.asarray(C._det(name, near, "fast"))).mean())
        t_fast = best(lambda: C._det(name, uniform, "fast")) * 1e6 / 10**6  # best() is in ms
        t_near = best(lambda: C._det(name, near, "auto"), 3) * 1e6 / 200_000
        log(f"{name:9s} fast uniform {t_fast:.1f}  declined on uniform {int(np.isnan(fast).sum())}  "
            f"auto near {t_near:.1f}  declined on near {declined:.3f}")  # fmt: skip


def main(argv=None) -> int:
    import scipy

    p = argparse.ArgumentParser(prog="tools/spatial_geometry_probe.py", description=__doc__.split("\n\n")[0])
    p.add_argument("sections", nargs="*", metavar="SECTION", help=f"sections (default all of {', '.join(SECTIONS)})")
    args = p.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS + EXTRA]
    if unknown:
        p.error(f"unknown section {unknown[0]!r} (choose from {', '.join(SECTIONS + EXTRA)})")
    log = Log()
    if "predicates" in args.sections:
        predicates_section(log, {})
        if all(s in EXTRA for s in args.sections):
            log(f"(records file not touched; log {log.path.relative_to(ROOT)})")
            return 0
    path = DATA / "scipy_geometry.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    records.pop("_provenance", None)
    table = {"exact": exact_section, "shape": shape_section, "edge": edge_section, "time": time_section}
    for s in SECTIONS:
        if not args.sections or s in args.sections:
            for k in [k for k in records if k.startswith(s + "/")]:
                del records[k]
            table[s](log, records)
    meta = {"tool": "tools/spatial_geometry_probe.py", "scipy": scipy.__version__, "numpy": np.__version__,
            "machine": platform.machine(), "system": platform.platform(),
            "date": time.strftime("%Y-%m-%d")}  # fmt: skip
    lines = [f"{json.dumps('_provenance')}: {json.dumps(meta)}"]
    lines += [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in sorted(records.items())]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    log(f"{len(records)} records -> {path.relative_to(ROOT)} (log {log.path.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
