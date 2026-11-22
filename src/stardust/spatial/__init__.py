"""stardust.spatial: distances between points and exact nearest-neighbour search
(docs/api/spatial.md, docs/design/spatial.md).

Usage: ``import stardust as sd`` then ``sd.spatial.cdist(XA, XB)``, ``sd.spatial.kneighbors(XA, XB, k=5)``
and ``sd.spatial.KDTree(data).query(x, k=3)``. The names are scipy.spatial's (and
scikit-learn's kneighbors / radius_neighbors for the one-call searches), in one flat namespace (no
sd.spatial.distance); inputs and options follow this library's rules (decisions SPT1-SPT28 in
docs/design/spatial.md §1.1).

The public names come milestone by milestone (docs/design/spatial.md §8): SPT1 the distances (pdist,
cdist, squareform and the fourteen distance functions), SPT2 kneighbors and radius_neighbors, SPT3 KDTree
with its query, SPT4 KDTree's radius, pair and count queries.
"""

from ._distance import (
    braycurtis,
    canberra,
    cdist,
    chebyshev,
    cityblock,
    correlation,
    cosine,
    euclidean,
    hamming,
    haversine,
    jaccard,
    mahalanobis,
    minkowski,
    pdist,
    seuclidean,
    sqeuclidean,
    squareform,
)
from ._kdtree import KDTree
from ._neighbors import kneighbors, radius_neighbors

__all__ = [
    "pdist",
    "cdist",
    "squareform",
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "euclidean",
    "hamming",
    "haversine",
    "jaccard",
    "mahalanobis",
    "minkowski",
    "seuclidean",
    "sqeuclidean",
    "KDTree",
    "kneighbors",
    "radius_neighbors",
]
