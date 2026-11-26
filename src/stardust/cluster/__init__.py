"""stardust.cluster: partitions of points into groups (docs/api/cluster.md, docs/design/cluster.md).

Usage: ``import stardust as sd`` then ``sd.cluster.kmeans(X, k, rng=sd.random.default_rng(0)).labels``,
``sd.cluster.linkage(X, method='ward')``, ``sd.cluster.agglomerative(X, method='average').cut(k=3)``,
``sd.cluster.dbscan(X, eps=0.5).labels`` and ``sd.cluster.adjusted_rand_score(a, b)``. Partitioning
(k-means, k-medoids, fuzzy c-means), hierarchical (scipy's linkage tools and DIANA), density (DBSCAN,
HDBSCAN, OPTICS), Gaussian mixtures and the clustering indices in one flat namespace (no submodules); the
fitting functions return <Name>Result objects, the scipy-named hierarchy tools take and return arrays
(decisions CL1-CL81 in docs/design/cluster.md §1.1).

Milestone CL0 made the common base of the results (_results.py) and the argument checks (_common.py); the
public names come with milestones CL1 to CL6 (CL1: the hierarchy, _hierarchy.py; CL2: the partitions, _kmeans.py,
_kmedoids.py, _fuzzy.py; CL3: the density methods, _density.py; CL4: the Gaussian mixtures, _mixture.py).
"""

from ._density import DBSCANResult, HDBSCANResult, OPTICSResult, dbscan, hdbscan, optics
from ._fuzzy import FuzzyCMeansResult, fuzzy_cmeans
from ._hierarchy import (
    CophenetResult,
    DendrogramResult,
    HierarchyResult,
    agglomerative,
    cophenet,
    cophenet_correlation,
    cut_tree,
    dendrogram,
    divisive,
    fcluster,
    inconsistent,
    is_isomorphic,
    is_monotonic,
    is_valid_linkage,
    leaders,
    leaves_list,
    linkage,
    maxdists,
    maxinconsts,
    maxRstat,
    num_obs_linkage,
    optimal_leaf_ordering,
)
from ._kmeans import BisectingKMeansResult, KMeansResult, bisecting_kmeans, kmeans, minibatch_kmeans
from ._kmedoids import KMedoidsResult, kmedoids
from ._mixture import GaussianMixtureResult, gaussian_mixture

__all__ = [
    "linkage",
    "agglomerative",
    "divisive",
    "HierarchyResult",
    "fcluster",
    "cut_tree",
    "cophenet",
    "cophenet_correlation",
    "CophenetResult",
    "leaves_list",
    "optimal_leaf_ordering",
    "dendrogram",
    "DendrogramResult",
    "inconsistent",
    "maxdists",
    "maxinconsts",
    "maxRstat",
    "leaders",
    "is_monotonic",
    "is_valid_linkage",
    "num_obs_linkage",
    "is_isomorphic",
    "kmeans",
    "minibatch_kmeans",
    "bisecting_kmeans",
    "KMeansResult",
    "BisectingKMeansResult",
    "kmedoids",
    "KMedoidsResult",
    "fuzzy_cmeans",
    "FuzzyCMeansResult",
    "dbscan",
    "DBSCANResult",
    "hdbscan",
    "HDBSCANResult",
    "optics",
    "OPTICSResult",
    "gaussian_mixture",
    "GaussianMixtureResult",
]
