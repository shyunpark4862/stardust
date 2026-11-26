"""stardust.cluster: partitions of points into groups (docs/api/cluster.md, docs/design/cluster.md).

Usage: ``import stardust as sd`` then ``sd.cluster.kmeans(X, k, rng=sd.random.default_rng(0)).labels``,
``sd.cluster.linkage(X, method='ward')``, ``sd.cluster.agglomerative(X, method='average').cut(k=3)``,
``sd.cluster.dbscan(X, eps=0.5).labels`` and ``sd.cluster.adjusted_rand_score(a, b)``. Partitioning
(k-means, k-medoids, fuzzy c-means), hierarchical (scipy's linkage tools and DIANA), density (DBSCAN,
HDBSCAN, OPTICS), Gaussian mixtures and the clustering indices in one flat namespace (no submodules); the
fitting functions return <Name>Result objects, the scipy-named hierarchy tools take and return arrays (decisions in
docs/design/cluster.md §1.1). The results' common base is _results.py, the argument checks _common.py; the hierarchy
is in _hierarchy.py, the partitions in _kmeans.py, _kmedoids.py and _fuzzy.py, the density methods in _density.py,
the Gaussian mixtures in _mixture.py, the indices, the gap statistic and Gower's distance in _metrics.py, _gap.py and
_gower.py, and affinity propagation in _affinity.py.
"""

from ._affinity import AffinityPropagationResult, affinity_propagation
from ._density import DBSCANResult, HDBSCANResult, OPTICSResult, dbscan, hdbscan, optics
from ._fuzzy import FuzzyCMeansResult, fuzzy_cmeans
from ._gap import GapResult, gap_statistic
from ._gower import gower_distance
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
from ._metrics import (
    HomogeneityResult,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    contingency_matrix,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    pair_confusion_matrix,
    rand_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)
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
    "silhouette_samples",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "adjusted_rand_score",
    "rand_score",
    "adjusted_mutual_info_score",
    "normalized_mutual_info_score",
    "mutual_info_score",
    "homogeneity_score",
    "completeness_score",
    "v_measure_score",
    "homogeneity_completeness_v_measure",
    "HomogeneityResult",
    "fowlkes_mallows_score",
    "pair_confusion_matrix",
    "contingency_matrix",
    "gap_statistic",
    "GapResult",
    "gower_distance",
    "affinity_propagation",
    "AffinityPropagationResult",
]
