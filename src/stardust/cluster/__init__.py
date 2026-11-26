"""stardust.cluster: partitions of points into groups (docs/api/cluster.md, docs/design/cluster.md).

Usage: ``import stardust as sd`` then ``sd.cluster.kmeans(X, k, rng=sd.random.default_rng(0)).labels``,
``sd.cluster.linkage(X, method='ward')``, ``sd.cluster.agglomerative(X, method='average').cut(k=3)``,
``sd.cluster.dbscan(X, eps=0.5).labels`` and ``sd.cluster.adjusted_rand_score(a, b)``. Partitioning
(k-means, k-medoids, fuzzy c-means), hierarchical (scipy's linkage tools and DIANA), density (DBSCAN,
HDBSCAN, OPTICS), Gaussian mixtures and the clustering indices in one flat namespace (no submodules); the
fitting functions return <Name>Result objects, the scipy-named hierarchy tools take and return arrays
(decisions CL1-CL50 in docs/design/cluster.md §1.1).

Milestone CL0 made the common base of the results (_results.py) and the argument checks (_common.py); the
public names come with milestones CL1 to CL6.
"""

__all__ = []
