"""stardust.embed: points moved to fewer dimensions (docs/api/embed.md, docs/design/embed.md).

Usage: ``import stardust as sd`` then ``sd.embed.pca(X, 2).embedding``, ``sd.embed.tsne(X).embedding``,
``sd.embed.umap(X, rng=sd.random.default_rng(0)).embedding`` and ``sd.embed.trustworthiness(X, Z)``. Linear
decompositions (PCA, truncated SVD, random projection, factor analysis, FastICA, NMF, kernel PCA), distance
preserving maps (classical and SMACOF MDS, Sammon), manifold learning on neighbour graphs (Isomap, LLE, spectral
embedding), stochastic neighbour embeddings (t-SNE, UMAP) and the embedding indices in one flat namespace (no
submodules); the fitting functions return <Name>Result objects (decisions EM1-EM48 in docs/design/embed.md §1.1).

Milestone EM0 made the common base of the results (_results.py) and the argument checks (_common.py, over the
learning areas' shared stardust._learn); the public names come with milestones EM1 to EM8.
"""

__all__ = []
