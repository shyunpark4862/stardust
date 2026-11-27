"""sd.embed's accuracy sweeps: one per reference family of tests/embed/ref (tools/embed_ref.py) once the functions
it measures are public.

EM0 made the references, the comparators' records (scikit-learn 1.9.1, scipy 1.18.1, statsmodels 0.15.0,
umap-learn 0.6.0, openTSNE 1.0.4 and R 4.6.2 with MASS, irlba, RSpectra, uwot, Rtsne and fastICA: tests/embed/data,
from tools/embed_probe.py) and the prototypes' measurements (tests/embed/data/prototype.json); each milestone adds
the sweep of its functions. The plan (docs/design/embed.md §4, §7.5), each judged through tests/embed/refdata.py:

  linear      EM1-EM3: the variances, singular values and components of pca's solvers and truncated_svd against
              the exact centred Gram's Arb eigenpairs (C_SVD u sqrt(min(n, p)) sigma_1, the Davis-Kahan bound
              for the components), kernel_pca's centred kernels and eigenpairs, factor_analysis's objective and
              gradient at given uniquenesses, one FastICA and one NMF step; comparators scikit-learn's PCA,
              TruncatedSVD, KernelPCA, R's prcomp and irlba, statsmodels' PCA
  mds         EM4: classical_mds's eigenpairs of B = -J D^2 J / 2, one Guttman transform, the three stresses,
              the isotonic regression; comparators scikit-learn's ClassicalMDS and MDS, R's cmdscale
  graph       EM5: the exact k-NN graphs, Isomap's geodesics and eigenpairs, LLE's local weights, the normalised
              Laplacian's eigenpairs; comparators scikit-learn's Isomap, LocallyLinearEmbedding,
              SpectralEmbedding
  sne         EM6-EM7: t-SNE's betas (the perplexity equation's root) and P, the exact gradient and KL at a given
              Y, UMAP's rho, sigma and fuzzy union; comparators scikit-learn's TSNE, openTSNE, umap-learn
  quality     EM5: trustworthiness and continuity correctly rounded (0.5 ULP); comparator scikit-learn's
              trustworthiness
"""

from __future__ import annotations


def sweeps():
    """No sweep until milestone EM1 makes the first functions public."""
    return []
