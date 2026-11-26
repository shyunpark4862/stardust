"""sd.cluster's accuracy sweeps: one per reference family of tests/cluster/ref (tools/cluster_ref.py) once the
functions it measures are public.

CL0 made the references, the comparators' records (scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2: tests/cluster/data,
from tools/cluster_probe.py) and the prototypes' measurements (tests/cluster/data/prototype.json); each milestone
adds the sweep of its functions. The plan (docs/design/cluster.md §7.5), each judged through
tests/cluster/refdata.py:

  hierarchy   CL1: linkage's heights on hierarchy.npz in units of the contract (single and complete C_DIST u h*,
              average C_AVERAGE gamma(ceil(log2 |A||B|) + 1) h*, ward, centroid and median the centroid bound),
              and the trees: the merges equal to the exact primitive tree where every gap is above NEAR_TIE, the
              near ties counted; comparators scipy's linkage (Lance-Williams heights) and R's hclust, live
  kmeans      CL2: the centres and inertia of the exact Lloyd fixed point from the same start (kmeans.npz),
              the labels at the rounded centres, the fixed point's check; comparator scikit-learn's KMeans from
              the same start
  kmedoids    CL2: PAM's objectives and medoids (and the optimum of the small sets); comparator R's pam
  density     CL3: DBSCAN's partition (the points near eps counted apart), HDBSCAN's core distances and minimum
              spanning tree; comparators scikit-learn's DBSCAN and HDBSCAN
  mixture     CL4: the log densities and responsibilities at given parameters (relative to max(1, |value|)) and
              the M step at given responsibilities; comparator scikit-learn's GaussianMixture at the same
              parameters (its one-pass diagonal covariance loses digits: 6.6e-12 measured at CL0)
  metrics     CL5: ARI, RI (correctly rounded: 0.5 ULP), FM, MI, NMI, AMI, homogeneity, completeness, V, the
              silhouettes, Calinski-Harabasz and Davies-Bouldin; comparator scikit-learn's indices
"""

from __future__ import annotations


def sweeps():
    """No sweep until milestone CL1 makes the first functions public."""
    return []
