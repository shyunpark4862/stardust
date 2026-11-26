// The cluster area's native core: one bind_* per file, called by bind/cluster_bind.cpp (docs/design/cluster.md §6).
#pragma once

#include <nanobind/nanobind.h>

namespace sd::cluster {

void bind_hierarchy(nanobind::module_& m);  // hierarchy.cpp: linkage, DIANA, cophenet, optimal leaf ordering
void bind_kmeans(nanobind::module_& m);     // kmeans.cpp: k-means, mini-batch and bisecting k-means
void bind_kmedoids(nanobind::module_& m);   // kmedoids.cpp: PAM and CLARA
void bind_fuzzy(nanobind::module_& m);      // fuzzy.cpp: fuzzy c-means
void bind_density(nanobind::module_& m);    // density.cpp: DBSCAN, HDBSCAN, OPTICS
void bind_mixture(nanobind::module_& m);    // mixture.cpp: Gaussian mixtures (EM)
void bind_metrics(nanobind::module_& m);    // metrics.cpp: the indices, the gap statistic, Gower's distance
void bind_affinity(nanobind::module_& m);   // affinity.cpp: affinity propagation

}  // namespace sd::cluster
