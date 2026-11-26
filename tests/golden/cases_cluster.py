"""Golden cases for sd.cluster (tools/golden.py has the contract).

The cases come with the milestones that make the names public (docs/design/cluster.md §8); milestone CL0 made none,
so there is no case yet.

The plan (docs/design/cluster.md §7.5): every public function on the representatives of tests/cluster/samples.py
rebuilt from literals and L.rng (separated, overlapping and elongated blobs, blobs of different sizes and
densities, rings and moons, integer lattices and their shuffles, an evenly spaced line, repeated and all-equal
points, a hexagon and its centre, pairs exactly eps apart, n = 1 to 3, fewer distinct points than k, collinear
points, a constant column, the narrow, shifted, huge and subnormal scales, the inversion triangles, weights with
zeros; labelings with their special cases and string labels) times each option's values (method, metric and
'precomputed', optimal_ordering, criterion; init, n_init, max_iter, tol; eps, min_samples, min_cluster_size,
cluster_selection_method; covariance_type, reg_covar; average_method, beta), the results' methods (cut, predict,
transform, sample with its rng, extract_xi, extract_dbscan, dbscan_clustering, dendrogram), NaN under each
nan_policy, the wrong calls (their errors are results), the same points in two orders (decision CL7), and cases
large enough to split into tasks at 3 and 10 threads (one worker below 1 GB). Results are the same at 1, 3 and 10
threads (decision CL9), so no case is thread dependent.
"""

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {}


def cases(L):
    """No case until milestone CL1 makes the first names public."""
    return iter(())
