"""Golden cases for sd.embed (tools/golden.py has the contract).

The cases come with the milestones that make the names public (docs/design/embed.md §8); milestone EM0 made none,
so there is no case yet.

The plan (docs/design/embed.md §7.5): every public function on the representatives of tests/embed/samples.py
rebuilt from literals and L.rng (low rank plus noise, repeated eigenvalues and a tiny gap, the shifted, tiny and
huge scales, a constant column, n < p, sparse and nonnegative data, mixed sources, a factor model; the Swiss roll,
the S curve, a circle, a ring, a sphere, two far apart blobs, integer lattices and their shuffles, repeated points,
collinear points; Gaussian blobs and synthetic digits; n = 1 to 3, fewer distinct points than n_components, all
equal points) times each option's values (svd_solver, algorithm, eigen_solver, kernel, rotation, init, method,
metric and 'precomputed', perplexity, n_neighbors, min_dist), the results' methods (transform, inverse_transform,
score_samples, reconstruction_error, summary), NaN under each nan_policy, the wrong calls (their errors are
results), the same points in two orders (decision EM12), the same rng state twice (decision EM11), and cases large
enough to split into tasks at 3 and 10 threads (one worker below 1 GB). The required promise is D-33 (decision
EM4): the thread-independent paths are the same at 1, 3 and 10 threads; a path that falls to D-33 by the rule of
EM4 (§3.11) lists its cases in THREAD_DEPENDENT with the reason, and its results are kept per thread count.
"""

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {}

# Cases whose results depend on the thread count (D-33 paths of docs/design/embed.md §3.11): the reason each does.
THREAD_DEPENDENT = {}


def cases(L):
    """No case until milestone EM1 makes the first names public."""
    return iter(())
