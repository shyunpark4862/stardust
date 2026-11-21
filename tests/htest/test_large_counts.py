"""sd.htest with large counts (docs/api/htest.md §7, §8; docs/design/htest.md H150-H154): the
2 x 2 fisher_exact and odds_ratio past 2^31 (a 128-bit mode, the binary search, the window of
the noncentral hypergeometric), the count limits of 2^53 (fisher_exact, odds_ratio, binomtest,
proportion_confint), poisson_means_test's step cap, cramervonmises_2samp's bounded try of
method='exact', and that each of these long calls leaves the GIL to other threads (API.md §3).
Every call here finishes in well under a second and holds little memory."""

import math
import threading
import time

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from .._bits import bits_equal

h = sd.htest
pytestmark = pytest.mark.usefixtures("restore_threads")


# ---- fisher_exact 2 x 2 (H150) -------------------------------------------------------------


@pytest.mark.parametrize("N", [1_500_000_000, 1_900_000_000, 2_147_483_660, 10**10, 10**11, 10**14])
def test_fisher_two_sided_past_2_31(N):
    """The mode's product past 2^63 (once a wrong p of 0.50 and an O(N) walk): p = 1 - (the
    tables between the two boundaries), near 1 - c / sqrt(N); the one-sided tails unchanged."""
    T = [[N, N + 7], [N + 3, N]]
    t0 = time.perf_counter()
    p = h.fisher_exact(T).pvalue
    assert time.perf_counter() - t0 < 1.0
    assert 0.9999 < p < 1.0
    # 1 - p falls as 1 / sqrt(N) (the middle's few tables, each of size about 1 / sd)
    q = h.fisher_exact([[10**10, 10**10 + 7], [10**10 + 3, 10**10]]).pvalue
    assert (1 - p) / (1 - q) == pytest.approx(math.sqrt(4 * 10**10 / (4 * N)), rel=1e-3)


def test_fisher_one_sided_and_far_tables():
    N = 1_900_000_000
    T = [[N, N + 7], [N + 3, N]]
    assert h.fisher_exact(T, alternative="less").pvalue == pytest.approx(0.4999633905456719, rel=1e-12)
    assert h.fisher_exact(T, alternative="greater").pvalue == pytest.approx(0.5000549141814002, rel=1e-12)
    for N in (10**6, 10**8, 10**9):  # |a - mode| = N / 6: the binary search, p = 0
        t0 = time.perf_counter()
        assert h.fisher_exact([[N, N // 2], [N // 2, N]]).pvalue == 0.0
        assert time.perf_counter() - t0 < 0.05


def test_fisher_matches_scipy_on_moderate_tables():
    """Tables a binary search reaches (|a - mode| past 2^16) and walks (the rest): scipy's rule."""
    rng = np.random.default_rng(11)
    for _ in range(40):
        s = int(rng.choice([50, 5000, 500_000]))
        T = [[int(v) for v in rng.integers(0, s, 2)], [int(v) for v in rng.integers(0, s, 2)]]
        if 0 in (T[0][0] + T[0][1], T[1][0] + T[1][1], T[0][0] + T[1][0], T[0][1] + T[1][1]):
            continue
        ours, theirs = h.fisher_exact(T).pvalue, st.fisher_exact(T).pvalue
        assert ours == pytest.approx(theirs, rel=1e-6, abs=1e-300), T


def test_fisher_significant_table_is_not_one():
    """A ratio to the mode past 2^1023 once compared as NaN, so p = 1 for a table like this."""
    for T in ([[4465, 754], [2559, 2595]], [[10585, 47194], [3181, 5305]], [[509, 3643], [1481, 1062]]):
        assert h.fisher_exact(T).pvalue < 1e-300
        assert h.fisher_exact(T).pvalue == pytest.approx(st.fisher_exact(T).pvalue, abs=1e-300)


def test_count_limit():
    msg = r"fisher_exact\(\): the table's counts must be below 2\*\*53"
    with pytest.raises(ValueError, match=msg):
        h.fisher_exact([[2**53, 1], [1, 1]])
    with pytest.raises(ValueError, match=msg):  # the total
        h.fisher_exact([[2**51, 2**51], [2**51, 2**51]])
    with pytest.raises(ValueError, match=msg):  # a batch: any table
        h.fisher_exact(sd.asarray([[[1.0, 2], [3, 4]], [[2.0**52, 2**52], [1, 1]]]))
    with pytest.raises(ValueError, match=r"odds_ratio\(\): the table's counts must be below 2\*\*53"):
        h.odds_ratio([[2**52, 2**52], [1, 1]])
    N = 2**51 - 10  # below the limit: the rule's p = 1 (the table within 1e-14 of the mode)
    assert h.fisher_exact([[N, N + 7], [N + 3, N]]).pvalue == 1.0


def test_fisher_threads_and_batch():
    tabs = [
        [[10**10, 10**10 + 7], [10**10 + 3, 10**10]],
        [[10**8, 5 * 10**7], [5 * 10**7, 10**8]],
        [[30, 70000], [70000, 10**9]],
    ]
    sd.set_num_threads(1)
    one = [h.fisher_exact(t).pvalue for t in tabs]
    sd.set_num_threads(10)
    batch = h.fisher_exact(sd.asarray(tabs)).pvalue
    assert bits_equal(batch, sd.asarray(one))


# ---- odds_ratio (H150) ---------------------------------------------------------------------


def test_odds_ratio_window():
    for N, lo in ((10**5, 0.98756228), (10**7, 0.99875997), (10**9, 0.99987603)):
        t0 = time.perf_counter()
        r = h.odds_ratio([[N, N + 7], [N + 3, N]])
        ci = r.conf_int()
        assert time.perf_counter() - t0 < 1.0
        assert r.statistic == pytest.approx(N * N / ((N + 7) * (N + 3)), rel=1e-6)
        assert float(ci[0]) == pytest.approx(lo, rel=1e-7)


# scipy 1.18.1's contingency.odds_ratio on supports past 2^16 (15-20 s each there: recorded here)
SCIPY_OR = {
    ((40000, 30000), (35000, 41000)): (1.5618999535423355, 1.5298559879597242, 1.5946248103670082),
    ((70000, 5), (3, 70000)): (288414073.1481234, 69544812.08309011, 1876862992.164582),
}


def test_odds_ratio_window_against_scipy():
    """A support past 2^16 (the window) and below it (the whole support): scipy's within 1e-7."""
    for T, (est, low, high) in SCIPY_OR.items():
        r = h.odds_ratio([list(T[0]), list(T[1])])
        assert r.statistic == pytest.approx(est, rel=1e-7)
        assert tuple(r.conf_int().tolist()) == pytest.approx((low, high), rel=1e-7)
    for T in ([[49303, 36443], [4, 18737]], [[415, 42], [2, 418]]):
        ours, theirs = h.odds_ratio(T), st.contingency.odds_ratio(T)
        assert ours.statistic == pytest.approx(theirs.statistic, rel=1e-7)
        c = theirs.confidence_interval()
        assert tuple(ours.conf_int().tolist()) == pytest.approx((c.low, c.high), rel=1e-7)


def test_odds_ratio_window_cap():
    msg = (
        r"odds_ratio\(\): the conditional odds ratio needs more than 4\.19e\+06 terms of its sums for this table; "
        r"use kind='sample'"
    )
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=msg):
        h.odds_ratio([[10**12, 10**12 + 7], [10**12 + 3, 10**12]])
    assert time.perf_counter() - t0 < 5.0
    assert h.odds_ratio([[10**12, 10**12 + 7], [10**12 + 3, 10**12]], kind="sample").statistic > 0
    # a narrow distribution at the estimate: the window is small there, whatever the counts
    r = h.odds_ratio([[10**12, 5], [3, 10**12]])
    assert r.statistic == pytest.approx(5.88609e22, rel=1e-5)


def test_odds_ratio_threads_and_batch():
    tabs = [[[10**7, 10**7 + 7], [10**7 + 3, 10**7]], [[5, 3], [2, 9]], [[40000, 30000], [35000, 41000]]]
    sd.set_num_threads(1)
    one = [h.odds_ratio(t) for t in tabs]
    sd.set_num_threads(10)
    batch = h.odds_ratio(sd.asarray(tabs))
    assert bits_equal(batch.statistic, sd.asarray([r.statistic for r in one]))
    lo = sd.asarray([float(r.conf_int()[0]) for r in one])
    assert bits_equal(batch.conf_int()[:, 0], lo)


# ---- binomtest, proportion_confint (H151) ---------------------------------------------------


def test_binomtest_limit():
    msg = r"binomtest\(\): 'n' must be below 2\*\*53"
    for k, n in ((3 * 2**53 // 4, 2**53), (3, 2**54), (2**60, 2**62)):
        t0 = time.perf_counter()
        with pytest.raises(ValueError, match=msg):
            h.binomtest(k, n, p=0.25)
        assert time.perf_counter() - t0 < 0.1
    n = 10**15
    assert h.binomtest(3 * n // 4, n, p=0.25).pvalue == 0.0
    # past sd.stats Binomial's range (2^52): NaN, not 1
    n = 2**52 + 100
    assert math.isnan(h.binomtest(n // 4 + 5, n, p=0.25).pvalue)
    with pytest.raises(ValueError, match=r"proportion_confint\(\): 'nobs' must be below 2\*\*53"):
        h.proportion_confint(3, 2.0**53)


# ---- poisson_means_test (H154) --------------------------------------------------------------


def test_poisson_means_step_cap():
    msg = r"poisson_means_test\(\): the counts are too large for the exact sum \(it needs more than 2e\+07 steps\)"
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=msg):
        h.poisson_means_test(10**13, 1.0, 10**13 + 10**7, 1.0)
    assert time.perf_counter() - t0 < 0.1
    r = h.poisson_means_test(10**11, 1.0, 10**11 + 10**6, 1.0)
    assert r.pvalue == pytest.approx(0.0253477, rel=1e-5)


# ---- cramervonmises_2samp(method='exact') (H152) --------------------------------------------


def test_cvm2_exact_try_is_bounded():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=150), rng.normal(size=153)
    with pytest.raises(ValueError, match=r"cramervonmises_2samp\(\): method='exact' needs about .* more than its cap"):
        h.cramervonmises_2samp(x, y, method="exact")
    sd.set_num_threads(4)
    X, Y = rng.normal(size=(4, 150)), rng.normal(size=(4, 153))
    with pytest.raises(ValueError, match=r"cramervonmises_2samp\(\): method='exact' needs about"):
        h.cramervonmises_2samp(X, Y, axis=1, method="exact")
    # within the cap: the exact p-value, a lane alone and in a batch
    a, b = rng.normal(size=(3, 12)), rng.normal(size=(3, 9))
    one = [h.cramervonmises_2samp(a[i], b[i], method="exact").pvalue for i in range(3)]
    assert bits_equal(h.cramervonmises_2samp(a, b, axis=1, method="exact").pvalue, sd.asarray(one))


# ---- the GIL (API.md §3) --------------------------------------------------------------------


class Watcher:
    """A Python thread that records its longest pause (tests/test_gil_release.py's)."""

    def __init__(self):
        self.longest = 0.0
        self.turns = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            self.longest = max(self.longest, now - last)
            last = now
            self.turns += 1

    def pause_during(self, call):
        self._thread.start()
        while self.turns == 0:
            time.sleep(0.001)
        time.sleep(0.002)
        self.longest = 0.0
        t0 = time.perf_counter()
        call()
        took = time.perf_counter() - t0
        time.sleep(0.002)
        self._stop.set()
        self._thread.join()
        return self.longest, took


@pytest.mark.serial
@pytest.mark.parametrize(
    "case",
    [
        lambda: h.fisher_exact(sd.asarray([[[10**10, 10**10 + 7], [10**10 + 3, 10**10]]] * 200)),
        lambda: h.odds_ratio([[10**9, 10**9 + 7], [10**9 + 3, 10**9]]).conf_int(),
        lambda: h.poisson_means_test(10**11, 1.0, 10**11 + 10**6, 1.0),
    ],
    ids=["fisher_exact", "odds_ratio", "poisson_means_test"],
)
def test_long_calls_leave_the_gil(case):
    sd.set_num_threads(1)
    longest, took = Watcher().pause_during(case)
    assert took > 0.05
    assert longest < 0.5 * took
