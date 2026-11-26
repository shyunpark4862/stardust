"""The gap statistic (src/stardust/cluster/_gap.py, csrc/cluster/metrics.cpp; docs/api/cluster.md §6;
docs/design/cluster.md §2.6, §3.9, §4.6; decisions CL37, CL41, CL87, CL88): log W of a given partition against the
exact references, R cluster's clusGap with PAM (log W bit for bit in effect, the reference mean statistically),
maxSE's rules against R, the sub-keys of the clusterings, the rng, the thread counts, the input order, the result
and the errors."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S

CL = sd.cluster
CORE = sd._core.cluster
REF = D.ref("metrics")
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _rng(seed):
    return sd.random.default_rng(seed)


def test_log_w_of_a_partition_within_the_contract():
    sets = set(REF.group_inputs["set"].tolist())
    for sid in S.point_ids():
        s = S.points(sid)
        if sid not in sets or s.truth is None:
            continue
        codes = np.unique(s.truth, return_inverse=True)[1].astype(np.int64)
        o = _np(CORE.lex_order(A(s.X)))
        K = int(codes.max()) + 1
        for power in (1, 2):
            m = REF.methods[f"logw_{power}"]
            at = np.nonzero(m.index == D.group_of(REF, sid))[0]
            if not len(at):
                continue
            ref = m.hi[at[0]] + m.lo[at[0]]
            got = CORE.gap_logw(A(np.ascontiguousarray(s.X[o])), A(codes[o]), K, float(power))
            assert abs(got - ref) <= D.logw_bound(s.n, s.p, K, power, ref), (sid, power, got, ref)


def test_r_clusgap_with_pam():
    rec = D.data("r")
    for sid in ("blobs/k3p2", "overlap/k3p2", "probe/300"):
        X = S.points(sid).X
        for dp in (1, 2):
            for sp in ("scaledPCA", "original"):
                v = np.array(rec[f"clusgap_pam/{sid}/{dp}/{sp}"]["value"])
                logw, elogw, se = v[:6], v[6:12], v[12:18]
                r = CL.gap_statistic(X, 6, rng=_rng(1), method="pam", d_power=dp, reference=sp.lower())
                # the data's log W: PAM's partition is R's (deterministic), the sums the same up to rounding
                assert np.allclose(_np(r.log_w), logw, rtol=1e-13, atol=1e-13), (sid, dp, sp)
                # the reference mean: another random stream, the same distribution (within a few standard errors)
                assert np.all(np.abs(_np(r.e_log_w) - elogw) <= 5 * np.maximum(_np(r.se), se)), (sid, dp, sp)
    gap = np.array(rec["cluster/clusgap"]["value"])
    r = CL.gap_statistic(S.points("blobs/k3p2").X, 6, rng=_rng(1), n_reference=20)
    assert r.k == int(gap[-1]) == 3


def test_max_se_against_r():
    from stardust.cluster._gap import max_se

    rec = D.data("r")["cluster/maxse"]
    names = [m.lower() for m in rec["methods"]]
    for row in rec["cases"]:
        f, se, want = row[0], row[1], row[2:]
        q = 0
        for m in names:
            for fac in rec["factors"]:
                assert max_se(f, se, m, fac) == int(want[q]), (f, se, m, fac)
                q += 1


def test_the_streams():
    # CL41, CL88: reference set b draws element b 2^40 + i p + d; the clusterings of set s (0 the data) at k use the
    # sub-key at start 2^23 + s k_max + k, so the data's log W at k is that of sd's kmeans with those two words
    X = S.points("blobs/k3p2").X
    k0, k1 = _rng(4).integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,)).tolist()
    r = CL.gap_statistic(X, 5, rng=_rng(4), n_reference=3)
    o = CORE.lex_order(A(X))
    Xk = A(np.ascontiguousarray(X[_np(o)]))
    for k in range(2, 6):
        a, b = CORE.subkey(k0, k1, 2**23 + k)
        km = CORE.kmeans(A(X), o, None, k, 0, None, 1, 300, 1e-4, a, b, 0)
        lw = CORE.gap_logw(Xk, km[0][0], k, 1.0)
        assert lw == _np(r.log_w)[k - 1], k
    assert _np(r.log_w)[0] == CORE.gap_logw(Xk, A(np.zeros(len(X), np.int64)), 1, 1.0)
    # the generator moves by two words; the same state gives the same bits
    g = _rng(4)
    CL.gap_statistic(X, 3, rng=g, n_reference=2)
    h = _rng(4)
    h.integers(0, 2, shape=(2,))
    assert g.integers(0, 2**60) == h.integers(0, 2**60)
    r2 = CL.gap_statistic(X, 5, rng=_rng(4), n_reference=3)
    assert bits_equal(_np(r.log_w_reference), _np(r2.log_w_reference))
    lr = _np(r.log_w_reference)
    assert len({tuple(row) for row in lr.tolist()}) == 3  # each reference set its own draw
    assert np.allclose(_np(r.e_log_w), lr.mean(axis=0), rtol=1e-14)
    assert np.allclose(_np(r.se), np.sqrt((1 + 1 / 3) * lr.var(axis=0, ddof=1)), rtol=1e-12)
    assert bits_equal(_np(r.gap), _np(r.e_log_w) - _np(r.log_w))


def test_threads_and_input_order():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(c, 1.0, (150, 3)) for c in (0.0, 4.0, 8.0)])
    out = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        for method in ("kmeans", "pam"):
            out.append(_np(CL.gap_statistic(X, 5, rng=_rng(2), n_reference=12, method=method).log_w_reference))
    sd.set_num_threads(10)
    assert bits_equal(out[0], out[2]) and bits_equal(out[0], out[4])
    assert bits_equal(out[1], out[3]) and bits_equal(out[1], out[5])
    perm = rng.permutation(len(X))
    a = CL.gap_statistic(X, 5, rng=_rng(2), n_reference=12)
    b = CL.gap_statistic(X[perm], 5, rng=_rng(2), n_reference=12)
    assert bits_equal(_np(a.log_w_reference), _np(b.log_w_reference)) and bits_equal(_np(a.log_w), _np(b.log_w))


def test_scale_and_nan():
    X = S.points("blobs/k3p2").X
    r = CL.gap_statistic(X, 4, rng=_rng(0), n_reference=5)
    for f, e in ((2.0**-1000, -1000), (2.0**600, 600)):  # CL43's grid: log W less e log 2
        rs = CL.gap_statistic(X * f, 4, rng=_rng(0), n_reference=5)
        assert np.allclose(_np(rs.log_w), _np(r.log_w) + e * math.log(2), rtol=0, atol=1e-11)
        assert rs.k == r.k
    Xn = X.copy()
    Xn[[2, 40]] = np.nan
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): the input contains NaN \(nan_policy='raise'\)$"):
        CL.gap_statistic(Xn, 4, rng=_rng(0))
    ro = CL.gap_statistic(Xn, 4, rng=_rng(0), n_reference=5, nan_policy="omit")
    rk = CL.gap_statistic(np.delete(X, [2, 40], axis=0), 4, rng=_rng(0), n_reference=5)
    assert ro.nobs == 88 and int(np.count_nonzero(_np(ro.omitted))) == 2
    assert bits_equal(_np(ro.log_w), _np(rk.log_w))


def test_result():
    r = CL.gap_statistic(S.points("blobs/k3p2").X, 6, rng=_rng(1), n_reference=10)
    assert repr(r) == f"GapResult(nobs=90, k={r.k}, k_max=6, method='kmeans')"
    text = r.summary()
    assert "Gap Statistic Results" in text and "scaledpca" in text and f"{r.k} *" in text
    for name in ("gap", "se", "log_w", "e_log_w"):
        assert getattr(r, name).shape == (6,)
    assert r.log_w_reference.shape == (10, 6)
    assert (r.method, r.reference, r.select, r.se_factor, r.n_reference, r.d_power) == (
        "kmeans", "scaledpca", "firstsemax", 1.0, 10, 1.0)  # fmt: skip
    with pytest.raises(AttributeError):
        r.k = 2
    with pytest.raises(TypeError, match=r"^summary\(\): "):
        r.summary(1)


def test_errors():
    X = S.points("blobs/k3p2").X
    g = _rng(0)
    with pytest.raises(TypeError, match=r"^gap_statistic\(\): missing 1 required keyword-only argument: 'rng'$"):
        CL.gap_statistic(X, 4)
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'k_max' must be >= 2, not 1$"):
        CL.gap_statistic(X, 1, rng=g)
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'k_max' \(4\) is larger than the number of distinct"):
        CL.gap_statistic(np.repeat([[0.0], [1.0], [2.0]], 3, axis=0), 4, rng=g)
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'n_reference' must be >= 2"):
        CL.gap_statistic(X, 4, rng=g, n_reference=1)
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'd_power' must be finite and > 0, not 0.0$"):
        CL.gap_statistic(X, 4, rng=g, d_power=0)
    with pytest.raises(
        ValueError, match=r"^gap_statistic\(\): 'select' must be lowercase: 'firstsemax', not 'firstSEmax'$"
    ):
        CL.gap_statistic(X, 4, rng=g, select="firstSEmax")
    with pytest.raises(
        ValueError, match=r"^gap_statistic\(\): 'reference' must be lowercase: 'scaledpca', not 'scaledPCA'$"
    ):
        CL.gap_statistic(X, 4, rng=g, reference="scaledPCA")
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'method' must be 'kmeans' or 'pam', not 'clara'$"):
        CL.gap_statistic(X, 4, rng=g, method="clara")
    with pytest.raises(ValueError, match=r"^gap_statistic\(\): 'se_factor' must be finite and >= 0"):
        CL.gap_statistic(X, 4, rng=g, se_factor=-1)
    with pytest.raises(TypeError, match=r"^gap_statistic\(\): unexpected keyword argument 'B'$"):
        CL.gap_statistic(X, 4, rng=g, B=10)
