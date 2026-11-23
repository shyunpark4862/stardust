"""sd.random (docs/api/random.md, docs/design/random.md)."""

import copy
import pickle
import threading

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd
from stardust.random import _SeedSequence

K = sd._core.random
M64 = 2**64 - 1


def ref_philox(ctr, key):
    c, k = list(ctr), list(key)
    for r in range(10):
        if r:
            k = [(k[0] + 0x9E3779B97F4A7C15) & M64, (k[1] + 0xBB67AE8584CAA73B) & M64]
        p0, p1 = 0xD2E7470EE14C6C93 * c[0], 0xCA5A826395121157 * c[2]
        c = [((p1 >> 64) ^ c[1] ^ k[0]) & M64, p1 & M64, ((p0 >> 64) ^ c[3] ^ k[1]) & M64, p0 & M64]
    return c


@pytest.mark.parametrize(
    "ctr,key,want",
    [
        # Random123's known-answer vectors for philox4x64-10
        ((0, 0, 0, 0), (0, 0), (0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B)),
        ((M64,) * 4, (M64, M64), (0x87B092C3013FE90B, 0x438C3C67BE8D0224, 0x9CC7D7C69CD777B6, 0xA09CAEBF594F0BA0)),
        (
            (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89),
            (0x452821E638D01377, 0xBE5466CF34E90C6C),
            (0xA528F45403E61D95, 0x38C72DBD566E9788, 0xA5A1610E72FD18B5, 0x57BD43B5E52B7FE6),
        ),
    ],
)
def test_philox_known_answers(ctr, key, want):
    assert K.philox_block(*ctr, *key) == want


def test_element_words_follow_the_counter_scheme():
    # element e: word e mod 4 of the block at (e / 4, 0, 0, 0), whatever the grouping
    key = (0x123456789, 0xABCDEF)
    for pos, n in ((0, 64), (5, 40), (13, 3), (2**40 + 7, 50)):
        got = [int(v) for v in np.asarray(K.bits(*key, pos, n)).view(np.uint64)]
        assert got == [ref_philox((e >> 2, 0, 0, 0), key)[e & 3] for e in range(pos, pos + n)]


@pytest.mark.parametrize("seed", [0, 1, 12345, 2**32, 2**64 + 3, 2**200 + 7, (1, 2, 3), (0,), (2**70, 5)])
def test_keys_are_numpys_seedsequence(seed):
    want = tuple(int(v) for v in np.random.SeedSequence(seed).generate_state(2, np.uint64))
    assert _SeedSequence(seed).key() == want


def test_spawn_matches_numpy_and_is_independent():
    kids = np.random.SeedSequence(42).spawn(3)
    gens = sd.random.default_rng(42).spawn(3)
    for k, g in zip(kids, gens):
        assert g._key == tuple(int(v) for v in k.generate_state(2, np.uint64))
    a, b = (np.asarray(g.random(10_000)) for g in gens[:2])
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.05
    # spawning again continues the numbering
    more = sd.random.default_rng(42)
    more.spawn(3)
    assert more.spawn(1)[0]._key == tuple(
        int(v) for v in np.random.SeedSequence(42, spawn_key=(3,)).generate_state(2, np.uint64)
    )


def test_the_same_on_any_thread_count_and_split_calls():
    before = sd.get_num_threads()
    try:
        outs = []
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            r = sd.random.default_rng(7)
            outs.append(
                [
                    np.asarray(r.random(300_001)).tobytes(),
                    np.asarray(r.integers(-(10**18), 10**18, shape=200_003)).tobytes(),
                    np.asarray(r.permutation(50_000)).tobytes(),
                    np.asarray(r.choice(1000, shape=100_000)).tobytes(),
                ]
            )
        assert outs[0] == outs[1] == outs[2]
    finally:
        sd.set_num_threads(before)
    a = sd.random.default_rng(9)
    whole = np.asarray(a.random(10_000))
    b = sd.random.default_rng(9)
    parts = np.concatenate([np.asarray(b.random(1)), np.asarray(b.random(3_333)), np.asarray(b.random(6_666))])
    assert whole.tobytes() == parts.tobytes()


def test_scalars_and_shapes():
    r = sd.random.default_rng(1)
    assert type(r.random()) is float and type(r.integers(10)) is int
    assert r.random(5).shape == (5,) and r.random((2, 3)).shape == (2, 3) and r.random((0,)).shape == (0,)
    assert r.integers(0, sd.asarray([1, 2, 3])).shape == (3,)
    assert r.integers(sd.asarray([[0], [10]]), sd.asarray([5, 50, 500]) + 20).shape == (2, 3)
    assert r.choice(5, shape=(2, 2)).shape == (2, 2) and type(r.choice(5)) is int


def test_bounds():
    r = sd.random.default_rng(2)
    x = np.asarray(r.integers(-3, 4, shape=100_000))
    assert x.min() == -3 and x.max() == 3
    x = np.asarray(r.integers(-3, 4, shape=100_000, endpoint=True))
    assert x.min() == -3 and x.max() == 4
    x = np.asarray(r.integers(-(2**63), 2**63 - 1, shape=1000, endpoint=True))  # the full range (r = 2^64)
    assert x.min() < -(2**61) and x.max() > 2**61
    assert r.integers(5, 6) == 5 and r.integers(5, 5, endpoint=True) == 5
    u = np.asarray(r.random(1_000_000))
    assert u.min() >= 0.0 and u.max() < 1.0


def test_statistics():
    r = sd.random.default_rng(3)
    u = np.asarray(r.random(200_000))
    assert st.kstest(u, "uniform").pvalue > 1e-4
    k = np.asarray(r.integers(0, 7, shape=700_000))
    assert st.chisquare(np.bincount(k, minlength=7)).pvalue > 1e-4
    big = np.asarray(r.integers(0, 3 * 2**61, shape=300_000))  # a range where rejection matters
    assert st.chisquare(np.bincount(big // 2**61, minlength=3)).pvalue > 1e-4
    perms = np.asarray([np.asarray(r.permutation(4)) for _ in range(24_000)])
    codes = perms @ np.array([64, 16, 4, 1])
    assert len(np.unique(codes)) == 24 and st.chisquare(np.unique(codes, return_counts=True)[1]).pvalue > 1e-4
    p = sd.asarray([0.1, 0.0, 0.6, 0.3])
    c = np.asarray(r.choice(4, shape=200_000, p=p))
    counts = np.bincount(c, minlength=4)
    assert counts[1] == 0 and st.chisquare(counts[[0, 2, 3]], 200_000 * np.array([0.1, 0.6, 0.3])).pvalue > 1e-4
    w = np.asarray(
        [np.asarray(r.choice(4, shape=2, replace=False, p=sd.asarray([0.5, 0.25, 0.25, 0.0]))) for _ in range(20_000)]
    )
    assert not (w == 3).any() and (w[:, 0] != w[:, 1]).all()
    first = np.bincount(w[:, 0], minlength=3)[:3] / 20_000
    assert abs(first[0] - 0.5) < 0.02  # the first draw of a weighted sample follows p


def test_without_replacement_and_permutations_of_arrays():
    r = sd.random.default_rng(4)
    s = np.asarray(r.choice(10**9, shape=1000, replace=False))
    assert len(set(s)) == 1000 and s.min() >= 0 and s.max() < 10**9
    assert sorted(np.asarray(r.choice(50, shape=50, replace=False))) == list(range(50))
    x = sd.asarray(np.arange(12.0).reshape(6, 2))
    y = np.asarray(r.permutation(x))
    assert sorted(map(tuple, y)) == sorted(map(tuple, np.asarray(x))) and np.array_equal(
        np.asarray(x), np.arange(12.0).reshape(6, 2)
    )
    pool = sd.asarray(np.array([10, 20, 30]))
    assert set(np.asarray(r.choice(pool, shape=100))) <= {10, 20, 30}


def test_state_pickle_copy():
    r = sd.random.default_rng(5)
    r.random(17)
    st_ = r.state
    assert st_["position"] == 17 and st_["bit_generator"] == "Philox4x64-10"
    a, b, c = pickle.loads(pickle.dumps(r)), copy.copy(r), copy.deepcopy(r)
    assert a.random(5).tolist() == b.random(5).tolist() == c.random(5).tolist() == r.random(5).tolist()
    s = sd.random.default_rng(0)
    s.state = st_
    assert s.random() == sd.random.default_rng(5).random(18)[17]
    assert sd.random.default_rng(r) is r


def test_threads_share_a_generator_without_reusing_numbers():
    r = sd.random.default_rng(6)
    got = []

    def work():
        for _ in range(50):
            got.append(np.asarray(r.integers(0, 2**62, shape=1000)))

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    allv = np.concatenate(got)
    assert r.state["position"] == 200_000 and len(np.unique(allv)) == allv.size  # no range handed out twice


@pytest.mark.parametrize(
    "call,exc,msg",
    [
        (lambda r: sd.random.default_rng(-1), ValueError, "default_rng(): 'seed' must be >= 0"),
        (
            lambda r: sd.random.default_rng(1.5),
            TypeError,
            "default_rng(): 'seed' must be an int, a sequence of ints or None, not float",
        ),
        (lambda r: sd.random.default_rng(True), TypeError, "not bool"),
        (lambda r: r.random(-1), ValueError, "random(): negative dimensions are not allowed"),
        (lambda r: r.random(2.0), TypeError, "random(): 'shape' must be an int or a tuple of ints, not float"),
        (lambda r: r.integers(5, 5), ValueError, "integers(): 'low' must be less than 'high'"),
        (
            lambda r: r.integers(6, 5, endpoint=True),
            ValueError,
            "integers(): 'low' must be less than or equal to 'high'",
        ),
        (lambda r: r.integers(0, 2**63), OverflowError, "integers(): 'high' is out of range"),
        (lambda r: r.integers(0.5, 3), TypeError, "integers(): 'low' must be an int or an int64 array, not float"),
        (lambda r: r.integers(0, sd.asarray([1.0])), TypeError, "'high' must be an int or an int64 array, not float64"),
        (lambda r: r.integers(0, 3, endpoint=1), TypeError, "integers(): 'endpoint' must be True or False, not int"),
        (lambda r: r.choice(3, shape=4, replace=False), ValueError, "choice(): cannot take a larger sample"),
        (lambda r: r.choice(3, p=sd.asarray([0.5, 0.6, -0.1])), ValueError, "choice(): 'p' must be >= 0"),
        (
            lambda r: r.choice(3, p=sd.asarray([0.5, 0.6, 0.1])),
            ValueError,
            "choice(): the probabilities 'p' must sum to 1",
        ),
        (lambda r: r.choice(sd.asarray(np.zeros((2, 2)))), ValueError, "choice(): 'a' must be 1-dimensional"),
        (lambda r: r.choice(0), ValueError, "choice(): cannot choose from an empty population"),
        (lambda r: r.permutation(-2), ValueError, "permutation(): 'x' must be >= 0"),
        (lambda r: r.permutation("a"), TypeError, "permutation(): 'x' must be an int or an array, not str"),
    ],
)
def test_errors(call, exc, msg):
    with pytest.raises(exc, match=msg.replace("(", r"\(").replace(")", r"\)")):
        call(sd.random.default_rng(0))


def test_choice_p_within_numpys_tolerance():
    r = sd.random.default_rng(8)
    p = sd.asarray([0.25, 0.25, 0.25, 0.25 + 1e-9])  # sums to 1 + 1e-9: accepted, normalised
    assert set(np.asarray(r.choice(4, shape=1000, p=p))) <= {0, 1, 2, 3}


# ---- 2026-11-18: messages, tiny probabilities, the stream against numpy's Philox ----


def test_integers_rejects_in_a_range_where_skipping_it_shows():
    # Without the rejection step a range of 3 * 2**61 gives the values 3, 3, 2 preimages in
    # turn (2**64 / range = 8/3): the residues mod 3 would come out 3/8, 3/8, 2/8.
    r = sd.random.default_rng(11)
    big = np.asarray(r.integers(0, 3 * 2**61, shape=300_000).tolist())
    counts = np.bincount(big % 3, minlength=3)
    assert abs(counts / big.size - 1 / 3).max() < 0.01


def test_stream_is_numpys_philox_four_elements_on():
    # the same key and counter-based stream; numpy starts one block (4 elements) later
    r = sd.random.default_rng(12345)
    ours = np.asarray(r.random((1000,)).tolist())
    key = np.random.SeedSequence(12345).generate_state(2, np.uint64)
    theirs = np.random.Generator(np.random.Philox(key=key)).random(996)
    assert np.array_equal(ours[4:], theirs)


def test_messages_name_the_method_and_the_argument():
    r = sd.random.default_rng(1)
    with pytest.raises(OverflowError, match=r"^choice\(\): 'a' is out of range"):
        r.choice(2**63)
    with pytest.raises(OverflowError, match=r"^choice\(\): 'a' is out of range"):
        r.choice(2**63, shape=2, replace=False)
    with pytest.raises(OverflowError, match=r"^permutation\(\): 'x' is out of range"):
        r.permutation(2**63)
    for call, name in (
        (lambda: r.random(()), "random"),
        (lambda: r.choice(5, shape=()), "choice"),
        (lambda: r.integers(5, shape=()), "integers"),
    ):
        with pytest.raises(ValueError, match=rf"^{name}\(\): 0-dimensional arrays are not supported \(stardust"):
            call()
    with pytest.raises(ValueError, match=r"^integers\(\): 'high' must be positive"):
        r.integers(0)
    with pytest.raises(ValueError, match=r"^integers\(\): 'high' must be >= 0"):
        r.integers(-1, endpoint=True)
    with pytest.raises(ValueError, match=r"^integers\(\): 'low' must be less than 'high'"):
        r.integers(3, 3)
    with pytest.raises(TypeError, match=r"^Generator\(\): make a generator with sd.random.default_rng"):
        sd.random.Generator()
    with pytest.raises(TypeError, match=r"^choice\(\): 'p' must be a real array"):
        r.choice(3, p=sd.asarray([0.5 + 0j, 0.25, 0.25]))


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("entropy", -1, ValueError),
        ("entropy", "a", TypeError),
        ("spawn_key", 5, TypeError),
        ("spawn_key", (True,), TypeError),
        ("n_children_spawned", -1, ValueError),
        ("position", 1.0, TypeError),
        ("position", True, TypeError),
        ("position", -1, ValueError),
    ],
)
def test_state_errors_share_one_prefix(key, value, error):
    r = sd.random.default_rng(1)
    before = r.state
    with pytest.raises(error, match=rf"^\.state: '{key}' "):
        r.state = {**before, key: value}
    assert r.state == before


def test_choice_keeps_a_tiny_probability_ahead_of_zero():
    # 1 / 1e-320 overflows: the element must still be drawn before one of probability 0
    r = sd.random.default_rng(3)
    p = sd.asarray([0.0, 1.0, 1e-320])
    for _ in range(50):
        assert sorted(r.choice(3, shape=2, replace=False, p=p).tolist()) == [1, 2]


def test_state_refuses_entropy_none():
    # None would reseed from the OS on every assignment: not the state it claims to be
    st = sd.random.default_rng(1).state
    st["entropy"] = None
    g = sd.random.default_rng(0)
    before = g.state
    with pytest.raises(TypeError, match=r"^\.state: 'entropy' must be an int or a sequence of ints, not NoneType$"):
        g.state = st
    assert g.state == before
    st["entropy"] = "x"
    with pytest.raises(TypeError, match=r"^\.state: 'entropy' must be an int or a sequence of ints, not str$"):
        g.state = st
    st["entropy"] = [1, True]
    with pytest.raises(TypeError, match=r"^\.state: 'entropy' must be an int or a sequence of ints, not a sequence"):
        g.state = st
    # default_rng still takes None (OS entropy)
    with pytest.raises(
        TypeError, match=r"^default_rng\(\): 'seed' must be an int, a sequence of ints or None, not str$"
    ):
        sd.random.default_rng("x")


def test_state_spawn_key_messages_agree():
    st = sd.random.default_rng(1).state
    g = sd.random.default_rng(0)
    st["spawn_key"] = 5
    with pytest.raises(TypeError, match=r"^\.state: 'spawn_key' must be a sequence of ints, not int$"):
        g.state = st
    st["spawn_key"] = [True]
    with pytest.raises(TypeError, match=r"^\.state: 'spawn_key' must be a sequence of ints, not a sequence of bool$"):
        g.state = st
    st["spawn_key"] = [1.5]
    with pytest.raises(TypeError, match=r"^\.state: 'spawn_key' must be a sequence of ints, not a sequence of float$"):
        g.state = st
    st["spawn_key"] = [-1]
    with pytest.raises(ValueError, match=r"^\.state: 'spawn_key' must be >= 0$"):
        g.state = st


def test_messages_in_the_shared_forms():
    r = sd.random.default_rng(1)
    with pytest.raises(TypeError, match=r"^choice\(\): 'p' must be an sd\.ndarray, not list$"):
        r.choice(3, p=[0.2, 0.3, 0.5])
    with pytest.raises(TypeError, match=r"^choice\(\): 'p' must be a real array, not complex128$"):
        r.choice(2, p=sd.asarray([0.5 + 0j, 0.5]))
    with pytest.raises(TypeError, match=r"^integers\(\): 'high' must be an int or an int64 array, not float64$"):
        r.integers(sd.asarray([1.0]))
    # shape: the array area's sentences (sd.zeros(()), sd.zeros(-1))
    for call, name in ((lambda: r.random(()), "random"), (lambda: r.integers(3, shape=()), "integers")):
        with pytest.raises(ValueError, match=rf"^{name}\(\): 0-dimensional arrays are not supported"):
            call()
    with pytest.raises(ValueError, match=r"^zeros\(\): 0-dimensional arrays are not supported"):
        sd.zeros(())
    for call in (lambda: r.random(-1), lambda: r.random((2, -1)), lambda: r.choice(3, shape=(-2,))):
        with pytest.raises(ValueError, match=r"^\w+\(\): negative dimensions are not allowed$"):
            call()


@pytest.mark.parametrize(
    "n, k",
    [(1000, 1000), (1000, 600), (1000, 62), (1000, 10), (1, 1), (0, 0)]
    # the hash table path (k <= n / 16) around its 16 draws made ahead, and at its edge
    + [(65536, 4096), (65536, 4097), (65536, 15), (65536, 16), (65536, 17), (65536, 1), (100003, 3001)],
)
def test_choice_without_replacement_is_the_permutations_head(n, k):
    # permutation(n) shuffles in its result, choice's paths in an array of n or a hash table:
    # the same steps, the same draws
    perm = sd.random.default_rng(5).permutation(n)
    head = sd.random.default_rng(5).choice(n, shape=k, replace=False)
    assert head.tolist() == perm.tolist()[:k]


@pytest.mark.parametrize(
    "call, msg",
    [
        # default_rng: the library's form (API.md §2.1)
        (
            lambda r: sd.random.default_rng(seed=1),
            "default_rng(): got some positional-only arguments passed as keyword arguments: 'seed'",
        ),
        (
            lambda r: sd.random.default_rng(1, 2),
            "default_rng(): takes from 0 to 1 positional arguments but 2 were given",
        ),
        # the Generator's methods: Python's own form (small calls, API.md §2.1)
        (lambda r: r.random(1, 2), "Generator.random() takes from 1 to 2 positional arguments but 3 were given"),
        (lambda r: r.choice(3, sise=2), "Generator.choice() got an unexpected keyword argument 'sise'"),
    ],
)
def test_call_shape_errors(call, msg):
    r = sd.random.default_rng(1)
    with pytest.raises(TypeError) as e:
        call(r)
    assert str(e.value) == msg


def test_sample_shape_messages_are_the_arrays():
    r = sd.random.default_rng(1)
    with pytest.raises(
        ValueError, match=r"^sample\(\): 0-dimensional arrays are not supported \(stardust has no 0-d arrays\)$"
    ):
        sd.stats.Normal().sample((), rng=r)
    with pytest.raises(ValueError, match=r"^sample\(\): negative dimensions are not allowed$"):
        sd.stats.Normal().sample((2, -1), rng=r)


def test_errors_from_inside_a_method_pass_unchanged():
    r = sd.random.default_rng(1)
    with pytest.raises(TypeError, match=r"^random\(\): 'shape' must be an int or a tuple of ints, not str$"):
        r.random("3")


def test_kernels_match_their_reference_forms():
    # the fast kernels give the plain forms' values (csrc/random/random.cpp): random()'s scalar,
    # integers' batched rejections (half the words rejected at r = 2^63 + 1), the weighted
    # sample without replacement (the k smallest race keys in stable order, ties included)
    import numpy as np

    k = sd._core.random
    K0, K1 = 0x1234567, 0xDEADBEEF12345
    for pos in [*range(40), 2**64 - 1, 2**63 + 5]:
        assert k.uniform1(K0, K1, pos) == float(k.uniform(K0, K1, pos, (1,))[0])
    n = 5000
    for lo, hi in ((-(2**62), 2**62 + 1), (0, 3 * 2**61), (0, 1000)):
        scalar = k.integers(
            K0, K1, 77, sd.broadcast_to(sd.asarray([lo]), (n,)), sd.broadcast_to(sd.asarray([hi]), (n,)), False
        )
        general = k.integers(K0, K1, 77, sd.full((n,), lo, dtype=sd.int64), sd.full((n,), hi, dtype=sd.int64), False)
        assert np.array_equal(np.asarray(scalar), np.asarray(general))
    rng = np.random.default_rng(3)
    for n, kk in ((20000, 300), (500, 300), (1000, 1000), (10, 0)):
        p = rng.dirichlet(np.ones(n))
        p[::7] = 0
        p[2] = p[5] = 1e-310  # keys cut at the largest double: ties, ordered by index
        p /= p.sum()
        ps = sd.asarray(p)
        got = np.asarray(k.race_choice(K0, K1, 9, ps, kk))
        want = np.asarray(sd.argsort(k.race_keys(K0, K1, 9, ps)))[:kk]
        assert np.array_equal(got, want)
