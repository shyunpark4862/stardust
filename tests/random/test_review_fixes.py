"""Bugs found by a code review: the state's entropy, the messages, the shuffle in place."""

import pytest

import stardust as sd


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
    with pytest.raises(ValueError, match=r"^\.state: 'spawn_key' must be non-negative$"):
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


@pytest.mark.parametrize("n, k", [(1000, 1000), (1000, 600), (1000, 62), (1000, 10), (1, 1), (0, 0)])
def test_choice_without_replacement_is_the_permutations_head(n, k):
    # permutation(n) shuffles in its result, choice's paths in an array of n or a hash map:
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
