"""A request beyond physical memory is a MemoryError named by the method, and the generator
does not move (API.md §2)."""

import pytest

import stardust as sd


def _calls():
    S = sd.stats
    mvn = S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray([[1.0, 0.0], [0.0, 1.0]]))
    return [
        ("permutation", lambda r: r.permutation(2**40)),
        ("random", lambda r: r.random(2**40)),
        ("integers", lambda r: r.integers(10, shape=2**40)),
        ("choice", lambda r: r.choice(10, shape=2**40)),
        ("choice", lambda r: r.choice(10, shape=2**40, p=sd.full(10, 0.1))),
        ("sample", lambda r: S.Normal().sample(2**40, rng=r)),
        ("sample", lambda r: mvn.sample(2**39, rng=r)),
        ("sample", lambda r: S.Wishart(df=3.0, scale=sd.asarray([[1.0, 0.0], [0.0, 1.0]])).sample(2**39, rng=r)),
    ]


def _memory_calls():
    # The memory a call takes besides its 8-byte indices (checked before the stream moves):
    # the shuffle's own entries, the elements copied out of an array.
    mem = sd._core._physical_memory()
    return [
        # a small sample of a huge population: a hash map, 64 bytes per element beside the result
        ("choice", lambda r: r.choice(2**62, shape=mem // 64, replace=False)),
        # a large sample: an array of the whole population beside the result
        ("choice", lambda r: r.choice(mem // 8, shape=mem // 64, replace=False)),
        # complex elements, 16 bytes each, taken from a population array
        ("choice", lambda r: r.choice(sd.broadcast_to(sd.zeros(1, dtype=sd.complex128), (2**40,)), shape=mem // 12)),
        # an array's permuted copy: n rows of 4096 elements
        ("permutation", lambda r: r.permutation(sd.broadcast_to(sd.zeros((1, 4096)), (2**26, 4096)))),
    ]


@pytest.mark.parametrize("i", range(8 + 4))
def test_memory_error_named_and_stream_untouched(i):
    name, call = (_calls() + _memory_calls())[i]
    rng = sd.random.default_rng(1)
    rng.random(3)
    before = rng.state["position"]
    with pytest.raises(
        MemoryError, match=rf"^{name}\(\): cannot allocate \d+ bytes: more than this machine's physical memory$"
    ):
        call(rng)
    assert rng.state["position"] == before
