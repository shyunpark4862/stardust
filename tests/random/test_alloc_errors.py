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


@pytest.mark.parametrize("i", range(8))
def test_memory_error_named_and_stream_untouched(i):
    name, call = _calls()[i]
    rng = sd.random.default_rng(1)
    rng.random(3)
    before = rng.state["position"]
    with pytest.raises(
        MemoryError, match=rf"^{name}\(\): cannot allocate \d+ bytes: more than this machine's physical memory$"
    ):
        call(rng)
    assert rng.state["position"] == before
