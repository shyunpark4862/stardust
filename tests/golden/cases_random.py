"""Golden cases for sd.random (tools/golden.py has the contract).

default_rng's seed forms, every Generator method with small and large (multithreaded) shapes,
broadcast bounds, extreme int64 ranges, weighted and unweighted choice with and without
replacement, permutation of ints and arrays, spawn, the state (read, assigned, copied, pickled),
and the documented errors. Every case returns the drawn values and the state after the draw,
so a change in how much of the stream a call consumes shows too. default_rng(None) (OS
entropy) is not a case.
"""

import copy
import pickle

import numpy as np

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {
    "Generator": "default_rng's class: every method case runs on a Generator and stores its state",
}

I64_MAX, I64_MIN = 2**63 - 1, -(2**63)
LARGE = 2**17 + 3


def cases(L):
    sd = L.sd
    R = sd.random
    a = L.arr

    def g(seed=12345):
        return R.default_rng(seed)

    def draw(fn, seed=12345):
        rng = g(seed)
        return {"r": fn(rng), "state": rng.state}

    # seeds
    for seed in (
        0,
        1,
        2,
        12345,
        2**32,
        2**63,
        2**64 + 7,
        10**40,
        [1, 2, 3],
        (0,),
        [2**70, 5],
        [],
        {3},
        -1,
        1.5,
        "a",
        True,
        [1, -2],
        [1.5],
        None,
    ):
        if seed is None:
            continue
        yield f"default_rng {seed!r}", lambda seed=seed: draw(lambda r: r.random(4), seed)
    yield "default_rng array seed", lambda: draw(lambda r: r.random(3), a(np.array([7, 8, 9])))
    yield "default_rng generator", lambda: (lambda r0: R.default_rng(r0) is r0)(g())
    yield "default_rng repr", lambda: repr(g())
    yield "err default_rng two args", lambda: R.default_rng(1, 2)
    yield "err default_rng keyword", lambda: R.default_rng(seed=1)

    # random
    for shape in (None, 0, 1, 5, (3, 4), (2, 0, 3), (LARGE,), (300, 500), (), -1, 2.0, (2, -1), [2, 3]):
        yield f"random shape={shape!r}", lambda shape=shape: draw(lambda r: r.random(shape))
    yield "random split calls", lambda: draw(lambda r: [r.random(3), r.random(5), r.random()])
    yield "random keyword shape", lambda: draw(lambda r: r.random(shape=3))

    # integers
    ints = [
        ("10", (10,), {}),
        ("0 10", (0, 10), {}),
        ("-5 5", (-5, 5), {}),
        ("1", (1,), {}),
        ("0 1 endpoint", (0, 1), {"endpoint": True}),
        ("5 5 endpoint", (5, 5), {"endpoint": True}),
        ("full int64", (I64_MIN, I64_MAX), {}),
        ("full int64 endpoint", (I64_MIN, I64_MAX), {"endpoint": True}),
        ("2**62", (2**62,), {}),
        ("odd range", (-3, 2**61 + 12345), {}),
        ("max-1 max", (I64_MAX - 1, I64_MAX), {}),
    ]
    for nm, args, kw in ints:
        for shape in (None, 7, (LARGE,)):
            yield (
                f"integers {nm} shape={shape}",
                lambda args=args, kw=kw, shape=shape: draw(lambda r: r.integers(*args, shape=shape, **kw)),
            )
    lo = a(np.array([0, 10, -100, 5], dtype=np.int64))
    hi = a(np.array([[100], [200], [1000]], dtype=np.int64))
    yield "integers broadcast", lambda: draw(lambda r: r.integers(lo, hi))
    yield "integers broadcast shape", lambda: draw(lambda r: r.integers(lo, hi, shape=(2, 3, 4)))
    yield "integers array high", lambda: draw(lambda r: r.integers(a(np.array([1, 2, 3, 2**40], dtype=np.int64))))
    yield "integers large broadcast", lambda: draw(lambda r: r.integers(lo, 50, shape=(LARGE // 4, 4)))
    for nm, args, kw in [
        ("err 0", (0,), {}),
        ("err 5 5", (5, 5), {}),
        ("err 6 5 endpoint", (6, 5), {"endpoint": True}),
        ("err big", (2**63,), {}),
        ("err float", (1.5,), {}),
        ("err bool", (True,), {}),
        ("err high float", (0, 2.5), {}),
        ("err broadcast", (lo, a(np.array([5, 6, 7], dtype=np.int64))), {}),
        ("err endpoint int", (5,), {"endpoint": 1}),
        ("err float array", (a(np.array([1.0])),), {}),
        ("err shape ()", (5,), {"shape": ()}),
        ("err no low", (), {}),
    ]:
        yield f"integers {nm}", lambda args=args, kw=kw: draw(lambda r: r.integers(*args, **kw))

    # choice
    pop = a(np.array([1.5, -2.0, 3.25, 0.0, 7.0, 9.5]))
    p = a(np.array([0.1, 0.0, 0.4, 0.2, 0.3, 0.0]))
    tiny = a(np.array([1e-310, 0.0, 1.0 - 1e-310, 0.0]))
    for nm, aa, kw in [
        ("int", 10, {}),
        ("int shape", 10, {"shape": 20}),
        ("int no replace", 10, {"shape": 10, "replace": False}),
        ("array", pop, {"shape": (3, 5)}),
        ("array p", pop, {"shape": 50, "p": p}),
        ("array p no replace", pop, {"shape": 4, "replace": False, "p": p}),
        ("tiny p no replace", 4, {"shape": 2, "replace": False, "p": tiny}),
        ("big population no replace", 10**6, {"shape": 1000, "replace": False}),
        ("large", 1000, {"shape": LARGE}),
        ("large p", pop, {"shape": LARGE, "p": p}),
        ("scalar", pop, {}),
        ("int array", a(np.arange(5, dtype=np.int64)), {"shape": 4}),
        ("bool array", a(np.array([True, False])), {"shape": 4}),
        ("complex array", a(np.array([1j, 2.0])), {"shape": 3}),
        ("err no replace too many", 5, {"shape": 6, "replace": False}),
        ("err no replace nonzero p", pop, {"shape": 5, "replace": False, "p": p}),
        ("err p sum", pop, {"p": a(np.full(6, 0.2))}),
        ("err p negative", 3, {"p": a(np.array([0.5, -0.1, 0.6]))}),
        ("err p nan", 2, {"p": a(np.array([np.nan, 1.0]))}),
        ("err p length", pop, {"p": a(np.array([0.5, 0.5]))}),
        ("err p complex", 2, {"p": a(np.array([0.5 + 0j, 0.5]))}),
        ("err a 2d", a(np.zeros((2, 2))), {}),
        ("err a 0", 0, {}),
        ("err a negative", -3, {}),
        ("err a float", 2.5, {}),
        ("err replace int", 3, {"replace": 1}),
        ("err a list", [1, 2], {}),
        ("err a big", 2**63, {}),
    ]:
        yield f"choice {nm}", lambda aa=aa, kw=kw: draw(lambda r: r.choice(aa, **kw))

    # permutation
    for nm, x in [
        ("int", 10),
        ("0", 0),
        ("large int", LARGE),
        ("array", pop),
        ("2d", a(np.arange(12.0).reshape(4, 3))),
        ("strided", a(np.arange(40.0).reshape(10, 4))[::-2, 1:]),
        ("large array", a(np.arange(LARGE, dtype=np.int64))),
        ("err negative", -1),
        ("err float", 1.5),
        ("err list", [1, 2]),
        ("err big", 2**63),
    ]:
        yield f"permutation {nm}", lambda x=x: draw(lambda r: r.permutation(x))
    yield "permutation keeps input", lambda: (lambda x: (draw(lambda r: r.permutation(x)), x))(a(np.arange(6.0)))

    # spawn, state, copies
    def spawned():
        r = g()
        kids = r.spawn(3)
        more = r.spawn(2)
        return {
            "kids": [k.state for k in kids + more],
            "draws": [k.random(3) for k in kids + more],
            "parent": r.state,
            "grandkids": [k.state for k in kids[0].spawn(2)],
        }

    yield "spawn", spawned
    yield "spawn 0", lambda: draw(lambda r: r.spawn(0))
    yield "err spawn negative", lambda: draw(lambda r: r.spawn(-1))
    yield "err spawn float", lambda: draw(lambda r: r.spawn(1.0))

    def state_roundtrip():
        r = g()
        r.random(10)
        st = r.state
        x1 = r.random(5)
        r.state = st
        return {"same": r.random(5) == x1, "state": r.state}

    yield "state roundtrip", state_roundtrip
    yield "state after large", lambda: draw(lambda r: (r.random(LARGE).shape, r.integers(10, shape=5)))
    for nm, val in [
        ("not dict", 1),
        ("missing key", {"position": 0}),
        ("bad position", dict(g().state, position=-1)),
        ("bad generator", dict(g().state, bit_generator="PCG64")),
    ]:
        yield f"err state {nm}", lambda val=val: setattr(g(), "state", val)

    def copies():
        r = g()
        r.random(7)
        out = {}
        for nm, c in (
            ("copy", copy.copy(r)),
            ("deepcopy", copy.deepcopy(r)),
            ("pickle", pickle.loads(pickle.dumps(r))),
        ):
            out[nm] = (c.state, c.random(3))
        out["original"] = (r.state, r.random(3))
        return out

    yield "copies", copies
    yield "err random huge", lambda: draw(lambda r: r.random((2**31, 2**31)))
    yield "err permutation huge", lambda: draw(lambda r: r.permutation(2**43))
