"""stardust.random: counter-based random numbers (docs/api/random.md, docs/design/random.md).

Usage: ``rng = sd.random.default_rng(seed)`` then ``rng.random(shape)``,
``rng.integers(low, high)``, ``rng.choice(a)``, ``rng.permutation(x)``. There
are no global functions and no global state (R3).

The generator is Philox4x64-10 (R5). Element e of the stream -- e is the
generator's position plus the element's index in the call -- is a pure function
of the key and e (csrc/random/philox.hpp), so the values do not depend on the
thread count, and a call split in two gives the same leading elements. The key
comes from the seed by numpy's SeedSequence hash (R6): the same key numpy's
Philox would get, but the counters are used differently, so the values are not
numpy's.
"""

import functools
import operator
import secrets
import threading

from .. import _calls
from .. import _core as _sd_core

_k = _sd_core.random

__all__ = ["default_rng", "Generator"]

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _type_name(v):
    return _sd_core._type_name(v)  # (numpy's types as numpy.bool: API.md §2.1)


# ---- seeding: numpy's SeedSequence, ported (numpy/random/bit_generator.pyx) ----------
#
# Ported rather than called so that sd.random does not need numpy; checked against
# numpy.random.SeedSequence(...).generate_state(2, numpy.uint64) in the tests.

_INIT_A, _MULT_A = 0x43B0D7E5, 0x931E8875
_INIT_B, _MULT_B = 0x8B51F9DD, 0x58F38DED
_MIX_MULT_L, _MIX_MULT_R = 0xCA01F9DD, 0x4973F715
_XSHIFT = 16
_POOL = 4


def _words(n):
    """A non-negative int as little-endian 32-bit words (0 -> [0])."""
    out = [n & _MASK32]
    n >>= 32
    while n:
        out.append(n & _MASK32)
        n >>= 32
    return out


def _entropy_words(x):
    if isinstance(x, int):
        return _words(x)
    out = []
    for v in x:
        out.extend(_entropy_words(v))
    return out


class _HashConst:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v


def _hashmix(value, h):
    value ^= h.v
    h.v = (h.v * _MULT_A) & _MASK32
    value = (value * h.v) & _MASK32
    return value ^ (value >> _XSHIFT)


def _mix(x, y):
    r = (_MIX_MULT_L * x - _MIX_MULT_R * y) & _MASK32
    return r ^ (r >> _XSHIFT)


class _SeedSequence:
    """numpy's SeedSequence (pool size 4): entropy, spawn key, children spawned so far."""

    def __init__(self, entropy, spawn_key=(), n_children_spawned=0):
        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.n_children_spawned = n_children_spawned
        run = _entropy_words(entropy)
        spawn = _entropy_words(self.spawn_key)
        if spawn and len(run) < _POOL:
            run = run + [0] * (_POOL - len(run))
        data = run + spawn
        h = _HashConst(_INIT_A)
        pool = [_hashmix(data[i] if i < len(data) else 0, h) for i in range(_POOL)]
        for src in range(_POOL):
            for dst in range(_POOL):
                if src != dst:
                    pool[dst] = _mix(pool[dst], _hashmix(pool[src], h))
        for src in range(_POOL, len(data)):
            for dst in range(_POOL):
                pool[dst] = _mix(pool[dst], _hashmix(data[src], h))
        self._pool = pool

    def key(self):
        """generate_state(2, uint64): the 128-bit Philox key as two 64-bit ints."""
        h = _INIT_B
        words = []
        for i in range(4):
            v = self._pool[i % _POOL] ^ h
            h = (h * _MULT_B) & _MASK32
            v = (v * h) & _MASK32
            words.append(v ^ (v >> _XSHIFT))
        return (words[0] | (words[1] << 32), words[2] | (words[3] << 32))

    def spawn(self, n):
        first = self.n_children_spawned
        self.n_children_spawned += n
        return [_SeedSequence(self.entropy, self.spawn_key + (i,)) for i in range(first, first + n)]


def _seed_entropy(seed, name, arg="seed", allow_none=True):
    """The seed as SeedSequence entropy: None (OS entropy, when allowed), an int >= 0, or a
    sequence of them."""
    kinds = "an int, a sequence of ints or None" if allow_none else "an int or a sequence of ints"
    if seed is None:
        if not allow_none:
            raise TypeError(f"{name}(): '{arg}' must be {kinds}, not NoneType")
        return secrets.randbits(128)
    if isinstance(seed, bool):
        raise TypeError(f"{name}(): '{arg}' must be {kinds}, not bool")
    try:
        v = operator.index(seed)
    except TypeError:
        v = None
    if v is not None:
        if v < 0:
            raise ValueError(f"{name}(): '{arg}' must be non-negative")
        return v
    if isinstance(seed, (str, bytes)) or not hasattr(seed, "__len__"):
        raise TypeError(f"{name}(): '{arg}' must be {kinds}, not {_type_name(seed)}")
    out = []
    for item in seed:
        if isinstance(item, bool):
            raise TypeError(f"{name}(): '{arg}' must be {kinds}, not a sequence of bool")
        try:
            iv = operator.index(item)
        except TypeError:
            raise TypeError(f"{name}(): '{arg}' must be {kinds}, not a sequence of {_type_name(item)}") from None
        if iv < 0:
            raise ValueError(f"{name}(): '{arg}' must be non-negative")
        out.append(iv)
    return tuple(out)


# ---- argument helpers -----------------------------------------------------------------


# The array area's sentences for a shape it cannot make (sd.zeros(()), sd.zeros(-1)).
_NO_0D = "0-dimensional arrays are not supported (stardust has no 0-d arrays)"
_NEGATIVE = "negative dimensions are not allowed"


def _extent(d, name):
    """One extent of `shape`: an int >= 0 (_count's checks; a negative one in the array
    area's sentence)."""
    if not isinstance(d, bool) and hasattr(d, "__index__") and operator.index(d) < 0:
        raise ValueError(f"{name}(): {_NEGATIVE}")
    return _count(d, name, "shape")


def _shape(shape, name):
    """None (a scalar), an int or a tuple of ints >= 0, as a tuple (None stays None)."""
    if shape is None:
        return None
    if isinstance(shape, bool):
        raise TypeError(f"{name}(): 'shape' must be an int or a tuple of ints, not bool")
    try:
        return (_extent(shape, name),)
    except TypeError:
        pass
    if not isinstance(shape, tuple):
        raise TypeError(f"{name}(): 'shape' must be an int or a tuple of ints, not {_type_name(shape)}")
    return tuple(_extent(d, name) for d in shape)


def _result_shape(shape, name):
    """_shape for the Generator's own methods, whose result has exactly this shape: () and
    negative extents are refused in the array area's sentences (no 0-d arrays)."""
    shp = _shape(shape, name)
    if shp == ():
        raise ValueError(f"{name}(): {_NO_0D}")
    return shp


def _sample_shape(shape):
    """sd.stats' sample(shape=None): None is one sample per parameter set (the parameters'
    shape, a float when they are scalars); an int or a tuple is added before that shape. () is
    refused as for the Generator's methods, in the same sentences: no 0-d arrays."""
    shp = _shape(shape, "sample")
    if shp is None:
        return ()
    if shp == ():
        raise ValueError(f"sample(): {_NO_0D}")
    return shp


def _count(v, name, arg):
    if isinstance(v, bool):
        raise TypeError(f"{name}(): '{arg}' must be an int, not bool")
    try:
        n = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {_type_name(v)}") from None
    if n < 0:
        raise ValueError(f"{name}(): '{arg}' must be non-negative")
    if n >= 2**63:
        raise OverflowError(f"{name}(): '{arg}' is out of range")
    return n


def _spawn_key_item(v):
    """One entry of a state's spawn_key: an int >= 0 (the messages of the sequence check)."""
    if isinstance(v, bool) or not hasattr(v, "__index__"):
        raise TypeError(f".state(): 'spawn_key' must be a sequence of ints, not a sequence of {_type_name(v)}")
    return _count(v, ".state", "spawn_key")


def _shuffle_bytes(n, k):
    """The memory _k.fisher_yates(…, n, k) takes, by its path (csrc/random/random.cpp): the
    result (8 k) and, for k < n, the shuffled entries -- a flat hash table of bit_ceil(2 k)
    16-byte slots (below 64 bytes per entry; 64 k is the bound) for k <= n / 16, else an
    array of n."""
    if k == n:
        return 8 * k
    if k <= n // 16:
        return 8 * k + 64 * k
    return 8 * k + 8 * n


def _check_alloc(nbytes, name):
    """MemoryError, with the caller's name, for a result larger than physical memory (the
    check every allocation makes, base/buffer.cpp; made here before the stream moves)."""
    if nbytes > _sd_core._physical_memory():
        raise MemoryError(f"{name}(): cannot allocate {nbytes} bytes: more than this machine's physical memory")


def _size(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _bool_opt(v, name, arg):
    if v is not True and v is not False:
        raise TypeError(f"{name}(): '{arg}' must be True or False, not {_type_name(v)}")
    return v


def _int_bound(v, name, arg):
    """An integers() bound: a Python int (or numpy integer) in int64, or an int64 ndarray."""
    import stardust as sd

    if isinstance(v, sd.ndarray):
        if v.dtype != sd.int64:
            raise TypeError(f"{name}(): '{arg}' must be an int or an int64 array, not {v.dtype}")
        return v
    if isinstance(v, bool):
        raise TypeError(f"{name}(): '{arg}' must be an int or an int64 array, not bool")
    try:
        iv = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int or an int64 array, not {_type_name(v)}") from None
    if not -(2**63) <= iv < 2**63:
        raise OverflowError(f"{name}(): '{arg}' is out of range")
    return iv


# ---- the Generator ----------------------------------------------------------------------


class Generator:
    """Random numbers from Philox4x64-10 at a position in its stream (docs/api/random.md §1).

    Make one with default_rng(seed). Calls take consecutive ranges of the stream;
    calls from several Python threads at once reserve their ranges atomically (R8),
    so no number is used twice, but which thread gets which range depends on the
    schedule: for reproducible threads, give each its own spawn().
    """

    __slots__ = ("_seq", "_key", "_pos", "_lock")

    def __init__(self, _seq=None, _pos=0):
        if not isinstance(_seq, _SeedSequence):
            raise TypeError("Generator(): make a generator with sd.random.default_rng(seed)")
        self._seq = _seq
        self._key = _seq.key()
        self._pos = _pos
        self._lock = threading.Lock()

    def _reserve(self, n):
        """Take n elements of the stream: (key, position), read together under the lock, so
        that a `state` assigned from another thread is seen whole or not at all."""
        with self._lock:
            key, p = self._key, self._pos
            self._pos = (p + n) & _MASK64
        return key, p

    def _take(self, n, name, nbytes=None):
        """_reserve(n) for a call `name` whose result takes nbytes (8 per element unless
        given): a request beyond the machine's physical memory is a MemoryError before the
        stream moves, so a refused call leaves the generator where it was (API.md §2)."""
        _check_alloc(8 * n if nbytes is None else nbytes, name)
        return self._reserve(n)

    def __repr__(self):
        return "Generator(Philox4x64-10)"

    # -- state, spawning, copies, pickling (R7)

    @property
    def state(self):
        """The generator's state: a dict with its seed sequence and position (writable)."""
        with self._lock:
            seq, pos = self._seq, self._pos
        return {
            "bit_generator": "Philox4x64-10",
            "entropy": seq.entropy,
            "spawn_key": seq.spawn_key,
            "n_children_spawned": seq.n_children_spawned,
            "position": pos,
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError(f".state: must be a dict, not {_type_name(value)}")
        need = ("bit_generator", "entropy", "spawn_key", "n_children_spawned", "position")
        missing = [k for k in need if k not in value]
        if missing:
            raise ValueError(f".state: missing key '{missing[0]}'")
        if value["bit_generator"] != "Philox4x64-10":
            raise ValueError(f".state: 'bit_generator' must be 'Philox4x64-10', not {value['bit_generator']!r}")
        try:
            # a state is a fixed point of the stream: None (fresh OS entropy) is not one
            entropy = _seed_entropy(value["entropy"], ".state", "entropy", allow_none=False)
            if isinstance(value["spawn_key"], (str, bytes)) or not hasattr(value["spawn_key"], "__iter__"):
                raise TypeError(
                    f".state(): 'spawn_key' must be a sequence of ints, not {_type_name(value['spawn_key'])}"
                )
            spawn_key = tuple(_spawn_key_item(v) for v in value["spawn_key"])
            n_spawned = _count(value["n_children_spawned"], ".state", "n_children_spawned")
        except (TypeError, ValueError, OverflowError) as e:  # the helpers write "name(): "
            raise type(e)(str(e).replace(".state(): ", ".state: ", 1)) from None
        pos = value["position"]
        if isinstance(pos, bool) or not hasattr(pos, "__index__"):
            raise TypeError(f".state: 'position' must be an int, not {_type_name(pos)}")
        pos = operator.index(pos)
        if not 0 <= pos <= _MASK64:
            raise ValueError(".state: 'position' must be an int in [0, 2**64)")
        seq = _SeedSequence(entropy, spawn_key, n_spawned)
        with self._lock:
            self._seq, self._key, self._pos = seq, seq.key(), pos

    def spawn(self, n, /):
        """n independent generators, each with its own key (numpy's SeedSequence.spawn)."""
        n = _count(n, "spawn", "n")
        with self._lock:
            seqs = self._seq.spawn(n)
        return [Generator(s) for s in seqs]

    def __reduce__(self):
        return (_restore, (self.state,))

    def __copy__(self):
        return _restore(self.state)

    def __deepcopy__(self, memo):
        return _restore(self.state)

    # -- the tools (R4)

    def random(self, shape=None, /):
        """Uniform float64 on [0, 1): a float, or an array of `shape`."""
        shp = _result_shape(shape, "random")
        n = 1 if shp is None else _size(shp)
        key, pos = self._take(n, "random")
        out = _k.uniform(key[0], key[1], pos, (1,) if shp is None else shp)
        return float(out[0]) if shp is None else out

    def integers(self, low, high=None, /, *, shape=None, endpoint=False):
        """int64 on [low, high) ([low, high] with endpoint=True); integers(high) is [0, high).

        low and high are ints or int64 arrays, broadcast together and with `shape`."""
        import stardust as sd

        one_bound = high is None
        lo = _int_bound(low, "integers", "high" if one_bound else "low")
        if one_bound:
            lo, hi = 0, lo
        else:
            hi = _int_bound(high, "integers", "high")
        _bool_opt(endpoint, "integers", "endpoint")
        shp = _result_shape(shape, "integers")
        arrays = [v for v in (lo, hi) if isinstance(v, sd.ndarray)]
        scalar_result = shp is None and not arrays
        if shp is None:
            try:
                shp = tuple(sd.broadcast_arrays(*arrays)[0].shape) if arrays else (1,)
            except ValueError:
                raise ValueError("integers(): 'low' and 'high' could not be broadcast together") from None
        try:
            lo_a = sd.broadcast_to(lo if isinstance(lo, sd.ndarray) else sd.asarray([lo]), shp)
            hi_a = sd.broadcast_to(hi if isinstance(hi, sd.ndarray) else sd.asarray([hi]), shp)
        except ValueError:
            raise ValueError(f"integers(): 'low' and 'high' cannot be broadcast to the shape {shp}") from None
        if _size(shp) > 0:
            # on the bounds as given (not broadcast to the result: a pass over 10^6 elements)
            if arrays:
                lo_b, hi_b = (
                    sd.broadcast_arrays(
                        sd.asarray(lo) if not isinstance(lo, sd.ndarray) else lo,
                        sd.asarray(hi) if not isinstance(hi, sd.ndarray) else hi,
                    )
                    if all(isinstance(v, sd.ndarray) for v in (lo, hi))
                    else (lo, hi)
                )
                bad = bool(sd.any(lo_b > hi_b)) if endpoint else bool(sd.any(lo_b >= hi_b))
            else:
                bad = lo > hi if endpoint else lo >= hi
            if bad:
                if one_bound:  # integers(high): the caller gave no 'low'
                    raise ValueError(f"integers(): 'high' must be {'non-negative' if endpoint else 'positive'}")
                rel = "less than or equal to" if endpoint else "less than"
                raise ValueError(f"integers(): 'low' must be {rel} 'high'")
        key, pos = self._take(_size(shp), "integers")
        out = _k.integers(key[0], key[1], pos, lo_a, hi_a, endpoint)
        return int(out[0]) if scalar_result else out

    def choice(self, a, /, *, shape=None, replace=True, p=None):
        """Elements of a 1-D array `a` (or of arange(a) for an int), with or without
        replacement, uniformly or with probabilities `p`."""
        import stardust as sd

        _bool_opt(replace, "choice", "replace")
        shp = _result_shape(shape, "choice")
        if isinstance(a, sd.ndarray):
            if a.ndim != 1:
                raise ValueError(f"choice(): 'a' must be 1-dimensional, not {a.ndim}-dimensional")
            pool, n = a, a.shape[0]
        else:
            if isinstance(a, bool):
                raise TypeError("choice(): 'a' must be an int or a 1-dimensional array, not bool")
            try:
                n = operator.index(a)
            except TypeError:
                raise TypeError(f"choice(): 'a' must be an int or a 1-dimensional array, not {_type_name(a)}") from None
            if n < 0:
                raise ValueError("choice(): 'a' must be non-negative")
            if n >= 2**63:
                raise OverflowError("choice(): 'a' is out of range")
            pool = None
        k = 1 if shp is None else _size(shp)
        # the indices and, from a population array, the elements taken (before integers()
        # or random() below take their part of the stream)
        taken = 0 if pool is None else k * pool.itemsize
        _check_alloc(8 * k + taken, "choice")
        if p is not None:
            if not isinstance(p, sd.ndarray):
                raise TypeError(f"choice(): 'p' must be an sd.ndarray, not {_type_name(p)}")
            if p.ndim != 1 or p.shape[0] != n:
                raise ValueError(f"choice(): 'p' must be 1-dimensional of length {n}")
            if p.dtype == sd.complex128:
                raise TypeError("choice(): 'p' must be a real array, not complex128")
            p = sd.astype(p, sd.float64)
            if bool(sd.any(sd.isnan(p))) or bool(sd.any(p < 0)):
                raise ValueError("choice(): 'p' must be non-negative")
            total = float(sd.sum(p))
            if abs(total - 1.0) > 1.4901161193847656e-08:  # sqrt(eps), numpy's tolerance
                raise ValueError("choice(): the probabilities 'p' must sum to 1")
        if k > 0 and n == 0:
            raise ValueError("choice(): cannot choose from an empty population")
        if not replace and k > n:
            raise ValueError("choice(): cannot take a larger sample than the population when replace=False")
        if replace:
            if p is None:
                idx = self.integers(0, n, shape=(k,)) if k > 0 else sd.zeros((0,), dtype=sd.int64)
            else:
                cdf = sd.cumulative_sum(p)
                cdf = cdf / cdf[n - 1]
                u = self.random((k,))
                idx = sd.minimum(sd.searchsorted(cdf, u, side="right"), n - 1)
        else:
            if p is None:  # the first k of a Fisher-Yates shuffle (O(k) for a small sample)
                key, pos = self._take(k, "choice", _shuffle_bytes(n, k) + taken)
                idx = _k.fisher_yates(key[0], key[1], pos, n, k)
            else:
                if k > int(sd.count_nonzero(p)):
                    raise ValueError("choice(): fewer non-zero probabilities in 'p' than the sample size")
                key, pos = self._take(n, "choice")
                keys = _k.race_keys(key[0], key[1], pos, p)
                idx = sd.argsort(keys)[:k]
        res = idx if pool is None else sd.take(pool, idx)
        if shp is None:
            return res[0]
        return sd.reshape(res, shp)

    def permutation(self, x, /):
        """A random permutation of arange(x) for an int, or of a copy of the array x
        along its first axis."""
        import stardust as sd

        if isinstance(x, sd.ndarray):
            n = x.shape[0]
            row = x.itemsize * _size(x.shape[1:])  # bytes of one row of the copy
        else:
            if isinstance(x, bool):
                raise TypeError("permutation(): 'x' must be an int or an array, not bool")
            try:
                n = operator.index(x)
            except TypeError:
                raise TypeError(f"permutation(): 'x' must be an int or an array, not {_type_name(x)}") from None
            if n < 0:
                raise ValueError("permutation(): 'x' must be non-negative")
            if n >= 2**63:
                raise OverflowError("permutation(): 'x' is out of range")
            row = 0
        # the permutation and, for an array, its permuted copy
        key, pos = self._take(n, "permutation", _shuffle_bytes(n, n) + n * row)
        # Fisher-Yates (sequential, each step's draw from its own element of the stream:
        # the same on any thread count). Measured against sorting random keys (parallel):
        # 3.0 ms for 10^6 against 8.6 on one thread and 3.8 on ten.
        perm = _k.fisher_yates(key[0], key[1], pos, n, n)
        return sd.take(x, perm, axis=0) if isinstance(x, sd.ndarray) else perm


def _restore(state):
    g = Generator(_SeedSequence(0))
    g.state = state
    return g


def default_rng(seed=None, /):
    """A Generator seeded by `seed`: an int >= 0, a sequence of them, or None (OS entropy).
    A Generator is returned as it is."""
    if isinstance(seed, Generator):
        return seed
    return Generator(_SeedSequence(_seed_entropy(seed, "default_rng")))


default_rng = _calls.function(default_rng)  # (API.md §2.1)
