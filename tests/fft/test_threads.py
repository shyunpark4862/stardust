"""Results of fft and ifft do not depend on the thread count or on Accelerate's thread mode
(docs/design/fft.md §1.3, §3.7, §7.3): at 1, 3 and 10 threads every case gives the same bytes on inputs
large enough to split into many tasks (batches along the last and the first axis, mixed-radix and
Bluestein lengths, one long line), and a child process with VECLIB_MAXIMUM_THREADS=1 gives the same
SHA-256 as one with Accelerate's default."""

import os
import subprocess
import sys

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal

pytestmark = pytest.mark.usefixtures("restore_threads")

F = sd.fft


def _results():
    g = np.random.default_rng(21)
    out = []
    for shape in ((300, 256), (64, 1009), (40, 1000), (2, 3, 4096)):
        x = g.uniform(-1, 1, shape) + 1j * g.uniform(-1, 1, shape)
        sx = sd.asarray(x)
        out.append(F.fft(sx))
        out.append(F.ifft(sx, axis=0, norm="ortho"))
        out.append(F.fft(sx, axis=-1, n=shape[-1] + 5))
    x = g.uniform(-1, 1, 1 << 17) + 1j * g.uniform(-1, 1, 1 << 17)
    out.append(F.fft(sd.asarray(x)))
    out.append(F.ifft(sd.asarray(x[:65537]), norm="forward"))
    return out


def test_one_three_ten_threads():
    got = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        got[t] = _results()
    for t in (3, 10):
        for k, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, k)


_CHILD = r"""
import hashlib, numpy as np, stardust as sd
g = np.random.default_rng(22)
h = hashlib.sha256()
for shape in ((200, 128), (30, 1009), (1, 1 << 16)):
    x = g.uniform(-1, 1, shape) + 1j * g.uniform(-1, 1, shape)
    h.update(np.asarray(sd.fft.fft(sd.asarray(x))).tobytes())
    h.update(np.asarray(sd.fft.ifft(sd.asarray(x), axis=0)).tobytes())
print(h.hexdigest())
"""


def test_accelerate_thread_mode_does_not_matter():
    digests = []
    for mode in ("1", None):
        env = dict(os.environ)
        env.pop("VECLIB_MAXIMUM_THREADS", None)
        if mode is not None:
            env["VECLIB_MAXIMUM_THREADS"] = mode
        r = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        digests.append(r.stdout.strip())
    assert digests[0] == digests[1] and len(digests[0]) == 64


def test_a_row_is_the_row_alone_at_every_thread_count():
    g = np.random.default_rng(23)
    x = g.uniform(-1, 1, (48, 1009)) + 1j * g.uniform(-1, 1, (48, 1009))
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        for axis in (-1, 0):
            got = np.asarray(F.fft(sd.asarray(x), axis=axis))
            lines = x if axis == -1 else x.T
            out = got if axis == -1 else got.T
            for k in range(0, lines.shape[0], 7):
                assert bits_equal(out[k], F.fft(sd.asarray(np.ascontiguousarray(lines[k])))), (t, axis, k)
