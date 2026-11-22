"""Results of the one-dimensional transforms (fft and ifft, and since milestone FT2 rfft, irfft, hfft,
ihfft and fft of real input) and of the n-dimensional ones (milestone FT3: every axis pass splits its
lines into tasks) do not depend on the thread count or on Accelerate's thread mode
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
    for shape in ((300, 256), (64, 1009), (40, 1000), (2, 3, 4096), (500, 26)):
        r = g.uniform(-1, 1, shape)
        sr = sd.asarray(r)
        out.append(F.rfft(sr))
        out.append(F.fft(sr, axis=0))
        out.append(F.ihfft(sr, axis=0, norm="ortho"))
        h = sd.asarray(r + 1j * g.uniform(-1, 1, shape))
        out.append(F.irfft(h))
        out.append(F.hfft(h, axis=0, n=2 * shape[0] + 1))
    r = g.uniform(-1, 1, 1 << 17)
    out.append(F.rfft(sd.asarray(r)))
    out.append(F.irfft(sd.asarray(r[:65537] + 0.5j)))
    # the n-dimensional transforms (FT3): passes along the last, middle and first axes, in place and not
    shape = (96, 128, 20)
    x = sd.asarray(g.uniform(-1, 1, shape) + 1j * g.uniform(-1, 1, shape))
    r = sd.asarray(g.uniform(-1, 1, shape))
    out.append(F.fftn(x))
    out.append(F.ifftn(x, s=(100, 20), axes=(0, 2), norm="ortho"))
    out.append(F.fftn(r))
    out.append(F.rfftn(r))
    out.append(F.ihfftn(r, axes=(2, 0)))
    out.append(F.irfftn(x))
    out.append(F.hfftn(x, s=(90, 31), axes=(1, 0)))
    out.append(F.fft2(sd.asarray(g.uniform(-1, 1, (34, 1009)) + 0.25j)))
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
    h.update(np.asarray(sd.fft.rfft(sd.asarray(x.real))).tobytes())
    h.update(np.asarray(sd.fft.irfft(sd.asarray(x))).tobytes())
    h.update(np.asarray(sd.fft.fft2(sd.asarray(x))).tobytes())
    h.update(np.asarray(sd.fft.irfftn(sd.asarray(x))).tobytes())
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


def test_a_real_row_is_the_row_alone_at_every_thread_count():
    g = np.random.default_rng(24)
    x = g.uniform(-1, 1, (48, 2018))
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        for axis in (-1, 0):
            got = np.asarray(F.rfft(sd.asarray(x), axis=axis))
            lines = x if axis == -1 else x.T
            out = got if axis == -1 else got.T
            for k in range(0, lines.shape[0], 7):
                assert bits_equal(out[k], F.rfft(sd.asarray(np.ascontiguousarray(lines[k])))), (t, axis, k)


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
