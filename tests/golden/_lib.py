"""The context the golden cases receive (tools/golden.py): the library and input helpers.

L.sd the library, L.np numpy (inputs only), L.rng(seed) a numpy Generator (inputs only),
L.arr(a) sd.asarray of a numpy array (zero copy where the buffer allows, so the
numpy strides are what the library sees). L.A, L.LA and L.F are sd, sd.linalg
and sd.frame (the names the merge-era case files use).
"""

import numpy as np

import stardust as sd


class Lib:
    def __init__(self):
        self.sd = self.A = sd
        self.LA = sd.linalg
        self.F = sd.frame
        self.np = np
        self._threads = sd

    @staticmethod
    def rng(seed):
        return np.random.Generator(np.random.PCG64(seed))

    def arr(self, a):
        return sd.asarray(np.asarray(a))
