import numpy as np
import numba as nb
from numba import uint64, int32
from numba.experimental import jitclass

spec = [
    ('n', int32),              # number of variables
    ('n_words', int32),       # number of uint64 words used to store the affine form
    ('data', uint64[:])       # word array
]

@jitclass(spec)
class Bitset:
    def __init__(self, n):
        self.n = n
        self.n_words = (n + 63) // 64
        self.data = np.zeros(self.n_words, dtype=np.uint64)

    def set(self, i):
        word = i >> 6        # i // 64
        bit  = i & 63        # i % 64
        self.data[word] |= (np.uint64(1) << np.uint64(bit))

    def unset(self, i):
        word = i >> 6
        bit  = i & 63
        self.data[word] &= ~(np.uint64(1) << np.uint64(bit))

    def get(self, i):
        word = i >> 6
        bit  = i & 63
        return (self.data[word] >> np.uint64(bit)) & np.uint64(1)

    def xor(self, other):
        for k in range(self.n_words):
            self.data[k] ^= other.data[k]

    def copy_from(self, other):
        for k in range(self.n_words):
            self.data[k] = other.data[k]

    def popcount(self):
        s = 0
        for k in range(self.n_words):
            s += self.data[k].bit_count()
        return s
