#! /usr/bin/env python
from functools import lru_cache
import numpy as np


def doubles(kmax, nmax):
    k = np.arange(1, kmax + 1, dtype=np.float64)
    n = np.arange(1, nmax + 1, dtype=np.float64)

    logk = np.log(k)[:, None]
    lognp1 = np.log(n + 1)[None, :]
    val = -(logk + 2.0 * k[:, None] * lognp1)
    return np.exp(val).sum(dtype=np.float64)


def doubles(kmax, nmax):
    n = np.arange(1, nmax + 1, dtype=np.float64)
    r = 1.0 / (n + 1) ** 2
    k = np.arange(1, kmax + 1, dtype=np.float64)[:, None]
    sn = (np.power(r[None, :], k) / k).sum(axis=0)
    return sn.sum()


def doubles(kmax, nmax):
    k = np.arange(1, kmax + 1, dtype=np.float64)[:, None]
    n = np.arange(1, nmax + 1, dtype=np.float64)[None, :]
    r = 1.0 / (n + 1.0) ** 2
    terms = (1.0 / k) * r**k
    return terms.sum(dtype=np.float64)


print(doubles(1, 3))
print(doubles(1, 10))
print(doubles_vec(20, 10000))
print(doubles(10, 1000))
