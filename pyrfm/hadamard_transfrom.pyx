# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from cython.view cimport array
import math


cdef void  _fht(double[:] out):
    cdef Py_ssize_t out_d, jj, j1, i2, step, t
    d = x.shape[0]
    out_d = out.shape[0]
    step = out_d/2
    cdef int degree = int(math.log2(out_d))
    cdef double temp1, temp2

    for t in range(degree, 1, -1):
        j1 = 0
        for jj in range(out_d/2):
            j2 = j1 + step
            temp1 = out[j1] + out[j2]
            temp2 = out[j1] - out[j2]
            out[j1] = temp1
            out[j2] = temp2
            if (jj+1)%step == 0:
                j1 += degree
            j1 += 1
        step /= 2


cdef void _fht_matrix(double[:, ::1] out):
    cdef Py_ssize_t d, n, out_d, i, jj, j1, j2, step, t
    cdef double temp1, temp2
    n, out_d = out.shape
    cdef int degree = int(math.log2(out_d))

    for i in range(n):
        step = out_d / 2
        for t in range(degree, 1, -1):
            j1 = 0
            for jj in range(out_n/2):
                j2 = j1 + step
                temp1 = out[i. j1] + out[i, j2]
                temp2 = out[i, j1] - out[i, j2]
                out[i, j1] = temp1
                out[i, j2] = temp2
                if (jj+1)%step == 0:
                    j1 += degree
                j1 += 1
            step /= 2


def fht(x):
    if x.ndim == 1:
        d = x.shape[0]
        out = np.zeros((2**math.ceil(math.log2(d)), ))
        out[:d] += x
        _fht(d, out)
        return out[:d]
    elif x.ndim == 2:
        n, d = x.shape
        out = np.zeros((n, 2**math.ceil(math.log2(d))))
        out[:, :d] += x
        _fht_matrix(x, out)
        return out[:, :d]

