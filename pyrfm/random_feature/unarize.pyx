# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from ..dataset_fast cimport RowDataset
from libc.math cimport sqrt, floor, ceil
from cython.view cimport array
from scipy.sparse import csr_matrix
import numpy as np
cimport numpy as np


cdef void _cunarize_sparse(double[:] values,
                           int[:] indices_out,
                           int[:] indptr,
                           RowDataset X,
                           int n_grids,
                           int n_nz):
    cdef double *x
    cdef int *indices
    cdef int n_nz_z
    cdef Py_ssize_t n_samples, i, jj, j, k, n

    n_samples = X.get_n_samples()
    cdef double sqrt_n_grids = sqrt(n_grids)    
    n = 0
    indptr[0] = 0
    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &x, &n_nz)
        indptr[i+1] = indptr[i]
        for jj in range(n_nz):
            j = indices[jj]
            n_nz_z = int(floor(x[jj]*n_grids))
            for k in range(n_nz_z):
                values[n] = 1. / sqrt_n_grids
                indices_out[n] = n_grids*j + k
                n += 1
                indptr[i+1] += 1
            if int(ceil(n_grids*x[jj])) > n_nz_z:
                values[n] = n_grids*x[jj] - n_nz_z
                values[n] /= sqrt_n_grids
                indices_out[n] = n_grids*j + n_nz_z
                n += 1
                indptr[i+1] += 1


cdef void _cunarize(double[:, ::1] output,
                    RowDataset X,
                    int n_grids):
    cdef double *x
    cdef int *indices
    cdef int n_nz, n_nz_z
    cdef Py_ssize_t n_samples, i, jj, j, k
    cdef double sqrt_n_grids = sqrt(n_grids)    
    n_samples = X.get_n_samples()

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &x, &n_nz)
        for jj in range(n_nz):
            j = indices[jj]
            n_nz_z = int(x[jj]*n_grids)
            for k in range(n_nz_z):
                output[i, j*n_grids+k] = 1. / sqrt_n_grids

            if x[jj] < 1:
                output[i, j*n_grids+n_nz_z] = n_grids*x[jj] - n_nz_z
                output[i, j*n_grids+n_nz_z] /= sqrt_n_grids


cdef void _make_sparse_mb(double[:] data,
                          int[:] row,
                          int[:] col,
                          RowDataset X,
                          int n_grids):
    cdef double *x
    cdef int *indices
    cdef int n_nz
    cdef Py_ssize_t n_samples, i, jj, j, offset, ind
    cdef double val

    n_samples = X.get_n_samples()
    n_features = X.get_n_features()
    offset = 0
    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &x, &n_nz)
        jj = 0
        for jj in range(n_nz):
            row[offset] = i
            row[offset+1] = i
            j = indices[jj]
            ind = int(floor((n_grids-1)*x[jj]))
            val = n_grids*x[jj] - ind
            col[offset] = j*n_grids+ind
            col[offset+1] = j*n_grids+ind+1
            if ind == 0:
                data[offset] = 0
            else:
                data[offset] = 1-val

            data[offset+1] = val
            offset += 2


def unarize(output, X, n_grids):
    _cunarize(output, X, n_grids)


def unarize_sparse(X, n_grids, n_nz, n_components):
    cdef int n_samples = X.get_n_samples()
    values = np.zeros((n_nz,), dtype=np.double)
    indices = np.zeros((n_nz,), dtype=np.int32)
    indptr = np.zeros((n_samples+1, ), dtype=np.int32) 
    _cunarize_sparse(values, indices, indptr, X, n_grids, n_nz)
    shape = (n_samples, n_components)
    return csr_matrix((values, indices, indptr), shape=shape)


def make_sparse_mb(data, row, col, X, n_grids):
    _make_sparse_mb(data, row, col, X, n_grids)
