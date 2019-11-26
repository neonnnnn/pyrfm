# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from ..dataset_fast cimport RowDataset
from libc.math cimport sqrt


cdef void _cunarize(double[:, ::1] output,
                    RowDataset X,
                    int n_grids):
    cdef double *x
    cdef int *indices
    cdef int n_nz, n_nz_z
    cdef Py_ssize_t n_samples, i, jj, j, k

    n_samples = X.get_n_samples()

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &x, &n_nz)
        for jj in range(n_nz):
            j = indices[jj]
            n_nz_z = int(x[jj]*n_grids)
            for k in range(n_nz_z):
                output[i, j*n_grids+k] = 1. / sqrt(n_grids)

            if x[jj] < 1:
                output[i, j*n_grids+n_nz_z] = n_grids*x[jj] - int(n_grids*x[jj])
                output[i, j*n_grids+n_nz_z] /= sqrt(n_grids)


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
            ind = int((n_grids-1)*x[jj])
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


def make_sparse_mb(data, row, col, X, n_grids):
    _make_sparse_mb(data, row, col, X, n_grids)
