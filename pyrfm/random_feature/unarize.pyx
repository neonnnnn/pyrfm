# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from lightning.impl.dataset_fast cimport RowDataset
from libc.math cimport sqrt


cdef void _cunarize(double[:, ::1] output,
                    RowDataset X,
                    int n_grids):
    cdef double *x
    cdef int *indices
    cdef int n_nz
    cdef Py_ssize_t n_samples, i, jj, j, k

    n_samples = X.get_n_samples()

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &x, &n_nz)
        for jj in range(n_nz):
            j = indices[jj]
            for k in range(int(x[jj]*n_grids)):
                output[i, j*n_grids+k] = 1. / sqrt(n_grids)

            if x[jj] < 1:
                output[i, j*n_grids+k] = (n_grids*x[jj] - int(n_grids*x[jj]))
                output[i, j*n_grids+k] /= sqrt(n_grids)


def unarize(output, X, n_grids):
    _cunarize(output, X, n_grids)