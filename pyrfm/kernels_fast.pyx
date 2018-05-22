# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from lightning.impl.dataset_fast import get_dataset, RowDataset, ColumnDataset
import numpy as np
cimport numpy as np
from cython.view cimport array


cdef void _anova(double[:, ::1] output,
                  RowDataset X,
                  ColumnDataset P,
                  int degree):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t X_samples, P_samples, i1, ii2, i2, jj, j, t

    X_samples = X.get_n_samples()
    P_samples = P.get_n_samples()

    cdef double[:, ::1] a = array((P_samples, degree+1), sizeof(double), 'd')
    a[:, 0] = 1.
    a[:, 1:] = 0.

    for i1 in range(X_samples):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz-1):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                for t in range(degree):
                    a[i2, degree-t] += a[i2, degree-t-1]*x[jj]*p[ii2]

        jj = x_n_nz-1
        j = x_indices[jj]
        P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
        ii2 = 0
        for i2 in range(P_samples):
            output[i1, i2] = 0
            for t in range(1, degree-1):
                a[i2, degree] = 0
            if i2  == p_indices[ii2]:
                output[i1, i2] += a[i2, degree-1]*x[jj]*p[ii2]
                ii2 += 1
            output[i1, i2] += a[i2, degree]

            a[i2, degree-1] = 0.
            a[i2, degree] = 0.


cdef void _all_subsets(double[:, ::1] output,
                       RowDataset X,
                       RowDataset P):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t X_samples, P_samples, i1, ii2, i2, jj, j
    X_samples = X.get_n_samples()
    P_samples = P.get_n_samples()

    for i1 in range(X_samples):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_row_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = x_indices[ii2]
                output[i1, i2] *= (1 + x[jj]*p[ii2])


def anova(X, P, degree):
    output = np.zeros((X.shape[0], P.shape[0]))
    _anova(output,
            get_dataset(X, order='c'),
            get_dataset(P, order='fortran'),
            degree)
    return output


def all_subsets(X, P):
    output = np.ones((X.shape[0], P.shape[0]))
    _all_subsets(output,
                  get_dataset(X, order='c'),
                  get_dataset(P, order='fortran'))
    return output
