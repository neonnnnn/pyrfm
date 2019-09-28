# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from .dataset_fast import get_dataset
from .dataset_fast cimport RowDataset, ColumnDataset
import numpy as np
cimport numpy as np
from cython.view cimport array
from libc.math cimport fmin
from libcpp.vector cimport vector
from scipy.sparse import csc_matrix


cdef void _canova(double[:, ::1] output,
                  RowDataset X,
                  ColumnDataset P,
                  int degree):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j, t

    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    cdef double[:, ::1] a = array((n_samples_p, degree+1), sizeof(double), 'd')
    for i2 in range(n_samples_p):
        a[i2, 0] = 1
        for t in range(degree):
            a[i2, 1+t] = 0
    for i1 in range(n_samples_x):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                for t in range(degree):
                    a[i2, degree-t] += a[i2, degree-t-1]*x[jj]*p[ii2]

        for i2 in range(n_samples_p):
            output[i1, i2] = a[i2, degree]
            a[i2, 0] = 1
            for t in range(degree):
                a[i2, 1+t] = 0


cdef _canova_sparse(RowDataset X,
                    ColumnDataset P,
                    int degree):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz, n_nz_all
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j, t

    cdef vector[double] data_vec
    cdef vector[int] row_vec
    cdef vector[int] col_vec

    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    cdef double[:, ::1] a = array((n_samples_p, degree+1), sizeof(double), 'd')

    for i2 in range(n_samples_p):
        a[i2, 0] = 1
        for t in range(degree):
            a[i2, 1+t] = 0

    n_nz_all = 0
    for i1 in range(n_samples_x):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                for t in range(degree):
                    a[i2, degree-t] += a[i2, degree-t-1]*x[jj]*p[ii2]

        for i2 in range(n_samples_p):
            if a[i2, degree] != 0:
                n_nz_all += 1
                data_vec.push_back(a[i2, degree])
                row_vec.push_back(i1)
                col_vec.push_back(i2)
            for t in range(degree):
                a[i2, 1+t] = 0
    """
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(n_nz_all,
                                                          dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(n_nz_all,
                                                       dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(n_nz_all,
                                                       dtype=np.int32)

    for i1 in range(n_nz_all):
        data[i1] = data_vec[i1]
        row[i1] = row_vec[i1]
        col[i1] = col_vec[i1]
    """
    return csc_matrix((data_vec, (row_vec, col_vec)),
                      shape=(n_samples_x, n_samples_p))

    #return csc_matrix((data, (row, col)), shape=(n_samples_x, n_samples_p))


cdef void _call_subsets(double[:, ::1] output,
                        RowDataset X,
                        ColumnDataset P):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j
    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    for i1 in range(n_samples_x):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                output[i1, i2] *= (1 + x[jj]*p[ii2])


cdef void _cintersection(double[:, ::1] output,
                         RowDataset X,
                         ColumnDataset P):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j
    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    for i1 in range(n_samples_x):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = p_indices[ii2]
                output[i1, i2] += fmin(x[jj], p[ii2])


cdef void _cchi_square(double[:, ::1] output,
                       RowDataset X,
                       ColumnDataset P):
    # chi square kernel: k(x,y)=\sum_{i=1}^{n}2x_iy_i/(x_i + y_i).
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int x_n_nz, p_n_nz
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j
    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    for i1 in range(n_samples_x):
        X.get_row_ptr(i1, &x_indices, &x, &x_n_nz)
        for jj in range(x_n_nz):
            j = x_indices[jj]
            P.get_column_ptr(j, &p_indices, &p, &p_n_nz)
            for ii2 in range(p_n_nz):
                i2 = x_indices[ii2]
                output[i1, i2] += 2*x[jj]*p[ii2] / (x[jj]+p[ii2])


def _anova(X, P, degree, dense_output=True):
    if dense_output:
        output = np.zeros((X.shape[0], P.shape[0]))

        _canova(output,
                get_dataset(X, order='c'),
                get_dataset(P, order='fortran'),
                degree)
    else:
        output = _canova_sparse(get_dataset(X, order='c'),
                                get_dataset(P, order='fortran'),
                                degree)

    return output


def _all_subsets(X, P):
    output = np.ones((X.shape[0], P.shape[0]))
    _call_subsets(output,
                  get_dataset(X, order='c'),
                  get_dataset(P, order='fortran'))
    return output


def _intersection(X, P):
    output = np.zeros((X.shape[0], P.shape[0]))
    _cintersection(output, get_dataset(X, order='c'),
                   get_dataset(P, order='fortran'))
    return output

def _chi_square(X, P):
    output = np.zeros(X.shape[0], P.shape[0])
    _cchi_square(output, get_dataset(X, order='c'),
                 get_dataset(P, order='fortran'))

    return output