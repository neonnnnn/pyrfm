# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from .dataset_fast import get_dataset
from .dataset_fast cimport RowDataset, ColumnDataset
import numpy as np
cimport numpy as np
from cython.view cimport array
from libc.math cimport fmin
from libcpp.vector cimport vector
from scipy.sparse import csr_matrix
import warnings


cdef inline double _sparse_dot(double* x,
                               int* indices_x,
                               int n_nz_x,
                               double* y,
                               int* indices_y,
                               int n_nz_y):
    cdef Py_ssize_t ii, i, jj, j
    cdef double dot = 0
    jj = 0
    for ii in range(n_nz_x):
        i = indices_x[ii]
        while jj < n_nz_y:
            j = indices_y[jj]
            if j >= i:
                break
            jj += 1

        if j == i:
            dot += x[ii] * y[jj]
            jj += 1

        if jj == n_nz_y:
            break
    return dot


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
    cdef int x_n_nz, p_n_nz
    cdef unsigned long int n_nz_all_all
    cdef Py_ssize_t n_samples_x, n_samples_p, i1, ii2, i2, jj, j, t
    n_samples_x = X.get_n_samples()
    n_samples_p = P.get_n_samples()

    cdef vector[double] data_vec
    cdef vector[int] col_vec
    cdef np.ndarray[np.float64_t, ndim=1] data
    cdef np.ndarray[np.int32_t, ndim=1] indptr
    cdef np.ndarray[np.int32_t, ndim=1] indices
    indptr = np.zeros(n_samples_x+1, dtype=np.int32)

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
                col_vec.push_back(i2)
                indptr[i1+1] += 1
            for t in range(degree):
                a[i2, 1+t] = 0
        indptr[i1+1] += indptr[i1]
    data = np.empty(n_nz_all, dtype=np.float64)
    indices = np.empty(n_nz_all, dtype=np.int32)

    for i1 in range(n_nz_all):
        data[i1] = data_vec[i1]
        indices[i1] = col_vec[i1]

    return csr_matrix((data, indices, indptr),
                      shape=(n_samples_x, n_samples_p))


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


cdef double _score(RowDataset X,
                   double[:, ::1] K,
                   int loss,
                   bint mean,
                   double* err_sup):
    cdef double* data1,
    cdef int* indices1
    cdef int n_nz1
    cdef double* data2
    cdef int* indices2
    cdef int n_nz2
    cdef Py_ssize_t i, j
    cdef int n_samples = X.get_n_samples()
    cdef double result = 0
    cdef double dot, err
    err_sup[0] = 0
    for i in range(n_samples):
        X.get_row_ptr(i, &indices1, &data1, &n_nz1)
        for j in range(i, n_samples):
            X.get_row_ptr(j, &indices2, &data2, &n_nz2)
            dot = _sparse_dot(data1, indices1, n_nz1, data2, indices2, n_nz2)
            if loss == 1:
                err = abs(K[i, j] - dot)
            elif loss == 2:
                err = (K[i, j] - dot) ** 2
            result += err
            if err_sup[0] < err:
                err_sup[0] = err

    if mean:
        result = result / (n_samples * (n_samples+1) / 2)
    return result


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


def _all_subsets(X, P, dense_output=True):
    output = np.ones((X.shape[0], P.shape[0]))
    if not dense_output:
        warnings.warn("The all-subsets kernel outputs np.ndarray"
                      "since its output matrix has no zero elements.")
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


def score(X, K, loss='l2', mean=True, return_max=False):
    """ Compute the approximation error of X

    .. math::

        \sum_{i=1}^{n}\sum_{j=i}^{n} loss(dot(X[i], X[j]), K[i,j]) / (n(n+1)/2)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Random feature matrix.

    K : np.ndarray, shape (n_samples, n_samples)
        Gram matrix.

    loss : str
        Which loss function. "l1" or "l2" can be used.

    mean : bool
        Whether compute mean or not. If false, return sum of the errs.

    return_max : bool
        Whether return the max error or not.

    Returns
    -------
    error : double
        Approximation error.

    sup : double, optional (only when return_max = True)
        Maximum approximation error.

    """

    if loss == 'l1':
        _loss = 1
    elif loss == 'l2':
        _loss = 2
    else:
        raise ValueError('loss {} is not supported, only "l1" or "l2" can be '
                         'used')
    cdef double sup, error
    error = _score(get_dataset(X, 'c'), K, _loss, mean, &sup)
    if return_max:
        return error, sup
    else:
        return error
