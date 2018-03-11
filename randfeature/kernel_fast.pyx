from lightning.impl.dataset_fast import get_dataset
from lightning.impl.dataset_fast cimport RowDataset
import numpy as np
cimport numpy as np
from cython.view cimport array


cdef void canova(double[:, ::1] output,
                 RowDataset X,
                 RowDataset P,
                 int degree):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int X_n_nz, P_n_nz
    cdef Py_ssize_t X_samples, P_samples, i, j, kk_p, kk_x, k, t
    cdef double[:] a = array((degree+1, ), sizeof(double), 'd')

    X_samples = X.get_n_samples()
    P_samples = P.get_n_samples()
    a[0] = 1

    for i in range(X_samples):
        X.get_row_ptr(i, &x_indices, &x, &X_n_nz)
        for j in range(P_samples):
            P.get_row_ptr(j, &p_indices, &p, &P_n_nz)
            kk_p = 0
            kk_x = 0
            for t in range(degree):
                a[degree-t] = 0

            while (kk_x < X_n_nz) and (kk_p < P_n_nz):
                if p_indices[kk_p] > x_indices[kk_x]:
                    kk_x += 1
                elif p_indices[kk_p] < x_indices[kk_x]:
                    kk_p += 1
                else:
                    k = p_indices[kk_p]
                    for t in range(degree):
                        a[degree-t] += a[degree-t-1]*x[k]*p[k]
                    kk_x += 1
                    kk_p += 1
            output[i, j] = a[degree]


cdef void call_subset(double[:, ::1] output,
                      RowDataset X,
                      RowDataset P):
    cdef double *x
    cdef double *p
    cdef int *x_indices
    cdef int *p_indices
    cdef int X_n_nz, P_n_nz
    cdef Py_ssize_t X_samples, P_samples, i, j, kk_p, kk_x, k
    X_samples = X.get_n_samples()
    P_samples = P.get_n_samples()

    for i in range(X_samples):
        X.get_row_ptr(i, &x_indices, &x, &X_n_nz)
        for j in range(P_samples):
            P.get_row_ptr(j, &p_indices, &p, &P_n_nz)
            kk_p = 0
            kk_x = 0
            output[i,j] = 1
            while (kk_x < X_n_nz) and (kk_p < P_n_nz):
                if p_indices[kk_p] > x_indices[kk_x]:
                    kk_x += 1
                elif p_indices[kk_p] < x_indices[kk_x]:
                    kk_p += 1
                else:
                    k = p_indices[kk_p]
                    output[i,j] *= (1+x[k]*p[k])
                    kk_x += 1
                    kk_p += 1


def anova(X, P, degree):
    output = np.zeros((X.shape[0], P.shape[0]))
    canova(output,
           get_dataset(X, order='c'),
           get_dataset(P, order='c'),
           degree)
    return output


def all_subset(X, P):
    output = np.ones((X.shape[0], P.shape[0]))
    call_subset(output,
                get_dataset(X, order='c'),
                get_dataset(P, order='c'))
    return output
