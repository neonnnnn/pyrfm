# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport cos, sin, sqrt
from lightning.impl.dataset_fast cimport ColumnDataset, RowDataset
from scipy.fftpack._fftpack import drfft
from cython.parallel import prange


cdef void random_fourier(double[:] z,
                         double* data,
                         int* indices,
                         int n_nz,
                         double[:, ::1] random_weights,
                         double[:] offset,
                         ):
    cdef Py_ssize_t n_components = z.shape[0]
    cdef Py_ssize_t n_features = random_weights.shape[1]
    cdef Py_ssize_t i, jj, j
    cdef Py_ssize_t index_offset = n_components/2
    # z = (cos, cos, ..., cos)
    if n_components == random_weights.shape[0]:
        for i in range(n_components):
            z[i] = 0
            for jj in range(n_nz):
                j = indices[jj]
                z[i] += data[jj]*random_weights[i, j]
            z[i] += offset[i]
            z[i] = cos(z[i])*sqrt(2/n_components)
    # z = (cos, ..., cos, sin, ..., sin)
    else:
        for i in range(index_offset):
            z[i] = 0
            z[i+index_offset] = 0
            for jj in range(n_nz):
                j = indices[jj]
                z[i] += data[jj]*random_weights[i, j]
            z[i+index_offset] = sin(z[i])/sqrt(n_components)
            z[i] = cos(z[i])/sqrt(n_components)


cdef void random_maclaurin(double[:] z,
                           double* data,
                           int* indices,
                           int n_nz,
                           double[:, ::1] random_weights,
                           int[:] orders,
                           double[:] p_choice,
                           double[:] coefs,
                           ):
    cdef Py_ssize_t n_components = len(z)
    cdef Py_ssize_t deg, i, jj, j
    cdef Py_ssize_t n_features, offset
    cdef double tmp
    n_features = random_weights.shape[1]
    offset = 0
    for i in range(n_components):
        z[i] = 0
        for deg in range(orders[i]):
            tmp = 0
            for jj in range(n_nz):
                j = indices[jj]
                tmp += data[jj]*random_weights[offset+deg, j]
            if deg == 0:
                z[i] = tmp
            else:
                z[i] *= tmp

            tmp = 0
        z[i] *= sqrt(coefs[orders[i]]/n_components)
        z[i] /= sqrt(p_choice[orders[i]])
        offset += orders[i]


cdef void tensor_sketch(double[:] z,
                        double[:] z_cache,
                        double* data,
                        int* indices,
                        int n_nz,
                        int degree,
                        int[:] hash_indices,
                        int[:] hash_signs):
    cdef Py_ssize_t n_components
    cdef Py_ssize_t jj, j
    cdef Py_ssize_t n_features, offset
    n_components = len(z)
    n_features = int(len(hash_indices) / degree)

    for j in range(n_components):
        z[j] = 0
        z_cache[j] = 0

    for jj in range(n_nz):
        j = indices[jj]
        z[hash_indices[j]] += data[jj]*hash_signs[j]

    drfft(z, direction=1, overwrite_x=True)

    for offset in range(n_features, n_features*degree, n_features):
        for j in range(n_components):
            z_cache[j] = 0

        for jj in range(n_nz):
            j = indices[jj]
            z_cache[hash_indices[j+offset]] += data[jj]*hash_signs[j+offset]

        drfft(z_cache, direction=1, overwrite_x=True)
        for j in range(n_components):
            z[j] *= z_cache[j]
    drfft(z, direction=-1, overwrite_x=True)


cdef inline void random_kernel(double[:] z,
                               double* data,
                               int* indices,
                               int n_nz,
                               double[:, ::1] random_weights,
                               int kernel,
                               int degree,
                               double[:] a):
    if kernel == 0:
        anova(z, data, indices, n_nz, random_weights, degree, a)
    elif kernel == 1:
        all_subsets(z, data, indices, n_nz, random_weights)
    else:
        raise ValueError('kernel = {} is not defined.'.format(kernel))


cdef inline void anova(double[:] z,
                       double* data,
                       int* indices,
                       int n_nz,
                       double[:, ::1] random_weights,
                       int degree,
                       double[:] a):
    cdef Py_ssize_t j, jj, deg, i
    cdef Py_ssize_t n_components = random_weights.shape[0]

    for i in range(n_components):
        for j in range(degree+1):
            a[j] = 0
        a[0] = 1

        for jj in range(n_nz):
            j = indices[jj]
            for deg in range(degree):
                a[degree-deg] += random_weights[i, j]*data[jj]*a[degree-deg-1]

        z[i] = a[degree] / sqrt(n_components)


cdef inline void all_subsets(double[:] z,
                             double* data,
                             int* indices,
                             int n_nz,
                             double[:, ::1] random_weights):
    cdef Py_ssize_t j, jj, i
    cdef Py_ssize_t n_components = random_weights.shape[0]

    for i in range(n_components):
        z[i] = 1
        for jj in range(n_nz):
            j = indices[jj]
            z[i] *= (1+data[jj]*random_weights[i, j])
        z[i] /= sqrt(n_components)
