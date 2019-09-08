# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc cimport stdlib
from libc.math cimport cos, sin, sqrt
from scipy.fftpack._fftpack import drfft
import numpy as np
cimport numpy as np
from sklearn.kernel_approximation import RBFSampler
from lightning.impl.dataset_fast import get_dataset
from lightning.impl.dataset_fast cimport RowDataset
from cython.view cimport array
from . import (RandomFourier, RandomKernel, RandomMaclaurin, TensorSketch,)


RANDOMFEATURES = {
    RandomFourier: CRandomFourier,
    RandomKernel: CRandomKernel,
    RandomMaclaurin: CRandomMaclaurin,
    TensorSketch: CTensorSketch,
    RBFSampler: CRBFSampler
}


cdef inline double dot(double* x,
                       int* indices,
                       int n_nz,
                       double* y):
    cdef Py_ssize_t j, jj
    cdef double result = 0
    for jj in range(n_nz):
        j = indices[jj]
        result += y[j]*x[jj]
    return result


cdef class BaseCRandomFeature(object):
    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):
        raise NotImplementedError("This is an abstract method.")


cdef class CRBFSampler(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_

    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        # z = (cos, cos, ..., cos)      
        for i in range(self.n_components):
            z[i] = self.random_offset[i]
        for jj in range(n_nz):
            j = indices[jj]
            for i in range(self.n_components):
                z[i] += data[jj] * self.random_weights[i, j]
        for i in range(self.n_components):
            z[i] = cos(z[i])*sqrt(2./self.n_components)


cdef class CRandomFourier(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]
        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_
        self.use_offset = transformer.use_offset

    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):

        cdef Py_ssize_t i, jj, j
        cdef Py_ssize_t index_offset = int(self.n_components/2)
        # z = (cos, cos, ..., cos)
        if self.use_offset:
            for i in range(self.n_components):
                z[i] = dot(data, indices, n_nz, &self.random_weights[i, 0])
                z[i] += self.random_offset[i]
                z[i] = cos(z[i])*sqrt(2./self.n_components)
        # z = (cos, ..., cos, sin, ..., sin)
        else:
            for i in range(index_offset):
                z[i] = dot(data, indices, n_nz, &self.random_weights[i, 0])
                z[i+index_offset] = sin(z[i])*sqrt(2./self.n_components)
                z[i] = cos(z[i])*sqrt(2./self.n_components)


cdef class CRandomMaclaurin(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]
        self.random_weights = transformer.random_weights_
        self.orders = transformer.orders_
        self.p_choice = transformer.p_choice
        self.coefs = transformer.coefs

    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, k, offset, deg, j, jj
        cdef double tmp
        offset = 0
        for i in range(self.n_components):
            z[i] = 1.
            deg = self.orders[i]
            for k in range(deg):
                z[i] *= dot(data, indices, n_nz,
                            &self.random_weights[offset+k, 0])
            z[i] *= sqrt(self.coefs[self.orders[i]]/self.n_components)
            z[i] /= sqrt(self.p_choice[self.orders[i]])
            offset += self.orders[i]


cdef class CTensorSketch(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = int(len(transformer.hash_indices_)/transformer.degree)
        self.degree = transformer.degree
        self.hash_indices = transformer.hash_indices_
        self.hash_signs = transformer.hash_signs_
        self.z_cache = array((self.n_components, ), sizeof(double), format='d')

    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):

        cdef Py_ssize_t jj, j, offset
        for j in range(self.n_components):
            z[j] = 0
            self.z_cache[j] = 0

        for jj in range(n_nz):
            j = indices[jj]
            z[self.hash_indices[j]] += data[jj]*self.hash_signs[j]
        drfft(z, direction=1, overwrite_x=True)
        for offset in range(self.n_features, self.n_features*self.degree, self.n_features):
            for j in range(self.n_components):
                self.z_cache[j] = 0

            for jj in range(n_nz):
                j = indices[jj]
                self.z_cache[self.hash_indices[j+offset]] \
                    += data[jj]*self.hash_signs[j+offset]

            drfft(self.z_cache, direction=1, overwrite_x=True)
            for j in range(self.n_components):
                z[j] *= self.z_cache[j]
        drfft(z, direction=-1, overwrite_x=True)



cdef class CRandomKernel(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]
        self.degree = transformer.degree
        # Now, not support for sparse rademacher
        self.random_weights = transformer.random_weights_
        if transformer.kernel in ["anova", "anova_cyhon"]:
            self.kernel = 0
        elif transformer.kernel == "all_subsets":
            self.kernel = 1
        else:
            raise ValueError('kernel = {} is not defined.'
                             .format(transformer.kernel))
        self.anova = array((self.degree+1, ), sizeof(double), format='d')
        cdef Py_ssize_t i
        self.anova[0] = 1
        for i in range(self.degree):
            self.anova[i+1] = 0

    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz):

        if self.kernel == 0:
            anova(z, data, indices, n_nz, self.random_weights, self.degree,
                  self.anova)
        elif self.kernel == 1:
            all_subsets(z, data, indices, n_nz, self.random_weights)

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


def transform_all_fast(X, transformer):
    cdef double[:, ::1] Z = np.zeros((X.shape[0], transformer.n_components),
                                     dtype=np.float64)
    cdef RowDataset dataset = get_dataset(X, order='c')
    cdef Py_ssize_t i
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    cdef double* data
    cdef int* indices
    cdef int n_nz
    cdef BaseCRandomFeature transformer_fast \
        = get_fast_random_feature(transformer)
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(Z[i], data, indices, n_nz)
    return Z


def get_fast_random_feature(transformer):
    if transformer.__class__ in RANDOMFEATURES:
        return RANDOMFEATURES[transformer.__class__](transformer)
    else:
        return None