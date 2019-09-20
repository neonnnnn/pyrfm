# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport cos, sin, sqrt
from scipy.fftpack._fftpack import drfft, zrfft, zfft
import numpy as np
cimport numpy as np
from sklearn.kernel_approximation import RBFSampler
from lightning.impl.dataset_fast import get_dataset
from lightning.impl.dataset_fast cimport RowDataset
from cython.view cimport array
from . import (RandomFourier, RandomKernel, RandomMaclaurin, TensorSketch,
               FastFood, SubsampledRandomHadamard, CompactRandomFeature,
               RandomProjection, OrthogonalRandomFeature,
               StructuredOrthogonalRandomFeature,
               SignedCirculantRandomMatrix)
from .utils_fast cimport _fwht1d


RANDOMFEATURES = {
    RandomFourier: CRandomFourier,
    RandomKernel: CRandomKernel,
    RandomMaclaurin: CRandomMaclaurin,
    TensorSketch: CTensorSketch,
    RBFSampler: CRBFSampler,
    FastFood: CFastFood,
    SubsampledRandomHadamard: CSubsampledRandomHadamard,
    RandomProjection: CRandomProjection,
    CompactRandomFeature: CCompactRandomFeature,
    OrthogonalRandomFeature: CRandomFourier,
    StructuredOrthogonalRandomFeature: CStructuredOrthogonalRandomFeature,
    SignedCirculantRandomMatrix: CSignedCirculantRandomMatrix
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
                        double* z,
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
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj, n_samples
        # z = (cos, cos, ..., cos)
        for i in range(self.n_components):
            z[i] = self.random_offset[i]
        for jj in range(n_nz):
            j = indices[jj]
            for i in range(self.n_components):
                z[i] += data[jj] * self.random_weights[j, i]

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
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
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
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, k, offset, deg
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
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):

        cdef Py_ssize_t i, jj, j, offset
        for i in range(self.n_components):
            self.z_cache[i] = 0
        for jj in range(n_nz):
            j = indices[jj]
            self.z_cache[self.hash_indices[j]] += data[jj]*self.hash_signs[j]
        drfft(self.z_cache, direction=1, overwrite_x=True)
        for i in range(self.n_components):
            z[i] = self.z_cache[i]

        for offset in range(self.n_features, self.n_features*self.degree, self.n_features):
            for i in range(self.n_components):
                self.z_cache[i] = 0

            for jj in range(n_nz):
                j = indices[jj]
                i = self.hash_indices[j+offset]
                self.z_cache[i] += data[jj]*self.hash_signs[j+offset]

            drfft(self.z_cache, direction=1, overwrite_x=True)
            for i in range(self.n_components):
                z[i] *= self.z_cache[i]

        for i in range(self.n_components):
            self.z_cache[i] = z[i]
        drfft(self.z_cache, direction=-1, overwrite_x=True)
        for i in range(self.n_components):
            z[i] = self.z_cache[i]


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
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):

        if self.kernel == 0:
            anova(z, data, indices, n_nz, self.random_weights, self.degree,
                  self.anova)
        elif self.kernel == 1:
            all_subsets(z, data, indices, n_nz, self.random_weights)


cdef class CFastFood(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_sign_.shape[1]
        self.random_weights = transformer.random_weights_
        self.random_sign = transformer.random_sign_
        self.fy_vec = transformer._fy_vector_
        self.random_scaling = transformer.random_scaling_
        self.random_offset = transformer.random_offset_
        self.cache = array((2**self.degree_hadamard, ), sizeof(double),
                           format='d')
        self.gamma = transformer.gamma
        self.random_fourier = transformer.random_fourier
        self.degree_hadamard = (self.n_features-1).bit_length()

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj, k, n_features_padded, t, n_stacks
        cdef double tmp, factor
        n_features_padded = 2**self.degree_hadamard
        n_stacks = self.n_components // n_features_padded
        for t in range(n_stacks):
            for j in range(n_features_padded):
                self.cache[j] = 0
            # Bx, B is the diagonal random signed matrix
            for jj in range(n_nz):
                j = indices[jj]
                self.cache[j] = data[jj] * self.random_sign[t, j]

            # HBx: H is the Walsh-Hadamard transform
            _fwht1d(&self.cache[0], self.degree_hadamard, normalize=False)
            
            # \Pi HBx: \Pi is the random permutation matrix
            for j in range(n_features_padded):
                tmp = self.cache[j]
                k = j+self.fy_vec[t, j]
                self.cache[j] = self.cache[k]
                self.cache[k] = tmp
            
            # SG\Pi HBx: G is the diagonal random weights,
            # S is the diagonal random scaling weights
            for j in range(n_features_padded):
                self.cache[j] *= self.random_weights[t, j]
                self.cache[j] *= self.random_scaling[t, j]
            
            _fwht1d(&self.cache[0], self.degree_hadamard, False)

            for j in range(n_features_padded):
                i = j + n_features_padded*t
                z[i] = self.cache[j] / sqrt(n_features_padded)

        for i in range(self.n_components):
            if self.random_fourier:
                z[i] = cos(sqrt(2*self.gamma)*z[i]+self.random_offset[i])
                z[i] *= sqrt(2)
            z[i] /= sqrt(self.n_components)


cdef class CSubsampledRandomHadamard(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = transformer.random_weights_
        self.random_indices_rows = transformer.random_indices_rows_
        self.degree_hadamard = (self.n_features-1).bit_length()
        self.cache = array((2**self.degree_hadamard, ), sizeof(double),
                           format='d')

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t j, jj, n_features_padded
        n_features_padded = 2**self.degree_hadamard
        cdef double factor = sqrt(n_features_padded/self.n_components)
        for i in range(n_features_padded):
            self.cache[i] = 0

        # Dx, D is a random diagonal sign matrix
        for jj in range(n_nz):
            j = indices[jj]
            self.cache[j] = data[jj] * self.random_weights[j]

        # HDx: H is the Walsh-Hadamard transform
        _fwht1d(&self.cache[0], self.degree_hadamard, normalize=True)

        # RHDx: R is a random n_components \times n_features_padded matrix,
        # which represents n_components indices of rows.
        for j in range(self.n_components):
            jj = self.random_indices_rows[j]
            z[j] = self.cache[jj]
            z[j] *= factor


cdef class CRandomProjection(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]
        self.random_weights = transformer.random_weights_


    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
        for i in range(self.n_components):
            z[i] = dot(data, indices, n_nz, &self.random_weights[i, 0])
            z[i] /= sqrt(self.n_components)


cdef class CCompactRandomFeature(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_components_up = transformer.transformer_up.n_components
        self.transformer_up = get_fast_random_feature(
            transformer.transformer_up
        )
        self.transformer_down = get_fast_random_feature(
            transformer.transformer_down
        )
        self.z_cache = array((self.n_components_up, ), sizeof(double),
                             format='d')
        self.indices = np.arange(self.n_components_up, dtype=np.int32)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j
        self.transformer_up.transform(&self.z_cache[0], data, indices, n_nz)
        self.transformer_down.transform(&z[0],
                                        &self.z_cache[0],
                                        &self.indices[0],
                                        self.n_components_up)


cdef class CSignedCirculantRandomMatrix(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]

        self.random_weights = transformer.random_weights_
        self.random_sign = transformer.random_sign_
        self.random_offset = transformer.random_offset_
        self.cache = np.zeros(self.n_features, dtype=np.complex)
        self.gamma = transformer.gamma
        self.random_fourier = transformer.random_fourier
        self.n_stacks = self.random_weights.shape[0]


    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t ii, j, jj, i
        cdef double factor = sqrt(2*self.gamma)
        for ii in range(self.n_stacks):
            for j in range(self.n_features):
                self.cache[j] = 0

            for jj in range(n_nz):
                j = indices[jj]
                self.cache[j] = data[jj]

            zfft(self.cache, direction=1, overwrite_x=True)
            for j in range(self.n_features):
                self.cache[j] *= self.random_weights[ii, j]

            zfft(self.cache, direction=-1, overwrite_x=True)

            for j in range(self.n_features):
                i = ii*self.n_features + j
                z[i] = self.cache[j].real * self.random_sign[ii, j]

        if self.random_fourier:
            for i in range(self.n_components):
                z[i] = cos(factor*z[i]+self.random_offset[i])*sqrt(2)

        #print(self.cache[0], z[0])
        for i in range(self.n_components):
            z[i] /= sqrt(self.n_components)
        #print(self.cache[0].real, z[0])


cdef class CStructuredOrthogonalRandomFeature(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_stacks = transformer.random_weights_.shape[0]
        self.n_features_padded = transformer.n_components //self.n_stacks
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1] - 2*self.n_features_padded

        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_
        self.cache = array((self.n_features_padded, ), sizeof(double),
                           format='d')
        self.degree_hadamard = (self.n_features-1).bit_length()
        self.gamma = transformer.gamma
        self.random_fourier = transformer.random_fourier

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, ii, j, jj, offset
        for ii in range(self.n_stacks):
            for i in range(self.n_features_padded):
                self.cache[i] = 0

            # D1x, D is a random diagonal sign matrix
            for jj in range(n_nz):
                j = indices[jj]
                self.cache[j] = data[jj] * self.random_weights[ii, j]

            # HDx: H is the Walsh-Hadamard transform
            _fwht1d(&self.cache[0], self.degree_hadamard, normalize=True)

            # Dx, D is a random diagonal sign matrix
            offset = self.n_features
            for j in range(self.n_features_padded):
                self.cache[j] *= self.random_weights[ii, j+offset]
            _fwht1d(&self.cache[0], self.degree_hadamard, normalize=True)
            offset += self.n_features_padded

            for j in range(self.n_features_padded):
                self.cache[j] *= self.random_weights[ii, j+offset]
            _fwht1d(&self.cache[0], self.degree_hadamard, normalize=True)

            for j in range(self.n_features_padded):
                i = ii*self.n_features_padded + j
                z[i] = self.cache[j]*sqrt(self.n_features_padded)

        if self.random_fourier:
            for i in range(self.n_components):
                z[i] = cos(z[i]*sqrt(2*self.gamma)+self.random_offset[i])
                z[i] *= sqrt(2)

        for i in range(self.n_components):
            z[i] /= sqrt(self.n_components)


cdef inline void anova(double* z,
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


cdef inline void all_subsets(double* z,
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


cdef void _transform_all_fast(double[:, ::1] Z, RowDataset dataset,
                              BaseCRandomFeature transformer_fast):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    cdef double* data
    cdef int* indices
    cdef int n_nz
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&Z[i, 0], data, indices, n_nz)


def transform_all_fast(X, transformer):
    cdef RowDataset dataset = get_dataset(X, order='c')

    cdef BaseCRandomFeature transformer_fast \
        = get_fast_random_feature(transformer)
    if transformer_fast is None:
        raise ValueError("transformer has no cython implementation.")

    cdef double[:, ::1] Z = np.zeros((X.shape[0], transformer.n_components),
                                     dtype=np.float64)

    _transform_all_fast(Z, dataset, transformer_fast)
    return np.asarray(Z)


def get_fast_random_feature(transformer):
    if transformer.__class__ in RANDOMFEATURES:
        return RANDOMFEATURES[transformer.__class__](transformer)
    else:
        return None