# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from libc.math cimport cos, sin, sqrt, cosh, pi, log
from scipy.fftpack._fftpack import zfft
import numpy as np
cimport numpy as np
from sklearn.kernel_approximation import (RBFSampler, SkewedChi2Sampler)
from sklearn.utils import check_random_state
from ..dataset_fast import get_dataset
from ..dataset_fast cimport RowDataset, ColumnDataset
from cython.view cimport array
from . import (RandomFourier, RandomKernel, RandomMaclaurin, TensorSketch,
               FastFood, SubsampledRandomHadamard, CompactRandomFeature,
               RandomProjection, OrthogonalRandomFeature,
               SubfeatureRandomMaclaurin, StructuredOrthogonalRandomFeature,
               SignedCirculantRandomMatrix, SubfeatureRandomKernel,
               AdditiveChi2Sampler, CountSketch)
from ..ffht.fht_fast cimport _fwht1d_fast
from libcpp.vector cimport vector
from scipy.sparse import csc_matrix, csr_matrix
from scipy.special import comb


RANDOMFEATURES = {
    RandomFourier: CRandomFourier,
    RandomKernel: CRandomKernel,
    RandomMaclaurin: CRandomMaclaurin,
    SubfeatureRandomMaclaurin: CSubfeatureRandomMaclaurin,
    TensorSketch: CTensorSketch,
    RBFSampler: CRBFSampler,
    AdditiveChi2Sampler: CAdditiveChi2Sampler,
    SkewedChi2Sampler: CSkewedChi2Sampler,
    FastFood: CFastFood,
    SubsampledRandomHadamard: CSubsampledRandomHadamard,
    RandomProjection: CRandomProjection,
    CompactRandomFeature: CCompactRandomFeature,
    OrthogonalRandomFeature: COrthogonalRandomFeature,
    StructuredOrthogonalRandomFeature: CStructuredOrthogonalRandomFeature,
    SignedCirculantRandomMatrix: CSignedCirculantRandomMatrix,
    SubfeatureRandomKernel: CSubfeatureRandomKernel,
    CountSketch: CCountSketch
}


cdef inline double dot_all(double* z,
                           double* x,
                           int* indices,
                           int n_nz,
                           RowDataset W,
                           Py_ssize_t n_components
                           ):
    cdef Py_ssize_t j, jj, i, ii
    cdef double* weights
    cdef int* indices_w
    cdef int n_nz_w
    for i in range(n_components):
        z[i] = 0

    for jj in range(n_nz):
        j = indices[jj]
        W.get_row_ptr(j, &indices_w, &weights, &n_nz_w)
        for ii in range(n_nz_w):
            i = indices_w[ii]
            z[i] += x[jj] * weights[ii]


cdef class BaseCRandomFeature(object):
    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        raise NotImplementedError("This is an abstract method.")


    cdef int get_n_components(self):
        return self.n_components


cdef class CRBFSampler(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_
        self.scale = sqrt(self.n_components/2.)
        
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
            z[i] = cos(z[i]) / self.scale


cdef class CSkewedChi2Sampler(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_
        self.skewedness = transformer.skewedness
        self.scale = sqrt(self.n_components/2.)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj, n_samples
        cdef double log_data_skewdness
        # z = (cos, cos, ..., cos)
        for i in range(self.n_components):
            z[i] = self.random_offset[i]

        for jj in range(n_nz):
            j = indices[jj]
            log_data_skewdness = log(data[jj]+self.skewedness)
            for i in range(self.n_components):
                z[i] += log_data_skewdness * self.random_weights[j, i]

        for i in range(self.n_components):
            z[i] = cos(z[i]) / self.scale


cdef class CAdditiveChi2Sampler(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_features = transformer.n_features
        self.sample_steps = transformer.sample_steps
        if transformer.sample_interval is None:
            self.sample_interval = transformer.sample_interval_
        else:
            self.sample_interval = transformer.sample_interval
        self.n_components = transformer.n_components

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        cdef Py_ssize_t offset = 0
        cdef double factor_nz, step_nz, log_step_nz
        for j in range(self.n_components):
            z[j] = 0
    
        for jj in range(n_nz):
            j = indices[jj]
            z[j] = sqrt(data[jj]*self.sample_interval)
            if data[jj] < 0:
                raise ValueError("Entries of X must be non-negative.")

        offset = self.n_features       
        for jj in range(n_nz):
            j = indices[jj]
            if data[jj] != 0:
                log_step_nz = self.sample_interval * log(data[jj])
                step_nz = 2 * data[jj] * self.sample_interval
                for i in range(1, self.sample_steps):
                    factor_nz = sqrt(step_nz / cosh(pi * i * self.sample_interval))
                    z[offset*(2*i-1)+j] = factor_nz * cos(i * log_step_nz) 
                    z[offset*2*i+j] = factor_nz * sin(i * log_step_nz)

        
cdef class CRandomFourier(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = get_dataset(transformer.random_weights_, 'c')
        self.random_offset = transformer.random_offset_
        self.use_offset = transformer.use_offset
        self.scale = sqrt(self.n_components/2.)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        cdef Py_ssize_t index_offset = int(self.n_components/2)
        dot_all(z, data, indices, n_nz, self.random_weights, self.n_components)
        # z = (cos, cos, ..., cos)
        if self.use_offset:
            for i in range(self.n_components):
                z[i] += self.random_offset[i]
                z[i] = cos(z[i])/self.scale
        # z = (cos, ..., cos, sin, ..., sin)
        else:
            for i in range(index_offset):
                z[i+index_offset] = sin(z[i])/self.scale
                z[i] = cos(z[i])/self.scale


cdef class CRandomMaclaurin(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = get_dataset(transformer.random_weights_,
                                          order='c')
        self.orders = transformer.orders_
        self.p_choice = transformer.p_choice
        self.coefs = transformer.coefs
        self.cache = array((transformer.random_weights_.shape[1], ),
                           sizeof(double), format='d')
        self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, ii, k, offset, deg, n_basis
        cdef double* weigths
        cdef int* indices_w
        cdef int n_nz_w

        n_basis = self.random_weights.get_n_features()
        dot_all(&self.cache[0], data, indices, n_nz, self.random_weights,
                n_basis)

        offset = 0
        for i in range(self.n_components):
            z[i] = 1.
            deg = self.orders[i]
            for k in range(deg):
                z[i] *= self.cache[offset]
                offset += 1
            z[i] *= sqrt(self.coefs[deg]/self.p_choice[deg])
            z[i] /= self.scale


cdef class CSubfeatureRandomMaclaurin(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.n_sub_features = transformer.n_sub_features
        self.random_weights = get_dataset(transformer.random_weights_,
                                          order='c')
        self.orders = transformer.orders_
        self.p_choice = transformer.p_choice
        self.coefs = transformer.coefs
        self.cache = array((transformer.random_weights_.shape[1], ),
                           sizeof(double), format='d')
        self.scale = sqrt(self.n_components)
        # scale_comb = \sqrt{comb(d, k) / comb(d-1, k-1)} = \sqrt{d/k}
        self.scale_comb = sqrt((self.n_features*1.0) / self.n_sub_features)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, ii, k, offset, deg, n_basis
        cdef double* weigths
        cdef int* indices_w
        cdef int n_nz_w

        n_basis = self.random_weights.get_n_features()
        dot_all(&self.cache[0], data, indices, n_nz, self.random_weights,
                n_basis)

        offset = 0
        for i in range(self.n_components):
            z[i] = 1.
            deg = self.orders[i]
            for k in range(deg):
                z[i] *= self.cache[offset]
                z[i] *= self.scale_comb
                offset += 1
            z[i] *= sqrt(self.coefs[deg]/self.p_choice[deg])
            z[i] /= self.scale


cdef class CTensorSketch(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = int(len(transformer.hash_indices_)/transformer.degree)
        self.degree = transformer.degree
        self.hash_indices = transformer.hash_indices_
        self.hash_signs = transformer.hash_signs_
        self.tmp1 = np.zeros(self.n_components, dtype=np.complex)
        self.tmp2 = np.zeros(self.n_components, dtype=np.complex)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):

        cdef Py_ssize_t i, jj, j, offset
        for i in range(self.n_components):
            self.tmp1[i] = 0
        for jj in range(n_nz):
            j = indices[jj]
            self.tmp1[self.hash_indices[j]] += data[jj]*self.hash_signs[j]

        zfft(self.tmp1, direction=1, overwrite_x=True)
        for offset in range(self.n_features, self.n_features*self.degree, self.n_features):
            for i in range(self.n_components):
                self.tmp2[i] = 0

            for jj in range(n_nz):
                j = indices[jj]
                i = self.hash_indices[j+offset]
                self.tmp2[i] += data[jj]*self.hash_signs[j+offset]

            zfft(self.tmp2, direction=1, overwrite_x=True)
            for i in range(self.n_components):
                self.tmp1[i] *= self.tmp2[i]

        zfft(self.tmp1, direction=-1, overwrite_x=True)
        for i in range(self.n_components):
            z[i] = self.tmp1[i].real


cdef class CRandomKernel(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.degree = transformer.degree
        self.random_weights = get_dataset(transformer.random_weights_,
                                          order='c')
        self.scale = sqrt(self.n_components)

        if transformer.kernel in ["anova", "anova_cython"]:
            self.kernel = 0
        elif transformer.kernel == "all_subsets":
            self.kernel = 1
        else:
            raise ValueError('kernel = {} is not defined.'
                             .format(transformer.kernel))
        self.anova = array((self.n_components, self.degree+1), sizeof(double),
                           format='d')
        cdef Py_ssize_t i, j
        for i in range(self.n_components):
            self.anova[i, 0] = 1
            for j in range(self.degree):
                self.anova[i, j+1] = 0

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
        if self.kernel == 0:
            anova(z, data, indices, n_nz, self.random_weights, self.degree,
                  self.anova, self.n_components)
        elif self.kernel == 1:
            all_subsets(z, data, indices, n_nz, self.random_weights,
                        self.n_components)
        for i in range(self.n_components):
            z[i] /= self.scale


cdef class CSubfeatureRandomKernel(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_sub_features = transformer.n_sub_features
        self.n_features = transformer.random_weights_.shape[0]
        self.degree = transformer.degree
        self.random_weights = get_dataset(transformer.random_weights_,
                                          order='c')
        if transformer.kernel not in ["anova", "anova_cython"]:
            raise ValueError('kernel = {} is not defined.'
                             .format(transformer.kernel))
        self.anova = array((self.n_components, self.degree+1), sizeof(double),
                           format='d')
        cdef Py_ssize_t i, j
        for i in range(self.n_components):
            self.anova[i, 0] = 1
            for j in range(self.degree):
                self.anova[i, j+1] = 0
        self.scale = comb(self.n_features, self.degree)
        self.scale /= comb(self.n_sub_features, self.degree)
        self.scale = sqrt(self.n_components/self.scale)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
        anova(z, data, indices, n_nz, self.random_weights, self.degree,
              self.anova, self.n_components)
        for i in range(self.n_components):
            z[i] /= self.scale


cdef class CFastFood(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_sign_.shape[1]
        self.gamma = transformer.gamma
        self.random_fourier = transformer.random_fourier
        self.degree_hadamard = (self.n_features-1).bit_length()
        self.random_weights = transformer.random_weights_
        self.random_sign = transformer.random_sign_
        self.fy_vec = transformer._fy_vector_
        self.random_scaling = transformer.random_scaling_
        self.random_offset = transformer.random_offset_
        self.n_stacks = self.random_weights.shape[0]
        self.use_offset = transformer.use_offset
        self.cache = array((2**self.degree_hadamard, ), sizeof(double),
                           format='d')
        if self.random_fourier:
            self.scale = sqrt(self.n_components/2.)
        else:
            self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj, k, n_features_padded, t
        cdef double tmp, scale_weight
        cdef Py_ssize_t index_offset = int(self.n_components/2)
        scale_weight = sqrt(2*self.gamma)
        n_features_padded = 2**self.degree_hadamard

        for t in range(self.n_stacks):
            for j in range(n_features_padded):
                self.cache[j] = 0
            # Bx, B is the diagonal random signed matrix
            for jj in range(n_nz):
                j = indices[jj]
                self.cache[j] = data[jj] * self.random_sign[t, j]

            # HBx: H is the Walsh-Hadamard transform
            _fwht1d_fast(&self.cache[0], self.degree_hadamard, normalize=False)
            
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
            
            _fwht1d_fast(&self.cache[0], self.degree_hadamard, False)

            for j in range(n_features_padded):
                i = j + n_features_padded*t
                z[i] = self.cache[j] / sqrt(n_features_padded)

        if self.random_fourier:
            if not self.use_offset:
                for i in range(index_offset):
                    z[i+index_offset] = sin(scale_weight*z[i])
                    z[i] = cos(scale_weight*z[i])
            else:
                for i in range(self.n_components):
                    z[i] = cos(scale_weight*z[i]+self.random_offset[i])
            
        for i in range(self.n_components):
            z[i] /= self.scale


cdef class CSubsampledRandomHadamard(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = transformer.random_weights_
        self.random_indices_rows = transformer.random_indices_rows_
        self.degree_hadamard = (self.n_features-1).bit_length()
        self.cache = array((2**self.degree_hadamard, ), sizeof(double),
                           format='d')
        cdef int n_features_padded = 2**self.degree_hadamard
        self.scale = sqrt((1.0*self.n_components)/n_features_padded)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t j, jj
        cdef int n_features_padded = 2**self.degree_hadamard

        for i in range(n_features_padded):
            self.cache[i] = 0

        # Dx, D is a random diagonal sign matrix
        for jj in range(n_nz):
            j = indices[jj]
            self.cache[j] = data[jj] * self.random_weights[j]

        # HDx: H is the Walsh-Hadamard transform
        _fwht1d_fast(&self.cache[0], self.degree_hadamard, normalize=True)

        # RHDx: R is a random n_components \times n_features_padded matrix,
        # which represents n_components indices of rows.
        for j in range(self.n_components):
            jj = self.random_indices_rows[j]
            z[j] = self.cache[jj]
            z[j] /= self.scale


cdef class CRandomProjection(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = get_dataset(transformer.random_weights_, 'c')
        self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
        dot_all(z, data, indices, n_nz, self.random_weights, self.n_components)
        for i in range(self.n_components):
            z[i] /= self.scale


cdef class CCountSketch(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[1]
        self.hash_indices = transformer.hash_indices_
        self.hash_signs = transformer.hash_signs_
    
    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, jj, j
        cdef int sign
        for i in range(self.n_components):
            z[i] = 0
        for jj in range(n_nz):
            j = indices[jj]
            sign = self.hash_signs[j]
            i = self.hash_indices[j]
            z[i] += sign * data[jj]


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
        self.use_offset = transformer.use_offset
        if self.random_fourier:
            self.scale = sqrt(self.n_components/2.)
        else:
            self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t ii, j, jj, i
        cdef Py_ssize_t index_offset = int(self.n_components / 2)
        cdef double scale_weight = sqrt(2*self.gamma)
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
            if self.use_offset:
                for i in range(self.n_components):
                    z[i] = cos(scale_weight*z[i]+self.random_offset[i])
            else:
                for i in range(index_offset):
                    z[i] *= scale_weight
                    z[i+index_offset] = sin(z[i])
                    z[i] = cos(z[i])

        for i in range(self.n_components):
            z[i] /= self.scale


cdef class COrthogonalRandomFeature(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_components = transformer.n_components
        self.n_features = transformer.random_weights_.shape[0]
        self.random_weights = get_dataset(transformer.random_weights_, 'c')
        self.random_offset = transformer.random_offset_
        self.use_offset = transformer.use_offset
        self.random_fourier = transformer.random_fourier
        if transformer.random_fourier:
            self.scale = sqrt(self.n_components/2.)
        else:
            self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        cdef Py_ssize_t index_offset = int(self.n_components/2)
        dot_all(z, data, indices, n_nz, self.random_weights, self.n_components)
        if self.random_fourier:
            # z = (cos, cos, ..., cos)
            if self.use_offset:
                for i in range(self.n_components):
                    z[i] += self.random_offset[i]
                    z[i] = cos(z[i]) / self.scale
            # z = (cos, ..., cos, sin, ..., sin)
            else:
                for i in range(index_offset):
                    z[i+index_offset] = sin(z[i]) / self.scale
                    z[i] = cos(z[i]) / self.scale
        else:
            for i in range(self.n_components):
                z[i] = z[i] / self.scale


cdef class CStructuredOrthogonalRandomFeature(BaseCRandomFeature):
    def __init__(self, transformer):
        self.n_stacks = transformer.random_weights_.shape[0]
        self.n_components = transformer.n_components
        self.random_fourier = transformer.random_fourier
        self.use_offset = transformer.use_offset
        
        if self.random_fourier and not self.use_offset:
            self.n_features_padded = self.n_components // (self.n_stacks*2)
        else:
            self.n_features_padded = self.n_components // self.n_stacks
        self.n_features = transformer.random_weights_.shape[1] - 2*self.n_features_padded
        self.random_weights = transformer.random_weights_
        self.random_offset = transformer.random_offset_
        self.cache = array((self.n_features_padded, ), sizeof(double),
                           format='d')
        self.degree_hadamard = (self.n_features-1).bit_length()
        self.gamma = transformer.gamma
        
        self.scale = sqrt(self.n_components)

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, ii, j, jj, offset
        cdef Py_ssize_t index_offset = int(self.n_components/2)
        cdef double scale_weight = sqrt(2*self.gamma)
        for ii in range(self.n_stacks):
            for i in range(self.n_features_padded):
                self.cache[i] = 0

            # D1x, D is a random diagonal sign matrix
            for jj in range(n_nz):
                j = indices[jj]
                self.cache[j] = data[jj] * self.random_weights[ii, j]

            # HDx: H is the Walsh-Hadamard transform
            _fwht1d_fast(&self.cache[0], self.degree_hadamard, normalize=True)

            # Dx, D is a random diagonal sign matrix
            offset = self.n_features
            for j in range(self.n_features_padded):
                self.cache[j] *= self.random_weights[ii, j+offset]
            _fwht1d_fast(&self.cache[0], self.degree_hadamard, normalize=True)
            offset += self.n_features_padded

            for j in range(self.n_features_padded):
                self.cache[j] *= self.random_weights[ii, j+offset]
            _fwht1d_fast(&self.cache[0], self.degree_hadamard, normalize=True)

            for j in range(self.n_features_padded):
                i = ii*self.n_features_padded + j
                z[i] = self.cache[j]*sqrt(self.n_features_padded)

        if self.random_fourier:
            if self.use_offset:
                for i in range(self.n_components):
                    z[i] *= scale_weight
                    z[i] = cos(z[i]+self.random_offset[i])
                    z[i] *= sqrt(2)
            else:
                for i in range(index_offset):
                    z[i] *= scale_weight
                    z[i+index_offset] = sin(z[i]) * sqrt(2)
                    z[i] = cos(z[i]) * sqrt(2)

        for i in range(self.n_components):
            z[i] /= self.scale


cdef inline void anova(double* z,
                       double* data,
                       int* indices,
                       int n_nz,
                       RowDataset random_weights,
                       int degree,
                       double[:, ::1] a,
                       int n_components):
    cdef Py_ssize_t j, jj, deg, i, ii
    cdef double* weights
    cdef int* indices_w
    cdef int n_nz_w
    # init dp table
    for i in range(n_components):
        a[i, 0] = 1
        for j in range(degree):
            a[i, j+1] = 0

    for jj in range(n_nz):
        j = indices[jj]
        random_weights.get_row_ptr(j, &indices_w, &weights, &n_nz_w)
        for ii in range(n_nz_w):
            i = indices_w[ii]
            for deg in range(degree):
                a[i, degree-deg] += weights[ii]*data[jj]*a[i, degree-deg-1]

    for i in range(n_components):
        z[i] = a[i, degree]


cdef inline void all_subsets(double* z,
                             double* data,
                             int* indices,
                             int n_nz,
                             RowDataset random_weights,
                             int n_components):
    cdef Py_ssize_t j, jj, i, ii
    cdef double* weights
    cdef int* indices_w
    cdef int n_nz_w
    for i in range(n_components):
        z[i] = 1
    for jj in range(n_nz):
        j = indices[jj]
        random_weights.get_row_ptr(j, &indices_w, &weights, &n_nz_w)
        for ii in range(n_nz_w):
            i = indices_w[ii]
            z[i] *= (1+data[jj]*weights[ii])


cdef _transform_all_fast_dense(RowDataset dataset,
                               BaseCRandomFeature transformer_fast):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    cdef Py_ssize_t n_components = transformer_fast.n_components
    cdef double* data
    cdef int* indices
    cdef int n_nz
    cdef double[:, ::1] Z = array((n_samples, n_components), sizeof(double),
                                  format='d')
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&Z[i, 0], data, indices, n_nz)
    return np.asarray(Z)


cdef _transform_all_fast_sparse(RowDataset dataset,
                                BaseCRandomFeature transformer_fast):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    cdef Py_ssize_t n_components = transformer_fast.n_components
    cdef double* data
    cdef int* indices
    cdef int n_nz, n_nz_output
    cdef double[:] z = array((n_components,), sizeof(double), format='d')
    cdef vector[double] data_vec
    cdef vector[int] col_vec
    cdef np.ndarray[np.float64_t, ndim=1] data_arr
    cdef np.ndarray[np.int32_t, ndim=1] indices_arr
    cdef np.ndarray[np.int32_t, ndim=1] indptr

    indptr = np.zeros(n_samples+1, dtpye=np.int32)
    n_nz_output = 0
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&z[0], data, indices, n_nz)
        for j in range(n_components):
            if z[j] != 0:
                data_vec.push_back(z[j])
                col_vec.push_back(i)
                indptr[i+1] += 1
                n_nz_output += 1
        indptr[i+1] += indptr[i]

    data_arr = np.empty((n_nz_output, ), dtype=np.float64)
    indices_arr = np.empty((n_nz_output, ), dtype=np.int32)
    for j in range(n_nz_output):
        data_arr[j] = data_vec[j]
        indices_arr[j] = col_vec[j]
    return csr_matrix((data_arr, indices_arr, indptr),
                      shape=(n_samples, n_components))


def transform_all_fast(X, transformer, dense_output=True):
    cdef RowDataset dataset = get_dataset(X, order='c')

    cdef BaseCRandomFeature transformer_fast \
        = get_fast_random_feature(transformer)
    if transformer_fast is None:
        raise ValueError("transformer has no cython implementation.")

    if dense_output:
        return _transform_all_fast_dense(dataset, transformer_fast)
    else:
        return _transform_all_fast_sparse(dataset, transformer_fast)


def get_fast_random_feature(transformer):
    if transformer.__class__ in RANDOMFEATURES:
        return RANDOMFEATURES[transformer.__class__](transformer)
    else:
        return None
