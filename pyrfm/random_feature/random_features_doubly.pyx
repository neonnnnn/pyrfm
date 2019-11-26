# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from libc.math cimport cos, sin, sqrt, cosh, log, M_PI, tan
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
               AdditiveChi2Sampler)
from ..sfmt.sfmt cimport SFMTRandomState
from .utils_random_fast cimport Categorical
from libc.limits cimport UINT_MAX
import warnings


RANDOMFEATURES = {
    RandomFourier: CDoublyRandomFourier,
    RandomKernel: CDoublyRandomKernel,
    RandomMaclaurin: CDoublyRandomMaclaurin,
    RBFSampler: CDoublyRBFSampler,
    SkewedChi2Sampler: CDoublySkewedChi2Sampler,
}


cdef class BaseCDoublyRandomFeature(BaseCRandomFeature):
    def __init__(self, transformer, n_features):
        self.n_features = n_features
        random_state = check_random_state(transformer.random_state)
        self.seed = random_state.randint(UINT_MAX)
        self.sfmt = SFMTRandomState(self.seed)
        self.random_weights = array((self.n_features, ), sizeof(double), 
                                     format='d')
        self.n_iter = 1

    cpdef void set_n_components(self, int n_iter):
        self.n_iter = n_iter

    cpdef void inc_n_components(self, int inc=1):
        self.n_iter = self.n_iter + inc

    cpdef void dec_n_components(self, int dec=1):
        self.n_iter = self.n_iter - dec

    cpdef int get_n_components(self):
        return self.n_iter
    
    cdef void sample_base(self):
        raise NotImplementedError("This is an abstract method.")

    cdef double _transform(self, double* data, int* indices, int n_nz):
        raise NotImplementedError("This is an abstract method.")

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        raise NotImplementedError("This is an abstract method.")

    cdef void init_gen_rand(self):
        self.sfmt.init_gen_rand(self.seed)
    
    cdef void pred_batch(self,
                         double[:, ::1] Z,
                         double[:] y_pred,
                         double[:] coef,
                         double intercept,
                         RowDataset X,
                         int* indices_samples,
                         int batch_size,
                         int start,
                         int stop):
        cdef Py_ssize_t n, nn, ii, s
        # data pointers
        cdef int* indices
        cdef double* data
        cdef int n_nz
        self.init_gen_rand()
        s = 0
        for nn in range(batch_size):
            y_pred[nn] = intercept
        for i in range(self.n_iter):
            self.sample_base()
            for nn in range(batch_size):
                n = indices_samples[nn]
                X.get_row_ptr(n, &indices, &data, &n_nz)
                if (start <= i) and (i < stop):
                    Z[nn, s] = self._transform(data, indices, n_nz)
                    y_pred[nn] += coef[i] * Z[nn, s]
                else:
                    y_pred[nn] += coef[i]*self._transform(data, indices, n_nz)
            if (start <= i) and (i < stop):
                s += 1
            

cdef class CDoublyRBFSampler(BaseCDoublyRandomFeature):
    def __init__(self, transformer, n_features):
        super(CDoublyRBFSampler, self).__init__(transformer, n_features)
        self.scale_weight = sqrt(2*transformer.gamma)
        self.scale = sqrt(2)

    cdef void sample_base(self):
        self.sfmt.genrand_randn_fill(&self.random_weights[0], 
                                     self.n_features, 0, 1.) 
        self.offset = self.sfmt.genrand_uniform(0, 2*M_PI)
    
    cdef double _transform(self, double* data, int* indices, int n_nz):
        cdef double z = 0
        cdef Py_ssize_t jj, j
        for jj in range(n_nz):
            j = indices[jj]
            z += data[jj] * self.random_weights[j]
        z *= self.scale_weight
        z += self.offset
        return cos(z) * self.scale

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        self.sfmt.init_gen_rand(self.seed)
        for i in range(self.n_iter):
            # sampling random weights
            self.sample_base()
            z[i] = self._transform(data, indices, n_nz)


cdef class CDoublyRandomFourier(BaseCDoublyRandomFeature):
    def __init__(self, transformer, n_features):
        super(CDoublyRandomFourier, self).__init__(transformer, n_features)
        cdef double gamma
        if transformer.gamma == 'auto':
            gamma = 1.0 / n_features
        else:
            gamma = transformer.gamma
        self.scale_weight = sqrt(2*gamma)
        self.use_offset = transformer.use_offset
        self.scale = sqrt(2)
        self.offset = 0.

    cpdef int get_n_components(self):
        if self.use_offset:
            return self.n_iter
        else:
            return 2*self.n_iter

    cdef void sample_base(self):
        self.sfmt.genrand_randn_fill(&self.random_weights[0], 
                                     self.n_features, 0, 1.) 
        self.offset = self.sfmt.genrand_uniform(0, 2*M_PI)

    cdef void pred_batch(self,
                         double[:, ::1] Z,
                         double[:] y_pred,
                         double[:] coef,
                         double intercept,
                         RowDataset X,
                         int* indices_samples,
                         int batch_size,
                         int start,
                         int stop):
        cdef Py_ssize_t n, nn, i, jj, j, s
        # data pointers
        cdef int* indices
        cdef double* data
        cdef int n_nz
        cdef double z
        s = 0
        self.sfmt.init_gen_rand(self.seed)
        for nn in range(batch_size):
            y_pred[nn] = intercept
        
        for i in range(self.n_iter):
            self.sfmt.genrand_randn_fill(&self.random_weights[0], 
                                         self.n_features, 0, 1.) 
            self.offset = self.sfmt.genrand_uniform(0, 2*M_PI)
            for nn in range(batch_size):
                n = indices_samples[nn]
                X.get_row_ptr(n, &indices, &data, &n_nz)
                z = 0
                # z = (cos, cos, ...)
                if self.use_offset:
                    for jj in range(n_nz):
                        j = indices[jj]
                        z += data[jj] * self.random_weights[j]
                    z *= self.scale_weight
                    z += self.offset
                    y_pred[nn] += cos(z) * self.scale * coef[i]
                    if Z is not None and (start <= i) and (i < stop):
                        Z[nn, s] = cos(z) * self.scale
                # z = (cos, sin, ...)
                else:
                    for jj in range(n_nz):
                        j = indices[jj]
                        z += data[jj] * self.random_weights[j]
                    z *= self.scale_weight
                    y_pred[nn] += coef[2*i] * cos(z) * self.scale
                    y_pred[nn] += coef[2*i+1] * sin(z) * self.scale
                    if (Z is not None) and (start <= 2*i) and (2*i+1 < stop):
                        Z[nn, 2*s] = cos(z) * self.scale
                        Z[nn, 2*s+1] = sin(z) * self.scale
            
            if self.use_offset:
                if (start <= i) and (i < stop):
                    s += 1
            else:
                if (start <= 2*i) and (2*i+1 < stop):
                    s += 1

    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, j, jj
        self.sfmt.init_gen_rand(self.seed)
        for i in range(self.n_iter):
            # sampling random bases
            self.sample_base()
            # z = (cos, cos, ..., cos)
            if self.use_offset:
                z[i] = 0
                for jj in range(n_nz):
                    j = indices[jj]
                    z[i] += data[jj] * self.random_weights[j]
                z[i] *= self.scale_weight
                z[i] += self.offset
                z[i] = cos(z[i]) * self.scale
            # z = (cos, ..., cos, sin, ..., sin)
            else:
                z[2*i] = 0
                for jj in range(n_nz):
                    j = indices[jj]
                    z[2*i] += data[jj] * self.random_weights[j]
                z[2*i] *= self.scale_weight
                z[2*i+1] = sin(z[2*i]) * self.scale
                z[2*i] = cos(z[2*i]) * self.scale


cdef class CDoublySkewedChi2Sampler(BaseCDoublyRandomFeature):
    def __init__(self, transformer, n_features):
        super(CDoublySkewedChi2Sampler, self).__init__(transformer, n_features)
        self.skewedness = transformer.skewedness
        self.scale = sqrt(2)

    cdef void sample_base(self):
        self.sfmt.genrand_uniform_fill(&self.random_weights[0],
                                       self.n_features, 0., 1.)
        self.offset = self.sfmt.genrand_uniform(0, 2*M_PI)
    
    cdef double _transform(self, double* data, int* indices, int n_nz):
        cdef double z = 0
        cdef Py_ssize_t jj, j
        cdef double rw, log_data_skewdness
        for jj in range(n_nz):
            j = indices[jj]
            log_data_skewdness = log(data[jj]+self.skewedness)
            rw = (1./M_PI)*log(tan(M_PI/2.*self.random_weights[j]))
            z += log_data_skewdness*rw
        return cos(z+self.offset) * self.scale

    cdef void transform(self, 
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, jj, j
        self.sfmt.init_gen_rand(self.seed)
        cdef double rw = 0
        for i in range(self.n_iter):
            z[i] = 0
            # sample bases
            self.sample_base()
            for jj in range(n_nz):
                j = indices[jj]
                log_data_skewdness = log(data[jj]+self.skewedness)
                rw = (1./M_PI)*log(tan(M_PI/2.*self.random_weights[j]))
                z[i] += log_data_skewdness*rw
            z[i] += self.offset
            z[i] = cos(z[i])*self.scale


cdef class CDoublyRandomMaclaurin(BaseCDoublyRandomFeature):
    def __init__(self, transformer, n_features):
        super(CDoublyRandomMaclaurin, self).__init__(transformer, n_features)
        cdef double gamma
        if transformer.gamma == 'auto':
            gamma = 1.0 / n_features
        else:
            gamma = transformer.gamma
        transformer._set_coefs(gamma)
        random_state = check_random_state(transformer.random_state)
        transformer._sample_orders(random_state)
        self.p_choice = transformer.p_choice
        self.cat = Categorical(self.p_choice, transformer.random_state)
        self.coefs = transformer.coefs
        if transformer.h01:
            warnings.warn("h01 heuristisc is not valid for doubly optimizer.")
    
    cdef void init_gen_rand(self):
        self.sfmt.init_gen_rand(self.seed)
        self.cat.init_gen_rand(self.seed)

    cdef void sample_base(self):
        self.order = self.cat.get_sample()
        self.sfmt.genrand_rademacher_fill(&self.random_weights[0],
                                          self.n_features)

    cdef double _transform(self, double* data, int* indices, int n_nz):
        cdef Py_ssize_t k, j, jj
        cdef double z, cache
        z = 1
        for k in range(self.order):
            cache = 0
            for jj in range(n_nz):
                j = indices[jj]
                cache += data[jj] * self.random_weights[j]
            z *= cache
        z *= sqrt(self.coefs[self.order]/self.p_choice[self.order])
        return z

    cdef void transform(self, 
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i, jj, j, k, offset
        cdef double cache
        self.sfmt.init_gen_rand(self.seed)
        self.cat.init_gen_rand(self.seed)
        offset = 0
        for i in range(self.n_iter):
            self.sample_base()
            
            z[i+offset] = 1.
            for k in range(self.order):
                cache = 0
                for jj in range(n_nz):
                    j = indices[jj]
                    cache += data[jj] * self.random_weights[j]
                z[i+offset] *= cache
            z[i+offset] *= sqrt(self.coefs[self.order]/self.p_choice[self.order])


cdef class CDoublyRandomKernel(BaseCDoublyRandomFeature):
    def __init__(self, transformer, n_features):
        super(CDoublyRandomKernel, self).__init__(transformer, n_features)
        self.degree = transformer.degree
        if transformer.kernel in ["anova", "anova_cython"]:
            self.kernel = 0
        elif transformer.kernel == "all_subsets":
            self.kernel = 1
        else:
            raise ValueError('kernel = {} is not defined.'
                             .format(transformer.kernel))
        self.anova = array((self.degree+1, ), sizeof(double), format='d')
        cdef Py_ssize_t j
        for j in range(self.degree):
            self.anova[j+1] = 0
        self.anova[0] = 1
   
    cdef double _transform(self, double* data, int* indices, int n_nz):
        if self.kernel == 0:
            return anova(data, indices, n_nz, self.random_weights,
                         self.degree, self.anova)
        else:
            return all_subsets(data, indices, n_nz, self.random_weights)

    cdef void sample_base(self):
        self.sfmt.genrand_rademacher_fill(&self.random_weights[0],
                                          self.n_features)

    cdef void transform(self, 
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz):
        cdef Py_ssize_t i
        self.sfmt.init_gen_rand(self.seed)
        for i in range(self.n_iter):
            self.sample_base()
            if self.kernel == 0:
                z[i] = anova(data, indices, n_nz, self.random_weights,
                             self.degree, self.anova)
            else:
                z[i] = all_subsets(data, indices, n_nz, self.random_weights)


cdef inline double anova(double* data,
                         int* indices,
                         int n_nz,
                         double[:] random_weights,
                         int degree,
                         double[:] a):
    cdef Py_ssize_t j, jj, deg
    # init dp table
    a[0] = 1
    for j in range(degree):
        a[j+1] = 0

    for jj in range(n_nz):
        j = indices[jj]
        for deg in range(degree):
            a[degree-deg] += random_weights[j]*data[jj]*a[degree-deg-1]
    return a[degree]
    

cdef inline double all_subsets(double* data,
                               int* indices,
                               int n_nz,
                               double[:] random_weights):
    cdef Py_ssize_t j, jj
    cdef double result = 1
    for jj in range(n_nz):
        j = indices[jj]
        result *= (1+data[jj]*random_weights[j])
    return result


cdef _transform_all_fast_dense(RowDataset dataset,
                               BaseCDoublyRandomFeature transformer_fast,
                               int n_iter):
    cdef Py_ssize_t i
    cdef Py_ssize_t n_samples = dataset.get_n_samples()
    if n_iter is not None:
        transformer_fast.set_n_components(n_iter)

    cdef Py_ssize_t n_components = transformer_fast.get_n_components()
    cdef double* data
    cdef int* indices
    cdef int n_nz
    cdef double[:, ::1] Z = array((n_samples, n_components), sizeof(double),
                                  format='d')
    for i in range(n_samples):
        dataset.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&Z[i, 0], data, indices, n_nz)
    return np.asarray(Z) / sqrt(Z.shape[1])


def transform_all_doubly(X, transformer, n_iter=None):
    cdef RowDataset dataset = get_dataset(X, order='c')

    cdef BaseCDoublyRandomFeature transformer_fast \
        = get_doubly_random_feature(transformer, X.shape[1])
    if transformer_fast is None:
        raise ValueError("transformer has no cython implementation.")
    
    return _transform_all_fast_dense(dataset, transformer_fast, n_iter)


def get_doubly_random_feature(transformer, n_features):
    if transformer.__class__ in RANDOMFEATURES:
        return RANDOMFEATURES[transformer.__class__](transformer, n_features)
    else:
        return None