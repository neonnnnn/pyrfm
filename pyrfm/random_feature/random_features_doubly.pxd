from ..sfmt.sfmt cimport SFMTRandomState
from libc.stdint cimport uint32_t, uint64_t
from .random_features_fast cimport BaseCRandomFeature
from .utils_random_fast cimport Categorical
from ..dataset_fast cimport RowDataset


cdef class BaseCDoublyRandomFeature(BaseCRandomFeature):
    cdef double[:] random_weights
    cdef SFMTRandomState sfmt
    cdef uint32_t seed
    cdef int n_iter

    cpdef int get_n_components(self)
    cpdef void set_n_components(self, int n_iter)
    cpdef void inc_n_components(self, int inc=*)
    cpdef void dec_n_components(self, int dec=*)
    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz)
    cdef void init_gen_rand(self)
    cdef double _transform(self, double* data, int* indices, int n_nz)
    cdef void sample_base(self)
    cdef void pred_batch(self,
                         double[:, ::1] Z,
                         double[:] y_pred,
                         double[:] coef,
                         double intercept,
                         RowDataset X,
                         int* indices_samples,
                         int batch_size,
                         int start,
                         int stop)


cdef class CDoublyRBFSampler(BaseCDoublyRandomFeature):
    cdef double scale_weight
    cdef double offset


cdef class CDoublyRandomFourier(BaseCDoublyRandomFeature):
    cdef double scale_weight
    cdef bint use_offset 
    cdef double offset


cdef class CDoublySkewedChi2Sampler(BaseCDoublyRandomFeature):
    cdef double skewedness
    cdef double offset


cdef class CDoublyRandomMaclaurin(BaseCDoublyRandomFeature):
    cdef double[:] coefs
    cdef Categorical cat
    cdef double[:] p_choice
    cdef int[:] orders
    cdef bint h01
    cdef Py_ssize_t order


cdef class CDoublyRandomKernel(BaseCDoublyRandomFeature):
    cdef int degree
    cdef int kernel 
    cdef double[:] anova
