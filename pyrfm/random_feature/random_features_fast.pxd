cdef class BaseCRandomFeature(object):
    cdef Py_ssize_t n_components
    cdef Py_ssize_t n_features
    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz)


cdef class CRBFSampler(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] random_offset


cdef class CRandomFourier(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] random_offset
    cdef bint use_offset


cdef class CRandomMaclaurin(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef int[:] orders
    cdef double[:] p_choice
    cdef double[:] coefs


cdef class CTensorSketch(BaseCRandomFeature):
    cdef int degree
    cdef int[:] hash_indices
    cdef int[:] hash_signs
    cdef double[:] z_cache


cdef class CRandomKernel(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef int degree
    cdef int kernel
    cdef double[:] anova


cdef class CFastFood(BaseCRandomFeature):
    cdef double gamma
    cdef bint random_fourier
    cdef double[:, ::1] random_weights
    cdef int[:, ::1] random_sign
    cdef int[:, ::1] fy_vec
    cdef double[:, ::1] random_scaling
    cdef double[:] random_offset
    cdef int degree_hadamard
    cdef double[:] cache


cdef class CSubsampledRandomHadamard(BaseCRandomFeature):
    cdef double[:] random_weights
    cdef int[:] random_indices_rows
    cdef int degree_hadamard
    cdef double[:] cache


cdef class CRandomProjection(BaseCRandomFeature):
    cdef double[:, ::1] random_weights


cdef class CCompactRandomFeature(BaseCRandomFeature):
    cdef int n_components_up
    cdef BaseCRandomFeature transformer_up
    cdef BaseCRandomFeature transformer_down
    cdef double[:] z_cache
    cdef int[:] indices
