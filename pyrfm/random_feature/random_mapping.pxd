cdef class BaseCRandomFeature(object):
    cdef Py_ssize_t n_components
    cdef Py_ssize_t n_features
    cdef void transform(self,
                        double[:] z,
                        double* data,
                        int* indices,
                        int n_nz)
    
cdef class CRandomFourier(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] offset
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

