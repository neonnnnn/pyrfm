from ..dataset_fast cimport RowDataset, ColumnDataset


cdef class BaseCRandomFeature(object):
    cdef Py_ssize_t n_components
    cdef Py_ssize_t n_features
    cdef double scale
    cdef void transform(self,
                        double* z,
                        double* data,
                        int* indices,
                        int n_nz)


cdef class CRBFSampler(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] random_offset


cdef class CAdditiveChi2Sampler(BaseCRandomFeature):
    cdef Py_ssize_t sample_steps
    cdef double sample_interval


cdef class CSkewedChi2Sampler(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] random_offset
    cdef double skewedness


cdef class CRandomFourier(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef double[:] random_offset
    cdef bint use_offset


cdef class CRandomMaclaurin(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef int[:] orders
    cdef double[:] p_choice
    cdef double[:] coefs
    cdef double[:] cache


cdef class CSubfeatureRandomMaclaurin(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef int[:] orders
    cdef double[:] p_choice
    cdef double[:] coefs
    cdef double[:] cache
    cdef int n_sub_features
    cdef double scale_comb


cdef class CTensorSketch(BaseCRandomFeature):
    cdef int degree
    cdef int[:] hash_indices
    cdef int[:] hash_signs
    cdef complex[:] tmp1
    cdef complex[:] tmp2


cdef class CRandomKernel(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef int degree
    cdef int kernel
    cdef double[:, ::1] anova


cdef class CSubfeatureRandomKernel(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef int degree
    cdef int kernel
    cdef double[:, ::1] anova
    cdef int n_sub_features


cdef class CFastFood(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef int[:, ::1] random_sign
    cdef int[:, ::1] fy_vec
    cdef double[:, ::1] random_scaling
    cdef double[:] random_offset
    cdef double[:] cache
    cdef double gamma
    cdef bint random_fourier
    cdef int degree_hadamard


cdef class CSubsampledRandomHadamard(BaseCRandomFeature):
    cdef double[:] random_weights
    cdef int[:] random_indices_rows
    cdef int degree_hadamard
    cdef double[:] cache


cdef class CRandomProjection(BaseCRandomFeature):
    cdef RowDataset random_weights


cdef class CCompactRandomFeature(BaseCRandomFeature):
    cdef int n_components_up
    cdef BaseCRandomFeature transformer_up
    cdef BaseCRandomFeature transformer_down
    cdef double[:] z_cache
    cdef int[:] indices


cdef class CSignedCirculantRandomMatrix(BaseCRandomFeature):
    cdef complex[:, ::1] random_weights
    cdef int[:, ::1] random_sign
    cdef double[:] random_offset
    cdef complex[:] cache
    cdef double gamma
    cdef bint random_fourier
    cdef int n_stacks


cdef class COrthogonalRandomFeature(BaseCRandomFeature):
    cdef RowDataset random_weights
    cdef double[:] random_offset
    cdef bint use_offset
    cdef bint random_fourier


cdef class CStructuredOrthogonalRandomFeature(BaseCRandomFeature):
    cdef double[:, ::1] random_weights
    cdef double[:] random_offset
    cdef double[:] cache
    cdef int degree_hadamard
    cdef double gamma
    cdef bint random_fourier
    cdef int n_features_padded
    cdef int n_stacks
