# cython: language_level=3

cdef class LossFunction:
    cdef double mu
    cdef double loss(self, double p, double y)
    cdef double dloss(self, double p, double y)
    cdef double conjugate(self, double alpha, double y)
    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale)