# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause
from libc.stdint cimport uint64_t, uint32_t


cdef extern from "SFMT.h":
    ctypedef struct sfmt_t:
        int idx


cdef class SFMTRandomState:
    cdef sfmt_t *sfmt_state
    cdef bint has_gauss
    cdef double gauss

    cpdef void init_gen_rand(self, uint32_t seed)
    cpdef uint32_t genrand_next_uint32(self)
    cpdef uint64_t genrand_next_uint64(self)
    cpdef uint32_t genrand_randint_uint32(self, uint32_t high) 
    cpdef double genrand_real_01_closed(self) # [0, 1]
    cpdef double genrand_real_01_ropen(self) # [0, 1)
    cpdef double genrand_real_01_open(self) # (0, 1)
    cpdef double genrand_randn(self, double loc=*, double scale=*)
    cpdef double genrand_laplace(self, double loc=*, double scale=*)
    cpdef double genrand_uniform(self, double low=*, double high=*)
    cpdef int genrand_rademacher(self)

    cdef void genrand_randn_fill(self, double* z, int n, double loc=*,
                                 double scale=*)
    cdef void genrand_laplace_fill(self, double* z, int n, double loc=*,
                                   double scale=*)
    cdef void genrand_uniform_fill(self, double* z, int n, double low=*,
                                   double high=*)
    cdef void genrand_rademacher_fill(self, double* z, int n)