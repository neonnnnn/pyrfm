# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from cython.view cimport array
from libc.limits cimport UINT_MAX
from libc.stdint cimport uint64_t, uint32_t
from ..sfmt.sfmt cimport SFMTRandomState


cdef class Categorical:
    cdef int[:] indices_another
    cdef double[:] frequent
    cdef public double sum
    cdef public uint32_t n_categories
    cdef public object random_state
    cdef SFMTRandomState sfmt

    cdef int get_sample(self)
    cdef void init_gen_rand(self, unsigned int seed)


cdef class Binomial:
    cdef public double p
    cdef public int n_categories
    cdef public object random_state
    cdef public Categorical cat

    cdef int get_sample(self)
