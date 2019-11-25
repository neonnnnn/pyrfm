# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause
from libc.stdint cimport uint64_t, uint32_t
from libc.math cimport log, cos, sin, sqrt, M_PI
from libc.stdlib cimport malloc, free


cdef extern from 'SFMT.h':
    ctypedef struct sfmt_t:
        int idx
    
    void sfmt_fill_array32(sfmt_t * sfmt, uint32_t * array, int size)
    void sfmt_fill_array64(sfmt_t * sfmt, uint64_t * array, int size)
    void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed)
    void sfmt_init_by_array(sfmt_t * sfmt, uint32_t * init_key, int key_length)
    const char * sfmt_get_idstring(sfmt_t * sfmt)
    int sfmt_get_min_array_size32(sfmt_t * sfmt)
    int sfmt_get_min_array_size64(sfmt_t * sfmt)
    void sfmt_gen_rand_all(sfmt_t * sfmt)

    uint32_t sfmt_genrand_uint32(sfmt_t * sfmt) 
    uint64_t sfmt_genrand_uint64(sfmt_t * sfmt)
    double sfmt_genrand_real1(sfmt_t * sfmt) # [0, 1]
    double sfmt_genrand_real2(sfmt_t * sfmt) # [0, 1)
    double sfmt_genrand_real3(sfmt_t * sfmt) # (0, 1)


cdef class SFMTRandomState(object):
    def __cinit__(self, uint32_t seed):
        self.sfmt_state = <sfmt_t*>malloc(sizeof(sfmt_t))
        sfmt_init_gen_rand(self.sfmt_state, seed)
        self.has_gauss = False
        self.gauss = 0.

    def __dealloc__(self):
        if self.sfmt_state != NULL:
            free(self.sfmt_state)
    
    cpdef void init_gen_rand(self, uint32_t seed):
        sfmt_init_gen_rand(self.sfmt_state, seed)

    cpdef uint32_t genrand_next_uint32(self):
        return sfmt_genrand_uint32(self.sfmt_state)

    cpdef uint64_t genrand_next_uint64(self):
        return sfmt_genrand_uint64(self.sfmt_state)       
    
    cpdef uint32_t genrand_randint_uint32(self, uint32_t high):
        # Fast Random Integer Generation in an Interval
        # https://arxiv.org/abs/1805.10941
        cdef uint32_t x = sfmt_genrand_uint32(self.sfmt_state)
        cdef uint64_t m = <uint64_t>x * <uint64_t>high
        cdef uint32_t l = <uint32_t>(m & 0XFFFFFFFFUL)
        cdef uint32_t t
        if l < high:
            t = -high % high
            while (l < t):
                x = sfmt_genrand_uint32(self.sfmt_state)
                m = <uint64_t>x * <uint64_t>high
                l = <uint32_t>(m & 0xFFFFFFFFUL)
        return m >> 32

    cpdef double genrand_real_01_closed(self):
        return sfmt_genrand_real1(self.sfmt_state)

    cpdef double genrand_real_01_ropen(self):
        return sfmt_genrand_real2(self.sfmt_state)
    
    cpdef double genrand_real_01_open(self):
        return sfmt_genrand_real3(self.sfmt_state)
    
    cpdef double genrand_randn(self, double loc=0, double scale=1):
        # box-muller
        cdef double z1, z2
        if self.has_gauss:
            self.has_gauss = False
            return self.gauss
        else:
            z1 = self.genrand_real_01_open()
            z2 = self.genrand_real_01_open()
            self.has_gauss = True
            self.gauss = loc + scale*sqrt(-2*log(z1))*cos(2*M_PI*z2)
            return loc + scale*sqrt(-2*log(z1))*sin(2*M_PI*z2)

    cpdef double genrand_laplace(self, double loc=0, double scale=1):
        cdef double z
        z = self.genrand_real_01_open()
        if z >= 0.5:
            return loc - scale*log(2.0-z-z)
        else:
            return loc + scale*log(z+z)
    
    cpdef double genrand_uniform(self, double low=0, double high=1):
        return low + (high - low)*self.genrand_real_01_closed()
    
    cpdef int genrand_rademacher(self):
        return 2*self.genrand_randint_uint32(2) - 1

    cdef void genrand_randn_fill(self, double* z, int n, double loc=0, 
                                 double scale=1):
        cdef Py_ssize_t i
        for i in range(n):
            z[i] = self.genrand_randn(loc, scale)

    cdef void genrand_laplace_fill(self, double* z, int n, double loc=0, 
                                   double scale=1):
        cdef Py_ssize_t i
        for i in range(n):
            z[i] = self.genrand_laplace(loc, scale)

    cdef void genrand_uniform_fill(self, double* z, int n, double low=0,
                                   double high=1):
        cdef Py_ssize_t i
        for i in range(n):
            z[i] = self.genrand_uniform(low, high)
    
    cdef void genrand_rademacher_fill(self, double* z, int n):
        cdef Py_ssize_t i
        for i in range(n):
            z[i] = self.genrand_rademacher()