# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

# Author: Kyohei Atarashi
# License: BSD-2-Clause

from cython.view cimport array
from sklearn.utils import check_random_state
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix
from libc.limits cimport UINT_MAX
from libc.stdint cimport uint64_t, uint32_t
from ..sfmt.sfmt cimport SFMTRandomState


# efficient sampling algorithm for sparse rademacher matrix
def sparse_rademacher(rng, int[:] size, double p_sparse):
    # size = (n_features, n_components)
    # Preprocess for walker alias method: O(n_components)
    cdef uint32_t i, j, n_nz, n_nz_i, n_components, n_features
    cdef uint32_t offset = 0
    n_features = size[0]
    n_components = size[1]
    cdef Binomial binom = Binomial(p=1-p_sparse, n=n_features,
                                   random_state=rng)
    # n_nzs[i]: number of nonzero elements in i-th columns of random matrix
    cdef int[:] n_nzs = np.zeros(n_components, dtype=np.int32)
    cdef int[:] arange = np.arange(n_features, dtype=np.int32)

    # sampling number of nonzero elements in each column : O(n_components)
    n_nz = 0
    for i in range(n_components):
        n_nzs[i] = binom.get_sample()
        n_nz += n_nzs[i]
    cdef uint32_t[:] row_ind = np.zeros(n_nz, dtype=np.uint32)
    cdef uint32_t[:] col_ind = np.zeros(n_nz, dtype=np.uint32)

    # sampling nonzero row indices : O(\sum_{i=1}^{n_components} nnz_i=nnz)
    offset = 0
    cdef SFMTRandomState sfmt = SFMTRandomState(rng.randint(UINT_MAX))
    for i in range(n_components):
        fisher_yates_shuffle(arange, n_features, n_nzs[i], sfmt)
        n_nz_i = n_nzs[i]
        for j in range(n_nz_i):
            col_ind[offset+j] = i
            row_ind[offset+j] = arange[j]
        offset += n_nz_i

    # sampling nonzero elements: O(nnz)
    data = (rng.randint(2, size=n_nz)*2-1) / np.sqrt(1-p_sparse)
    return csr_matrix((data, (row_ind, col_ind)), shape=size)


cdef inline void fisher_yates_shuffle(int[:] permutation, uint32_t length, 
                                      uint32_t n, SFMTRandomState sfmt):
    cdef uint32_t i, j
    cdef int tmp
    for i in range(n):
        j = sfmt.genrand_randint_uint32(length-i)
        tmp = permutation[i]
        permutation[i] = permutation[j+i]
        permutation[j+i] = tmp


cdef class Categorical:
    def __init__(self, frequent, random_state=None):
        self.n_categories = len(frequent)
        self.random_state = check_random_state(random_state)
        self.sfmt = SFMTRandomState(<uint32_t>self.random_state.randint(UINT_MAX))

    def __cinit__(self, frequent, random_state=None):
        cdef int i, j, ii, n_overfull, n_underfull
        cdef uint32_t n_categories = len(frequent)
        self.indices_another = np.zeros(n_categories, dtype=np.int32)
        self.frequent = np.zeros(n_categories, dtype=np.double)
        cdef int[:] underfull = array((n_categories, ), sizeof(int), format='i')
        cdef int[:] overfull = array((n_categories, ), sizeof(int), format='i')
        self.sum = 0
        for i in range(n_categories):
            self.frequent[i] = frequent[i]*n_categories
            self.sum += frequent[i]
            overfull[i] = -1
            underfull[i] = -1
            self.indices_another[i] = -1

        n_overfull = 0
        n_underfull = 0
        for i in range(n_categories):
            if self.frequent[i] >= self.sum:
                overfull[n_overfull] = i
                n_overfull += 1
            elif self.frequent[i] < self.sum:
                underfull[n_underfull] = i
                n_underfull += 1
        # init alias table
        ii = 0
        while n_overfull > 0:
            i = underfull[ii]
            if i == -1:
                break
            j = overfull[n_overfull-1]
            self.indices_another[i] = j
            self.frequent[j] -= self.sum - self.frequent[i]

            if self.frequent[j] == self.sum:
                n_overfull -= 1
            elif self.frequent[j] < self.sum:
                n_overfull -= 1
                underfull[n_underfull] = j
                n_underfull += 1
            ii += 1

    def get_samples(self, size):
        cdef uint32_t i, j
        cdef int _size
        cdef double u
        if isinstance(size, int):
            _size = size
        else:
            _size = np.prod(size)
        ret = np.zeros(_size, dtype=np.int32)
        for i in range(_size):
            u = self.random_state.uniform(0, self.sum)
            j = self.sfmt.genrand_randint_uint32(self.n_categories)
            if u <= self.frequent[j]:
                ret[i] = j
            else:
                ret[i] = self.indices_another[j]
        return ret.reshape(size)

    cdef int get_sample(self):
        cdef uint32_t i  = self.sfmt.genrand_randint_uint32(self.n_categories)
        cdef double u = self.sfmt.genrand_uniform(0., self.sum)
        if u <= self.frequent[i]:
            return i
        else:
            return self.indices_another[i]

    cdef void init_gen_rand(self, unsigned int seed):
        self.sfmt.init_gen_rand(seed)


cdef class Binomial:
    def __init__(self, p, n, random_state=None):
        self.p = p
        self.n_categories = n
        self.random_state = check_random_state(random_state)

    def __cinit__(self, p, n, random_state=None):
        cdef double[:] frequent = array((n+1, ), sizeof(double), format='d')
        cdef int i
        cdef int mode= int((n+1) * p)
        frequent[mode] = 1000.
        for i in range(mode+1, n+1):
            frequent[i] = frequent[i-1] * p / (1-p)
            frequent[i] *= (n-i+1) / i
        for i in range(mode-1, -1, -1):
            frequent[i] = (i+1)*frequent[i+1] * (1-p) / p
            frequent[i] /= (n-i)

        self.cat = Categorical(frequent, random_state)

    def get_samples(self, size):
        return self.cat.get_samples(size)

    cdef int get_sample(self):
        return self.cat.get_sample()


cdef inline void _get_subfeatures_indices(int[:] indices,
                                          int[:] perm,
                                          uint32_t n_components,
                                          uint32_t n_features,
                                          uint32_t n_sub_features,
                                          rng):
    cdef uint32_t offset, i, j
    cdef int ii
    offset = 0
    cdef SFMTRandomState sfmt = SFMTRandomState(rng.randint(UINT_MAX))
    for j in range(n_components):
        fisher_yates_shuffle(perm, n_features, n_sub_features, sfmt)
        for i in range(n_sub_features):
            indices[offset+i] = perm[i]
        offset += n_sub_features


def get_subfeatures_indices(int n_components, int n_features,
                            int n_sub_features, rng):
    cdef int size = n_components * n_sub_features
    cdef np.ndarray[int, ndim=1] indices = np.zeros(size, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] perm = np.arange(n_features, dtype=np.int32)

    _get_subfeatures_indices(indices, perm, n_components, n_features,
                             n_sub_features, rng)
    return indices


cdef void _fisher_yates_shuffle_with_indices(int[:] permutation,
                                             int[:] fisher_yates_vec,
                                             SFMTRandomState sfmt):
    cdef Py_ssize_t i, length, j
    cdef int tmp
    length = permutation.shape[0]
    for i in range(length):
        j = sfmt.genrand_randint_uint32(length-i)
        fisher_yates_vec[i] = j
        tmp = permutation[i]
        permutation[i] = permutation[i+j]
        permutation[i+j] = tmp


def fisher_yates_shuffle_with_indices(int n, rng):
    # sampled indices matrix (represent exchanging)
    cdef np.ndarray[int, ndim=1] fy_vec = np.zeros(n, dtype=np.int32)
    # shuffled matrix
    cdef np.ndarray[int, ndim=1] perm = np.arange(n, dtype=np.int32)
    cdef SFMTRandomState sfmt = SFMTRandomState(rng.randint(UINT_MAX))
    _fisher_yates_shuffle_with_indices(perm, fy_vec, sfmt)
    return perm, fy_vec

