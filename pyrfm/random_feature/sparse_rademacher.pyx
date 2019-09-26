# cython: language_level=3
# cython: cdivision=True
# cython: boudscheck=False
# cython: wraparound=False

from cython.view cimport array
from sklearn.utils import check_random_state
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix


# efficient sampling algorithm for sparse rademacher matrix
def sparse_rademacher(rng, int[:] size, double p_sparse):
    # size = (n_features, n_components)
    # Preprocess for walker alias method: O(n_components)
    cdef Py_ssize_t i, j, n_nz
    cdef Py_ssize_t n_components, n_features
    cdef Py_ssize_t offset = 0
    n_features = size[0]
    n_components = size[1]
    cdef Binomial binom = Binomial(p=1-p_sparse, n=n_features,
                                   random_state=rng)
    # n_nzs[i]: number of nonzero elements in i-th columns of random matrix
    cdef int[:] n_nzs = np.zeros(n_components, dtype=np.int32)
    cdef int[:] arange = np.arange(n_features, dtype=np.int32)

    # sampling number of nonzero elements in each column : O(n_features)
    n_nz = 0
    for i in range(n_components):
        n_nzs[i] = binom.get_sample()
        n_nz += n_nzs[i]
    cdef int[:] row_ind = np.zeros(n_nz, dtype=np.int32)
    cdef int[:] col_ind = np.zeros(n_nz, dtype=np.int32)

    # sampling nonzero row indices : O(\sum_{j=1}^{n_features} nnz_j=nnz)
    offset = 0
    for i in range(n_components):
        fisher_yates_shuffle(arange, n_nzs[i], rng)
        for j in range(n_nzs[i]):
            col_ind[offset+j] = i
            row_ind[offset+j] = arange[j]
        offset += n_nzs[i]

    # sampling nonzero elements: O(nnz)
    data = (rng.randint(2, size=n_nz)*2-1) / np.sqrt(1-p_sparse)
    return csr_matrix((data, (row_ind, col_ind)), shape=size)


cdef inline void fisher_yates_shuffle(int[:] permutation, int n, rng):
    cdef Py_ssize_t i, j
    cdef int tmp
    cdef lengh = len(permutation)
    for i in range(n):
        j = rng.randint(0, lengh-i)
        tmp = permutation[i]
        permutation[i] = permutation[j+i]
        permutation[j+i] = tmp


cdef class Categorical:
    cdef int[:] indices_another
    cdef double[:] frequent
    cdef public double sum
    cdef public int n_categories
    cdef public object random_state

    def __init__(self, frequent, random_state=None):
        self.n_categories = len(frequent)
        self.random_state = check_random_state(random_state)

    def __cinit__(self, frequent, random_state=None):
        cdef int i, j, ii, n_overfull, n_underfull
        cdef int n_categories = len(frequent)
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
            if self.frequent[i] > self.sum:
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
        cdef int i, j
        cdef int _size
        cdef double u
        if isinstance(size, int):
            _size = size
        else:
            _size = np.prod(size)
        ret = np.zeros(_size, dtype=np.int32)
        for i in range(_size):
            u = self.random_state.uniform(0, self.sum)
            j = self.random_state.randint(0, self.n_categories)
            if u <= self.frequent[j]:
                ret[i] = j
            else:
                ret[i] = self.indices_another[j]
        return ret.reshape(size)

    cdef inline int get_sample(self):
        cdef int i  = self.random_state.randint(0, self.n_categories)
        cdef double u = self.random_state.uniform(0, self.sum)
        if u <= self.frequent[i]:
            return i
        else:
            return self.indices_another[i]


cdef class Binomial:
    cdef public double p
    cdef public int n_categories
    cdef public object random_state
    cdef public Categorical cat

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

    cdef inline int get_sample(self):
        return self.cat.get_sample()
