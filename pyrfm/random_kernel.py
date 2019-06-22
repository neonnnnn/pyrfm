import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix, issparse
from math import sqrt
from .kernels import anova, all_subsets, anova_fast


def _anova(degree=2, dense_output=True):
    def __anova(X, P):
        return anova(X, P, degree, dense_output)

    return __anova


def _anova_fast(degree=2, dense_output=True):
    def __anova_fast(X, P):
        return anova_fast(X, P, degree, dense_output=dense_output)
    return __anova_fast


def dot():
    def _dot(X, Y, dense_output=True):
        return safe_sparse_dot(X, Y.T, dense_output)

    return _dot


def get_random_matrix(rng, distribution, size, p=0.):
    # size = (n_components, n_features)
    if distribution == 'rademacher':
        return (rng.randint(2, size=size)*2 - 1).astype(np.float64z)
    elif distribution in ['gaussian', 'normal']:
        return rng.normal(0, 1, size)
    elif distribution == 'uniform':
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    elif distribution == 'laplace':
        return rng.laplace(0, 1./np.sqrt(2), size)
    elif distribution == 'sparse_rademacher':
        """
        mat = rng.choice([-1., 0., 1.], size=size,
                         p=[(1-p)/2., p, (1-p)/2.])

        return csr_matrix(mat) / np.sqrt(1-p)
        """
        # n_nzs : (n_features, )
        # n_nzs[j] is n_nz of random_weights[:, j]
        n_nzs = rng.binomial(size[0], 1-p, size[1])

        indptr = np.append(0, np.cumsum(n_nzs))
        indices = rng.choice(np.arange(size[0]), size=n_nzs[0], replace=False)
        arange = np.arange(size[0])

        for nnz in n_nzs[1:]:
            indices = np.append(indices,
                                rng.choice(arange, size=nnz, replace=False))

        data = (rng.randint(2, size=np.sum(n_nzs))*2-1) / np.sqrt(1-p)
        return csc_matrix((data, indices.ravel(), indptr), shape=size)

    else:
        raise ValueError('{} distribution is not implemented. Please use'
                         'rademacher, gaussian (normal), uniform or laplace.'
                         .format(distribution))


def get_feature_indices(rng, n_sub_features, n_features, n_components):
    arange = np.arange(n_features)
    mat = np.array(
        [rng.choice(arange, size=n_sub_features, replace=False)
         for _ in range(n_components)]
    )
    return mat.ravel()


class RandomKernel(BaseEstimator, TransformerMixin):
    """Approximates feature map of the ANOVA/all-subsets kernel by Monte Carlo
    approximation by Random Kernel Feature map.

    Parameters
    ----------
    n_components : int
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    kernel : str
        Kernel to be approximated.
        "anova", "dot", or "all-subsets" can be used.

    degree : int
        Parameter of the ANOVA kernel.

    distribution : str
        Distribution for random_weights_.
        "rademacher", "gaussian", "laplace", "uniform", or "sparse_rademacher"
        can be used.

    dense_output : bool, default=True
        Whether randomized feature matrix is dense or sparse.
        For kernel='anova', if dense_output = False,
        distribution='sparse_rademacher', and X is sparse matrix, output random
        feature matrix will become sparse matrix.
        For kernel='anova_cython', if dense_output=False, output random feature
        matrix will become sparse matrix.

    p_sparse : float
        Sparsity parameter for "sparse_rademacher" distribution.
        If p_sparse = 0, "sparse_rademacher" is equivalent to "rademacher".

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [1] Random Feature Maps for the Itemset Kernel.
    Kyohei Atarashi, Subhransu Maji, and Satoshi Oyama
    In AAAI 2019.
    (https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/73469/1/aaai19_3875_camera_ready.pdf)
    """
    def __init__(self, n_components=100, kernel='anova', degree=2,
                 distribution='rademacher', dense_output=True, p_sparse=0.,
                 random_state=None):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.distribution = distribution
        self.dense_output = dense_output
        self.p_sparse = p_sparse
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        size = (self.n_components, n_features)
        distribution = self.distribution.lower()
        self.random_weights_ = get_random_matrix(random_state, distribution,
                                                 size, self.p_sparse)

        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if isinstance(self.kernel, str):
            if self.kernel == 'anova':
                kernel_ = _anova(self.degree, self.dense_output)
            elif self.kernel == 'anova_cython':
                kernel_ = _anova_fast(self.degree, self.dense_output)
            elif self.kernel == 'all_subsets':
                kernel_ = all_subsets
            elif self.kernel == 'dot':
                kernel_ = dot()
            else:
                raise ValueError('Kernel {} is not supported. '
                                 'Use "anova", "all_subsets" or "dot"'
                                 .format(self.kernel))
        else:
            kernel_ = self.kernel
        output = kernel_(X, self.random_weights_).astype(np.float64)

        output /= sqrt(self.n_components)
        return output


class RandomSubsetKernel(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, n_sub_features=5, kernel='anova',
                 degree=2, distribution='rademacher', dense_output=False,
                 random_state=None):
        self.n_components = n_components
        self.n_sub_features = n_sub_features
        self.degree = degree
        self.kernel = kernel
        self.distribution = distribution
        self.random_state = random_state
        self.dense_output=dense_output

    def fit(self, X, y=None):
        if self.kernel not in ['anova', 'anova_cython']:
            raise ValueError("RandomSubsetKernel now does not support"
                             " {} kernel.".format(self.kernel))

        if self.n_sub_features < self.degree:
            raise ValueError("n_sub_features < degree.")

        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        size = (self.n_sub_features * self.n_components, )
        distribution = self.distribution.lower()
        data = get_random_matrix(random_state, distribution, size=size)
        row = np.repeat(np.arange(self.n_components), self.n_sub_features)
        col = get_feature_indices(random_state, self.n_sub_features,
                                  n_features, self.n_components)
        self.random_weights_ = csc_matrix((data, (row, col)),
                                          shape=(self.n_components, n_features))
        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        const = np.arange(n_features, n_features-self.degree, -1)
        denominator = np.arange(self.n_sub_features,
                                self.n_sub_features-self.degree,
                                -1)
        const = np.prod(np.sqrt(const / denominator))
        if isinstance(self.kernel, str):
            if self.kernel == 'anova':
                kernel_ = _anova(self.degree, self.dense_output)
            elif self.kernel == 'anova_cython':
                kernel_ = _anova_fast(self.degree, self.dense_output)
            else:
                raise ValueError('Kernel {} is not supported. '
                                 'Use "anova" or "dot"'
                                 .format(self.kernel))
        else:
            kernel_ = self.kernel
        output = kernel_(X, self.random_weights_).astype(np.float64)

        output /= sqrt(self.n_components)
        output *= const
        return output
