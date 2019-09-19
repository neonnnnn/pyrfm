import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from .utils import next_pow_of_two, get_random_matrix


def _get_random_matrix(distribution):
    return lambda rng, size: get_random_matrix(rng, distribution, size)


class SubsampledRandomHadamard(BaseEstimator, TransformerMixin):
    """Approximates feature maps of the product between random matrix and
    feature vectors by Subsampled Randomized Hadamard Transform that uses
    diagonal matrices, the Walsh-Hadamard matrix, and submatrix of the
    identity matrix.
    This class can be used as a sub-routine for approximating the product
    between random matrix and feature vectors in some random features.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    distribution : str or function (default="rademacher")
        A function for sampling random basis whose arguments
        are random_state and size.
        Its arguments must be random_state and size.
        If None, the standard rademacher distribution is used.

    Attributes
    ----------
    random_weights_ : array, shape (n_features)
        The sampled basis.
        It is sampled by using self.distribution, which is the rademacher
        distribution default.

    random_indices_rows_ : array, shape (n_components)
        The indices of rows sampled from [0, \ldots, n_features_padded-1]
        uniformly, where n_features_padded is the smallest power of two number
        that is bigger than n_features.

    References
    ----------
    [1] Improved Analysis of the Subsampled Randomized Hadamard Transform.
    Joel A Tropp.
    Advances in Adaptive Data Analysis,
    (https://arxiv.org/pdf/1011.1595.pdf)
    """

    def __init__(self, n_components=100,  gamma=0.5, distribution="rademacher",
                 random_state=None):
        self.n_components = n_components
        self.distribution = distribution
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        if isinstance(self.distribution, str):
            self.distribution = _get_random_matrix(self.distribution)

        n_features_padded = next_pow_of_two(n_features)
        if n_features_padded < self.n_components:
            raise ValueError("n_components is bigger than next power of two"
                             "of n_features.")
        self.random_weights_ = self.distribution(random_state, n_features)
        self.random_weights_ = self.random_weights_.astype(np.float64)
        perm = random_state.permutation(n_features_padded).astype(np.int32)
        self.random_indices_rows_ = perm[:self.n_components]

        return self

    def transform(self, X):
        from .random_features_fast import transform_all_fast
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        return transform_all_fast(X, self)
