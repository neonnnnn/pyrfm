# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix
from scipy.fftpack import fft, ifft
from sklearn.utils.validation import check_is_fitted


def _index_hash(n_inputs, n_outputs, degree, rng):
    """
    # h(j) = (a*j + b mod p) mod n_outputs,
    # where p is a prime number that is enough large (p >> n_outputs)
    p = 2**61 - 1
    a = rng.randint(p, size)
    b = rng.randint(p, size)
    return (((a * np.arange(n_outputs)) % p + b) % p) % n_outputs
    """
    return rng.randint(n_outputs, size=(degree, n_inputs), dtype=np.int32)


def _sign_hash(n_inputs, degree, rng):
    return 2*rng.randint(2, size=(degree, n_inputs), dtype=np.int32) - 1


def _make_projection_matrices(i_hash, s_hash, n_components):
    degree, d = i_hash.shape
    val = s_hash.ravel()
    row = i_hash.ravel()
    col = np.arange(d*degree)
    random_weights = csc_matrix((val, (row, col)),
                                shape=(n_components, d*degree))

    return random_weights


class TensorSketch(BaseEstimator, TransformerMixin):
    """Approximates feature map of a polynomial kernel by Monte Carlo
    approximation by using Tensor Sketch.

    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    degree : int (default=2)
        Parameter of the polynomial product kernel.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    hash_indices_ : array, shape (degree, n_features)
        Hash matrix for CountSketch.

    hash_signs_ : array, shape (degree, n_features)
        Sign matrix for CountSketch.

    random_weights_ : list of csc_matrix, len=degree
        The sampled basis created by hash_indices and hash_signs for
        convenience.

    References
    ----------
    [1] Fast and scalable polynomial kernels via explicit feature maps.
    Ninh Pham and Rasmus Pagh.
    In KDD 2013.
    (http://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)

    """
    def __init__(self, n_components=100, degree=2, random_state=None):
        self.n_components = n_components
        self.degree = degree
        self.random_state = random_state

    def fit(self, X, y=None):
        """Generate hash functions according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        i_hash = _index_hash(n_features, self.n_components, self.degree,
                             random_state)
        s_hash = _sign_hash(n_features, self.degree, random_state)
        self.hash_indices_ = i_hash.ravel()
        self.hash_signs_= s_hash.ravel()
        self.random_weights_ = _make_projection_matrices(i_hash, s_hash,
                                                         self.n_components)

        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        X = check_array(X, True)
        n_samples, n_features = X.shape
        P = safe_sparse_dot(X, self.random_weights_[:, :n_features].T, True)
        output = fft(P)
        for offset in range(n_features, n_features*self.degree, n_features):
            random_weight = self.random_weights_[:, offset:offset+n_features]
            P = safe_sparse_dot(X, random_weight.T, True)
            output *= fft(P)

        return ifft(output).real
