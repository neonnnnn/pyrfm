# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix
from sklearn.utils.validation import check_is_fitted


def _index_hash(n_inputs, n_outputs, rng):
    return rng.randint(n_outputs, size=(n_inputs), dtype=np.int32)


def _sign_hash(n_inputs, rng):
    return 2*rng.randint(2, size=(n_inputs), dtype=np.int32) - 1

def _make_projection_matrices(hash_indices, hash_signs, n_components):
    n_features = hash_indices.shape[0]
    col = np.arange(n_features)
    random_weights = csc_matrix((hash_signs, (hash_indices, col)),
                                shape=(n_components, n_features))

    return random_weights

class CountSketch(BaseEstimator, TransformerMixin):
    """Approximates feature map of a linear kernel by Count Sketch, 
    a.k.a feature hashing.
    
    Parameters
    ----------
    n_components : int (default=100)
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    dense_output : bool (default=True)
        If dense_output = False and X is sparse matrix, output random
        feature matrix will become sparse matrix.

    Attributes
    ----------
    hash_indices_ : np.array, shape (n_features)
        Hash table that represents the embedded indices.
    
    hash_signs_ : np.array, shape (n_features)
        Sign array.

    random_weights_ : np.array, shape (n_features)
        Projection matrix created by hash_indices_ and hash_signs_.

    References
    ----------
    [1] Finding Frequent Items in Data Streams.
    Moses Charikar, Kevin Chen, and Martin Farach-Colton.
    In ICALP 2002.
    (https://www.cs.rutgers.edu/~farach/pubs/FrequentStream.pdf)

    [2] Fast and scalable polynomial kernels via explicit feature maps.
    Ninh Pham and Rasmus Pagh.
    In KDD 2013.
    (http://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)

    [3] Feature Hashing for Large Scale Multitask Learning.
    Kilian Weinberger, Anirban Dasgupta ANIRBAN, John Langford, Alex Smola,
    and Josh Attenberg.
    In ICML 2009.
    (http://alex.smola.org/papers/2009/Weinbergeretal09.pdf)

    """
    def __init__(self, n_components=100, degree=2, random_state=None,
                 dense_output=False):
        self.n_components = n_components
        self.degree = degree
        self.random_state = random_state
        self.dense_output = dense_output

    def fit(self, X, y=None):
        """Generate hash functions according to n_features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        self.hash_indices_ = _index_hash(n_features, self.n_components, 
                                         random_state)
        self.hash_signs_ = _sign_hash(n_features, random_state)
        self.random_weights_ = _make_projection_matrices(self.hash_indices_,
                                                         self.hash_signs_,
                                                         self.n_components)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        return safe_sparse_dot(X, self.random_weights_.T, self.dense_output)