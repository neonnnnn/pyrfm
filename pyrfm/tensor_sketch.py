from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix
from scipy.fftpack import fft, ifft
from sklearn.utils.validation import check_is_fitted


def _index_hash(n_inputs, n_outputs, size, rng):
    """
    # h(j) = (a*j + b mod p) mod n_outputs,
    # where p is a prime number that is enough large (p >> n_outputs)
    p = 2**61 - 1
    a = rng.randint(p, size)
    b = rng.randint(p, size)
    return (((a * np.arange(n_outputs)) % p + b) % p) % n_outputs
    """
    return rng.randint(n_outputs, size=(size, n_inputs))


def _bit_hash(degree, d, rng):
    return 2*rng.randint(2, size=(degree, d)) - 1


def _make_projection_matrices(i_hash, b_hash, n_components):
    degree, d = i_hash.shape
    random_weights = []
    for pi in range(degree):
        random_weights.append(csc_matrix((b_hash[pi], (range(d), i_hash[pi])),
                                         shape=(d, n_components)))
    return random_weights


class TensorSketch(BaseEstimator, TransformerMixin):
    """Approximates feature map of a polynomial kernel by Monte Carlo
    approximation by using Tensor Sketch.

    Parameters
    ----------
    n_components : int
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    degree : int
        Parameter of the polynomial product kernel.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

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
        X = check_array(X, True)
        n_features = X.shape[1]

        random_state = check_random_state(self.random_state)
        i_hash = _index_hash(n_features, self.n_components, self.degree,
                             random_state)
        b_hash = _bit_hash(self.degree, n_features, random_state)
        self.random_weights_ = _make_projection_matrices(i_hash, b_hash,
                                                         self.n_components)

        return self

    def transform(self, X):
        check_is_fitted(self, "random_weights_")
        X = check_array(X, True)
        P = safe_sparse_dot(X, self.random_weights_[0], True)
        output = fft(P)
        for random_weight in self.random_weights_[1:]:
            P = safe_sparse_dot(X, random_weight, True)
            output *= fft(P)

        return ifft(output).real
