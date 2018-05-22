from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix
from scipy.fftpack import fft, ifft


def _index_hash(D, degree, d, rng):
    return rng.randint(D, size=(degree, d))


def _bit_hash(degree, d, rng):
    return 2*rng.randint(2, size=(degree, d)) - 1


def _make_projection_matrices(i_hash, b_hash, D):
    degree, d = i_hash.shape
    projs = []
    for pi in range(degree):
        projs.append(csc_matrix((b_hash[pi], (range(d), i_hash[pi])),
                                shape=(d, D)))
    return projs


class TensorSketch(BaseEstimator, TransformerMixin):
    def __init__(self, D=100, degree=2, random_state=1):
        self.D = D
        self.degree = degree
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        d = check_array(X, True).shape[1]
        i_hash = _index_hash(self.D, self.degree, d, random_state)
        b_hash = _bit_hash(self.degree, d, random_state)
        self.projs_ = _make_projection_matrices(i_hash, b_hash, self.D)

        return self

    def transform(self, raw_X):
        n, d = check_array(raw_X, True).shape
        P = safe_sparse_dot(raw_X, self.projs_[0], True)
        output = fft(P)
        for proj in self.projs_[1:]:
            P = safe_sparse_dot(raw_X, proj, True)
            output *= fft(P)

        return ifft(output).real
