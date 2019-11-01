import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.kernel_approximation import SkewedChi2Sampler
from pyrfm.random_feature.random_features_fast import transform_all_fast

# generate data
rng = np.random.RandomState(0)
n_samples = 300
n_features = 50
X = rng.random_sample(size=(n_samples, n_features))
Y = rng.random_sample(size=(n_samples, n_features))


def test_additivechi2sampler():
    # approximate kernel mapping
    transformer = SkewedChi2Sampler()
    X_trans = transformer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans, transform_all_fast(X, transformer))