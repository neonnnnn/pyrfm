import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_less_equal, assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import (CompactRandomFeature, RandomProjection,
                   SubsampledRandomHadamard,
                   RandomMaclaurin, RandomFourier, RandomKernel, TensorSketch)
from sklearn.kernel_approximation import RBFSampler


def polynomial(X, Y, degree, bias=0):
    return (safe_sparse_dot(X, Y.T, True)+bias)**degree


def exp_kernel(X, Y, gamma):
    return np.exp(safe_sparse_dot(X, Y.T) * gamma)


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)
X_sp = csr_matrix(X)


def test_compact_random_feature_random_maclaurin():
    # compute exact kernel
    for down in [RandomProjection, SubsampledRandomHadamard]:
        for degree in range(2, 5):
            # approximate kernel mapping
            transform_up = RandomMaclaurin(n_components=100, degree=degree,
                                           random_state=0, kernel='poly')
            transform_down = down(n_components=50, random_state=0)
            X_trans_naive = transform_down.fit_transform(
                transform_up.fit_transform(X)
            )

            transform_up = RandomMaclaurin(n_components=100, degree=degree,
                                           random_state=0, kernel='poly')
            transform_down = down(n_components=50, random_state=0)
            transformer = CompactRandomFeature(transformer_up=transform_up,
                                               degree=degree,
                                               transformer_down=transform_down)
            X_trans = transformer.fit_transform(X)
            assert_allclose(X_trans_naive, X_trans)


def test_compact_random_feature_random_fourier():
    for down in [RandomProjection, SubsampledRandomHadamard]:
        for gamma in [0.1, 1, 10]:
            # approximate kernel mapping
            transform_up = RandomFourier(n_components=100, gamma=gamma,
                                         random_state=0)
            transform_down = down(n_components=50, random_state=0)
            X_trans_naive = transform_down.fit_transform(
                transform_up.fit_transform(X)
            )

            transform_up = RandomFourier(n_components=100, gamma=gamma,
                                         random_state=0)
            transform_down = down(n_components=50, random_state=0)
            transformer = CompactRandomFeature(transformer_up=transform_up,
                                               transformer_down=transform_down)
            X_trans = transformer.fit_transform(X)
            assert_allclose(X_trans_naive, X_trans)


def test_compact_random_feature_tensor_sketch():
    for down in [RandomProjection, SubsampledRandomHadamard]:
        for degree in range(2, 5):
            # approximate kernel mapping
            transform_up = TensorSketch(degree=degree, random_state=0,
                                        n_components=100)
            transform_down = down(n_components=50, random_state=0)
            X_trans_naive = transform_down.fit_transform(
                transform_up.fit_transform(X)
            )

            transform_up = TensorSketch(degree=degree, random_state=0,
                                        n_components=100)
            transform_down = down(n_components=50, random_state=0)
            transformer = CompactRandomFeature(transformer_up=transform_up,
                                               transformer_down=transform_down)
            X_trans = transformer.fit_transform(X)
            assert_allclose(X_trans_naive, X_trans)


def test_compact_random_feature_random_kernel():
    for down in [RandomProjection, SubsampledRandomHadamard]:
        for degree in range(2, 5):
            # approximate kernel mapping
            transform_up = RandomKernel(degree=degree, random_state=0,
                                        n_components=100)
            transform_down = down(n_components=50, random_state=0)
            X_trans_naive = transform_down.fit_transform(
                transform_up.fit_transform(X)
            )

            transform_up = RandomKernel(degree=degree, random_state=0,
                                        n_components=100)
            transform_down = down(n_components=50, random_state=0)
            transformer = CompactRandomFeature(transformer_up=transform_up,
                                               transformer_down=transform_down)
            X_trans = transformer.fit_transform(X)
            assert_allclose(X_trans_naive, X_trans)


def test_compact_random_feature_random_kernel_all_subsets():
    for down in [RandomProjection, SubsampledRandomHadamard]:
        # approximate kernel mapping
        transform_up = RandomKernel(kernel='all_subsets', random_state=0,
                                    n_components=100)
        transform_down = down(n_components=50, random_state=0)
        X_trans_naive = transform_down.fit_transform(
            transform_up.fit_transform(X)
        )

        transform_up = RandomKernel(kernel='all_subsets', random_state=0,
                                    n_components=100)
        transform_down = down(n_components=50, random_state=0)
        transformer = CompactRandomFeature(transformer_up=transform_up,
                                           transformer_down=transform_down)
        X_trans = transformer.fit_transform(X)
        assert_allclose(X_trans_naive, X_trans)


def test_compact_random_feature_rbfsampler():
    for down in [RandomProjection, SubsampledRandomHadamard]:
        for gamma in [0.1, 1, 10]:
            # approximate kernel mapping
            transform_up = RBFSampler(gamma=gamma, random_state=0,
                                      n_components=100)
            transform_down = down(n_components=50, random_state=0)
            X_trans_naive = transform_down.fit_transform(
                transform_up.fit_transform(X)
            )
            transform_up = RBFSampler(gamma=gamma, random_state=0,
                                      n_components=100)
            transform_down = down(n_components=50, random_state=0)
            transformer = CompactRandomFeature(transformer_up=transform_up,
                                               transformer_down=transform_down)
            X_trans = transformer.fit_transform(X)
            assert_allclose(X_trans_naive, X_trans)
