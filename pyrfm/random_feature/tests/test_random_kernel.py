import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import assert_less_equal
from pyrfm import anova, all_subsets, RandomKernel


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)


def test_anova_kernel():
    # compute exact kernel
    distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                     'sparse_rademacher']
    for degree in range(2, 5):
        kernel = anova(X, Y, degree)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomKernel(n_components=1000, random_state=rng,
                                        kernel='anova', degree=degree,
                                        distribution=dist, p_sparse=0.5)

            X_trans = rk_transform.fit_transform(X)
            Y_trans = rk_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.0001)
            assert_less_equal(np.max(error), 0.001)  # nothing too far off
            assert_less_equal(np.mean(error), 0.0005)  # mean is fairly close


def test_all_subsets_kernel():
    # compute exact kernel
    distributions = ['rademacher', 'gaussian', 'laplace', 'uniform',
                     'sparse_rademacher']
    kernel = all_subsets(X, Y)
    for dist in distributions:
        # approximate kernel mapping
        rk_transform = RandomKernel(n_components=2000, random_state=rng,
                                    kernel='all_subsets',
                                    distribution=dist, p_sparse=0.5)
        X_trans = rk_transform.fit_transform(X)
        Y_trans = rk_transform.transform(Y)
        kernel_approx = np.dot(X_trans, Y_trans.T)

        error = kernel - kernel_approx
        assert_less_equal(np.abs(np.mean(error)), 0.01)
        assert_less_equal(np.max(error), 0.1)  # nothing too far off
        assert_less_equal(np.mean(error), 0.05)  # mean is fairly close