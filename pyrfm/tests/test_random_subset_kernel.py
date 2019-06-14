import numpy as np
from scipy.sparse import csr_matrix


from sklearn.utils.testing import assert_less_equal, assert_almost_equal
from sklearn.utils.extmath import safe_sparse_dot
from pyrfm import anova, RandomSubsetKernel, RandomKernel


# generate data
rng = np.random.RandomState(0)
X = rng.random_sample(size=(300, 50))
Y = rng.random_sample(size=(300, 50))
X /= np.sum(X, axis=1, keepdims=True)
Y /= np.sum(Y, axis=1, keepdims=True)

X_sp = rng.random_sample(size=(300, 2000))
Y_sp = rng.random_sample(size=(300, 2000))
X_sp[X_sp<0.99] = 0
Y_sp[Y_sp<0.99] = 0
X_sp /= np.sum(X_sp, axis=1, keepdims=True)
Y_sp /= np.sum(Y_sp, axis=1, keepdims=True)

X_sp = csr_matrix(X_sp)
Y_sp = csr_matrix(Y_sp)


def test_anova_kernel():
    # compute exact kernel
    distributions = ['rademacher']
    for degree in range(2, 5):
        kernel = anova(X, Y, degree)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomSubsetKernel(n_components=1000,
                                              random_state=rng,
                                              kernel='anova', degree=degree,
                                              distribution=dist,
                                              n_sub_features=25)
            X_trans = rk_transform.fit_transform(X)
            Y_trans = rk_transform.transform(Y)
            kernel_approx = np.dot(X_trans, Y_trans.T)

            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.001)
            assert_less_equal(np.max(error), 0.01)  # nothing too far off
            assert_less_equal(np.mean(error), 0.005)  # mean is fairly close


def test_anova_kernel_sparse():
    # compute exact kernel
    distributions = ['rademacher']
    for degree in range(2, 4):
        kernel = anova(X_sp, Y_sp, degree, True)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomKernel(n_components=2000*4,
                                        random_state=rng,
                                        kernel='anova', degree=degree,
                                        distribution=dist)
            X_trans = rk_transform.fit_transform(X_sp)
            Y_trans = rk_transform.transform(Y_sp)
            kernel_approx = safe_sparse_dot(X_trans, Y_trans.T, True)
            error = kernel - kernel_approx
            assert_less_equal(np.abs(np.mean(error)), 0.001)
            assert_less_equal(np.max(error), 0.01)  # nothing too far off
            assert_less_equal(np.mean(error), 0.005)  # mean is fairly close


def test_anova_kernel_sparse_sparse_rademacher():
    # compute exact kernel
    distributions = ['sparse_rademacher']
    for degree in range(2, 4):
        kernel = anova(X_sp, Y_sp, degree, True)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomKernel(n_components=2000*4,
                                        random_state=rng,
                                        kernel='anova', degree=degree,
                                        distribution=dist, p_sparse=0.99)
            X_trans = rk_transform.fit_transform(X_sp)
            Y_trans = rk_transform.transform(Y_sp)
            Y_trans_nnz = np.sum(Y_trans != 0)
            kernel_approx = safe_sparse_dot(X_trans, Y_trans.T, True)
            error = kernel - kernel_approx
            print('nnz_sparse_rade {}'.format(Y_trans_nnz/np.prod(Y_trans.shape)))
            print('abs(mean(error)) {}'.format(np.abs(np.mean(error))))
            print('mean(abs(error)) {}'.format(np.mean(np.abs(error))))
            print('max(abs(error)) {}'.format(np.max(np.abs(error))))
            assert_less_equal(np.abs(np.mean(error)), 0.001)
            assert_less_equal(np.max(error), 0.01)  # nothing too far off
            assert_less_equal(np.mean(error), 0.005)  # mean is fairly close


def test_anova_kernel_sparse_subset():
    # compute exact kernel
    distributions = ['rademacher']
    n_components = 2000*4
    n_sub_features = 20
    for degree in range(2, 4):
        kernel = anova(X_sp, Y_sp, degree, True)
        for dist in distributions:
            # approximate kernel mapping
            rk_transform = RandomSubsetKernel(n_components=n_components,
                                              random_state=rng,
                                              kernel='anova', degree=degree,
                                              distribution=dist,
                                              n_sub_features=n_sub_features)
            X_trans = rk_transform.fit_transform(X_sp)
            Y_trans = rk_transform.transform(Y_sp)
            assert_almost_equal(rk_transform.random_weights_.nnz,
                                n_components*n_sub_features)

            kernel_approx = safe_sparse_dot(X_trans, Y_trans.T, True)
            error = kernel - kernel_approx
            print('nnz_sparse_subset {}'.format(Y_trans.nnz/np.prod(Y_trans.shape)))
            print('abs(mean(error)) {}'.format(np.abs(np.mean(error))))
            print('mean(abs(error)) {}'.format(np.mean(np.abs(error))))
            print('max(abs(error)) {}'.format(np.max(np.abs(error))))

            assert_less_equal(np.abs(np.mean(error)), 0.001)
            assert_less_equal(np.max(error), 0.1)  # nothing too far off
            assert_less_equal(np.mean(error), 0.005)  # mean is fairly close
