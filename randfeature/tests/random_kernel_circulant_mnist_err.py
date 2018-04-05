from load_mnist import load_data
from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernelSignedCirculant, anova
import numpy as np
from sklearn.svm import LinearSVC, SVC
import timeit
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse


def safe_power(X, degree):
    if issparse(X):
        return X.power(degree)
    else:
        return X ** degree


def _anova(degree):
    def __anova(X, P):
        """
        if degree == 2:
            ret = safe_sparse_dot(X, P.T, True)**2
            ret -= safe_sparse_dot(safe_power(X, 2), safe_power(P, 2).T, True)

            return 0.5 * ret
        else:
        """
        return anova(X, P, degree)
    return __anova


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train[:10000], y_train[:10000]

    gram = anova(X_train, X_train, 2)
    nnz = np.where(gram != 0.)
    d = X_train.shape[1]
    for t in [1, 2, 3, 4]:
        print('compute random signed circulant kernel map...')
        abs_err = 0
        rel_err = 0
        time = 0
        for i in range(5):
            s = timeit.default_timer()
            rk = RandomKernelSignedCirculant(t, random_state=i)
            rk.fit(X_train)
            X_train_rk = rk.transform(X_train)
            e = timeit.default_timer()
            time += e - s
            gram_rk = np.dot(X_train_rk, X_train_rk.T)
            abs_err += np.mean(np.abs(gram[nnz[0], nnz[1]] - gram_rk[nnz[0], nnz[1]]))
            rel_err += np.mean(np.abs(1-gram_rk[nnz[0], nnz[1]]/gram[nnz[0], nnz[1]]))
        abs_err /= 5.
        rel_err /= 5.
        time /= 5.
        print('D:{}, Absolute Err:{}, Relative Err:{}, Time:{}'
              .format(t*d, abs_err, rel_err, time))


"""
"""
