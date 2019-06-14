from .load_mnist import load_data
from pyrfm import SignedCirculantRandomKernel, anova
import numpy as np
import timeit
from scipy.sparse import issparse


def safe_power(X, degree):
    if issparse(X):
        return X.power(degree)
    else:
        return X ** degree


def _anova(degree):
    def __anova(X, P):
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
