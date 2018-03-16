from load_mnist import load_data
from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernel, anova
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
    for D in [500, 1000, 1500, 2000, 2500, 3000]:
        print('compute random kernel map...')
        abs_err = 0
        rel_err = 0
        time = 0
        for i in range(5):
            s = timeit.default_timer()
            rk = RandomKernel(D, random_state=i)
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
              .format(D, abs_err, rel_err, time))


"""
Linear model Accuracy:0.9041
Poly model Accuracy:0.9695
D:100, Accuracy:0.754
D:200, Accuracy:0.8631
D:300, Accuracy:0.8922
D:400, Accuracy:0.8929
D:500, Accuracy:0.9108
D:1000, Accuracy:0.9282
D:1500, Accuracy:0.9504
"""
