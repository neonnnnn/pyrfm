from load_mnist import load_data
from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernel, anova
import numpy as np
from sklearn.svm import LinearSVC, SVC
import timeit
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from random_kernel_gaussian import RandomKernelGaussian


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
    X_train, y_train = X_train[:20000], y_train[:20000]
    """
    s = timeit.default_timer()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    e = timeit.default_timer()
    print('Linear model Accuracy:{}, Time:{}'.format(acc, e-s))

    s = timeit.default_timer()
    clf = SVC(kernel=_anova(2))
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    e = timeit.default_timer()
    print('Kernel SVM Accuracy:{}, Time:{}'.format(acc, e-s))
    """
    for D in [784, 784*2, 784*3, 784*4, 784*5]:
        time = 0.
        test_acc = 0.
        for i in range(5):
            print('compute random kernel gaussian map...')
            s = timeit.default_timer()
            rk = RandomKernelGaussian(D)
            rk.fit(X_test)
            X_train_rk = rk.transform(X_train)
            X_test_rk = rk.transform(X_test)
            print('fit LinearSVC...')
            clf = LinearSVC()
            clf.fit(X_train_rk, y_train)
            test_acc += clf.score(X_test_rk, y_test)
            time += timeit.default_timer() - s
        print('D:{}, Accuracy:{}, Time:{}'.format(D, test_acc/5., time/5))


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
