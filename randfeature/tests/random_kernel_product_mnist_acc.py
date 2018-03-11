from load_mnist import load_data
from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernelProduct, anova
import numpy as np
from sklearn.svm import LinearSVC, SVC


def _anova(degree):
    def __anova(X, P):
        return anova(X, P, degree)
    return __anova


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train[:20000], y_train[:20000]
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Linear model Accuracy:{}'.format(clf.score(X_test, y_test)))
    clf = SVC(kernel=_anova(2))
    clf.fit(X_train, y_train)
    print('Poly model Accuracy:{}'.format(clf.score(X_test, y_test)))

    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000, 3000]:
        rkp = RandomKernelProduct(D)
        print('compute random kernel product...')
        rkp.fit(X_test)
        X_train_rkp = rkp.transform(X_train)
        X_test_rkp = rkp.transform(X_test)
        print('fit LinearSVC...')
        clf = LinearSVC()
        clf.fit(X_train_rkp, y_train)
        test_acc = clf.score(X_test_rkp, y_test)
        print('D:{}, Accuracy:{}'.format(D, test_acc))

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
