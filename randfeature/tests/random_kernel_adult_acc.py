from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernelProduct, anova
import numpy as np
from sklearn.svm import LinearSVC, SVC
from load_a9a import load_data


def _anova(degree=2):
    def __anova(X, Y):
        return anova(X, Y, degree)
    return __anova


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    clf = LinearSVC()

    clf.fit(X_train, y_train)
    print('Linear model Accuracy:{}'.format(clf.score(X_test, y_test)))
    clf = SVC(kernel=_anova(degree), degree=2)
    clf.fit(X_train, y_train)
    print('ANOVA kernel model Accuracy:{}'.format(clf.score(X_test, y_test)))

    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
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

