from .load_mnist import load_data
from randfeature import RandomKernel, anova
from sklearn.svm import LinearSVC, SVC
import timeit


def _anova(degree):
    def __anova(X, P):
        return anova(X, P, degree)
    return __anova


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train[:20000], y_train[:20000]

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

    for D in [784, 784*2, 784*3, 784*4, 784*5]:
        time = 0.
        test_acc = 0.
        for i in range(5):
            s = timeit.default_timer()
            rk = RandomKernel(D)
            rk.fit(X_test)
            X_train_rk = rk.transform(X_train)
            X_test_rk = rk.transform(X_test)
            clf = LinearSVC(dual=False)
            clf.fit(X_train_rk, y_train)
            test_acc += clf.score(X_test_rk, y_test)
            time += timeit.default_timer() - s
        print('D:{}, Accuracy:{}, Time:{}'.format(D, test_acc/5., time/5))
