from .load_mnist import load_data
from pyrfm import SignedCirculantRandomKernel, anova
from sklearn.svm import LinearSVC, SVC
import timeit
from scipy.sparse import issparse


def safe_power(X, degree):
    if issparse(X):
        return X.power(degree)
    else:
        return X ** degree


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train[:20000], y_train[:20000]
    d = X_train.shape[1]
    for t in [1, 2, 3, 4, 5]:
        test_acc = 0.
        time = 0.
        for i in range(5):
            s = timeit.default_timer()
            rk = RandomKernelSignedCirculant(t, random_state=i)
            rk.fit(X_test)
            X_train_rk = rk.transform(X_train)
            X_test_rk = rk.transform(X_test)
            clf = LinearSVC(dual=False)
            clf.fit(X_train_rk, y_train)
            test_acc += clf.score(X_test_rk, y_test)
            e = timeit.default_timer()
            time += e-s
        print('D:{}, Accuracy:{}, Time:{}'.format(t*d, test_acc/5, time/5))

