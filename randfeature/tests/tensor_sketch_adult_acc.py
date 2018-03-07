
from sklearn.datasets import load_svmlight_file
from randfeature import TensorSketch
import numpy as np
from sklearn.svm import LinearSVC, SVC


if __name__ == '__main__':
    X_train, y_train = load_svmlight_file('a9a', 123)
    X_test, y_test = load_svmlight_file('a9a.t', 123)
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Linear model Accuracy:{}'.format(clf.score(X_test, y_test)))
    clf = SVC(kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    print('Poly model Accuracy:{}'.format(clf.score(X_test, y_test)))

    for D in [100, 200, 300, 400, 500, 1000]:
        ts = TensorSketch(D, 2)
        print('compute tensor sketcing...')
        ts.fit(X_test)
        X_train_ts = ts.transform(X_train)
        X_test_ts = ts.transform(X_test)
        print('done.')
        print('fit LinearSVC...')
        clf = LinearSVC()
        clf.fit(X_train_ts, y_train)
        print('done.')
        test_acc = clf.score(X_test_ts, y_test)
        print('D:{}, Accuracy:{}'.format(D, test_acc))

