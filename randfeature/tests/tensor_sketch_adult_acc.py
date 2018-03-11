from sklearn.datasers import load_svmlight_file
from randfeature import TensorSketch
import numpy as np
from sklearn.svm import LinearSVC, SVC
from load_a9a import load_data


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Linear model Accuracy:{}'.format(clf.score(X_test, y_test)))
    clf = SVC(kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    print('Poly model Accuracy:{}'.format(clf.score(X_test, y_test)))

    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
        rs = TensorSketch(D, 2)
        print('compute tensor sketcing...')
        rs.fit(X_test)
        X_train_rs = rs.transform(X_train)
        X_test_rs = rs.transform(X_test)
        print('fit LinearSVC...')
        clf = LinearSVC()
        clf.fit(X_train_rs, y_train)
        test_acc = clf.score(X_test_rs, y_test)
        print('D:{}, Accuracy:{}'.format(D, test_acc))

