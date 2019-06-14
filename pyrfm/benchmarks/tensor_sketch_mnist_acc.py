from .load_mnist import load_data
from pyrfm import TensorSketch
from sklearn.svm import LinearSVC, SVC


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    print('Linear model Accuracy:{}'.format(clf.score(X_test, y_test)))
    clf = SVC(kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    print('Poly model Accuracy:{}'.format(clf.score(X_test, y_test)))

    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
        ts = TensorSketch(D, 2)
        print('compute tensor sketcing...')
        ts.fit(X_test)
        X_train_ts = ts.transform(X_train)
        X_test_ts = ts.transform(X_test)
        print('fit LinearSVC...')
        clf = LinearSVC(dual=False)
        clf.fit(X_train_ts, y_train)
        test_acc = clf.score(X_test_ts, y_test)
        print('D:{}, Accuracy:{}'.format(D, test_acc))

