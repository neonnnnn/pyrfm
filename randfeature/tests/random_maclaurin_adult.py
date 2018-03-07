from sklearn.datasets import load_svmlight_file
from randfeature import RandomMaclaurin
import numpy as np


if __name__ == '__main__':
    X_train, y_train = load_svmlight_file('a9a', 123)
    X_test, y_test = load_svmlight_file('a9a.t', 123)
    X_test = X_test
    print('compute gram matrix...')
    gram = (X_test*X_test.T).power(2)
    nnz = gram.nonzero()
    gram = gram[nnz[0], nnz[1]]
    print('done.')
    for D in [500, 1000, 1500, 2000, 2500, 3000]:
        ts = RandomMaclaurin(D, 10)
        print('compute rancom maclaurin...')
        ts.fit(X_test)
        X_test_ts = ts.transform(X_test)
        print('done.')
        gram_ts = np.dot(X_test_ts, X_test_ts.T)
        relative_error = np.mean(np.abs(1 - gram_ts[nnz[0], nnz[1]]/gram))
        print('D:{}, Relative_Error:{}'.format(D, relative_error))


