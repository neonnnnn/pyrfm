from pyrfm import RandomMaclaurin
import numpy as np
from .load_a9a import load_data


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    print('compute gram matrix...')
    gram = (X_test[:5000]*X_test[:5000].T).power(2)
    nnz = gram.nonzero()
    gram = gram[nnz[0], nnz[1]]
    print('done.')
    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
        ts = RandomMaclaurin(D, 10)
        print('compute random Maclaurin...')
        ts.fit(X_test)
        X_test_ts = ts.transform(X_test)
        gram_ts = np.dot(X_test_ts, X_test_ts.T)
        relative_error = np.mean(np.abs(1 - gram_ts[nnz[0], nnz[1]]/gram))
        print('D:{}, Relative_Error:{}'.format(D, relative_error))
