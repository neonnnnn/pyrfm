from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernel, anova
import numpy as np
from load_a9a import load_data
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_test = normalize(X_test[:5000])
    print('compute gram matrix...')
    gram = anova(X_test, X_test, 2)
    nnz = np.where(np.abs(gram) > 1e-10)
    gram = gram[nnz[0], nnz[1]]
    print('done.')
    for D in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
        rk = RandomKernel(D)
        print('compute random kernel product...')
        rk.fit(X_test)
        X_test_rk = rk.transform(X_test)
        gram_rk = np.dot(X_test_rk, X_test_rk.T)
        print(np.max(gram_rk))
        relative_error = np.mean(np.abs(1 - gram_rk[nnz[0], nnz[1]]/gram))
        print('D:{}, Relative_Error:{}'.format(D, relative_error))
