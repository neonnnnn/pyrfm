from sklearn.datasets import load_svmlight_file
from randfeature import RandomKernelProduct, anova
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
        rkp = RandomKernelProduct(D)
        print('compute random kernel product...')
        rkp.fit(X_test)
        X_test_rkp = rkp.transform(X_test)
        gram_rkp = np.dot(X_test_rkp, X_test_rkp.T)
        print(np.max(gram_rkp))
        relative_error = np.mean(np.abs(1 - gram_rkp[nnz[0], nnz[1]]/gram))
        print('D:{}, Relative_Error:{}'.format(D, relative_error))
