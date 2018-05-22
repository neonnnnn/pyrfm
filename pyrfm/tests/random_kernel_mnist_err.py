from .load_mnist import load_data
from randfeature import RandomKernel, anova
import numpy as np
import timeit


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train = X_train[:10000], y_train[:10000]

    gram = anova(X_train, X_train, 2)
    nnz = np.where(gram != 0.)
    for D in [1, 2, 3, 4, 5]:
        print('compute random kernel map...')
        abs_err = 0
        rel_err = 0
        time = 0
        for i in range(5):
            s = timeit.default_timer()
            rk = RandomKernel(D*784, random_state=i)
            rk.fit(X_train)
            X_train_rk = rk.transform(X_train)
            e = timeit.default_timer()
            time += e - s
            gram_rk = np.dot(X_train_rk, X_train_rk.T)
            abs_err += np.mean(np.abs(gram[nnz[0], nnz[1]] - gram_rk[nnz[0], nnz[1]]))
            rel_err += np.mean(np.abs(1-gram_rk[nnz[0], nnz[1]]/gram[nnz[0], nnz[1]]))
        abs_err /= 5.
        rel_err /= 5.
        time /= 5.
        print('D:{}, Absolute Err:{}, Relative Err:{}, Time:{}'
              .format(D, abs_err, rel_err, time))
