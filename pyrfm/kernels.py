import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import safe_sparse_dot
from .kernels_fast import _anova, _all_subsets, _intersection, _chi_square
from sklearn.utils import check_array


def safe_power(X, degree, dense_output=False):
    if issparse(X):
        ret = X.power(degree)
        if dense_output and issparse(ret):
            return ret.toarray()
        else:
            return ret
    else:
        return X ** degree


def safe_np_elem_prod(X, Y, dense_output=False):
    if not (issparse(X) or issparse(Y)):
        return X*Y
    else:
        if issparse(X):
            ret = X.multiply(Y)
        else:
            ret = Y.multiply(X)

        if dense_output:
            return ret.toarray()
        else:
            return ret


def D(X, P, degree, dense_output=True):
    return safe_sparse_dot(safe_power(X, degree), safe_power(P, degree).T,
                           dense_output)


def anova(X, P, degree, dense_output=True):
    X = check_array(X, True)
    P = check_array(P, True)

    if degree == 2:
        H2 = safe_power(safe_sparse_dot(X, P.T, dense_output), 2, dense_output)
        D2 = D(X, P, 2, dense_output)
        A = (H2-D2) / 2.
    elif degree == 3:
        dot = safe_sparse_dot(X, P.T, dense_output)
        A = safe_power(dot, 3, dense_output)
        A -= 3. * safe_np_elem_prod(D(X, P, 2, dense_output), dot,
                                    dense_output)
        A += 2. * D(X, P, 3, dense_output)
        A /= 6.
    else:
        n1 = X.shape[0]
        n2 = P.shape[0]
        Ds = [safe_sparse_dot(X, P.T, dense_output)]
        Ds += [D(X, P, t, dense_output) for t in range(2, degree+1)]
        anovas = [1., Ds[0]]
        for m in range(2, degree+1):
            A = np.zeros((n1, n2))
            sign = 1.
            for t in range(1, m+1):
                A += sign * safe_np_elem_prod(anovas[m-t], Ds[t-1],
                                              dense_output)
                sign *= -1.
            A /= (1.0*m)
            anovas.append(A)
        A = anovas[-1]

    if issparse(A) and dense_output:
        A = A.toarray()
    return A


def pairwise(X, P, dense_output=True, symmetric=False):
    if X.shape[1] % 2 != 0:
        raise ValueError('X.shape[1] is not even.')

    n_features = X.shape[1]//2

    K1 = safe_sparse_dot(X[:, :n_features], P[:, :n_features], dense_output)
    K2 = safe_sparse_dot(X[:, n_features:], P[:, n_features:], dense_output)
    K = safe_np_elem_prod(K1, K2, dense_output)
    if symmetric:
        K1 = safe_sparse_dot(X[:, :n_features], P[:, n_features:], dense_output)
        K2 = safe_sparse_dot(X[:, n_features:], P[:, :n_features], dense_output)
        K += safe_np_elem_prod(K1, K2, dense_output)
        K *= 0.5
    return K


def hellinger(X, P):
    X = check_array(X, True)
    P = check_array(P, True)
    return safe_sparse_dot(np.sqrt(X), np.sqrt(P))


def all_subsets(X, P):
    X = check_array(X, True)
    P = check_array(P, True)
    return _all_subsets(X, P)


def anova_fast(X, P, degree, dense_output=True):
    X = check_array(X, True)
    P = check_array(P, True)
    return _anova(X, P, degree, dense_output)


def intersection(X, P):
    X = check_array(X, True)
    P = check_array(P, True)
    return _intersection(X, P)


def chi_square(X, P):
    X = check_array(X, True)
    P = check_array(P, True)
    return _chi_square(X, P)

