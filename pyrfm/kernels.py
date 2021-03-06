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
                           dense_output=dense_output)


def anova(X, P, degree, dense_output=True):
    """Compute ANOVA kernel by pure numpy.

    .. math::

        k(x, y) = \\sum_{j_1 < \cdots < j_m} x_{j_1}p_{j_1}\\cdots x_{j_m}p_{j_m}

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    degree : int
        Degree of the ANOVA kernel (m in above equation).

    dense_output : bool (default=True)
        Whether to output np.ndarray or not (csr_matrix).

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """
    X = check_array(X, True)
    P = check_array(P, True)

    if degree == 2:
        H2 = safe_power(safe_sparse_dot(X, P.T, dense_output=dense_output), 
                        2, dense_output)
        D2 = D(X, P, 2, dense_output)
        A = (H2-D2) / 2.
    elif degree == 3:
        dot = safe_sparse_dot(X, P.T, dense_output=dense_output)
        A = safe_power(dot, 3, dense_output)
        A -= 3. * safe_np_elem_prod(D(X, P, 2, dense_output), dot,
                                    dense_output)
        A += 2. * D(X, P, 3, dense_output)
        A /= 6.
    else:
        n1 = X.shape[0]
        n2 = P.shape[0]
        Ds = [safe_sparse_dot(X, P.T, dense_output=dense_output)]
        Ds += [D(X, P, t, dense_output) for t in range(2, degree+1)]
        anovas = [1., Ds[0]]
        for m in range(2, degree+1):
            A = safe_np_elem_prod(anovas[m-1], Ds[0], dense_output)
            sign = -1.
            for t in range(2, m+1):
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
    """Compute pairwise kernel.

    .. math::

        k((x, a), (y, b)) = \sum_{t=1}^{m-1} \mathrm{ANOVA}^t(x, y)\mathrm{ANOVA}^{t-m}}a, b)

    Now only degree (m) = 2 supported.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    dense_output : bool (default=True)
        Whether to output np.ndarray or not (csr_matrix).

    symmetric : bool (default=False)
        Whether to symmetrize or not.

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """
    if X.shape[1] % 2 != 0:
        raise ValueError('X.shape[1] is not even.')

    n_features = X.shape[1]//2

    K1 = safe_sparse_dot(X[:, :n_features], P[:, :n_features], 
                         dense_output=dense_output)
    K2 = safe_sparse_dot(X[:, n_features:], P[:, n_features:], 
                         dense_output=dense_output)
    K = safe_np_elem_prod(K1, K2, dense_output=dense_output)
    if symmetric:
        K1 = safe_sparse_dot(X[:, :n_features], P[:, n_features:], 
                             dense_output=dense_output)
        K2 = safe_sparse_dot(X[:, n_features:], P[:, :n_features], 
                             dense_output=dense_output)
        K += safe_np_elem_prod(K1, K2, dense_output=dense_output)
        K *= 0.5
    return K


def hellinger(X, P):
    """Compute hellinger kernel.

    .. math:: 
    
        k(x, y) = \sum_{j=1}^d \sqrt{x_j}\sqrt{y_j}

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """

    X = check_array(X, True)
    P = check_array(P, True)
    return safe_sparse_dot(np.sqrt(X), np.sqrt(P))


def all_subsets(X, P, dense_output=True):
    """Compute all-subsets kernel.

    .. math::

        k(x, y) = \prod_{j=1}^d (1+x_jy_j)

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """

    X = check_array(X, True)
    P = check_array(P, True)
    return _all_subsets(X, P, dense_output)


def anova_fast(X, P, degree, dense_output=True):
    """Compute ANOVA kernel by Cython implementation.

    .. math::

        k(x, y) = \sum_{j_1 < \cdots < j_m} x_{j_1}p_{j_1}\cdots x_{j_m}p_{j_m}

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    degree : int
        Degree of the ANOVA kernel (m in above equation).

    dense_output : bool (default=True)
        Whether to output np.ndarray or not (csr_matrix).

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """

    X = check_array(X, True)
    P = check_array(P, True)
    return _anova(X, P, degree, dense_output)


def intersection(X, P):
    """Compute intersection kernel.

    .. math::

        k(x, y) = \sum_{j=1}^{d} \min (x_j, y_j)

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """
    X = check_array(X, True)
    P = check_array(P, True)
    return _intersection(X, P)


def chi_square(X, P):
    """Compute chi squared kernel.

    .. math::

        k(x,y) = \sum_{i=1}^{n}2x_iy_i/(x_i + y_i)

    Parameters
    ----------
    X : {array-like, sparse matrix} shape (n_samples1, n_features)
        Feature matrix.

    P : {array-like, sparse matrix} shape (n_samples2, n_features)
        Feature matrix.

    Returns
    -------
    gram_matrix : array-like, shape (n_samples1, n_samples2)

    """
    X = check_array(X, True)
    P = check_array(P, True)
    return _chi_square(X, P)


def kernel_alignment(K, y, scaling=True):
    """Compute kernel alignment.

    Parameters
    ----------
    K : array, shape (n_sample, n_samples)
        Gram matrix.

    y : array, shape (n_samples, )
        Label.
    
    scaling : bool (default=True)
        Whether to scale or not.
        If True, result is divided by \sqrt{KK}*n_samples

    Returns
    -------
    score : double

    """
    score = np.dot(y, np.dot(K, y))
    if scaling:
        score /= (np.sqrt(np.sum(K**2))*len(y))
    return score
