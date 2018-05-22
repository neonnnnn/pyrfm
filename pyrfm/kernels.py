import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import safe_sparse_dot


def safe_power(X, degree):
    if issparse(X):
        return X.power(degree)
    else:
        return X ** degree


def D(X, P, degree):
    return safe_sparse_dot(safe_power(X, degree),
                           safe_power(P, degree).T,
                           True)


def anova(X, P, degree):
    if degree == 2:
        anova = (safe_sparse_dot(X, P.T, True)**2 - D(X, P, 2))
        return anova / 2.
    elif degree == 3:
        dot = safe_sparse_dot(X, P.T, True)
        anova = dot**3
        anova -= 3. * D(X, P, 2)*dot
        anova += 2. * D(X, P, 3)
        return anova / 6.
    else:
        n1 = X.shape[0]
        n2 = P.shape[0]
        Ds = [safe_sparse_dot(X, P.T, True)]
        Ds += [D(X, P, t) for t in range(2, degree+1)]
        anovas = [1., Ds[0]]
        for m in range(2, degree+1):
            anova = np.zeros((n1, n2))
            sign = 1.
            for t in range(1, m+1):
                anova += sign * anovas[m-t] * Ds[t-1]
                sign *= -1.
            anova /= (1.0*m)
            anovas.append(anova)
        return anovas[-1]
