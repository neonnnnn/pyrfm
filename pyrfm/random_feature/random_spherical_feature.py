import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot
from scipy.optimize import lbfgsb
from sklearn.utils.validation import check_is_fitted
from scipy.special import gamma, j0, j1, lambertw


class ShpericalRandomFeature(BaseEstimator, TransformerMixin):
    """Approximates feature map of a polynomial kernel by Monte Carlo
    approximation by using Random Spherical Feature map.

    Parameters
    ----------
    n_components : int
        Number of Monte Carlo samples per original features.
        Equals the dimensionality of the computed (mapped) feature space.

    degree : int
        Parameter of the polynomial product kernel.

    n_gaussians : int, default=10
        Number of gaussians.

    n_grids : int, default=500
        Number of grid for approximating the inverse Fourier transform of
        the sum of Gaussians

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_weights_ : array, shape (n_components, n_features) (use_offset=True)
                                   or (n_components/2, n_features) (otherwise)
        The sampled basis.

    References
    ----------
    [1] Fast and scalable polynomial kernels via explicit feature maps.
    Ninh Pham and Rasmus Pagh.
    In KDD 2013.
    (http://chbrown.github.io/kdd-2013-usb/kdd/p239.pdf)
    """
    def __init__(self, n_components=100, degree=2, n_gaussians=10,
                 n_grids=500, warm_start=False, random_state=None):
        self.n_components = n_components
        self.degree = degree
        self.n_gaussians = n_gaussians
        self.n_grids = n_grids
        self.warm_start = warm_start
        self.random_state = random_state

    def _find_cutoff(self, n_features, eps, sigma):
        x = -2 / (n_features - 1) * (eps * sigma)
        if n_features < 200:
            x *= gamma(n_features/2)
            x = x**(2/(n_features-1))
        else:
            x = x**(2/(n_features-1))
            x *= (4*np.pi)**(1./(n_features-1))
            coef = np.exp(-1)*(n_features/2 + 1/(6*n_features - n_features/5))
            x *= coef**(1/(1-1/n_features))
        return np.max(sigma * np.sqrt(-2*(n_features-1)*lambertw(x, -1)))

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)

        if not (self.warm_start and hasattr(self, 'coefs_')):
            self.coefs_ = np.zeros(self.n_components)

        if not (self.warm_start and hasattr(self, 'log_vars_')):
            self.log_vars_ = np.zeros(self.n_components)

        z = np.linspace(0, 2, self.n_grids)
        self.coefs_ = np.zeros(self.n_gaussians)

        self.log_sigmas_ = np.zeros(self.n_gaussians)

        def _approx_kernel_cost_grad(theta):
            coefs = theta[:self.n_gaussians]
            log_sigmas = theta[self.n_gaussians:]
            sigmas = np.exp(log_sigmas)
            w_max = self._find_cutoff(n_features, 1e-6/(abs(coefs)), sigmas)
            w = np.linspace(0, 2, self.n_grids)
            # grids
            samples = (1 / (2*sigmas))[:, np.newaxis] * w[np.newaxis, :]

            approx = 0
            grad = 0
            return approx, grad

        return self
