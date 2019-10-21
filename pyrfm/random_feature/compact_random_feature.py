# Author: Kyohei Atarashi
# License: BSD-2-Clause

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted
from . import RandomMaclaurin


class CompactRandomFeature(BaseEstimator, TransformerMixin):
    """Approximates feature map of the RBF kernel by Monte Carlo
    approximation by Random Fourier Feature map.

    You can construct (the simplest) CompactRandomFeature by combining random
    features and sklearn.random_projection, e.g.,

        >>> trans_up = RandomMaclaurin(n_components=n_components_up)
        >>> trans_down = SparseRandomProjecion(n_components=n_components.
                                           density=1)
        >>> X_trans_up = trans_up.fit_transform(X)
        >>> X_trans_down = trans_down.fit_transform(X_trans_down)

    The advantages of this CompactRandomFeature is its memory efficiency.
    Above-mentioned combinatorial approach might occur the memory error in
    the up projection when the size of the original feature matrix is large.
    CompactRandomFeature for a random feature map with a cython implementation
    avoid the memory error because it does not compute all of up projection
    random features at the same time.
    Although you can avoid MemoryError by creating mini-batches of the training
    instances, this CompactRandomFeature class save this step.

    Parameters
    ----------
    transformer_up : sklearn transformer object (default=None)
        A random feature map object.
        If None, RandomMaclaurin is used.

    transformer_down: str or sklearn transformer object (default=None)
        Transformer for down projection.
        {"srht"} can be used.
        If None, structured projection cannot be used. Standard RandomProjection
        is used.

    n_components : int (default=10)
        Number of Monte Carlo samples per randomized features of transformer.
        It corresponds to E in original paper [1].
        Equals the dimensionality of the computed (mapped) feature space.

    n_components_up : int (default=100)
        Number of Monte Calro samples per original features.
        It corresponds to D in original paper [1].
        It is used when transformer = "random_maclaurin" or "tensor_sketch"

    degree : int (default=2)
        Degree of polynomial kernel.
        This argument is used when transformer = None.

    h01 : int (default=False)
        Using h01 heuristics or not.
        This argument is used when transformer = None.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If np.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    random_weights_ : array, shape (n_components, transformer.n_components)
    or (transformer.n_components)
        The sampled basis for down projection.

    References
    ----------
    [1] Compact Random Feature Maps.
    Raffay Hamid, Ying Xiao, Alex Gittens, and Dennis DeCoste.
    In ICML 2014.
    (http://proceedings.mlr.press/v32/hamid14.pdf)

    """

    def __init__(self, transformer_up=None, transformer_down=None,
                 n_components=10, n_components_up=100, degree=2, h01=False,
                 random_state=None):
        self.transformer_up = transformer_up
        self.transformer_down = transformer_down
        self.n_components = n_components
        self.n_components_up = n_components_up
        self.degree = degree
        self.h01 = h01
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the transformers for up projection and down projection.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the transformer.
        """
        random_state = check_random_state(self.random_state)
        X = check_array(X, accept_sparse=True)

        # fit up projection transformer
        if self.transformer_up is None:
            self.transformer_up = RandomMaclaurin(self.n_components_up,
                                                  degree=self.degree,
                                                  h01=self.h01,
                                                  random_state=random_state)
        self.transformer_up.fit(X)
        _, n_components_up = self.transformer_up.transform(X[:2]).shape

        # fit down projection transformer
        if not isinstance(self.transformer_down, TransformerMixin):
            from . import DOWNSAMPLES
            if self.transformer_down in DOWNSAMPLES.keys():
                self.transformer_down = DOWNSAMPLES[self.transformer_down](
                    n_components=self.n_components, random_state=random_state
                )
            else:
                raise ValueError("{0} not in {1}. Use {1} or transformer."
                                 .format(self.transformer_down,
                                         DOWNSAMPLES.keys()))
        self.transformer_down.fit(self.transformer_up.transform(X[:2]))
        self.n_components = self.transformer_down.n_components
        self.random_weights_ = self.transformer_down.random_weights_
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        from .random_features_fast import transform_all_fast
        check_is_fitted(self, "random_weights_")
        X = check_array(X, accept_sparse=True)
        Z = transform_all_fast(X, self)
        return Z
