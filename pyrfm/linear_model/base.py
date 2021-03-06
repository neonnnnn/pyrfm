# Author: Kyohei Atarashi
# License: BSD-2-Clause

import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target
from ..random_feature.random_features_fast import get_fast_random_feature
from ..random_feature.random_features_doubly import get_doubly_random_feature
from ..random_feature import LearningKernelwithRandomFeature
from ..dataset_fast import get_dataset
from .utils import _predict_fast, _predict_fast_doubly
from scipy import sparse
from ..random_feature.maji_berg import MB
from ..random_feature import AdditiveChi2Sampler


def sigmoid(pred):
    return np.exp(np.minimum(0, pred)) / (1.+np.exp(-np.abs(pred)))


class BaseLinear(BaseEstimator, metaclass=ABCMeta):
    def _valid_params(self):
        if not self.C > 0:
            raise ValueError("C <= 0.")

        if not self.alpha >= 0:
            raise ValueError("alpha < 0.")

        if not self.tol >= 0:
            raise ValueError("tol <  0")

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported. Only {}"
                             "are supported.".format(self.loss,
                                                     self.LOSSES.key()))
        if not isinstance(self.max_iter, int):
            raise TypeError("max_iter is not int.")

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose is not bool.")

        if hasattr(self, "fast_solver"):
            if not isinstance(self.warm_start, bool):
                raise TypeError("fast_solver is not bool.")

        if hasattr(self, "warm_start"):
            if not isinstance(self.warm_start, bool):
                raise TypeError("warm_start is not bool.")

        if hasattr(self, "fit_intercept"):
            if not isinstance(self.fit_intercept, bool):
                raise TypeError("fit_intercept is not bool.")

        if hasattr(self, "normalize") and not isinstance(self.normalize, bool):
            raise TypeError("normalize is not bool.")

        if hasattr(self, "shuffle") and not isinstance(self.shuffle, bool):
            raise TypeError("shuffle is not bool.")

        if hasattr(self, "l1_ratio") and not (0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1].")

        if hasattr(self, "eps") and not (0 < self.eps):
            raise ValueError("eps <= 0.")

        if hasattr(self, "eta0") and not (0 < self.eta0):
            raise ValueError("eta0 <= 0.")

    # for stochastic solver
    def _init_params(self, X, y):
        if not (self.warm_start and self._check_transformer_is_fitted()):
            self.transformer.fit(X, y)
        
        n_components = self.transformer.n_components

        if not (self.warm_start and hasattr(self, 'coef_')):
            self.coef_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'intercept_')):
            self.intercept_ = np.zeros(1)

        if not (self.warm_start and hasattr(self, 't_')):
            self.t_ = 1

        if self.normalize:
            if not (self.warm_start and hasattr(self, 'mean_')):
                self.mean_ = np.zeros((n_components, ))

            if not (self.warm_start and hasattr(self, 'var_')):
                self.var_ = np.zeros((n_components,))
        else:
            self.mean_ = None
            self.var_ = None

    def _check_transformer_is_fitted(self):
        if not isinstance(self.transformer, TransformerMixin):
            raise ValueError("transformer is not an instance of TransformerMixin.")
        if isinstance(self.transformer, AdditiveChi2Sampler):
            if not hasattr(self.transformer, "sample_interval"):
                return False
        elif isinstance(self.transformer, MB):
            if not hasattr(self.transformer, "n_grids_"):
                return False
        elif not hasattr(self.transformer, "random_weights_"):
            return False
        if isinstance(self.transformer, LearningKernelwithRandomFeature):
            if not hasattr(self.transformer, "importance_weights_"):
                return False
            
        return True

    def _predict(self, X):
        check_is_fitted(self, 'coef_')
        coef = self.coef_
        if hasattr(self, 'intercept_'):
            intercept = self.intercept_
        else:
            intercept = 0.
    
        # if averaging, use averaged coef and intercept
        if hasattr(self, 'average'):
            if isinstance(self.average, bool):
                averaging = self.average
            else:
                averaging = self.average > 0
        
            if averaging:
                coef = self.coef_average_
                intercept = self.intercept_average_
    
        if getattr(self, 'stochastic', False):
            y_pred = np.zeros(X.shape[0])
            is_sparse = sparse.issparse(X)
    
            transformer_fast = get_fast_random_feature(self.transformer)
            if transformer_fast is None:
                for i, xi in enumerate(X):
                    if is_sparse:
                        xi_trans = self.transformer.transform(xi).ravel()
                    else:
                        xi_trans = self.transformer.transform(
                            np.atleast_2d(xi)).ravel()

                    if self.normalize:
                        xi_trans = (xi_trans - self.mean_)
                        xi_trans /= np.sqrt(self.var_)+1e-6

                    y_pred[i] = safe_sparse_dot(xi_trans, coef.T)

            else:
                _predict_fast(coef, get_dataset(X, order='c'), y_pred,
                              self.mean_, self.var_, transformer_fast)
        else:
            X_trans = self.transformer.transform(X)
            y_pred = safe_sparse_dot(X_trans, coef.T)

        y_pred += intercept

        if y_pred.ndim != 1:
            y_pred = y_pred.ravel()

        return y_pred


class LinearClassifierMixin(BaseLinear, ClassifierMixin):
    def decision_function(self, X):
        return self._predict(X)

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
            Predicted target values for X
        """
        pred = self._predict(X)
        out = self.label_binarizer_.inverse_transform(pred)

        if hasattr(self, 'label_encoder_'):
            out = self.label_encoder_.inverse_transform(out)

        return out

    def predict_proba(self, X):
        if self.loss != 'logistic':
            raise AttributeError('Only "logistic" loss supports predict_proba.')
        else:
            pred = self._predict(X)
            if pred.ndim == 1:
                pred = sigmoid(pred)
            else:
                pred = softmax(pred)
        return pred

    def _check_X_y(self, X, y, accept_sparse=True):
        is_2d = hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] >= 2
        if is_2d or type_of_target(y) != 'binary':
            raise TypeError("Only binary targets supported. For training "
                            "multiclass or multilabel models, you may use the "
                            "OneVsRest or OneVsAll metaestimators in "
                            "scikit-learn.")

        X, Y = check_X_y(X, y, dtype=np.double, accept_sparse=accept_sparse,
                         multi_output=False)

        self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self.label_binarizer_.fit_transform(Y).ravel().astype(np.double)
        return X, y


class LinearRegressorMixin(BaseLinear, RegressorMixin):
    def predict(self, X):
        """Perform regression on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
            Predicted target values for X
        """
        return self._predict(X)

    def _check_X_y(self, X, y, accept_sparse=True):
        X, y = check_X_y(X, y, accept_sparse=accept_sparse, multi_output=False,
                         dtype=np.double, y_numeric=True)
        y = y.astype(np.double).ravel()
        return X, y
