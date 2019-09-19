import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.utils.multiclass import type_of_target
from ..random_feature.random_features_fast import get_fast_random_feature
from lightning.impl.dataset_fast import get_dataset
from .stochastic_predict import _predict_fast
from scipy import sparse


def sigmoid(pred):
    return np.exp(np.minimum(0, pred)) / (1.+np.exp(-np.abs(pred)))


class BaseLinear(six.with_metaclass(ABCMeta, BaseEstimator)):
    # for stochastic solver
    def _init_params(self, n_components):
        if not (self.warm_start and hasattr(self, 'coef_')):
            self.coef_ = np.zeros(n_components)

        if not (self.warm_start and hasattr(self, 'intercept_')):
            self.intercept_ = np.zeros((1,))

        if not (self.warm_start and hasattr(self, 't_')):
            self.t_ = 1

        if self.loss not in self.LOSSES:
            raise ValueError("loss {} is not supported.".format(self.loss))

        if self.normalize:
            if not (self.warm_start and hasattr(self, 'mean_')):
                self.mean_ = np.zeros((n_components, ))

            if not (self.warm_start and hasattr(self, 'var_')):
                self.var_ = np.zeros((n_components,))
        else:
            self.mean_ = None
            self.var_ = None

    def _predict(self, X):
        check_is_fitted(self, 'coef_')
        if hasattr(self, "transformer_"):
            transformer = self.transformer_
        else:
            transformer = self.transformer
        if getattr(self, 'stochastic', False):
            y_pred = np.zeros(X.shape[0])
            is_sparse = sparse.issparse(X)
            transformer_fast = get_fast_random_feature(transformer)
            if transformer_fast is None or not self.fast_solver:
                for i, xi in enumerate(X):
                    if is_sparse:
                        xi_trans = transformer.transform(xi).ravel()
                    else:
                        xi_trans = transformer.transform(
                            np.atleast_2d(xi)).ravel()

                    if self.normalize:
                        xi_trans = (xi_trans - self.mean_)
                        xi_trans /= np.sqrt(self.var_)+1e-6

                    y_pred[i] = safe_sparse_dot(xi_trans, self.coef_)

            else:
                _predict_fast(self.coef_, get_dataset(X, order='c'), y_pred,
                              self.mean_, self.var_, transformer_fast)
        else:
            X_trans = transformer.transform(X)
            y_pred = safe_sparse_dot(X_trans, self.coef_.T)

        if self.fit_intercept and hasattr(self, 'intercept_'):
            y_pred += self.intercept_

        if y_pred.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred


class LinearClassifierMixin(BaseLinear, ClassifierMixin):
    def decision_function(self, X):
        return self._predict(X)

    def predict(self, X):
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
        return self._predict(X)

    def _check_X_y(self, X, y, accept_sparse=True):
        X, y = check_X_y(X, y, accept_sparse=accept_sparse, multi_output=False,
                         dtype=np.double, y_numeric=True)
        y = y.astype(np.double).ravel()
        return X, y
