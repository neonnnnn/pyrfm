import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.utils.multiclass import type_of_target
from ..random_feature import (RandomFourier, RandomMaclaurin, TensorSketch,
                              RandomKernel)
from lightning.impl.dataset_fast import get_dataset
from .stochastic_predict import _predict_fast
from scipy import sparse


def sigmoid(pred):
    return np.exp(np.minimum(0, pred)) / (1.+np.exp(-np.abs(pred)))


class BaseLinear(six.with_metaclass(ABCMeta, BaseEstimator)):
    TRANSFORMERS = (RandomFourier, RandomMaclaurin, TensorSketch, RandomKernel)

    def _predict(self, X):
        check_is_fitted(self, 'coef_')
        if getattr(self, 'stochastic', False):
            y_pred = np.zeros(X.shape[0])
            id_transform = self._get_id_transformer()
            is_sparse = sparse.issparse(X)

            if id_transform == -1 or not self.fast_solver:
                for i, xi in enumerate(X):
                    if is_sparse:
                        xi_trans = self.transformer.transform(xi).ravel()
                    else:
                        xi_trans = self.transformer.transform(
                            np.atleast_2d(xi)).ravel()

                    if self.normalize:
                        xi_trans = (xi_trans - self.mean_)
                        xi_trans /= np.sqrt(self.var_)+1e-6

                    y_pred[i] = safe_sparse_dot(xi_trans, self.coef_)
                    y_pred[i] += self.intercept_

            else:
                params = self._get_transformer_params(id_transform)
                _predict_fast(self.coef_, self.intercept_,
                              get_dataset(X, order='c'), y_pred, self.mean_,
                              self.var_, self.transformer,
                              id_transform, **params)
        else:
            X_trans = self.transformer_.transform(X)
            y_pred = safe_sparse_dot(X_trans, self.coef_.T)

        if self.fit_intercept and hasattr(self, 'intercept_'):
            y_pred += self.intercept_

        if y_pred.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def _get_id_transformer(self):
        if isinstance(self.transformer, self.TRANSFORMERS):
            id_transformer = self.TRANSFORMERS.index(type(self.transformer))
        else:
            id_transformer = -1
        if isinstance(self.transformer, RandomKernel):
            if self.transformer.kernel not in ['anova', 'all_subsets']:
                id_transformer = -1
        return id_transformer

    def _get_transformer_params(self, id_transformer):
        params = {}
        params['random_weights'] = getattr(self.transformer, 'random_weights_',
                                           None)
        params['offset'] = getattr(self.transformer, 'offset_', None)
        params['orders'] = getattr(self.transformer, 'orders_', None)
        params['p_choice'] = getattr(self.transformer, 'p_choice', None)
        params['coefs_maclaurin'] = getattr(self.transformer, 'coefs', None)
        params['hash_indices'] = getattr(self.transformer, 'hash_indices_',
                                         None)
        params['hash_signs'] = getattr(self.transformer, 'hash_signs_', None)
        params['degree'] = getattr(self.transformer, 'degree', -1)
        kernel = -1

        if id_transformer == 2:
            params['random_weights'] = None
        if id_transformer == 3:
            if self.transformer.kernel == 'anova':
                kernel = 0
            elif self.transformer.kernel == 'all_subsets':
                kernel = 1
        params['kernel'] = kernel
        return params


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
