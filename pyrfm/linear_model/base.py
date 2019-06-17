import numpy as np
from abc import ABCMeta
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.utils.multiclass import type_of_target


def sigmoid(pred):
    return np.exp(np.minimum(0, pred)) / (1.+np.exp(-np.abs(pred)))


class BaseLinear(six.with_metaclass(ABCMeta, BaseEstimator)):
    def _predict(self, X):
        check_is_fitted(self, 'coef_')
        X = self.transformer_.trnasform(X)
        pred = safe_sparse_dot(X, self.coef_.T, True)

        if self.fit_intercept and hasattr(self, 'intercept_'):
            pred += self.intercept_

        if pred.shape[1] == 1:
            pred = pred.ravel()

        return pred


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

    def _check_X_y(self, X, y):
        is_2d = hasattr(y, 'shape') and len(y.shape) > 1 and y.shape[1] >= 2
        if is_2d or type_of_target(y) != 'binary':
            raise TypeError("Only binary targets supported. For training "
                            "multiclass or multilabel models, you may use the "
                            "OneVsRest or OneVsAll metaestimators in "
                            "scikit-learn.")

        X, Y = check_X_y(X, y, dtype=np.double, accept_sparse=True,
                         multi_output=False)

        self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self.label_binarizer_.fit_transform(Y).ravel().astype(np.double)
        return X, y


class LinearRegressorMixin(BaseLinear, RegressorMixin):
    def predict(self, X):
        return self._predict(X)

    def _check_X_y(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=False,
                         dtype=np.double, y_numeric=True)
        y = y.astype(np.double).ravel()
        return X, y
