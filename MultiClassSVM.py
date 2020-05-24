import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class OneVsAllClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_classes):
        self.n_classes = n_classes
        self.estimators = [clone(estimator) for _ in range(n_classes)]
        self.fitted = False

    def fit(self, X, y=None):
        for i in range(len(self.estimators)):
            y_list = np.zeros(len(y))
            y_list = np.where(y == i, 1, 0)
            self.estimators[i].fit(X, y_list)
        self.fitted = True
        return self

    def decision_function(self, X):
        if not self.fitted:
            raise RuntimeError("You must train classifier before predicting data.")

        if not hasattr(self.estimators[0], "decision_function"):
            raise AttributeError("Base estimator doesn't have a decision_function attribute.")
        ret_row = X.shape[0]
        ret_col = self.n_classes
        ret_mat = np.zeros((ret_row, ret_col))
        for i in range(self.n_classes):
            ret_mat[:, i] = self.estimators[i].decision_function(X)
        return ret_mat

    def predict(self, X):
        all_predicts = self.decision_function(X)
        predict = np.argmax(all_predicts, axis=1)
        return predict
