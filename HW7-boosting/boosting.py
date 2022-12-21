from __future__ import annotations

from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        part = int(x.shape[0] * self.subsample)

        sample_idx = np.random.randint(0, x.shape[0], part)
        x_samples = x[sample_idx, :]
        y_samples = y[sample_idx]
        pred_samples = predictions[sample_idx]

        s = -self.loss_derivative(y_samples, pred_samples)

        new_model = self.base_model_class(**self.base_model_params)
        new_model.fit(x_samples, s)
        new_preds = predictions + new_model.predict(x)

        new_gamma = self.find_optimal_gamma(y, predictions, new_preds) * self.learning_rate

        self.gammas.append(new_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        self.history['train'].append(self.loss_fn(y_train, train_predictions))
        self.history['val'].append(self.loss_fn(y_valid, valid_predictions))

        bad_rounds = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)

            train_loss = self.loss_fn(y_train, train_predictions)
            self.history['train'].append(train_loss)

            if self.early_stopping_rounds is not None:
                valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
                val_loss = self.loss_fn(y_valid, valid_predictions)
                self.history['val'].append(val_loss)

                if val_loss >= self.history['val'][-2]:
                    bad_rounds += 1
                else:
                    bad_rounds = 0

                if bad_rounds >= self.early_stopping_rounds:
                    break

        if self.plot:
            plt.plot(self.history['train'], label='train')
            plt.xlabel('total models')
            plt.ylabel('loss')

            if 'val' in self.history:
                plt.plot(self.history['val'], label='val')

            plt.legend()
            plt.show()

    def predict_proba(self, x):
        res = 0
        for gamma, model in zip(self.gammas, self.models):
            res += gamma * model.predict(x)
        res = self.sigmoid(res)

        return np.vstack([1 - res, res]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        if len(self.models) == 0:
            raise Exception('Model is not fitted')

        sum_imp = np.zeros((self.models[0].feature_importances_.shape[0],))
        for model in self.models:
            sum_imp += model.feature_importances_

        avg_imp = sum_imp / len(self.models)
        e_x = np.exp(avg_imp - np.max(avg_imp))

        return e_x / e_x.sum(axis=0)
