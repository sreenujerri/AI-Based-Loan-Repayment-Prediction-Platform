import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestClassifier(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 50)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

        reg = self.train(
            x_train,
            y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

        return reg.score(x_test, y_test)


class AdaBoostModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = AdaBoostClassifier(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.99)

        reg = self.train(
            x_train,
            y_train,
            n_estimators=n_estimators,
           
            learning_rate=learning_rate
        )

        return reg.score(x_test, y_test)



class LogisticRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LogisticRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning using Optuna.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: self.model.optimize(
                trial,
                self.x_train,
                self.y_train,
                self.x_test,
                self.y_test
            ),
            n_trials=n_trials
        )

        return study.best_trial.params