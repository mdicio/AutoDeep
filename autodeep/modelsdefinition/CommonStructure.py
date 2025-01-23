import numpy as np
import torch
import random


class BaseModel:
    """
    Base class for all models.
    """

    def __init__(self, random_state=4200, problem_type=None):
        super(BaseModel, self).__init__()
        """
        Constructor for the base model class.
        
        Parameters
        ----------
        model_params : dict
            Dictionary containing the hyperparameters for the model.
        """
        self.parameters = None
        self.model = None
        self.random_state = random_state
        self.problem_type = problem_type
        self.kwargs = kwargs

    def train(self, X_train, y_train):
        """
        Method to train the model on training data.

        Parameters
        ----------
        X_train : ndarray
            Training data input.
        y_train : ndarray
            Training data labels.
        """
        raise NotImplementedError

    def predict(self, X_test, predict_proba=False):
        """
        Method to generate predictions on test data.

        Parameters
        ----------
        X_test : ndarray
            Test data input.

        Returns
        -------
        ndarray
            Array of model predictions.
        """
        raise NotImplementedError
