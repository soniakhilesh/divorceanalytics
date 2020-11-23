import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *


class kNNClassifierIP(ClassifierMixin, BaseEstimator):
    """ k-NN with weighted features. Weights are learnable parameters.

    Parameters
    ----------
    k : int
        Number of neighbors
    p : int
        p-norm
    lambda_: float
        multi objective
    gurobi_MIPGap
    gurobi_TimeLimit
    gurobi_LogToConsole

    Attributes
    ----------
    classes_
    model_ : Gurobi Model
    weights_
    """

    def __init__(self, k=5, p=2, lambda_=0.1, gurobi_MIPGap=0.01, gurobi_TimeLimit=None, gurobi_LogToConsole=0):
        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.gurobi_MIPGap = gurobi_MIPGap
        self.gurobi_TimeLimit = gurobi_TimeLimit
        self.gurobi_LogToConsole = gurobi_LogToConsole

    def fit(self, X, y):
        """ Learn feature weights.

        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray
        y : pandas Series or NumPy ndarray

        Returns
        -------
        self
        """
        #
        # Input validation, model setup
        #

        # Check that dimensions are consistent, convert X and y to ndarrays
        X, y = check_X_y(X, y)

        # Check that all entries of X are normalized to [0,1]
        if not np.all(np.logical_and(X >= -np.finfo(float).eps, X <= 1 + np.finfo(float).eps)):
            raise ValueError("Features must be normalized to [0,1]")

        # N = # instances, d = # features
        N, d = np.shape(X)

        # Define classes
        self.classes_ = unique_labels(y)

        # Define cost
        c = {}
        for i in range(N):
            for j in range(N):
                if i == j or j == i:
                    # assign a very high cost
                    c[i, j] = np.inf
                else:
                    if y[i] == y[j]:
                        # same labels: cost 0
                        c[i, j] = 0
                    else:
                        # different labels: cost 1
                        c[i, j] = 1

        # define distance
        distance = {}
        for i in range(N):
            for j in range(N):
                for f in range(d):
                    if i == j:
                        distance[i, j, f] = 0
                    else:
                        distance[i, j, f] = self.lambda_ * abs(X[i, f] - X[j, f]) ** self.p
        #
        # Model
        #

        m = Model("kNN-IP")
        self.model_ = m
        m.Params.LogToConsole = self.gurobi_LogToConsole
        m.Params.MIPGap = self.gurobi_MIPGap
        if self.gurobi_TimeLimit is not None:
            m.Params.TimeLimit = self.gurobi_TimeLimit

        # Decision variables
        w = m.addVars(range(d), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
        z = m.addVars(range(N), range(N), obj=c, vtype=GRB.BINARY, name="z")
        u = m.addVars(range(N), range(N), range(d), obj=distance, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="u")

        # Objective
        m.ModelSense = GRB.MINIMIZE

        # Constraints
        m.addConstr(w.sum() == 1, name="w_sum")
        m.addConstrs((z.sum(i, '*') == self.k for i in range(N)), name="pick_kNN")
        m.addConstrs((u[i, j, f] >= w[f] - (1 - z[i, j]) for i in range(N) for j in range(N)
                      for f in range(d)), name="relationship")

        # Pack variables and data into m
        m._vars = (w, z, u)
        m._X_y = X, y
        m._k = self.k
        m._p = self.p

        # Solve model
        m.optimize()

        #
        # Save weights
        #

        self.weights_ = m.getAttr('x', w)

        return self

    def predict(self, X):
        """ Classify instances.

        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray

        Returns
        -------
        y : pandas Series
        """
        check_is_fitted(self, ['classes_', 'model_', 'weights_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X)  # Convert to ndarray
        d = np.shape(X)[1]
        k = self.model_._k
        p = self.model_._p
        w = self.weights_
        X_train, y_train = self.model_._X_y
        # Note: Previously I was using numpy.apply_along_axis(), but this truncates class labels if they are strings of differing lengths
        y_pred = []
        for x in X:
            nearest_neighbors = []
            for i, xi in enumerate(X_train):
                dist = sum(w[j] * (abs(x[j] - xi[j]) ** p) for j in range(d))  # Not taking (1/p)th power
                nearest_neighbors.append((i, dist))
            nearest_neighbors.sort(key=lambda tup: tup[1])
            nearest_neighbors = nearest_neighbors[:k]
            labels = [y_train[nn[0]] for nn in nearest_neighbors]
            prediction = max(set(labels), key=labels.count)
            y_pred.append(prediction)
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
