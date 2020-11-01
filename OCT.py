import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from gurobipy import *
from graphviz import Digraph
from pydotplus import graph_from_dot_data
from decision_tree_functions import *

class OCTClassifier(ClassifierMixin, BaseEstimator):
    """ Implementation of OCT from Optimal classification trees.
    
    The model is slightly modified to be consistent with conventions introduced
    in the paper by Aghaei et al. For example, there is no min_samples_leaf
    hyperparameter, and the objective is changed from minimizing in-sample
    misclassification loss to maximizing in-sample accuracy. The model is as
    simple as can be while still preserving key ideas in the original paper.
    
    Parameters
    ----------
    max_depth
    lambda_
    warm_start : None or tuple
        If a warm start is provided, it must be a tuple of the form (branch_rules, classification_rules)
    gurobi_MIPGap
    gurobi_TimeLimit
    gurobi_LogToConsole
    
    Attributes
    ----------
    classes_
    n_features_
    model_ : Gurobi Model
    branch_nodes_
    classification_rules_
    """
    def __init__(self, max_depth, lambda_=0.0, warm_start=None, gurobi_MIPGap=0.01, gurobi_TimeLimit=None, gurobi_LogToConsole=0):
        if (lambda_ < 0.0) or (lambda_ > 1.0):
            raise ValueError("lambda_ must be in [0,1]")
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.warm_start = warm_start
        self.gurobi_MIPGap = gurobi_MIPGap
        self.gurobi_TimeLimit = gurobi_TimeLimit
        self.gurobi_LogToConsole = gurobi_LogToConsole
    
    def fit(self, X, y):
        """ Trains a classification tree using the OCT model.
        
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
        
        # N = # instances, p = # features
        N, p = np.shape(X)
        self.n_features_ = p
        
        # Define classes
        self.classes_ = unique_labels(y)
        
        # Construct nodes
        internal_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        
        # Define left/right-branch ancestors
        left_branch_ancestors, right_branch_ancestors = define_branch_ancestors(leaf_nodes)
        
        # epsilon, big-M values for enforcing splits
        epsilon = compute_epsilon(X)
        M_left = 1
        M_right = 1 + min(epsilon.values())
        
        # Define Y
        Y = {}
        for i in range(N):
            for k in self.classes_:
                Y[i,k] = +1 if y[i] == k else -1
        
        #
        # OCT model
        #
        
        m = Model("OCT")
        self.model_ = m
        m.Params.LogToConsole = self.gurobi_LogToConsole
        m.Params.MIPGap = self.gurobi_MIPGap
        if self.gurobi_TimeLimit is not None:
            m.Params.TimeLimit = self.gurobi_TimeLimit
        
        # Decision variables
        a = m.addVars(internal_nodes, range(p), vtype=GRB.BINARY, name="a")
        b = m.addVars(internal_nodes, name="b")
        c = m.addVars(self.classes_, leaf_nodes, vtype=GRB.BINARY, name="c")
        d = m.addVars(internal_nodes, vtype=GRB.BINARY, name="d")
        w = m.addVars(leaf_nodes, range(N), vtype=GRB.BINARY, name="w")
        z = m.addVars(leaf_nodes, name="z")
        Nkn = m.addVars(self.classes_, leaf_nodes, name="Nkn")
        Nn = m.addVars(leaf_nodes, name="Nn")
        
        # Objective
        m.setObjective((1-self.lambda_)*z.sum() - self.lambda_*d.sum(), GRB.MAXIMIZE)
        
        # Constraints
        m.addConstrs((a.sum(n,'*') == d[n] for n in internal_nodes), name="split_a")
        m.addConstrs((b[n] <= d[n] for n in internal_nodes), name="split_b")
        m.addConstrs((d[n] <= d[int(n/2)] for n in internal_nodes if n != 1), name="split_only_if_parent_splits")
        m.addConstrs((w.sum('*',i) == 1 for i in range(N)), name="assign_points_to_leaves")
        m.addConstrs((quicksum(a[ancestor,j]*X[i,j] for j in range(p)) <= b[ancestor] + M_left*(1 - w[n,i]) for n in leaf_nodes for ancestor in left_branch_ancestors[n] for i in range(N)), name="left_branching")
        # Incorrect
        #m.addConstrs((quicksum(a[ancestor,j]*(X[i,j] - epsilon[j]) for j in range(p)) >= b[ancestor] - M_right*(1 - w[n,i]) for n in leaf_nodes for ancestor in right_branch_ancestors[n] for i in range(N)), name="right_branching")
        # Correct but not the fastest
        m.addConstrs((quicksum(a[ancestor,j]*X[i,j] for j in range(p)) - min(epsilon.values()) >= b[ancestor] - M_right*(1 - w[n,i]) for n in leaf_nodes for ancestor in right_branch_ancestors[n] for i in range(N)), name="right_branching")
        # Correct (I think) and faster than the previous implementation, but kinda Frankensteined together
        #M_right = 1 + 0.5*max(epsilon.values()) + 0.5*min(epsilon.values())
        #m.addConstrs((quicksum(a[ancestor,j]*(X[i,j] - 0.5*epsilon[j]) for j in range(p)) - 0.5*min(epsilon.values()) >= b[ancestor] - M_right*(1 - w[n,i]) for n in L for ancestor in right_branch_ancestors[n] for i in range(N)), name="right_branching")
        m.addConstrs((Nkn[k,n] == 0.5*quicksum((1+Y[i,k])*w[n,i] for i in range(N)) for k in self.classes_ for n in leaf_nodes), name="Nkn_defn")
        m.addConstrs((Nn[n] == w.sum(n,'*') for n in leaf_nodes), name="Nn_defn")
        m.addConstrs((c.sum('*',n) == 1 for n in leaf_nodes), "classify_at_leaves")
        m.addConstrs((z[n] <= Nkn[k,n] + N*(1 - c[k,n]) for k in self.classes_ for n in leaf_nodes), name="loss_defn1")
        m.addConstrs((z[n] <= Nn[n] for n in leaf_nodes), name="loss_defn2")
        
        # Pack variables and data into m
        m._vars = (a, b, c, d, w, z, Nkn, Nn)
        m._X_y = X, y
        m._n_features = p
        m._classes = self.classes_
        m._nodes = internal_nodes, leaf_nodes
        m._warm_start = self.warm_start
        
        # Use warm start if one is provided
        if self.warm_start is not None:
            OCTClassifier._load_warm_start(m)
        
        # Solve model
        m.optimize()
        
        #
        # Use learned parameters to define classifier
        #
        
        self.branch_rules_, self.classification_rules_ = OCTClassifier._learned_parameters_to_rules(m)
        
        return self
    
    @staticmethod
    def _load_warm_start(model):
        """ Load a warm start into Gurobi variables. """
        (a, b, c, d, w, z, Nkn, Nn) = model._vars
        X, y = model._X_y
        p = model._n_features
        classes = model._classes
        internal_nodes, leaf_nodes = model._nodes
        init_branch_rules, init_classification_rules = model._warm_start
        # Set Start for a, b, d
        for n in internal_nodes:
            if n in init_branch_rules:
                a_init, b_init = init_branch_rules[n]
                for j in range(p):
                    a[n,j].Start = a_init[j]
                b[n].Start = b_init
                d[n].Start = 1
            else:
                for j in range(p):
                    a[n,j].Start = 0
                b[n].Start = 0
                d[n].Start = 0
        # Set Start for c
        # First initialize c[k,n].Start for all k, n
        for n in leaf_nodes:
            for k in classes:
                c[k,n].Start = 0
            c[classes[0],n].Start = 1 # Must set c[k,n].Start = 1 for some k for each n for feasibility
        # Next set c[k,n].Start = 1 for select k, n
        for n in init_classification_rules:
            class_ = init_classification_rules[n]
            child = n
            while child not in leaf_nodes:
                child *= 2
            c[classes[0],child].Start = 0 # Was set to 1 initially
            c[class_,child].Start = 1
        # Set Start for w, z, Nkn, Nn
        z_start = {}
        Nkn_start = {}
        Nn_start = {}
        for n in leaf_nodes:
            z_start[n] = 0
            for k in classes:
                Nkn_start[k,n] = 0
            Nn_start[n] = 0
        for i,x in enumerate(X):
            for n in leaf_nodes:
                w[n,i].Start = 0
            # Find which leaf n and class k datapoint i is assigned to
            n = 1
            while n in init_branch_rules:
                a_init, b_init = init_branch_rules[n]
                if np.dot(a_init,x) <= b_init:
                    n = 2*n
                else:
                    n = 2*n + 1
            k = init_classification_rules[n]
            while n not in leaf_nodes:
                n *= 2
            w[n,i].Start = 1
            z_start[n] += (k == y[i])
            Nkn_start[y[i],n] += 1
            Nn_start[n] += 1
        for n in leaf_nodes:
            z[n].Start = z_start[n]
            for k in classes:
                Nkn[k,n].Start = Nkn_start[k,n]
            Nn[n].Start = Nn_start[n]
    
    @staticmethod
    def _learned_parameters_to_rules(model):
        """ Given optimal values of a, b, c, and d, construct decision tree rules. """
        branch_rules = {}
        classification_rules = {}
        
        # Extract variables and data from model
        (a, b, c, d, _, _, _, _) = model._vars
        a_vals = model.getAttr('x', a)
        b_vals = model.getAttr('x', b)
        c_vals = model.getAttr('x', c)
        d_vals = model.getAttr('x', d)
        p = model._n_features
        classes = model._classes
        internal_nodes, leaf_nodes = model._nodes
        tree_nodes = internal_nodes + leaf_nodes
        
        for n in tree_nodes:
            parent_applies_split = (int(n/2) in branch_rules)
            if parent_applies_split or (n == 1):
                n_applies_split = False
                if n in internal_nodes:
                    n_applies_split = (d_vals[n] > 0.5)
                if n_applies_split:
                    lhs = np.asarray([a_vals[n,j] for j in range(p)])
                    # Added tolerance deals with numerical issues in finding in-sample accuracy
                    rhs = b_vals[n] + model.Params.FeasibilityTol
                    branch_rules[n] = (lhs, rhs)
                else:
                    # Take left branch until at a leaf
                    child = n
                    while child not in leaf_nodes:
                        child *= 2
                    class_index = np.argmax([c_vals[k,child] for k in classes])
                    classification_rules[n] = classes[class_index]
        
        return branch_rules, classification_rules
    
    def predict(self, X):
        """ Classify instances.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray
        
        Returns
        -------
        y : pandas Series
        """
        check_is_fitted(self,['classes_','n_features_','model_','branch_rules_','classification_rules_'])
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Convert to ndarray
        # Note: Previously I was using numpy.apply_along_axis(), but this truncates class labels if they are strings of differing lengths
        y_pred = []
        for x in X:
            y_pred.append(classify_with_rules(x, self.branch_rules_, self.classification_rules_))
        y_pred = pd.Series(y_pred, index=index)
        return y_pred
