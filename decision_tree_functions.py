import numpy as np
import math
from gurobipy import *
from sklearn.tree import _tree
from graphviz import Digraph
from pydotplus import graph_from_dot_data

def define_branch_ancestors(leaf_nodes):
    """ Define the left/right-branch ancestors of leaf nodes.
    
    Returns
    -------
    left_branch_ancestors : dict mapping leaf nodes to left-branch ancestors
    right_branch_ancestors : dict mapping leaf nodes to right-branch ancestors
    """
    left_branch_ancestors = {}
    right_branch_ancestors = {}
    for n in leaf_nodes:
        lba = set()
        rba = set()
        curr_node = n
        while curr_node > 1:
            parent = int(curr_node/2)
            if curr_node == 2*parent:
                lba.add(parent)
            else:
                rba.add(parent)
            curr_node = parent
        left_branch_ancestors[n] = lba
        right_branch_ancestors[n] = rba
    return left_branch_ancestors, right_branch_ancestors

def construct_flow_network(max_depth):
    """ Given max depth, construct the flow network. """
    # Nodes (not including source s and sink t)
    internal_nodes = list(range(1, 2**max_depth))
    leaf_nodes = list(range(2**max_depth, 2**(max_depth+1)))
    
    # Edges
    edges = []
    edges.append(('s',1))
    for n in internal_nodes:
        edges.append((n,2*n))
        edges.append((n,2*n+1))
    for n in internal_nodes + leaf_nodes:
        edges.append((n,'t'))
    
    return internal_nodes, leaf_nodes, edges

def compute_epsilon(X):
    """ Compute epsilon for enforcing univariate splits on numerical data.
    
    Parameters
    ----------
    X : NumPy array of shape (N,p)
    
    Returns
    -------
    epsilon : dictionary mapping features {0,...,p-1} to epsilon value
    """
    cols_sorted = np.sort(X, axis=0)
    epsilon = {}
    N, p = np.shape(X)
    for j in range(p):
        min_diff = 1 # Rely on assumption that features are scaled to [0,1]
        for i in range(N-1):
            diff = cols_sorted[i+1,j] - cols_sorted[i,j]
            if (diff < min_diff) and (diff > 0):
                min_diff = diff
        # TODO If warm starting b with CART, shouldn't divide epsilon by 2, should adjust thresholds
        epsilon[j] = min_diff/2 # Divide by 2 if using CART for warm starting b
    return epsilon

def cart_tree_to_rules(cart_tree):
    """ Convert a fitted DecisionTreeClassifier to rules. """
    branch_rules = {}
    classification_rules = {}
    
    # Pull attributes from cart_tree
    n_features_ = cart_tree.n_features_
    classes_ = cart_tree.classes_
    tree_ = cart_tree.tree_
    
    # "node" is the node in the CART tree, "n" is the node in our tree
    def recurse(node, n):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Split at node in CART tree is of the form X[j] <= b; retrieve j and b
            j = tree_.feature[node]
            b = tree_.threshold[node]
            # Construct NumPy ndarray to represent LHS vector a in split ax <= b
            a = np.zeros(n_features_)
            a[j] = 1
            branch_rules[n] = (a,b)
            recurse(tree_.children_left[node], 2*n)
            recurse(tree_.children_right[node], 2*n + 1)
        else:
            value = tree_.value[node]
            class_index = np.argmax(value)
            classification_rules[n] = classes_[class_index]
    
    recurse(0,1)
    
    return branch_rules, classification_rules

def classify_with_rules(x, branch_rules, classification_rules):
    """ Classify a single instance using rules. """
    n = 1
    while True:
        if n in branch_rules:
            a, b = branch_rules[n]
            if np.dot(a,x) <= b:
                n = 2*n
            else:
                n = 2*n + 1
        else:
            prediction = classification_rules[n]
            return prediction

def view_tree(decision_tree, feature_names=None, name="Tree", write_dot=False, write_png=False):
    """ Convert tree to DOT format, then to PNG image.
    
    Generate a GraphViz representation of the tree, optionally write it to a DOT
    file, optionally create a PNG image. Each node is labeled with either the
    split "X[j] < threshold" if the node is a branch node, or the class if the
    node is a leaf node.
    
    TODO: Maybe it is helpful to know how many samples pass through each node?
    
    Parameters
    ----------
    decision_tree : our decision tree
    feature_names : list
    name
    write_dot
    write_png
    
    Returns
    -------
    dot_data : string representation of the tree in GraphViz dot format
    """
    tree = Digraph(name)
    
    # Create nodes
    for n in decision_tree.branch_rules_:
        label = "Node {}\n".format(n)
        (a,b) = decision_tree.branch_rules_[n]
        for j in range(decision_tree.n_features_):
            if not math.isclose(a[j], 0, abs_tol=10**-5):
                if feature_names is None:
                    label += "{}X[{}] + ".format(a[j], j)
                else:
                    label += "{}({}) + ".format(a[j], feature_names[j])
        label = label[:-2] # Remove trailing "+ "
        label += "<= {}".format(b)
        tree.node(name=str(n), label=label, shape='box')
    for n in decision_tree.classification_rules_:
        label = "Node {}\n".format(n)
        label += "class = {}".format(decision_tree.classification_rules_[n])
        tree.node(name=str(n), label=label, shape='box')
    
    # Create edges
    for n in decision_tree.branch_rules_:
        tree.edge(tail_name=str(n), head_name=str(2*n), label="Yes")
        tree.edge(tail_name=str(n), head_name=str(2*n + 1), label="No")
    
    dot_data = tree.source
    
    if write_dot:
        dot_filename = '{}.dot'.format(name)
        tree.save(dot_filename)
    
    if write_png:
        png_filename = '{}.png'.format(name)
        # Create pydotplus.graphviz.Dot object to write to PNG
        t = graph_from_dot_data(dot_data)
        t.write_png(png_filename)
    
    return dot_data
