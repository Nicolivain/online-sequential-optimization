import numpy as np
import matplotlib.pyplot as plt


def accuracy(y_pred, y_true):
    """
    Compute the accuracy of the provided predictions
    y_pred (n) : prediciton
    y_true (n) : true value to predict
    """
    if len(y_pred) != len(y_true):
        raise KeyError("prediction and truth must be the same size")
    return np.sum(y_pred == y_true)/len(y_true)

def error(y_pred, y_true):
    """
    Compute the error of the provided predictions
    y_pred (n) : prediciton
    y_true (n) : true value to predict
    """
    if len(y_pred) != len(y_true):
        raise KeyError("prediction and truth must be the same size")
    return np.sum(y_pred != y_true)/len(y_true)


def plot_loss(loss, graph_title=None):
    """
    a useful function to plot curves
    :param loss: list (n) or dict (key, list(n)) values or batch of values to be plotted
    :param graph_title: (str) grah
    """

    if type(loss) == list:
        idx = [i for i in range(len(loss))]
        return plt.plot(idx, loss, title=graph_title)
    else:
        ax = None
        for key, vals in loss.items():
            idx = [i for i in range(len(vals))]
            ax = plt.plot(idx, vals, title=graph_title, legend=key)
        return ax

def compute_accuracies(wts, X, y_true):
    """
    Compute the accuracy wrt time of the provided predictions and data
    wts (txm) : weigths at each time step of the algo
    X (nxm) : data to be predicted
    y_true (n) : true value to predict
    """
    accs = []

    for weigts in wts:
        y_pred = np.sign(X.dot(weigts))
        acc = accuracy(y_pred, y_true)
        accs.append(acc)
    return accs

def rate(wts, X, y):
    """
    wts : weights provided during the online fitting
    X : test data
    y : test labels
    """
    acc = []
    for w in wts:
        acc.append(np.mean(y*X.dot(w) > 0))
    return acc
    

