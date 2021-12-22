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
