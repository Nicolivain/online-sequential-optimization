"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""


def GradientDescent(model, X, y, lr, epoch, l, verbose=0):
    """
    Unconstrained GD
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param verbose: (int) print epoch results every n epochs
    """
    losses = []
    for i in range(epoch):
        model.w -= lr * model.gradLoss(X, y, l)
        current_loss = model.loss(X, y, l)
        losses += [current_loss]
        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))
    return losses

