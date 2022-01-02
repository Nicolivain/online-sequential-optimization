"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""
import random as rd
import numpy as np
from Algorithms.Projector import *


def sgd(model, X, y, epoch, l, verbose=0, lr=1):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param model: the model
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param verbose: (int) print epoch results every n epochs
    """

    losses = []
    wts = [model.w]
    n, _ = X.shape
    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        dlr = lr / (l * t)
        new_wts = (1 - 1 / t) * wts[-1] - dlr * model.gradLoss(sample_x, sample_y, l)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    return losses, np.array(wts)


def projected_sgd(model, X, y, epoch, l, z=1, verbose=0, lr=1):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param model: the model
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) = 1 so that it is fixed to 1/(lambda*t) in the descent step
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius for projection on the l1-ball
        :param verbose: (int) print epoch results every n epochs
        """

    losses = []
    wts = [model.w]
    n, _ = X.shape
    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        dlr = lr / (l * t)
        new_wts = (1 - 1 / t) * wts[-1] - dlr * model.gradLoss(sample_x, sample_y, l)
        new_wts  = proj_l1(new_wts, z)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    
    return losses, np.array(wts)
