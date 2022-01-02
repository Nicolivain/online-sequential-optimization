"""
This file contains functions for the mirror descent algorithm applied at the SVM problem
"""
import random as rd
from Algorithms.Projector import *
import numpy as np


def smd(model, X, y, epoch, l, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    losses = []
    wts = [np.zeros(len(model.w))]
    n, _ = X.shape

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        dlr = lr / np.sqrt(t)
        new_wts = wts[-1] - dlr * model.gradLoss(sample_x, sample_y, l)
        new_wts = proj_l1(new_wts, z)
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


def seg(model, X, y, lr, epoch, l, z=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    tetatp = np.zeros(d)
    tetatm = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        etat = lr * np.sqrt(1 / t)
        tetatm -= etat * model.gradLoss(sample_x, sample_y, l)
        tetatp += etat * model.gradLoss(sample_x, sample_y, l)
        tetat = np.r_[tetatm, tetatp]
        new_wts = np.exp(tetat)/np.sum(np.exp(tetat))
        new_wts  = z * (new_wts[0:d] - new_wts[d:])
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


def adagrad(model, X, y, epoch, l, z=1, verbose=0, lr=0.1):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param lr: (float) learning rate
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param verbose: (int) print epoch results every n epochs
    """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    Sts = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        Sts += model.gradLoss(sample_x, sample_y, l)**2
        Dt = np.diag(np.sqrt(Sts))
        yts = wts[-1] - lr * np.linalg.inv(Dt).dot(model.gradLoss(sample_x, sample_y, l))
        new_wts = weighted_proj_l1(yts, np.diag(Dt), z)
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
