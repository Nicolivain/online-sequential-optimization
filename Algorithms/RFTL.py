"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""
import random as rd
import numpy as np
from Algorithms.Projector import proj_l1

# TODO : use the mean to update the model (cf .R) because that's the interesting result and not value 

def smd(model, X, y, lr, epoch, l, z=1, verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
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
        idx = rd.randint(0, n)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        # t = i + 1
        # lr = 1 / np.sqrt(t) # TODO : check why it doesn't work with decresing lr
        new_wts = wts[-1] - lr * model.gradLoss(sample_x, sample_y, l)
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
    return losses


def seg(model, X, y, lr, epoch, l, z=1, verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
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
        idx = rd.randint(0, n)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        etat = np.sqrt(1 / t) 
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
    return losses

def adagrad(model, X, y, lr, epoch, l, z=1, verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
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
        idx = rd.randint(0, n)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1
        etat = np.sqrt(1 / t) 
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
    return losses