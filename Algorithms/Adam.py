"""
This file contains functions for Adam applied at the SVM problem
https://arxiv.org/pdf/1412.6980.pdf
"""
import random as rd
import numpy as np
from Algorithms.Projector import proj_l1, weighted_proj_simplex


def adam(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = betas[1] * vt_1s + (1 - betas[1]) * model.gradLoss(sample_x, sample_y, l)**2
        mtchap = mts/(1 - betas[0]**t)
        vtchap = vts/(1 - betas[1]**t)

        new_wts = wts[-1] - lr * mtchap / (np.sqrt(vtchap) + 10e-8)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        mt_1s = mts
        vt_1s = vts

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def adamP(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], p=2, verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param p: (int) norm to be considered (1 <= p )
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = betas[1]**p * vt_1s + (1 - betas[1]**p) * np.abs(model.gradLoss(sample_x, sample_y, l))**p
        mtchap = mts/(1 - betas[0]**t)
        vtchap = vts/(1 - betas[1]**t)

        new_wts = wts[-1] - lr * mtchap / (np.power(vtchap, 1/p) + 10e-8)
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        mt_1s = mts
        vt_1s = vts

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def adamproj(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = betas[1] * vt_1s + (1 - betas[1]) * model.gradLoss(sample_x, sample_y, l)**2
        mtchap = mts/(1 - betas[0]**t)
        vtchap = vts/(1 - betas[1]**t)

        new_wts = wts[-1] - lr * mtchap / (np.sqrt(vtchap + 10e-8))
        new_wts  = weighted_proj_simplex(new_wts, np.diag(vts))
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        mt_1s = mts
        vt_1s = vts

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def adaMax(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = np.maximum(betas[1] * vt_1s, np.abs(model.gradLoss(sample_x, sample_y, l)))

        new_wts = wts[-1] - lr /(1 - betas[0]**t) * mts / vts
        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        mt_1s = mts
        vt_1s = vts

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def adamTemporal(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)
    tetats = np.zeros(d)
    tetat_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = betas[1] * vt_1s + (1 - betas[1]) * model.gradLoss(sample_x, sample_y, l)**2
        mtchap = mts/(1 - betas[0]**t)
        vtchap = vts/(1 - betas[1]**t)

        tetats = wts[-1] - lr * mtchap / (np.sqrt(vtchap) + 10e-8)
        new_wts = (betas[1] * tetat_1s + (1 - betas[1]) * tetats) / (1 - betas[1]**t)

        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)
        
        mt_1s = mts
        vt_1s = vts
        tetat_1s = tetats

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)


def adaMaxTemporal(model, X, y, lr, epoch, l, z=1, betas = [0.9, 0.999], verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param betas:(1x2) exponential decay rates for the moment estimates
        :param verbose: (int) print epoch results every n epochs
        """

    n, d = X.shape
    losses = []
    wts = [1 / (2*d) * np.zeros(d)]
    mts = np.zeros(d)
    mt_1s = np.zeros(d)
    vts = np.zeros(d)
    vt_1s = np.zeros(d)
    tetats = np.zeros(d)
    tetat_1s = np.zeros(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        t = i + 1

        mts = betas[0] * mt_1s + (1 - betas[0]) * model.gradLoss(sample_x, sample_y, l)
        vts = np.maximum(betas[1] * vt_1s, np.abs(model.gradLoss(sample_x, sample_y, l)))

        tetats = wts[-1] - lr /(1 - betas[0]**t) * mts / vts
        new_wts = (betas[1] * tetat_1s + (1 - betas[1]) * tetats) / (1 - betas[1]**t)

        wts.append(new_wts)
        model.w = new_wts

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        mt_1s = mts
        vt_1s = vts
        tetat_1s = tetats

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    return losses, np.array(wts)