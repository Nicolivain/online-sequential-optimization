"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""
import random as rd
import numpy as np
from Algorithms.Projector import *

# TODO : use the mean to update the model (cf .R) because that's the interesting result and not value

def ons(model, X, y, epoch, l, gamma, z=1, verbose=0):
    """
        Gradient descent algorithms applied with the CO pb il loss and uses the gradloss function to update parameters
        :param X: (nxm) data
        :param y: (n)  labels
        :param lr: (float) learning rate
        :param epoch: (int) maximum number of iteration of the algorithm
        :param l:  (float) regularization parameter (lambda)
        :param z: (float) radius of the l1-ball
        :param verbose: (int) print epoch results every n epochs
        """

    losses = []
    n, d = X.shape
    model.w = np.zeros(d)
    wts = [model.w]  # initalization = 0
    A = np.diag([1 / gamma**2 for i in range(d)])
    Ainv = np.diag([gamma**2 for i in range(d)])

    for i in range(epoch):
        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt
        grad = model.gradLoss(sample_x, sample_y, l)
        A += grad @ grad.T  # Hessian approximated by grad@grad.T

        # Ainv = np.linalg.inv(A) #test using the inversion of A_t at each step t

        Ainstg = Ainv@grad #vect

        #Ainv -= (1 / (1 + grad.T @ Ainstg)) * Ainstg @ (grad.T @ Ainv)
        Ainv = np.linalg.inv(A)
        new_wts = wts[-1] - (1/gamma) * Ainv @ grad
        new_wts = weighted_proj_l1(new_wts, np.diag(A), z=z)
        #new_wts  = proj_l1(new_wts, z, np.diag(A))
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