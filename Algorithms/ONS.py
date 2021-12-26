"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""
import random as rd
import numpy as np
from Algorithms.Projector import weighted_proj_l1

# TODO : use the mean to update the model (cf .R) because that's the interesting result and not value

def ONS(model, X, y, epoch, l, gamma, z=1, verbose=0):
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
    wts = [np.zeros(len(model.w))]
    n, _ = X.shape
    A = np.diag([1 / gamma ** 2 for i in range(n)])
    Ainv = np.diag([gamma ** 2 for i in range(n)])

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # update the last xt

        new_wts = wts[-1] - 1/gamma * Ainv @ model.gradLoss(sample_x, sample_y, l)
        new_wts  = weighted_proj_l1(new_wts, np.diag(A), z)
        wts.append(new_wts)
        model.w = new_wts
        A += model.gradLoss(sample_x, sample_y, l) @ model.gradLoss(sample_x, sample_y, l).T
        Ainv = np.linalg.inv(A)

        # loss
        current_loss = model.loss(X, y, l)
        losses.append(current_loss)

        if verbose > 0 and i % verbose == 0:
            print("Epoch {:3d} : Loss = {:1.4f}".format(i, current_loss))

    # update wts:
    model.w = np.mean(wts, axis=0)
    return losses, wts