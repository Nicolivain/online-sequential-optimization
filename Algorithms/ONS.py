import random as rd
from Algorithms.Projector import *
import numpy as np


def ons(model, X, y, epoch, l, gamma, z=1, lr=1, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses the gradloss function to update parameters
    :param model: the model
    :param X: (nxm) data
    :param y: (n)  labels
    :param gamma: weight tuning and initialisation of A
    :param epoch: (int) maximum number of iteration of the algorithm
    :param l:  (float) regularization parameter (lambda)
    :param z: (float) radius of the l1-ball
    :param lr: (float) the learning rate
    :param verbose: (int) print epoch results every n epochs
    """

    losses = []
    n, d = X.shape
    model.w = np.zeros(d)
    wts = [model.w]  # initalization = 0
    A = np.diag([1 / gamma**2 for i in range(d)])

    for i in range(epoch):
        # sample
        idx = rd.randint(0, n-1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility

        # compute gradient
        grad = model.gradLoss(sample_x, sample_y, l)

        # matrices update
        A += grad @ grad.T  # Hessian approximated by grad@grad.T
        # Ainstg = Ainv@grad # vect
        # Ainv -= (1 / (1 + grad.T @ Ainstg)) * Ainstg @ (grad.T @ Ainv)
        # using the linalg inversion of A_t at each step t
        Ainv = np.linalg.inv(A)

        # update the last xt
        yt = wts[-1] - lr * (1 / gamma) * Ainv @ grad
        new_wts = weighted_proj_l1(yt, np.diag(A), z)
        # new_wts  = proj_l1ONS(yt, z, A, weighted=True)
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
