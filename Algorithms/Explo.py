"""
This file contains functions for Exploration algorithm applied at the SVM problem
"""
import random as rd
import numpy as np
from Algorithms.Projector import proj_l1, weighted_proj_simplex


def sreg(model, X, y, lr, epoch, l, z=1, verbose=0):
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
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility
        Jt = np.random.randint(0, d, 1) # sample the direction

        # update the last xt
        t = i + 1
        etat = np.sqrt(1 / t) 

        tetatm[Jt] -= (etat * model.gradLoss(sample_x, sample_y, l))[Jt]
        tetatp[Jt] += (etat * model.gradLoss(sample_x, sample_y, l))[Jt]
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

def sbeg(model, X, y, lr, epoch, l, z=1, verbose=0):
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
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility
        Jt = np.random.randint(0, d, 1) # sample the direction
        sgt = np.random.randint(0, 2, 1, dtype=bool) # sample the sign : True is + and False is -

        # update the last xt
        t = i + 1
        etat = np.sqrt(1 / t) 

        # if sign is > 0 then we modify the first part 
        if sgt :
            tetatm[Jt] -= (etat * model.gradLoss(sample_x, sample_y, l))[Jt] 
        # sign is < 0 then we modify this part 
        else :
            tetatp[Jt] += (etat * model.gradLoss(sample_x, sample_y, l))[Jt]
        tetat = np.r_[tetatm, tetatp]
        new_wts = np.exp(tetat)/np.sum(np.exp(tetat)) * (1 - etat) + etat / (2 * d)

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