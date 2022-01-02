import random as rd
import numpy as np


def sreg(model, X, y, epoch, l, z=1, lr=1, verbose=0):
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
    wts = [np.zeros(d)]
    tetatp = 1 / (2 * d) * np.ones(d)
    tetatm = 1 / (2 * d) * np.ones(d)

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility
        Jt = np.random.randint(0, d, 1)  # sample the direction

        # update the last xt
        t = i + 1
        etat = lr / np.sqrt(d * t)

        instg_j = model.gradLoss(sample_x, sample_y, l)[Jt]
        tetatm[Jt] = np.exp(-etat * d * instg_j) * tetatm[Jt]
        tetatp[Jt] = np.exp(etat * d * instg_j) * tetatp[Jt]
        tetat = np.r_[tetatm, tetatp]  # vect of size 2d !
        new_wts = tetat / np.sum(tetat)
        new_wts = z * (new_wts[0:d] - new_wts[d:])
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


def sbeg(model, X, y, epoch, l, z=1, lr=1, verbose=0):
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
    wts = [np.zeros(d)]
    tetatp = 1 / (2 * d) * np.ones(d)
    tetatm = 1 / (2 * d) * np.ones(d)
    prob = np.r_[tetatm, tetatp]

    for i in range(epoch):

        # sample
        idx = rd.randint(0, n - 1)
        sample_x = X[idx, :].reshape(1, -1)
        sample_y = np.array(y[idx])  # need an array for compatibility
        # sample an arm according to weights
        arm = np.random.choice(2 * d, 1, p=prob)
        # index corresponding to our representation
        Jt = arm * (arm <= d - 1) + (arm - d) * (arm >= d)
        # corresponding sign : + if arm <= d-1 and - else
        sgt = 1 - 2 * (arm >= d)

        # update the last xt
        t = i + 1
        etat = lr / np.sqrt(d * t)

        instg_j = model.gradLoss(sample_x, sample_y, l)[Jt]
        if arm < d:
            # sgt is > 0 and we modify the first part
            tetatm[Jt] = np.exp(- sgt * etat * d * instg_j) * tetatm[Jt]
        else:
            # sgt is < 0 then we modify the second part of weights
            tetatp[Jt] = np.exp(- sgt * etat * d * instg_j) * tetatp[Jt]

        tetat = np.r_[tetatm, tetatp]
        new_wts = (tetat / np.sum(tetat)) * (1 - etat) + etat / (2 * d)  # 1/2d is the initial draw probability
        prob = new_wts
        new_wts = z * (new_wts[0:d] - new_wts[d:])
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
