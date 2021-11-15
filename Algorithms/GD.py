"""
This file contains functions for gradient descent algorithm applied at the SVM problem
"""
import numpy as np
from utils import LinearSVM

def GradientDescent(model, X, y, lr, epoch, l, verbose=0):
    """
    Gradient descent algorithms applied with the CO pb il loss and uses tjhe gradloss function to update parameters
    X (nxm) : data
    y (n) : labels 
    lr (float) : learning rate
    epoch (int) : maximum number of iteration of the algorithm
    l (float) : regularization parameter (lambda)"""
    n, m = X.shape
    params = np.random.rand(m)
    losses = []
    for i in range(epoch):
        params -= lr * model.gradLoss(params, X, y, l)
        current_loss = model.loss(params, X, y, l)
        losses += [current_loss]
        if verbose > 0  and i % verbose == 0:
            print(f"epoch {i} : loss = {current_loss}")
    return params, losses