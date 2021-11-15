"""
file with usefull functions
"""

import numpy as np

class LinearSVM():
    def __init__(self) -> None:
        pass

    def accuracy(self, y_pred, y_true):
        """
        Compute the accuracy of the provided predictions
        y_pred (n) : prediciton
        y_true (n) : true value to predict
        """
        if len(y_pred) != len(y_true):
            raise KeyError("prediction and truth must be the same size")
        return(np.sum(y_pred==y_true)/len(y_true))

    def loss(self, x, a, b, l):
        """
        Compute the loss for the data and provided parameters
        x (m) : params 
        a (nxm) : data
        b (n) : labels
        l (float ): regularization parameter (lambda)"""
        temp = 1 - np.dot(a, x) * b
        temp[np.where(temp < 0)] = 0
        return np.mean(temp) + l/2*np.linalg.norm(x)
        
    def gradLoss(self, x, a, b, l, ):
        """
        Compute the gradient of the loss wrt the current parameters and data
        x (m) : params 
        a (nxm) : data
        b (n) : labels
        l (float) : regularization parameter (lambda)"""
        temp = 1 - np.dot(a, x)
        grad = - (np.repeat(b[:, np.newaxis], a.shape[1], 1)) * a # reshape b to nxm to use term-by-term multiplication
        grad[np.where(temp < 0)] = 0
        return np.sum(grad, 0) + l * x

    def predict(self, x, data):
        """
        predict values using coeff x and new data data
        /!\ return values in -1/1 format instead of -1/0 as labels
        """
        return -1 * (data.dot(x) < 0)