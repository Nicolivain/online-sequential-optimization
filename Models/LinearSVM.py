import numpy as np


class LinearSVM:
    def __init__(self, m) -> None:
        self.w = np.random.rand(m)

    def loss(self, a, b, l):
        """
        Compute the loss for the data and provided parameters
        a (nxm) : data
        b (n) : labels
        l (float ): regularization parameter (lambda)
        """

        temp = 1 - np.dot(a, self.w) * b
        temp[np.where(temp < 0)] = 0
        return np.mean(temp) + l / 2 * np.linalg.norm(self.w)

    def gradLoss(self, a, b, l):
        """
        Compute the gradient of the loss wrt the current parameters and data
        x (m) : params
        a (nxm) : data
        b (n) : labels
        l (float) : regularization parameter (lambda)"""
        temp = 1 - np.dot(a, self.w)
        if b.shape != ():
            grad = - (np.repeat(b[:, np.newaxis], a.shape[1], 1)) * a  # reshape b to nxm to use term-by-term multiplication
        else:
            grad = -b * a  # no need for repeat if n=1
        grad[np.where(temp < 0)] = 0
        return np.sum(grad, 0) + l * self.w

    def predict(self, data):
        """
        predict values using coeff x and new data data
        data (list) : the data for prediction
        """

        return - np.sign(data.dot(self.w)) # TODO : check here why - ? 
