import numpy as np
from copy import deepcopy


def proj_simplex(vect):
    """
    Projects the vector vect on the simplex
    :param vect: vector of size (n)
    """

    if sum([abs(vi) for vi in vect]) == 1:
        return vect

    nvc  = list(deepcopy(vect))
    topk = [nvc.pop(np.argmax(vect).min())]  # min in case of equality eg len(argmax)>1
    d0 = 1
    while nvc[np.argmax(nvc)] > 1/d0 * (sum(topk) - 1):
        topk.append(nvc.pop(np.argmax(nvc).min()))
        d0 += 1
        if d0 > len(vect) - 1:
            break

    theta = 1/d0 * (sum(topk) - 1)

    soft_threshold = [vi/abs(vi) * max(vi-theta, 0) if vi != 0 else 0 for vi in vect]  # we make sure to keep the sign
    return np.array(soft_threshold)

def proj_l1(vect, z) : 
    """
    Projects the vector vect on the L1-ball
    :param vect: vector of size (n)
    :param z: radius of the l1-ball considered
    """

    if (np.abs(vect) <= 1).sum():
        return vect
    wstar = proj_simplex(np.abs(vect) / z)
    return np.sign(vect) * wstar # produit terme à terme

def weighted_proj_simplex(vect, weight):
    """
    Projects the vector vect on the simplex
    :param vect: vector of size (nxn)
    """

    if sum([abs(vi) for vi in vect]) == 1:
        return vect

    Dx = weight.dot(vect)
    nvc  = list(deepcopy(Dx))
    topk = [nvc.pop(np.argmax(Dx).min())]  # min in case of equality eg len(argmax)>1
    d0 = 1
    Dsorted = np.sort(np.diag(weight))
    sumd0 = np.sum(Dsorted[0:d0])
    while nvc[np.argmax(nvc)] > 1/sumd0 * (sum(topk) - 1):
        topk.append(nvc.pop(np.argmax(nvc).min()))
        d0 += 1
        sumd0 = np.sum(Dsorted[0:d0])
        if d0 > len(Dx) - 1:
            break

    theta = 1/sumd0 * (sum(topk) - 1)

    soft_threshold = [vi/abs(vi) * max(vi-theta, 0) if vi != 0 else 0 for vi in Dx]  # we make sure to keep the sign
    return np.linalg.inv(weight).dot(np.array(soft_threshold))


if __name__ == '__main__':
    # Unit Testing
    x = np.random.rand(10) * 2 - 1
    p = proj_simplex(x)
    p2 = proj_l1(x, 1)
    print(x)
    print(p)
    print(sum([abs(vi) for vi in p]))
    print(p2)
    print(np.abs(p2) < np.ones(10))
    print(np.abs(p2) < 1)
    d = np.random.rand(10)
    D = np.diag(d)
    p3 = weighted_proj_simplex(x, D)
    print(D)
    print(x)
    print(p3)
