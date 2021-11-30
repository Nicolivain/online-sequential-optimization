import numpy as np
from copy import deepcopy


def proj_l1(vect):
    """
    Projects the vector vect on the L1-ball
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
        if d0 > len(vect):
            break

    theta = 1/d0 * (sum(topk) - 1)

    soft_threshold = [max(vi-theta, 0) for vi in vect]
    return soft_threshold


if __name__ == '__main__':
    # Unit Testing
    x = np.random.rand(10)
    p = proj_l1(x)
    print(sum([abs(vi) for vi in p]))
