import numpy as np


def proj_simplex(x):
    if sum([abs(vi) for vi in x]) <= 1:
        return x
    x_s = np.sort(x, kind='quicksort')[::-1]
    cum_s = np.cumsum(x_s)
    res = x_s - (cum_s - 1) / (np.arange(len(x)) + 1)
    d0 = np.max(np.where(res > 0)) + 1  # because index start at 0
    theta = (cum_s[d0 - 1] - 1) / d0

    return np.maximum(0, x - theta)


def proj_l1(x, z=1):
    if np.sum(np.abs(x)) <= z:
        return x
    else:
        w = proj_simplex(abs(x) / z)
    return z * np.sign(x) * w


def weighted_proj_l1(x, d, z=1):
    """
    Weighted projection on the l1-ball of
    :param x: (size n) the input vector
    :param d: (size n) weights
    :param z: (int) radius of the L1 ball considered
    """
    if np.sum(np.abs(x)) <= z:
        return x
    else:
        proj = weighted_proj_simplex(abs(x) / z, np.diag(d))
    return z * np.sign(x) * proj


def weighted_proj_simplex(x, D):
    """
    :param x: a vector
    :param D: a matrix
    """

    dx = np.abs(np.dot(D, x))
    sorted_indices = np.argsort(-dx, kind='quicksort')
    sx = np.cumsum(x[sorted_indices])
    sd = np.cumsum(1 / np.diag(D)[sorted_indices])
    res = dx[sorted_indices] - (sx - 1) / sd
    d0 = np.max(np.where(res > 0)[0])
    theta = (sx[d0] - 1) / sd[d0]

    return np.dot(np.linalg.inv(D), np.maximum(0, dx - theta))


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
    p3 = weighted_proj_l1(x, np.diag(D))
    print(D)
    print(x)
    print(p3)
