import numpy as np
import math

# def proj_simplex(x):
#     if sum([abs(vi) for vi in x]) <= 1:
#         return x
#     x_s = np.sort(x, kind='quicksort')[::-1]
#     cum_s = np.cumsum(x_s)
#     res = x_s - (cum_s-1)/(np.arange(len(x))+1)
#     d0 = np.max(np.where(res>0)) + 1 #because index start at 0
#     theta = (cum_s[d0-1]-1)/d0
#
#     return np.maximum(0,x-theta)
#
# def weighted_proj_simplex(x,D):
#     """
#     x : vector, D : diag matrix
#     """
#     dx = np.abs(np.dot(D,x))
#     sorted_indices = np.argsort(-dx, kind='quicksort')
#     sx = np.cumsum(x[sorted_indices])
#     sd = np.cumsum(1/np.diag(D)[sorted_indices])
#     res = dx[sorted_indices] - (sx-1)/sd
#     d0 = np.max(np.where(res>0)[0])
#     theta = (sx[d0]-1)/sd[d0]
#
#     return np.dot(np.linalg.inv(D),np.maximum(0,dx-theta))
#
# def proj_l1(x,z=1,d=0,weighted=False):
#     if np.sum(np.abs(x)) <= z:
#         return x
#     else:
#         if weighted:
#             w = weighted_proj_simplex(abs(x)/z,d)
#         else :
#             w = proj_simplex(abs(x)/z)
#         return z*np.sign(x)*w


def proj_simplex(v, z=1):
    u = np.sort(v)
    su = np.cumsum(u)
    res = u - (su - z) / (np.arange(len(v)) + 1)
    rho = np.max(np.where(res > 0))
    theta = (su[rho]-z)/(rho+1)

    w = np.clip(v - theta, 0, np.max(v-theta))
    return w


def proj_l1(vect, z=1):
    v = np.abs(vect)
    if np.sum(v) > z :  # outside of the l1-ball
        u = proj_simplex(v, z)
        vect = np.sign(vect)*u
    return vect


def weighted_proj_l1(vect, w, z=1):
    """
    Weighted projection on the l1-ball of
    - vect of size (n)
    - using weights w
    - and the radius z of the l1-ball considered
    """
    if np.sum(np.abs(vect))>z & math.isinf(z) == False & math.isnan(z) == False:
        v = np.abs(vect*w)
        ind = np.argsort(-v)
        sum_vect = np.cumsum(np.abs(vect)[ind])
        sum_w = np.cumsum(1/w[ind])
        rho = np.max(np.where(v[ind]>(sum_vect - z)/sum_w))
        theta = (sum_vect[rho] - z) / sum_w[rho]
        vect = np.sign(vect)*np.clip(np.abs(vect) - theta/w, 0, np.max(np.abs(vect) - theta/w))
    return vect


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
