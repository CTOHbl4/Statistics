import numpy as np


def get_ito(a, b, start, length, dt=1):
    '''
    a, b are functions of x and t, return same shape as start.
    '''
    x = start
    curr = start
    for ii in range(1, length):
        curr = curr + np.random.normal(loc=a(curr, ii*dt), scale=b(curr, ii*dt), size=start.shape)
        x = np.dstack([x, curr])

    return x


def get_hist(dX, num):
    x = np.sort(dX, -1)
    Fx = np.arange(3*num)
    return x, np.array(Fx)
