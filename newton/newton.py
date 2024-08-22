import numpy as np


def derivative1(x, func):
    """function returning the derivative of a function at point x."""
    # f' = f(x+h) - f(x) / h
    h = 0.00001
    return (func(x + h) - func(x)) / h


def derivative2(x, func):
    """function returning the second derivative of a function at point x."""
    h = 0.00001
    # f'' = f'(x+h) - f'(x) / h
    return (derivative1(x + h, func) - derivative1(x, func)) / h


def update(x_t, func):
    """helper function for performing Netwon's method."""
    x_t1 = x_t - (derivative1(x_t, func) / derivative2(x_t, func))
    return x_t1


def optimize(x_0, func, tol=1e-3, max_iter=10000):
    """Newton's method for optimization function."""
    x_t = x_0
    x_t1 = update(x_t, func)
    print(x_t, x_t1)
    iter = 0
    while (np.abs(x_t - x_t1) < tol) and (iter<max_iter):
        x_t = x_t1
        x_t1 = update(x_t, func)
        iter += 1
    return x_t1
