import numba
from numba.typed import List
from numpy import cos, sqrt, pi, e, exp
import numpy as np

"""
Continuous benchmark functions wrappers to speed up calculations
"""

@numba.jit(nopython=True)
def sphere__(solution=None):
    return np.sum(solution ** 2)

@numba.jit(nopython=True)
def elliptic__(solution):
    dim = len(solution)
    result = 0.0
    for i in range(0, dim):
        result += (10 ** 6) ** (i / (dim - 1)) * solution[i] ** 2
    return result


@numba.jit(nopython=True)
def rastrigin__(solution=None):
    return np.sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)


@numba.jit(nopython=True)
def ackley__(solution=None):
    return -20 * exp(-0.2 * sqrt(np.sum(solution ** 2) / len(solution))) - exp(
        np.sum(cos(2 * pi * solution)) / len(solution)) + 20 + e


@numba.jit(nopython=True)
def schwefel__(solution):
    size = len(solution)
    sumspace = List()
    [sumspace.append(np.sum(solution[:i]) ** 2) for i in range(1, size)]
    return sum(sumspace)

@numba.jit(nopython=True)
def rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result
