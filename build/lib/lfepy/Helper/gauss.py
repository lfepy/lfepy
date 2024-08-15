import numpy as np


def gauss(x, sigma):
    """
    Calculate the value of the Gaussian (normal) distribution at a given point.

    :param x: The point or points at which to evaluate the Gaussian function.
    :type x: float or numpy.ndarray
    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float

    :returns: The value(s) of the Gaussian function at the given point(s).
    :rtype: float or numpy.ndarray

    :example:
        >>> gauss(0, 1)
        0.3989422804014327
        >>> gauss(np.array([0, 1, 2]), 1)
        array([0.39894228, 0.24197072, 0.05399097])
    """
    # Compute the Gaussian function value
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))