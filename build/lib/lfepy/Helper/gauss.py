import numpy as np


def gauss(x, sigma):
    """
    Calculate the value of the Gaussian (normal) distribution at a given point.

    Args:
        x (float or numpy.ndarray): The point or points at which to evaluate the Gaussian function.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float or numpy.ndarray: The value(s) of the Gaussian function at the given point(s).

    Example:
        >>> gauss(0, 1)
        0.3989422804014327
        >>> gauss(np.array([0, 1, 2]), 1)
        array([0.39894228, 0.24197072, 0.05399097])
    """
    # Compute the Gaussian function value
    return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))