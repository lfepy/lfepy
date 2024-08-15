import numpy as np
from lfepy.Helper.gauss import gauss


def dgauss(x, sigma):
    """
    Compute the derivative of the Gaussian (normal) distribution with respect to x.

    :param x: The point or points at which to evaluate the derivative.
    :type x: float or numpy.ndarray
    :param sigma: The standard deviation of the Gaussian distribution.
    :type sigma: float

    :returns: The derivative of the Gaussian function at the given point(s).
    :rtype: float or numpy.ndarray

    :example:
        >>> dgauss(0, 1)
        -0.0
        >>> dgauss(np.array([0, 1, 2]), 1)
        array([-0., -0.24197072, -0.10798193])
    """
    return -x * gauss(x, sigma) / sigma ** 2