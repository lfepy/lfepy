import numpy as np
from lfepy.Helper.gauss import gauss


def dgauss(x, sigma):
    """
    Compute the derivative of the Gaussian function with respect to x.

    This function calculates the derivative of the Gaussian function, also known as the
    Gaussian derivative, which is useful in various image processing tasks such as edge
    detection and feature extraction. The derivative is computed based on the standard
    deviation of the Gaussian distribution.

    Args:
        x (float or numpy.ndarray): The point or points at which to evaluate the derivative of the Gaussian function.
        sigma (float): The standard deviation of the Gaussian distribution, which controls the width of the Gaussian curve.

    Returns:
        float or numpy.ndarray: The derivative of the Gaussian function at the given point(s). The type matches the input `x` type.

    Example:
        >>> dgauss(0, 1)
        -0.0
        >>> dgauss(np.array([0, 1, 2]), 1)
        array([-0., -0.24197072, -0.10798193])
    """
    return -x * gauss(x, sigma) / sigma ** 2