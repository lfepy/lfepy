import numpy as np


def roundn(x, n):
    """
    Round a number or array of numbers to a specified number of decimal places.

    The function rounds the input `x` to `n` decimal places. If `n` is negative, it rounds to the left of the decimal point.
    If `n` is zero, it rounds to the nearest integer.

    Args:
        x (float or array-like): The number or array of numbers to be rounded.
        n (int): The number of decimal places to round to. If `n` is negative, `x` is rounded to the left of the decimal point.
                 If `n` is zero, `x` is rounded to the nearest integer.

    Returns:
        float or array-like: The rounded number or array of numbers.

    Examples:
        >>> roundn(123.456, 2)
        123.46
        >>> roundn(123.456, -1)
        120.0
        >>> roundn(123.456, 0)
        123.0
    """
    # Check if n is negative
    if n < 0:
        # Calculate the power of 10 to shift the decimal point left
        p = 10 ** -n
        # Multiply x by p, round to the nearest integer, then divide by p to shift the decimal point back
        x = np.round(p * x) / p
    elif n > 0:
        # Calculate the power of 10 to shift the decimal point right
        p = 10 ** n
        # Divide x by p, round to the nearest integer, then multiply by p to shift the decimal point back
        x = p * np.round(x / p)
    else:
        # If n is zero, round x to the nearest integer
        x = np.round(x)

    return x