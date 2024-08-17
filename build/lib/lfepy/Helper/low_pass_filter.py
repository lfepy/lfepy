import numpy as np


def low_pass_filter(size, cutoff, n):
    """
    Creates a low-pass Butterworth filter.

    Args:
        size (tuple of int or int): The size of the filter. If a single integer is provided, the filter will be square with that size.
        cutoff (float): The cutoff frequency for the filter. Must be between 0 and 0.5.
        n (int): The order of the Butterworth filter. Must be an integer greater than or equal to 1.

    Returns:
        np.ndarray: The low-pass Butterworth filter in the frequency domain.

    Raises:
        ValueError: If `cutoff` is not in the range [0, 0.5], or if `n` is not an integer greater than or equal to 1.

    Example:
        >>> filter_size = (256, 256)
        >>> cutoff_frequency = 0.1
        >>> order = 2
        >>> lp_filter = low_pass_filter(filter_size, cutoff_frequency, order)
        >>> print(lp_filter.shape)
        (256, 256)
    """
    # Validate input parameters
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('Cutoff frequency must be between 0 and 0.5')

    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be an integer >= 1')

    # Set filter size
    if isinstance(size, int):
        rows = cols = size
    else:
        rows, cols = size

    # Create coordinate grid
    xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)

    # Calculate Butterworth filter
    f = 1 / (1 + (radius / cutoff) ** (2 * n))

    # Shift the filter to center it
    return np.fft.ifftshift(f)