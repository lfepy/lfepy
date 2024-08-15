import numpy as np


def construct_Gabor_filters(num_of_orient, num_of_scales, size1, fmax=0.25,
                            ni=np.sqrt(2), gamma=np.sqrt(2), separation=np.sqrt(2)):
    """
    Constructs a bank of Gabor filters.

    :param num_of_orient: Number of orientations.
    :type num_of_orient: int
    :param num_of_scales: Number of scales.
    :type num_of_scales: int
    :param size1: Size of the filters. Can be an integer for square filters or a tuple for rectangular filters.
    :type size1: int or tuple
    :param fmax: Maximum frequency. Default is 0.25.
    :type fmax: float, optional
    :param ni: Bandwidth parameter. Default is sqrt(2).
    :type ni: float, optional
    :param gamma: Aspect ratio. Default is sqrt(2).
    :type gamma: float, optional
    :param separation: Frequency separation factor. Default is sqrt(2).
    :type separation: float, optional

    :returns: A dictionary containing the spatial and frequency representations of the Gabor filters.
    :rtype: dict

    :example:
        >>> import matplotlib.pyplot as plt
        >>> num_of_orient = 8
        >>> num_of_scales = 5
        >>> filter_size = 31
        >>> gabor_filters = construct_Gabor_filters(num_of_orient, num_of_scales, filter_size)
        >>> fig, axes = plt.subplots(num_of_scales, num_of_orient, figsize=(20, 10))
        >>> for u in range(num_of_scales):
        ...     for v in range(num_of_orient):
        ...         ax = axes[u, v]
        ...         ax.imshow(np.real(gabor_filters['spatial'][u, v]), cmap='gray')
        ...         ax.axis('off')
        >>> plt.show()
    """
    # Initialize the filter bank
    filter_bank = {
        'spatial': np.empty((num_of_scales, num_of_orient), dtype=object),
        'freq': np.empty((num_of_scales, num_of_orient), dtype=object),
        'scales': num_of_scales,
        'orient': num_of_orient
    }

    # Check and adjust the size input
    if isinstance(size1, int):
        size1 = (size1, size1)
    elif len(size1) == 2:
        size1 = tuple(size1)
    else:
        raise ValueError("The parameter determining the size of the filters is not valid.")

    # Construct Gabor filter bank
    sigma_x = size1[1]
    sigma_y = size1[0]

    for u in range(num_of_scales):  # for each scale
        fu = fmax / (separation ** u)
        alfa = fu / gamma
        beta = fu / ni

        for v in range(num_of_orient):  # for each orientation
            theta_v = (v / num_of_orient) * np.pi
            gabor = np.zeros((2 * sigma_y, 2 * sigma_x), dtype=np.complex128)

            for x in range(-sigma_x, sigma_x):
                for y in range(-sigma_y, sigma_y):
                    xc = x * np.cos(theta_v) + y * np.sin(theta_v)
                    yc = -x * np.sin(theta_v) + y * np.cos(theta_v)
                    gabor[y + sigma_y, x + sigma_x] = (
                            (fu ** 2 / (np.pi * gamma * ni)) *
                            np.exp(-(alfa ** 2 * xc ** 2 + beta ** 2 * yc ** 2)) *
                            np.exp(1j * 2 * np.pi * fu * xc)
                    )

            filter_bank['spatial'][u, v] = gabor
            filter_bank['freq'][u, v] = np.fft.fft2(gabor)

    return filter_bank