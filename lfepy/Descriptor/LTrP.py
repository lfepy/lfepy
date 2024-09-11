import numpy as np
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LTrP(image, **kwargs):
    """
    Compute Local Transitional Pattern (LTrP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LTrP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LTrP_hist (numpy.ndarray): Histogram(s) of LTrP descriptors.
            imgDesc (numpy.ndarray): LTrP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LTrP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Jabid, and O. Chae,
        Local Transitional Pattern: A Robust Facial Image Descriptor for Automatic Facial Expression Recognition,
        Proc. International Conference on Computer Convergence Technology, Seoul, Korea,
        2011, pp. 333-44.

        T. Jabid, and O. Chae,
        Facial Expression Recognition Based on Local Transitional Pattern,
        International Information Institute (Tokyo), Information,
        15 (2012) 2007.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define link list for LTrP computation
    link_list = [[[4, 4], [5, 5]], [[4, 3], [5, 3]], [[4, 2], [5, 1]], [[3, 2], [3, 1]],
                 [[2, 2], [1, 1]], [[2, 3], [1, 3]], [[2, 4], [1, 5]], [[3, 4], [3, 5]]]

    # Initialize variables
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape
    imgDesc = np.zeros_like(x_c)

    # Compute LTrP descriptors
    for n, corners in enumerate(link_list):
        corner1, corner2 = corners
        x_p1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_p2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        imgDesc += np.logical_xor((x_p1 - x_c) >= 0, (x_p2 - x_c) >= 0) * 2**(len(link_list) - n - 1)

    # Set bin vectors
    options['binVec'] = np.arange(256)

    # Compute LTrP histogram
    LTrP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LTrP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LTrP_hist = LTrP_hist / np.sum(LTrP_hist)

    return LTrP_hist, imgDesc
