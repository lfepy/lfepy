import numpy as np
from lfepy.Helper import descriptor_PHOG, phogDescriptor_hist
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_bin, validate_angle, validate_L


def PHOG(image, **kwargs):
    """
    Compute Pyramid Histogram of Oriented Gradients (PHOG) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing PHOG extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            bin (int): Number of bins for the histogram. Default is 8.
            angle (int): Range of gradient angles in degrees. Default is 360.
            L (int): Number of pyramid levels. Default is 2.

    Returns:
        tuple: A tuple containing:
            PHOG_hist (numpy.ndarray): Histogram of PHOG descriptors.
            imgDesc (list of dicts): List of dictionaries containing PHOG descriptors for each pyramid level.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = PHOG(image, mode='nh', bin=8, angle=360, L=2)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        A. Bosch, A. Zisserman, and X. Munoz,
        Representing shape with a spatial pyramid kernel,
        Proceedings of the 6th ACM international conference on Image and video retrieval, ACM,
        2007, pp. 401-408.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    bin = validate_bin(options)
    angle = validate_angle(options)
    L = validate_L(options)

    # Define the region of interest (ROI)
    roi = [0, image.shape[0], 0, image.shape[1]]

    # Compute PHOG descriptors
    _, bh_roi, bv_roi = descriptor_PHOG(image, bin, angle, L, roi)

    # Collect descriptors
    imgDesc = [{'fea': bh_roi}, {'fea': bv_roi}]

    # Compute PHOG histogram
    PHOG_hist = phogDescriptor_hist(bh_roi, bv_roi, L, bin)

    # Normalize the histogram if required
    if 'mode' in options and options['mode'] == 'nh':
        PHOG_hist = PHOG_hist / np.sum(PHOG_hist)

    return PHOG_hist, imgDesc