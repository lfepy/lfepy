import numpy as np
from lfepy.Helper import descriptor_LBP, get_mapping
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_mappingType, validate_radius


def LBP(image, **kwargs):
    """
    Compute Local Binary Patterns (LBP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LBP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            radius (int): Radius for LBP computation. Default is 1.
            mappingType (str): Type of mapping for LBP computation. Options: 'full', 'ri', 'u2', 'riu2'. Default is 'full'.

    Returns:
        tuple: A tuple containing:
            LBP_hist (numpy.ndarray): Histogram(s) of LBP descriptors.
            imgDesc (numpy.ndarray): LBP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` or `mappingType` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LBP(image, mode='nh', radius=1, mappingType='full')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Ojala, M. Pietikainen, and T. Maenpaa,
        Multi-resolution gray-scale and rotation invariant texture classification with local binary patterns,
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 24, pp. 971-987, 2002.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    options, radius, neighbors = validate_radius(options)
    options, mapping = validate_mappingType(options, radius, neighbors)

    mode = options['mode']

    # Extract LBP descriptors
    _, imgDesc = descriptor_LBP(image, radius, neighbors, mapping, mode)

    # Compute LBP histogram
    LBP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LBP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LBP_hist = LBP_hist / np.sum(LBP_hist)

    return LBP_hist, imgDesc
