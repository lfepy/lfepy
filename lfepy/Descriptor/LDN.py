import numpy as np
from lfepy.Helper import descriptor_LDN
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_mask_LDN, validate_msize


def LDN(image, **kwargs):
    """
    Compute Local Difference Number (LDN) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LDN extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            mask (str): Mask type for LDN computation. Options: 'gaussian', 'kirsch', 'sobel', or 'prewitt'. Default is 'kirsch'.
            msize (int): Mask size if 'mask' is set to 'kirsch'. Default is 3.
            start (float): Starting sigma value if 'mask' is set to 'gaussian'. Default is 0.5.

    Returns:
        tuple: A tuple containing:
            LDN_hist (numpy.ndarray): Histogram(s) of LDN descriptors.
            imgDesc (list): List of dictionaries containing LDN descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LDN(image, mode='nh', mask='kirsch', msize=3, start=0.5)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        A.R. Rivera, J.R. Castillo, and O.O. Chae,
        Local Directional Number Pattern for Face Analysis: Face and Expression Recognition,
        IEEE Transactions on Image Processing,
        vol. 22, 2013, pp. 1740-1752.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    mask = validate_mask_LDN(options)
    msize = validate_msize(options)

    imgDesc = []
    options['binVec'] = []

    # Set bin vector
    uniqueBin = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17,
                          19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33,
                          34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49,
                          50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62])

    # Compute LDN descriptors
    if mask == 'gaussian':
        start = options.get('start', 0.5)
        imgDesc.append({'fea': descriptor_LDN(image, mask='gaussian', sigma=start)})
        options['binVec'].append(uniqueBin)
        imgDesc.append({'fea': descriptor_LDN(image, mask='gaussian', sigma=2 * start)})
        options['binVec'].append(uniqueBin)
        imgDesc.append({'fea': descriptor_LDN(image, mask='gaussian', sigma=3 * start)})
        options['binVec'].append(uniqueBin)

    else:
        imgDesc.append({'fea': descriptor_LDN(image, mask=mask, msize=msize)})
        options['binVec'].append(uniqueBin)

    # Compute LDN histogram
    LDN_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LDN_hist.append(hh)
    LDN_hist = np.array(LDN_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LDN_hist = LDN_hist / np.sum(LDN_hist)

    return LDN_hist, imgDesc
