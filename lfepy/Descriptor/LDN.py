import numpy as np
from lfepy.Helper import descriptor_LDN


def LDN(image, **kwargs):
    """
    Compute Local Difference Number (LDN) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LDN extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.mask: Mask type for LDN computation. Options: 'gaussian', 'kirsch', 'sobel', or 'prewitt'. Default: 'kirsch'.
    :type kwargs.mask: str
    :param kwargs.msize: Mask size if 'mask' is set to 'kirsch'. Default: 3.
    :type kwargs.msize: int
    :param kwargs.start: Starting sigma value if 'mask' is set to 'gaussian'. Default: 0.5.
    :type kwargs.start: float

    :returns:
        - LDN_hist: Histogram(s) of LDN descriptors.
        - imgDesc: List of dictionaries containing LDN descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = LDN(image, mode='nh', mask='kirsch', msize=3, start=0.5)
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        A.R. Rivera, J.R. Castillo, and O.O. Chae,
        Local directional number pattern for face analysis: Face and expression recognition.
        IEEE transactions on image processing 22 (2013) 1740-1752.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision
    image = np.double(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Handle keyword arguments
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    # Extract histogram mode
    if 'mode' not in options:
        options.update({'mode': 'nh'})

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    # Extract options or set defaults
    mask = options.get('mask', 'kirsch')
    msize = options.get('msize', 3)

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
