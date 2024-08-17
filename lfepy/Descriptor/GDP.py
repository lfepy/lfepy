import numpy as np
from scipy.signal import convolve2d
from lfepy.Helper import get_mapping


def GDP(image, **kwargs):
    """
    Compute Gradient Directional Pattern (GDP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing GDP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            mask (str): Mask type for gradient computation. Options: 'sobel', 'prewitt'. Default is 'sobel'.
            t (float): Threshold value for gradient angle difference. Default is 22.5.

    Returns:
        tuple: A tuple containing:
            GDP_hist (numpy.ndarray): Histogram(s) of GDP descriptors.
            imgDesc (numpy.ndarray): GDP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` or `mask` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = GDP(image, mode='nh', mask='sobel', t=22.5)

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Ahmed,
        "Gradient directional pattern: a robust feature descriptor for facial expression recognition",
        in *Electronics letters*,
        vol. 48, no. 23, pp. 1203-1204, 2012.

        W. Chu,
        Facial expression recognition based on local binary pattern and gradient directional pattern,
        in Green Computing and Communications (GreenCom), 2013 IEEE and Internet of Things (iThings/CPSCom), IEEE,
        2013, pp. 1458-1462.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision if needed
    if image.dtype != np.float64:
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

    # Extract mask method
    if 'mask' not in options:
        options['mask'] = 'sobel'
        t = 22.5
    elif options['mask'] == 'sobel' and 't' not in options:
        t = 22.5
    elif options['mask'] == 'prewitt' and 't' not in options:
        t = 330
    else:
        if 't' in options:
            t = options['t']
        else:
            t = 22.5

    # Validate the mask
    valid_masks = ['sobel', 'prewitt']
    if options['mask'] not in valid_masks and 'mask' in options:
        raise ValueError(f"Invalid mask '{options['mask']}'. Valid masks are {valid_masks}.")

    EPSILON = 0.0000001

    # Define masks for sobel or prewitt
    if options['mask'] == 'sobel':
        maskA = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        maskB = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        link = np.array([[1, 2], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3], [2, 3], [1, 3]])
    elif options['mask'] == 'prewitt':
        maskA = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        maskB = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        link = np.array([[3, 1], [3, 2], [3, 3], [2, 3], [1, 3], [1, 2], [1, 1], [2, 1]])

    Gx = convolve2d(image, maskA, 'same')
    Gy = convolve2d(image, maskB, 'same')
    angles = np.arctan2(Gy, Gx + EPSILON)
    angles = np.degrees(angles) + 90

    x_c = angles[1:-1, 1:-1]
    rSize, cSize = x_c.shape
    GDPdecimal = np.zeros((rSize, cSize))
    for n in range(link.shape[0]):
        corner = link[n]
        x_i = angles[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        GDPdecimal += np.double(((x_i - x_c) <= t) & ((x_i - x_c) >= -t)) * 2 ** (8 - n - 1)

    if options['mask'] == 'prewitt':
        mapping = get_mapping(8, 'u2')
        for r in range(GDPdecimal.shape[0]):
            for c in range(GDPdecimal.shape[1]):
                GDPdecimal[r, c] = mapping['table'][int(GDPdecimal[r, c])]
        binNum = mapping['num']
    else:
        binNum = 256

    imgDesc = GDPdecimal

    # Set bin vectors
    options['binVec'] = np.arange(binNum)

    # Compute GDP histogram
    GDP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        GDP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        GDP_hist = GDP_hist / np.sum(GDP_hist)

    return GDP_hist, imgDesc
