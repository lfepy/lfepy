import numpy as np
from lfepy.Helper import descriptor_LPQ


def LPQ(image, **kwargs):
    """
    Compute Local Phase Quantization (LPQ) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LPQ extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            windowSize (int): Size of the sliding window for LPQ. Default is 5.

    Returns:
        tuple: A tuple containing:
            LPQ_hist (numpy.ndarray): Histogram of LPQ descriptors.
            imgDesc (numpy.ndarray): LPQ descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LPQ(image, mode='nh', windowSize=5)

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        V. Ojansivu, and J. Heikkilä,
        Blur Insensitive Texture Classification Using Local Phase Quantization,
        International Conference on Image and Signal Processing, Springer,
        2008, pp. 236-243.

        A. Dhall, A. Asthana, R. Goecke, and T. Gedeon,
        Emotion Recognition Using PHOG and LPQ Features,
        Automatic Face & Gesture Recognition and Workshops (FG 2011), IEEE,
        2011, pp. 878-883.
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

    wSz = options.get('windowSize', 5)

    imgDesc, _ = descriptor_LPQ(image, wSz)

    options['binVec'] = np.arange(256)

    # Compute LPQ histogram
    LPQ_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LPQ_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LPQ_hist = LPQ_hist / np.sum(LPQ_hist)

    return LPQ_hist, imgDesc
