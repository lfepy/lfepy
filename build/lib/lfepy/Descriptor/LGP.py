import numpy as np


def LGP(image, **kwargs):
    """
    Compute Local Gradient Pattern (LGP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGP_hist (numpy.ndarray): Histogram(s) of LGP descriptors.
            imgDesc (numpy.ndarray): LGP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Local Gradient Pattern-A Novel Feature Representation for Facial Expression Recognition,
        Journal of AI and Data Mining 2,
        (2014), pp. 33-38.
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

    # Compute binary patterns for four pairs of pixels
    a1a3 = np.double((image[:-2, :-2]) - (image[2:, 2:]) > 0)
    a2a4 = np.double((image[:-2, 2:]) - (image[2:, :-2]) > 0)
    path1 = a1a3 * 2 + a2a4 * 1

    b1b3 = np.double((image[:-2, 1:-1]) - (image[2:, 1:-1]) > 0)
    b2b4 = np.double((image[1:-1, 2:]) - (image[1:-1, :-2]) > 0)
    path2 = b1b3 * 2 + b2b4 * 1 + 4

    # Combine paths to form the final descriptor
    imgDesc = path1 + path2

    # Set bin vectors
    options['binVec'] = np.arange(4, 11)

    # Compute LGP histogram
    LGP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LGP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LGP_hist = LGP_hist / np.sum(LGP_hist)

    return LGP_hist, imgDesc
