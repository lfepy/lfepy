import numpy as np


def LMP(image, **kwargs):
    """
    Compute Local Monotonic Pattern (LMP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LMP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LMP_hist (numpy.ndarray): Histogram(s) of LMP descriptors.
            imgDesc (numpy.ndarray): LMP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LMP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Mohammad, and M.L. Ali,
        Robust Facial Expression Recognition Based on Local Monotonic Pattern (LMP),
        Computer and Information Technology (ICCIT), 2011 14th International Conference on, IEEE,
        2011, pp. 572-576.
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

    # Define link list for LMP computation
    link = [[[3, 4], [3, 5]],
            [[2, 4], [1, 5]],
            [[2, 3], [1, 3]],
            [[2, 2], [1, 1]],
            [[3, 2], [3, 1]],
            [[4, 2], [5, 1]],
            [[4, 3], [5, 3]],
            [[4, 4], [5, 5]]]

    # Compute LMP descriptors
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape

    # Initialize LMP descriptor matrix
    imgDesc = np.zeros((rSize, cSize))

    for n in range(8):
        corner = link[n]
        x_i1 = image[corner[0][0] - 1:corner[0][0] + rSize - 1, corner[0][1] - 1:corner[0][1] + cSize - 1]
        x_i2 = image[corner[1][0] - 1:corner[1][0] + rSize - 1, corner[1][1] - 1:corner[1][1] + cSize - 1]
        imgDesc += np.double((((x_i1 - x_c) >= 0) & ((x_i2 - x_i1) >= 0)) * 2 ** (8 - n - 1))

    # Set bin vectors
    options['binVec'] = np.arange(256)

    # Compute LMP histogram
    LMP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LMP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LMP_hist = LMP_hist / np.sum(LMP_hist)

    return LMP_hist, imgDesc
