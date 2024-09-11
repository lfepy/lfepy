import numpy as np
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def MBP(image, **kwargs):
    """
    Compute Median Binary Pattern (MBP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MBP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            MBP_hist (numpy.ndarray): Histogram(s) of MBP descriptors.
            imgDesc (numpy.ndarray): MBP descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MBP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust facial expression recognition based on median ternary pattern (MTP),
        Electrical Information and Communication Technology (EICT), 2013 International Conference on, IEEE,
        2014, pp. 1-5.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for MBP computation
    link = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])
    ImgIntensity = np.zeros((rSize * cSize, link.shape[0]))

    # Compute MBP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        x_slice = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        ImgIntensity[:, n] = x_slice.reshape(-1)

    medianMat = np.median(ImgIntensity, axis=1)
    MBP = np.double(ImgIntensity > medianMat.reshape(-1, 1))
    imgDesc = np.array([int(''.join(map(str, row.astype(np.uint8))), 2) for row in MBP]).reshape(rSize, cSize)

    # Set bin vectors
    options['binVec'] = np.arange(256)

    # Compute MBP histogram
    MBP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        MBP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        MBP_hist = MBP_hist / np.sum(MBP_hist)

    return MBP_hist, imgDesc
