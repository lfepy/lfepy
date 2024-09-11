import numpy as np
from scipy.signal import convolve2d
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_epsi


def LDTP(image, **kwargs):
    """
    Compute Local Directional Texture Pattern (LDTP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LDTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            epsi (int): Threshold value for texture difference. Default is 15.

    Returns:
        tuple: A tuple containing:
            LDTP_hist (numpy.ndarray): Histogram(s) of LDTP descriptors.
            imgDesc (numpy.ndarray): LDTP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LDTP(image, mode='nh', epsi=15)

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        A.R. Rivera, J.R. Castillo, and O. Chae,
        Local Directional Texture Pattern Image Descriptor,
        Pattern Recognition Letters,
        vol. 51, 2015, pp. 94-100.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    epsi = validate_epsi(options)

    # Define Kirsch Masks
    Kirsch = [np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute mask responses
    maskResponses = np.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch):
        maskResponses[:, :, i] = convolve2d(image, kirsch_mask, mode='same')

    maskResponsesAbs = np.abs(maskResponses) / 8

    ind = np.argsort(maskResponsesAbs[1:-1, 1:-1, :], axis=2)
    prin1 = ind[:, :, 0]
    prin2 = ind[:, :, 1]

    linkList = [[[2, 3], [2, 1]], [[1, 3], [3, 1]], [[1, 2], [3, 2]], [[1, 1], [3, 3]],
                [[2, 1], [2, 3]], [[3, 1], [1, 3]], [[3, 2], [1, 2]], [[3, 3], [1, 1]]]

    x_c = image[1:-1, 1:-1]
    rSize, cSize = x_c.shape
    diffIntensity = np.zeros((rSize, cSize, 8))

    for n, link in enumerate(linkList):
        corner1 = link[0]
        corner2 = link[1]
        x_1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        diffIntensity[:, :, n] = x_1 - x_2

    diffResP = np.zeros((rSize, cSize))
    diffResN = np.zeros((rSize, cSize))
    for d in range(8):
        diffResIns = diffIntensity[:, :, d]
        diffResP[prin1 == d] = diffResIns[prin1 == d]
        diffResN[prin2 == d] = diffResIns[prin2 == d]

    diffResP[np.logical_and(diffResP <= epsi, diffResP >= -epsi)] = 0
    diffResP[diffResP < -epsi] = 1
    diffResP[diffResP > epsi] = 2
    diffResN[np.logical_and(diffResN <= epsi, diffResN >= -epsi)] = 0
    diffResN[diffResN < -epsi] = 1
    diffResN[diffResN > epsi] = 2

    imgDesc = 16 * prin1 + 4 * diffResP + diffResN

    # Define unique bins for histogram
    uniqueBin = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 32, 33,
                          34, 36, 37, 38, 40, 41, 42, 48, 49, 50, 52, 53, 54, 56, 57, 58, 64, 65,
                          66, 68, 69, 70, 72, 73, 74, 80, 81, 82, 84, 85, 86, 88, 89, 90, 96, 97, 98,
                          100, 101, 102, 104, 105, 106, 112, 113, 114, 116, 117, 118, 120, 121, 122])

    # Set binVec option
    options['binVec'] = uniqueBin

    # Compute LDTP histogram
    LDTP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LDTP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LDTP_hist = LDTP_hist / np.sum(LDTP_hist)

    return LDTP_hist, imgDesc