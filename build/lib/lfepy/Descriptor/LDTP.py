import numpy as np
from scipy.signal import convolve2d


def LDTP(image, **kwargs):
    """
    Compute Local Directional Texture Pattern (LDTP) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LDTP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.epsi: Threshold value for texture difference. Default: 15.
    :type kwargs.epsi: int

    :returns:
        - LDTP_hist: Histogram(s) of LDTP descriptors.
        - imgDesc: LDTP descriptors.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = LDTP(image, mode='nh', epsi=15)
        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        A.R. Rivera, J.R. Castillo, and O. Chae,
        Local directional texture pattern image descriptor.
        Pattern Recognition Letters 51 (2015) 94-100.
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

    # Extract threshold value for texture difference or use default
    epsi = options.get('epsi', 15)

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