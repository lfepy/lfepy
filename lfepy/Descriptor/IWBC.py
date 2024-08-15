import numpy as np


def IWBC(image, **kwargs):
    """
    Compute Improved Weber Contrast (IWBC) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing IWBC extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.scale: Scale factor for IWBC computation. Default: 1.
    :type kwargs.scale: int

    :returns:
        - IWBC_hist: Histogram(s) of IWBC descriptors.
        - imgDesc: List of dictionaries containing IWBC descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = IWBC(image, mode='nh', scale=1)
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        B.-Q. Yang, T. Zhang, C.-C. Gu, K.-J. Wu, and X.-P. Guan,
        A novel face recognition method based on iwld and iwbc.
        Multimedia Tools and Applications 75 (2016) 6979.
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

    # Define scale-specific cell configurations
    scaleCell = {
        1: np.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1]]),
        2: np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5],
                     [5, 5], [5, 4], [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [2, 1]]),
        3: np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7],
                     [7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [7, 2], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1]])}

    # Extract scale factor or use default
    scale = options.get('scale', 1)

    # Define constants and angles for IWBC computation
    BELTA = 5
    ALPHA = 3
    EPSILON = 0.0000001
    ANGLE = 5 * np.pi / 4
    ANGLEDiff = 2 * np.pi / (scale * 8)

    # Extract central region of the image
    numNeigh = scale * 8
    x_c = image[scale:-scale, scale:-scale]
    rSize, cSize = x_c.shape
    DEx = np.zeros((rSize, cSize))
    DEy = np.zeros((rSize, cSize))
    link = scaleCell[scale]
    for n in range(numNeigh):
        corner = link[n]
        x_i = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        DEx += (x_i - x_c) * np.cos(ANGLE)
        DEy += (x_i - x_c) * np.sin(ANGLE)
        ANGLE -= ANGLEDiff

    # Compute EPSx and EPSy
    EPSx = np.arctan((ALPHA * DEx) / (x_c + BELTA))
    EPSy = np.arctan((ALPHA * DEy) / (x_c + BELTA))
    signEPSx = np.sign(EPSx)
    signEPSy = np.sign(EPSy)

    # Convert EPSx and EPSy to degrees
    EPSxDeg = EPSx * 180 / np.pi
    EPSyDeg = EPSy * 180 / np.pi
    # Compute NWM (Normalized Weber Magnitude)
    NWM = np.sqrt(EPSxDeg ** 2 + EPSyDeg ** 2)
    EPSx[EPSx == 0] = EPSILON
    # Compute NWO (Normalized Weber Orientation)
    NWO = np.arctan(EPSy / EPSx) * 180 / np.pi
    NWO[EPSx < 0] += 180
    NWO[(EPSx > 0) & (EPSy < 0)] += 360

    # Define binary maps B_x and B_y
    B_x = np.ones_like(signEPSx)
    B_x[signEPSx == 1] = 0
    B_y = np.ones_like(signEPSy)
    B_y[signEPSy == 1] = 0

    # Initialize variables for scale 2 computation
    scale2 = 1
    numNeigh = scale2 * 8
    link = scaleCell[scale2]

    # Compute LBMP (Local Binary Magnitude Pattern)
    x_c = NWM[scale2:-scale2, scale2:-scale2]
    rSize, cSize = x_c.shape
    LBMP = np.zeros((rSize, cSize))
    for i in range(numNeigh):
        corner = link[i]
        x_i = NWM[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        diff = x_i - x_c
        diff[(diff == 0) | (diff > 0)] = 1
        diff[diff < 0] = 0
        LBMP += diff * 2 ** (numNeigh - i - 1)

    # Compute IWBC_M (Magnitude Component of Improved Weber Contrast)
    IWBC_M = LBMP + B_y[scale2:-scale2, scale2:-scale2] * 2 ** numNeigh
    IWBC_M += B_x[scale2:-scale2, scale2:-scale2] * 2 ** (numNeigh + 1)

    NWO[NWO == 360] = 0
    NWO[(NWO >= 0) & (NWO < 90)] = 0
    NWO[(NWO >= 90) & (NWO < 180)] = 1
    NWO[(NWO >= 180) & (NWO < 270)] = 2
    NWO[(NWO >= 270) & (NWO < 360)] = 3

    # Convert NWO to discrete orientation bins
    x_c = NWO[scale2:-scale2, scale2:-scale2]
    LXOP = np.zeros((rSize, cSize))
    for i in range(numNeigh):
        corner = link[i]
        x_i = NWO[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        diff = ~(x_i == x_c)
        LXOP += diff * 2 ** (numNeigh - i - 1)

    IWBC_O = LXOP + B_y[scale2:-scale2, scale2:-scale2] * 2 ** numNeigh
    IWBC_O += B_x[scale2:-scale2, scale2:-scale2] * 2 ** (numNeigh + 1)

    imgDesc = [{'fea': IWBC_M}, {'fea': IWBC_O}]

    # Set bin vectors
    binVec = [np.arange(0, 2 ** (numNeigh + 2)), np.arange(0, 2 ** (numNeigh + 2))]
    options['binVec'] = binVec

    # Compute IWBC histogram
    IWBC_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            IWBC_hist.append(hh)
    IWBC_hist = np.array(IWBC_hist)
    if 'mode' in options and options['mode'] == 'nh':
        IWBC_hist = IWBC_hist / np.sum(IWBC_hist)

    return IWBC_hist, imgDesc
