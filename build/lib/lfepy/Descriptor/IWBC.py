import numpy as np
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def IWBC(image, **kwargs):
    """
    Compute Improved Weber Contrast (IWBC) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing IWBC extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            scale (int): Scale factor for IWBC computation. Default is 1.

    Returns:
        tuple: A tuple containing:
            IWBC_hist (numpy.ndarray): Histogram(s) of IWBC descriptors.
            imgDesc (list): List of dictionaries containing IWBC descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = IWBC(image, mode='nh', scale=1)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        B.-Q. Yang, T. Zhang, C.-C. Gu, K.-J. Wu, and X.-P. Guan,
        A novel face recognition method based on IWLD and IWBC,
        Multimedia Tools and Applications,
        vol. 75, pp. 6979, 2016.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

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
