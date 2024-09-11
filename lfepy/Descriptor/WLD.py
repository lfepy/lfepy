import numpy as np
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_T, validate_N, validate_scaleTop


def WLD(image, **kwargs):
    """
    Compute Weber Local Descriptor (WLD) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing WLD extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            T (int): Number of bins for gradient orientation. Default is 8.
            N (int): Number of bins for differential excitation. Default is 4.
            scaleTop (int): Number of scales to consider for WLD computation. Default is 1.

    Returns:
        tuple: A tuple containing:
            WLD_hist (numpy.ndarray): Histogram of WLD descriptors.
            imgDesc (list of dicts): List of dictionaries containing WLD descriptors for each scale.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = WLD(image, mode='nh', T=8, N=4, scaleTop=1)

        >>> plt.imshow(imgDesc[0]['fea']['GO'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()
        >>> plt.imshow(imgDesc[1]['fea']['DE'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        S. Li, D. Gong, and Y. Yuan,
        Face recognition using Weber local descriptors.,
        Neurocomputing,
        122 (2013) 272-283.

        S. Liu, Y. Zhang, and K. Liu,
        Facial expression recognition under partial occlusion based on Weber Local Descriptor histogram and decision fusion,
        Control Conference (CCC), 2014 33rd Chinese, IEEE,
        2014, pp. 4664-4668.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    T = validate_T(options)
    N = validate_N(options)
    scaleTop = validate_scaleTop(options)

    scaleCell = {(1, 1): np.array([[1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1]]),
                 (1, 2): np.array([[3, 2], [1, 2], [2, 1], [2, 3]]),
                 (2, 1): np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5],
                                   [5, 5], [5, 4], [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [2, 1]]),
                 (2, 2): np.array([[5, 3], [1, 3], [3, 1], [3, 5]]),
                 (3, 1): np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [2, 7],
                                   [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [7, 6], [7, 5], [7, 4],
                                   [7, 3], [7, 2], [7, 1], [6, 1], [5, 1], [4, 1], [3, 1], [2, 1]]),
                 (3, 2): np.array([[7, 4], [1, 4], [4, 1], [4, 7]])}

    BELTA = 5
    ALPHA = 3
    EPSILON = 1e-7

    imgDescs = []

    # Compute WLD descriptors
    for scale in range(1, scaleTop + 1):
        numNeigh = scale * 8
        x_c = image[scale:-scale, scale:-scale]
        rSize, cSize = x_c.shape
        link1 = scaleCell[(scale, 1)]
        V00 = np.zeros_like(x_c)

        for corner in link1:
            x_i = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
            V00 += x_i

        V00 -= numNeigh * x_c
        imgDE = np.degrees(np.arctan(ALPHA * V00 / (x_c + BELTA))) + 90

        link2 = scaleCell[(scale, 2)]
        V04 = (image[link2[2, 0] - 1:link2[2, 0] + rSize - 1, link2[2, 1] - 1:link2[2, 1] + cSize - 1] -
               image[link2[3, 0] - 1:link2[3, 0] + rSize - 1, link2[3, 1] - 1:link2[3, 1] + cSize - 1])
        V03 = (image[link2[0, 0] - 1:link2[0, 0] + rSize - 1, link2[0, 1] - 1:link2[0, 1] + cSize - 1] -
               image[link2[1, 0] - 1:link2[1, 0] + rSize - 1, link2[1, 1] - 1:link2[1, 1] + cSize - 1])

        V03[V03 == 0] = EPSILON
        imgGO = np.degrees(np.arctan(V04 / V03))
        imgGO[V03 < 0] += 180
        imgGO[(V03 >= 0) & (V04 < 0)] += 360

        imgDesc = [{'fea': {'GO': imgGO}}, {'fea': {'DE': imgDE}}]
        imgDescs.append(imgDesc)

    options['binVec'] = []
    options['wldHist'] = 1

    # Compute WLD histogram
    WLD_hist = []
    for desc in imgDescs:
        imgGO = desc[0]['fea']['GO']
        imgDE = desc[1]['fea']['DE']

        range_GO = 360 / T
        imgGO = np.floor(imgGO / range_GO)

        range_DE = 180 / N
        imgDE = np.floor(imgDE / range_DE)

        hh = []
        for t in range(T):
            orien = imgDE[imgGO == t]
            orienHist, _ = np.histogram(orien, bins=range(N + 1))
            hh.extend(orienHist)
        WLD_hist.extend(hh)

    WLD_hist = np.array(WLD_hist)
    if 'mode' in options and options['mode'] == 'nh':
        WLD_hist = WLD_hist / np.sum(WLD_hist)

    return WLD_hist, imgDesc