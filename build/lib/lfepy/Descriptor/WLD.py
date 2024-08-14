from lfepy.Helper.helper import np


def WLD(image, **kwargs):
    """
    Compute Weber Local Descriptor (WLD) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing WLD extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.T: Number of bins for gradient orientation. Default: 8.
    :type kwargs.T: int
    :param kwargs.N: Number of bins for differential excitation. Default: 4.
    :type kwargs.N: int
    :param kwargs.scaleTop: Number of scales to consider for WLD computation. Default: 1.
    :type kwargs.scaleTop: int

    :returns:
        - WLD_hist: Histogram(s) of WLD descriptors.
        - imgDesc: List of dictionaries containing WLD descriptors.
    :rtype: tuple of (numpy.ndarray, list of dicts)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = WLD(image, mode='nh', T=8, N=4, scaleTop=1)
        >>> plt.imshow(imgDesc[0]['fea']['GO'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()
        >>> plt.imshow(imgDesc[1]['fea']['DE'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        S. Li, D. Gong, and Y. Yuan,
        Face recognition using Weber local descriptors.
        Neurocomputing 122 (2013) 272-283.

        S. Liu, Y. Zhang, and K. Liu,
        Facial expression recognition under partial occlusion based on Weber Local Descriptor histogram and decision fusion,
        Control Conference (CCC), 2014 33rd Chinese,
        IEEE, 2014, pp. 4664-4668.
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

    # Extract T value or set default
    if 'T' not in options:
        options['T'] = 8
    T = options.get('T', 4)

    # Extract N value or set default
    if 'N' not in options:
        options['N'] = 4
    N = options.get('N', 4)

    scaleTop = options.get('scaleTop', 1)

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
