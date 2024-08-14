from lfepy.Helper.helper import np, monofilt, descriptor_LBP, lxp_phase


def MBC(image, **kwargs):
    """
    Compute Monogenic Binary Coding (MBC) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing MBC extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.mbcMode: Mode for MBC computation. Options: 'A' (amplitude), 'O' (orientation), 'P' (phase). Default: 'A'.
    :type kwargs.mbcMode: str

    :returns:
        - MBC_hist: Histogram of MBC descriptors.
        - imgDesc: List of dictionaries containing MBC descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = MBC(image, mode='nh', mbcMode='A')
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        M. Yang, L. Zhang, S.C.-K. Shiu, and D. Zhang,
        Monogenic binary coding: An efficient local feature extraction approach to face recognition.
        IEEE Transactions on Information Forensics and Security 7 (2012) 1738-1751.

        X.X. Xia, Z.L. Ying, and W.J. Chu,
        Facial Expression Recognition Based on Monogenic Binary Coding,
        Applied Mechanics and Materials,
        Trans Tech Publ, 2014, pp. 437-440.
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

    # Extract MBC mode or set default
    if 'mbcMode' not in options:
        options['mbcMode'] = 'A'

    # Validate the MBC mode
    valid_MBC = ['A', 'O', 'P']
    if options['mbcMode'] not in valid_MBC:
        raise ValueError(f"Invalid mbc Mode '{options['mbcMode']}'. Valid mbc Modes are {valid_MBC}.")

    # Parameters for monogenic filters
    minWaveLength = 4
    sigmaOnf = 0.64
    mult = 1.7
    nscale = 3
    neigh = 8
    MAPPING = 0

    imgDesc = []
    options['binVec'] = []

    # Amplitude-based MBC
    if options['mbcMode'] == 'A':
        orientWrap = 0
        radius = 3
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = np.uint8((A1[v] - np.min(A1[v])) / (np.max(A1[v]) - np.min(A1[v])) * 255)
            LBPHIST, _ = descriptor_LBP(Tem_img, radius, neigh, MAPPING, 'i')
            matrix2 = np.zeros(np.shape(h1f1[v]))
            matrix3 = np.zeros(np.shape(h2f1[v]))
            matrix2[h1f1[v] > 0] = 0
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius:-radius, radius:-radius]
            matrix3[h2f1[v] > 0] = 0
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius:-radius, radius:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + np.double(LBPHIST)
            N_LBPHIST = np.uint16(N_LBPHIST)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(np.arange(1024))

    # Orientation-based MBC
    elif options['mbcMode'] == 'O':
        orientWrap = 0
        radius = 4
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = np.uint16((theta1[v] - np.min(theta1[v])) / (np.max(theta1[v]) - np.min(theta1[v])) * 360)
            LBPHIST = lxp_phase(Tem_img, radius, neigh, 0, 'i')
            matrix2 = np.zeros(np.shape(h1f1[v]))
            matrix3 = np.zeros(np.shape(h2f1[v]))
            matrix2[h1f1[v] > 0] = 0
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius + 1:-radius, radius + 1:-radius]
            matrix3[h2f1[v] > 0] = 0
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius + 1:-radius, radius + 1:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + np.double(LBPHIST)
            N_LBPHIST = np.uint16(N_LBPHIST)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(np.arange(1024))

    # Phase-based MBC
    elif options['mbcMode'] == 'P':
        orientWrap = 1
        radius = 4
        f1, h1f1, h2f1, A1, theta1, psi1 = monofilt(image, nscale, minWaveLength, mult, sigmaOnf, orientWrap)
        for v in range(nscale):
            Tem_img = np.uint16((psi1[v] - np.min(psi1[v])) / (np.max(psi1[v]) - np.min(psi1[v])) * 360)
            LBPHIST = lxp_phase(Tem_img, radius, neigh, 0, 'i')
            matrix2 = np.zeros(np.shape(h1f1[v]))
            matrix3 = np.zeros(np.shape(h2f1[v]))
            matrix2[h1f1[v] > 0] = 0
            matrix2[h1f1[v] <= 0] = 1
            matrix2 = matrix2[radius + 1:-radius, radius + 1:-radius]
            matrix3[h2f1[v] > 0] = 0
            matrix3[h2f1[v] <= 0] = 1
            matrix3 = matrix3[radius + 1:-radius, radius + 1:-radius]
            N_LBPHIST = matrix2 * 512 + matrix3 * 256 + np.double(LBPHIST)
            N_LBPHIST = np.uint16(N_LBPHIST)
            imgDesc.append({'fea': N_LBPHIST})
            options['binVec'].append(np.arange(1024))

    # Compute MBC histogram
    MBC_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            MBC_hist.append(hh)
    MBC_hist = np.array(MBC_hist)
    if 'mode' in options and options['mode'] == 'nh':
        MBC_hist = MBC_hist / np.sum(MBC_hist)

    return MBC_hist, imgDesc
