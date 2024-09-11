import numpy as np
from lfepy.Helper import monofilt, descriptor_LBP, lxp_phase
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_mbcMode


def MBC(image, **kwargs):
    """
    Compute Monogenic Binary Coding (MBC) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MBC extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            mbcMode (str): Mode for MBC computation. Options: 'A' (amplitude), 'O' (orientation), 'P' (phase). Default is 'A'.

    Returns:
        tuple: A tuple containing:
            MBC_hist (numpy.ndarray): Histogram of MBC descriptors.
            imgDesc (list): List of dictionaries containing MBC descriptors.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` or `mbcMode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MBC(image, mode='nh', mbcMode='A')

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M. Yang, L. Zhang, S.C.-K. Shiu, and D. Zhang,
        Monogenic binary coding: An efficient local feature extraction approach to face recognition,
        IEEE Transactions on Information Forensics and Security,
        7 (2012) 1738-1751.

        X.X. Xia, Z.L. Ying, and W.J. Chu,
        Facial Expression Recognition Based on Monogenic Binary Coding,
        Applied Mechanics and Materials, Trans Tech Publ,
        2014, pp. 437-440.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    options = validate_mbcMode(options)

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
