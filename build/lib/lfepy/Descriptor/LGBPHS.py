from lfepy.Helper.helper import np, gabor_filter, descriptor_LBP, get_mapping


def LGBPHS(image, **kwargs):
    """
    Compute Local Gabor Binary Pattern Histogram Sequence (LGBPHS) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LGBPHS extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.uniformLBP: Flag to use uniform LBP. Default: 1 (use uniform LBP).
    :type kwargs.uniformLBP: int
    :param kwargs.scaleNum: Number of scales for Gabor filters. Default: 5.
    :type kwargs.scaleNum: int
    :param kwargs.orienNum: Number of orientations for Gabor filters. Default: 8.
    :type kwargs.orienNum: int

    :returns:
        - LGBPHS_hist: Histogram(s) of LGBPHS descriptors.
        - imgDesc: List of dictionaries containing LGBPHS descriptors for each scale and orientation.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = LGBPHS(image, mode='nh', uniformLBP=1, scaleNum=5, orienNum=8)
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        W. Zhang, S. Shan, W. Gao, X. Chen, and H. Zhang,
        Local gabor binary pattern histogram sequence (lgbphs): A novel non-statistical model for face representation and recognition,
        Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on,
        IEEE, 2005, pp. 786-791.
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

    uniformLBP = options.get('uniformLBP', 1)
    scaleNum = options.get('scaleNum', 5)
    orienNum = options.get('orienNum', 8)

    # Compute Gabor magnitude responses
    gaborMag = np.abs(gabor_filter(image, 8, 5))

    options['binVec'] = []
    imgDesc = []

    # Compute LGBPHS descriptors
    for s in range(scaleNum):
        for o in range(orienNum):
            gaborResIns = gaborMag[:, :, o, s]
            if uniformLBP == 1:
                mapping = get_mapping(8, 'u2')
                _, codeImg = descriptor_LBP(gaborResIns, 1, 8, mapping, 'uniform')
                options['binVec'].append(np.arange(59))
            else:
                _, codeImg = descriptor_LBP(gaborResIns, 1, 8, None, 'default')
                options['binVec'].append(np.arange(256))

            imgDesc.append({'fea': codeImg})

    # Compute LGBPHS histogram
    LGBPHS_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LGBPHS_hist.append(hh)
    LGBPHS_hist = np.array(LGBPHS_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LGBPHS_hist = LGBPHS_hist / np.sum(LGBPHS_hist)

    return LGBPHS_hist, imgDesc

