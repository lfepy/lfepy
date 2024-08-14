from lfepy.Helper.helper import np, gabor_filter


def LGDiP(image, **kwargs):
    """
    Compute Local Gabor Directional Pattern (LGDiP) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LGDiP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LGDiP_hist: Histogram(s) of LGDiP descriptors.
        - imgDesc: List of dictionaries containing LGDiP descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = LGDiP(image, mode='nh')
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        S.Z. Ishraque, A.H. Banna, and O. Chae,
        Local Gabor directional pattern for facial expression recognition,
        Computer and Information Technology (ICCIT), 2012 15th International Conference on,
        IEEE, 2012, pp. 164-167.
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
        options['mode'] = 'nh'

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    # Define unique bin values
    uniqueBin = np.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44,
                          49, 50, 52, 56, 67, 69, 70, 73, 74, 76, 81, 82, 84, 88, 97, 98,
                          100, 104, 112, 131, 133, 134, 137, 138, 140, 145, 146, 148, 152,
                          161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Initialize variables
    ro, co = image.shape
    imgDesc = []
    options['binVec'] = []

    # Compute Gabor magnitude
    gaborMag = abs(gabor_filter(image, 8, 5))

    for scale in range(5):
        ind = np.argsort(gaborMag[:, :, :, scale], axis=2)[:, :, ::-1]
        bit8array = np.zeros((ro, co, 8))

        bit8array[np.isin(ind, [1, 2, 3])] = 1
        codeImg = np.zeros((ro, co))

        for r in range(ro):
            codebit = np.flip(bit8array[r, :, :], axis=1).reshape(co, -1)
            codeImg[r, :] = np.packbits(codebit.astype(np.uint8), axis=1).reshape(-1)

        imgDesc.append({'fea': codeImg})
        options['binVec'].append(uniqueBin)

    # Compute LGDiP histogram
    LGDiP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LGDiP_hist.append(hh)
    LGDiP_hist = np.array(LGDiP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LGDiP_hist = LGDiP_hist / np.sum(LGDiP_hist)

    return LGDiP_hist, imgDesc
