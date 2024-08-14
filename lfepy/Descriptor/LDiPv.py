from lfepy.Helper.helper import np, convolve2d


def LDiPv(image, **kwargs):
    """
    Compute Local Directional Pattern Variance (LDiPv) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LDiPv extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LDiPv_hist: Histogram(s) of LDiPv descriptors.
        - imgDesc: LDiPv descriptors themselves.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = LDiPv(image, mode='nh')
        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        M.H. Kabir, T. Jabid, and O. Chae,
        A local directional pattern variance (LDPv) based face descriptor for human facial expression recognition,
        Advanced Video and Signal Based Surveillance (AVSS), 2010 Seventh IEEE International Conference on,
        IEEE, 2010, pp. 526-532.
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

    # Define Kirsch masks
    Kirsch = [np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute the response of the image to each Kirsch mask
    maskResponses = np.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch, start=1):
        maskResponses[:, :, i - 1] = convolve2d(image, kirsch_mask, mode='same')

    # Take the absolute value of the mask responses
    maskResponsesAbs = np.abs(maskResponses)

    # Sort the mask responses to find the strongest responses
    ind = np.argsort(maskResponsesAbs, axis=2)[:, :, ::-1]

    # Create a binary 8-bit array based on the top 3 strongest responses
    bit8array = np.zeros((image.shape[0], image.shape[1], 8))
    bit8array[(ind == 0) | (ind == 1) | (ind == 2)] = 1

    # Generate the LDiPv descriptor for each pixel
    imgDesc = np.zeros_like(image)
    for r in range(image.shape[0]):
        codebit = np.reshape(bit8array[r, :, ::-1], (image.shape[1], -1))
        imgDesc[r, :] = np.array([int(''.join(map(str, row.astype(np.uint8))), 2) for row in codebit])

    # Define the unique bins for the histogram
    uniqueBin = np.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56, 67, 69,
                          70, 73, 74, 76, 81, 82, 84, 88, 97, 98, 100, 104, 112, 131, 133, 134, 137, 138, 140, 145,
                          146, 148, 152, 161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Compute the variance of the mask responses
    varianceImg = np.var(maskResponsesAbs, axis=2)
    options['weight'] = varianceImg
    options['binVec'] = uniqueBin

    # Compute LDiPv histogram
    LDiPv_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LDiPv_hist[i] = np.sum(options['weight'][imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LDiPv_hist = LDiPv_hist / np.sum(LDiPv_hist)

    return LDiPv_hist, imgDesc
