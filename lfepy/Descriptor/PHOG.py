from lfepy.Helper.helper import np, descriptor_PHOG, phogDescriptor_hist


def PHOG(image, **kwargs):
    """
    Compute Pyramid Histogram of Oriented Gradients (PHOG) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing PHOG extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str
    :param kwargs.bin: Number of bins for the histogram. Default: 8.
    :type kwargs.bin: int
    :param kwargs.angle: Range of gradient angles. Default: 360.
    :type kwargs.angle: int
    :param kwargs.L: Number of pyramid levels. Default: 2.
    :type kwargs.L: int

    :returns:
        - PHOG_hist: Histogram of PHOG descriptors.
        - imgDesc: List of dictionaries containing PHOG descriptors.
    :rtype: tuple of (numpy.ndarray, list of dicts)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = PHOG(image, mode='nh', bin=8, angle=360, L=2)
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        A. Bosch, A. Zisserman, and X. Munoz,
        Representing shape with a spatial pyramid kernel,
        Proceedings of the 6th ACM international conference on Image and video retrieval,
        ACM, 2007, pp. 401-408.
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

    # Set defaults
    bin = options.get('bin', 8)
    angle = options.get('angle', 360)
    L = options.get('L', 2)

    # Define the region of interest (ROI)
    roi = [0, image.shape[0], 0, image.shape[1]]

    # Compute PHOG descriptors
    _, bh_roi, bv_roi = descriptor_PHOG(image, bin, angle, L, roi)

    # Collect descriptors
    imgDesc = [{'fea': bh_roi}, {'fea': bv_roi}]

    # Compute PHOG histogram
    PHOG_hist = phogDescriptor_hist(bh_roi, bv_roi, L, bin)

    # Normalize the histogram if required
    if 'mode' in options and options['mode'] == 'nh':
        PHOG_hist = PHOG_hist / np.sum(PHOG_hist)

    return PHOG_hist, imgDesc

