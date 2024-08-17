import numpy as np
from lfepy.Helper import descriptor_LBP, get_mapping


def LBP(image, **kwargs):
    """
    Compute Local Binary Patterns (LBP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LBP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            radius (int): Radius for LBP computation. Default is 1.
            mappingType (str): Type of mapping for LBP computation. Options: 'full', 'ri', 'u2', 'riu2'. Default is 'full'.

    Returns:
        tuple: A tuple containing:
            LBP_hist (numpy.ndarray): Histogram(s) of LBP descriptors.
            imgDesc (numpy.ndarray): LBP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` or `mappingType` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LBP(image, mode='nh', radius=1, mappingType='full')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        T. Ojala, M. Pietikainen, and T. Maenpaa,
        Multi-resolution gray-scale and rotation invariant texture classification with local binary patterns,
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 24, pp. 971-987, 2002.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision if needed
    if image.dtype != np.float64:
        image = np.double(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Handle keyword arguments
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    # Extract radius and compute number of neighbors or use defaults
    if 'radius' in options:
        radius = options['radius']
        neighbors = 8 * radius
    else:
        radius = 1
        neighbors = 8

    # Extract histogram mode
    if 'mode' not in options:
        options['mode'] = 'nh'

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    mode = options['mode']

    # Handle mapping type and adjust bin vector accordingly
    if 'mappingType' in options and options['mappingType'] != 'full':
        mappingType = options['mappingType']
        mapping = get_mapping(neighbors, mappingType)
        if mappingType == 'u2':
            if radius == 1:
                options['binVec'] = np.arange(0, 59)
            elif radius == 2:
                options['binVec'] = np.arange(0, 243)
        elif mappingType == 'ri':
            if radius == 1:
                options['binVec'] = np.arange(0, 36)
            elif radius == 2:
                options['binVec'] = np.arange(0, 4117)
        else:
            if radius == 1:
                options['binVec'] = np.arange(0, 10)
            elif radius == 2:
                options['binVec'] = np.arange(0, 16)
    else:
        mapping = 0
        options['binVec'] = np.arange(0, 256)

    # Extract LBP descriptors
    _, imgDesc = descriptor_LBP(image, radius, neighbors, mapping, mode)

    # Compute LBP histogram
    LBP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LBP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LBP_hist = LBP_hist / np.sum(LBP_hist)

    return LBP_hist, imgDesc
