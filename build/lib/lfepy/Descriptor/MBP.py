import numpy as np


def MBP(image, **kwargs):
    """
    Compute Median Binary Pattern (MBP) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing MBP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - MBP_hist: Histogram(s) of MBP descriptors.
        - imgDesc: MBP descriptors.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = MBP(image, mode='nh')
        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust facial expression recognition based on median ternary pattern (MTP),
        Electrical Information and Communication Technology (EICT), 2013 International Conference on,
        IEEE, 2014, pp. 1-5.
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

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for MBP computation
    link = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])
    ImgIntensity = np.zeros((rSize * cSize, link.shape[0]))

    # Compute MBP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        x_slice = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        ImgIntensity[:, n] = x_slice.reshape(-1)

    medianMat = np.median(ImgIntensity, axis=1)
    MBP = np.double(ImgIntensity > medianMat.reshape(-1, 1))
    imgDesc = np.array([int(''.join(map(str, row.astype(np.uint8))), 2) for row in MBP]).reshape(rSize, cSize)

    # Set bin vectors
    options['binVec'] = np.arange(256)

    # Compute MBP histogram
    MBP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        MBP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        MBP_hist = MBP_hist / np.sum(MBP_hist)

    return MBP_hist, imgDesc
