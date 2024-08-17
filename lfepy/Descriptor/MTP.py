import numpy as np


def MTP(image, **kwargs):
    """
    Compute Median Ternary Pattern (MTP) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing MTP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            t (float): Threshold value for MTP computation. Default is 10.

    Returns:
        tuple: A tuple containing:
            MTP_hist (numpy.ndarray): Histogram(s) of MTP descriptors.
            imgDesc (list of dicts): List of dictionaries containing MTP descriptors for positive and negative thresholds.

    Raises:
        TypeError: If `image` is not a valid `numpy.ndarray`.
        ValueError: If `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = MTP(image, mode='nh', t=10)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir,
        Robust facial expression recognition based on median ternary pattern (MTP),
        Electrical Information and Communication Technology (EICT), 2013 International Conference on, IEEE,
        2014, pp. 1-5.
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

    # Extract histogram mode
    if 'mode' not in options:
        options.update({'mode': 'nh'})

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    # Extract threshold value or set default to 10
    t = options.get('t', 10)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for MTP computation
    link = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])
    ImgIntensity = np.zeros((rSize * cSize, link.shape[0]))

    # Compute MTP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        x_slice = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1]
        ImgIntensity[:, n] = x_slice.reshape(-1)

    medianMat = np.median(ImgIntensity, axis=1)

    Pmtp = np.double(ImgIntensity > (medianMat + t).reshape(-1, 1))
    Nmtp = np.double(ImgIntensity < (medianMat - t).reshape(-1, 1))

    imgDesc = [{'fea': np.array([int(''.join(map(str, row.astype(np.uint8))), 2) for row in Pmtp]).reshape(rSize, cSize)},
               {'fea': np.array([int(''.join(map(str, row.astype(np.uint8))), 2) for row in Nmtp]).reshape(rSize, cSize)}]
    options['binVec'] = [np.arange(256), np.arange(256)]

    # Compute MTP histogram
    MTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            MTP_hist.append(hh)
    MTP_hist = np.array(MTP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        MTP_hist = MTP_hist / np.sum(MTP_hist)

    return MTP_hist, imgDesc