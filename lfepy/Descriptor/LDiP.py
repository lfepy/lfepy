import numpy as np
from scipy.signal import convolve2d


def LDiP(image, **kwargs):
    """
    Compute Local Directional Pattern (LDiP) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LDiP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LDiP_hist: Histogram(s) of LDiP descriptors.
        - imgDesc: LDiP descriptors.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = LDiP(image, mode='nh')
        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        T. Jabid, M.H. Kabir, and O. Chae,
        Local directional pattern (LDP)–A robust image descriptor for object recognition,
        Advanced Video and Signal Based Surveillance (AVSS), 2010 Seventh IEEE International Conference on,
        IEEE, 2010, pp. 482-487.
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

    # Define Kirsch Masks
    Kirsch = [np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
              np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
              np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
              np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
              np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
              np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
              np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
              np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]

    # Compute mask responses
    maskResponses = np.zeros((image.shape[0], image.shape[1], 8))
    for i, kirsch_mask in enumerate(Kirsch):
        maskResponses[:, :, i] = np.abs(convolve2d(image, kirsch_mask, mode='same'))

    # Sort responses and construct binary pattern
    ind = np.argsort(maskResponses, axis=2)[:, :, ::-1]
    bit8array = np.zeros((image.shape[0], image.shape[1], 8))
    bit8array[np.logical_or(np.logical_or(ind == 0, ind == 1), ind == 2)] = 1
    imgDesc = np.zeros_like(image)
    for r in range(image.shape[0]):
        codebit = np.reshape(bit8array[r, :, 7::-1], (image.shape[1], -1))
        imgDesc[r, :] = np.packbits(codebit.astype(bool), axis=1).flatten()

    # Define unique bins for histogram
    uniqueBin = np.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 49, 50, 52, 56, 67, 69,
                          70, 73, 74, 76, 81, 82, 84, 88, 97, 98, 100, 104, 112, 131, 133, 134, 137, 138, 140,
                          145, 146, 148, 152, 161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Set binVec option
    options['binVec'] = uniqueBin

    # Compute LDiP histogram
    LDiP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LDiP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LDiP_hist = LDiP_hist / np.sum(LDiP_hist)

    return LDiP_hist, imgDesc
