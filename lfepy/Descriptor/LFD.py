import numpy as np
from lfepy.Helper import descriptor_LBP, descriptor_LPQ


def LFD(image, **kwargs):
    """
    Compute Local Frequency Descriptor (LFD) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LFD extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LFD_hist: Histogram(s) of LFD descriptors.
        - imgDesc: List of dictionaries containing LFD descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = LFD(image, mode='nh')
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        Z. Lei, T. Ahonen, M. Pietikäinen, and S.Z. Li,
        Local frequency descriptor for low-resolution face recognition,
        Automatic Face & Gesture Recognition and Workshops (FG 2011), 2011 IEEE International Conference on,
        IEEE, 2011, pp. 161-166.
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

    _, filterResp = descriptor_LPQ(image, 5)
    magn = np.abs(filterResp)

    imgDesc = [{'fea': descriptor_LBP(magn, 1, 8)[1]}]

    CoorX = np.sign(np.real(filterResp))
    CoorY = np.sign(np.imag(filterResp))

    quadrantMat = np.ones_like(filterResp, dtype=np.uint8)
    quadrantMat[CoorX == -1 & (CoorY == 1)] = 2
    quadrantMat[CoorX == -1 & (CoorY == -1)] = 3
    quadrantMat[CoorX == 1 & (CoorY == -1)] = 4

    rSize, cSize = quadrantMat.shape[0] - 2, quadrantMat.shape[1] - 2
    link = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
    x_c = quadrantMat[1:-1, 1:-1]
    pattern = np.zeros_like(x_c, dtype=np.uint8)

    for n, (i, j) in enumerate(link):
        x_i = quadrantMat[i - 1:i + rSize - 1, j - 1:j + cSize - 1]
        pattern += (x_c == x_i).astype(np.uint8) * (2 ** (len(link) - n - 1))

    imgDesc.append({'fea': pattern.astype(np.float64)})

    options['binVec'] = [np.arange(256)] * 2

    # Compute LFD histogram
    LFD_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LFD_hist.append(hh)
    LFD_hist = np.array(LFD_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LFD_hist = LFD_hist / np.sum(LFD_hist)

    return LFD_hist, imgDesc
