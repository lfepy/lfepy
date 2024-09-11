import numpy as np
from lfepy.Helper import descriptor_LBP, descriptor_LPQ
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LFD(image, **kwargs):
    """
    Compute Local Frequency Descriptor (LFD) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LFD extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LFD_hist (numpy.ndarray): Histogram(s) of LFD descriptors.
            imgDesc (list): List of dictionaries containing LFD descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LFD(image, mode='nh')

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        Z. Lei, T. Ahonen, M. Pietikäinen, and S.Z. Li,
        Local Frequency Descriptor for Low-Resolution Face Recognition,
        Automatic Face & Gesture Recognition and Workshops (FG 2011), IEEE,
        2011, pp. 161-166.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

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
