import numpy as np
from lfepy.Helper import gabor_filter, descriptor_LBP, get_mapping
from lfepy.Validator import validate_image, validate_kwargs, validate_mode, validate_uniformLBP, validate_scaleNum, validate_orienNum


def LGBPHS(image, **kwargs):
    """
    Compute Local Gabor Binary Pattern Histogram Sequence (LGBPHS) descriptors and histograms from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGBPHS extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.
            uniformLBP (int): Flag to use uniform LBP. Default is 1 (use uniform LBP).
            scaleNum (int): Number of scales for Gabor filters. Default is 5.
            orienNum (int): Number of orientations for Gabor filters. Default is 8.

    Returns:
        tuple: A tuple containing:
            LGBPHS_hist (numpy.ndarray): Histogram(s) of LGBPHS descriptors.
            imgDesc (list): List of dictionaries containing LGBPHS descriptors for each scale and orientation.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGBPHS(image, mode='nh', uniformLBP=1, scaleNum=5, orienNum=8)

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        W. Zhang, S. Shan, W. Gao, X. Chen, and H. Zhang,
        Local Gabor Binary Pattern Histogram Sequence (LGBPHS): A Novel Non-Statistical Model for Face Representation and Recognition,
        ICCV 2005: Tenth IEEE International Conference on Computer Vision, IEEE,
        2005, pp. 786-791.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)
    uniformLBP = validate_uniformLBP(options)
    scaleNum = validate_scaleNum(options)
    orienNum = validate_orienNum(options)

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