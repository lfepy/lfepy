import numpy as np
from lfepy.Helper import get_mapping, phase_cong3, descriptor_LBP
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def BPPC(image, **kwargs):
    """
    Compute Binary Phase Pattern Congruency (BPPC) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing BPPC extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            BPPC_hist (numpy.ndarray): Histogram(s) of BPPC descriptors.
            imgDesc (list): List of dictionaries containing BPPC descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option ('nh' or 'h').

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = BPPC(image, mode='nh')

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        S. Shojaeilangari, W.-Y. Yau, J. Li, and E.-K. Teoh,
        Feature extraction through binary pattern of phase congruency for facial expression recognition,
        in Control Automation Robotics & Vision (ICARCV), 2012 12th International Conference on, IEEE,
        2012, pp. 166-170.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    options['binVec'] = []

    # Compute phase congruency
    _, _, phaseAngle2, _, pc, EO = phase_cong3(image, 4, 6, 3)
    imgDesc = []
    phaseAngle = phaseAngle2[1:-1, 1:-1]

    # Compute BPPC descriptors
    for o in range(6):
        imgDesc.append({'pc': pc[o]})
        mapping = get_mapping(8, 'u2')
        _, codeImg = descriptor_LBP(imgDesc[o]['pc'], 1, 8, mapping, 'nh')

        angleInd = np.floor(phaseAngle / 60)
        imgDesc[o]['fea'] = codeImg + angleInd * 59
        options['binVec'].append(np.arange(177))

    # Compute BPPC histogram
    BPPC_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            BPPC_hist.append(hh)
    BPPC_hist = np.array(BPPC_hist)
    if 'mode' in options and options['mode'] == 'nh':
        BPPC_hist = BPPC_hist / np.sum(BPPC_hist)

    return BPPC_hist, imgDesc
