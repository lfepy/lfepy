import numpy as np
from lfepy.Helper import gabor_filter
from lfepy.Descriptor.LTrP import LTrP
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LGTrP(image, **kwargs):
    """
    Compute Local Gabor Transitional Pattern (LGTrP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGTrP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGTrP_hist (numpy.ndarray): Histogram(s) of LGTrP descriptors.
            imgDesc (numpy.ndarray): LGTrP descriptors.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGTrP(image, mode='nh')

        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        M.S. Islam,
        Local Gradient Pattern-A Novel Feature Representation for Facial Expression Recognition,
        Journal of AI and Data Mining 2,
        (2014), pp. 33-38.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Compute LGTrP descriptor
    gaborMag = np.abs(gabor_filter(image, 8, 1))
    gaborTotal = gaborMag[:, :, 0, 0]

    for o in range(1, 8):
        gaborTotal += gaborMag[:, :, o, 0]

    imgDescGabor = gaborTotal / 8
    _, imgDesc = LTrP(imgDescGabor)

    # Set bin vector
    options['binVec'] = np.arange(256)

    # Compute LGTrP histogram
    LGTrP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LGTrP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LGTrP_hist = LGTrP_hist / np.sum(LGTrP_hist)

    return LGTrP_hist, imgDesc
