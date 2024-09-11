import numpy as np
from lfepy.Helper import gabor_filter
from lfepy.Validator import validate_image, validate_kwargs, validate_mode


def LGDiP(image, **kwargs):
    """
    Compute Local Gabor Directional Pattern (LGDiP) histograms and descriptors from an input image.

    Args:
        image (numpy.ndarray): Input image (preferably in NumPy array format).
        **kwargs (dict): Additional keyword arguments for customizing LGDiP extraction.
            mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            LGDiP_hist (numpy.ndarray): Histogram(s) of LGDiP descriptors.
            imgDesc (list): List of dictionaries containing LGDiP descriptors for each scale.

    Raises:
        TypeError: If the `image` is not a valid `numpy.ndarray`.
        ValueError: If the `mode` in `kwargs` is not a valid option.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.image import imread

        >>> image = imread("Path")
        >>> histogram, imgDesc = LGDiP(image, mode='nh')

        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    References:
        S.Z. Ishraque, A.H. Banna, and O. Chae,
        Local Gabor Directional Pattern for Facial Expression Recognition,
        ICCIT 2012: 15th International Conference on Computer and Information Technology, IEEE,
        2012, pp. 164-167.
    """
    # Input data validation
    image = validate_image(image)
    options = validate_kwargs(**kwargs)
    options = validate_mode(options)

    # Define unique bin values
    uniqueBin = np.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44,
                          49, 50, 52, 56, 67, 69, 70, 73, 74, 76, 81, 82, 84, 88, 97, 98,
                          100, 104, 112, 131, 133, 134, 137, 138, 140, 145, 146, 148, 152,
                          161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224])

    # Initialize variables
    ro, co = image.shape
    imgDesc = []
    options['binVec'] = []

    # Compute Gabor magnitude
    gaborMag = abs(gabor_filter(image, 8, 5))

    for scale in range(5):
        ind = np.argsort(gaborMag[:, :, :, scale], axis=2)[:, :, ::-1]
        bit8array = np.zeros((ro, co, 8))

        bit8array[np.isin(ind, [1, 2, 3])] = 1
        codeImg = np.zeros((ro, co))

        for r in range(ro):
            codebit = np.flip(bit8array[r, :, :], axis=1).reshape(co, -1)
            codeImg[r, :] = np.packbits(codebit.astype(np.uint8), axis=1).reshape(-1)

        imgDesc.append({'fea': codeImg})
        options['binVec'].append(uniqueBin)

    # Compute LGDiP histogram
    LGDiP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LGDiP_hist.append(hh)
    LGDiP_hist = np.array(LGDiP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LGDiP_hist = LGDiP_hist / np.sum(LGDiP_hist)

    return LGDiP_hist, imgDesc
