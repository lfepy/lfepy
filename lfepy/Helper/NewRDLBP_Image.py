import numpy as np
from lfepy.Helper.cirInterpSingleRadiusNew import cirInterpSingleRadiusNew


def NewRDLBP_Image(img, imgPre, lbpRadius, lbpRadiusPre, lbpPoints, mapping=None, mode='h'):
    """
    Compute the Radial Difference Local Binary Pattern (RDLBP) between two images.

    Args:
        img (numpy.ndarray): 2D grayscale image.
        imgPre (numpy.ndarray): 2D grayscale image for comparison.
        lbpRadius (int): Radius of the circular neighborhood for the current image.
        lbpRadiusPre (int): Radius of the circular neighborhood for the comparison image.
        lbpPoints (int): Number of points used in the LBP pattern.
        mapping (dict or None, optional): Mapping dictionary for converting the LBP result to a different bin scheme.
            If provided, must contain 'num' (number of bins) and 'table' (mapping from old bin to new bin). Default is None.
        mode (str, optional): Mode for output. 'h' or 'hist' for histogram of the RDLBP, 'nh' for normalized histogram. Default is 'h'.

    Returns:
        numpy.ndarray: RDLBP descriptor, either as a histogram or image depending on the `mode` parameter.

    Raises:
        ValueError: If `mapping` is provided but does not contain the required keys.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> imgPre = data.coins()
        >>> lbpRadius = 1
        >>> lbpRadiusPre = 1
        >>> lbpPoints = 8
        >>> hist = NewRDLBP_Image(img, imgPre, lbpRadius, lbpRadiusPre, lbpPoints, mode='nh')
        >>> print(hist.shape)
        (256,)
    """
    # Extract circularly interpolated blocks from the current image
    blocks1, _, _ = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
    blocks1 = blocks1.T

    # Adjust the comparison image size based on radii and extract circularly interpolated blocks
    imgPre = imgPre[lbpRadius - lbpRadiusPre: -lbpRadius + lbpRadiusPre, lbpRadius - lbpRadiusPre: -lbpRadius + lbpRadiusPre]
    blocks2, _, _ = cirInterpSingleRadiusNew(imgPre, lbpPoints, lbpRadiusPre)
    blocks2 = blocks2.T

    # Compute the radial difference between the two images
    radialDiff = blocks1 - blocks2
    radialDiff[radialDiff >= 0] = 1
    radialDiff[radialDiff < 0] = 0

    # Compute the LBP value by weighting the binary differences
    bins = 2 ** lbpPoints
    weight = 2 ** np.arange(lbpPoints)
    radialDiff = radialDiff * weight
    radialDiff = np.sum(radialDiff, axis=1)

    # Apply mapping if it is defined
    if mapping is not None:
        bins = mapping['num']
        result = np.array([mapping['table'][int(r)] for r in radialDiff], dtype=np.uint32)
    else:
        result = radialDiff

    # Return result as histogram or image depending on mode
    if mode in ['h', 'hist', 'nh']:
        hist_result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            hist_result = hist_result / np.sum(hist_result)
        return hist_result
    else:
        # Return result as matrix of unsigned integers
        max_val = bins - 1
        if max_val <= np.iinfo(np.uint8).max:
            return result.astype(np.uint8)
        elif max_val <= np.iinfo(np.uint16).max:
            return result.astype(np.uint16)
        else:
            return result.astype(np.uint32)