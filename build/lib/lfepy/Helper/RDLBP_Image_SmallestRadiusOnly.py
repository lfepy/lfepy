import numpy as np
from lfepy.Helper.cirInterpSingleRadiusNew import cirInterpSingleRadiusNew


def RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode):
    """
    Compute the Radial Difference Local Binary Pattern (RDLBP) for an image with a focus on the smallest radius.

    :param imgCenSmooth: Smoothed image from which the radial difference is computed.
    :type imgCenSmooth: numpy.ndarray
    :param img: Original image for extracting circularly interpolated blocks.
    :type img: numpy.ndarray
    :param lbpRadius: Radius of the circular neighborhood for LBP.
    :type lbpRadius: int
    :param lbpPoints: Number of points used in the LBP pattern.
    :type lbpPoints: int
    :param mapping: Optional mapping dictionary for converting LBP result to a different bin scheme.
        Must contain 'num' (number of bins) and 'table' (mapping from old bin to new bin).
    :type mapping: dict or None
    :param mode: Output mode. 'h' or 'hist' for histogram of the RDLBP, 'nh' for normalized histogram.
    :type mode: str

    :returns: RDLBP descriptor, either as a histogram or image depending on the `mode` parameter.
    :rtype: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> imgCenSmooth = data.coins()
        >>> lbpRadius = 1
        >>> lbpPoints = 8
        >>> mapping = {'num': 256, 'table': np.arange(256)}
        >>> hist = RDLBP_Image_SmallestRadiusOnly(imgCenSmooth, img, lbpRadius, lbpPoints, mapping, mode='nh')
        >>> print(hist.shape)
        (256,)  # Example output shape for normalized histogram
    """
    # Extract circularly interpolated blocks from the original image
    blocks1, dx, dy = cirInterpSingleRadiusNew(img, lbpPoints, lbpRadius)
    blocks1 = blocks1.T

    # Adjust the smoothed image size based on the radius
    imgTemp = imgCenSmooth[lbpRadius:-lbpRadius, lbpRadius:-lbpRadius]
    # Create a tiled version of the smoothed image to match the size of the LBP blocks
    blocks2 = np.tile(imgTemp.ravel(), (lbpPoints, 1)).T

    # Compute the radial difference between the blocks of the original image and the smoothed image
    radialDiff = blocks1 - blocks2
    radialDiff[radialDiff >= 0] = 1
    radialDiff[radialDiff < 0] = 0

    # Compute the LBP value by weighting the binary differences
    bins = 2 ** lbpPoints
    weight = 2 ** np.arange(lbpPoints)
    radialDiff = radialDiff * weight
    radialDiff = np.sum(radialDiff, axis=1)

    # Reshape the result to match the dimensions of the original image
    result = radialDiff
    result = np.reshape(result, (dx + 1, dy + 1))

    # Apply mapping if it is defined
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

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