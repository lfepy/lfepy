import numpy as np
from lfepy.Helper.cirInterpSingleRadius_ct import cirInterpSingleRadius_ct


def NILBP_Image_ct(img, lbpPoints, mapping, mode, lbpRadius):
    """
    Compute the Neighborhood Binary Pattern (NILBP) descriptor for an image using circular interpolation.

    Args:
        img (numpy.ndarray): 2D grayscale image.
        lbpPoints (int): Number of points used in the LBP pattern.
        mapping (dict or None): A dictionary containing 'num' (number of bins) and 'table' (mapping table).
            If None, no mapping is applied.
        mode (str): Mode for output. 'h' or 'hist' for histogram of the NILBP, 'nh' for normalized histogram.
        lbpRadius (int): Radius of the circular neighborhood for computing LBP.

    Returns:
        numpy.ndarray: NILBP descriptor, either as a histogram or image depending on the `mode` parameter.

    Example:
        >>> import numpy as np
        >>> from skimage import data
        >>> img = data.camera()
        >>> lbpPoints = 8
        >>> lbpRadius = 1
        >>> mapping = {'num': 256, 'table': np.arange(256)}
        >>> descriptor = NILBP_Image_ct(img, lbpPoints, mapping, mode='nh', lbpRadius=lbpRadius)
        >>> print(descriptor.shape)
        (256,)
    """
    # Compute LBP blocks and dimensions
    blocks, dx, dy = cirInterpSingleRadius_ct(img, lbpPoints, lbpRadius)
    blocks = blocks.T  # Transpose to match the expected shape

    # Centering the blocks by subtracting the mean
    blocks = blocks - np.mean(blocks, axis=1, keepdims=True)

    # Binarize the blocks
    blocks[blocks >= 0] = 1
    blocks[blocks < 0] = 0

    # Calculate the LBP value for each block
    weight = 2 ** np.arange(lbpPoints)
    blocks = blocks * weight
    blocks = np.sum(blocks, axis=1)

    # Reshape the result to match the image dimensions
    result = blocks
    result = np.reshape(result, (dx + 1, dy + 1))

    # Apply mapping if provided
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

    # Compute the histogram or convert result to appropriate type
    if mode in ['h', 'hist', 'nh']:
        result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / np.sum(result)
    else:
        # Determine the appropriate data type for the result
        if (bins - 1) <= np.iinfo(np.uint8).max:
            result = result.astype(np.uint8)
        elif (bins - 1) <= np.iinfo(np.uint16).max:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)

    return result
