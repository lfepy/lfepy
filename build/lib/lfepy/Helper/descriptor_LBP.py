import numpy as np
from lfepy.Helper.roundn import roundn
from lfepy.Helper.get_mapping import get_mapping


def descriptor_LBP(*varargin):
    """
    Compute the Local Binary Pattern (LBP) of an image with various options for radius, neighbors, mapping, and mode.

    The function computes the LBP of a grayscale image based on the provided parameters, including radius, number of
    neighbors, and optional mapping and mode settings. It returns either the LBP histogram or the LBP code image.

    Args:
        image (numpy.ndarray): The input image, expected to be a 2D numpy array (grayscale).
        radius (int, optional): The radius of the LBP. Determines the distance of the sampling points from the center pixel.
        neighbors (int, optional): The number of sampling points in the LBP.
        mapping (dict or None, optional): The mapping information for LBP codes. Should contain 'samples' and 'table' if provided. If `None`, no mapping is applied.
        mode (str, optional): The mode for LBP calculation. Options are:
            'h' (histogram): Returns LBP histogram.
            'hist' (histogram): Same as 'h', returns LBP histogram.
            'nh' (normalized histogram): Returns normalized LBP histogram. Default is 'nh'.

    Returns:
        tuple: A tuple containing:
            result (numpy.ndarray): The LBP histogram or LBP image based on the `mode` parameter.
            codeImage (numpy.ndarray): The LBP code image, which contains the LBP codes for each pixel.

    Raises:
        ValueError: If the number of input arguments is incorrect or if the provided `mapping` is incompatible with the number of `neighbors`.
        ValueError: If the input image is too small for the given `radius`.
        ValueError: If the dimensions of `spoints` are not valid.

    Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)
        >>> result, codeImage = descriptor_LBP(image, 1, 8, None, 'nh')
        >>> print(result)
        >>> print(codeImage)
    """
    # Check the number of input arguments
    if len(varargin) < 1 or len(varargin) > 5:
        raise ValueError("Wrong number of input arguments")

    image = varargin[0]

    if len(varargin) == 1:
        # Default parameters
        spoints = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors = 8
        mapping = get_mapping(8, 'riu2')
        mode = 'nh'

    if (len(varargin) == 2) and (len(str(varargin[1])) == 1):
        raise ValueError('Input arguments')

    if (len(varargin) > 2) and (len(str(varargin[1])) == 1):
        radius = varargin[1]
        neighbors = varargin[2]

        spoints = np.zeros((neighbors, 2))

        # Angle step
        a = 2 * np.pi / neighbors

        for i in range(neighbors):
            spoints[i, 0] = -radius * np.sin((i - 1) * a)
            spoints[i, 1] = radius * np.cos((i - 1) * a)

        if len(varargin) >= 4:
            mapping = varargin[3]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 5:
            mode = varargin[4]
        else:
            mode = 'h'

    if (len(varargin) > 1) and (len(str(varargin[1])) > 1):
        spoints = varargin[1]
        neighbors = spoints.shape[0]

        if len(varargin) >= 3:
            mapping = varargin[2]
            if isinstance(mapping, dict) and mapping['samples'] != neighbors:
                raise ValueError('Incompatible mapping')
        else:
            mapping = 0

        if len(varargin) >= 4:
            mode = varargin[3]
        else:
            mode = 'nh'

    # Determine the dimensions of the input image
    ysize, xsize = image.shape

    miny = np.min(spoints[:, 0])
    maxy = np.max(spoints[:, 0])
    minx = np.min(spoints[:, 1])
    maxx = np.max(spoints[:, 1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey = np.ceil(max(maxy, 0)) - np.floor(min(miny, 0))
    bsizex = np.ceil(max(maxx, 0)) - np.floor(min(minx, 0))

    # Coordinates of origin (0,0) in the block
    origy = int(1 - np.floor(min(miny, 0)))
    origx = int(1 - np.floor(min(minx, 0)))

    # Minimum allowed size for the input image depends on the radius of the used LBP operator
    if xsize < bsizex or ysize < bsizey:
        raise ValueError("Too small input image. Should be at least (2*radius+1) x (2*radius+1)")

    # Calculate dx and dy
    dx = int(xsize - bsizex)
    dy = int(ysize - bsizey)

    # Fill the center pixel matrix C
    C = image[origy - 1:origy + dy - 1, origx - 1:origx + dx - 1]
    d_C = np.double(C)

    bins = 2 ** neighbors

    # Initialize the result matrix with zeros
    result = np.zeros((dy, dx))

    # Compute the LBP code image
    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx
        # Calculate floors, ceils and rounds for the x and y
        fy = int(np.floor(y))
        cy = int(np.ceil(y))
        ry = int(np.round(y))
        fx = int(np.floor(x))
        cx = int(np.ceil(x))
        rx = int(np.round(x))
        # Check if interpolation is needed
        if (np.abs(x - rx) < 1e-6) and (np.abs(y - ry) < 1e-6):
            # Interpolation is not needed, use original datatypes
            N = image[ry - 1:ry + dy - 1, rx - 1:rx + dx - 1]
            D = N >= C
        else:
            # Interpolation needed, use double type images
            ty = y - fy
            tx = x - fx

            # Calculate the interpolation weights
            w1 = roundn((1 - tx) * (1 - ty), -6)
            w2 = roundn(tx * (1 - ty), -6)
            w3 = roundn((1 - tx) * ty, -6)
            w4 = roundn(1 - w1 - w2 - w3, -6)

            # Compute interpolated pixel values
            N = w1 * image[fy - 1:fy + dy - 1, fx - 1:fx + dx - 1] + w2 * image[fy - 1:fy + dy - 1, cx - 1:cx + dx - 1] + \
                w3 * image[cy - 1:cy + dy - 1, fx - 1:fx + dx - 1] + w4 * image[cy - 1:cy + dy - 1, cx - 1:cx + dx - 1]
            N = roundn(N, -4)
            D = N >= d_C

        # Update the result matrix
        v = 2 ** i
        result = result + v * D

    # Apply mapping if it is defined
    if isinstance(mapping, dict):
        bins = mapping['num']
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = mapping['table'][int(result[i, j])]

    codeImage = result

    if mode in ['h', 'hist', 'nh']:
        # Return with LBP histogram if mode equals 'hist'
        result = np.histogram(result, bins=np.arange(bins + 1))[0]
        if mode == 'nh':
            result = result / np.sum(result)
    else:
        # Otherwise return a matrix of unsigned integers
        if bins - 1 <= np.iinfo(np.uint8).max:
            result = result.astype(np.uint8)
        elif bins - 1 <= np.iinfo(np.uint16).max:
            result = result.astype(np.uint16)
        else:
            result = result.astype(np.uint32)

    return result, codeImage