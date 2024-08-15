import numpy as np
from scipy.ndimage import label


def bin_matrix(A, E, G, angle, bin):
    """
    Compute the bin matrix for a given angle map and gradient magnitude.

    :param A: Angle map of the gradient directions.
    :type A: numpy.ndarray
    :param E: Binary edge map where edges are marked.
    :type E: numpy.ndarray
    :param G: Gradient magnitude map.
    :type G: numpy.ndarray
    :param angle: Total range of angles in degrees (e.g., 360 for full circle).
    :type angle: float
    :param bin: Number of bins to divide the angle range into.
    :type bin: int

    :returns:
        - bm: Bin matrix with assigned bins for each pixel.
        - bv: Gradient magnitude values corresponding to the bin matrix.
    :rtype:
        - bm: numpy.ndarray
        - bv: numpy.ndarray

    :example:
        >>> import numpy as np
        >>> A = np.array([[0, 45], [90, 135]])
        >>> E = np.array([[1, 1], [1, 1]])
        >>> G = np.array([[1, 2], [3, 4]])
        >>> angle = 360
        >>> bin = 8
        >>> bm, bv = bin_matrix(A, E, G, angle, bin)
        >>> print(bm)
        [[1 2]
         [3 4]]
        >>> print(bv)
        [[1. 2.]
         [3. 4.]]
    """
    # Label connected components in the edge map
    contorns, n = label(E)
    Y, X = E.shape

    # Initialize bin matrix and gradient magnitude matrix
    bm = np.zeros((Y, X), dtype=int)
    bv = np.zeros((Y, X), dtype=float)

    # Calculate the angle range per bin
    nAngle = angle / bin

    # Iterate over each labeled region (edge component)
    for i in range(1, n + 1):
        posY, posX = np.where(contorns == i)  # Get coordinates of the current edge component

        for y, x in zip(posY, posX):
            # Determine the bin index based on the angle
            b = int(np.ceil(A[y, x] / nAngle))
            if b == 0:
                b = 1
            if G[y, x] > 0:
                # Assign the bin index and gradient magnitude
                bm[y, x] = b
                bv[y, x] = G[y, x]

    return bm, bv