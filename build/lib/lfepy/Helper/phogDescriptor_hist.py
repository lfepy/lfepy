import numpy as np


def phogDescriptor_hist(bh, bv, L, bin):
    """
    Compute the histogram of the Pyramid Histogram of Oriented Gradients (PHOG) descriptor.

    This function calculates the PHOG descriptor histogram by computing histograms for multiple pyramid levels,
    where each level represents different spatial resolutions. It uses the bin matrix `bh` to determine the orientation
    of gradients and the gradient magnitude matrix `bv` to compute the histogram values.

    Args:
        bh (numpy.ndarray): Bin matrix of the image, where each pixel is assigned a bin index.
        bv (numpy.ndarray): Gradient magnitude matrix corresponding to the bin matrix.
        L (int): Number of pyramid levels.
        bin (int): Number of bins for the histogram.

    Returns:
        numpy.ndarray: Normalized histogram of the PHOG descriptor.

    Example:
        >>> import numpy as np
        >>> bh = np.array([[1, 2], [2, 1]])
        >>> bv = np.array([[1, 2], [2, 1]])
        >>> L = 2
        >>> bin = 4
        >>> phog_hist = phogDescriptor_hist(bh, bv, L, bin)
        >>> print(phog_hist)
        [0.1 0.2 0.2 0.1 0.1 0.1 0.1 0.1]
    """
    p = []

    # Compute histogram for level 0 (original image)
    for b in range(1, bin + 1):
        ind = (bh == b)
        p.append(np.sum(bv[ind]))

    # Compute histograms for pyramid levels
    for l in range(1, L + 1):
        x = bh.shape[1] // (2 ** l)  # Width of each cell at level l
        y = bh.shape[0] // (2 ** l)  # Height of each cell at level l
        xx = 0

        # Traverse through the image cells at level l
        while xx + x <= bh.shape[1]:
            yy = 0
            while yy + y <= bh.shape[0]:
                # Extract cell regions for bin and magnitude
                bh_cella = bh[yy:yy + y, xx:xx + x]
                bv_cella = bv[yy:yy + y, xx:xx + x]

                # Compute histogram for each cell
                for b in range(1, bin + 1):
                    ind = (bh_cella == b)
                    p.append(np.sum(bv_cella[ind]))

                yy += y
            xx += x

    # Convert list to numpy array
    p = np.array(p)

    # Normalize the histogram
    if np.sum(p) != 0:
        p = p / np.sum(p)

    return p