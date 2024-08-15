import numpy as np


def LAP(image, **kwargs):
    """
    Compute Local Arc Pattern (LAP) descriptors and histograms from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LAP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LAP_hist: Histogram(s) of LAP descriptors.
        - imgDesc: List of dictionaries containing LAP descriptors.
    :rtype: tuple of (numpy.ndarray, list)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open("Path")
        >>> histogram, imgDesc = LAP(image, mode='nh')
        >>> plt.imshow(imgDesc[0]['fea'], cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        M.S. Islam, and S. Auwatanamongkol,
        Facial Expression Recognition using Local Arc Pattern.
        Trends in Applied Sciences Research 9 (2014) 113.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision
    image = np.double(image)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Handle keyword arguments
    if kwargs is None:
        options = {}
    else:
        options = kwargs

    # Extract histogram mode
    if 'mode' not in options:
        options.update({'mode': 'nh'})

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    # Define patterns and compute descriptors
    linkList1 = [[(2, 2), (4, 4)], [(2, 3), (4, 3)], [(2, 4), (4, 2)], [(3, 4), (3, 2)]]
    linkList2 = [[(1, 1), (5, 5)], [(1, 2), (5, 4)], [(1, 3), (5, 3)], [(1, 4), (5, 2)],
                 [(1, 5), (5, 1)], [(2, 5), (4, 1)], [(3, 5), (3, 1)], [(4, 5), (2, 1)]]
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape
    pattern1 = np.zeros_like(x_c)

    for n in range(len(linkList1)):
        corner1 = linkList1[n][0]
        corner2 = linkList1[n][1]
        x_1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        pattern1 += ((x_1 - x_2) > 0).astype(float) * 2 ** (len(linkList1) - n - 1)

    pattern2 = np.zeros_like(x_c)
    for n in range(len(linkList2)):
        corner1 = linkList2[n][0]
        corner2 = linkList2[n][1]
        x_1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        pattern2 += ((x_1 - x_2) > 0).astype(float) * 2 ** (len(linkList2) - n - 1)

    imgDesc = [{'fea': pattern1}, {'fea': pattern2}]

    # Set bin vectors
    binVec = [np.arange(0, 2 ** len(linkList1)), np.arange(0, 2 ** len(linkList2))]
    options['binVec'] = binVec

    # Compute LAP histogram
    LAP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LAP_hist.append(hh)
    LAP_hist = np.array(LAP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LAP_hist = LAP_hist / np.sum(LAP_hist)

    return LAP_hist, imgDesc