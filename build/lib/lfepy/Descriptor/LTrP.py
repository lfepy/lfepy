from lfepy.Helper.helper import np


def LTrP(image, **kwargs):
    """
    Compute Local Transitional Pattern (LTrP) histograms and descriptors from an input image.

    :param image: Input image (preferably in NumPy array format).
    :type image: numpy.ndarray
    :param kwargs: Additional keyword arguments for customizing LTrP extraction.
    :type kwargs: dict
    :param kwargs.mode: Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
    :type kwargs.mode: str

    :returns:
        - LTrP_hist: Histogram(s) of LTrP descriptors.
        - imgDesc: LTrP descriptors.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)

    :example:
        >>> from PIL import Image
        >>> import matplotlib.pyplot as plt
        >>> image = Image.open(Path)
        >>> histogram, imgDesc = LTrP(image, mode='nh')
        >>> plt.imshow(imgDesc, cmap='gray')
        >>> plt.axis('off')
        >>> plt.show()

    :references:
        T. Jabid, and O. Chae,
        Local Transitional Pattern: A Robust Facial Image Descriptor for Automatic Facial Expression Recognition,Proc.
        International Conference on Computer Convergence Technology,
        Seoul, Korea, 2011, pp. 333-44.

        T. Jabid, and O. Chae,
        Facial Expression Recognition Based on Local Transitional Pattern.
        International Information Institute (Tokyo).
        Information 15 (2012) 2007.
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

    # Define link list for LTrP computation
    link_list = [[[4, 4], [5, 5]], [[4, 3], [5, 3]], [[4, 2], [5, 1]], [[3, 2], [3, 1]],
                 [[2, 2], [1, 1]], [[2, 3], [1, 3]], [[2, 4], [1, 5]], [[3, 4], [3, 5]]]

    # Initialize variables
    x_c = image[2:-2, 2:-2]
    rSize, cSize = x_c.shape
    imgDesc = np.zeros_like(x_c)

    # Compute LTrP descriptors
    for n, corners in enumerate(link_list):
        corner1, corner2 = corners
        x_p1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_p2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        imgDesc += np.logical_xor((x_p1 - x_c) >= 0, (x_p2 - x_c) >= 0) * 2**(len(link_list) - n - 1)

    # Set bin vectors
    options['binVec'] = np.arange(256)

    # Compute LTrP histogram
    LTrP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LTrP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LTrP_hist = LTrP_hist / np.sum(LTrP_hist)

    return LTrP_hist, imgDesc
