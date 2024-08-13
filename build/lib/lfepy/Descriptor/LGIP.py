from lfepy.Helper.helper import np
from PIL import Image
import matplotlib.pyplot as plt


def LGIP(image, **kwargs):
    """
        Compute Local Gradient Increasing Pattern (LGIP) descriptors and histograms from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing LGIP extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.

        Returns:
            - LGIP_hist (numpy.ndarray): Histogram(s) of LGIP descriptors.
            - imgDesc (numpy.ndarray): LGIP descriptors themselves.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = LGIP(image, mode='nh')
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - Z. Lubing, and W. Han, Local gradient increasing pattern for facial expression recognition, Image Processing (ICIP), 2012 19th IEEE International Conference on, IEEE, 2012, pp. 2601-2604.
    """
    # Input validation
    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("The image must be a valid numpy.ndarray.")

    # Convert the input image to double precision
    image = np.double(image)

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

    # Get image dimensions
    r, c = image.shape

    # Compute LGIP components
    v000 = np.double(-image[1:-3, 2:-2] + image[1:-3, 3:-1] - 2 * image[2:-2, 2:-2] +
                     2 * image[2:-2, 3:-1] - image[3:-1, 2:-2] + image[3:-1, 3:-1] > 0)
    v001 = np.double(-image[1:-3, 1:-3] + image[:-4, 2:-2] - 2 * image[2:-2, 2:-2] +
                     2 * image[1:-3, 3:-1] - image[3:-1, 3:-1] + image[2:-2, 4:])
    v010 = np.double(-image[2:-2, 1:-3] + image[1:-3, 1:-3] - 2 * image[2:-2, 2:-2] +
                     2 * image[1:-3, 2:-2] - image[2:-2, 3:-1] + image[1:-3, 3:-1] > 0)
    v011 = np.double(-image[3:-1, 1:-3] + image[2:-2, 0:-4] - 2 * image[2:-2, 2:-2] +
                     2 * image[1:-3, 1:-3] - image[1:-3, 3:-1] + image[0:-4, 2:-2] > 0)
    v100 = np.double(-image[1:-3, 2:-2] + image[1:-3, 1:-3] - 2 * image[2:-2, 2:-2] +
                     2 * image[2:-2, 1:-3] - image[3:-1, 2:-2] + image[3:-1, 1:-3] > 0)
    v101 = np.double(-image[1:-3, 1:-3] + image[2:-2, 0:-4] - 2 * image[2:-2, 2:-2] +
                     2 * image[3:-1, 1:-3] - image[3:-1, 3:-1] + image[4:, 2:-2] > 0)
    v110 = np.double(-image[2:-2, 1:-3] + image[3:-1, 1:-3] - 2 * image[2:-2, 2:-2] +
                     2 * image[3:-1, 2:-2] - image[2:-2, 3:-1] + image[3:-1, 3:-1] > 0)
    v111 = np.double(-image[3:-1, 1:-3] + image[4:, 2:-2] - 2 * image[2:-2, 2:-2] +
                     2 * image[3:-1, 3:-1] - image[1:-3, 3:-1] + image[2:-2, 4:] > 0)

    # Compute Orientation Tensor Vectors (OTV)
    OTVx = np.ravel(v000 + v001 + v111 - v011 - v100 - v101)
    OTVy = np.ravel(v001 + v010 + v011 - v101 - v110 - v111)

    # Define pattern mask
    patternMask = np.array([[-1, -1, 30, 29, 28, -1, -1],
                            [-1, 16, 15, 14, 13, 12, -1],
                            [31, 17, 4, 3, 2, 11, 27],
                            [32, 18, 5, 0, 1, 10, 26],
                            [33, 19, 6, 7, 8, 9, 25],
                            [-1, 20, 21, 22, 23, 24, -1],
                            [-1, -1, 34, 35, 36, -1, -1]])

    # Clip OTV values to be within the pattern mask range
    OTVx_clipped = np.clip(OTVx + 4, 0, 6).astype(int)
    OTVy_clipped = np.clip(OTVy + 4, 0, 6).astype(int)

    # Map the OTV values to pattern mask indices
    idx = np.ravel_multi_index((OTVx_clipped, OTVy_clipped), patternMask.shape)
    LGIP = patternMask.flat[idx]
    imgDesc = np.reshape(LGIP, (r - 4, c - 4))

    # Set bin vectors
    options['binVec'] = np.arange(0, 37)

    # Compute LGIP histogram
    LGIP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LGIP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LGIP_hist = LGIP_hist / np.sum(LGIP_hist)

    return LGIP_hist, imgDesc
