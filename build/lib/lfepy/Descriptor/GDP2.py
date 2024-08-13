from lfepy.Helper.helper import np
from PIL import Image
import matplotlib.pyplot as plt


def GDP2(image, **kwargs):
    """
        Compute Gradient Direction Pattern (GDP2) descriptors and histograms from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing GDP2 extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.

        Returns:
            - GDP2_hist (numpy.ndarray): Histogram(s) of GDP2 descriptors.
            - imgDesc (numpy.ndarray): GDP2 descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = GDP2(image, mode='nh')
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - M.S. Islam, Gender Classification using Gradient Direction Pattern. Science International 25 (2013).
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

    # Define link list
    linkList = [[[1, 1], [3, 3]], [[1, 2], [3, 2]], [[1, 3], [3, 1]], [[2, 3], [2, 1]]]

    # Compute pattern
    x_c = image[1:-1, 1:-1]
    rSize, cSize = x_c.shape

    # Initialize pattern with zeros
    pattern = np.zeros_like(x_c)

    for n in range(len(linkList)):
        corner1 = linkList[n][0]
        corner2 = linkList[n][1]
        x_1 = image[corner1[0] - 1:corner1[0] + rSize - 1, corner1[1] - 1:corner1[1] + cSize - 1]
        x_2 = image[corner2[0] - 1:corner2[0] + rSize - 1, corner2[1] - 1:corner2[1] + cSize - 1]
        pattern += np.double(((x_1 - x_2) >= 0) * 2 ** (len(linkList) - n - 1))

    imgDesc = pattern

    binNum = 2 ** len(linkList)
    transitionSelected = [0, 1, 3, 7, 8, 12, 14, 15]
    options['selected'] = transitionSelected

    # Set bin vectors
    options['binVec'] = np.arange(binNum)

    # Compute GDP2 histogram
    GDP2_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        GDP2_hist[i] = np.sum([imgDesc == bin_val])
    GDP2_hist = GDP2_hist[transitionSelected]
    if 'mode' in options and options['mode'] == 'nh':
        GDP2_hist = GDP2_hist / np.sum(GDP2_hist)

    return GDP2_hist, imgDesc
