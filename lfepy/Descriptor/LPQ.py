from lfepy.Helper.helper import np, descriptor_LPQ
from PIL import Image
import matplotlib.pyplot as plt


def LPQ(image, **kwargs):
    """
        Compute Local Phase Quantization (LPQ) histograms and descriptors from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing LPQ extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
                - windowSize (int): Size of the sliding window for LPQ. Default: 5.

        Returns:
            - LPQ_hist (numpy.ndarray): Histogram of LPQ descriptors.
            - imgDesc (numpy.ndarray): LPQ descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = LPQ(image, mode='nh', windowSize=5)
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - V. Ojansivu, and J. Heikkilä, Blur insensitive texture classification using local phase quantization, International conference on image and signal processing, Springer, 2008, pp. 236-243.
            - A. Dhall, A. Asthana, R. Goecke, and T. Gedeon, Emotion recognition using PHOG and LPQ features, Automatic Face & Gesture Recognition and Workshops (FG 2011), 2011 IEEE International Conference on, IEEE, 2011, pp. 878-883.
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

    wSz = options.get('windowSize', 5)

    imgDesc, _ = descriptor_LPQ(image, wSz)

    options['binVec'] = np.arange(256)

    # Compute LPQ histogram
    LPQ_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LPQ_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LPQ_hist = LPQ_hist / np.sum(LPQ_hist)

    return LPQ_hist, imgDesc
