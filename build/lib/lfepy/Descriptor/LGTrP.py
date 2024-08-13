from lfepy.Helper.helper import np, gabor_filter
from lfepy.Descriptor.LTrP import LTrP
from PIL import Image
import matplotlib.pyplot as plt


def LGTrP(image, **kwargs):
    """
        Compute Local Gabor Transitional Pattern (LGTrP) histograms and descriptors from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing LGTrP extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.

        Returns:
            - LGTrP_hist (numpy.ndarray): Histogram(s) of LGTrP descriptors.
            - imgDesc (numpy.ndarray): LGTrP descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = LGTrP(image, mode='nh')
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - M.S. Islam, Local gradient pattern-A novel feature representation for facial expression recognition. Journal of AI and Data Mining 2 (2014) 33-38.
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
        options['mode'] = 'nh'

    # Validate the mode
    valid_modes = ['nh', 'h']
    if options['mode'] not in valid_modes:
        raise ValueError(f"Invalid mode '{options['mode']}'. Valid options are {valid_modes}.")

    # Compute LGTrP descriptor
    gaborMag = np.abs(gabor_filter(image, 8, 1))
    gaborTotal = gaborMag[:, :, 0, 0]

    for o in range(8):
        gaborTotal += gaborMag[:, :, o, 0]

    imgDescGabor = gaborTotal / 8
    _, imgDesc = LTrP(imgDescGabor)

    # Set bin vector
    options['binVec'] = np.arange(256)

    # Compute LGTrP histogram
    LGTrP_hist = np.zeros(len(options['binVec']))
    for i, bin_val in enumerate(options['binVec']):
        LGTrP_hist[i] = np.sum([imgDesc == bin_val])
    if 'mode' in options and options['mode'] == 'nh':
        LGTrP_hist = LGTrP_hist / np.sum(LGTrP_hist)

    return LGTrP_hist, imgDesc
