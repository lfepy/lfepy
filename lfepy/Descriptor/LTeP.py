from lfepy.Helper.helper import np
from PIL import Image
import matplotlib.pyplot as plt


def LTeP(image, **kwargs):
    """
        Compute Local Ternary Pattern (LTeP) histograms and descriptors from an input image.

        Parameters:
            - img (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing LTeP extraction.
                - t (int): Threshold value for ternary pattern computation. Default: 2.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.

        Returns:
            - LTeP_hist (numpy.ndarray): Histogram(s) of LTeP descriptors.
            - imgDesc (list): List of dictionaries containing LTeP descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = LTeP(image, mode='nh', windowSize=5)
            plt.imshow(imgDesc, cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - F. Bashar, A. Khan, F. Ahmed, and M.H. Kabir, Robust facial expression recognition based on median ternary pattern (MTP), Electrical Information and Communication Technology (EICT), 2013 International Conference on, IEEE, 2014, pp. 1-5.
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

    # Extract threshold or use default
    t = options.get('t', 2)

    # Initialize variables
    rSize = image.shape[0] - 2
    cSize = image.shape[1] - 2

    # Define link list for LTeP computation
    link = np.array([[2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]])
    ImgIntensity = np.zeros((rSize * cSize, 8))

    # Compute LTeP descriptors
    for n in range(link.shape[0]):
        corner = link[n, :]
        ImgIntensity[:, n] = image[corner[0] - 1:corner[0] + rSize - 1, corner[1] - 1:corner[1] + cSize - 1].flatten()

    centerMat = image[1:-1, 1:-1].flatten()

    Pltp = np.double(ImgIntensity > (centerMat[:, None] + t))
    Nltp = np.double(ImgIntensity < (centerMat[:, None] - t))

    imgDesc = [{'fea': Pltp.dot(2**np.arange(Pltp.shape[-1])).reshape(rSize, cSize)},
               {'fea': Nltp.dot(2**np.arange(Nltp.shape[-1])).reshape(rSize, cSize)}]

    # Set bin vectors
    options['binVec'] = [np.arange(256), np.arange(256)]

    # Compute LTeP histogram
    LTeP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            LTeP_hist.append(hh)
    LTeP_hist = np.array(LTeP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        LTeP_hist = LTeP_hist / np.sum(LTeP_hist)

    return LTeP_hist, imgDesc
