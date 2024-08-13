from lfepy.Helper.helper import np, convolve2d
from lfepy.Descriptor.LTeP import LTeP
from PIL import Image
import matplotlib.pyplot as plt


def GLTP(image, **kwargs):
    """
        Compute Gradient-based Local Ternary Pattern (GLTP) histograms and descriptors from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing GLTP extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.
                - t (int): Threshold value for ternary pattern computation. Default: 10.
                - DGLP (int): Flag to include Directional Gradient-based Local Pattern. If set to 1, includes DGLP. Default: 0.

        Returns:
            - GLTP_hist (numpy.ndarray): Histogram(s) of GLTP descriptors.
            - imgDesc (list): List of dictionaries containing GLTP descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = GLTP(image, mode='nh', t=10, DGLP=1)
            plt.imshow(imgDesc[0]['fea'], cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - M. Valstar, and M. Pantic, Fully automatic facial action unit detection and temporal analysis, Computer Vision and Pattern Recognition Workshop, 2006. CVPRW'06. Conference on, IEEE, 2006, pp. 149-149.
            - F. Ahmed, and E. Hossain, Automated facial expression recognition using gradient-based ternary texture patterns. Chinese Journal of Engineering 2013 (2013).
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

    # Extract the DGLP
    if 'DGLP' not in options:
        options.update({'DGLP': 0})

    # Validate the DGLP
    valid_DGLP = [0, 1]
    if options['DGLP'] not in valid_DGLP:
        raise ValueError(f"Invalid DGLP '{options['DGLP']}'. Valid DGLP are {valid_DGLP}.")

    EPSILON = 1e-7

    # Define Sobel masks for gradient computation
    maskA = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    maskB = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Compute gradients
    Gx = convolve2d(image, maskA, mode='same')
    Gy = convolve2d(image, maskB, mode='same')

    # Compute gradient magnitude
    img_gradient = np.abs(Gx) + np.abs(Gy)

    # Compute Local Ternary Pattern (LTeP) on gradient image
    _, imgDesc = LTeP(img_gradient, t=options.get('t', 10))
    options['binVec'] = [np.arange(256) for _ in range(2)]

    # If DGLP flag is set, include directional gradient pattern
    if options['DGLP'] == 1:
        r, c = Gx.shape
        img_angle = np.arctan2(Gy, Gx + EPSILON)
        img_angle = np.degrees(img_angle)
        img_angle[Gx < 0] += 180
        img_angle[(Gx >= 0) & (Gy < 0)] += 360
        img_angle = img_angle[1:r-1, 1:c-1]
        img_angle = np.floor(img_angle / 22.5).astype(int)

        imgDesc.append({'fea': img_angle})
        options['binVec'].append(np.arange(16))

    # Compute GLTP histogram
    GLTP_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            GLTP_hist.append(hh)
    GLTP_hist = np.array(GLTP_hist)
    if 'mode' in options and options['mode'] == 'nh':
        GLTP_hist = GLTP_hist / np.sum(GLTP_hist)

    return GLTP_hist, imgDesc
