from lfepy.Helper.helper import np, get_mapping, phase_cong3, descriptor_LBP
from PIL import Image
import matplotlib.pyplot as plt


def BPPC(image, **kwargs):
    """
        Compute Binary Phase Pattern Concatenation (BPPC) histograms and descriptors from an input image.

        Parameters:
            - image (numpy.ndarray): Input image (preferably in NumPy array format).
            - **kwargs (dict): Additional keyword arguments for customizing BPPC extraction.
                - mode (str): Mode for histogram computation. Options: 'nh' (normalized histogram) or 'h' (histogram). Default: 'nh'.

        Returns:
            - BPPC_hist (numpy.ndarray): Histogram(s) of BPPC descriptors.
            - imgDesc (list): List of dictionaries containing BPPC descriptors.

        Example:
            image = Image.open(Path)
            histogram, imgDesc = BPPC(image, mode='nh')
            plt.imshow(imgDesc[0]['fea'], cmap='gray')
            plt.axis('off')
            plt.show()

        References:
            - S. Shojaeilangari, W.-Y. Yau, J. Li, and E.-K. Teoh, Feature extraction through binary pattern of phase congruency for facial expression recognition, Control Automation Robotics & Vision (ICARCV), 2012 12th International Conference on, IEEE, 2012, pp. 166-170.
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

    options['binVec'] = []

    # Compute phase congruency
    _, _, phaseAngle2, _, pc, EO = phase_cong3(image, 4, 6, 3)
    imgDesc = []
    phaseAngle = phaseAngle2[1:-1, 1:-1]

    # Compute BPPC descriptors
    for o in range(6):
        imgDesc.append({'pc': pc[o]})
        mapping = get_mapping(8, 'u2')
        _, codeImg = descriptor_LBP(imgDesc[o]['pc'], 1, 8, mapping, 'nh')

        angleInd = np.floor(phaseAngle / 60)
        imgDesc[o]['fea'] = codeImg + angleInd * 59
        options['binVec'].append(np.arange(177))

    # Compute BPPC histogram
    BPPC_hist = []
    for s in range(len(imgDesc)):
        imgReg = imgDesc[s]['fea']
        for i, bin_val in enumerate(options['binVec'][s]):
            hh = np.sum([imgReg == bin_val])
            BPPC_hist.append(hh)
    BPPC_hist = np.array(BPPC_hist)
    if 'mode' in options and options['mode'] == 'nh':
        BPPC_hist = BPPC_hist / np.sum(BPPC_hist)

    return BPPC_hist, imgDesc
