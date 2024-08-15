import numpy as np
from scipy.fft import fft2, ifft2, ifftshift
from lfepy.Helper.low_pass_filter import low_pass_filter


def phase_cong3(image, nscale=4, norient=6, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
                dThetaOnSigma=1.5, k=2.0, cutOff=0.5, g=10):
    """
    Computes the phase congruency of an image using a multiscale, multi-orientation approach.
    Phase congruency is a measure of the image's local contrast, based on the phase information of its frequency components.
    This method is used for edge detection and texture analysis.

    :param image: Input grayscale image as a 2D numpy array.
    :type image: numpy.ndarray
    :param nscale: Number of scales to be used in the analysis (default is 4).
    :type nscale: int, optional
    :param norient: Number of orientations to be used in the analysis (default is 6).
    :type norient: int, optional
    :param minWaveLength: Minimum wavelength of the log-Gabor filters (default is 3).
    :type minWaveLength: float, optional
    :param mult: Scaling factor for the wavelength of the log-Gabor filters (default is 2.1).
    :type mult: float, optional
    :param sigmaOnf: Standard deviation of the Gaussian function used in the log-Gabor filter (default is 0.55).
    :type sigmaOnf: float, optional
    :param dThetaOnSigma: Angular spread of the Gaussian function relative to the orientation (default is 1.5).
    :type dThetaOnSigma: float, optional
    :param k: Constant to adjust the threshold for noise (default is 2.0).
    :type k: float, optional
    :param cutOff: Cut-off parameter for weighting function (default is 0.5).
    :type cutOff: float, optional
    :param g: Gain parameter for the weighting function (default is 10).
    :type g: float, optional

    :returns: A tuple containing:
        - M: The measure of local phase congruency.
        - m: The measure of local phase concavity.
        - ori: Orientation of the phase congruency.
        - featType: Feature type (complex representation of phase congruency).
        - PC: List of phase congruency maps for each orientation.
        - EO: List of complex responses for each scale and orientation.
    :rtype: tuple of numpy.ndarray and list

    :raises ValueError: If the input image is not a 2D numpy array.

    :notes:
        - The function assumes the input image is grayscale.
        - The log-Gabor filters are used to analyze the image at different scales and orientations.
        - The phase congruency is computed based on the response of these filters, and noise is estimated and thresholded.
        - The result includes orientation information, which can be useful for edge detection and texture analysis.

    :example:
        >>> import numpy as np
        >>> image = np.random.rand(256, 256)
        >>> M, m, ori, featType, PC, EO = phase_cong3(image)
    """
    epsilon = .0001

    thetaSigma = np.pi / norient / dThetaOnSigma

    rows, cols = image.shape
    imagefft = fft2(image)

    zero = np.zeros((rows, cols))
    EO = [[None] * norient for _ in range(nscale)]
    covx2 = zero.copy()
    covy2 = zero.copy()
    covxy = zero.copy()

    estMeanE2n = []
    PC = []
    ifftFilterArray = [None] * nscale

    if cols % 2:
        xrange = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) / (cols - 1)
    else:
        xrange = np.linspace(-cols / 2, cols / 2 - 1, cols) / cols

    if rows % 2:
        yrange = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) / (rows - 1)
    else:
        yrange = np.linspace(-rows / 2, rows / 2 - 1, rows) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    del x, y, theta

    lp = low_pass_filter([rows, cols], .45, 15)
    logGabor = [None] * nscale

    for s in range(nscale):
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] *= lp
        logGabor[s][0, 0] = 0

    spread = [None] * norient

    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    for o in range(norient):
        angl = o * np.pi / norient
        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        sumAn_ThisOrient = zero.copy()
        Energy = zero.copy()

        for s in range(nscale):
            filter_ = logGabor[s] * spread[o]
            ifftFilt = np.real(ifft2(filter_)) * np.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            EO[s][o] = ifft2(imagefft * filter_)
            An = np.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO[s][o])
            sumO_ThisOrient += np.imag(EO[s][o])

            if s == 0:
                EM_n = np.sum(filter_ ** 2)
                maxAn = An
            else:
                maxAn = np.maximum(maxAn, An)

        XEnergy = np.sqrt(sumE_ThisOrient ** 2 + sumO_ThisOrient ** 2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for s in range(nscale):
            E = np.real(EO[s][o])
            O = np.imag(EO[s][o])
            Energy += E * MeanE + O * MeanO - np.abs(E * MeanO - O * MeanE)

        medianE2n = np.median(np.abs(EO[0][o]) ** 2)
        meanE2n = -medianE2n / np.log(0.5)
        estMeanE2n.append(meanE2n)

        noisePower = meanE2n / EM_n
        EstSumAn2 = zero.copy()
        for s in range(nscale):
            EstSumAn2 += ifftFilterArray[s] ** 2

        EstSumAiAj = zero.copy()
        for si in range(nscale - 1):
            for sj in range(si + 1, nscale):
                EstSumAiAj += ifftFilterArray[si] * ifftFilterArray[sj]

        sumEstSumAn2 = np.sum(EstSumAn2)
        sumEstSumAiAj = np.sum(EstSumAiAj)

        EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj
        tau = np.sqrt(EstNoiseEnergy2 / 2)
        EstNoiseEnergy = tau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = np.sqrt((2 - np.pi / 2) * tau ** 2)
        T = EstNoiseEnergy + k * EstNoiseEnergySigma
        T /= 1.7

        Energy = np.maximum(Energy - T, zero)

        width = sumAn_ThisOrient / (maxAn + epsilon) / nscale
        weight = 1.0 / (1 + np.exp((cutOff - width) * g))

        PC.append(weight * Energy / sumAn_ThisOrient)
        featType = E + 1j * O

        covx = PC[o] * np.cos(angl)
        covy = PC[o] * np.sin(angl)
        covx2 += covx ** 2
        covy2 += covy ** 2
        covxy += covx * covy

    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy *= 4 / norient

    denom = np.sqrt(covxy ** 2 + (covx2 - covy2) ** 2) + epsilon
    sin2theta = covxy / denom
    cos2theta = (covx2 - covy2) / denom
    ori = np.arctan2(sin2theta, cos2theta) / 2
    ori = np.rad2deg(ori)
    ori[ori < 0] += 180

    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    return M, m, ori, featType, PC, EO
