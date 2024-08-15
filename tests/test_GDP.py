import unittest
import numpy as np
from lfepy.Descriptor import GDP  # Replace with the actual module name


class TestGDP(unittest.TestCase):

    def setUp(self):
        # Create a sample image for testing (e.g., 8x8 grayscale image)
        self.image = np.array([
            [52, 55, 61, 59, 79, 61, 76, 61],
            [62, 59, 55, 104, 94, 85, 59, 71],
            [63, 65, 66, 113, 144, 104, 63, 72],
            [64, 70, 70, 126, 154, 109, 71, 69],
            [67, 73, 68, 106, 122, 88, 68, 68],
            [68, 79, 60, 70, 77, 66, 58, 75],
            [69, 85, 64, 58, 55, 61, 65, 83],
            [70, 87, 69, 68, 65, 73, 78, 90]
        ], dtype=np.uint8)

    def test_gdp_default_mode(self):
        # Test GDP with default parameters
        gdp_hist, imgDesc = GDP(self.image)
        self.assertIsInstance(gdp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(gdp_hist.ndim, 1)  # Should be a 1D array

    def test_gdp_histogram_mode(self):
        # Test GDP with histogram mode ('h')
        gdp_hist, imgDesc = GDP(self.image, mode='h')
        self.assertIsInstance(gdp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(gdp_hist.ndim, 1)  # Should be a 1D array

    def test_gdp_normalization_mode(self):
        # Test if the GDP histogram is normalized in 'nh' mode
        gdp_hist, _ = GDP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(gdp_hist), 1.0)

    def test_gdp_mask_sobel(self):
        # Test GDP with Sobel mask
        gdp_hist, imgDesc = GDP(self.image, mask='sobel')
        self.assertIsInstance(gdp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)

    def test_gdp_mask_prewitt(self):
        # Test GDP with Prewitt mask
        gdp_hist, imgDesc = GDP(self.image, mask='prewitt')
        self.assertIsInstance(gdp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)

    def test_gdp_threshold(self):
        # Test GDP with a custom threshold value
        gdp_hist, imgDesc = GDP(self.image, t=30.0)
        self.assertIsInstance(gdp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)

    def test_gdp_invalid_mode(self):
        # Test GDP with an invalid mode
        with self.assertRaises(ValueError):
            GDP(self.image, mode='invalid_mode')

    def test_gdp_invalid_mask(self):
        # Test GDP with an invalid mask
        with self.assertRaises(ValueError):
            GDP(self.image, mask='invalid_mask')

    def test_gdp_with_none_image(self):
        # Test GDP with None as input
        with self.assertRaises(TypeError):
            GDP(None)


if __name__ == '__main__':
    unittest.main()
