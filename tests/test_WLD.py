import unittest
import numpy as np
from lfepy.Descriptor import WLD  # Replace with the actual module name


class TestWLD(unittest.TestCase):

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

    def test_wld_default_params(self):
        # Test WLD with default parameters
        wld_hist, imgDesc = WLD(self.image)
        self.assertIsInstance(wld_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)  # Check that imgDesc is not empty
        self.assertTrue(len(wld_hist) > 0)  # Check that histogram is not empty
        self.assertIn('fea', imgDesc[0])
        self.assertIn('fea', imgDesc[1])
        self.assertIn('GO', imgDesc[0]['fea'])
        self.assertIn('DE', imgDesc[1]['fea'])
        self.assertIsInstance(imgDesc[0]['fea']['GO'], np.ndarray)
        self.assertIsInstance(imgDesc[1]['fea']['DE'], np.ndarray)

    def test_wld_custom_params(self):
        # Test WLD with custom parameters
        wld_hist, imgDesc = WLD(self.image, mode='h', T=16, N=8, scaleTop=2)
        self.assertIsInstance(wld_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(len(wld_hist) > 0)

    def test_wld_invalid_mode(self):
        # Test WLD with an invalid mode
        with self.assertRaises(ValueError):
            WLD(self.image, mode='invalid_mode')

    def test_wld_with_none_image(self):
        # Test WLD with None as input
        with self.assertRaises(TypeError):
            WLD(None)

    def test_wld_with_non_array_image(self):
        # Test WLD with a non-numpy array image
        with self.assertRaises(TypeError):
            WLD("invalid_image")

    def test_wld_feature_extraction(self):
        # Check if the feature extraction part of WLD works
        wld_hist, imgDesc = WLD(self.image)
        self.assertTrue(len(wld_hist) > 0)
        self.assertTrue(len(imgDesc) > 0)
        self.assertIn('fea', imgDesc[0])
        self.assertIn('fea', imgDesc[1])
        self.assertIn('GO', imgDesc[0]['fea'])
        self.assertIn('DE', imgDesc[1]['fea'])
        self.assertIsInstance(imgDesc[0]['fea']['GO'], np.ndarray)
        self.assertIsInstance(imgDesc[1]['fea']['DE'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
