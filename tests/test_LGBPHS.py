import unittest
import numpy as np
from lfepy.Descriptor import LGBPHS  # Replace with the actual module name


class TestLGBPHS(unittest.TestCase):

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

    def test_lgbphs_default_params(self):
        # Test LGBPHS with default parameters
        lgbphs_hist, imgDesc = LGBPHS(self.image)
        self.assertIsInstance(lgbphs_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(lgbphs_hist.ndim == 1)  # Should be a 1D array

    def test_lgbphs_custom_params(self):
        # Test LGBPHS with custom parameters
        lgbphs_hist, imgDesc = LGBPHS(self.image, mode='h', uniformLBP=0, scaleNum=3, orienNum=4)
        self.assertIsInstance(lgbphs_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(lgbphs_hist.ndim == 1)  # Should be a 1D array

    def test_lgbphs_histogram_mode(self):
        # Test LGBPHS with histogram mode ('h')
        lgbphs_hist, imgDesc = LGBPHS(self.image, mode='h')
        self.assertIsInstance(lgbphs_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(lgbphs_hist.ndim == 1)  # Should be a 1D array

    def test_lgbphs_normalization_mode(self):
        # Test if the LGBPHS histogram is normalized in 'nh' mode
        lgbphs_hist, _ = LGBPHS(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(lgbphs_hist), 1.0)

    def test_lgbphs_invalid_mode(self):
        # Test LGBPHS with an invalid mode
        with self.assertRaises(ValueError):
            LGBPHS(self.image, mode='invalid_mode')

    def test_lgbphs_with_none_image(self):
        # Test LGBPHS with None as input
        with self.assertRaises(TypeError):
            LGBPHS(None)

    def test_lgbphs_with_non_array_image(self):
        # Test LGBPHS with a non-numpy array image
        with self.assertRaises(TypeError):
            LGBPHS("invalid_image")

    def test_lgbphs_feature_extraction(self):
        # Check if the feature extraction part of LGBPHS works
        lgbphs_hist, imgDesc = LGBPHS(self.image)
        self.assertTrue(len(lgbphs_hist) > 0)
        self.assertTrue(len(imgDesc) > 0)
        for desc in imgDesc:
            self.assertIn('fea', desc)
            self.assertEqual(desc['fea'].ndim, 2)
            self.assertEqual(desc['fea'].dtype, np.float64)


if __name__ == '__main__':
    unittest.main()
