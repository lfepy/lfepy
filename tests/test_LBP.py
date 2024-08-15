import unittest
import numpy as np
from lfepy.Descriptor import LBP  # Replace with the actual module name


class TestLBP(unittest.TestCase):

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

    def test_lbp_default_params(self):
        # Test LBP with default parameters
        lbp_hist, imgDesc = LBP(self.image)
        self.assertIsInstance(lbp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(lbp_hist.ndim, 1)  # Should be a 1D array

    def test_lbp_with_radius(self):
        # Test LBP with a specific radius
        lbp_hist, imgDesc = LBP(self.image, radius=2)
        self.assertIsInstance(lbp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(lbp_hist.ndim, 1)  # Should be a 1D array

    def test_lbp_mapping_type(self):
        # Test LBP with different mapping types
        for mapping_type in ['full', 'ri', 'u2', 'riu2']:
            lbp_hist, imgDesc = LBP(self.image, radius=1, mappingType=mapping_type)
            self.assertIsInstance(lbp_hist, np.ndarray)
            self.assertIsInstance(imgDesc, np.ndarray)
            self.assertEqual(lbp_hist.ndim, 1)  # Should be a 1D array

    def test_lbp_normalization_mode(self):
        # Test if the LBP histogram is normalized in 'nh' mode
        lbp_hist, _ = LBP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(lbp_hist), 1.0)

    def test_lbp_invalid_mode(self):
        # Test LBP with an invalid mode
        with self.assertRaises(ValueError):
            LBP(self.image, mode='invalid_mode')

    def test_lbp_invalid_mapping_type(self):
        # Test LBP with an invalid mapping type
        with self.assertRaises(ValueError):
            LBP(self.image, mappingType='invalid_mapping')

    def test_lbp_with_none_image(self):
        # Test LBP with None as input
        with self.assertRaises(TypeError):
            LBP(None)

    def test_lbp_with_non_array_image(self):
        # Test LBP with a non-numpy array image
        with self.assertRaises(TypeError):
            LBP("invalid_image")

    def test_lbp_shape(self):
        # Ensure that the image descriptor shape is correctly handled
        lbp_hist, imgDesc = LBP(self.image)
        self.assertTrue(imgDesc.ndim == 2)  # imgDesc should be a 2D array

    def test_lbp_feature_extraction(self):
        # Check if the feature extraction part of LBP works
        lbp_hist, imgDesc = LBP(self.image)
        self.assertTrue(len(lbp_hist) > 0)
        self.assertTrue(imgDesc.size > 0)


if __name__ == '__main__':
    unittest.main()
