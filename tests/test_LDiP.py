import unittest
import numpy as np
from lfepy.Descriptor import LDiP  # Replace with the actual module name


class TestLDiP(unittest.TestCase):

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

    def test_ldip_default_mode(self):
        # Test LDiP with default parameters
        ldip_hist, imgDesc = LDiP(self.image)
        self.assertIsInstance(ldip_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(imgDesc.shape, self.image.shape)  # Check if imgDesc has the same shape as the input image
        self.assertEqual(ldip_hist.ndim, 1)  # Should be a 1D array

    def test_ldip_histogram_mode(self):
        # Test LDiP with histogram mode ('h')
        ldip_hist, imgDesc = LDiP(self.image, mode='h')
        self.assertIsInstance(ldip_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(imgDesc.shape, self.image.shape)  # Check if imgDesc has the same shape as the input image
        self.assertEqual(ldip_hist.ndim, 1)  # Should be a 1D array

    def test_ldip_normalization_mode(self):
        # Test if the LDiP histogram is normalized in 'nh' mode
        ldip_hist, _ = LDiP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(ldip_hist), 1.0)

    def test_ldip_invalid_mode(self):
        # Test LDiP with an invalid mode
        with self.assertRaises(ValueError):
            LDiP(self.image, mode='invalid_mode')

    def test_ldip_with_none_image(self):
        # Test LDiP with None as input
        with self.assertRaises(TypeError):
            LDiP(None)

    def test_ldip_with_non_array_image(self):
        # Test LDiP with a non-numpy array image
        with self.assertRaises(TypeError):
            LDiP("invalid_image")

    def test_ldip_shape(self):
        # Ensure that the image descriptor shape is correctly handled
        ldip_hist, imgDesc = LDiP(self.image)
        self.assertEqual(imgDesc.shape, self.image.shape)  # Check that imgDesc shape matches the input image shape

    def test_ldip_feature_extraction(self):
        # Check if the feature extraction part of LDiP works
        ldip_hist, imgDesc = LDiP(self.image)
        self.assertTrue(np.any(imgDesc >= 0))  # Check if imgDesc contains non-negative values
        unique_values = np.unique(imgDesc)
        self.assertTrue(np.all(np.in1d(unique_values, np.array([
            7, 11, 13, 14, 19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44,
            49, 50, 52, 56, 67, 69, 70, 73, 74, 76, 81, 82, 84, 88, 97, 98,
            100, 104, 112, 131, 133, 134, 137, 138, 140, 145, 146, 148, 152,
            161, 162, 164, 168, 176, 193, 194, 196, 200, 208, 224
        ]))))  # Check if the values in imgDesc are in the expected bin set


if __name__ == '__main__':
    unittest.main()
