import unittest
import numpy as np
from lfepy.Descriptor import LDTP  # Replace with the actual module name


class TestLDTP(unittest.TestCase):

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

    def test_ldtp_default_mode(self):
        # Test LDTP with default parameters
        ldtp_hist, imgDesc = LDTP(self.image)
        self.assertIsInstance(ldtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(ldtp_hist.ndim, 1)  # Should be a 1D array
        self.assertEqual(imgDesc.ndim, 2)  # Should be a 2D array

    def test_ldtp_histogram_mode(self):
        # Test LDTP with histogram mode ('h')
        ldtp_hist, imgDesc = LDTP(self.image, mode='h')
        self.assertIsInstance(ldtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(ldtp_hist.ndim, 1)  # Should be a 1D array
        self.assertEqual(imgDesc.ndim, 2)  # Should be a 2D array

    def test_ldtp_normalization_mode(self):
        # Test if the LDTP histogram is normalized in 'nh' mode
        ldtp_hist, _ = LDTP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(ldtp_hist), 1.0)

    def test_ldtp_invalid_mode(self):
        # Test LDTP with an invalid mode
        with self.assertRaises(ValueError):
            LDTP(self.image, mode='invalid_mode')

    def test_ldtp_invalid_eps_value(self):
        # Test LDTP with an invalid epsi value
        ldtp_hist, imgDesc = LDTP(self.image, epsi=-1)
        self.assertIsInstance(ldtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertEqual(ldtp_hist.ndim, 1)
        self.assertEqual(imgDesc.ndim, 2)

    def test_ldtp_with_none_image(self):
        # Test LDTP with None as input
        with self.assertRaises(TypeError):
            LDTP(None)

    def test_ldtp_with_non_array_image(self):
        # Test LDTP with a non-numpy array image
        with self.assertRaises(TypeError):
            LDTP("invalid_image")

    def test_ldtp_shape(self):
        # Ensure that the LDTP descriptor shape is correctly handled
        ldtp_hist, imgDesc = LDTP(self.image)
        self.assertTrue(len(ldtp_hist) > 0)  # Ensure that histogram is not empty
        self.assertTrue(imgDesc.size > 0)  # Ensure that imgDesc is not empty

    def test_ldtp_feature_extraction(self):
        # Check if the feature extraction part of LDTP works
        ldtp_hist, imgDesc = LDTP(self.image)
        self.assertTrue(len(ldtp_hist) > 0)
        self.assertEqual(imgDesc.ndim, 2)
        unique_values = np.unique(imgDesc)
        self.assertTrue(
            np.all(np.in1d(unique_values, np.arange(0, 123))))  # Check if feature values are within expected range


if __name__ == '__main__':
    unittest.main()
