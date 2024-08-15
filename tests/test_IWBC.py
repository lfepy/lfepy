import unittest
import numpy as np
from lfepy.Descriptor import IWBC  # Replace with the actual module name


class TestIWBC(unittest.TestCase):

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

    def test_iwbc_default_mode(self):
        # Test IWBC with default parameters
        iwbc_hist, imgDesc = IWBC(self.image)
        self.assertIsInstance(iwbc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(iwbc_hist.ndim, 1)  # Should be a 1D array

    def test_iwbc_histogram_mode(self):
        # Test IWBC with histogram mode ('h')
        iwbc_hist, imgDesc = IWBC(self.image, mode='h')
        self.assertIsInstance(iwbc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(iwbc_hist.ndim, 1)  # Should be a 1D array

    def test_iwbc_normalization_mode(self):
        # Test if the IWBC histogram is normalized in 'nh' mode
        iwbc_hist, _ = IWBC(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(iwbc_hist), 1.0)

    def test_iwbc_scale(self):
        # Test IWBC with a custom scale value
        iwbc_hist, imgDesc = IWBC(self.image, scale=2)
        self.assertIsInstance(iwbc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_iwbc_invalid_mode(self):
        # Test IWBC with an invalid mode
        with self.assertRaises(ValueError):
            IWBC(self.image, mode='invalid_mode')

    def test_iwbc_invalid_scale(self):
        # Test IWBC with an invalid scale
        with self.assertRaises(KeyError):
            IWBC(self.image, scale=4)

    def test_iwbc_with_none_image(self):
        # Test IWBC with None as input
        with self.assertRaises(TypeError):
            IWBC(None)

    def test_iwbc_with_non_array_image(self):
        # Test IWBC with a non-numpy array image
        with self.assertRaises(TypeError):
            IWBC("invalid_image")

    def test_iwbc_shape(self):
        # Ensure that the image descriptor shape is correctly handled
        iwbc_hist, imgDesc = IWBC(self.image, scale=1)
        self.assertEqual(len(imgDesc), 2)  # Check for magnitude and orientation features

    def test_iwbc_feature_extraction(self):
        # Check if the feature extraction part of IWBC works
        iwbc_hist, imgDesc = IWBC(self.image, scale=1)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue('fea' in imgDesc[0])
        self.assertEqual(len(iwbc_hist), 2 * (2 ** (8 + 2)))  # Adjust based on expected histogram size


if __name__ == '__main__':
    unittest.main()
