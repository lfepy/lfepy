import unittest
import numpy as np
from lfepy.Descriptor import LAP  # Replace with the actual module name


class TestLAP(unittest.TestCase):

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

    def test_lap_default_mode(self):
        # Test LAP with default parameters
        lap_hist, imgDesc = LAP(self.image)
        self.assertIsInstance(lap_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(lap_hist.ndim, 1)  # Should be a 1D array

    def test_lap_histogram_mode(self):
        # Test LAP with histogram mode ('h')
        lap_hist, imgDesc = LAP(self.image, mode='h')
        self.assertIsInstance(lap_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(lap_hist.ndim, 1)  # Should be a 1D array

    def test_lap_normalization_mode(self):
        # Test if the LAP histogram is normalized in 'nh' mode
        lap_hist, _ = LAP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(lap_hist), 1.0)

    def test_lap_invalid_mode(self):
        # Test LAP with an invalid mode
        with self.assertRaises(ValueError):
            LAP(self.image, mode='invalid_mode')

    def test_lap_with_none_image(self):
        # Test LAP with None as input
        with self.assertRaises(TypeError):
            LAP(None)

    def test_lap_with_non_array_image(self):
        # Test LAP with a non-numpy array image
        with self.assertRaises(TypeError):
            LAP("invalid_image")

    def test_lap_shape(self):
        # Ensure that the image descriptor shape is correctly handled
        lap_hist, imgDesc = LAP(self.image)
        self.assertEqual(len(imgDesc), 2)  # Check for the two feature patterns

    def test_lap_feature_extraction(self):
        # Check if the feature extraction part of LAP works
        lap_hist, imgDesc = LAP(self.image)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue('fea' in imgDesc[0])


if __name__ == '__main__':
    unittest.main()
