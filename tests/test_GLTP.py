import unittest
import numpy as np
from lfepy.Descriptor import GLTP  # Replace with the actual module name


class TestGLTP(unittest.TestCase):

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

    def test_gltp_default_mode(self):
        # Test GLTP with default parameters
        gltp_hist, imgDesc = GLTP(self.image)
        self.assertIsInstance(gltp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(gltp_hist.ndim, 1)  # Should be a 1D array

    def test_gltp_histogram_mode(self):
        # Test GLTP with histogram mode ('h')
        gltp_hist, imgDesc = GLTP(self.image, mode='h')
        self.assertIsInstance(gltp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty
        self.assertEqual(gltp_hist.ndim, 1)  # Should be a 1D array

    def test_gltp_normalization_mode(self):
        # Test if the GLTP histogram is normalized in 'nh' mode
        gltp_hist, _ = GLTP(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(gltp_hist), 1.0)

    def test_gltp_threshold(self):
        # Test GLTP with a custom threshold value
        gltp_hist, imgDesc = GLTP(self.image, t=20)
        self.assertIsInstance(gltp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 0)  # Check if imgDesc is not empty

    def test_gltp_dglp_flag(self):
        # Test GLTP with DGLP flag set to 1
        gltp_hist, imgDesc = GLTP(self.image, DGLP=1)
        self.assertIsInstance(gltp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertGreater(len(imgDesc), 1)  # Check if imgDesc includes angle feature

    def test_gltp_invalid_mode(self):
        # Test GLTP with an invalid mode
        with self.assertRaises(ValueError):
            GLTP(self.image, mode='invalid_mode')

    def test_gltp_invalid_dglp(self):
        # Test GLTP with an invalid DGLP flag
        with self.assertRaises(ValueError):
            GLTP(self.image, DGLP=2)

    def test_gltp_with_none_image(self):
        # Test GLTP with None as input
        with self.assertRaises(TypeError):
            GLTP(None)

    def test_gltp_with_non_array_image(self):
        # Test GLTP with a non-numpy array image
        with self.assertRaises(TypeError):
            GLTP("invalid_image")

    def test_gltp_shape(self):
        # Ensure that the image descriptor shape is correctly handled
        gltp_hist, imgDesc = GLTP(self.image)
        self.assertEqual(len(imgDesc), 2)  # Check for basic and optional features

    def test_gltp_feature_extraction(self):
        # Check if the feature extraction part of GLTP works
        # Here we would check the content of imgDesc and GLTP_hist for expected values.
        gltp_hist, imgDesc = GLTP(self.image, t=10, DGLP=1)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue('fea' in imgDesc[0])


if __name__ == '__main__':
    unittest.main()
