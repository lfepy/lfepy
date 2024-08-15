import unittest
import numpy as np
from lfepy.Descriptor import MTP  # Replace with the actual module name


class TestMTP(unittest.TestCase):

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

    def test_mtp_default_params(self):
        # Test MTP with default parameters
        mtp_hist, imgDesc = MTP(self.image)
        self.assertIsInstance(mtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)  # There should be 2 descriptors (positive and negative)
        self.assertTrue(len(mtp_hist) > 0)  # Check that histogram is not empty
        for desc in imgDesc:
            self.assertIn('fea', desc)
            self.assertIsInstance(desc['fea'], np.ndarray)

    def test_mtp_custom_mode(self):
        # Test MTP with custom mode
        mtp_hist, imgDesc = MTP(self.image, mode='nh')
        self.assertIsInstance(mtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertTrue(len(mtp_hist) > 0)

    def test_mtp_custom_threshold(self):
        # Test MTP with a custom threshold value
        mtp_hist, imgDesc = MTP(self.image, t=20)
        self.assertIsInstance(mtp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertTrue(len(mtp_hist) > 0)

    def test_mtp_invalid_mode(self):
        # Test MTP with an invalid mode
        with self.assertRaises(ValueError):
            MTP(self.image, mode='invalid_mode')

    def test_mtp_with_none_image(self):
        # Test MTP with None as input
        with self.assertRaises(TypeError):
            MTP(None)

    def test_mtp_with_non_array_image(self):
        # Test MTP with a non-numpy array image
        with self.assertRaises(TypeError):
            MTP("invalid_image")

    def test_mtp_feature_extraction(self):
        # Check if the feature extraction part of MTP works
        mtp_hist, imgDesc = MTP(self.image)
        self.assertTrue(len(mtp_hist) > 0)
        self.assertEqual(len(imgDesc), 2)
        for desc in imgDesc:
            self.assertTrue('fea' in desc)
            self.assertIsInstance(desc['fea'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
