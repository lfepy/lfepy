import unittest
import numpy as np
from lfepy.Descriptor import LTeP  # Replace with the actual module name


class TestLTeP(unittest.TestCase):

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

    def test_ltep_default_params(self):
        # Test LTeP with default parameters
        ltep_hist, imgDesc = LTeP(self.image)
        self.assertIsInstance(ltep_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertTrue(len(ltep_hist) == 512)  # 256 bins for each of the 2 descriptors
        self.assertEqual(imgDesc[0]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertEqual(imgDesc[1]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_ltep_custom_threshold(self):
        # Test LTeP with a custom threshold value
        threshold = 3
        ltep_hist, imgDesc = LTeP(self.image, t=threshold)
        self.assertIsInstance(ltep_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertTrue(len(ltep_hist) == 512)  # 256 bins for each of the 2 descriptors
        self.assertEqual(imgDesc[0]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertEqual(imgDesc[1]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_ltep_custom_params(self):
        # Test LTeP with custom parameters
        ltep_hist, imgDesc = LTeP(self.image, mode='h', t=2)
        self.assertIsInstance(ltep_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(len(imgDesc), 2)
        self.assertTrue(len(ltep_hist) == 512)  # 256 bins for each of the 2 descriptors
        self.assertEqual(imgDesc[0]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertEqual(imgDesc[1]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_ltep_invalid_mode(self):
        # Test LTeP with an invalid mode
        with self.assertRaises(ValueError):
            LTeP(self.image, mode='invalid_mode')

    def test_ltep_with_none_image(self):
        # Test LTeP with None as input
        with self.assertRaises(TypeError):
            LTeP(None)

    def test_ltep_with_non_array_image(self):
        # Test LTeP with a non-numpy array image
        with self.assertRaises(TypeError):
            LTeP("invalid_image")

    def test_ltep_feature_extraction(self):
        # Check if the feature extraction part of LTeP works
        ltep_hist, imgDesc = LTeP(self.image)
        self.assertTrue(len(ltep_hist) == 512)
        self.assertEqual(imgDesc[0]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertEqual(imgDesc[1]['fea'].shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertTrue(np.all(np.isin(imgDesc[0]['fea'], np.arange(0, 256))))
        self.assertTrue(np.all(np.isin(imgDesc[1]['fea'], np.arange(0, 256))))


if __name__ == '__main__':
    unittest.main()
