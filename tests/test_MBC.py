import unittest
import numpy as np
from lfepy.Descriptor import MBC  # Replace with the actual module name


class TestMBC(unittest.TestCase):

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

    def test_mbc_default_params(self):
        # Test MBC with default parameters
        mbc_hist, imgDesc = MBC(self.image)
        self.assertIsInstance(mbc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(mbc_hist) > 0)  # Check that histogram is not empty
        self.assertTrue(all(isinstance(desc, dict) and 'fea' in desc for desc in imgDesc))
        self.assertEqual(len(imgDesc), 3)  # Should have 3 descriptors (A, O, P)

    def test_mbc_custom_mode(self):
        # Test MBC with a custom mode
        with self.assertRaises(ValueError):
            mbc_hist, imgDesc = MBC(self.image, mode='h', mbcMode='O')
            self.assertIsInstance(mbc_hist, np.ndarray)
            self.assertIsInstance(imgDesc, list)
            self.assertTrue(len(mbc_hist) > 0)  # Check that histogram is not empty
            self.assertTrue(all(isinstance(desc, dict) and 'fea' in desc for desc in imgDesc))
            self.assertEqual(len(imgDesc), 1)  # Only one descriptor for mode 'O'

    def test_mbc_invalid_mode(self):
        # Test MBC with an invalid mode
        with self.assertRaises(ValueError):
            MBC(self.image, mode='invalid_mode')

    def test_mbc_invalid_mbcMode(self):
        # Test MBC with an invalid mbcMode
        with self.assertRaises(ValueError):
            MBC(self.image, mbcMode='invalid_mode')

    def test_mbc_with_none_image(self):
        # Test MBC with None as input
        with self.assertRaises(TypeError):
            MBC(None)

    def test_mbc_with_non_array_image(self):
        # Test MBC with a non-numpy array image
        with self.assertRaises(TypeError):
            MBC("invalid_image")

    def test_mbc_feature_extraction(self):
        # Check if the feature extraction part of MBC works
        mbc_hist, imgDesc = MBC(self.image)
        self.assertTrue(len(mbc_hist) > 0)
        self.assertEqual(len(imgDesc), 3)
        for desc in imgDesc:
            self.assertTrue('fea' in desc)
            self.assertTrue(np.issubdtype(type(desc['fea']), np.ndarray))  # Ensure 'fea' contains numeric data


if __name__ == '__main__':
    unittest.main()
