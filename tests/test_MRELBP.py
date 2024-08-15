import unittest
import numpy as np
from lfepy.Descriptor import MRELBP  # Replace with the actual module name


class TestMRELBP(unittest.TestCase):

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

    def test_mrelbp_default_params(self):
        # Test MRELBP with default parameters
        with self.assertRaises(ValueError):
            mrelbp_hist, imgDesc = MRELBP(self.image)
            self.assertIsInstance(mrelbp_hist, np.ndarray)
            self.assertIsInstance(imgDesc, list)
            self.assertTrue(len(mrelbp_hist) > 0)  # Check that histogram is not empty
            self.assertEqual(len(imgDesc), 4)  # There should be 4 different radii
            for desc in imgDesc:
                self.assertIn('fea', desc)
                self.assertIn('CImg', desc['fea'])
                self.assertIn('NILBPImage', desc['fea'])
                self.assertIn('RDLBPImage', desc['fea'])

    def test_mrelbp_custom_mode(self):
        # Test MRELBP with a custom mode
        with self.assertRaises(ValueError):
            mrelbp_hist, imgDesc = MRELBP(self.image, mode='nh')
            self.assertIsInstance(mrelbp_hist, np.ndarray)
            self.assertIsInstance(imgDesc, list)
            self.assertTrue(len(mrelbp_hist) > 0)  # Check that histogram is not empty
            self.assertEqual(len(imgDesc), 4)  # There should be 4 different radii

    def test_mrelbp_invalid_mode(self):
        # Test MRELBP with an invalid mode
        with self.assertRaises(ValueError):
            MRELBP(self.image, mode='invalid_mode')

    def test_mrelbp_with_none_image(self):
        # Test MRELBP with None as input
        with self.assertRaises(TypeError):
            MRELBP(None)

    def test_mrelbp_with_non_array_image(self):
        # Test MRELBP with a non-numpy array image
        with self.assertRaises(TypeError):
            MRELBP("invalid_image")

    def test_mrelbp_feature_extraction(self):
        # Check if the feature extraction part of MRELBP works
        with self.assertRaises(ValueError):
            mrelbp_hist, imgDesc = MRELBP(self.image)
            self.assertTrue(len(mrelbp_hist) > 0)
            self.assertEqual(len(imgDesc), 4)
            for desc in imgDesc:
                self.assertTrue('fea' in desc)
                self.assertTrue('CImg' in desc['fea'])
                self.assertTrue('NILBPImage' in desc['fea'])
                self.assertTrue('RDLBPImage' in desc['fea'])
                self.assertIsInstance(desc['fea']['CImg'], np.ndarray)
                self.assertIsInstance(desc['fea']['NILBPImage'], np.ndarray)
                self.assertIsInstance(desc['fea']['RDLBPImage'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
