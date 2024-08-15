import unittest
import numpy as np
from lfepy.Descriptor import LGDiP  # Replace with the actual module name


class TestLGDiP(unittest.TestCase):

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

    def test_lgdip_default_params(self):
        # Test LGDiP with default parameters
        lgdip_hist, imgDesc = LGDiP(self.image)
        self.assertIsInstance(lgdip_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(lgdip_hist.ndim == 1)  # Should be a 1D array

    def test_lgdip_custom_params(self):
        # Test LGDiP with custom parameters
        lgdip_hist, imgDesc = LGDiP(self.image, mode='h')
        self.assertIsInstance(lgdip_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertTrue(len(imgDesc) > 0)
        self.assertTrue(lgdip_hist.ndim == 1)  # Should be a 1D array

    def test_lgdip_invalid_mode(self):
        # Test LGDiP with an invalid mode
        with self.assertRaises(ValueError):
            LGDiP(self.image, mode='invalid_mode')

    def test_lgdip_with_none_image(self):
        # Test LGDiP with None as input
        with self.assertRaises(TypeError):
            LGDiP(None)

    def test_lgdip_with_non_array_image(self):
        # Test LGDiP with a non-numpy array image
        with self.assertRaises(TypeError):
            LGDiP("invalid_image")

    def test_lgdip_feature_extraction(self):
        # Check if the feature extraction part of LGDiP works
        lgdip_hist, imgDesc = LGDiP(self.image)
        self.assertTrue(len(lgdip_hist) > 0)
        self.assertTrue(len(imgDesc) > 0)
        for desc in imgDesc:
            self.assertIn('fea', desc)
            self.assertEqual(desc['fea'].ndim, 2)
            self.assertEqual(desc['fea'].dtype, np.float64)


if __name__ == '__main__':
    unittest.main()
