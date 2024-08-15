import unittest
import numpy as np
from lfepy.Descriptor import LGP  # Replace with the actual module name


class TestLGP(unittest.TestCase):

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

    def test_lgp_default_params(self):
        # Test LGP with default parameters
        lgp_hist, imgDesc = LGP(self.image)
        self.assertIsInstance(lgp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertTrue(len(lgp_hist) == 7)  # Should have 7 bins
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_lgp_custom_params(self):
        # Test LGP with custom parameters
        lgp_hist, imgDesc = LGP(self.image, mode='h')
        self.assertIsInstance(lgp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertTrue(len(lgp_hist) == 7)  # Should have 7 bins
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))

    def test_lgp_invalid_mode(self):
        # Test LGP with an invalid mode
        with self.assertRaises(ValueError):
            LGP(self.image, mode='invalid_mode')

    def test_lgp_with_none_image(self):
        # Test LGP with None as input
        with self.assertRaises(TypeError):
            LGP(None)

    def test_lgp_with_non_array_image(self):
        # Test LGP with a non-numpy array image
        with self.assertRaises(TypeError):
            LGP("invalid_image")

    def test_lgp_feature_extraction(self):
        # Check if the feature extraction part of LGP works
        lgp_hist, imgDesc = LGP(self.image)
        self.assertTrue(len(lgp_hist) == 7)
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 2, self.image.shape[1] - 2))
        self.assertTrue(np.all(np.isin(imgDesc, np.arange(4, 11))))


if __name__ == '__main__':
    unittest.main()
