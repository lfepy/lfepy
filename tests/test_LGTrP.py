import unittest
import numpy as np
from lfepy.Descriptor import LGTrP  # Replace with the actual module name


class TestLGTrP(unittest.TestCase):

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

    def test_lgtrp_default_params(self):
        # Test LGTrP with default parameters
        lgtrp_hist, imgDesc = LGTrP(self.image)
        self.assertIsInstance(lgtrp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertTrue(len(lgtrp_hist) == 256)  # Should have 256 bins
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 4, self.image.shape[1] - 4))

    def test_lgtrp_custom_params(self):
        # Test LGTrP with custom parameters
        lgtrp_hist, imgDesc = LGTrP(self.image, mode='h')
        self.assertIsInstance(lgtrp_hist, np.ndarray)
        self.assertIsInstance(imgDesc, np.ndarray)
        self.assertTrue(len(lgtrp_hist) == 256)  # Should have 256 bins
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 4, self.image.shape[1] - 4))

    def test_lgtrp_invalid_mode(self):
        # Test LGTrP with an invalid mode
        with self.assertRaises(ValueError):
            LGTrP(self.image, mode='invalid_mode')

    def test_lgtrp_with_none_image(self):
        # Test LGTrP with None as input
        with self.assertRaises(TypeError):
            LGTrP(None)

    def test_lgtrp_with_non_array_image(self):
        # Test LGTrP with a non-numpy array image
        with self.assertRaises(TypeError):
            LGTrP("invalid_image")

    def test_lgtrp_feature_extraction(self):
        # Check if the feature extraction part of LGTrP works
        lgtrp_hist, imgDesc = LGTrP(self.image)
        self.assertTrue(len(lgtrp_hist) == 256)
        self.assertEqual(imgDesc.shape, (self.image.shape[0] - 4, self.image.shape[1] - 4))
        self.assertTrue(np.all(np.isin(imgDesc, np.arange(256))))


if __name__ == '__main__':
    unittest.main()
