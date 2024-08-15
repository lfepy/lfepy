import unittest
import numpy as np
from lfepy.Descriptor import BPPC  # Replace with the actual module name


class TestBPPC(unittest.TestCase):

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

    def test_bppc_default_mode(self):
        # Test BPPC with default mode
        bppc_hist, imgDesc = BPPC(self.image)
        self.assertIsInstance(bppc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(bppc_hist.ndim, 1)  # Should be a 1D array
        self.assertGreater(len(imgDesc), 0)  # imgDesc should not be empty

    def test_bppc_histogram_mode(self):
        # Test BPPC with histogram mode ('h')
        bppc_hist, imgDesc = BPPC(self.image, mode='h')
        self.assertIsInstance(bppc_hist, np.ndarray)
        self.assertIsInstance(imgDesc, list)
        self.assertEqual(bppc_hist.ndim, 1)  # Should be a 1D array

    def test_bppc_invalid_mode(self):
        # Test BPPC with an invalid mode
        with self.assertRaises(ValueError):
            BPPC(self.image, mode='invalid_mode')

    def test_bppc_normalization(self):
        # Test if the BPPC histogram is normalized in 'nh' mode
        bppc_hist, _ = BPPC(self.image, mode='nh')
        self.assertAlmostEqual(np.sum(bppc_hist), 1.0)

    def test_bppc_large_image(self):
        # Test BPPC with a larger image to ensure it handles larger inputs
        large_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        bppc_hist, imgDesc = BPPC(large_image)
        self.assertIsInstance(bppc_hist, np.ndarray)
        self.assertGreater(len(bppc_hist), 0)

    def test_bppc_with_none_image(self):
        # Test BPPC with None as input
        with self.assertRaises(TypeError):
            BPPC(None)


if __name__ == '__main__':
    unittest.main()
