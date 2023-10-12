"""
Test math_utils module.
"""
import unittest

import numpy as np

from common_utils import math_utils

# Example data
array = np.random.rand(100)
size = 5

class TestMathUtils(unittest.TestCase):

    def test_sample(self):
        """
        Test if the size of samples matches the expected size
        """
        samples = math_utils.random_sampling(array, size)

        self.assertEqual(samples.shape, (10000, size))

    def test_sample_range(self):
        """
        Test if samples fall within a specified range
        """
        samples = math_utils.random_sampling(array, size)

        # check range 
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples < 100))

if __name__ == '__main__':
    unittest.main()
