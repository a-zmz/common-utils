import unittest

from common_utils import __version__

class Test(unittest.TestCase):

    def test_version(self):
        assert __version__ == '0.1.0'

if __name__ == '__main__':
    unittest.main()
