"""
Test file_utils module.
"""
import unittest
import os
import tempfile
from pathlib import Path

from common_utils import file_utils

class TestFileUtils(unittest.TestCase):

    def clean_up(self):
        # Clean up the temporary directories & files after testing
        [os.remove(copy) for copy in self.copies]
        os.remove(self.file.name)
        print("> Test passed.")

    def test_copy_files(self):
        # Create a temporary directories for testing
        dests = [tempfile.mkdtemp() for _ in range(2)]

        # Create a temporary files
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.source = '/tmp/'
        name = self.file.name.split('/')[-1]

        # Copy the files from the source to destinations
        file_utils.copy_files(self.source, dests, name)

        # Check if the files were copied correctly
        self.copies = []
        for dest in dests:
            copy = dest + '/' + name
            self.copies.append(copy)
            self.assertTrue(Path(copy).exists())

        self.clean_up()

if __name__ == '__main__':
    unittest.main()
