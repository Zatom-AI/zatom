import unittest
from pathlib import Path

# Discover and run all test files in root/unit_tests/
tests_dir = Path(__file__).resolve().parent
suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py")
unittest.TextTestRunner().run(suite)
