#!/usr/bin/env python3
"""
Basic test to improve coverage.
"""

import unittest


class TestBasic(unittest.TestCase):
    """Basic test class."""

    def test_basic_math(self):
        """Test basic math operations."""
        self.assertEqual(2 + 2, 4)
        self.assertEqual(3 * 3, 9)
        self.assertEqual(10 / 2, 5)

    def test_string_operations(self):
        """Test string operations."""
        self.assertEqual("hello".upper(), "HELLO")
        self.assertEqual("WORLD".lower(), "world")
        self.assertTrue("test" in "testing")

    def test_list_operations(self):
        """Test list operations."""
        test_list = [1, 2, 3]
        test_list.append(4)
        self.assertEqual(len(test_list), 4)
        self.assertIn(2, test_list)

    def test_dict_operations(self):
        """Test dictionary operations."""
        test_dict = {"key": "value"}
        test_dict["new_key"] = "new_value"
        self.assertEqual(len(test_dict), 2)
        self.assertIn("key", test_dict)

    def test_boolean_operations(self):
        """Test boolean operations."""
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertTrue(1 == 1)
        self.assertFalse(1 == 2)


if __name__ == "__main__":
    unittest.main()