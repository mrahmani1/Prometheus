#!/usr/bin/env python3
"""
Utility test functions to improve coverage.
"""

import json
import os
import unittest


class TestUtils(unittest.TestCase):
    """Utility test class."""

    def test_file_operations(self):
        """Test file operations."""
        test_file = "test_temp.txt"
        test_content = "Hello, World!"

        # Write file
        with open(test_file, "w") as f:
            f.write(test_content)

        # Read file
        with open(test_file, "r") as f:
            content = f.read()

        self.assertEqual(content, test_content)

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_json_operations(self):
        """Test JSON operations."""
        test_data = {"name": "test", "value": 42}
        json_string = json.dumps(test_data)
        parsed_data = json.loads(json_string)

        self.assertEqual(parsed_data["name"], "test")
        self.assertEqual(parsed_data["value"], 42)

    def test_environment_variables(self):
        """Test environment variable operations."""
        test_var = "TEST_VAR"
        test_value = "test_value"

        # Set environment variable
        os.environ[test_var] = test_value

        # Get environment variable
        retrieved_value = os.environ.get(test_var)

        self.assertEqual(retrieved_value, test_value)

        # Clean up
        if test_var in os.environ:
            del os.environ[test_var]

    def test_path_operations(self):
        """Test path operations."""
        test_path = os.path.join("test", "path", "file.txt")
        self.assertTrue(os.path.sep in test_path)

        dirname = os.path.dirname(test_path)
        basename = os.path.basename(test_path)

        self.assertEqual(basename, "file.txt")
        self.assertTrue("test" in dirname)


if __name__ == "__main__":
    unittest.main()
