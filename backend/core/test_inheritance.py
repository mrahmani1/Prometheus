"""Test inheritance of helper methods"""

from interdimensional_helpers import InterdimensionalMathHelpers

# Test if methods exist
helper = InterdimensionalMathHelpers()
print("Available methods in InterdimensionalMathHelpers:")
methods = [method for method in dir(helper) if method.startswith('_find') or method.startswith('_compute')]
for method in methods:
    print(f"  - {method}")

# Test specific method
if hasattr(helper, '_find_nilpotent_elements'):
    print("\n✓ _find_nilpotent_elements method exists")
else:
    print("\n✗ _find_nilpotent_elements method missing")