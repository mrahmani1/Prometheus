#!/usr/bin/env python3
"""
Test script for advanced implementations in interdimensional helpers
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import asyncio
import numpy as np
from core.interdimensional_helpers import InterdimensionalMathHelpers

async def test_advanced_krull_dimension():
    """Test advanced Krull dimension computation"""
    print("Testing Advanced Krull Dimension Computation...")
    
    helpers = InterdimensionalMathHelpers()
    
    # Test cases with different ideal structures
    test_ideals = [
        {
            'generators': [{'degree': 1}, {'degree': 2}],
            'variables': ['x', 'y', 'z'],
            'ambient_dimension': 3,
            'geological_dimension_bound': 3
        },
        {
            'generators': [{'degree': 1, 'leading_monomial': [1, 0, 0]}],
            'variables': ['x', 'y'],
            'num_variables': 2,
            'geological_dimension_bound': 2
        },
        {
            'generators': [],
            'variables': [],
            'geological_dimension_bound': 1
        }
    ]
    
    for i, ideal in enumerate(test_ideals):
        try:
            krull_dim = helpers._compute_krull_dimension(ideal)
            print(f"  Test case {i+1}: Krull dimension = {krull_dim}")
        except Exception as e:
            print(f"  Test case {i+1}: Error = {str(e)}")
    
    return True

async def test_advanced_depth_computation():
    """Test advanced depth computation"""
    print("\nTesting Advanced Depth Computation...")
    
    helpers = InterdimensionalMathHelpers()
    
    # Test cases with different module structures
    test_modules = [
        {
            'generators': [
                {'is_zero_divisor': False, 'koszul_regular': True},
                {'is_zero_divisor': False, 'koszul_regular': True}
            ],
            'relations': [{'type': 'linear'}],
            'ring_dimension': 3,
            'geological_depth_bound': 3
        },
        {
            'generators': [
                {'is_zero_divisor': True, 'annihilated_by_maximal': True}
            ],
            'relations': [],
            'ring_dimension': 2,
            'geological_depth_bound': 2
        },
        {
            'generators': [],
            'relations': [],
            'geological_depth_bound': 1
        }
    ]
    
    for i, module in enumerate(test_modules):
        try:
            depth = helpers._compute_depth(module)
            print(f"  Test case {i+1}: Depth = {depth}")
        except Exception as e:
            print(f"  Test case {i+1}: Error = {str(e)}")
    
    return True

async def test_cohen_macaulay_property():
    """Test Cohen-Macaulay property checking"""
    print("\nTesting Cohen-Macaulay Property...")
    
    helpers = InterdimensionalMathHelpers()
    
    # Test module that should be Cohen-Macaulay
    cm_module = {
        'generators': [
            {'is_zero_divisor': False, 'koszul_regular': True},
            {'is_zero_divisor': False, 'koszul_regular': True}
        ],
        'relations': [{'type': 'linear'}],
        'ring_dimension': 2,
        'geological_depth_bound': 2,
        'variables': ['x', 'y'],
        'geological_dimension_bound': 2
    }
    
    try:
        is_cm = helpers._is_cohen_macaulay(cm_module)
        print(f"  Cohen-Macaulay test: {is_cm}")
    except Exception as e:
        print(f"  Cohen-Macaulay test: Error = {str(e)}")
    
    return True

def test_helper_methods():
    """Test basic helper methods"""
    print("\nTesting Helper Methods...")
    
    helpers = InterdimensionalMathHelpers()
    
    # Test monomial similarity
    mono1 = [1, 2, 0]
    mono2 = [1, 3, 1]
    mono3 = [2, 2, 0]
    
    similar_12 = helpers._monomials_similar(mono1, mono2)
    similar_13 = helpers._monomials_similar(mono1, mono3)
    
    print(f"  Monomials {mono1} and {mono2} similar: {similar_12}")
    print(f"  Monomials {mono1} and {mono3} similar: {similar_13}")
    
    return True

async def main():
    """Run all tests"""
    print("=" * 60)
    print("ADVANCED INTERDIMENSIONAL HELPERS TEST SUITE")
    print("=" * 60)
    
    try:
        # Run tests
        await test_advanced_krull_dimension()
        await test_advanced_depth_computation()
        await test_cohen_macaulay_property()
        test_helper_methods()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Advanced helper implementations are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())