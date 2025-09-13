"""
Simple validation script for geochemical analysis implementation
"""

import sys
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        from modules.production_geochemical_analysis import analyze_production_geochemistry
        logger.info("✓ Production geochemical analysis module imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import production geochemical analysis: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    try:
        from modules.production_geochemical_analysis import ProductionGeochemicalAnalyzer
        
        # Test initialization
        analyzer = ProductionGeochemicalAnalyzer()
        logger.info("✓ ProductionGeochemicalAnalyzer initialized successfully")
        
        # Test sample ID generation
        location = {'latitude': 44.4280, 'longitude': -110.5885}
        sample_id = analyzer._generate_sample_id(location)
        assert sample_id.startswith('PROD_GEOCHEM_')
        logger.info(f"✓ Sample ID generation works: {sample_id}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False

async def test_analysis_structure():
    """Test that analysis returns proper structure"""
    try:
        from modules.production_geochemical_analysis import analyze_production_geochemistry
        
        location = {'latitude': 44.4280, 'longitude': -110.5885, 'elevation': 2400}
        
        result = await analyze_production_geochemistry(location)
        
        # Check required fields
        required_fields = [
            'sample_id', 'analysis_timestamp', 'data_sources', 'major_elements',
            'trace_elements', 'tectonic_setting', 'magma_source', 'overall_confidence'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        logger.info("✓ Analysis returns proper structure")
        logger.info(f"  Sample ID: {result['sample_id']}")
        logger.info(f"  Tectonic Setting: {result['tectonic_setting']}")
        logger.info(f"  Confidence: {result['overall_confidence']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Analysis structure test failed: {e}")
        return False

def test_trace_element_imports():
    """Test trace element analysis imports"""
    try:
        from modules.enhanced_trace_element_analysis import analyze_enhanced_trace_elements
        logger.info("✓ Enhanced trace element analysis module imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import enhanced trace element analysis: {e}")
        return False

async def main():
    """Run all validation tests"""
    logger.info("Starting implementation validation...")
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Trace Element Imports", test_trace_element_imports),
        ("Analysis Structure", test_analysis_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\nValidation Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All validation tests passed!")
        return True
    else:
        logger.error("❌ Some validation tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)