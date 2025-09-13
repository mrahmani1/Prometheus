"""
Simple test for Enhanced Tectonic Setting Database
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from tectonic_setting_enhancements import EnhancedTectonicSettingDatabase
    print("✓ Successfully imported EnhancedTectonicSettingDatabase")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

async def test_basic_functionality():
    """Test basic functionality of enhanced tectonic setting database"""
    
    print("Testing Enhanced Tectonic Setting Database...")
    
    # Create database instance (using mock URL for testing)
    database_url = "postgresql://test:test@localhost:5432/test"
    
    try:
        enhanced_db = EnhancedTectonicSettingDatabase(database_url)
        print("✓ Successfully created EnhancedTectonicSettingDatabase instance")
    except Exception as e:
        print(f"✗ Failed to create database instance: {e}")
        return False
    
    # Test distance calculations
    try:
        test_lat, test_lon = 35.3606, 138.7274  # Mount Fuji
        
        # Test individual distance calculations
        trench_distance = await enhanced_db._distance_to_nearest_trench(test_lat, test_lon)
        print(f"✓ Trench distance calculation: {trench_distance:.1f} km")
        
        arc_distance = await enhanced_db._distance_to_nearest_volcanic_arc(test_lat, test_lon)
        print(f"✓ Volcanic arc distance calculation: {arc_distance:.1f} km")
        
        hotspot_distance = await enhanced_db._distance_to_nearest_hotspot(test_lat, test_lon)
        print(f"✓ Hotspot distance calculation: {hotspot_distance:.1f} km")
        
    except Exception as e:
        print(f"✗ Distance calculation failed: {e}")
        return False
    
    # Test comprehensive distance calculations
    try:
        comprehensive_distances = await enhanced_db._calculate_comprehensive_tectonic_distances(test_lat, test_lon)
        print(f"✓ Comprehensive distance calculations completed")
        print(f"  Found {len(comprehensive_distances)} tectonic features")
        
        # Show nearest features
        sorted_distances = sorted(comprehensive_distances.items(), key=lambda x: x[1])
        print("  Nearest features:")
        for feature, distance in sorted_distances[:3]:
            print(f"    {feature}: {distance:.1f} km")
            
    except Exception as e:
        print(f"✗ Comprehensive distance calculation failed: {e}")
        return False
    
    # Test tectonic setting hierarchy analysis
    try:
        setting_hierarchy = enhanced_db._analyze_tectonic_setting_hierarchy(
            comprehensive_distances, test_lat, test_lon
        )
        print(f"✓ Tectonic setting hierarchy analysis completed")
        print(f"  Primary setting: {setting_hierarchy['primary']['setting']}")
        print(f"  Secondary settings: {len(setting_hierarchy['secondary'])}")
        
    except Exception as e:
        print(f"✗ Tectonic hierarchy analysis failed: {e}")
        return False
    
    print("\n🎉 All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    if success:
        print("\n✅ Enhanced Tectonic Setting Database is working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)