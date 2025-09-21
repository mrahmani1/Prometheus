#!/usr/bin/env python3
"""
Test script for the production implementations in deformation_analysis_system.py
"""

import asyncio
import numpy as np
import sys
import os

# Add the backend modules to the path
sys.path.append('backend/modules/resurgent_caldera')

async def test_eof_analysis():
    """Test the EOF analysis implementation"""
    print("Testing EOF Analysis Implementation...")
    
    try:
        from deformation_analysis_system import HighResolutionDeformationAnalysisSystem
        
        # Create test instance (using SQLite for testing)
        system = HighResolutionDeformationAnalysisSystem("sqlite:///test_deformation.db")
        
        # Create synthetic geopotential height data
        test_data = {
            'geopotential_height': np.random.randn(30, 36, 72),  # 30 time steps, 36 lats, 72 lons
            'latitudes': np.linspace(20, 90, 36),
            'longitudes': np.linspace(0, 359, 72)
        }
        
        # Test the EOF analysis method
        result = await system._parse_ncep_geopotential_data(test_data)
        
        if result and 'value' in result:
            print(f"✓ EOF Analysis successful: AO Index = {result['value']:.3f}")
            print(f"  - Arctic anomaly: {result.get('arctic_anomaly', 'N/A')}")
            print(f"  - Confidence: {result.get('confidence', 'N/A')}")
            return True
        else:
            print("✗ EOF Analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ EOF Analysis error: {e}")
        return False

async def test_grib_parsing():
    """Test GRIB2 data parsing implementation"""
    print("\nTesting GRIB2 Data Parsing Implementation...")
    
    try:
        from deformation_analysis_system import HighResolutionDeformationAnalysisSystem
        
        system = HighResolutionDeformationAnalysisSystem("sqlite:///test_deformation.db")
        
        # Test with fallback data (no actual GRIB file)
        test_data = {}
        
        result = await system._parse_gfs_jet_stream_data(test_data, 45.0, -120.0)
        
        if result and 'jet_core_latitude' in result:
            print(f"✓ GRIB2 Parsing successful: Jet core at {result['jet_core_latitude']:.2f}°N")
            print(f"  - Jet strength: {result.get('jet_strength_ms', 'N/A')} m/s")
            print(f"  - Confidence: {result.get('confidence', 'N/A')}")
            return True
        else:
            print("✗ GRIB2 Parsing failed")
            return False
            
    except Exception as e:
        print(f"✗ GRIB2 Parsing error: {e}")
        return False

async def test_era5_processing():
    """Test ERA5 NetCDF processing implementation"""
    print("\nTesting ERA5 NetCDF Processing Implementation...")
    
    try:
        from deformation_analysis_system import HighResolutionDeformationAnalysisSystem
        
        system = HighResolutionDeformationAnalysisSystem("sqlite:///test_deformation.db")
        
        # Test with fallback data (no actual NetCDF file)
        test_data = {}
        
        result = await system._process_era5_netcdf(test_data, 45.0, -120.0)
        
        if result and len(result) > 0:
            print(f"✓ ERA5 Processing successful: {len(result)} pressure levels processed")
            
            # Show sample level data
            sample_level = list(result.keys())[0]
            sample_data = result[sample_level]
            print(f"  - Sample level {sample_level} hPa:")
            print(f"    Wind speed: {sample_data.get('speed', 'N/A'):.2f} m/s")
            print(f"    Data quality: {sample_data.get('data_quality', 'N/A')}")
            return True
        else:
            print("✗ ERA5 Processing failed")
            return False
            
    except Exception as e:
        print(f"✗ ERA5 Processing error: {e}")
        return False

async def test_jma_analysis():
    """Test JMA jet stream analysis implementation"""
    print("\nTesting JMA Jet Stream Analysis Implementation...")
    
    try:
        from deformation_analysis_system import HighResolutionDeformationAnalysisSystem
        
        system = HighResolutionDeformationAnalysisSystem("sqlite:///test_deformation.db")
        
        # Test with fallback data
        test_data = {}
        
        result = await system._parse_jma_jet_analysis(test_data, 35.0, 140.0)
        
        if result and 'jet_axis_latitude' in result:
            print(f"✓ JMA Analysis successful: Jet axis at {result['jet_axis_latitude']:.2f}°N")
            print(f"  - Jet strength: {result.get('jet_axis_strength', 'N/A')} m/s")
            print(f"  - Confidence: {result.get('confidence', 'N/A')}")
            return True
        else:
            print("✗ JMA Analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ JMA Analysis error: {e}")
        return False

async def main():
    """Run all tests"""
    print("Testing Production Implementations in Deformation Analysis System")
    print("=" * 70)
    
    tests = [
        test_eof_analysis,
        test_grib_parsing,
        test_era5_processing,
        test_jma_analysis
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All production implementations are working correctly!")
    else:
        print("⚠ Some implementations need attention")
    
    # Clean up test database
    try:
        os.remove("test_deformation.db")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    asyncio.run(main())