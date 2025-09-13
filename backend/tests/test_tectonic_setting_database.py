"""
Comprehensive tests for Tectonic Setting Database

Tests all aspects of the tectonic classification system including:
- Tectonic setting classification accuracy
- Plate motion analysis
- Stress regime determination
- Volcanic province identification
- Database operations and data integrity
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from tectonic_setting_database import (
    TectonicSettingDatabase, TectonicSettingType, PlateMotionType, 
    StressRegime, TectonicParameters, Base
)
from tectonic_database_init import TectonicDatabaseInitializer

class TestTectonicSettingDatabase:
    """Test suite for tectonic setting database functionality"""
    
    @pytest.fixture
    async def database_setup(self):
        """Set up test database"""
        # Use in-memory SQLite for testing
        test_db_url = "sqlite:///:memory:"
        
        # Initialize database
        tectonic_db = TectonicSettingDatabase(test_db_url)
        
        # Create tables
        Base.metadata.create_all(bind=tectonic_db.engine)
        
        yield tectonic_db
        
        # Cleanup
        Base.metadata.drop_all(bind=tectonic_db.engine)
    
    @pytest.mark.asyncio
    async def test_subduction_zone_classification(self, database_setup):
        """Test classification of subduction zone volcanoes"""
        
        tectonic_db = database_setup
        
        # Test Mount Fuji (Japan subduction zone)
        classification = await tectonic_db.classify_tectonic_setting(
            'mount_fuji', 35.3606, 138.7274
        )
        
        assert classification['setting_type'] in ['subduction_zone', 'volcanic_arc', 'fore_arc']
        assert classification['plate_motion_type'] == 'convergent'
        assert classification['stress_regime'] in ['compressional', 'transpressional']
        assert classification['confidence_score'] > 0.5
        
        # Verify tectonic parameters
        params = classification['tectonic_parameters']
        assert params.convergence_rate > 0  # Positive for convergence
        assert params.slab_dip is not None
        assert params.crustal_thickness > 20  # Thickened arc crust
    
    @pytest.mark.asyncio
    async def test_hotspot_classification(self, database_setup):
        """Test classification of hotspot volcanoes"""
        
        tectonic_db = database_setup
        
        # Test Kilauea (Hawaiian hotspot)
        classification = await tectonic_db.classify_tectonic_setting(
            'kilauea', 19.4069, -155.2834
        )
        
        assert classification['setting_type'] == 'hotspot'
        assert classification['volcanic_province']['province_name'] == 'Hawaiian_Hotspot_Province'
        assert classification['confidence_score'] > 0.6
        
        # Verify hotspot-specific parameters
        params = classification['tectonic_parameters']
        assert params.heat_flow > 100  # Elevated heat flow
        assert params.lithospheric_age is not None
    
    @pytest.mark.asyncio
    async def test_continental_rift_classification(self, database_setup):
        """Test classification of continental rift volcanoes"""
        
        tectonic_db = database_setup
        
        # Test Nyiragongo (East African Rift)
        classification = await tectonic_db.classify_tectonic_setting(
            'nyiragongo', -1.5167, 29.25
        )
        
        assert classification['setting_type'] == 'continental_rift'
        assert classification['stress_regime'] in ['extensional', 'transtensional']
        assert classification['volcanic_province']['province_name'] == 'East_African_Rift_Province'
        
        # Verify rift parameters
        params = classification['tectonic_parameters']
        assert params.convergence_rate < 0  # Negative for extension
        assert params.crustal_thickness < 40  # Thinned crust
    
    @pytest.mark.asyncio
    async def test_mid_ocean_ridge_classification(self, database_setup):
        """Test classification of mid-ocean ridge volcanoes"""
        
        tectonic_db = database_setup
        
        # Test Iceland (Mid-Atlantic Ridge)
        classification = await tectonic_db.classify_tectonic_setting(
            'iceland_volcano', 64.1, -21.9
        )
        
        assert classification['setting_type'] in ['mid_ocean_ridge', 'hotspot']
        assert classification['plate_motion_type'] == 'divergent'
        assert classification['stress_regime'] == 'extensional'
        
        # Verify ridge parameters
        params = classification['tectonic_parameters']
        assert params.convergence_rate < 0  # Negative for divergence
        assert params.crustal_thickness <= 10  # Thin oceanic crust
    
    @pytest.mark.asyncio
    async def test_distance_calculations(self, database_setup):
        """Test accuracy of distance calculations to tectonic features"""
        
        tectonic_db = database_setup
        
        # Test distances for Mount St. Helens (Cascadia)
        distances = await tectonic_db._calculate_tectonic_distances(46.1912, -122.1944)
        
        assert 'trench' in distances
        assert 'ridge' in distances
        assert 'transform' in distances
        assert 'hotspot' in distances
        
        # Verify reasonable distance values
        assert 0 < distances['trench'] < 1000  # Should be within 1000 km
        assert distances['trench'] < distances['ridge']  # Closer to trench than ridge
    
    @pytest.mark.asyncio
    async def test_plate_motion_analysis(self, database_setup):
        """Test plate motion analysis accuracy"""
        
        tectonic_db = database_setup
        
        # Test San Andreas Fault region
        plate_motion = await tectonic_db._analyze_plate_motion(37.0, -122.0)
        
        assert plate_motion['type'] in ['transform', 'complex']
        assert 'Pacific' in plate_motion['plates_involved']
        assert 'North American' in plate_motion['plates_involved']
        assert plate_motion['motion_rate'] > 10  # Significant motion
    
    @pytest.mark.asyncio
    async def test_stress_regime_determination(self, database_setup):
        """Test stress regime determination"""
        
        tectonic_db = database_setup
        
        # Test different tectonic settings
        test_locations = [
            (35.0, 140.0, 'compressional'),  # Japan subduction
            (0.0, -110.0, 'extensional'),    # East Pacific Rise
            (-1.0, 29.0, 'extensional'),     # East African Rift
            (37.0, -122.0, 'strike_slip')    # San Andreas
        ]
        
        for lat, lon, expected_regime in test_locations:
            stress_regime = await tectonic_db._determine_stress_regime(lat, lon)
            
            assert stress_regime['regime'] in [expected_regime, 'transtensional', 'transpressional']
            assert 0.0 <= stress_regime['confidence'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_tectonic_parameters_calculation(self, database_setup):
        """Test tectonic parameter calculations"""
        
        tectonic_db = database_setup
        
        # Test subduction zone parameters
        distances = {'trench': 150, 'ridge': 2000, 'transform': 500, 'hotspot': 1000}
        primary_setting = {'type': 'volcanic_arc'}
        
        params = await tectonic_db._calculate_tectonic_parameters(
            35.0, 140.0, primary_setting, distances
        )
        
        assert isinstance(params, TectonicParameters)
        assert params.convergence_rate is not None
        assert params.slab_dip is not None
        assert params.crustal_thickness is not None
        assert params.heat_flow is not None
    
    @pytest.mark.asyncio
    async def test_volcanic_province_classification(self, database_setup):
        """Test volcanic province classification"""
        
        tectonic_db = database_setup
        
        test_cases = [
            (35.0, 140.0, 'Pacific_Ring_of_Fire'),
            (19.0, -155.0, 'Hawaiian_Hotspot_Province'),
            (-1.0, 29.0, 'East_African_Rift_Province'),
            (44.0, -110.0, 'Yellowstone_Hotspot_Province')
        ]
        
        for lat, lon, expected_province in test_cases:
            primary_setting = {'type': 'test_setting'}
            province = await tectonic_db._classify_volcanic_province(lat, lon, primary_setting)
            
            assert province['province_name'] == expected_province
            assert province['province_type'] is not None
            assert province['dominant_magma_type'] is not None
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, database_setup):
        """Test confidence score calculation"""
        
        tectonic_db = database_setup
        
        # High confidence case (clear subduction zone)
        distances_high = {'trench': 50, 'ridge': 2000, 'transform': 1000, 'hotspot': 1500}
        plate_motion_high = {'type': 'convergent', 'motion_rate': 80, 'plates_involved': ['Pacific', 'Eurasian']}
        stress_regime_high = {'regime': 'compressional', 'confidence': 0.9}
        params_high = TectonicParameters(convergence_rate=80, slab_dip=30, crustal_thickness=35)
        
        confidence_high = tectonic_db._calculate_classification_confidence(
            distances_high, plate_motion_high, stress_regime_high, params_high
        )
        
        # Low confidence case (ambiguous setting)
        distances_low = {'trench': 800, 'ridge': 900, 'transform': 850, 'hotspot': 750}
        plate_motion_low = {'type': 'diffuse', 'motion_rate': 2, 'plates_involved': ['Unknown']}
        stress_regime_low = {'regime': 'neutral', 'confidence': 0.3}
        params_low = TectonicParameters()
        
        confidence_low = tectonic_db._calculate_classification_confidence(
            distances_low, plate_motion_low, stress_regime_low, params_low
        )
        
        assert confidence_high > confidence_low
        assert 0.0 <= confidence_high <= 1.0
        assert 0.0 <= confidence_low <= 1.0
    
    @pytest.mark.asyncio
    async def test_tectonic_evolution_context(self, database_setup):
        """Test tectonic evolution context retrieval"""
        
        tectonic_db = database_setup
        
        # First classify a volcano
        classification = await tectonic_db.classify_tectonic_setting(
            'test_volcano', 35.0, 140.0
        )
        
        # Then get evolution context
        evolution = await tectonic_db.get_tectonic_evolution_context('test_volcano')
        
        assert 'current_setting' in evolution
        assert 'evolutionary_stage' in evolution
        assert 'formation_mechanism' in evolution
        assert 'future_evolution' in evolution
        assert 'temporal_context' in evolution
    
    def test_tectonic_setting_enum_values(self):
        """Test tectonic setting enumeration values"""
        
        expected_settings = [
            'subduction_zone', 'mid_ocean_ridge', 'transform_fault', 'hotspot',
            'continental_rift', 'back_arc_basin', 'fore_arc', 'intraplate',
            'collision_zone', 'extensional_basin', 'strike_slip_zone', 'volcanic_arc'
        ]
        
        actual_settings = [setting.value for setting in TectonicSettingType]
        
        for expected in expected_settings:
            assert expected in actual_settings
    
    def test_plate_motion_enum_values(self):
        """Test plate motion enumeration values"""
        
        expected_motions = ['convergent', 'divergent', 'transform', 'complex', 'diffuse']
        actual_motions = [motion.value for motion in PlateMotionType]
        
        for expected in expected_motions:
            assert expected in actual_motions
    
    def test_stress_regime_enum_values(self):
        """Test stress regime enumeration values"""
        
        expected_regimes = [
            'extensional', 'compressional', 'strike_slip', 
            'transtensional', 'transpressional', 'neutral'
        ]
        actual_regimes = [regime.value for regime in StressRegime]
        
        for expected in expected_regimes:
            assert expected in actual_regimes
    
    @pytest.mark.asyncio
    async def test_database_storage_and_retrieval(self, database_setup):
        """Test database storage and retrieval operations"""
        
        tectonic_db = database_setup
        
        # Classify and store a volcano
        volcano_id = 'test_storage_volcano'
        classification = await tectonic_db.classify_tectonic_setting(
            volcano_id, 40.0, -120.0
        )
        
        # Verify storage by retrieving evolution context
        evolution = await tectonic_db.get_tectonic_evolution_context(volcano_id)
        
        assert evolution is not None
        assert evolution['current_setting'] == classification['setting_type']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, database_setup):
        """Test error handling for invalid inputs"""
        
        tectonic_db = database_setup
        
        # Test invalid coordinates
        with pytest.raises(Exception):
            await tectonic_db.classify_tectonic_setting('invalid', 200.0, 400.0)
        
        # Test non-existent volcano for evolution context
        with pytest.raises(ValueError):
            await tectonic_db.get_tectonic_evolution_context('non_existent_volcano')
    
    def test_tectonic_parameters_dataclass(self):
        """Test TectonicParameters dataclass functionality"""
        
        params = TectonicParameters(
            convergence_rate=50.0,
            slab_dip=30.0,
            crustal_thickness=35.0,
            heat_flow=80.0
        )
        
        assert params.convergence_rate == 50.0
        assert params.slab_dip == 30.0
        assert params.crustal_thickness == 35.0
        assert params.heat_flow == 80.0
        assert params.lithospheric_age is None  # Default value
    
    @pytest.mark.asyncio
    async def test_classification_consistency(self, database_setup):
        """Test consistency of classifications for the same location"""
        
        tectonic_db = database_setup
        
        # Classify the same location multiple times
        volcano_id = 'consistency_test'
        lat, lon = 35.0, 140.0
        
        classifications = []
        for i in range(3):
            classification = await tectonic_db.classify_tectonic_setting(
                f'{volcano_id}_{i}', lat, lon
            )
            classifications.append(classification)
        
        # Verify consistency
        first_classification = classifications[0]
        for classification in classifications[1:]:
            assert classification['setting_type'] == first_classification['setting_type']
            assert classification['plate_motion_type'] == first_classification['plate_motion_type']
            assert classification['stress_regime'] == first_classification['stress_regime']
    
    @pytest.mark.asyncio
    async def test_global_coverage(self, database_setup):
        """Test classification coverage for global locations"""
        
        tectonic_db = database_setup
        
        # Test locations across different continents and oceans
        global_locations = [
            ('north_america', 45.0, -120.0),
            ('south_america', -20.0, -70.0),
            ('europe', 45.0, 10.0),
            ('asia', 35.0, 100.0),
            ('africa', 0.0, 30.0),
            ('australia', -25.0, 130.0),
            ('pacific_ocean', 0.0, -150.0),
            ('atlantic_ocean', 30.0, -30.0),
            ('indian_ocean', -20.0, 80.0)
        ]
        
        successful_classifications = 0
        
        for location_name, lat, lon in global_locations:
            try:
                classification = await tectonic_db.classify_tectonic_setting(
                    location_name, lat, lon
                )
                
                # Verify basic classification structure
                assert 'setting_type' in classification
                assert 'confidence_score' in classification
                assert 0.0 <= classification['confidence_score'] <= 1.0
                
                successful_classifications += 1
                
            except Exception as e:
                pytest.fail(f"Classification failed for {location_name}: {str(e)}")
        
        assert successful_classifications == len(global_locations)

class TestTectonicDatabaseInitializer:
    """Test suite for database initialization"""
    
    @pytest.fixture
    def temp_database(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db_url = f"sqlite:///{temp_file.name}"
        
        yield db_url
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_database):
        """Test complete database initialization process"""
        
        initializer = TectonicDatabaseInitializer(temp_database)
        
        # Test table creation
        await initializer._create_tables()
        
        # Verify tables exist
        engine = create_engine(temp_database)
        inspector = engine.inspect(engine)
        table_names = inspector.get_table_names()
        
        expected_tables = [
            'tectonic_settings', 'plate_boundaries', 
            'stress_measurements', 'volcanic_provinces'
        ]
        
        for table in expected_tables:
            assert table in table_names
    
    @pytest.mark.asyncio
    async def test_reference_data_loading(self, temp_database):
        """Test loading of reference data"""
        
        initializer = TectonicDatabaseInitializer(temp_database)
        
        # Initialize database
        await initializer._create_tables()
        
        # Test individual data loading functions
        await initializer._load_plate_boundaries()
        await initializer._load_volcanic_provinces()
        await initializer._load_stress_measurements()
        
        # Verify data was loaded
        session = initializer.SessionLocal()
        
        try:
            from tectonic_setting_database import PlateBoundary, VolcanicProvince, StressMeasurement
            
            boundary_count = session.query(PlateBoundary).count()
            province_count = session.query(VolcanicProvince).count()
            stress_count = session.query(StressMeasurement).count()
            
            assert boundary_count > 0
            assert province_count > 0
            assert stress_count > 0
            
        finally:
            session.close()

# Performance and integration tests
class TestTectonicDatabasePerformance:
    """Performance tests for tectonic database operations"""
    
    @pytest.mark.asyncio
    async def test_classification_performance(self, database_setup):
        """Test classification performance for multiple volcanoes"""
        
        tectonic_db = database_setup
        
        # Test locations
        test_locations = [
            (f'perf_test_{i}', np.random.uniform(-60, 60), np.random.uniform(-180, 180))
            for i in range(10)
        ]
        
        import time
        start_time = time.time()
        
        for volcano_id, lat, lon in test_locations:
            await tectonic_db.classify_tectonic_setting(volcano_id, lat, lon)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 10 classifications in reasonable time
        assert total_time < 30.0  # 30 seconds max
        
        avg_time_per_classification = total_time / len(test_locations)
        assert avg_time_per_classification < 5.0  # 5 seconds per classification max

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])