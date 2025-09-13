"""
Database initialization and setup for Tectonic Setting Database

This module handles database schema creation, initial data loading,
and configuration for the comprehensive tectonic classification system.
"""

import asyncio
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import requests
import json
from datetime import datetime

from tectonic_setting_database import (
    Base, TectonicSetting, PlateBoundary, StressMeasurement, 
    VolcanicProvince, TectonicSettingDatabase
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TectonicDatabaseInitializer:
    """Initialize and populate the tectonic setting database"""
    
    def __init__(self, database_url: str):
        """Initialize database connection"""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    async def initialize_database(self):
        """Complete database initialization process"""
        
        logger.info("Starting tectonic database initialization...")
        
        # Create all tables
        await self._create_tables()
        
        # Load reference data
        await self._load_plate_boundaries()
        await self._load_volcanic_provinces()
        await self._load_stress_measurements()
        await self._create_spatial_indices()
        
        logger.info("Tectonic database initialization completed successfully")
    
    async def _create_tables(self):
        """Create all database tables"""
        
        logger.info("Creating database tables...")
        
        # Create tables using SQLAlchemy
        Base.metadata.create_all(bind=self.engine)
        
        # Create additional PostGIS extensions if needed
        with self.engine.connect() as conn:
            # Enable PostGIS extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            
            # Create custom functions for tectonic analysis
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION calculate_distance_to_boundary(
                    point_geom geometry,
                    boundary_type text
                ) RETURNS float AS $$
                BEGIN
                    RETURN (
                        SELECT MIN(ST_Distance(point_geom, geometry))
                        FROM plate_boundaries
                        WHERE boundary_type = $2
                    );
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            conn.commit()
        
        logger.info("Database tables created successfully")
    
    async def _load_plate_boundaries(self):
        """Load global plate boundary data"""
        
        logger.info("Loading plate boundary data...")
        
        session = self.SessionLocal()
        
        try:
            # Major plate boundaries with detailed geometry
            plate_boundaries_data = [
                {
                    'boundary_name': 'San Andreas Fault System',
                    'boundary_type': 'transform',
                    'geometry': 'LINESTRING(-124.5 40.0, -115.5 32.5)',
                    'relative_motion_rate': 35.0,
                    'motion_direction': 135.0,
                    'motion_obliquity': 15.0,
                    'length': 1200.0,
                    'seismic_activity_rate': 50.0,
                    'data_sources': ['USGS', 'CGS', 'PB2002']
                },
                {
                    'boundary_name': 'Japan Trench',
                    'boundary_type': 'convergent',
                    'geometry': 'LINESTRING(142.0 36.0, 144.0 40.0)',
                    'relative_motion_rate': 83.0,
                    'motion_direction': 285.0,
                    'motion_obliquity': 10.0,
                    'length': 800.0,
                    'seismic_activity_rate': 200.0,
                    'data_sources': ['JMA', 'NIED', 'MORVEL']
                },
                {
                    'boundary_name': 'Mid-Atlantic Ridge North',
                    'boundary_type': 'divergent',
                    'geometry': 'LINESTRING(-30.0 60.0, -20.0 10.0)',
                    'relative_motion_rate': 25.0,
                    'motion_direction': 90.0,
                    'motion_obliquity': 0.0,
                    'length': 6000.0,
                    'seismic_activity_rate': 10.0,
                    'data_sources': ['NOAA', 'MORVEL', 'RIDGE2000']
                },
                {
                    'boundary_name': 'Peru-Chile Trench',
                    'boundary_type': 'convergent',
                    'geometry': 'LINESTRING(-71.0 -18.0, -75.0 -46.0)',
                    'relative_motion_rate': 67.0,
                    'motion_direction': 78.0,
                    'motion_obliquity': 20.0,
                    'length': 5900.0,
                    'seismic_activity_rate': 150.0,
                    'data_sources': ['USGS', 'CSN', 'IGP']
                },
                {
                    'boundary_name': 'East Pacific Rise',
                    'boundary_type': 'divergent',
                    'geometry': 'LINESTRING(-110.0 -55.0, -130.0 55.0)',
                    'relative_motion_rate': 150.0,
                    'motion_direction': 90.0,
                    'motion_obliquity': 0.0,
                    'length': 9000.0,
                    'seismic_activity_rate': 20.0,
                    'data_sources': ['NOAA', 'MORVEL', 'RIDGE2000']
                },
                {
                    'boundary_name': 'Mariana Trench',
                    'boundary_type': 'convergent',
                    'geometry': 'LINESTRING(142.0 11.0, 147.0 25.0)',
                    'relative_motion_rate': 60.0,
                    'motion_direction': 285.0,
                    'motion_obliquity': 5.0,
                    'length': 2550.0,
                    'seismic_activity_rate': 80.0,
                    'data_sources': ['JMA', 'USGS', 'MORVEL']
                },
                {
                    'boundary_name': 'North Anatolian Fault',
                    'boundary_type': 'transform',
                    'geometry': 'LINESTRING(30.0 40.0, 42.0 41.0)',
                    'relative_motion_rate': 25.0,
                    'motion_direction': 90.0,
                    'motion_obliquity': 5.0,
                    'length': 1500.0,
                    'seismic_activity_rate': 30.0,
                    'data_sources': ['KOERI', 'AFAD', 'EMSC']
                },
                {
                    'boundary_name': 'Alpine Fault',
                    'boundary_type': 'transform',
                    'geometry': 'LINESTRING(166.0 -46.0, 171.0 -42.0)',
                    'relative_motion_rate': 38.0,
                    'motion_direction': 45.0,
                    'motion_obliquity': 10.0,
                    'length': 600.0,
                    'seismic_activity_rate': 15.0,
                    'data_sources': ['GNS', 'GeoNet', 'MORVEL']
                }
            ]
            
            for boundary_data in plate_boundaries_data:
                plate_boundary = PlateBoundary(
                    boundary_name=boundary_data['boundary_name'],
                    boundary_type=boundary_data['boundary_type'],
                    geometry=boundary_data['geometry'],
                    relative_motion_rate=boundary_data['relative_motion_rate'],
                    motion_direction=boundary_data['motion_direction'],
                    motion_obliquity=boundary_data['motion_obliquity'],
                    length=boundary_data['length'],
                    seismic_activity_rate=boundary_data['seismic_activity_rate'],
                    data_sources=boundary_data['data_sources']
                )
                
                session.add(plate_boundary)
            
            session.commit()
            logger.info(f"Loaded {len(plate_boundaries_data)} plate boundaries")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading plate boundaries: {str(e)}")
            raise
        finally:
            session.close()
    
    async def _load_volcanic_provinces(self):
        """Load volcanic province data"""
        
        logger.info("Loading volcanic province data...")
        
        session = self.SessionLocal()
        
        try:
            volcanic_provinces_data = [
                {
                    'province_name': 'Pacific Ring of Fire',
                    'province_type': 'convergent_margin_complex',
                    'boundary_geometry': 'POLYGON((-180 -60, 180 -60, 180 60, -180 60, -180 -60))',
                    'area': 40000000.0,  # km²
                    'dominant_tectonic_setting': 'subduction_zone',
                    'tectonic_evolution': 'Complex subduction system around Pacific Ocean',
                    'formation_mechanism': 'Pacific Plate subduction',
                    'volcano_count': 850,
                    'dominant_magma_type': 'calc-alkaline',
                    'eruption_style': 'explosive',
                    'initiation_age': 200.0,  # Ma
                    'current_activity_level': 'very_high'
                },
                {
                    'province_name': 'Mid-Atlantic Ridge Province',
                    'province_type': 'divergent_margin',
                    'boundary_geometry': 'POLYGON((-50 -70, -10 -70, -10 70, -50 70, -50 -70))',
                    'area': 10000000.0,
                    'dominant_tectonic_setting': 'mid_ocean_ridge',
                    'tectonic_evolution': 'Atlantic Ocean spreading center',
                    'formation_mechanism': 'seafloor spreading',
                    'volcano_count': 200,
                    'dominant_magma_type': 'tholeiitic',
                    'eruption_style': 'effusive',
                    'initiation_age': 180.0,
                    'current_activity_level': 'moderate'
                },
                {
                    'province_name': 'East African Rift Province',
                    'province_type': 'continental_rift',
                    'boundary_geometry': 'POLYGON((28 -15, 45 -15, 45 20, 28 20, 28 -15))',
                    'area': 3000000.0,
                    'dominant_tectonic_setting': 'continental_rift',
                    'tectonic_evolution': 'Active continental rifting and breakup',
                    'formation_mechanism': 'lithospheric extension',
                    'volcano_count': 150,
                    'dominant_magma_type': 'alkaline',
                    'eruption_style': 'mixed',
                    'initiation_age': 30.0,
                    'current_activity_level': 'high'
                },
                {
                    'province_name': 'Yellowstone Hotspot Province',
                    'province_type': 'hotspot',
                    'boundary_geometry': 'POLYGON((-112 42, -108 42, -108 46, -112 46, -112 42))',
                    'area': 200000.0,
                    'dominant_tectonic_setting': 'hotspot',
                    'tectonic_evolution': 'Hotspot track across North America',
                    'formation_mechanism': 'mantle plume',
                    'volcano_count': 50,
                    'dominant_magma_type': 'rhyolitic',
                    'eruption_style': 'super_explosive',
                    'initiation_age': 17.0,
                    'current_activity_level': 'moderate'
                },
                {
                    'province_name': 'Hawaiian Hotspot Province',
                    'province_type': 'hotspot',
                    'boundary_geometry': 'POLYGON((-160 18, -154 18, -154 22, -160 22, -160 18))',
                    'area': 150000.0,
                    'dominant_tectonic_setting': 'hotspot',
                    'tectonic_evolution': 'Hawaiian-Emperor hotspot chain',
                    'formation_mechanism': 'mantle plume',
                    'volcano_count': 130,
                    'dominant_magma_type': 'basaltic',
                    'eruption_style': 'effusive',
                    'initiation_age': 85.0,
                    'current_activity_level': 'high'
                },
                {
                    'province_name': 'Andes Volcanic Province',
                    'province_type': 'continental_arc',
                    'boundary_geometry': 'POLYGON((-80 -50, -65 -50, -65 10, -80 10, -80 -50))',
                    'area': 8000000.0,
                    'dominant_tectonic_setting': 'volcanic_arc',
                    'tectonic_evolution': 'Andean orogeny and arc volcanism',
                    'formation_mechanism': 'Nazca Plate subduction',
                    'volcano_count': 450,
                    'dominant_magma_type': 'calc-alkaline',
                    'eruption_style': 'explosive',
                    'initiation_age': 200.0,
                    'current_activity_level': 'high'
                }
            ]
            
            for province_data in volcanic_provinces_data:
                volcanic_province = VolcanicProvince(
                    province_name=province_data['province_name'],
                    province_type=province_data['province_type'],
                    boundary_geometry=province_data['boundary_geometry'],
                    area=province_data['area'],
                    dominant_tectonic_setting=province_data['dominant_tectonic_setting'],
                    tectonic_evolution=province_data['tectonic_evolution'],
                    formation_mechanism=province_data['formation_mechanism'],
                    volcano_count=province_data['volcano_count'],
                    dominant_magma_type=province_data['dominant_magma_type'],
                    eruption_style=province_data['eruption_style'],
                    initiation_age=province_data['initiation_age'],
                    current_activity_level=province_data['current_activity_level'],
                    data_sources=['internal_classification', 'GVP', 'geological_surveys']
                )
                
                session.add(volcanic_province)
            
            session.commit()
            logger.info(f"Loaded {len(volcanic_provinces_data)} volcanic provinces")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading volcanic provinces: {str(e)}")
            raise
        finally:
            session.close()
    
    async def _load_stress_measurements(self):
        """Load representative stress measurement data"""
        
        logger.info("Loading stress measurement data...")
        
        session = self.SessionLocal()
        
        try:
            # Representative stress measurements from different tectonic settings
            stress_measurements_data = [
                {
                    'location': 'POINT(138.7 35.4)',  # Japan
                    'measurement_type': 'focal_mechanism',
                    'sigma1_magnitude': 120.0,
                    'sigma1_azimuth': 285.0,
                    'sigma1_plunge': 15.0,
                    'sigma2_magnitude': 80.0,
                    'sigma2_azimuth': 195.0,
                    'sigma2_plunge': 5.0,
                    'sigma3_magnitude': 40.0,
                    'sigma3_azimuth': 105.0,
                    'sigma3_plunge': 75.0,
                    'stress_ratio': 0.5,
                    'differential_stress': 80.0,
                    'mean_stress': 80.0,
                    'depth': 15.0,
                    'uncertainty': 10.0,
                    'data_source': 'NIED_focal_mechanisms'
                },
                {
                    'location': 'POINT(-122.2 46.2)',  # Cascadia
                    'measurement_type': 'borehole_breakout',
                    'sigma1_magnitude': 90.0,
                    'sigma1_azimuth': 45.0,
                    'sigma1_plunge': 10.0,
                    'sigma2_magnitude': 60.0,
                    'sigma2_azimuth': 315.0,
                    'sigma2_plunge': 5.0,
                    'sigma3_magnitude': 30.0,
                    'sigma3_azimuth': 225.0,
                    'sigma3_plunge': 80.0,
                    'stress_ratio': 0.5,
                    'differential_stress': 60.0,
                    'mean_stress': 60.0,
                    'depth': 5.0,
                    'uncertainty': 15.0,
                    'data_source': 'USGS_borehole_data'
                },
                {
                    'location': 'POINT(-70.0 -25.0)',  # Andes
                    'measurement_type': 'focal_mechanism',
                    'sigma1_magnitude': 150.0,
                    'sigma1_azimuth': 90.0,
                    'sigma1_plunge': 5.0,
                    'sigma2_magnitude': 100.0,
                    'sigma2_azimuth': 0.0,
                    'sigma2_plunge': 10.0,
                    'sigma3_magnitude': 50.0,
                    'sigma3_azimuth': 270.0,
                    'sigma3_plunge': 80.0,
                    'stress_ratio': 0.5,
                    'differential_stress': 100.0,
                    'mean_stress': 100.0,
                    'depth': 20.0,
                    'uncertainty': 12.0,
                    'data_source': 'CSN_seismic_network'
                },
                {
                    'location': 'POINT(35.0 5.0)',  # East African Rift
                    'measurement_type': 'focal_mechanism',
                    'sigma1_magnitude': 60.0,
                    'sigma1_azimuth': 0.0,
                    'sigma1_plunge': 85.0,
                    'sigma2_magnitude': 40.0,
                    'sigma2_azimuth': 90.0,
                    'sigma2_plunge': 5.0,
                    'sigma3_magnitude': 20.0,
                    'sigma3_azimuth': 0.0,
                    'sigma3_plunge': 0.0,
                    'stress_ratio': 0.5,
                    'differential_stress': 40.0,
                    'mean_stress': 40.0,
                    'depth': 10.0,
                    'uncertainty': 20.0,
                    'data_source': 'IRIS_AfricaArray'
                }
            ]
            
            for stress_data in stress_measurements_data:
                stress_measurement = StressMeasurement(
                    location=stress_data['location'],
                    measurement_type=stress_data['measurement_type'],
                    sigma1_magnitude=stress_data['sigma1_magnitude'],
                    sigma1_azimuth=stress_data['sigma1_azimuth'],
                    sigma1_plunge=stress_data['sigma1_plunge'],
                    sigma2_magnitude=stress_data['sigma2_magnitude'],
                    sigma2_azimuth=stress_data['sigma2_azimuth'],
                    sigma2_plunge=stress_data['sigma2_plunge'],
                    sigma3_magnitude=stress_data['sigma3_magnitude'],
                    sigma3_azimuth=stress_data['sigma3_azimuth'],
                    sigma3_plunge=stress_data['sigma3_plunge'],
                    stress_ratio=stress_data['stress_ratio'],
                    differential_stress=stress_data['differential_stress'],
                    mean_stress=stress_data['mean_stress'],
                    depth=stress_data['depth'],
                    uncertainty=stress_data['uncertainty'],
                    data_source=stress_data['data_source'],
                    measurement_date=datetime.utcnow()
                )
                
                session.add(stress_measurement)
            
            session.commit()
            logger.info(f"Loaded {len(stress_measurements_data)} stress measurements")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading stress measurements: {str(e)}")
            raise
        finally:
            session.close()
    
    async def _create_spatial_indices(self):
        """Create spatial indices for efficient queries"""
        
        logger.info("Creating spatial indices...")
        
        with self.engine.connect() as conn:
            # Spatial indices for geometry columns
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_tectonic_settings_location ON tectonic_settings USING GIST (location);",
                "CREATE INDEX IF NOT EXISTS idx_plate_boundaries_geometry ON plate_boundaries USING GIST (geometry);",
                "CREATE INDEX IF NOT EXISTS idx_stress_measurements_location ON stress_measurements USING GIST (location);",
                "CREATE INDEX IF NOT EXISTS idx_volcanic_provinces_boundary ON volcanic_provinces USING GIST (boundary_geometry);",
                
                # Regular indices for common queries
                "CREATE INDEX IF NOT EXISTS idx_tectonic_settings_volcano_id ON tectonic_settings (volcano_id);",
                "CREATE INDEX IF NOT EXISTS idx_tectonic_settings_type ON tectonic_settings (setting_type);",
                "CREATE INDEX IF NOT EXISTS idx_plate_boundaries_type ON plate_boundaries (boundary_type);",
                "CREATE INDEX IF NOT EXISTS idx_stress_measurements_type ON stress_measurements (measurement_type);"
            ]
            
            for index_sql in indices:
                conn.execute(text(index_sql))
            
            conn.commit()
        
        logger.info("Spatial indices created successfully")

async def initialize_tectonic_database(database_url: str):
    """Main function to initialize the tectonic database"""
    
    try:
        initializer = TectonicDatabaseInitializer(database_url)
        await initializer.initialize_database()
        
        # Test the database with sample classifications
        await test_database_functionality(database_url)
        
        logger.info("Tectonic database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize tectonic database: {str(e)}")
        raise

async def test_database_functionality(database_url: str):
    """Test database functionality with sample classifications"""
    
    logger.info("Testing database functionality...")
    
    tectonic_db = TectonicSettingDatabase(database_url)
    
    # Test volcanoes
    test_cases = [
        {'id': 'test_fuji', 'lat': 35.3606, 'lon': 138.7274, 'expected': 'subduction_zone'},
        {'id': 'test_kilauea', 'lat': 19.4069, 'lon': -155.2834, 'expected': 'hotspot'},
        {'id': 'test_vesuvius', 'lat': 40.8214, 'lon': 14.4289, 'expected': 'subduction_zone'},
        {'id': 'test_nyiragongo', 'lat': -1.5167, 'lon': 29.25, 'expected': 'continental_rift'}
    ]
    
    successful_tests = 0
    
    for test_case in test_cases:
        try:
            classification = await tectonic_db.classify_tectonic_setting(
                test_case['id'], test_case['lat'], test_case['lon']
            )
            
            logger.info(f"Test {test_case['id']}: {classification['setting_type']} "
                       f"(confidence: {classification['confidence_score']:.2f})")
            
            successful_tests += 1
            
        except Exception as e:
            logger.error(f"Test failed for {test_case['id']}: {str(e)}")
    
    logger.info(f"Database functionality test completed: {successful_tests}/{len(test_cases)} tests passed")

if __name__ == "__main__":
    # Example usage
    database_url = "postgresql://prometheus:prometheus123@localhost:5432/prometheus"
    asyncio.run(initialize_tectonic_database(database_url))