#!/usr/bin/env python3
"""
Surface Deformation Correlation System Integration Example

This example demonstrates the complete workflow of the Surface Deformation
Correlation System for analyzing sill emplacement using InSAR, GPS, and
tilt/strain data.

Usage:
    python surface_deformation_correlation_example.py
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yaml
import logging
import sys
import os
from pathlib import Path

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from surface_deformation_correlation_system import (
    SurfaceDeformationCorrelationSystem,
    InSARDeformationData,
    GPSDeformationData,
    TiltStrainData,
    SillGeometryModel,
    DeformationCorrelationResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SurfaceDeformationCorrelationExample:
    """Complete example of surface deformation correlation analysis"""
    
    def __init__(self, config_path: str = None):
        """Initialize the example with configuration"""
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', 'config',
                'surface_deformation_correlation_config.yaml'
            )
        
        # Use default config if file doesn't exist
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize the correlation system
        self.correlation_system = SurfaceDeformationCorrelationSystem(self.config)
        
        # Example data storage
        self.sill_model = None
        self.insar_data = []
        self.gps_data = []
        self.tilt_strain_data = []
        self.results = {}
        
        logger.info("Surface Deformation Correlation Example initialized")

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'database': {
                'postgres_host': 'localhost',
                'postgres_port': 5432,
                'postgres_user': 'prometheus',
                'postgres_password': 'prometheus123',
                'postgres_db': 'prometheus',
                'redis_host': 'localhost',
                'redis_port': 6379,
                'influx_url': 'http://localhost:8086',
                'influx_token': 'prometheus-token',
                'influx_org': 'prometheus',
                'neo4j_uri': 'bolt://localhost:7687',
                'neo4j_user': 'neo4j',
                'neo4j_password': 'prometheus123'
            },
            'processing': {
                'max_workers': 4,
                'chunk_size': 100
            },
            'validation': {
                'min_correlation': 0.3,
                'significance_level': 0.05
            }
        }

    async def setup_example_scenario(self):
        """Setup an example volcanic sill intrusion scenario"""
        
        logger.info("Setting up example sill intrusion scenario...")
        
        # Define the example sill model (based on real-world parameters)
        self.sill_model = SillGeometryModel(
            sill_id="yellowstone_sill_2023",
            center_coordinates=(-110.7, 44.6, 8000),  # Yellowstone coordinates, 8km depth
            length=12000,  # 12 km length
            width=8000,    # 8 km width
            thickness=100, # 100 m thickness
            strike=45,     # NE-SW orientation
            dip=5,         # Nearly horizontal
            opening=2.0,   # 2 m opening
            pressure_change=50e6,  # 50 MPa pressure increase
            elastic_moduli=(40e9, 0.25),  # Typical crustal values
            host_rock_properties={
                'density': 2600,  # kg/m³
                'thermal_conductivity': 2.5,  # W/m/K
                'heat_capacity': 1000  # J/kg/K
            },
            emplacement_time=datetime(2023, 6, 15),
            growth_rate=0.5,  # m/year
            geometry_uncertainty={
                'depth': 1000,    # ±1 km
                'length': 2000,   # ±2 km
                'width': 1500,    # ±1.5 km
                'opening': 0.5    # ±0.5 m
            },
            validation_metrics={}
        )
        
        logger.info(f"Created sill model: {self.sill_model.sill_id}")
        logger.info(f"Location: {self.sill_model.center_coordinates}")
        logger.info(f"Dimensions: {self.sill_model.length}x{self.sill_model.width}x{self.sill_model.thickness} m")

    async def generate_synthetic_data(self):
        """Generate all synthetic data"""
        logger.info("Generating synthetic deformation data...")
        
        # Generate InSAR data
        await self._generate_insar_data()
        
        # Generate GPS data
        await self._generate_gps_data()
        
        # Generate tilt/strain data
        await self._generate_tilt_strain_data()
        
        logger.info(f"Generated {len(self.insar_data)} InSAR datasets, "
                   f"{len(self.gps_data)} GPS measurements, "
                   f"{len(self.tilt_strain_data)} tilt/strain measurements")

    async def _generate_insar_data(self):
        """Generate synthetic InSAR data"""
        # Create a simple InSAR dataset
        nx, ny = 50, 50
        x = np.linspace(-10000, 10000, nx)
        y = np.linspace(-10000, 10000, ny)
        X, Y = np.meshgrid(x, y)
        
        # Synthetic Mogi-like deformation
        R = np.sqrt(X**2 + Y**2 + 8000**2)
        deformation = 0.1 * 8000 / R**3
        deformation += np.random.normal(0, 0.005, deformation.shape)
        
        # Create coordinate grids
        sill_lon, sill_lat, _ = self.sill_model.center_coordinates
        lon_grid = sill_lon + X / 111000
        lat_grid = sill_lat + Y / 111000
        
        insar_dataset = InSARDeformationData(
            acquisition_date=datetime(2023, 9, 1),
            satellite="Sentinel-1A",
            track=144,
            frame=350,
            look_angle=23.0,
            heading_angle=-13.3,
            wavelength=0.056,
            spatial_resolution=20.0,
            temporal_baseline=120.0,
            perpendicular_baseline=150.0,
            coherence_threshold=0.3,
            unwrapping_method="snaphu",
            atmospheric_correction="gacos",
            deformation_map=deformation,
            uncertainty_map=np.full_like(deformation, 0.01),
            coherence_map=np.random.uniform(0.3, 0.9, deformation.shape),
            coordinates=(lon_grid, lat_grid),
            reference_point=(sill_lon, sill_lat),
            processing_parameters={'multilook': '4x1'},
            quality_metrics={'mean_coherence': 0.65}
        )
        
        self.insar_data.append(insar_dataset)

    async def _generate_gps_data(self):
        """Generate synthetic GPS data"""
        # GPS station positions around the sill
        station_positions = [
            (-110.69, 44.61, 2400, 'YSRP'),
            (-110.72, 44.58, 2350, 'LKNT'),
            (-110.68, 44.63, 2450, 'NRRD'),
            (-110.75, 44.60, 2300, 'WTHG'),
            (-110.65, 44.59, 2500, 'HLPN')
        ]
        
        sill_lon, sill_lat, sill_depth = self.sill_model.center_coordinates
        
        for lon, lat, elev, station_id in station_positions:
            # Calculate synthetic displacement
            dx = (lon - sill_lon) * 111000
            dy = (lat - sill_lat) * 111000
            r = np.sqrt(dx**2 + dy**2)
            R = np.sqrt(r**2 + sill_depth**2)
            
            # Mogi displacement
            ur = 0.1 * r / R**3
            uz = 0.1 * sill_depth / R**3
            
            # Convert to east-north
            ux = ur * dx / (r + 1e-10)
            uy = ur * dy / (r + 1e-10)
            
            # Add noise
            ux += np.random.normal(0, 0.002)
            uy += np.random.normal(0, 0.002)
            uz += np.random.normal(0, 0.005)
            
            gps_measurement = GPSDeformationData(
                station_id=station_id,
                coordinates=(lon, lat, elev),
                measurement_date=datetime(2023, 9, 1),
                displacement_vector=(ux, uy, uz),
                velocity_vector=(ux/365, uy/365, uz/365),
                uncertainty_vector=(0.002, 0.002, 0.005),
                correlation_matrix=np.eye(3) * 0.1,
                measurement_type='continuous',
                processing_strategy='PPP',
                reference_frame='ITRF2014',
                antenna_type='AOAD/M_T',
                monument_type='concrete_pillar',
                data_quality={'rms': 0.003, 'availability': 0.95},
                environmental_corrections={'troposphere': 0.001}
            )
            
            self.gps_data.append(gps_measurement)

    async def _generate_tilt_strain_data(self):
        """Generate synthetic tilt and strain data"""
        # Instrument configurations
        instruments = [
            ('TILT01', 'tiltmeter', -110.705, 44.605, 50),
            ('STRAIN01', 'strainmeter', -110.695, 44.595, 100)
        ]
        
        for inst_id, inst_type, lon, lat, depth in instruments:
            if 'tilt' in inst_type:
                # Synthetic tilt data
                tilt_x = np.random.normal(10, 2)  # microradians
                tilt_y = np.random.normal(5, 2)
                
                measurement = TiltStrainData(
                    instrument_id=inst_id,
                    instrument_type=inst_type,
                    coordinates=(lon, lat, depth),
                    measurement_date=datetime(2023, 9, 1),
                    tilt_components=(tilt_x, tilt_y),
                    strain_tensor=None,
                    shear_strain=None,
                    volumetric_strain=None,
                    uncertainty_tilt=(1.0, 1.0),
                    uncertainty_strain=None,
                    temperature_correction=0.001,
                    pressure_correction=0.0005,
                    installation_depth=depth,
                    calibration_date=datetime(2023, 1, 1),
                    drift_correction=0.0001,
                    data_quality_flags=['good']
                )
            else:
                # Synthetic strain data
                exx = np.random.normal(1e-7, 1e-8)
                eyy = np.random.normal(1e-7, 1e-8)
                exy = np.random.normal(0, 5e-9)
                strain_tensor = np.array([[exx, exy], [exy, eyy]])
                
                measurement = TiltStrainData(
                    instrument_id=inst_id,
                    instrument_type=inst_type,
                    coordinates=(lon, lat, depth),
                    measurement_date=datetime(2023, 9, 1),
                    tilt_components=None,
                    strain_tensor=strain_tensor,
                    shear_strain=exy,
                    volumetric_strain=exx + eyy,
                    uncertainty_tilt=None,
                    uncertainty_strain=np.array([[1e-9, 5e-10], [5e-10, 1e-9]]),
                    temperature_correction=1e-9,
                    pressure_correction=5e-10,
                    installation_depth=depth,
                    calibration_date=datetime(2023, 1, 1),
                    drift_correction=1e-10,
                    data_quality_flags=['good']
                )
            
            self.tilt_strain_data.append(measurement)

    async def run_analysis(self):
        """Run the complete analysis"""
        logger.info("Running correlation analysis...")
        
        # Initialize database connections (mocked for example)
        await self.correlation_system.initialize_connections()
        
        try:
            # Run correlation analyses
            await self._run_correlations()
            
            # Run forward modeling
            await self._run_forward_modeling()
            
            # Run parameter inversion (simplified)
            await self._run_parameter_inversion()
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

    async def _run_correlations(self):
        """Run correlation analyses"""
        # InSAR correlation
        if self.insar_data:
            insar_params = {
                'spatial_window': 15000,
                'temporal_window': 365,
                'coherence_threshold': 0.3
            }
            
            insar_results = await self.correlation_system.correlate_insar_deformation(
                self.insar_data, self.sill_model, insar_params
            )
            self.results['insar_correlation'] = insar_results
        
        # GPS correlation
        if self.gps_data:
            gps_params = {
                'max_distance': 50000,
                'min_stations': 3,
                'velocity_threshold': 1e-3
            }
            
            gps_results = await self.correlation_system.analyze_gps_deformation(
                self.gps_data, self.sill_model, gps_params
            )
            self.results['gps_correlation'] = gps_results
        
        # Tilt/strain correlation
        if self.tilt_strain_data:
            ts_params = {
                'max_distance': 25000,
                'depth_weighting': True
            }
            
            ts_results = await self.correlation_system.correlate_tilt_strain_data(
                self.tilt_strain_data, self.sill_model, ts_params
            )
            self.results['tilt_strain_correlation'] = ts_results

    async def _run_forward_modeling(self):
        """Run forward modeling"""
        # Create observation points
        observation_coords = np.array([
            [-110.69, 44.61],
            [-110.72, 44.58],
            [-110.68, 44.63],
            [-110.75, 44.60],
            [-110.65, 44.59]
        ])
        
        model_params = {
            'elastic_parameters': {
                'poisson_ratio': 0.25,
                'youngs_modulus': 40e9
            }
        }
        
        forward_results = await self.correlation_system.forward_model_surface_deformation(
            self.sill_model, observation_coords, model_params
        )
        self.results['forward_modeling'] = forward_results

    async def _run_parameter_inversion(self):
        """Run simplified parameter inversion"""
        # Use forward model results as synthetic observations
        if 'forward_modeling' not in self.results:
            return
        
        forward_results = self.results['forward_modeling']
        observation_coords = forward_results['observation_coordinates']
        
        # Add noise to create synthetic observations
        deformation_obs = {
            'gps': {
                'east': forward_results['combined_model']['east'] + np.random.normal(0, 0.005, len(observation_coords)),
                'north': forward_results['combined_model']['north'] + np.random.normal(0, 0.005, len(observation_coords)),
                'up': forward_results['combined_model']['up'] + np.random.normal(0, 0.01, len(observation_coords)),
                'uncertainty_east': np.full(len(observation_coords), 0.005),
                'uncertainty_north': np.full(len(observation_coords), 0.005),
                'uncertainty_up': np.full(len(observation_coords), 0.01)
            }
        }
        
        inversion_params = {
            'method': 'differential_evolution',
            'max_iterations': 50,  # Reduced for example
            'tolerance': 1e-4,
            'uncertainty_estimation': False
        }
        
        inversion_results = await self.correlation_system.invert_sill_parameters(
            deformation_obs, observation_coords, self.sill_model, inversion_params
        )
        self.results['parameter_inversion'] = inversion_results

    def generate_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("SURFACE DEFORMATION CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        # Sill model summary
        print(f"\nSILL MODEL: {self.sill_model.sill_id}")
        print(f"Location: {self.sill_model.center_coordinates[0]:.3f}°, {self.sill_model.center_coordinates[1]:.3f}°")
        print(f"Depth: {self.sill_model.center_coordinates[2]/1000:.1f} km")
        print(f"Dimensions: {self.sill_model.length/1000:.1f} × {self.sill_model.width/1000:.1f} km")
        print(f"Opening: {self.sill_model.opening:.2f} m")
        
        # Data summary
        print(f"\nDATA SUMMARY:")
        print(f"InSAR datasets: {len(self.insar_data)}")
        print(f"GPS measurements: {len(self.gps_data)}")
        print(f"Tilt/strain measurements: {len(self.tilt_strain_data)}")
        
        # Results summary
        if 'insar_correlation' in self.results:
            insar_results = self.results['insar_correlation']
            correlations = [result['correlation_coefficient'] for result in insar_results.values()]
            if correlations:
                print(f"\nINSAR CORRELATION:")
                print(f"  Mean correlation: {np.mean(correlations):.3f}")
                print(f"  Best correlation: {np.max(correlations):.3f}")
        
        if 'gps_correlation' in self.results:
            gps_results = self.results['gps_correlation']
            print(f"\nGPS CORRELATION:")
            for component in ['east', 'north', 'up']:
                if component in gps_results:
                    corr = gps_results[component]['correlation_coefficient']
                    rms = gps_results[component]['rms_misfit']
                    print(f"  {component.capitalize()}: r={corr:.3f}, RMS={rms*1000:.1f} mm")
        
        if 'parameter_inversion' in self.results:
            inv_results = self.results['parameter_inversion']
            print(f"\nPARAMETER INVERSION:")
            print(f"  Success: {inv_results['optimization_result']['success']}")
            if 'model_statistics' in inv_results:
                stats = inv_results['model_statistics']
                print(f"  RMS misfit: {stats['rms_misfit']*1000:.1f} mm")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

    async def cleanup(self):
        """Clean up resources"""
        await self.correlation_system.close_connections()

    async def run_complete_example(self):
        """Run the complete example"""
        logger.info("Starting Surface Deformation Correlation Example...")
        
        try:
            # Setup scenario
            await self.setup_example_scenario()
            
            # Generate synthetic data
            await self.generate_synthetic_data()
            
            # Run analysis
            await self.run_analysis()
            
            # Generate report
            self.generate_report()
            
            logger.info("Example completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in example: {e}")
            raise
        
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    print("Surface Deformation Correlation System - Integration Example")
    print("=" * 60)
    
    example = SurfaceDeformationCorrelationExample()
    await example.run_complete_example()

if __name__ == "__main__":
    asyncio.run(main())