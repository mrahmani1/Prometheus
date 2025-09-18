"""
Stacked-Sill Geometry Modeling Engine - Core Implementation
Advanced 3D geometric modeling of stacked-sill intrusion complexes

This module implements sophisticated algorithms for:
- 3D sill shape reconstruction from seismic and structural data
- Thickness variation modeling based on host rock properties
- Lateral extent determination using geophysical constraints
- Sill interconnection and feeding relationship analysis
- Structural control analysis on sill emplacement geometry
- Multi-scale sill modeling from individual sills to regional complexes
- Temporal sill geometry evolution and growth modeling
- Validation against exposed sill complexes and field measurements
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import pickle
import gzip
from scipy import optimize, interpolate, spatial, ndimage
import warnings

# Database integration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.database_integration import DatabaseIntegration
    from core.gis_postgresql_operations import GISPostgreSQLOperations
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    warnings.warn("Database integration not available. Data persistence will be limited.")

# ============================================================================
# Data Models and Structures
# ============================================================================

@dataclass
class StructuralData:
    """Structural geological data for sill reconstruction"""
    fracture_networks: np.ndarray  # 3D fracture orientations and positions
    bedding_planes: np.ndarray  # Host rock bedding orientations
    fault_systems: np.ndarray  # Fault plane orientations and properties
    joint_sets: np.ndarray  # Joint orientations and spacing
    lithological_contacts: np.ndarray  # Lithological boundary positions
    structural_measurements: Dict[str, np.ndarray]  # Field measurements
    coordinates: Dict[str, np.ndarray]  # x, y, z coordinates (km)
    uncertainty: Dict[str, np.ndarray]  # Measurement uncertainties
    data_quality: Dict[str, float]  # Quality indicators
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SeismicStructuralData:
    """Seismic data optimized for sill detection"""
    reflection_data: np.ndarray  # 3D seismic reflection amplitudes
    coherence_attributes: np.ndarray  # Seismic coherence for discontinuities
    amplitude_attributes: np.ndarray  # Amplitude-based sill indicators
    frequency_attributes: np.ndarray  # Frequency domain attributes
    coordinates: Dict[str, np.ndarray]  # x, y, z coordinates (km)
    acquisition_geometry: Dict[str, Any]  # Seismic acquisition parameters
    processing_parameters: Dict[str, Any]  # Data processing details
    resolution: Dict[str, float]  # Spatial resolution limits
    signal_to_noise: float  # Data quality metric
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class HostRockProperties:
    """Host rock mechanical and thermal properties"""
    rock_type: str  # Lithological classification
    mechanical_properties: Dict[str, float]  # Strength, modulus, etc.
    thermal_properties: Dict[str, float]  # Conductivity, capacity, etc.
    permeability: float  # Hydraulic permeability (m²)
    porosity: float  # Rock porosity (fraction)
    density: float  # Rock density (kg/m³)
    elastic_moduli: Dict[str, float]  # Young's modulus, Poisson's ratio
    failure_criteria: Dict[str, float]  # Tensile/shear strength
    anisotropy_tensor: np.ndarray  # Mechanical anisotropy
    layering_properties: Dict[str, Any]  # Stratification characteristics
    alteration_zones: List[Dict[str, Any]]  # Hydrothermally altered regions
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SillGeometry3D:
    """3D sill geometry representation"""
    sill_id: str
    center_coordinates: Tuple[float, float, float]  # x, y, z (km)
    lateral_extent: Dict[str, float]  # Length, width (km)
    thickness_distribution: np.ndarray  # Thickness variation (m)
    volume: float  # Sill volume (km³)
    surface_mesh: np.ndarray  # 3D surface mesh points
    shape_parameters: Dict[str, float]  # Geometric descriptors
    orientation: Dict[str, float]  # Strike, dip angles
    aspect_ratios: Tuple[float, float]  # Length/width, thickness/width
    irregularity_index: float  # Shape complexity measure
    emplacement_depth: float  # Average emplacement depth (km)
    host_rock_contact: np.ndarray  # Contact surface geometry
    internal_structure: Dict[str, Any]  # Internal heterogeneity
    uncertainty_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    confidence_level: float = 0.95
    reconstruction_method: str = 'seismic_structural'
    validation_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SillInterconnection:
    """Sill interconnection and feeding relationships"""
    connection_id: str
    parent_sill: str
    child_sill: str
    connection_type: str  # 'direct', 'indirect', 'feeder'
    connection_geometry: np.ndarray  # 3D connection pathway
    flow_capacity: float  # Magma transport capacity (m³/s)
    pressure_drop: float  # Pressure loss across connection (Pa)
    geometric_complexity: float  # Connection tortuosity
    activation_threshold: float  # Pressure threshold for flow (Pa)
    temporal_activity: List[Tuple[datetime, float]]  # Activity history
    uncertainty: Dict[str, float]  # Parameter uncertainties
    validation_evidence: List[str]  # Supporting evidence types
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SillComplex:
    """Complete stacked-sill complex representation"""
    complex_id: str
    individual_sills: List[SillGeometry3D]
    interconnections: List[SillInterconnection]
    complex_geometry: Dict[str, Any]  # Overall complex properties
    stratigraphic_levels: List[float]  # Emplacement levels (km depth)
    total_volume: float  # Total complex volume (km³)
    emplacement_sequence: List[str]  # Temporal emplacement order
    structural_controls: Dict[str, Any]  # Controlling structures
    thermal_evolution: Dict[str, Any]  # Thermal history
    mechanical_evolution: Dict[str, Any]  # Stress field evolution
    hazard_assessment: Dict[str, float]  # Associated hazards
    uncertainty_analysis: Dict[str, Any]  # Complex-wide uncertainties
    timestamp: datetime = field(default_factory=datetime.now)

# ============================================================================
# Stacked-Sill Geometry Modeling Engine
# ============================================================================

class StackedSillGeometryEngine:
    """
    Advanced stacked-sill geometry modeling engine with comprehensive
    3D reconstruction, structural analysis, and temporal evolution capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the sill geometry modeling engine"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize database connection if available
        if DATABASE_AVAILABLE:
            self.db_integration = DatabaseIntegration()
            self.gis_ops = GISPostgreSQLOperations()
        else:
            self.db_integration = None
            self.gis_ops = None
        
        # Initialize analysis algorithms
        self._initialize_analysis_algorithms()
        
        # Cache for computed results
        self.geometry_cache = {}
        self.analysis_cache = {}
        
        # Load validation datasets
        self.validation_complexes = self._load_validation_complexes()
        
        self.logger.info("Stacked-Sill Geometry Engine initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters"""
        return {
            'reconstruction': {
                'min_sill_thickness': 0.5,  # meters
                'max_sill_thickness': 500.0,  # meters
                'min_lateral_extent': 0.1,  # km
                'max_lateral_extent': 100.0,  # km
                'resolution_levels': [1.0, 0.5, 0.25, 0.1],  # km
                'smoothing_factor': 0.2,
                'noise_threshold': 0.1
            },
            'structural_analysis': {
                'fracture_density_threshold': 0.1,  # fractures/m³
                'bedding_influence_factor': 0.8,
                'fault_control_distance': 2.0,  # km
                'anisotropy_threshold': 0.3
            },
            'interconnection': {
                'connection_search_radius': 5.0,  # km
                'flow_capacity_threshold': 0.01,  # m³/s
                'pressure_gradient_limit': 1e6,  # Pa/m
                'geometric_complexity_limit': 3.0
            },
            'temporal_evolution': {
                'growth_rate_limits': [1e-6, 1e-3],  # km³/year
                'cooling_rate_typical': 1e-4,  # K/year
                'crystallization_threshold': 0.5,  # fraction
                'prediction_uncertainty': 0.2
            },
            'validation': {
                'cross_validation_folds': 5,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95,
                'field_validation_weight': 0.7
            },
            'database': {
                'cache_results': True,
                'auto_save': True,
                'compression': 'gzip',
                'backup_frequency': 'daily'
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__ + ".StackedSillGeometryEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        return logger
    
    def _initialize_analysis_algorithms(self):
        """Initialize analysis algorithm components"""
        self.reconstruction_methods = {
            'seismic_structural': self._seismic_structural_reconstruction,
            'structural_only': self._structural_reconstruction,
            'seismic_only': self._seismic_reconstruction,
            'integrated': self._integrated_reconstruction
        }
        
        self.thickness_models = {
            'mechanical': self._mechanical_thickness_model,
            'thermal': self._thermal_thickness_model,
            'hybrid': self._hybrid_thickness_model
        }
        
        self.logger.info("Analysis algorithms initialized")
    
    def _load_validation_complexes(self) -> Dict[str, Any]:
        """Load validation datasets from well-studied sill complexes"""
        
        validation_complexes = {
            # Karoo Large Igneous Province - South Africa
            'karoo_basin': {
                'location': (24.0, -30.0),
                'complex_type': 'continental_flood_basalt',
                'total_volume': 2500.0,  # km³
                'sill_count': 150,
                'depth_range': (0.5, 8.0),  # km
                'age_range': (183, 177),  # Ma
                'host_rocks': ['sandstone', 'shale', 'mudstone'],
                'reference': 'Svensen et al. (2012), Galerne et al. (2008)'
            },
            
            # Ferrar Dolerite - Antarctica
            'ferrar_dolerite': {
                'location': (160.0, -77.0),
                'complex_type': 'continental_flood_basalt',
                'total_volume': 1800.0,  # km³
                'sill_count': 200,
                'depth_range': (0.2, 6.0),  # km
                'age_range': (184, 182),  # Ma
                'host_rocks': ['sandstone', 'coal_measures', 'shale'],
                'reference': 'Elliot & Fleming (2004), Ross et al. (2008)'
            },
            
            # Whin Sill - Northern England
            'whin_sill': {
                'location': (-2.5, 55.0),
                'complex_type': 'regional_sill_complex',
                'total_volume': 45.0,  # km³
                'sill_count': 25,
                'depth_range': (0.1, 2.5),  # km
                'age_range': (295, 295),  # Ma
                'host_rocks': ['limestone', 'sandstone', 'shale'],
                'reference': 'Liss et al. (2004), Woodcock & Strachan (2012)'
            }
        }
        
        return validation_complexes
    
    # ========================================================================
    # Core Public API Methods
    # ========================================================================
    
    async def reconstruct_sill_geometry_3d(
        self,
        seismic_data: SeismicStructuralData,
        structural_data: StructuralData,
        host_rock_props: HostRockProperties,
        method: str = 'integrated'
    ) -> List[SillGeometry3D]:
        """
        Reconstruct 3D sill geometries from seismic and structural data
        
        Args:
            seismic_data: Seismic reflection and attribute data
            structural_data: Structural geological measurements
            host_rock_props: Host rock mechanical properties
            method: Reconstruction method to use
            
        Returns:
            List of reconstructed sill geometries
        """
        try:
            self.logger.info(f"Starting 3D sill reconstruction using {method}")
            
            # Validate input data
            self._validate_sill_input_data(seismic_data, structural_data, host_rock_props)
            
            # Preprocess data for sill detection
            processed_data = await self._preprocess_sill_data(
                seismic_data, structural_data, host_rock_props
            )
            
            # Perform reconstruction based on selected method
            if method in self.reconstruction_methods:
                sill_geometries = await self.reconstruction_methods[method](processed_data)
            else:
                raise ValueError(f"Unknown reconstruction method: {method}")
            
            # Post-process and validate results
            validated_geometries = await self._validate_sill_geometries(
                sill_geometries, processed_data
            )
            
            # Calculate uncertainty bounds for each sill
            for geometry in validated_geometries:
                geometry.uncertainty_bounds = await self._calculate_sill_uncertainty(
                    geometry, processed_data
                )
            
            # Save results to database if available
            if self.db_integration and self.config['database']['auto_save']:
                await self._save_sill_geometries_to_database(validated_geometries)
            
            self.logger.info(f"Successfully reconstructed {len(validated_geometries)} sills")
            return validated_geometries
            
        except Exception as e:
            self.logger.error(f"Sill reconstruction failed: {str(e)}")
            raise
    
    async def model_thickness_variation(
        self,
        sill_geometry: SillGeometry3D,
        host_rock_props: HostRockProperties,
        method: str = 'hybrid'
    ) -> np.ndarray:
        """
        Model thickness variation based on host rock properties
        
        Args:
            sill_geometry: Base sill geometry
            host_rock_props: Host rock mechanical properties
            method: Thickness modeling method
            
        Returns:
            3D thickness distribution array
        """
        try:
            self.logger.info(f"Modeling thickness variation using {method} method")
            
            # Extract base parameters
            base_thickness = np.mean(sill_geometry.thickness_distribution)
            lateral_coords = self._extract_lateral_coordinates(sill_geometry)
            
            # Apply selected thickness model
            if method in self.thickness_models:
                thickness_variation = await self.thickness_models[method](
                    sill_geometry, host_rock_props, lateral_coords
                )
            else:
                raise ValueError(f"Unknown thickness modeling method: {method}")
            
            # Apply constraints and smoothing
            thickness_variation = self._apply_thickness_constraints(
                thickness_variation, base_thickness
            )
            
            # Validate against field observations
            validation_score = await self._validate_thickness_model(
                thickness_variation, sill_geometry
            )
            
            self.logger.info(f"Thickness modeling completed with validation score: {validation_score:.3f}")
            return thickness_variation
            
        except Exception as e:
            self.logger.error(f"Thickness modeling failed: {str(e)}")
            raise
    
    async def determine_lateral_extent(
        self,
        seismic_data: SeismicStructuralData,
        structural_constraints: StructuralData,
        sill_center: Tuple[float, float, float]
    ) -> Dict[str, float]:
        """
        Determine lateral extent using geophysical constraints
        
        Args:
            seismic_data: Seismic data for extent mapping
            structural_constraints: Structural controls on extent
            sill_center: Sill center coordinates
            
        Returns:
            Dictionary with extent parameters
        """
        try:
            self.logger.info("Determining sill lateral extent from geophysical data")
            
            # Extract seismic attributes around sill center
            local_attributes = self._extract_local_seismic_attributes(
                seismic_data, sill_center
            )
            
            # Apply edge detection algorithms
            sill_boundaries = await self._detect_sill_boundaries(
                local_attributes, structural_constraints
            )
            
            # Calculate extent parameters
            extent_params = self._calculate_extent_parameters(sill_boundaries)
            
            # Apply structural controls
            controlled_extent = self._apply_structural_controls(
                extent_params, structural_constraints
            )
            
            # Validate against regional geology
            validation_score = await self._validate_lateral_extent(
                controlled_extent, structural_constraints
            )
            
            controlled_extent['validation_score'] = validation_score
            
            self.logger.info(f"Lateral extent determined with validation score: {validation_score:.3f}")
            return controlled_extent
            
        except Exception as e:
            self.logger.error(f"Lateral extent determination failed: {str(e)}")
            raise
    
    async def analyze_sill_interconnections(
        self,
        sill_geometries: List[SillGeometry3D],
        structural_data: StructuralData
    ) -> List[SillInterconnection]:
        """
        Analyze sill interconnection and feeding relationships
        
        Args:
            sill_geometries: List of individual sill geometries
            structural_data: Structural controls on connections
            
        Returns:
            List of sill interconnections
        """
        try:
            self.logger.info(f"Analyzing interconnections between {len(sill_geometries)} sills")
            
            interconnections = []
            
            # Analyze all pairwise connections
            for i, sill_a in enumerate(sill_geometries):
                for j, sill_b in enumerate(sill_geometries[i+1:], i+1):
                    
                    # Calculate geometric relationship
                    geometric_analysis = self._analyze_geometric_relationship(sill_a, sill_b)
                    
                    # Check for potential connection
                    if geometric_analysis['connection_potential'] > 0.3:
                        
                        # Detailed connection analysis
                        connection = await self._analyze_detailed_connection(
                            sill_a, sill_b, structural_data, geometric_analysis
                        )
                        
                        if connection:
                            interconnections.append(connection)
            
            # Analyze feeding relationships
            feeding_network = await self._analyze_feeding_relationships(
                interconnections, sill_geometries
            )
            
            # Update interconnections with feeding information
            for interconnection in interconnections:
                interconnection.connection_type = feeding_network.get(
                    interconnection.connection_id, 'lateral'
                )
            
            # Validate interconnection network
            network_validation = await self._validate_interconnection_network(
                interconnections, sill_geometries
            )
            
            self.logger.info(f"Identified {len(interconnections)} sill interconnections")
            return interconnections
            
        except Exception as e:
            self.logger.error(f"Interconnection analysis failed: {str(e)}")
            raise
    
    async def analyze_structural_controls(
        self,
        sill_geometry: SillGeometry3D,
        structural_data: StructuralData
    ) -> Dict[str, Any]:
        """
        Analyze structural controls on sill emplacement geometry
        
        Args:
            sill_geometry: Sill geometry to analyze
            structural_data: Structural geological data
            
        Returns:
            Dictionary of structural control analysis results
        """
        try:
            self.logger.info("Analyzing structural controls on sill emplacement")
            
            # Analyze bedding plane influence
            bedding_influence = self._analyze_bedding_influence(
                sill_geometry, structural_data.bedding_planes
            )
            
            # Analyze fracture network control
            fracture_control = self._analyze_fracture_control(
                sill_geometry, structural_data.fracture_networks
            )
            
            # Analyze fault system influence
            fault_influence = self._analyze_fault_influence(
                sill_geometry, structural_data.fault_systems
            )
            
            # Analyze lithological controls
            lithological_control = self._analyze_lithological_control(
                sill_geometry, structural_data.lithological_contacts
            )
            
            # Integrate all structural controls
            integrated_controls = self._integrate_structural_controls(
                bedding_influence, fracture_control, fault_influence, lithological_control
            )
            
            return integrated_controls
            
        except Exception as e:
            self.logger.error(f"Structural control analysis failed: {str(e)}")
            raise
    
    async def model_multi_scale_sill_complex(
        self,
        individual_sills: List[SillGeometry3D],
        interconnections: List[SillInterconnection],
        regional_constraints: Dict[str, Any]
    ) -> SillComplex:
        """
        Model multi-scale sill complex from individual sills to regional complexes
        
        Args:
            individual_sills: List of individual sill geometries
            interconnections: Sill interconnection relationships
            regional_constraints: Regional geological constraints
            
        Returns:
            Complete sill complex model
        """
        try:
            self.logger.info(f"Modeling multi-scale sill complex with {len(individual_sills)} sills")
            
            # Calculate complex-wide properties
            total_volume = sum(sill.volume for sill in individual_sills)
            stratigraphic_levels = sorted(list(set(sill.emplacement_depth for sill in individual_sills)))
            
            # Determine emplacement sequence based on depth and interconnections
            emplacement_sequence = await self._determine_emplacement_sequence(
                individual_sills, interconnections
            )
            
            # Analyze structural controls at complex scale
            complex_structural_controls = await self._analyze_complex_structural_controls(
                individual_sills, regional_constraints
            )
            
            # Model thermal evolution of the complex
            thermal_evolution = await self._model_complex_thermal_evolution(
                individual_sills, interconnections
            )
            
            # Model mechanical evolution and stress field changes
            mechanical_evolution = await self._model_complex_mechanical_evolution(
                individual_sills, interconnections
            )
            
            # Assess hazards associated with the sill complex
            hazard_assessment = await self._assess_complex_hazards(
                individual_sills, interconnections
            )
            
            # Perform uncertainty analysis for the entire complex
            uncertainty_analysis = await self._analyze_complex_uncertainty(
                individual_sills, interconnections
            )
            
            # Create sill complex object
            sill_complex = SillComplex(
                complex_id=f"complex_{np.random.randint(1000, 9999)}",
                individual_sills=individual_sills,
                interconnections=interconnections,
                complex_geometry={
                    'total_extent': self._calculate_complex_extent(individual_sills),
                    'depth_distribution': stratigraphic_levels,
                    'geometric_complexity': self._calculate_geometric_complexity(individual_sills)
                },
                stratigraphic_levels=stratigraphic_levels,
                total_volume=total_volume,
                emplacement_sequence=emplacement_sequence,
                structural_controls=complex_structural_controls,
                thermal_evolution=thermal_evolution,
                mechanical_evolution=mechanical_evolution,
                hazard_assessment=hazard_assessment,
                uncertainty_analysis=uncertainty_analysis
            )
            
            # Save complex to database if available
            if self.db_integration and self.config['database']['auto_save']:
                await self._save_sill_complex_to_database(sill_complex)
            
            self.logger.info(f"Successfully modeled sill complex with total volume: {total_volume:.2f} km³")
            return sill_complex
            
        except Exception as e:
            self.logger.error(f"Multi-scale sill complex modeling failed: {str(e)}")
            raise
    
    async def validate_against_field_observations(
        self,
        sill_geometries: List[SillGeometry3D],
        field_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate sill geometries against exposed sill complexes and field measurements
        
        Args:
            sill_geometries: Reconstructed sill geometries
            field_data: Field observation data
            
        Returns:
            Validation scores and metrics
        """
        try:
            self.logger.info(f"Validating {len(sill_geometries)} sill geometries against field observations")
            
            validation_results = {}
            
            # Validate against known sill complexes
            for complex_name, complex_data in self.validation_complexes.items():
                if complex_name in field_data:
                    validation_score = await self._validate_against_known_complex(
                        sill_geometries, complex_data, field_data[complex_name]
                    )
                    validation_results[f'{complex_name}_validation'] = validation_score
            
            # Validate geometric parameters
            geometric_validation = await self._validate_geometric_parameters(
                sill_geometries, field_data
            )
            validation_results['geometric_validation'] = geometric_validation
            
            # Validate thickness distributions
            thickness_validation = await self._validate_thickness_distributions(
                sill_geometries, field_data
            )
            validation_results['thickness_validation'] = thickness_validation
            
            # Validate interconnection patterns
            if 'interconnection_data' in field_data:
                interconnection_validation = await self._validate_interconnection_patterns(
                    sill_geometries, field_data['interconnection_data']
                )
                validation_results['interconnection_validation'] = interconnection_validation
            
            # Calculate overall validation score
            overall_score = np.mean(list(validation_results.values()))
            validation_results['overall_validation'] = overall_score
            
            self.logger.info(f"Field validation completed with overall score: {overall_score:.3f}")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Field validation failed: {str(e)}")
            raise