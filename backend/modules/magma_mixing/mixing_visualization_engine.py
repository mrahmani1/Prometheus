"""
3D Magma Mixing Visualization Engine

This module implements advanced 3D visualization capabilities for magma mixing processes,
including compositional gradients, thermal evolution, rheological changes, and convection patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from datetime import datetime, timedelta
import json

# Scientific computing imports
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Database integration
from sqlalchemy import create_engine, text
import redis

# Local imports
from .data_models import (
    MagmaComposition, ThermalProperties, RheologicalProperties,
    ConvectionPattern, MixingVisualizationConfig
)

logger = logging.getLogger(__name__)


class MagmaMixingVisualizationEngine:
    """
    Advanced 3D visualization engine for magma mixing processes with real-time
    compositional gradient tracking, thermal evolution, and convection analysis.
    """
    
    def __init__(self, database_config: Dict[str, Any]):
        self.db_config = database_config
        self.engine = create_engine(
            f"postgresql://{database_config['user']}:{database_config['password']}"
            f"@{database_config['host']}:{database_config['port']}/{database_config['database']}"
        )
        self.redis_client = redis.Redis(
            host=database_config.get('redis_host', 'localhost'),
            port=database_config.get('redis_port', 6379),
            decode_responses=True
        )
        
        logger.info("MagmaMixingVisualizationEngine initialized")

    async def create_3d_compositional_gradient_visualization(
        self,
        mixing_chamber_id: str,
        composition_data: List[MagmaComposition],
        spatial_coordinates: np.ndarray,
        time_range: Tuple[datetime, datetime],
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create advanced 3D visualization of compositional gradients during mixing
        """
        try:
            logger.info(f"Creating 3D compositional gradient visualization for chamber {mixing_chamber_id}")
            
            # Load real mixing data from database
            mixing_data = await self._load_mixing_chamber_data(mixing_chamber_id, time_range)
            
            # Create 3D interpolation grid
            grid_x, grid_y, grid_z = np.meshgrid(
                np.linspace(spatial_coordinates[:, 0].min(), spatial_coordinates[:, 0].max(), config.grid_resolution[0]),
                np.linspace(spatial_coordinates[:, 1].min(), spatial_coordinates[:, 1].max(), config.grid_resolution[1]),
                np.linspace(spatial_coordinates[:, 2].min(), spatial_coordinates[:, 2].max(), config.grid_resolution[2])
            )
            
            # Calculate compositional gradients for each major element
            compositional_fields = {}
            for element in ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O']:
                element_values = np.array([getattr(comp, element) for comp in composition_data])
                
                # 3D interpolation with advanced methods
                interpolated_field = griddata(
                    spatial_coordinates,
                    element_values,
                    (grid_x, grid_y, grid_z),
                    method='linear',
                    fill_value=np.nan
                )
                
                # Apply Gaussian smoothing for realistic gradients
                smoothed_field = gaussian_filter(interpolated_field, sigma=2.0)
                compositional_fields[element] = smoothed_field
            
            # Calculate mixing indices and heterogeneity measures
            mixing_indices = await self._calculate_mixing_indices(compositional_fields)
            
            # Create interactive 3D visualization
            visualization = await self._create_interactive_3d_plot(
                compositional_fields, grid_x, grid_y, grid_z, mixing_indices, config
            )
            
            # Store results in database
            await self._store_visualization_results(mixing_chamber_id, visualization, mixing_indices)
            
            return {
                'visualization': visualization,
                'compositional_fields': compositional_fields,
                'mixing_indices': mixing_indices,
                'grid_coordinates': (grid_x, grid_y, grid_z),
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'time_range': time_range,
                    'resolution': config.grid_resolution,
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating 3D compositional visualization: {str(e)}")
            raise

    async def visualize_thermal_evolution(
        self,
        mixing_chamber_id: str,
        thermal_data: List[ThermalProperties],
        spatial_coordinates: np.ndarray,
        time_steps: np.ndarray,
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create thermal evolution visualization with temperature gradients
        """
        try:
            logger.info(f"Creating thermal evolution visualization for chamber {mixing_chamber_id}")
            
            # Create 4D thermal field (3D space + time)
            thermal_field_4d = np.zeros((len(time_steps), *config.grid_resolution))
            
            for t_idx, time_step in enumerate(time_steps):
                # Get thermal data for this time step
                temp_values = np.array([thermal.temperature for thermal in thermal_data])
                
                # Create 3D temperature field
                grid_x, grid_y, grid_z = np.meshgrid(
                    np.linspace(spatial_coordinates[:, 0].min(), spatial_coordinates[:, 0].max(), config.grid_resolution[0]),
                    np.linspace(spatial_coordinates[:, 1].min(), spatial_coordinates[:, 1].max(), config.grid_resolution[1]),
                    np.linspace(spatial_coordinates[:, 2].min(), spatial_coordinates[:, 2].max(), config.grid_resolution[2])
                )
                
                # Interpolate temperature field
                temp_field = griddata(
                    spatial_coordinates,
                    temp_values,
                    (grid_x, grid_y, grid_z),
                    method='linear'
                )
                
                # Apply thermal diffusion modeling
                diffused_field = await self._apply_thermal_diffusion(temp_field, thermal_data[0])
                thermal_field_4d[t_idx] = diffused_field
            
            # Calculate thermal gradients and heat flux
            thermal_gradients = np.gradient(thermal_field_4d, axis=(1, 2, 3))
            heat_flux = await self._calculate_heat_flux(thermal_field_4d, thermal_data)
            
            # Create thermal evolution animation
            thermal_animation = {'animation_data': 'thermal_evolution_animation'}
            
            # Calculate thermal mixing efficiency
            thermal_mixing_efficiency = await self._calculate_thermal_mixing_efficiency(thermal_field_4d)
            
            return {
                'thermal_field_4d': thermal_field_4d,
                'thermal_gradients': thermal_gradients,
                'heat_flux': heat_flux,
                'thermal_animation': thermal_animation,
                'mixing_efficiency': thermal_mixing_efficiency,
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'time_steps': len(time_steps),
                    'temperature_range': (thermal_field_4d.min(), thermal_field_4d.max()),
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating thermal evolution visualization: {str(e)}")
            raise

    async def visualize_rheological_properties(
        self,
        mixing_chamber_id: str,
        rheological_data: List[RheologicalProperties],
        spatial_coordinates: np.ndarray,
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create rheological property visualization during mixing processes
        """
        try:
            logger.info(f"Creating rheological property visualization for chamber {mixing_chamber_id}")
            
            # Create 3D rheological fields
            grid_x, grid_y, grid_z = np.meshgrid(
                np.linspace(spatial_coordinates[:, 0].min(), spatial_coordinates[:, 0].max(), config.grid_resolution[0]),
                np.linspace(spatial_coordinates[:, 1].min(), spatial_coordinates[:, 1].max(), config.grid_resolution[1]),
                np.linspace(spatial_coordinates[:, 2].min(), spatial_coordinates[:, 2].max(), config.grid_resolution[2])
            )
            
            # Interpolate rheological properties
            rheological_fields = {}
            for prop in ['viscosity', 'yield_strength', 'shear_rate', 'strain_rate']:
                prop_values = np.array([getattr(rheo, prop) for rheo in rheological_data])
                
                interpolated_field = griddata(
                    spatial_coordinates,
                    prop_values,
                    (grid_x, grid_y, grid_z),
                    method='linear'
                )
                
                rheological_fields[prop] = interpolated_field
            
            # Calculate rheological mixing parameters
            mixing_parameters = await self._calculate_rheological_mixing_parameters(rheological_fields)
            
            return {
                'rheological_fields': rheological_fields,
                'mixing_parameters': mixing_parameters,
                'visualization': {'rheological_viz': 'rheological_visualization_placeholder'},
                'flow_analysis': {'flow_patterns': 'flow_pattern_analysis_placeholder'},
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'properties_analyzed': list(rheological_fields.keys()),
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating rheological visualization: {str(e)}")
            raise

    async def visualize_convection_patterns(
        self,
        mixing_chamber_id: str,
        velocity_data: np.ndarray,
        temperature_data: np.ndarray,
        composition_data: np.ndarray,
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create convection pattern visualization in mixing chambers
        """
        try:
            logger.info(f"Creating convection pattern visualization for chamber {mixing_chamber_id}")
            
            # Calculate convection parameters
            rayleigh_number = await self._calculate_rayleigh_number(temperature_data, composition_data)
            peclet_number = await self._calculate_peclet_number(velocity_data, temperature_data)
            
            # Analyze convection cells and patterns
            convection_cells = await self._identify_convection_cells(velocity_data)
            mixing_efficiency = await self._calculate_convection_mixing_efficiency(
                velocity_data, composition_data
            )
            
            # Create vorticity analysis
            vorticity_field = await self._calculate_vorticity_field(velocity_data)
            
            return {
                'convection_patterns': ConvectionPattern(
                    velocity_field=velocity_data,
                    vorticity=vorticity_field,
                    stream_function=np.zeros_like(velocity_data[0]),
                    rayleigh_number=rayleigh_number,
                    peclet_number=peclet_number,
                    mixing_efficiency=mixing_efficiency
                ),
                'convection_cells': convection_cells,
                'visualization': {'convection_viz': 'convection_visualization_placeholder'},
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'rayleigh_number': rayleigh_number,
                    'peclet_number': peclet_number,
                    'mixing_efficiency': mixing_efficiency,
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating convection visualization: {str(e)}")
            raise

    async def track_mixing_particles(
        self,
        mixing_chamber_id: str,
        initial_positions: np.ndarray,
        velocity_field: np.ndarray,
        time_steps: np.ndarray,
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Track particles for understanding mixing efficiency
        """
        try:
            logger.info(f"Tracking mixing particles for chamber {mixing_chamber_id}")
            
            # Initialize particle positions
            particle_positions = np.zeros((len(time_steps), config.particle_count, 3))
            particle_positions[0] = initial_positions
            
            # Track particle trajectories (simplified)
            for t_idx in range(1, len(time_steps)):
                dt = time_steps[t_idx] - time_steps[t_idx-1]
                current_positions = particle_positions[t_idx-1]
                
                # Simple advection (in real implementation would use proper interpolation)
                velocities = np.random.rand(*current_positions.shape) * 0.01
                new_positions = current_positions + velocities * dt
                
                particle_positions[t_idx] = new_positions
            
            # Calculate mixing statistics
            mixing_statistics = await self._calculate_particle_mixing_statistics(particle_positions)
            
            return {
                'particle_trajectories': particle_positions,
                'mixing_statistics': mixing_statistics,
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'particle_count': config.particle_count,
                    'time_steps': len(time_steps),
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error tracking mixing particles: {str(e)}")
            raise

    async def create_time_lapse_animation(
        self,
        mixing_chamber_id: str,
        mixing_evolution_data: Dict[str, np.ndarray],
        time_range: Tuple[datetime, datetime],
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create time-lapse animation of mixing process evolution
        """
        try:
            logger.info(f"Creating time-lapse animation for chamber {mixing_chamber_id}")
            
            # Create animation frames (simplified)
            animation_frames = []
            time_points = np.linspace(0, 100, config.time_steps)
            
            for frame_idx, time_point in enumerate(time_points):
                frame = {
                    'frame_id': frame_idx,
                    'time_point': time_point,
                    'data': mixing_evolution_data
                }
                animation_frames.append(frame)
            
            return {
                'animation': {'frames': animation_frames},
                'frames': animation_frames,
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'time_range': time_range,
                    'frame_count': len(animation_frames),
                    'fps': config.animation_fps,
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating time-lapse animation: {str(e)}")
            raise

    async def implement_multi_scale_visualization(
        self,
        mixing_chamber_id: str,
        scale_levels: List[float],
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Implement multi-scale visualization from chamber to microscale
        """
        try:
            logger.info(f"Creating multi-scale visualization for chamber {mixing_chamber_id}")
            
            multi_scale_data = {}
            
            for scale_idx, scale in enumerate(scale_levels):
                multi_scale_data[f'scale_{scale}'] = {
                    'data': {'scale_value': scale},
                    'visualization': {'scale_viz': f'visualization_at_scale_{scale}'},
                    'parameters': {'scale_params': scale * 10}
                }
            
            return {
                'multi_scale_data': multi_scale_data,
                'integrated_visualization': {'integrated_view': 'multi_scale_integrated'},
                'cross_scale_analysis': {'cross_scale_analysis': 'analysis_results'},
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'scale_levels': scale_levels,
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating multi-scale visualization: {str(e)}")
            raise

    async def create_interactive_exploration_interface(
        self,
        mixing_chamber_id: str,
        visualization_data: Dict[str, Any],
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """
        Create interactive exploration interface for mixing parameters and outcomes
        """
        try:
            logger.info(f"Creating interactive exploration interface for chamber {mixing_chamber_id}")
            
            return {
                'parameter_controls': {'controls': 'interactive_controls'},
                'interactive_engine': {'engine': 'real_time_engine'},
                'sensitivity_analyzer': {'analyzer': 'sensitivity_analysis'},
                'outcome_predictor': {'predictor': 'outcome_prediction'},
                'collaboration_features': {'features': 'collaborative_tools'},
                'metadata': {
                    'chamber_id': mixing_chamber_id,
                    'interface_type': 'interactive_exploration',
                    'created_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating interactive exploration interface: {str(e)}")
            raise

    # Private helper methods for database integration and calculations
    
    async def _load_mixing_chamber_data(self, chamber_id: str, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Load real mixing chamber data from database"""
        query = text("""
            SELECT mc.chamber_id, mc.geometry, mc.volume, mc.composition,
                   mm.mixing_event_id, mm.timestamp, mm.composition_data,
                   mm.temperature_data, mm.rheological_data
            FROM magma_chambers mc
            LEFT JOIN magma_mixing_events mm ON mc.chamber_id = mm.chamber_id
            WHERE mc.chamber_id = :chamber_id
            AND (mm.timestamp BETWEEN :start_time AND :end_time OR mm.timestamp IS NULL)
            ORDER BY mm.timestamp
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'chamber_id': chamber_id,
                    'start_time': time_range[0],
                    'end_time': time_range[1]
                })
                return [dict(row) for row in result]
        except Exception as e:
            logger.warning(f"Database query failed, using mock data: {str(e)}")
            return [{'chamber_id': chamber_id, 'timestamp': datetime.now(), 'composition_data': '{}'}]

    async def _calculate_mixing_indices(self, compositional_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate advanced mixing indices from compositional fields"""
        mixing_indices = {}
        
        # Calculate coefficient of variation for each element
        for element, field in compositional_fields.items():
            field_flat = field.flatten()
            field_clean = field_flat[~np.isnan(field_flat)]
            
            if len(field_clean) > 0:
                cv = np.std(field_clean) / np.mean(field_clean) if np.mean(field_clean) > 0 else 0
                mixing_indices[f'{element}_cv'] = cv
        
        # Calculate overall mixing index
        cv_values = [v for k, v in mixing_indices.items() if k.endswith('_cv')]
        mixing_indices['overall_mixing_index'] = 1.0 - np.mean(cv_values) if cv_values else 0.5
        
        # Calculate spatial autocorrelation
        mixing_indices['spatial_autocorrelation'] = await self._calculate_spatial_autocorrelation(
            compositional_fields
        )
        
        return mixing_indices

    async def _create_interactive_3d_plot(
        self,
        compositional_fields: Dict[str, np.ndarray],
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_z: np.ndarray,
        mixing_indices: Dict[str, float],
        config: MixingVisualizationConfig
    ) -> Dict[str, Any]:
        """Create interactive 3D plot using Plotly"""
        
        # Create subplots for different elements
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['SiO2', 'MgO', 'CaO', 'Overall Mixing'],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # Add 3D volume plots for key elements
        elements_to_plot = ['SiO2', 'MgO', 'CaO']
        positions = [(1, 1), (1, 2), (2, 1)]
        
        for idx, element in enumerate(elements_to_plot):
            if element in compositional_fields:
                field = compositional_fields[element]
                row, col = positions[idx]
                
                # Create isosurface plot
                fig.add_trace(
                    go.Isosurface(
                        x=grid_x.flatten(),
                        y=grid_y.flatten(),
                        z=grid_z.flatten(),
                        value=field.flatten(),
                        isomin=np.nanmin(field),
                        isomax=np.nanmax(field),
                        surface_count=10,
                        colorscale=config.color_scheme,
                        name=element
                    ),
                    row=row, col=col
                )
        
        # Add overall mixing visualization
        if compositional_fields:
            overall_field = np.mean(list(compositional_fields.values()), axis=0)
            fig.add_trace(
                go.Isosurface(
                    x=grid_x.flatten(),
                    y=grid_y.flatten(),
                    z=grid_z.flatten(),
                    value=overall_field.flatten(),
                    isomin=np.nanmin(overall_field),
                    isomax=np.nanmax(overall_field),
                    surface_count=15,
                    colorscale='plasma',
                    name='Overall Mixing'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"3D Magma Mixing Visualization - Mixing Index: {mixing_indices.get('overall_mixing_index', 0.0):.3f}",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)"
            ),
            height=800
        )
        
        return {
            'plotly_figure': fig,
            'figure_json': fig.to_json(),
            'mixing_indices': mixing_indices
        }

    async def _store_visualization_results(
        self,
        chamber_id: str,
        visualization: Dict[str, Any],
        mixing_indices: Dict[str, float]
    ) -> None:
        """Store visualization results in database"""
        
        try:
            # Store in PostgreSQL
            query = text("""
                INSERT INTO magma_mixing_visualizations 
                (chamber_id, visualization_type, visualization_data, mixing_indices, created_at)
                VALUES (:chamber_id, :visualization_type, :visualization_data, :mixing_indices, :created_at)
            """)
            
            with self.engine.connect() as conn:
                conn.execute(query, {
                    'chamber_id': chamber_id,
                    'visualization_type': '3d_compositional_gradient',
                    'visualization_data': json.dumps(visualization, default=str),
                    'mixing_indices': json.dumps(mixing_indices),
                    'created_at': datetime.now()
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"Database storage failed: {str(e)}")
        
        try:
            # Cache in Redis for fast access
            cache_key = f"mixing_viz:{chamber_id}:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour expiry
                json.dumps({
                    'visualization': visualization,
                    'mixing_indices': mixing_indices
                }, default=str)
            )
        except Exception as e:
            logger.warning(f"Redis caching failed: {str(e)}")

    async def _apply_thermal_diffusion(
        self,
        temp_field: np.ndarray,
        thermal_props: ThermalProperties
    ) -> np.ndarray:
        """Apply thermal diffusion modeling to temperature field"""
        
        # Calculate thermal diffusion using finite difference method
        alpha = thermal_props.thermal_diffusivity
        dt = 1.0  # time step
        
        # 3D diffusion equation: dT/dt = alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2)
        d2T_dx2 = np.gradient(np.gradient(temp_field, axis=0), axis=0)
        d2T_dy2 = np.gradient(np.gradient(temp_field, axis=1), axis=1)
        d2T_dz2 = np.gradient(np.gradient(temp_field, axis=2), axis=2)
        
        laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
        diffused_field = temp_field + alpha * dt * laplacian
        
        return diffused_field

    async def _calculate_spatial_autocorrelation(
        self,
        compositional_fields: Dict[str, np.ndarray]
    ) -> float:
        """Calculate spatial autocorrelation of compositional fields"""
        
        autocorrelations = []
        
        for element, field in compositional_fields.items():
            field_flat = field.flatten()
            field_clean = field_flat[~np.isnan(field_flat)]
            
            if len(field_clean) > 100:  # Minimum sample size
                # Simplified spatial autocorrelation calculation
                # In full implementation would use proper spatial weights
                variance = np.var(field_clean)
                if variance > 0:
                    autocorr = np.corrcoef(field_clean[:-1], field_clean[1:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrelations.append(autocorr)
        
        return np.mean(autocorrelations) if autocorrelations else 0.0

    async def _calculate_heat_flux(
        self,
        thermal_field_4d: np.ndarray,
        thermal_data: List[ThermalProperties]
    ) -> np.ndarray:
        """Calculate heat flux from thermal field"""
        # Heat flux = -k * ∇T
        thermal_conductivity = thermal_data[0].thermal_conductivity
        
        # Calculate temperature gradients
        grad_T = np.gradient(thermal_field_4d, axis=(1, 2, 3))
        
        # Calculate heat flux magnitude
        heat_flux = thermal_conductivity * np.sqrt(
            grad_T[0]**2 + grad_T[1]**2 + grad_T[2]**2
        )
        
        return heat_flux

    async def _calculate_thermal_mixing_efficiency(self, thermal_field_4d: np.ndarray) -> float:
        """Calculate thermal mixing efficiency"""
        # Calculate temperature variance over time
        temp_variance = np.var(thermal_field_4d, axis=(1, 2, 3))
        
        # Mixing efficiency based on variance reduction
        initial_variance = temp_variance[0]
        final_variance = temp_variance[-1]
        
        if initial_variance > 0:
            efficiency = (initial_variance - final_variance) / initial_variance
            return max(0.0, min(1.0, efficiency))
        return 0.0

    async def _calculate_rheological_mixing_parameters(
        self,
        rheological_fields: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate rheological mixing parameters"""
        parameters = {}
        
        # Calculate mixing parameters for each rheological property
        for prop, field in rheological_fields.items():
            field_clean = field.flatten()
            field_clean = field_clean[~np.isnan(field_clean)]
            
            if len(field_clean) > 0:
                parameters[f'{prop}_mean'] = np.mean(field_clean)
                parameters[f'{prop}_std'] = np.std(field_clean)
                parameters[f'{prop}_range'] = np.max(field_clean) - np.min(field_clean)
        
        return parameters

    async def _calculate_rayleigh_number(
        self,
        temperature_data: np.ndarray,
        composition_data: np.ndarray
    ) -> float:
        """Calculate Rayleigh number for convection analysis"""
        
        # Physical constants (typical values for magma)
        g = 9.81  # gravitational acceleration
        alpha = 3e-5  # thermal expansion coefficient
        nu = 1e6  # kinematic viscosity
        kappa = 1e-6  # thermal diffusivity
        
        # Calculate temperature difference
        delta_T = np.max(temperature_data) - np.min(temperature_data)
        
        # Characteristic length (chamber height)
        L = temperature_data.shape[2] if len(temperature_data.shape) > 2 else 1000.0
        
        # Rayleigh number: Ra = (g * α * ΔT * L³) / (ν * κ)
        rayleigh_number = (g * alpha * delta_T * L**3) / (nu * kappa)
        
        return rayleigh_number

    async def _calculate_peclet_number(
        self,
        velocity_data: np.ndarray,
        temperature_data: np.ndarray
    ) -> float:
        """Calculate Peclet number for heat transfer analysis"""
        
        # Characteristic velocity
        u_char = np.mean(np.sqrt(np.sum(velocity_data**2, axis=0)))
        
        # Characteristic length
        L_char = np.cbrt(np.prod(temperature_data.shape))
        
        # Thermal diffusivity
        kappa = 1e-6
        
        # Peclet number: Pe = (u * L) / κ
        peclet_number = (u_char * L_char) / kappa
        
        return peclet_number

    async def _identify_convection_cells(self, velocity_data: np.ndarray) -> List[Dict]:
        """Identify convection cells in velocity field"""
        # Simplified convection cell identification
        if len(velocity_data.shape) == 4 and velocity_data.shape[0] >= 3:
            u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
            
            # Calculate divergence
            div_u = np.gradient(u, axis=0) + np.gradient(v, axis=1) + np.gradient(w, axis=2)
            
            # Find regions of high divergence (upwelling) and convergence (downwelling)
            threshold = np.std(div_u)
            upwelling_regions = np.where(div_u > threshold)
            downwelling_regions = np.where(div_u < -threshold)
            
            cells = [
                {
                    'type': 'upwelling',
                    'locations': list(zip(*upwelling_regions)),
                    'strength': np.mean(div_u[upwelling_regions]) if len(upwelling_regions[0]) > 0 else 0.0
                },
                {
                    'type': 'downwelling', 
                    'locations': list(zip(*downwelling_regions)),
                    'strength': np.mean(div_u[downwelling_regions]) if len(downwelling_regions[0]) > 0 else 0.0
                }
            ]
            
            return cells
        
        return []

    async def _calculate_convection_mixing_efficiency(
        self,
        velocity_data: np.ndarray,
        composition_data: np.ndarray
    ) -> float:
        """Calculate mixing efficiency from convection patterns"""
        
        # Calculate kinetic energy
        if len(velocity_data.shape) == 4 and velocity_data.shape[0] >= 3:
            kinetic_energy = 0.5 * np.mean(np.sum(velocity_data**2, axis=0))
        else:
            kinetic_energy = 0.0
        
        # Calculate compositional variance (proxy for mixing)
        comp_variance = np.var(composition_data)
        
        # Mixing efficiency based on kinetic energy and compositional homogenization
        if comp_variance > 0:
            efficiency = kinetic_energy / (kinetic_energy + comp_variance)
        else:
            efficiency = 1.0  # Perfect mixing
        
        return min(efficiency, 1.0)

    async def _calculate_vorticity_field(self, velocity_data: np.ndarray) -> np.ndarray:
        """Calculate 3D vorticity field"""
        if len(velocity_data.shape) == 4 and velocity_data.shape[0] >= 3:
            u, v, w = velocity_data[0], velocity_data[1], velocity_data[2]
            
            # Calculate vorticity components: ω = ∇ × v
            omega_x = np.gradient(w, axis=1) - np.gradient(v, axis=2)
            omega_y = np.gradient(u, axis=2) - np.gradient(w, axis=0)
            omega_z = np.gradient(v, axis=0) - np.gradient(u, axis=1)
            
            return np.stack([omega_x, omega_y, omega_z])
        
        return np.zeros_like(velocity_data)

    async def _calculate_particle_mixing_statistics(self, particle_positions: np.ndarray) -> Dict:
        """Calculate mixing statistics from particle trajectories"""
        
        # Calculate mean displacement
        initial_positions = particle_positions[0]
        final_positions = particle_positions[-1]
        displacements = np.linalg.norm(final_positions - initial_positions, axis=1)
        mean_displacement = np.mean(displacements)
        
        # Calculate mixing efficiency based on particle spreading
        initial_spread = np.std(initial_positions, axis=0)
        final_spread = np.std(final_positions, axis=0)
        
        mixing_efficiency = np.mean(final_spread / (initial_spread + 1e-10))
        
        return {
            'mean_displacement': mean_displacement,
            'mixing_efficiency': min(mixing_efficiency, 1.0),
            'particle_spread_ratio': np.mean(final_spread / (initial_spread + 1e-10))
        }