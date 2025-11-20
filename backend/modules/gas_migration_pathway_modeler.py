"""
3D Gas Migration Pathway Modeling

This module implements advanced 3D gas migration pathway modeling through crustal fracture networks
for the Prometheus AI Platform's volatile degassing analysis system.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import psycopg2
from scipy import optimize, integrate, interpolate
from scipy.spatial import distance_matrix, KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Import existing database and core modules
try:
    from .database_integration import DatabaseManager
    from .geological_data_structures import VolcanicSystem, GeochemicalData
except ImportError:
    # Fallback for direct execution
    try:
        from database_integration import DatabaseManager
        from geological_data_structures import VolcanicSystem, GeochemicalData
    except ImportError:
        # Create minimal mock classes for testing
        class DatabaseManager:
            async def execute_query(self, query, params=None):
                return []
        
        class VolcanicSystem:
            def __init__(self):
                self.system_id = 'test'
                self.latitude = 0.0
                self.longitude = 0.0
        
        class GeochemicalData:
            pass

logger = logging.getLogger(__name__)

@dataclass
class GasMigrationPathway:
    """Represents a 3D gas migration pathway through crustal fractures"""
    pathway_id: str
    source_location: Tuple[float, float, float]  # (x, y, depth)
    surface_location: Tuple[float, float]  # (x, y)
    pathway_nodes: List[Tuple[float, float, float]]
    permeability_profile: np.ndarray
    pressure_gradient: np.ndarray
    flow_velocity: np.ndarray
    gas_composition_evolution: Dict[str, np.ndarray]
    structural_controls: List[str]
    efficiency_score: float
    uncertainty_bounds: Dict[str, Tuple[float, float]]

class GasMigrationPathwayModeler:
    """Advanced 3D gas migration pathway modeling through crustal fracture networks"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.fracture_network = None
        self.permeability_field = None
        self.pressure_field = None
        self.pathway_cache = {}
        
    async def initialize_fracture_network(self, volcanic_system: VolcanicSystem) -> nx.Graph:
        """Initialize 3D fracture network from geological and geophysical data"""
        try:
            # Query structural geology data from database
            structural_query = """
            SELECT fracture_id, x_coord, y_coord, depth, strike, dip, length, 
                   aperture, permeability, connectivity_index
            FROM structural_geology 
            WHERE volcanic_system_id = %s AND fracture_type IN ('fault', 'fracture', 'joint')
            """
            
            fracture_data = await self.db_manager.execute_query(
                structural_query, (volcanic_system.system_id,)
            )
            
            # Create 3D fracture network graph
            fracture_network = nx.Graph()
            
            for fracture in fracture_data:
                node_id = fracture['fracture_id']
                fracture_network.add_node(
                    node_id,
                    position=(fracture['x_coord'], fracture['y_coord'], fracture['depth']),
                    strike=fracture['strike'],
                    dip=fracture['dip'],
                    length=fracture['length'],
                    aperture=fracture['aperture'],
                    permeability=fracture['permeability'],
                    connectivity=fracture['connectivity_index']
                )
            
            # Calculate fracture connectivity based on spatial proximity and orientation
            positions = np.array([fracture_network.nodes[node]['position'] for node in fracture_network.nodes()])
            
            if len(positions) > 1:
                # Build KDTree for efficient spatial queries
                kdtree = KDTree(positions)
                
                for i, node1 in enumerate(fracture_network.nodes()):
                    # Find nearby fractures within connection distance
                    nearby_indices = kdtree.query_ball_point(positions[i], r=1000.0)  # 1km radius
                    
                    for j in nearby_indices:
                        if i != j:
                            node2 = list(fracture_network.nodes())[j]
                            
                            # Calculate connection probability based on geometric compatibility
                            connection_prob = self._calculate_fracture_connection_probability(
                                fracture_network.nodes[node1], fracture_network.nodes[node2]
                            )
                            
                            if connection_prob > 0.3:  # Threshold for connection
                                distance = np.linalg.norm(positions[i] - positions[j])
                                fracture_network.add_edge(
                                    node1, node2,
                                    distance=distance,
                                    connection_probability=connection_prob,
                                    flow_capacity=self._calculate_flow_capacity(
                                        fracture_network.nodes[node1], fracture_network.nodes[node2]
                                    )
                                )
            
            self.fracture_network = fracture_network
            logger.info(f"Initialized fracture network with {len(fracture_network.nodes())} nodes and {len(fracture_network.edges())} edges")
            
            return fracture_network
            
        except Exception as e:
            logger.error(f"Error initializing fracture network: {str(e)}")
            raise
    
    def _calculate_fracture_connection_probability(self, fracture1: Dict, fracture2: Dict) -> float:
        """Calculate probability of connection between two fractures based on geometry"""
        # Calculate angular difference between fracture orientations
        strike_diff = abs(fracture1['strike'] - fracture2['strike'])
        dip_diff = abs(fracture1['dip'] - fracture2['dip'])
        
        # Normalize angular differences
        strike_diff = min(strike_diff, 360 - strike_diff) / 180.0
        dip_diff = dip_diff / 90.0
        
        # Calculate geometric compatibility
        orientation_compatibility = 1.0 - 0.5 * (strike_diff + dip_diff)
        
        # Consider aperture and permeability compatibility
        aperture_ratio = min(fracture1['aperture'], fracture2['aperture']) / max(fracture1['aperture'], fracture2['aperture'])
        permeability_factor = np.sqrt(fracture1['permeability'] * fracture2['permeability']) / 1e-12  # Normalize
        
        # Combined connection probability
        connection_prob = orientation_compatibility * aperture_ratio * min(permeability_factor, 1.0)
        
        return max(0.0, min(1.0, connection_prob))
    
    def _calculate_flow_capacity(self, fracture1: Dict, fracture2: Dict) -> float:
        """Calculate flow capacity between connected fractures"""
        # Use harmonic mean of permeabilities
        k1, k2 = fracture1['permeability'], fracture2['permeability']
        harmonic_mean_k = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
        
        # Consider aperture effects
        min_aperture = min(fracture1['aperture'], fracture2['aperture'])
        
        # Flow capacity based on cubic law for fracture flow
        flow_capacity = harmonic_mean_k * min_aperture**3 / 12.0  # Cubic law coefficient
        
        return flow_capacity
    
    async def model_permeability_field(self, volcanic_system: VolcanicSystem, 
                                     grid_resolution: Tuple[int, int, int] = (100, 100, 50)) -> np.ndarray:
        """Model 3D permeability field for gas migration"""
        try:
            # Query rock property data
            rock_query = """
            SELECT x_coord, y_coord, depth, rock_type, porosity, permeability, 
                   alteration_degree, fracture_density
            FROM rock_properties 
            WHERE volcanic_system_id = %s
            """
            
            rock_data = await self.db_manager.execute_query(
                rock_query, (volcanic_system.system_id,)
            )
            
            if not rock_data:
                logger.warning("No rock property data found, using default permeability field")
                return self._create_default_permeability_field(grid_resolution)
            
            # Create 3D grid
            x_coords = np.array([r['x_coord'] for r in rock_data])
            y_coords = np.array([r['y_coord'] for r in rock_data])
            depths = np.array([r['depth'] for r in rock_data])
            permeabilities = np.array([r['permeability'] for r in rock_data])
            
            # Define grid bounds
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            z_min, z_max = depths.min(), depths.max()
            
            # Create regular grid
            x_grid = np.linspace(x_min, x_max, grid_resolution[0])
            y_grid = np.linspace(y_min, y_max, grid_resolution[1])
            z_grid = np.linspace(z_min, z_max, grid_resolution[2])
            
            X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            
            # Interpolate permeability values to grid
            points = np.column_stack((x_coords, y_coords, depths))
            grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
            
            # Use RBF interpolation for smooth permeability field
            from scipy.interpolate import Rbf
            try:
                rbf = Rbf(x_coords, y_coords, depths, np.log10(permeabilities + 1e-20), 
                         function='multiquadric', smooth=0.1)
                
                log_permeability_grid = rbf(X, Y, Z)
                permeability_field = 10**log_permeability_grid
            except Exception as rbf_error:
                logger.warning(f"RBF interpolation failed: {rbf_error}, using nearest neighbor")
                # Fallback to nearest neighbor interpolation
                from scipy.spatial.distance import cdist
                
                permeability_field = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        for k in range(X.shape[2]):
                            grid_point = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                            distances = cdist([grid_point], points)[0]
                            nearest_idx = np.argmin(distances)
                            permeability_field[i,j,k] = permeabilities[nearest_idx]
            
            # Apply geological constraints
            permeability_field = self._apply_geological_constraints(
                permeability_field, X, Y, Z, rock_data
            )
            
            self.permeability_field = permeability_field
            logger.info(f"Created permeability field with shape {permeability_field.shape}")
            
            return permeability_field
            
        except Exception as e:
            logger.error(f"Error modeling permeability field: {str(e)}")
            # Ensure we always set the permeability field
            self.permeability_field = self._create_default_permeability_field(grid_resolution)
            return self.permeability_field
    
    def _create_default_permeability_field(self, grid_resolution: Tuple[int, int, int]) -> np.ndarray:
        """Create default permeability field when no data is available"""
        # Create depth-dependent permeability field
        permeability_field = np.zeros(grid_resolution)
        
        for k in range(grid_resolution[2]):
            depth_factor = k / grid_resolution[2]  # Normalized depth
            # Decrease permeability with depth (typical geological trend)
            base_permeability = 1e-12 * np.exp(-depth_factor * 2)  # m²
            
            # Add spatial heterogeneity
            noise = np.random.lognormal(0, 0.5, (grid_resolution[0], grid_resolution[1]))
            permeability_field[:, :, k] = base_permeability * noise
        
        return permeability_field
    
    def _apply_geological_constraints(self, permeability_field: np.ndarray, 
                                    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                    rock_data: List[Dict]) -> np.ndarray:
        """Apply geological constraints to permeability field"""
        # Enhance permeability near fractures and faults
        if self.fracture_network:
            for node in self.fracture_network.nodes():
                node_pos = self.fracture_network.nodes[node]['position']
                node_perm = self.fracture_network.nodes[node]['permeability']
                
                # Find grid points near fracture
                distances = np.sqrt((X - node_pos[0])**2 + (Y - node_pos[1])**2 + (Z - node_pos[2])**2)
                influence_radius = 100.0  # meters
                
                # Apply fracture influence
                influence_mask = distances < influence_radius
                influence_factor = np.exp(-distances[influence_mask] / (influence_radius / 3))
                permeability_field[influence_mask] = np.maximum(
                    permeability_field[influence_mask],
                    node_perm * influence_factor
                )
        
        # Apply depth-dependent scaling
        for k in range(permeability_field.shape[2]):
            depth_factor = 1.0 - k / permeability_field.shape[2]  # Surface = 1, deep = 0
            permeability_field[:, :, k] *= (0.1 + 0.9 * depth_factor)
        
        return permeability_field    

    async def simulate_pressure_driven_flow(self, source_locations: List[Tuple[float, float, float]],
                                          source_pressures: List[float]) -> Dict[str, np.ndarray]:
        """Simulate pressure-driven gas flow from deep sources"""
        try:
            if self.permeability_field is None:
                logger.warning("Permeability field not initialized, creating default field")
                self.permeability_field = self._create_default_permeability_field((100, 100, 50))
            
            # Set up finite difference grid for pressure equation
            nx, ny, nz = self.permeability_field.shape
            dx = dy = dz = 100.0  # Grid spacing in meters
            
            # Initialize pressure field
            pressure_field = np.zeros((nx, ny, nz))
            
            # Set boundary conditions
            # Surface pressure (atmospheric)
            pressure_field[:, :, 0] = 101325.0  # Pa
            
            # Source pressures at depth
            for i, (source_loc, source_pressure) in enumerate(zip(source_locations, source_pressures)):
                # Find nearest grid point
                ix = int(source_loc[0] / dx)
                iy = int(source_loc[1] / dy)
                iz = int(source_loc[2] / dz)
                
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    pressure_field[ix, iy, iz] = source_pressure
            
            # Solve steady-state pressure equation using finite differences
            pressure_field = self._solve_pressure_equation(pressure_field, self.permeability_field, dx, dy, dz)
            
            # Calculate flow velocities using Darcy's law
            velocity_field = self._calculate_velocity_field(pressure_field, self.permeability_field, dx, dy, dz)
            
            # Calculate flow pathways using streamline tracing
            pathways = await self._trace_streamlines(velocity_field, source_locations, dx, dy, dz)
            
            self.pressure_field = pressure_field
            
            return {
                'pressure_field': pressure_field,
                'velocity_field': velocity_field,
                'pathways': pathways
            }
            
        except Exception as e:
            logger.error(f"Error simulating pressure-driven flow: {str(e)}")
            raise
    
    def _solve_pressure_equation(self, pressure_field: np.ndarray, permeability_field: np.ndarray,
                                dx: float, dy: float, dz: float, max_iterations: int = 100) -> np.ndarray:
        """Solve steady-state pressure equation using iterative method"""
        nx, ny, nz = pressure_field.shape
        pressure_new = pressure_field.copy()
        
        # Gauss-Seidel iteration
        for iteration in range(max_iterations):
            pressure_old = pressure_new.copy()
            
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    for k in range(1, nz-1):
                        # Skip source and boundary points
                        if pressure_field[i, j, k] > 101325.0:  # Source point
                            continue
                        
                        # Calculate permeability at cell faces (harmonic mean)
                        k_x_plus = 2 * permeability_field[i, j, k] * permeability_field[i+1, j, k] / \
                                  (permeability_field[i, j, k] + permeability_field[i+1, j, k])
                        k_x_minus = 2 * permeability_field[i, j, k] * permeability_field[i-1, j, k] / \
                                   (permeability_field[i, j, k] + permeability_field[i-1, j, k])
                        k_y_plus = 2 * permeability_field[i, j, k] * permeability_field[i, j+1, k] / \
                                  (permeability_field[i, j, k] + permeability_field[i, j+1, k])
                        k_y_minus = 2 * permeability_field[i, j, k] * permeability_field[i, j-1, k] / \
                                   (permeability_field[i, j, k] + permeability_field[i, j-1, k])
                        k_z_plus = 2 * permeability_field[i, j, k] * permeability_field[i, j, k+1] / \
                                  (permeability_field[i, j, k] + permeability_field[i, j, k+1])
                        k_z_minus = 2 * permeability_field[i, j, k] * permeability_field[i, j, k-1] / \
                                   (permeability_field[i, j, k] + permeability_field[i, j, k-1])
                        
                        # Finite difference approximation
                        numerator = (k_x_plus * pressure_new[i+1, j, k] + k_x_minus * pressure_new[i-1, j, k]) / dx**2 + \
                                   (k_y_plus * pressure_new[i, j+1, k] + k_y_minus * pressure_new[i, j-1, k]) / dy**2 + \
                                   (k_z_plus * pressure_new[i, j, k+1] + k_z_minus * pressure_new[i, j, k-1]) / dz**2
                        
                        denominator = (k_x_plus + k_x_minus) / dx**2 + (k_y_plus + k_y_minus) / dy**2 + \
                                     (k_z_plus + k_z_minus) / dz**2
                        
                        if denominator > 0:
                            pressure_new[i, j, k] = numerator / denominator
            
            # Check convergence
            max_change = np.max(np.abs(pressure_new - pressure_old))
            if max_change < 100.0:  # 100 Pa tolerance (relaxed for faster convergence)
                logger.info(f"Pressure equation converged after {iteration+1} iterations")
                break
        
        return pressure_new
    
    def _calculate_velocity_field(self, pressure_field: np.ndarray, permeability_field: np.ndarray,
                                 dx: float, dy: float, dz: float) -> Dict[str, np.ndarray]:
        """Calculate velocity field using Darcy's law"""
        nx, ny, nz = pressure_field.shape
        
        # Initialize velocity components
        vx = np.zeros((nx-1, ny, nz))
        vy = np.zeros((nx, ny-1, nz))
        vz = np.zeros((nx, ny, nz-1))
        
        # Gas properties (assume CO2 at volcanic conditions)
        gas_viscosity = 2e-5  # Pa·s
        gas_density = 2.0     # kg/m³
        
        # Calculate x-component of velocity
        for i in range(nx-1):
            for j in range(ny):
                for k in range(nz):
                    k_face = 2 * permeability_field[i, j, k] * permeability_field[i+1, j, k] / \
                            (permeability_field[i, j, k] + permeability_field[i+1, j, k])
                    dp_dx = (pressure_field[i+1, j, k] - pressure_field[i, j, k]) / dx
                    vx[i, j, k] = -k_face * dp_dx / gas_viscosity
        
        # Calculate y-component of velocity
        for i in range(nx):
            for j in range(ny-1):
                for k in range(nz):
                    k_face = 2 * permeability_field[i, j, k] * permeability_field[i, j+1, k] / \
                            (permeability_field[i, j, k] + permeability_field[i, j+1, k])
                    dp_dy = (pressure_field[i, j+1, k] - pressure_field[i, j, k]) / dy
                    vy[i, j, k] = -k_face * dp_dy / gas_viscosity
        
        # Calculate z-component of velocity (including buoyancy)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz-1):
                    k_face = 2 * permeability_field[i, j, k] * permeability_field[i, j, k+1] / \
                            (permeability_field[i, j, k] + permeability_field[i, j, k+1])
                    dp_dz = (pressure_field[i, j, k+1] - pressure_field[i, j, k]) / dz
                    # Include gravitational effects
                    gravity_term = gas_density * 9.81  # N/m³
                    vz[i, j, k] = -k_face * (dp_dz - gravity_term) / gas_viscosity
        
        return {'vx': vx, 'vy': vy, 'vz': vz}
    
    async def _trace_streamlines(self, velocity_field: Dict[str, np.ndarray], 
                          source_locations: List[Tuple[float, float, float]],
                          dx: float, dy: float, dz: float) -> List[GasMigrationPathway]:
        """Trace streamlines from source locations to surface"""
        pathways = []
        
        vx, vy, vz = velocity_field['vx'], velocity_field['vy'], velocity_field['vz']
        
        for i, source_loc in enumerate(source_locations):
            # Initialize streamline at source location
            x, y, z = source_loc
            pathway_nodes = [(x, y, z)]
            
            # Trace streamline using 4th-order Runge-Kutta
            dt = 10.0  # Time step in seconds
            max_steps = 10000
            
            for step in range(max_steps):
                # Get velocity at current position by interpolation
                velocity = self._interpolate_velocity(x, y, z, velocity_field, dx, dy, dz)
                
                if np.linalg.norm(velocity) < 1e-10:  # Stagnation point
                    break
                
                # Runge-Kutta 4th order integration
                k1 = dt * np.array(velocity)
                k2 = dt * np.array(self._interpolate_velocity(x + k1[0]/2, y + k1[1]/2, z + k1[2]/2, 
                                                            velocity_field, dx, dy, dz))
                k3 = dt * np.array(self._interpolate_velocity(x + k2[0]/2, y + k2[1]/2, z + k2[2]/2, 
                                                            velocity_field, dx, dy, dz))
                k4 = dt * np.array(self._interpolate_velocity(x + k3[0], y + k3[1], z + k3[2], 
                                                            velocity_field, dx, dy, dz))
                
                # Update position
                dx_dt = (k1 + 2*k2 + 2*k3 + k4) / 6
                x += dx_dt[0]
                y += dx_dt[1]
                z += dx_dt[2]
                
                pathway_nodes.append((x, y, z))
                
                # Check if reached surface
                if z <= 0:
                    break
                
                # Check bounds
                if not (0 <= x <= vx.shape[0]*dx and 0 <= y <= vy.shape[1]*dy):
                    break
            
            # Create pathway object
            if len(pathway_nodes) > 1:
                pathway = GasMigrationPathway(
                    pathway_id=f"pathway_{i}",
                    source_location=source_loc,
                    surface_location=(pathway_nodes[-1][0], pathway_nodes[-1][1]),
                    pathway_nodes=pathway_nodes,
                    permeability_profile=self._extract_permeability_profile(pathway_nodes),
                    pressure_gradient=self._calculate_pressure_gradient_along_path(pathway_nodes),
                    flow_velocity=self._calculate_velocity_along_path(pathway_nodes, velocity_field, dx, dy, dz),
                    gas_composition_evolution={},  # To be filled by composition modeling
                    structural_controls=await self._identify_structural_controls(pathway_nodes),
                    efficiency_score=self._calculate_pathway_efficiency(pathway_nodes, velocity_field, dx, dy, dz),
                    uncertainty_bounds={}  # To be filled by uncertainty analysis
                )
                pathways.append(pathway)
        
        return pathways
    
    def _interpolate_velocity(self, x: float, y: float, z: float, 
                             velocity_field: Dict[str, np.ndarray],
                             dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        """Interpolate velocity at arbitrary position"""
        vx, vy, vz = velocity_field['vx'], velocity_field['vy'], velocity_field['vz']
        
        # Convert to grid indices
        i = x / dx
        j = y / dy
        k = z / dz
        
        # Trilinear interpolation for each velocity component
        try:
            # X-velocity (defined at i+0.5, j, k)
            i_vx = max(0, min(vx.shape[0]-1, int(i)))
            j_vx = max(0, min(vx.shape[1]-1, int(j)))
            k_vx = max(0, min(vx.shape[2]-1, int(k)))
            vel_x = vx[i_vx, j_vx, k_vx]
            
            # Y-velocity (defined at i, j+0.5, k)
            i_vy = max(0, min(vy.shape[0]-1, int(i)))
            j_vy = max(0, min(vy.shape[1]-1, int(j)))
            k_vy = max(0, min(vy.shape[2]-1, int(k)))
            vel_y = vy[i_vy, j_vy, k_vy]
            
            # Z-velocity (defined at i, j, k+0.5)
            i_vz = max(0, min(vz.shape[0]-1, int(i)))
            j_vz = max(0, min(vz.shape[1]-1, int(j)))
            k_vz = max(0, min(vz.shape[2]-1, int(k)))
            vel_z = vz[i_vz, j_vz, k_vz]
            
            return (vel_x, vel_y, vel_z)
            
        except (IndexError, ValueError):
            return (0.0, 0.0, 0.0)
    
    def _extract_permeability_profile(self, pathway_nodes: List[Tuple[float, float, float]]) -> np.ndarray:
        """Extract permeability values along pathway"""
        if self.permeability_field is None:
            return np.ones(len(pathway_nodes)) * 1e-12
        
        permeabilities = []
        dx = dy = dz = 100.0  # Grid spacing
        
        for x, y, z in pathway_nodes:
            i = max(0, min(self.permeability_field.shape[0]-1, int(x / dx)))
            j = max(0, min(self.permeability_field.shape[1]-1, int(y / dy)))
            k = max(0, min(self.permeability_field.shape[2]-1, int(z / dz)))
            
            permeabilities.append(self.permeability_field[i, j, k])
        
        return np.array(permeabilities)
    
    def _calculate_pressure_gradient_along_path(self, pathway_nodes: List[Tuple[float, float, float]]) -> np.ndarray:
        """Calculate pressure gradient along pathway"""
        if self.pressure_field is None:
            return np.zeros(len(pathway_nodes))
        
        gradients = []
        dx = dy = dz = 100.0  # Grid spacing
        
        for i, (x, y, z) in enumerate(pathway_nodes):
            if i == 0 or i == len(pathway_nodes) - 1:
                gradients.append(0.0)
                continue
            
            # Calculate gradient magnitude using central differences
            prev_x, prev_y, prev_z = pathway_nodes[i-1]
            next_x, next_y, next_z = pathway_nodes[i+1]
            
            # Get pressures at neighboring points
            prev_pressure = self._get_pressure_at_point(prev_x, prev_y, prev_z, dx, dy, dz)
            next_pressure = self._get_pressure_at_point(next_x, next_y, next_z, dx, dy, dz)
            
            # Calculate distance
            distance = np.sqrt((next_x - prev_x)**2 + (next_y - prev_y)**2 + (next_z - prev_z)**2)
            
            if distance > 0:
                gradient = abs(next_pressure - prev_pressure) / distance
            else:
                gradient = 0.0
            
            gradients.append(gradient)
        
        return np.array(gradients)
    
    def _get_pressure_at_point(self, x: float, y: float, z: float, 
                              dx: float, dy: float, dz: float) -> float:
        """Get pressure at specific point"""
        if self.pressure_field is None:
            return 101325.0  # Atmospheric pressure
        
        i = max(0, min(self.pressure_field.shape[0]-1, int(x / dx)))
        j = max(0, min(self.pressure_field.shape[1]-1, int(y / dy)))
        k = max(0, min(self.pressure_field.shape[2]-1, int(z / dz)))
        
        return self.pressure_field[i, j, k]
    
    def _calculate_velocity_along_path(self, pathway_nodes: List[Tuple[float, float, float]],
                                     velocity_field: Dict[str, np.ndarray],
                                     dx: float, dy: float, dz: float) -> np.ndarray:
        """Calculate velocity magnitude along pathway"""
        velocities = []
        
        for x, y, z in pathway_nodes:
            velocity = self._interpolate_velocity(x, y, z, velocity_field, dx, dy, dz)
            velocity_magnitude = np.linalg.norm(velocity)
            velocities.append(velocity_magnitude)
        
        return np.array(velocities)
    
    async def _identify_structural_controls(self, pathway_nodes: List[Tuple[float, float, float]]) -> List[str]:
        """Identify structural controls along pathway using advanced spatial analysis"""
        controls = []
        
        if self.fracture_network is None:
            return controls
        
        # Advanced structural control identification using multiple criteria
        pathway_array = np.array(pathway_nodes)
        
        # Query structural geology database for comprehensive analysis
        try:
            # Get detailed structural features from database
            structural_query = """
            SELECT structure_id, structure_type, strike, dip, geometry, 
                   permeability_enhancement, stress_orientation, age_ma,
                   ST_AsText(geometry) as geometry_wkt
            FROM geological_structures 
            WHERE ST_DWithin(geometry, ST_MakeLine(ARRAY[%s]), %s)
            ORDER BY ST_Distance(geometry, ST_MakeLine(ARRAY[%s]))
            """
            
            # Create pathway linestring for spatial query
            pathway_wkt_points = [f"POINT({x} {y} {z})" for x, y, z in pathway_nodes]
            pathway_linestring = f"LINESTRING({', '.join([f'{x} {y} {z}' for x, y, z in pathway_nodes])})"
            
            # Execute spatial query with 200m buffer
            try:
                structural_features = await self.db_manager.execute_query(
                    structural_query, (pathway_linestring, 200.0, pathway_linestring)
                )
            except RuntimeError:
                # Fallback if already in event loop
                structural_features = []
            
            # Analyze each structural feature's influence
            for feature in structural_features:
                control_analysis = self._analyze_structural_control_influence(
                    feature, pathway_array
                )
                
                if control_analysis['influence_score'] > 0.3:  # Significant influence threshold
                    control_description = (
                        f"{feature['structure_type']}_{feature['structure_id']}_"
                        f"strike{feature['strike']:.0f}_dip{feature['dip']:.0f}_"
                        f"influence{control_analysis['influence_score']:.2f}"
                    )
                    controls.append(control_description)
            
            # Add fracture network analysis
            for node in self.fracture_network.nodes():
                fracture_data = self.fracture_network.nodes[node]
                fracture_pos = np.array(fracture_data['position'])
                
                # Calculate minimum distance to pathway
                distances = np.linalg.norm(pathway_array - fracture_pos, axis=1)
                min_distance = np.min(distances)
                closest_idx = np.argmin(distances)
                
                if min_distance < 100.0:  # Within 100m influence zone
                    # Calculate fracture influence based on orientation and permeability
                    influence_score = self._calculate_fracture_influence(
                        fracture_data, pathway_array[closest_idx], min_distance
                    )
                    
                    if influence_score > 0.4:
                        control_description = (
                            f"fracture_{node}_strike{fracture_data['strike']:.0f}_"
                            f"dip{fracture_data['dip']:.0f}_perm{fracture_data['permeability']:.2e}_"
                            f"dist{min_distance:.1f}m_influence{influence_score:.2f}"
                        )
                        controls.append(control_description)
            
            # Add lithological controls
            lithology_controls = await self._identify_lithological_controls(pathway_array)
            controls.extend(lithology_controls)
            
            # Add stress field controls
            stress_controls = await self._identify_stress_field_controls(pathway_array)
            controls.extend(stress_controls)
            
        except Exception as e:
            logger.warning(f"Advanced structural control analysis failed, using basic method: {e}")
            # Fallback to basic fracture proximity analysis
            for node in self.fracture_network.nodes():
                fracture_pos = self.fracture_network.nodes[node]['position']
                
                for pathway_pos in pathway_nodes:
                    distance = np.linalg.norm(np.array(pathway_pos) - np.array(fracture_pos))
                    
                    if distance < 50.0:  # Within 50m of fracture
                        controls.append(f"fracture_{node}")
                        break
        
        return list(set(controls))  # Remove duplicates
    
    def _calculate_pathway_efficiency(self, pathway_nodes: List[Tuple[float, float, float]],
                                    velocity_field: Dict[str, np.ndarray],
                                    dx: float, dy: float, dz: float) -> float:
        """Calculate pathway efficiency score"""
        if len(pathway_nodes) < 2:
            return 0.0
        
        # Calculate total path length
        total_length = 0.0
        for i in range(1, len(pathway_nodes)):
            segment_length = np.linalg.norm(
                np.array(pathway_nodes[i]) - np.array(pathway_nodes[i-1])
            )
            total_length += segment_length
        
        # Calculate direct distance
        direct_distance = np.linalg.norm(
            np.array(pathway_nodes[-1]) - np.array(pathway_nodes[0])
        )
        
        # Path efficiency (1 = straight line, 0 = very tortuous)
        path_efficiency = direct_distance / total_length if total_length > 0 else 0.0
        
        # Average velocity along path
        velocities = self._calculate_velocity_along_path(pathway_nodes, velocity_field, dx, dy, dz)
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0.0
        
        # Combined efficiency score
        efficiency_score = path_efficiency * min(avg_velocity / 1e-6, 1.0)  # Normalize velocity
        
        return efficiency_score
    
    async def model_gas_composition_evolution(self, pathways: List[GasMigrationPathway],
                                            initial_composition: Dict[str, float],
                                            temperature_profile: np.ndarray,
                                            pressure_profile: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Model gas composition evolution along migration pathways"""
        try:
            composition_evolution = {}
            
            for pathway in pathways:
                pathway_composition = await self._calculate_pathway_composition_evolution(
                    pathway, initial_composition, temperature_profile, pressure_profile
                )
                composition_evolution[pathway.pathway_id] = pathway_composition
            
            return composition_evolution
            
        except Exception as e:
            logger.error(f"Error modeling gas composition evolution: {str(e)}")
            raise
    
    async def _calculate_pathway_composition_evolution(self, pathway: GasMigrationPathway,
                                                     initial_composition: Dict[str, float],
                                                     temperature_profile: np.ndarray,
                                                     pressure_profile: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate detailed gas composition evolution along a single pathway"""
        try:
            n_nodes = len(pathway.pathway_nodes)
            gas_species = list(initial_composition.keys())
            
            # Initialize composition arrays
            composition_evolution = {species: np.zeros(n_nodes) for species in gas_species}
            
            # Set initial composition
            for species in gas_species:
                composition_evolution[species][0] = initial_composition[species]
            
            # Query geochemical reaction data
            reaction_data = await self._get_geochemical_reaction_data(pathway)
            
            # Calculate evolution along pathway
            for i in range(1, n_nodes):
                # Get local conditions
                depth = pathway.pathway_nodes[i][2]
                temperature = self._interpolate_profile_value(temperature_profile, depth)
                pressure = self._interpolate_profile_value(pressure_profile, depth)
                
                # Calculate reaction rates
                reaction_rates = self._calculate_reaction_rates(
                    composition_evolution, i-1, temperature, pressure, reaction_data
                )
                
                # Update composition using reaction kinetics
                dt = self._calculate_residence_time(pathway, i-1, i)
                
                for species in gas_species:
                    # Apply reaction rate equation
                    rate = reaction_rates.get(species, 0.0)
                    composition_evolution[species][i] = (
                        composition_evolution[species][i-1] + rate * dt
                    )
                    
                    # Apply physical constraints
                    composition_evolution[species][i] = max(0.0, 
                        min(1.0, composition_evolution[species][i]))
                
                # Normalize to ensure sum = 1
                total = sum(composition_evolution[species][i] for species in gas_species)
                if total > 0:
                    for species in gas_species:
                        composition_evolution[species][i] /= total
            
            return composition_evolution
            
        except Exception as e:
            logger.error(f"Error calculating pathway composition evolution: {str(e)}")
            # Return initial composition for all nodes as fallback
            return {species: np.full(n_nodes, initial_composition[species]) 
                   for species in initial_composition.keys()}
    
    async def _get_geochemical_reaction_data(self, pathway: GasMigrationPathway) -> Dict[str, Any]:
        """Get geochemical reaction data for pathway"""
        try:
            # Query reaction database
            reaction_query = """
            SELECT reaction_type, reactants, products, activation_energy, 
                   pre_exponential_factor, temperature_range, pressure_range,
                   catalysts, inhibitors
            FROM geochemical_reactions 
            WHERE reaction_type IN ('gas_water', 'gas_rock', 'gas_gas', 'dissolution', 'precipitation')
            """
            
            reactions = await self.db_manager.execute_query(reaction_query)
            
            # Organize reaction data by type
            reaction_data = {
                'gas_water_reactions': [],
                'gas_rock_reactions': [],
                'gas_gas_reactions': [],
                'dissolution_reactions': [],
                'precipitation_reactions': []
            }
            
            for reaction in reactions:
                reaction_type = reaction['reaction_type']
                if reaction_type in reaction_data:
                    reaction_data[f"{reaction_type}_reactions"].append(reaction)
            
            return reaction_data
            
        except Exception as e:
            logger.warning(f"Failed to get geochemical reaction data: {e}")
            return {}
    
    def _calculate_reaction_rates(self, composition_evolution: Dict[str, np.ndarray],
                                 node_index: int, temperature: float, pressure: float,
                                 reaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate reaction rates for gas species"""
        try:
            rates = {}
            
            # Current composition
            current_comp = {species: comp[node_index] 
                           for species, comp in composition_evolution.items()}
            
            # CO2 dissolution in water
            if 'CO2' in current_comp:
                co2_dissolution_rate = self._calculate_co2_dissolution_rate(
                    current_comp['CO2'], temperature, pressure
                )
                rates['CO2'] = -co2_dissolution_rate  # Negative for loss
            
            # H2S oxidation
            if 'H2S' in current_comp and 'O2' in current_comp:
                h2s_oxidation_rate = self._calculate_h2s_oxidation_rate(
                    current_comp['H2S'], current_comp['O2'], temperature
                )
                rates['H2S'] = -h2s_oxidation_rate
                rates['SO2'] = rates.get('SO2', 0.0) + h2s_oxidation_rate
            
            # CH4 oxidation
            if 'CH4' in current_comp and 'O2' in current_comp:
                ch4_oxidation_rate = self._calculate_ch4_oxidation_rate(
                    current_comp['CH4'], current_comp['O2'], temperature
                )
                rates['CH4'] = -ch4_oxidation_rate
                rates['CO2'] = rates.get('CO2', 0.0) + ch4_oxidation_rate
            
            # Water vapor equilibrium
            if 'H2O' in current_comp:
                h2o_equilibrium_rate = self._calculate_h2o_equilibrium_rate(
                    current_comp['H2O'], temperature, pressure
                )
                rates['H2O'] = h2o_equilibrium_rate
            
            return rates
            
        except Exception as e:
            logger.warning(f"Failed to calculate reaction rates: {e}")
            return {}
    
    def _calculate_co2_dissolution_rate(self, co2_fraction: float, 
                                      temperature: float, pressure: float) -> float:
        """Calculate CO2 dissolution rate in water using Henry's law"""
        try:
            # Henry's law constant for CO2 (temperature dependent)
            # H = H0 * exp(d_ln_H/d(1/T) * (1/T - 1/T0))
            H0 = 3.4e-2  # mol/(L·atm) at 298K
            d_ln_H_d_inv_T = 2400  # K
            T0 = 298.15  # K
            
            henry_constant = H0 * np.exp(d_ln_H_d_inv_T * (1/temperature - 1/T0))
            
            # CO2 partial pressure
            co2_pressure = co2_fraction * pressure / 101325.0  # Convert to atm
            
            # Equilibrium CO2 concentration in water
            equilibrium_conc = henry_constant * co2_pressure
            
            # Dissolution rate (simplified first-order kinetics)
            k_dissolution = 1e-6  # s^-1 (rate constant)
            dissolution_rate = k_dissolution * co2_fraction * (1 - equilibrium_conc)
            
            return max(0.0, dissolution_rate)
            
        except Exception as e:
            logger.warning(f"Failed to calculate CO2 dissolution rate: {e}")
            return 0.0
    
    def _calculate_h2s_oxidation_rate(self, h2s_fraction: float, o2_fraction: float, 
                                     temperature: float) -> float:
        """Calculate H2S oxidation rate"""
        try:
            # Arrhenius equation parameters for H2S oxidation
            A = 1e12  # Pre-exponential factor (s^-1)
            Ea = 150000  # Activation energy (J/mol)
            R = 8.314  # Gas constant (J/(mol·K))
            
            # Rate constant
            k = A * np.exp(-Ea / (R * temperature))
            
            # Second-order reaction rate
            oxidation_rate = k * h2s_fraction * o2_fraction
            
            return max(0.0, oxidation_rate)
            
        except Exception as e:
            logger.warning(f"Failed to calculate H2S oxidation rate: {e}")
            return 0.0
    
    def _calculate_ch4_oxidation_rate(self, ch4_fraction: float, o2_fraction: float,
                                     temperature: float) -> float:
        """Calculate CH4 oxidation rate"""
        try:
            # Arrhenius equation parameters for CH4 oxidation
            A = 8e13  # Pre-exponential factor (s^-1)
            Ea = 200000  # Activation energy (J/mol)
            R = 8.314  # Gas constant (J/(mol·K))
            
            # Rate constant
            k = A * np.exp(-Ea / (R * temperature))
            
            # Second-order reaction rate
            oxidation_rate = k * ch4_fraction * o2_fraction
            
            return max(0.0, oxidation_rate)
            
        except Exception as e:
            logger.warning(f"Failed to calculate CH4 oxidation rate: {e}")
            return 0.0
    
    def _calculate_h2o_equilibrium_rate(self, h2o_fraction: float, 
                                      temperature: float, pressure: float) -> float:
        """Calculate H2O vapor equilibrium rate"""
        try:
            # Clausius-Clapeyron equation for water vapor pressure
            # ln(P) = -L/(R*T) + constant
            L = 40660  # Latent heat of vaporization (J/mol)
            R = 8.314  # Gas constant (J/(mol·K))
            P0 = 101325  # Reference pressure (Pa)
            T0 = 373.15  # Reference temperature (K)
            
            # Saturation vapor pressure
            ln_P_sat = np.log(P0) - L/R * (1/temperature - 1/T0)
            P_sat = np.exp(ln_P_sat)
            
            # Equilibrium H2O fraction
            equilibrium_fraction = P_sat / pressure
            
            # Equilibration rate (first-order)
            k_equilibrium = 1e-4  # s^-1
            equilibrium_rate = k_equilibrium * (equilibrium_fraction - h2o_fraction)
            
            return equilibrium_rate
            
        except Exception as e:
            logger.warning(f"Failed to calculate H2O equilibrium rate: {e}")
            return 0.0
    
    def _calculate_residence_time(self, pathway: GasMigrationPathway, 
                                 start_idx: int, end_idx: int) -> float:
        """Calculate gas residence time between pathway nodes"""
        try:
            # Calculate segment length
            start_pos = np.array(pathway.pathway_nodes[start_idx])
            end_pos = np.array(pathway.pathway_nodes[end_idx])
            segment_length = np.linalg.norm(end_pos - start_pos)
            
            # Average velocity in segment
            if len(pathway.flow_velocity) > start_idx:
                avg_velocity = (pathway.flow_velocity[start_idx] + 
                               pathway.flow_velocity[end_idx]) / 2.0
            else:
                avg_velocity = 1e-6  # Default velocity (m/s)
            
            # Residence time
            if avg_velocity > 0:
                residence_time = segment_length / avg_velocity
            else:
                residence_time = 86400.0  # Default 1 day
            
            return residence_time
            
        except Exception as e:
            logger.warning(f"Failed to calculate residence time: {e}")
            return 86400.0  # Default 1 day
    
    def _interpolate_profile_value(self, profile: np.ndarray, depth: float) -> float:
        """Interpolate value from depth profile"""
        try:
            if len(profile) == 0:
                return 0.0
            
            # Assume profile is indexed by depth with regular spacing
            max_depth = len(profile) * 100.0  # Assuming 100m spacing
            depth_index = depth / 100.0
            
            if depth_index <= 0:
                return profile[0]
            elif depth_index >= len(profile) - 1:
                return profile[-1]
            else:
                # Linear interpolation
                idx = int(depth_index)
                frac = depth_index - idx
                return profile[idx] * (1 - frac) + profile[idx + 1] * frac
                
        except Exception as e:
            logger.warning(f"Failed to interpolate profile value: {e}")
            return 0.0
    
    async def perform_uncertainty_quantification(self, pathways: List[GasMigrationPathway],
                                               n_monte_carlo: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive uncertainty quantification using Monte Carlo methods"""
        try:
            uncertainty_results = {}
            
            for pathway in pathways:
                pathway_uncertainty = await self._calculate_pathway_uncertainty(
                    pathway, n_monte_carlo
                )
                uncertainty_results[pathway.pathway_id] = pathway_uncertainty
            
            return uncertainty_results
            
        except Exception as e:
            logger.error(f"Error performing uncertainty quantification: {str(e)}")
            raise
    
    async def _calculate_pathway_uncertainty(self, pathway: GasMigrationPathway,
                                           n_monte_carlo: int) -> Dict[str, Any]:
        """Calculate uncertainty for a single pathway using Monte Carlo simulation"""
        try:
            # Parameter uncertainty distributions
            parameter_uncertainties = await self._get_parameter_uncertainties(pathway)
            
            # Monte Carlo simulation
            results = {
                'flow_velocity': [],
                'pressure_gradient': [],
                'efficiency_score': [],
                'arrival_time': [],
                'gas_flux': []
            }
            
            for i in range(n_monte_carlo):
                # Sample uncertain parameters
                sampled_params = self._sample_uncertain_parameters(parameter_uncertainties)
                
                # Recalculate pathway with sampled parameters
                modified_pathway = self._modify_pathway_with_parameters(pathway, sampled_params)
                
                # Calculate metrics
                results['flow_velocity'].append(np.mean(modified_pathway.flow_velocity))
                results['pressure_gradient'].append(np.mean(modified_pathway.pressure_gradient))
                results['efficiency_score'].append(modified_pathway.efficiency_score)
                
                # Calculate arrival time
                arrival_time = self._calculate_total_arrival_time(modified_pathway)
                results['arrival_time'].append(arrival_time)
                
                # Calculate gas flux
                gas_flux = self._calculate_gas_flux(modified_pathway)
                results['gas_flux'].append(gas_flux)
            
            # Calculate statistics
            uncertainty_stats = {}
            for metric, values in results.items():
                values_array = np.array(values)
                uncertainty_stats[metric] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'percentile_5': np.percentile(values_array, 5),
                    'percentile_25': np.percentile(values_array, 25),
                    'percentile_50': np.percentile(values_array, 50),
                    'percentile_75': np.percentile(values_array, 75),
                    'percentile_95': np.percentile(values_array, 95),
                    'coefficient_of_variation': np.std(values_array) / np.mean(values_array) if np.mean(values_array) != 0 else 0
                }
            
            return uncertainty_stats
            
        except Exception as e:
            logger.error(f"Error calculating pathway uncertainty: {str(e)}")
            return {}
    
    async def _get_parameter_uncertainties(self, pathway: GasMigrationPathway) -> Dict[str, Dict[str, float]]:
        """Get comprehensive parameter uncertainty distributions using advanced statistical analysis"""
        try:
            # Advanced uncertainty database query with spatial and temporal correlations
            uncertainty_query = """
            SELECT pu.parameter_name, pu.distribution_type, pu.mean_value, pu.std_deviation,
                   pu.min_value, pu.max_value, pu.skewness, pu.kurtosis,
                   pu.correlation_matrix, pu.spatial_correlation_length,
                   pu.temporal_correlation_time, pu.measurement_uncertainty,
                   pu.model_uncertainty, pu.epistemic_uncertainty,
                   pc.correlation_coefficient, pc.correlation_type
            FROM parameter_uncertainties pu
            LEFT JOIN parameter_correlations pc ON pu.parameter_name = pc.parameter_1
            WHERE pu.parameter_category = 'gas_migration'
            AND pu.applicable_depth_range @> %s
            ORDER BY pu.confidence_level DESC
            """
            
            # Use pathway depth for depth-dependent uncertainties
            avg_depth = np.mean([node[2] for node in pathway.pathway_nodes])
            
            try:
                uncertainty_data = await self.db_manager.execute_query(
                    uncertainty_query, (avg_depth,)
                )
            except RuntimeError:
                uncertainty_data = []
            
            # Advanced uncertainty characterization
            if uncertainty_data:
                uncertainties = self._process_advanced_uncertainty_data(uncertainty_data, pathway)
            else:
                uncertainties = self._generate_advanced_default_uncertainties(pathway)
            
            # Add pathway-specific uncertainty adjustments
            uncertainties = self._adjust_uncertainties_for_pathway(uncertainties, pathway)
            
            # Add cross-parameter correlations
            uncertainties = self._add_parameter_correlations(uncertainties, pathway)
            
            return uncertainties
                
        except Exception as e:
            logger.warning(f"Failed to get parameter uncertainties: {e}")
            return self._generate_advanced_default_uncertainties(pathway)
    
    def _process_advanced_uncertainty_data(self, uncertainty_data: List[Dict], 
                                         pathway: GasMigrationPathway) -> Dict[str, Dict[str, float]]:
        """Process advanced uncertainty data from database"""
        try:
            uncertainties = {}
            
            for param in uncertainty_data:
                param_name = param['parameter_name']
                
                # Advanced distribution parameters
                uncertainties[param_name] = {
                    'type': param['distribution_type'],
                    'mean': param['mean_value'],
                    'std': param['std_deviation'],
                    'min': param.get('min_value'),
                    'max': param.get('max_value'),
                    'skewness': param.get('skewness', 0.0),
                    'kurtosis': param.get('kurtosis', 3.0),
                    'spatial_correlation_length': param.get('spatial_correlation_length', 1000.0),
                    'temporal_correlation_time': param.get('temporal_correlation_time', 86400.0),
                    'measurement_uncertainty': param.get('measurement_uncertainty', 0.1),
                    'model_uncertainty': param.get('model_uncertainty', 0.2),
                    'epistemic_uncertainty': param.get('epistemic_uncertainty', 0.15)
                }
            
            return uncertainties
            
        except Exception as e:
            logger.warning(f"Failed to process advanced uncertainty data: {e}")
            return {}
    
    def _generate_advanced_default_uncertainties(self, pathway: GasMigrationPathway) -> Dict[str, Dict[str, float]]:
        """Generate advanced default uncertainty distributions"""
        try:
            # Depth-dependent uncertainty scaling
            avg_depth = np.mean([node[2] for node in pathway.pathway_nodes])
            depth_factor = 1.0 + avg_depth / 5000.0  # Increase uncertainty with depth
            
            # Pathway length-dependent uncertainty
            pathway_length = sum(
                np.linalg.norm(np.array(pathway.pathway_nodes[i+1]) - np.array(pathway.pathway_nodes[i]))
                for i in range(len(pathway.pathway_nodes) - 1)
            )
            length_factor = 1.0 + pathway_length / 10000.0  # Increase uncertainty with length
            
            # Advanced uncertainty distributions with multiple uncertainty sources
            advanced_uncertainties = {
                'matrix_permeability': {
                    'type': 'lognormal',
                    'mean': -13.0,  # log10(m²)
                    'std': 2.0 * depth_factor,
                    'min': -18.0,
                    'max': -10.0,
                    'skewness': 0.5,
                    'kurtosis': 3.5,
                    'measurement_uncertainty': 0.3,
                    'model_uncertainty': 0.4,
                    'epistemic_uncertainty': 0.2,
                    'spatial_correlation_length': 500.0,
                    'temporal_correlation_time': 86400.0
                },
                'fracture_permeability': {
                    'type': 'lognormal',
                    'mean': -11.0,  # log10(m²)
                    'std': 3.0 * depth_factor,
                    'min': -16.0,
                    'max': -8.0,
                    'skewness': 1.0,
                    'kurtosis': 4.0,
                    'measurement_uncertainty': 0.5,
                    'model_uncertainty': 0.6,
                    'epistemic_uncertainty': 0.3
                },
                'porosity': {
                    'type': 'beta',  # Bounded between 0 and 1
                    'mean': 0.15,
                    'std': 0.05 * depth_factor,
                    'min': 0.01,
                    'max': 0.4,
                    'alpha': 2.0,  # Beta distribution parameters
                    'beta': 8.0,
                    'measurement_uncertainty': 0.1,
                    'model_uncertainty': 0.15,
                    'epistemic_uncertainty': 0.1
                },
                'pressure_gradient': {
                    'type': 'gamma',
                    'mean': 1.0,
                    'std': 0.3 * length_factor,
                    'min': 0.1,
                    'max': 5.0,
                    'shape': 4.0,  # Gamma distribution parameters
                    'scale': 0.25,
                    'measurement_uncertainty': 0.2,
                    'model_uncertainty': 0.25,
                    'epistemic_uncertainty': 0.15
                },
                'gas_density': {
                    'type': 'normal',
                    'mean': 2.0,  # kg/m³
                    'std': 0.4,
                    'min': 0.5,
                    'max': 5.0,
                    'measurement_uncertainty': 0.1,
                    'model_uncertainty': 0.2,
                    'epistemic_uncertainty': 0.1
                },
                'gas_viscosity': {
                    'type': 'lognormal',
                    'mean': np.log(2e-5),  # ln(Pa·s)
                    'std': 0.3,
                    'min': 1e-6,
                    'max': 1e-4,
                    'measurement_uncertainty': 0.15,
                    'model_uncertainty': 0.2,
                    'epistemic_uncertainty': 0.1
                },
                'fracture_aperture': {
                    'type': 'lognormal',
                    'mean': -3.0,  # log10(m)
                    'std': 1.5 * depth_factor,
                    'min': -5.0,
                    'max': -1.0,
                    'measurement_uncertainty': 0.4,
                    'model_uncertainty': 0.5,
                    'epistemic_uncertainty': 0.3
                },
                'temperature': {
                    'type': 'normal',
                    'mean': 15.0 + 25.0 * avg_depth / 1000.0,  # Geothermal gradient
                    'std': 5.0 + avg_depth / 2000.0,
                    'min': 0.0,
                    'max': 200.0,
                    'measurement_uncertainty': 0.05,
                    'model_uncertainty': 0.1,
                    'epistemic_uncertainty': 0.05
                },
                'stress_ratio': {
                    'type': 'normal',
                    'mean': 1.2,
                    'std': 0.3,
                    'min': 0.5,
                    'max': 3.0,
                    'measurement_uncertainty': 0.2,
                    'model_uncertainty': 0.3,
                    'epistemic_uncertainty': 0.2
                }
            }
            
            return advanced_uncertainties
            
        except Exception as e:
            logger.warning(f"Failed to generate advanced default uncertainties: {e}")
            return {}
    
    def _adjust_uncertainties_for_pathway(self, uncertainties: Dict[str, Dict[str, float]],
                                        pathway: GasMigrationPathway) -> Dict[str, Dict[str, float]]:
        """Adjust uncertainties based on pathway-specific characteristics"""
        try:
            # Pathway efficiency affects uncertainty
            efficiency = pathway.efficiency_score
            efficiency_factor = 2.0 - efficiency  # Lower efficiency = higher uncertainty
            
            # Structural controls affect uncertainty
            num_controls = len(pathway.structural_controls)
            control_factor = 1.0 / (1.0 + num_controls / 10.0)  # More controls = lower uncertainty
            
            # Apply adjustments
            for param_name, param_dist in uncertainties.items():
                if 'std' in param_dist:
                    param_dist['std'] *= efficiency_factor * control_factor
                
                # Adjust epistemic uncertainty based on pathway complexity
                if 'epistemic_uncertainty' in param_dist:
                    param_dist['epistemic_uncertainty'] *= efficiency_factor
            
            return uncertainties
            
        except Exception as e:
            logger.warning(f"Failed to adjust uncertainties for pathway: {e}")
            return uncertainties
    
    def _add_parameter_correlations(self, uncertainties: Dict[str, Dict[str, float]],
                                  pathway: GasMigrationPathway) -> Dict[str, Dict[str, float]]:
        """Add cross-parameter correlations to uncertainty analysis"""
        try:
            # Define correlation matrix for key parameters
            correlation_matrix = {
                ('matrix_permeability', 'porosity'): 0.7,  # Strong positive correlation
                ('fracture_permeability', 'fracture_aperture'): 0.9,  # Very strong correlation
                ('pressure_gradient', 'gas_density'): 0.5,  # Moderate correlation
                ('temperature', 'gas_viscosity'): -0.6,  # Negative correlation
                ('stress_ratio', 'fracture_permeability'): 0.4,  # Moderate correlation
                ('porosity', 'fracture_aperture'): -0.3  # Weak negative correlation
            }
            
            # Add correlation information to uncertainties
            for param_name, param_dist in uncertainties.items():
                correlations = {}
                for (param1, param2), corr_coeff in correlation_matrix.items():
                    if param1 == param_name:
                        correlations[param2] = corr_coeff
                    elif param2 == param_name:
                        correlations[param1] = corr_coeff
                
                param_dist['correlations'] = correlations
            
            return uncertainties
            
        except Exception as e:
            logger.warning(f"Failed to add parameter correlations: {e}")
            return uncertainties
    
    def _sample_uncertain_parameters(self, uncertainties: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Sample uncertain parameters using advanced statistical methods with correlations"""
        try:
            sampled_params = {}
            
            # First pass: sample independent parameters
            independent_samples = {}
            
            for param_name, param_dist in uncertainties.items():
                dist_type = param_dist.get('type', 'normal')
                mean = param_dist.get('mean', 0.0)
                std = param_dist.get('std', 1.0)
                min_val = param_dist.get('min')
                max_val = param_dist.get('max')
                
                # Sample from appropriate distribution
                if dist_type == 'normal':
                    sample = np.random.normal(mean, std)
                elif dist_type == 'lognormal':
                    sample = np.random.lognormal(mean, std)
                elif dist_type == 'uniform':
                    min_bound = min_val if min_val is not None else mean - 2*std
                    max_bound = max_val if max_val is not None else mean + 2*std
                    sample = np.random.uniform(min_bound, max_bound)
                elif dist_type == 'beta':
                    alpha = param_dist.get('alpha', 2.0)
                    beta = param_dist.get('beta', 5.0)
                    sample = np.random.beta(alpha, beta)
                    # Scale to desired range
                    if min_val is not None and max_val is not None:
                        sample = min_val + sample * (max_val - min_val)
                elif dist_type == 'gamma':
                    shape = param_dist.get('shape', 2.0)
                    scale = param_dist.get('scale', 1.0)
                    sample = np.random.gamma(shape, scale)
                elif dist_type == 'weibull':
                    shape = param_dist.get('shape', 2.0)
                    scale = param_dist.get('scale', 1.0)
                    sample = np.random.weibull(shape) * scale
                else:
                    sample = mean  # Default to mean value
                
                # Apply bounds if specified
                if min_val is not None:
                    sample = max(sample, min_val)
                if max_val is not None:
                    sample = min(sample, max_val)
                
                independent_samples[param_name] = sample
            
            # Second pass: apply correlations using Cholesky decomposition
            sampled_params = self._apply_parameter_correlations(
                independent_samples, uncertainties
            )
            
            # Third pass: apply physical constraints and consistency checks
            sampled_params = self._apply_physical_constraints(sampled_params)
            
            return sampled_params
            
        except Exception as e:
            logger.warning(f"Failed to sample uncertain parameters: {e}")
            # Fallback to simple sampling
            return {param: param_dist.get('mean', 1.0) for param, param_dist in uncertainties.items()}
    
    def _apply_parameter_correlations(self, independent_samples: Dict[str, float],
                                    uncertainties: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Apply parameter correlations using advanced statistical methods"""
        try:
            # Extract correlation information
            param_names = list(independent_samples.keys())
            n_params = len(param_names)
            
            if n_params < 2:
                return independent_samples
            
            # Build correlation matrix
            correlation_matrix = np.eye(n_params)
            
            for i, param1 in enumerate(param_names):
                correlations = uncertainties[param1].get('correlations', {})
                for j, param2 in enumerate(param_names):
                    if param2 in correlations:
                        correlation_matrix[i, j] = correlations[param2]
                        correlation_matrix[j, i] = correlations[param2]  # Symmetric
            
            # Ensure positive definite matrix
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Cholesky decomposition for correlated sampling
            try:
                L = np.linalg.cholesky(correlation_matrix)
                
                # Convert samples to standard normal
                standard_samples = np.array([
                    (independent_samples[param] - uncertainties[param].get('mean', 0.0)) / 
                    uncertainties[param].get('std', 1.0)
                    for param in param_names
                ])
                
                # Apply correlation
                correlated_samples = L @ standard_samples
                
                # Convert back to original distributions
                correlated_params = {}
                for i, param in enumerate(param_names):
                    mean = uncertainties[param].get('mean', 0.0)
                    std = uncertainties[param].get('std', 1.0)
                    correlated_params[param] = mean + std * correlated_samples[i]
                
                return correlated_params
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                logger.warning("Cholesky decomposition failed, using independent samples")
                return independent_samples
            
        except Exception as e:
            logger.warning(f"Failed to apply parameter correlations: {e}")
            return independent_samples
    
    def _apply_physical_constraints(self, sampled_params: Dict[str, float]) -> Dict[str, float]:
        """Apply physical constraints to sampled parameters"""
        try:
            # Ensure physical consistency
            
            # Permeability constraints
            if 'matrix_permeability' in sampled_params:
                sampled_params['matrix_permeability'] = max(1e-20, sampled_params['matrix_permeability'])
            
            if 'fracture_permeability' in sampled_params:
                sampled_params['fracture_permeability'] = max(1e-20, sampled_params['fracture_permeability'])
            
            # Porosity constraints
            if 'porosity' in sampled_params:
                sampled_params['porosity'] = max(0.001, min(0.99, sampled_params['porosity']))
            
            # Density constraints
            if 'gas_density' in sampled_params:
                sampled_params['gas_density'] = max(0.1, min(10.0, sampled_params['gas_density']))
            
            # Viscosity constraints
            if 'gas_viscosity' in sampled_params:
                sampled_params['gas_viscosity'] = max(1e-6, min(1e-3, sampled_params['gas_viscosity']))
            
            # Temperature constraints
            if 'temperature' in sampled_params:
                sampled_params['temperature'] = max(0.0, min(500.0, sampled_params['temperature']))
            
            # Aperture constraints
            if 'fracture_aperture' in sampled_params:
                sampled_params['fracture_aperture'] = max(1e-6, min(0.1, sampled_params['fracture_aperture']))
            
            # Consistency checks
            # Fracture permeability should generally be higher than matrix permeability
            if ('matrix_permeability' in sampled_params and 
                'fracture_permeability' in sampled_params):
                if sampled_params['fracture_permeability'] < sampled_params['matrix_permeability']:
                    sampled_params['fracture_permeability'] = sampled_params['matrix_permeability'] * 10.0
            
            return sampled_params
            
        except Exception as e:
            logger.warning(f"Failed to apply physical constraints: {e}")
            return sampled_params
    
    def _modify_pathway_with_parameters(self, pathway: GasMigrationPathway,
                                      sampled_params: Dict[str, float]) -> GasMigrationPathway:
        """Modify pathway properties with sampled uncertain parameters using advanced physics-based scaling"""
        try:
            # Create a deep copy of the pathway with advanced property scaling
            modified_pathway = GasMigrationPathway(
                pathway_id=pathway.pathway_id + "_mc_sample",
                source_location=pathway.source_location,
                surface_location=pathway.surface_location,
                pathway_nodes=pathway.pathway_nodes.copy(),
                permeability_profile=pathway.permeability_profile.copy(),
                pressure_gradient=pathway.pressure_gradient.copy(),
                flow_velocity=pathway.flow_velocity.copy(),
                gas_composition_evolution={k: v.copy() for k, v in pathway.gas_composition_evolution.items()},
                structural_controls=pathway.structural_controls.copy(),
                efficiency_score=pathway.efficiency_score,
                uncertainty_bounds={}
            )
            
            # Advanced parameter modifications with physics-based scaling
            
            # 1. Matrix permeability modifications
            if 'matrix_permeability' in sampled_params:
                matrix_perm_factor = 10**sampled_params['matrix_permeability'] / 1e-13  # Convert from log scale
                modified_pathway.permeability_profile *= matrix_perm_factor
            
            # 2. Fracture permeability modifications (affects high-permeability zones)
            if 'fracture_permeability' in sampled_params:
                fracture_perm = 10**sampled_params['fracture_permeability']
                # Apply to high-permeability zones (top 20% of profile)
                high_perm_threshold = np.percentile(modified_pathway.permeability_profile, 80)
                high_perm_mask = modified_pathway.permeability_profile >= high_perm_threshold
                modified_pathway.permeability_profile[high_perm_mask] = fracture_perm
            
            # 3. Porosity effects on permeability (Kozeny-Carman relationship)
            if 'porosity' in sampled_params:
                porosity = sampled_params['porosity']
                reference_porosity = 0.15
                # k ∝ φ³/(1-φ)² (Kozeny-Carman)
                porosity_factor = (
                    (porosity**3 / (1-porosity)**2) / 
                    (reference_porosity**3 / (1-reference_porosity)**2)
                )
                modified_pathway.permeability_profile *= porosity_factor
            
            # 4. Pressure gradient modifications with depth dependence
            if 'pressure_gradient' in sampled_params:
                pressure_factor = sampled_params['pressure_gradient']
                # Apply depth-dependent scaling
                depths = np.array([node[2] for node in pathway.pathway_nodes])
                depth_scaling = 1.0 + (depths / 5000.0) * (pressure_factor - 1.0)
                modified_pathway.pressure_gradient *= depth_scaling
            
            # 5. Gas property effects on flow velocity
            gas_density = sampled_params.get('gas_density', 2.0)
            gas_viscosity = sampled_params.get('gas_viscosity', 2e-5)
            temperature = sampled_params.get('temperature', 50.0)
            
            # Recalculate flow velocities using Darcy's law with updated properties
            modified_velocity = self._recalculate_velocity_with_properties(
                modified_pathway, gas_density, gas_viscosity, temperature
            )
            modified_pathway.flow_velocity = modified_velocity
            
            # 6. Stress effects on fracture aperture and permeability
            if 'stress_ratio' in sampled_params:
                stress_ratio = sampled_params['stress_ratio']
                stress_enhancement = self._calculate_stress_enhancement_factor(stress_ratio)
                # Apply to fracture-dominated zones
                fracture_zones = modified_pathway.permeability_profile > 1e-12
                modified_pathway.permeability_profile[fracture_zones] *= stress_enhancement
            
            # 7. Temperature effects on gas properties and rock properties
            if 'temperature' in sampled_params:
                temp_effects = self._calculate_temperature_effects(
                    sampled_params['temperature'], modified_pathway
                )
                modified_pathway.permeability_profile *= temp_effects['permeability_factor']
                modified_pathway.flow_velocity *= temp_effects['velocity_factor']
            
            # 8. Fracture aperture effects (cubic law)
            if 'fracture_aperture' in sampled_params:
                aperture = sampled_params['fracture_aperture']
                reference_aperture = 1e-3  # 1mm
                # k ∝ aperture³ (cubic law)
                aperture_factor = (aperture / reference_aperture)**3
                # Apply to fracture zones
                fracture_zones = modified_pathway.permeability_profile > 1e-13
                modified_pathway.permeability_profile[fracture_zones] *= aperture_factor
            
            # 9. Recalculate derived properties
            modified_pathway = self._recalculate_derived_properties(
                modified_pathway, sampled_params
            )
            
            # 10. Update efficiency score with new properties
            modified_pathway.efficiency_score = self._calculate_modified_efficiency_score(
                modified_pathway, sampled_params
            )
            
            # 11. Apply physical consistency checks
            modified_pathway = self._apply_pathway_consistency_checks(modified_pathway)
            
            return modified_pathway
            
        except Exception as e:
            logger.warning(f"Failed to modify pathway with parameters: {e}")
            return pathway
    
    def _recalculate_velocity_with_properties(self, pathway: GasMigrationPathway,
                                            gas_density: float, gas_viscosity: float,
                                            temperature: float) -> np.ndarray:
        """Recalculate flow velocities with updated gas properties"""
        try:
            # Darcy's law: v = -k/μ × ∇P
            velocities = []
            
            for i, perm in enumerate(pathway.permeability_profile):
                if i < len(pathway.pressure_gradient):
                    pressure_grad = pathway.pressure_gradient[i]
                    
                    # Include buoyancy effects
                    depth = pathway.pathway_nodes[i][2] if i < len(pathway.pathway_nodes) else 1000.0
                    buoyancy_force = gas_density * 9.81  # N/m³
                    
                    # Darcy velocity with buoyancy
                    darcy_velocity = perm * (pressure_grad + buoyancy_force) / gas_viscosity
                    
                    # Temperature correction for gas expansion
                    temp_kelvin = temperature + 273.15
                    expansion_factor = temp_kelvin / 323.15  # Reference 50°C
                    
                    velocity = darcy_velocity * expansion_factor
                    velocities.append(max(0.0, velocity))
                else:
                    velocities.append(0.0)
            
            return np.array(velocities)
            
        except Exception as e:
            logger.warning(f"Failed to recalculate velocity with properties: {e}")
            return pathway.flow_velocity
    
    def _calculate_stress_enhancement_factor(self, stress_ratio: float) -> float:
        """Calculate stress-dependent permeability enhancement"""
        try:
            # Higher stress ratios can enhance fracture permeability through dilatancy
            # But excessive stress can also close fractures
            
            if stress_ratio < 1.0:
                # Low differential stress - minimal enhancement
                return 1.0
            elif stress_ratio < 2.0:
                # Moderate stress - enhancement through dilatancy
                return 1.0 + 2.0 * (stress_ratio - 1.0)
            else:
                # High stress - potential fracture closure
                return max(0.5, 3.0 - 0.5 * (stress_ratio - 2.0))
                
        except Exception as e:
            logger.warning(f"Failed to calculate stress enhancement factor: {e}")
            return 1.0
    
    def _calculate_temperature_effects(self, temperature: float, 
                                     pathway: GasMigrationPathway) -> Dict[str, float]:
        """Calculate temperature effects on pathway properties"""
        try:
            reference_temp = 50.0  # °C
            temp_diff = temperature - reference_temp
            
            # Thermal expansion of fractures
            thermal_expansion_coeff = 1e-5  # /K
            permeability_factor = 1.0 + 3.0 * thermal_expansion_coeff * temp_diff  # Cubic law
            
            # Gas viscosity temperature dependence (Sutherland's law)
            temp_kelvin = temperature + 273.15
            ref_temp_kelvin = reference_temp + 273.15
            viscosity_ratio = (temp_kelvin / ref_temp_kelvin)**0.7
            
            # Velocity inversely proportional to viscosity
            velocity_factor = 1.0 / viscosity_ratio
            
            return {
                'permeability_factor': max(0.5, min(2.0, permeability_factor)),
                'velocity_factor': max(0.5, min(2.0, velocity_factor))
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate temperature effects: {e}")
            return {'permeability_factor': 1.0, 'velocity_factor': 1.0}
    
    def _recalculate_derived_properties(self, pathway: GasMigrationPathway,
                                      sampled_params: Dict[str, float]) -> GasMigrationPathway:
        """Recalculate derived pathway properties"""
        try:
            # Update gas composition evolution if temperature changed significantly
            if 'temperature' in sampled_params:
                temp_change = abs(sampled_params['temperature'] - 50.0)
                if temp_change > 20.0:  # Significant temperature change
                    # Simplified composition adjustment
                    for species, evolution in pathway.gas_composition_evolution.items():
                        if species == 'H2O':
                            # Water vapor increases with temperature
                            temp_factor = 1.0 + temp_change / 100.0
                            pathway.gas_composition_evolution[species] = evolution * temp_factor
                        elif species == 'CO2':
                            # CO2 solubility decreases with temperature
                            temp_factor = 1.0 - temp_change / 200.0
                            pathway.gas_composition_evolution[species] = evolution * max(0.5, temp_factor)
            
            return pathway
            
        except Exception as e:
            logger.warning(f"Failed to recalculate derived properties: {e}")
            return pathway
    
    def _calculate_modified_efficiency_score(self, pathway: GasMigrationPathway,
                                           sampled_params: Dict[str, float]) -> float:
        """Calculate efficiency score for modified pathway"""
        try:
            # Base efficiency calculation
            if len(pathway.pathway_nodes) < 2:
                return 0.0
            
            # Path tortuosity
            total_length = sum(
                np.linalg.norm(np.array(pathway.pathway_nodes[i+1]) - np.array(pathway.pathway_nodes[i]))
                for i in range(len(pathway.pathway_nodes) - 1)
            )
            
            direct_distance = np.linalg.norm(
                np.array(pathway.pathway_nodes[-1]) - np.array(pathway.pathway_nodes[0])
            )
            
            path_efficiency = direct_distance / total_length if total_length > 0 else 0.0
            
            # Flow efficiency based on velocity
            avg_velocity = np.mean(pathway.flow_velocity) if len(pathway.flow_velocity) > 0 else 0.0
            velocity_efficiency = min(avg_velocity / 1e-5, 1.0)  # Normalize to 10 μm/s
            
            # Permeability efficiency
            avg_permeability = np.mean(pathway.permeability_profile)
            perm_efficiency = min(np.log10(avg_permeability / 1e-15) / 5.0, 1.0)
            
            # Combined efficiency with parameter-dependent weighting
            efficiency = (
                0.4 * path_efficiency +
                0.4 * velocity_efficiency +
                0.2 * perm_efficiency
            )
            
            return max(0.0, min(1.0, efficiency))
            
        except Exception as e:
            logger.warning(f"Failed to calculate modified efficiency score: {e}")
            return 0.5
    
    def _apply_pathway_consistency_checks(self, pathway: GasMigrationPathway) -> GasMigrationPathway:
        """Apply physical consistency checks to modified pathway"""
        try:
            # Ensure permeability values are physically reasonable
            pathway.permeability_profile = np.clip(
                pathway.permeability_profile, 1e-20, 1e-8
            )
            
            # Ensure pressure gradients are positive
            pathway.pressure_gradient = np.maximum(pathway.pressure_gradient, 0.0)
            
            # Ensure velocities are non-negative
            pathway.flow_velocity = np.maximum(pathway.flow_velocity, 0.0)
            
            # Ensure gas compositions sum to 1.0
            for species, evolution in pathway.gas_composition_evolution.items():
                pathway.gas_composition_evolution[species] = np.clip(evolution, 0.0, 1.0)
            
            return pathway
            
        except Exception as e:
            logger.warning(f"Failed to apply pathway consistency checks: {e}")
            return pathway
    
    def _calculate_total_arrival_time(self, pathway: GasMigrationPathway) -> float:
        """Calculate total gas arrival time at surface"""
        try:
            total_time = 0.0
            
            for i in range(len(pathway.pathway_nodes) - 1):
                segment_time = self._calculate_residence_time(pathway, i, i + 1)
                total_time += segment_time
            
            return total_time
            
        except Exception as e:
            logger.warning(f"Failed to calculate total arrival time: {e}")
            return 86400.0  # Default 1 day
    
    def _calculate_gas_flux(self, pathway: GasMigrationPathway) -> float:
        """Calculate gas flux through pathway"""
        try:
            # Average cross-sectional area (simplified)
            avg_permeability = np.mean(pathway.permeability_profile)
            avg_pressure_gradient = np.mean(pathway.pressure_gradient)
            
            # Darcy flux
            gas_viscosity = 2e-5  # Pa·s (CO2 at volcanic conditions)
            darcy_flux = avg_permeability * avg_pressure_gradient / gas_viscosity
            
            # Convert to volumetric flux (m³/s per unit area)
            gas_density = 2.0  # kg/m³
            volumetric_flux = darcy_flux / gas_density
            
            return max(0.0, volumetric_flux)
            
        except Exception as e:
            logger.warning(f"Failed to calculate gas flux: {e}")
            return 0.0
    
    def _analyze_structural_control_influence(self, feature: Dict[str, Any], 
                                            pathway_array: np.ndarray) -> Dict[str, float]:
        """Analyze structural feature influence on gas migration pathway using advanced multi-physics modeling"""
        try:
            # Advanced geometric analysis with 3D spatial relationships
            from shapely.wkt import loads as wkt_loads
            from shapely.geometry import Point, LineString
            
            geometry = wkt_loads(feature['geometry_wkt'])
            structure_type = feature.get('structure_type', 'unknown')
            strike = np.radians(feature.get('strike', 0.0))
            dip = np.radians(feature.get('dip', 90.0))
            perm_enhancement = feature.get('permeability_enhancement', 1.0)
            stress_orientation = np.radians(feature.get('stress_orientation', 0.0))
            age_ma = feature.get('age_ma', 0.0)
            
            # Create pathway linestring for advanced spatial analysis
            pathway_line = LineString([(p[0], p[1]) for p in pathway_array])
            
            # Advanced distance calculations with 3D geometry
            min_distance = float('inf')
            intersection_points = []
            proximity_segments = []
            
            for i, point in enumerate(pathway_array):
                pathway_point = Point(point[0], point[1])
                distance_2d = geometry.distance(pathway_point)
                
                # Include depth-dependent distance weighting
                depth_weight = 1.0 + point[2] / 5000.0  # Deeper = more influence
                weighted_distance = distance_2d / depth_weight
                
                if weighted_distance < min_distance:
                    min_distance = weighted_distance
                
                # Identify intersection and near-intersection points
                if distance_2d < 50.0:  # Within 50m
                    intersection_points.append((i, point, distance_2d))
                    
                if i > 0:
                    segment = LineString([(pathway_array[i-1][0], pathway_array[i-1][1]), 
                                        (point[0], point[1])])
                    if geometry.intersects(segment) or geometry.distance(segment) < 100.0:
                        proximity_segments.append((i-1, i, geometry.distance(segment)))
            
            # Advanced distance influence with structure-type specific parameters
            structure_params = {
                'major_fault': {'decay_length': 2000.0, 'max_influence': 1.0},
                'fault': {'decay_length': 1000.0, 'max_influence': 0.9},
                'fracture': {'decay_length': 300.0, 'max_influence': 0.7},
                'joint': {'decay_length': 100.0, 'max_influence': 0.5},
                'shear_zone': {'decay_length': 1500.0, 'max_influence': 0.95},
                'dike': {'decay_length': 500.0, 'max_influence': 0.8}
            }
            
            params = structure_params.get(structure_type, {'decay_length': 500.0, 'max_influence': 0.6})
            distance_influence = params['max_influence'] * np.exp(-min_distance / params['decay_length'])
            
            # Advanced orientation analysis using 3D vector mechanics
            orientation_influence = self._calculate_advanced_orientation_influence(
                feature, pathway_array, strike, dip
            )
            
            # Stress-dependent permeability enhancement
            stress_factor = self._calculate_stress_permeability_coupling(
                stress_orientation, strike, dip, perm_enhancement
            )
            permeability_influence = min(np.log10(stress_factor + 1) / 4.0, 1.0)
            
            # Age-dependent influence with reactivation potential
            age_influence = self._calculate_age_reactivation_influence(age_ma, structure_type)
            
            # Connectivity analysis using network topology
            connectivity_influence = self._calculate_structure_connectivity_influence(
                feature, pathway_array, intersection_points, proximity_segments
            )
            
            # Fluid flow enhancement factor based on aperture and roughness
            flow_enhancement = self._calculate_flow_enhancement_factor(
                feature, min_distance, orientation_influence
            )
            
            # Multi-physics coupling effects
            thermal_influence = self._calculate_thermal_influence(feature, pathway_array)
            chemical_influence = self._calculate_chemical_alteration_influence(feature, age_ma)
            
            # Advanced weighted combination using information theory
            influences = np.array([
                distance_influence,
                orientation_influence, 
                permeability_influence,
                age_influence,
                connectivity_influence,
                flow_enhancement,
                thermal_influence,
                chemical_influence
            ])
            
            # Entropy-based weighting (higher entropy = more uncertainty = lower weight)
            entropies = np.array([0.2, 0.15, 0.25, 0.3, 0.1, 0.2, 0.35, 0.4])  # Estimated uncertainties
            weights = (1.0 - entropies) / np.sum(1.0 - entropies)
            
            # Combined influence using weighted geometric mean for robustness
            influences_safe = np.maximum(influences, 0.01)  # Avoid zeros
            influence_score = np.prod(influences_safe ** weights)
            
            # Uncertainty quantification using Monte Carlo principles
            uncertainty_components = {
                'geometric_uncertainty': 0.1 * (1.0 - orientation_influence),
                'distance_uncertainty': 0.05 * min_distance / params['decay_length'],
                'permeability_uncertainty': 0.2 * (1.0 - permeability_influence),
                'age_uncertainty': 0.15 * min(age_ma / 50.0, 1.0),
                'connectivity_uncertainty': 0.1 * (1.0 - connectivity_influence),
                'flow_uncertainty': 0.12 * (1.0 - flow_enhancement)
            }
            
            total_uncertainty = np.sqrt(sum(unc**2 for unc in uncertainty_components.values()))
            
            # Confidence intervals
            confidence_95_lower = influence_score * (1.0 - 1.96 * total_uncertainty)
            confidence_95_upper = influence_score * (1.0 + 1.96 * total_uncertainty)
            confidence_95 = (confidence_95_lower, confidence_95_upper)
            
            return {
                'influence_score': float(influence_score),
                'distance_influence': float(distance_influence),
                'orientation_influence': float(orientation_influence),
                'permeability_influence': float(permeability_influence),
                'age_influence': float(age_influence),
                'connectivity_influence': float(connectivity_influence),
                'flow_enhancement': float(flow_enhancement),
                'thermal_influence': float(thermal_influence),
                'chemical_influence': float(chemical_influence),
                'stress_factor': float(stress_factor),
                'min_distance': float(min_distance),
                'intersection_count': len(intersection_points),
                'proximity_segments': len(proximity_segments),
                'structure_type': structure_type,
                'uncertainty_components': uncertainty_components,
                'total_uncertainty': float(total_uncertainty),
                'confidence_interval_95': {
                    'lower': float(max(0.0, influence_score - 1.96 * total_uncertainty * influence_score)),
                    'upper': float(min(1.0, influence_score + 1.96 * total_uncertainty * influence_score))
                },
                'weights_used': weights.tolist(),
                'individual_influences': influences.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze structural control influence: {e}")
            return {'influence_score': 0.0}
    
    def _calculate_advanced_orientation_influence(self, feature: Dict[str, Any], 
                                                pathway_array: np.ndarray, 
                                                strike: float, dip: float) -> float:
        """Calculate advanced orientation-based influence using 3D vector mechanics and flow dynamics"""
        try:
            # Calculate 3D structural vectors
            strike_vector = np.array([np.cos(strike), np.sin(strike), 0.0])
            dip_vector = np.array([
                -np.sin(strike) * np.cos(dip),
                np.cos(strike) * np.cos(dip),
                np.sin(dip)
            ])
            normal_vector = np.cross(strike_vector, dip_vector)
            
            if len(pathway_array) < 2:
                return 0.5
            
            # Analyze pathway segments individually for detailed orientation analysis
            segment_influences = []
            pathway_segments = np.diff(pathway_array, axis=0)
            
            for segment in pathway_segments:
                segment_length = np.linalg.norm(segment)
                if segment_length > 0:
                    segment_unit = segment / segment_length
                    
                    # Calculate multiple orientation metrics
                    
                    # 1. Strike-parallel alignment (optimal for lateral flow)
                    strike_alignment = abs(np.dot(segment_unit, strike_vector))
                    
                    # 2. Dip-parallel alignment (optimal for vertical flow)
                    dip_alignment = abs(np.dot(segment_unit, dip_vector))
                    
                    # 3. Normal-perpendicular alignment (flow within fracture plane)
                    normal_perpendicular = 1.0 - abs(np.dot(segment_unit, normal_vector))
                    
                    # 4. Aperture-dependent flow efficiency
                    aperture = feature.get('aperture', 1e-3)  # Default 1mm
                    reynolds_number = self._calculate_reynolds_number(segment_unit, aperture)
                    flow_regime_factor = self._get_flow_regime_factor(reynolds_number)
                    
                    # 5. Roughness effects on flow
                    roughness = feature.get('roughness', 1e-5)  # Default smooth
                    roughness_factor = 1.0 / (1.0 + 10.0 * roughness / aperture)
                    
                    # Combined orientation influence with flow physics
                    segment_influence = (
                        0.4 * max(strike_alignment, dip_alignment) +  # Best directional alignment
                        0.3 * normal_perpendicular +                  # Within fracture plane
                        0.2 * flow_regime_factor +                    # Flow efficiency
                        0.1 * roughness_factor                        # Surface effects
                    )
                    
                    segment_influences.append(segment_influence)
            
            if segment_influences:
                # Weight by segment length for overall pathway influence
                segment_lengths = [np.linalg.norm(seg) for seg in pathway_segments]
                total_length = sum(segment_lengths)
                
                if total_length > 0:
                    weighted_influence = sum(
                        inf * length / total_length 
                        for inf, length in zip(segment_influences, segment_lengths)
                    )
                else:
                    weighted_influence = np.mean(segment_influences)
            else:
                weighted_influence = 0.5
            
            return max(0.0, min(1.0, weighted_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate advanced orientation influence: {e}")
            return 0.5
    
    def _calculate_reynolds_number(self, velocity_vector: np.ndarray, aperture: float) -> float:
        """Calculate Reynolds number for fracture flow"""
        try:
            # Assume typical gas properties at volcanic conditions
            gas_density = 2.0  # kg/m³ (CO2 at elevated T,P)
            gas_viscosity = 2e-5  # Pa·s
            
            velocity_magnitude = np.linalg.norm(velocity_vector)
            if velocity_magnitude == 0:
                return 0.0
            
            # Hydraulic diameter for parallel plates (fracture approximation)
            hydraulic_diameter = 2.0 * aperture
            
            reynolds_number = gas_density * velocity_magnitude * hydraulic_diameter / gas_viscosity
            return reynolds_number
            
        except Exception as e:
            logger.warning(f"Failed to calculate Reynolds number: {e}")
            return 100.0  # Default moderate Re
    
    def _get_flow_regime_factor(self, reynolds_number: float) -> float:
        """Get flow efficiency factor based on Reynolds number"""
        try:
            if reynolds_number < 1.0:  # Creeping flow
                return 0.3
            elif reynolds_number < 100.0:  # Laminar flow
                return 1.0
            elif reynolds_number < 2000.0:  # Transitional
                return 0.8
            else:  # Turbulent flow
                return 0.6
                
        except Exception as e:
            logger.warning(f"Failed to get flow regime factor: {e}")
            return 0.7
    
    def _calculate_stress_permeability_coupling(self, stress_orientation: float, 
                                              strike: float, dip: float, 
                                              base_permeability: float) -> float:
        """Calculate stress-dependent permeability enhancement using geomechanical principles"""
        try:
            # Calculate stress tensor components in fracture coordinate system
            stress_angle_to_strike = abs(stress_orientation - strike)
            stress_angle_to_dip = abs(stress_orientation - (strike + np.pi/2))
            
            # Normalize angles to [0, π/2]
            stress_angle_to_strike = min(stress_angle_to_strike, np.pi - stress_angle_to_strike)
            stress_angle_to_dip = min(stress_angle_to_dip, np.pi - stress_angle_to_dip)
            
            # Stress-induced aperture changes (Barton-Bandis model)
            normal_stress_factor = np.cos(stress_angle_to_strike)**2
            shear_stress_factor = np.sin(2 * stress_angle_to_strike)
            
            # Aperture change due to normal stress (compression reduces aperture)
            normal_stress_effect = 1.0 - 0.5 * normal_stress_factor
            
            # Aperture change due to shear stress (can increase aperture through dilatancy)
            shear_dilatancy_effect = 1.0 + 0.3 * shear_stress_factor
            
            # Combined stress effect on permeability (cubic law: k ∝ aperture³)
            aperture_ratio = normal_stress_effect * shear_dilatancy_effect
            permeability_ratio = aperture_ratio**3
            
            # Include pressure solution effects for long-term changes
            pressure_solution_factor = 0.9  # Slight reduction due to mineral precipitation
            
            enhanced_permeability = base_permeability * permeability_ratio * pressure_solution_factor
            
            return max(0.1, enhanced_permeability)  # Minimum 10% of original
            
        except Exception as e:
            logger.warning(f"Failed to calculate stress-permeability coupling: {e}")
            return base_permeability
    
    def _calculate_age_reactivation_influence(self, age_ma: float, structure_type: str) -> float:
        """Calculate age-dependent influence considering reactivation potential"""
        try:
            if age_ma <= 0:
                return 1.0  # Recent/active structure
            
            # Structure-specific aging parameters
            aging_params = {
                'major_fault': {'half_life': 50.0, 'reactivation_potential': 0.8},
                'fault': {'half_life': 20.0, 'reactivation_potential': 0.6},
                'fracture': {'half_life': 5.0, 'reactivation_potential': 0.4},
                'joint': {'half_life': 2.0, 'reactivation_potential': 0.2},
                'shear_zone': {'half_life': 100.0, 'reactivation_potential': 0.9}
            }
            
            params = aging_params.get(structure_type, {'half_life': 10.0, 'reactivation_potential': 0.5})
            
            # Exponential decay with reactivation potential
            base_decay = np.exp(-age_ma / params['half_life'])
            reactivation_boost = params['reactivation_potential'] * np.exp(-age_ma / 100.0)
            
            age_influence = base_decay + reactivation_boost
            
            return max(0.05, min(1.0, age_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate age reactivation influence: {e}")
            return 0.5
    
    def _calculate_structure_connectivity_influence(self, feature: Dict[str, Any], 
                                                  pathway_array: np.ndarray,
                                                  intersection_points: List,
                                                  proximity_segments: List) -> float:
        """Calculate connectivity influence using network topology analysis"""
        try:
            # Base connectivity from intersections and proximity
            intersection_score = min(len(intersection_points) / 5.0, 1.0)  # Normalize to max 5 intersections
            proximity_score = min(len(proximity_segments) / 10.0, 1.0)    # Normalize to max 10 segments
            
            # Network connectivity analysis
            if self.fracture_network:
                structure_id = feature.get('structure_id', 'unknown')
                
                if structure_id in self.fracture_network.nodes():
                    # Calculate network centrality measures
                    degree_centrality = self.fracture_network.degree(structure_id) / max(1, len(self.fracture_network.nodes()) - 1)
                    
                    # Betweenness centrality (how often structure is on shortest paths)
                    try:
                        betweenness = nx.betweenness_centrality(self.fracture_network).get(structure_id, 0.0)
                    except:
                        betweenness = 0.0
                    
                    # Clustering coefficient (local connectivity)
                    try:
                        clustering = nx.clustering(self.fracture_network, structure_id)
                    except:
                        clustering = 0.0
                    
                    network_connectivity = 0.5 * degree_centrality + 0.3 * betweenness + 0.2 * clustering
                else:
                    network_connectivity = 0.1  # Isolated structure
            else:
                network_connectivity = 0.5  # No network data
            
            # Combined connectivity influence
            connectivity_influence = (
                0.4 * intersection_score +
                0.3 * proximity_score +
                0.3 * network_connectivity
            )
            
            return max(0.0, min(1.0, connectivity_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate structure connectivity influence: {e}")
            return 0.5
    
    def _calculate_flow_enhancement_factor(self, feature: Dict[str, Any], 
                                         min_distance: float, 
                                         orientation_influence: float) -> float:
        """Calculate flow enhancement factor based on fracture properties and geometry"""
        try:
            aperture = feature.get('aperture', 1e-3)  # m
            length = feature.get('length', 100.0)     # m
            roughness = feature.get('roughness', 1e-5) # m
            
            # Aperture-based flow enhancement (cubic law)
            aperture_factor = (aperture / 1e-3)**2  # Normalized to 1mm reference
            
            # Length-based connectivity enhancement
            length_factor = min(np.log10(length / 10.0), 2.0)  # Normalized to 10m reference
            
            # Distance-dependent effectiveness
            distance_factor = np.exp(-min_distance / 200.0)  # 200m characteristic distance
            
            # Roughness penalty
            roughness_penalty = 1.0 / (1.0 + 100.0 * roughness / aperture)
            
            # Orientation bonus
            orientation_bonus = 1.0 + 0.5 * orientation_influence
            
            flow_enhancement = (
                aperture_factor * length_factor * distance_factor * 
                roughness_penalty * orientation_bonus
            )
            
            return max(0.1, min(5.0, flow_enhancement))  # Reasonable bounds
            
        except Exception as e:
            logger.warning(f"Failed to calculate flow enhancement factor: {e}")
            return 1.0
    
    def _calculate_thermal_influence(self, feature: Dict[str, Any], 
                                   pathway_array: np.ndarray) -> float:
        """Calculate thermal influence on gas migration through structural features"""
        try:
            # Depth-dependent temperature gradient
            surface_temp = 15.0  # °C
            geothermal_gradient = 25.0  # °C/km
            
            pathway_depths = pathway_array[:, 2]
            avg_depth = np.mean(pathway_depths)
            avg_temperature = surface_temp + geothermal_gradient * avg_depth / 1000.0
            
            # Temperature effects on gas properties
            # Gas viscosity decreases with temperature (enhances flow)
            temp_kelvin = avg_temperature + 273.15
            viscosity_ratio = (temp_kelvin / 288.15)**(-0.7)  # Power law approximation
            
            # Gas density decreases with temperature (affects buoyancy)
            density_ratio = 288.15 / temp_kelvin
            
            # Thermal expansion of fractures
            thermal_expansion_coeff = 1e-5  # /K for rock
            temp_diff = avg_temperature - surface_temp
            thermal_expansion = 1.0 + thermal_expansion_coeff * temp_diff
            
            # Combined thermal influence
            thermal_influence = (
                0.4 * viscosity_ratio +      # Flow enhancement
                0.3 * (1.0 / density_ratio) + # Buoyancy effects  
                0.3 * thermal_expansion      # Aperture enhancement
            )
            
            return max(0.5, min(2.0, thermal_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate thermal influence: {e}")
            return 1.0
    
    def _calculate_chemical_alteration_influence(self, feature: Dict[str, Any], age_ma: float) -> float:
        """Calculate chemical alteration influence on fracture permeability"""
        try:
            # Chemical alteration processes
            
            # 1. Mineral precipitation (reduces permeability)
            precipitation_rate = 0.1 * age_ma / 10.0  # Normalized to 10 Ma
            precipitation_factor = np.exp(-precipitation_rate)
            
            # 2. Dissolution (increases permeability)
            dissolution_rate = 0.05 * age_ma / 10.0
            dissolution_factor = 1.0 + dissolution_rate
            
            # 3. Hydrothermal alteration (variable effects)
            hydrothermal_intensity = feature.get('hydrothermal_alteration', 0.0)  # 0-1 scale
            if hydrothermal_intensity > 0.5:
                # High alteration can seal fractures
                alteration_factor = 0.5 + 0.5 * (1.0 - hydrothermal_intensity)
            else:
                # Low alteration may enhance permeability
                alteration_factor = 1.0 + 0.5 * hydrothermal_intensity
            
            # 4. Clay mineral formation
            clay_content = feature.get('clay_content', 0.0)  # 0-1 scale
            clay_factor = 1.0 / (1.0 + 10.0 * clay_content)
            
            # Combined chemical influence
            chemical_influence = (
                precipitation_factor * dissolution_factor * 
                alteration_factor * clay_factor
            )
            
            return max(0.1, min(2.0, chemical_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate chemical alteration influence: {e}")
            return 1.0
    
    def _calculate_structure_connectivity(self, feature: Dict[str, Any], 
                                        pathway_array: np.ndarray) -> float:
        """Calculate structure connectivity factor using advanced network analysis"""
        try:
            if not self.fracture_network:
                return 0.5
            
            structure_id = feature.get('structure_id', feature.get('fracture_id', 'unknown'))
            
            if structure_id not in self.fracture_network.nodes():
                return 0.1  # Isolated structure
            
            # Advanced network metrics
            node_degree = self.fracture_network.degree(structure_id)
            max_degree = max(dict(self.fracture_network.degree()).values()) if self.fracture_network.nodes() else 1
            degree_centrality = node_degree / max_degree
            
            # Calculate effective connectivity considering edge weights
            weighted_degree = sum(
                self.fracture_network[structure_id][neighbor].get('flow_capacity', 1.0)
                for neighbor in self.fracture_network.neighbors(structure_id)
            )
            
            # Normalize weighted degree
            max_weighted_degree = max([
                sum(self.fracture_network[node][neighbor].get('flow_capacity', 1.0)
                    for neighbor in self.fracture_network.neighbors(node))
                for node in self.fracture_network.nodes()
            ]) if self.fracture_network.nodes() else 1.0
            
            weighted_centrality = weighted_degree / max_weighted_degree
            
            # Path diversity (number of alternative paths)
            try:
                # Find paths to surface nodes (shallow structures)
                surface_nodes = [
                    node for node in self.fracture_network.nodes()
                    if self.fracture_network.nodes[node]['position'][2] < 500.0  # Within 500m of surface
                ]
                
                if surface_nodes:
                    path_count = 0
                    for surface_node in surface_nodes[:5]:  # Limit to 5 surface nodes for efficiency
                        try:
                            paths = list(nx.all_simple_paths(
                                self.fracture_network, structure_id, surface_node, cutoff=10
                            ))
                            path_count += len(paths)
                        except:
                            continue
                    
                    path_diversity = min(path_count / 10.0, 1.0)  # Normalize to max 10 paths
                else:
                    path_diversity = 0.0
            except:
                path_diversity = 0.0
            
            # Combined connectivity factor
            connectivity_factor = (
                0.4 * degree_centrality +
                0.4 * weighted_centrality +
                0.2 * path_diversity
            )
            
            return max(0.1, min(1.0, connectivity_factor))
            
        except Exception as e:
            logger.warning(f"Failed to calculate structure connectivity: {e}")
            return 0.5
    
    def _calculate_fracture_influence(self, fracture_data: Dict[str, Any], 
                                    pathway_point: np.ndarray, distance: float) -> float:
        """Calculate fracture influence on gas migration using advanced multi-physics modeling"""
        try:
            # Extract comprehensive fracture properties
            aperture = fracture_data.get('aperture', 1e-3)  # m
            permeability = fracture_data.get('permeability', 1e-12)  # m²
            strike = np.radians(fracture_data.get('strike', 0.0))
            dip = np.radians(fracture_data.get('dip', 90.0))
            length = fracture_data.get('length', 100.0)  # m
            connectivity = fracture_data.get('connectivity', 0.5)
            roughness = fracture_data.get('roughness', 1e-5)  # m
            filling_material = fracture_data.get('filling_material', 'none')
            
            # Advanced distance-based influence with fracture-specific parameters
            # Larger fractures have longer influence zones
            influence_radius = 50.0 + 200.0 * np.log10(max(length / 10.0, 1.0))
            distance_factor = np.exp(-distance / influence_radius)
            
            # Aperture influence using cubic law for fracture flow
            # Flow rate ∝ aperture³
            aperture_factor = (aperture / 1e-3)**1.5  # Power between linear and cubic
            aperture_factor = min(aperture_factor, 20.0)  # Cap at reasonable value
            
            # Permeability influence with stress-dependent corrections
            depth = pathway_point[2]
            effective_stress = 2700 * 9.81 * depth  # Lithostatic stress (Pa)
            
            # Stress-dependent aperture closure (Barton-Bandis model)
            initial_aperture = aperture
            jrc = 10.0  # Joint roughness coefficient (typical value)
            jcs = 100e6  # Joint compressive strength (Pa)
            
            stress_factor = 1.0 / (1.0 + effective_stress / jcs)
            effective_aperture = initial_aperture * stress_factor
            
            # Effective permeability considering stress effects
            effective_permeability = permeability * (effective_aperture / initial_aperture)**3
            perm_factor = min(np.log10(effective_permeability / 1e-18) / 6.0, 1.0)
            
            # Advanced orientation analysis
            fracture_normal = np.array([
                np.sin(dip) * np.sin(strike + np.pi/2),
                np.sin(dip) * np.cos(strike + np.pi/2),
                np.cos(dip)
            ])
            
            fracture_strike_vector = np.array([
                np.cos(strike),
                -np.sin(strike),
                0.0
            ])
            
            # Assume vertical gas flow for buoyancy-driven migration
            vertical_flow_vector = np.array([0.0, 0.0, 1.0])
            
            # Calculate optimal flow direction within fracture plane
            fracture_dip_vector = np.cross(fracture_normal, fracture_strike_vector)
            
            # Flow efficiency based on alignment with fracture orientation
            strike_alignment = abs(np.dot(vertical_flow_vector, fracture_strike_vector))
            dip_alignment = abs(np.dot(vertical_flow_vector, fracture_dip_vector))
            normal_alignment = abs(np.dot(vertical_flow_vector, fracture_normal))
            
            # Optimal flow is within fracture plane (low normal alignment)
            orientation_factor = max(strike_alignment, dip_alignment) * (1.0 - normal_alignment)
            
            # Connectivity influence using network theory
            connectivity_factor = connectivity**0.5  # Square root to moderate the effect
            
            # Roughness effects on flow (Zimmerman-Bodvarsson model)
            relative_roughness = roughness / effective_aperture
            roughness_factor = 1.0 / (1.0 + 8.8 * relative_roughness**1.5)
            
            # Filling material effects
            filling_factors = {
                'none': 1.0,
                'clay': 0.1,
                'calcite': 0.3,
                'quartz': 0.7,
                'gouge': 0.05,
                'breccia': 0.6
            }
            filling_factor = filling_factors.get(filling_material, 0.5)
            
            # Thermal effects on fracture properties
            temperature = 15.0 + 25.0 * depth / 1000.0  # Geothermal gradient
            thermal_expansion = 1.0 + 1e-5 * (temperature - 15.0)  # Rock thermal expansion
            
            # Gas property changes with temperature and pressure
            pressure = 101325 + 2700 * 9.81 * depth  # Hydrostatic + lithostatic
            gas_density = self._calculate_gas_density(temperature, pressure)
            gas_viscosity = self._calculate_gas_viscosity(temperature, pressure)
            
            # Reynolds number for flow regime determination
            characteristic_velocity = 1e-6  # m/s (typical gas migration velocity)
            reynolds_number = gas_density * characteristic_velocity * effective_aperture / gas_viscosity
            
            # Flow regime factor
            if reynolds_number < 1.0:
                flow_regime_factor = 1.0  # Stokes flow (optimal)
            elif reynolds_number < 100.0:
                flow_regime_factor = 0.9  # Laminar flow
            else:
                flow_regime_factor = 0.7  # Turbulent flow (less efficient)
            
            # Fracture maturity effects (older fractures may be more sealed)
            age_factor = 1.0  # Assume young/active fracture if no age data
            
            # Combined influence using weighted geometric mean for robustness
            factors = np.array([
                distance_factor,
                aperture_factor / 20.0,  # Normalize
                perm_factor,
                orientation_factor,
                connectivity_factor,
                roughness_factor,
                filling_factor,
                thermal_expansion,
                flow_regime_factor,
                age_factor
            ])
            
            # Weights based on physical importance and uncertainty
            weights = np.array([0.25, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05, 0.03, 0.04, 0.03])
            
            # Ensure no zero values for geometric mean
            factors_safe = np.maximum(factors, 0.01)
            
            # Calculate influence using weighted geometric mean
            influence = np.prod(factors_safe ** weights)
            
            # Apply physical bounds and scaling
            influence = max(0.0, min(1.0, influence))
            
            return float(influence)
            
        except Exception as e:
            logger.warning(f"Failed to calculate fracture influence: {e}")
            return 0.0
    
    def _calculate_gas_density(self, temperature: float, pressure: float) -> float:
        """Calculate gas density using equation of state"""
        try:
            # Assume CO2 as primary gas component
            # Using ideal gas law with compressibility correction
            R = 8.314  # J/(mol·K)
            M = 0.044  # kg/mol (CO2 molar mass)
            T = temperature + 273.15  # Convert to Kelvin
            
            # Compressibility factor for CO2 (simplified)
            Z = 1.0 - 0.1 * pressure / 1e6  # Approximate correction
            
            density = pressure * M / (Z * R * T)
            return max(0.1, density)  # Minimum reasonable density
            
        except Exception as e:
            logger.warning(f"Failed to calculate gas density: {e}")
            return 2.0  # Default CO2 density
    
    def _calculate_gas_viscosity(self, temperature: float, pressure: float) -> float:
        """Calculate gas viscosity using Sutherland's law"""
        try:
            # Sutherland's law for CO2
            T = temperature + 273.15  # Convert to Kelvin
            T0 = 273.15  # Reference temperature
            mu0 = 1.37e-5  # Reference viscosity (Pa·s)
            S = 240.0  # Sutherland constant for CO2
            
            viscosity = mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)
            
            # Pressure correction (approximate)
            pressure_factor = 1.0 + 1e-8 * pressure
            
            return viscosity * pressure_factor
            
        except Exception as e:
            logger.warning(f"Failed to calculate gas viscosity: {e}")
            return 2e-5  # Default gas viscosity
    
    async def _identify_lithological_controls(self, pathway_array: np.ndarray) -> List[str]:
        """Identify lithological controls along pathway using advanced petrophysical analysis"""
        controls = []
        
        try:
            # Advanced multi-table query for comprehensive lithological analysis
            lithology_query = """
            SELECT rp.rock_type, rp.porosity, rp.permeability, rp.alteration_degree, 
                   rp.formation_name, rp.age_ma, rp.grain_size, rp.sorting, rp.cementation,
                   rp.fracture_density, rp.weathering_degree, rp.mineral_composition,
                   gc.sio2_content, gc.al2o3_content, gc.fe2o3_content, gc.cao_content,
                   gc.mgo_content, gc.k2o_content, gc.na2o_content,
                   ST_AsText(rp.geometry) as geometry_wkt,
                   ST_Distance(rp.geometry, ST_MakeLine(ARRAY[%s])) as distance
            FROM rock_properties rp
            LEFT JOIN geochemical_analysis gc ON rp.sample_id = gc.sample_id
            WHERE ST_DWithin(rp.geometry, ST_MakeLine(ARRAY[%s]), %s)
            ORDER BY ST_Distance(rp.geometry, ST_MakeLine(ARRAY[%s]))
            """
            
            # Create pathway linestring for spatial analysis
            pathway_linestring = f"LINESTRING({', '.join([f'{x} {y} {z}' for x, y, z in pathway_array])})"
            
            try:
                lithology_data = await self.db_manager.execute_query(
                    lithology_query, (pathway_linestring, pathway_linestring, 200.0, pathway_linestring)
                )
            except RuntimeError:
                # Fallback if already in event loop
                lithology_data = []
            
            # Advanced lithological analysis
            for rock_unit in lithology_data:
                # Extract comprehensive rock properties
                rock_type = rock_unit.get('rock_type', 'unknown')
                porosity = rock_unit.get('porosity', 0.1)
                permeability = rock_unit.get('permeability', 1e-15)
                alteration = rock_unit.get('alteration_degree', 0.0)
                fracture_density = rock_unit.get('fracture_density', 0.0)
                weathering = rock_unit.get('weathering_degree', 0.0)
                grain_size = rock_unit.get('grain_size', 'medium')
                sorting = rock_unit.get('sorting', 'moderate')
                cementation = rock_unit.get('cementation', 'moderate')
                distance = rock_unit.get('distance', 0.0)
                
                # Advanced petrophysical modeling
                petrophysical_analysis = self._perform_advanced_petrophysical_analysis(rock_unit)
                
                # Multi-scale permeability analysis
                matrix_permeability = permeability
                fracture_permeability = self._calculate_fracture_permeability(
                    fracture_density, rock_type, alteration
                )
                effective_permeability = self._calculate_effective_permeability(
                    matrix_permeability, fracture_permeability, porosity
                )
                
                # Geochemical influence on gas-rock interactions
                geochemical_reactivity = self._calculate_geochemical_reactivity(rock_unit)
                
                # Depth-dependent property evolution
                depth_effects = self._calculate_depth_dependent_properties(
                    pathway_array, rock_unit, distance
                )
                
                # Anisotropy analysis for directional permeability
                permeability_anisotropy = self._calculate_permeability_anisotropy(
                    rock_type, grain_size, sorting, fracture_density
                )
                
                # Capillary pressure and relative permeability effects
                capillary_effects = self._calculate_capillary_effects(
                    porosity, permeability, grain_size, sorting
                )
                
                # Thermal effects on rock properties
                thermal_effects = self._calculate_thermal_rock_effects(
                    pathway_array, rock_unit
                )
                
                # Calculate comprehensive lithological favorability
                favorability_components = {
                    'matrix_permeability': self._normalize_permeability_score(matrix_permeability),
                    'fracture_permeability': self._normalize_permeability_score(fracture_permeability),
                    'effective_permeability': self._normalize_permeability_score(effective_permeability),
                    'porosity_score': self._calculate_porosity_score(porosity),
                    'alteration_score': self._calculate_alteration_score(alteration, rock_type),
                    'geochemical_score': geochemical_reactivity,
                    'anisotropy_score': permeability_anisotropy,
                    'capillary_score': capillary_effects,
                    'thermal_score': thermal_effects,
                    'distance_score': np.exp(-distance / 100.0)  # Distance decay
                }
                
                # Weighted combination of favorability components
                weights = {
                    'matrix_permeability': 0.2,
                    'fracture_permeability': 0.25,
                    'effective_permeability': 0.15,
                    'porosity_score': 0.1,
                    'alteration_score': 0.1,
                    'geochemical_score': 0.05,
                    'anisotropy_score': 0.05,
                    'capillary_score': 0.05,
                    'thermal_score': 0.03,
                    'distance_score': 0.02
                }
                
                overall_favorability = sum(
                    favorability_components[component] * weights[component]
                    for component in favorability_components
                )
                
                # Advanced uncertainty quantification
                uncertainty_factors = self._calculate_lithological_uncertainty(
                    rock_unit, petrophysical_analysis, distance
                )
                
                if overall_favorability > 0.3:  # Significant lithological control threshold
                    # Create detailed control description with advanced metrics
                    control_desc = (
                        f"lithology_{rock_type}_formation_{rock_unit.get('formation_name', 'unknown')}_"
                        f"matrix_perm{matrix_permeability:.2e}_fracture_perm{fracture_permeability:.2e}_"
                        f"eff_perm{effective_permeability:.2e}_porosity{porosity:.3f}_"
                        f"alteration{alteration:.2f}_fracture_density{fracture_density:.2f}_"
                        f"geochemical_reactivity{geochemical_reactivity:.2f}_"
                        f"anisotropy{permeability_anisotropy:.2f}_"
                        f"favorability{overall_favorability:.3f}_"
                        f"uncertainty{uncertainty_factors['total_uncertainty']:.3f}_"
                        f"distance{distance:.1f}m"
                    )
                    controls.append(control_desc)
                    
                    # Add specific rock-type controls
                    rock_specific_controls = self._identify_rock_specific_controls(
                        rock_type, petrophysical_analysis, overall_favorability
                    )
                    controls.extend(rock_specific_controls)
            
            # Add regional lithological trends
            regional_controls = await self._identify_regional_lithological_trends(
                pathway_array, lithology_data
            )
            controls.extend(regional_controls)
            
        except Exception as e:
            logger.warning(f"Failed to identify lithological controls: {e}")
        
        return controls
    
    def _perform_advanced_petrophysical_analysis(self, rock_unit: Dict[str, Any]) -> Dict[str, float]:
        """Perform advanced petrophysical analysis of rock unit"""
        try:
            rock_type = rock_unit.get('rock_type', 'unknown')
            porosity = rock_unit.get('porosity', 0.1)
            permeability = rock_unit.get('permeability', 1e-15)
            grain_size = rock_unit.get('grain_size', 'medium')
            sorting = rock_unit.get('sorting', 'moderate')
            cementation = rock_unit.get('cementation', 'moderate')
            
            # Kozeny-Carman permeability estimation
            kozeny_carman_perm = self._calculate_kozeny_carman_permeability(
                porosity, grain_size, sorting
            )
            
            # Archie's law for electrical properties (proxy for pore connectivity)
            archie_factor = self._calculate_archie_factor(porosity, cementation)
            
            # Pore throat size distribution analysis
            pore_throat_analysis = self._analyze_pore_throat_distribution(
                porosity, permeability, grain_size
            )
            
            # Tortuosity calculation
            tortuosity = self._calculate_tortuosity(porosity, rock_type, cementation)
            
            # Specific surface area estimation
            specific_surface_area = self._calculate_specific_surface_area(
                porosity, grain_size, rock_type
            )
            
            return {
                'kozeny_carman_permeability': kozeny_carman_perm,
                'archie_factor': archie_factor,
                'mean_pore_throat_size': pore_throat_analysis['mean_size'],
                'pore_throat_heterogeneity': pore_throat_analysis['heterogeneity'],
                'tortuosity': tortuosity,
                'specific_surface_area': specific_surface_area,
                'pore_connectivity': archie_factor / tortuosity if tortuosity > 0 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Failed to perform petrophysical analysis: {e}")
            return {}
    
    def _calculate_fracture_permeability(self, fracture_density: float, 
                                       rock_type: str, alteration: float) -> float:
        """Calculate fracture-controlled permeability"""
        try:
            # Base fracture permeability from density
            if fracture_density <= 0:
                return 1e-18  # Minimal fracture permeability
            
            # Rock-type specific fracture characteristics
            fracture_params = {
                'granite': {'base_aperture': 1e-4, 'connectivity': 0.3},
                'basalt': {'base_aperture': 5e-5, 'connectivity': 0.4},
                'limestone': {'base_aperture': 2e-4, 'connectivity': 0.6},
                'sandstone': {'base_aperture': 1e-4, 'connectivity': 0.5},
                'shale': {'base_aperture': 1e-5, 'connectivity': 0.2},
                'volcanic': {'base_aperture': 1e-4, 'connectivity': 0.7}
            }
            
            params = fracture_params.get(rock_type, {'base_aperture': 1e-4, 'connectivity': 0.4})
            
            # Alteration effects on fracture aperture
            alteration_factor = 1.0 + 2.0 * alteration  # Alteration can increase aperture
            effective_aperture = params['base_aperture'] * alteration_factor
            
            # Fracture permeability using cubic law
            # k_fracture = (aperture³ × density × connectivity) / 12
            fracture_permeability = (
                effective_aperture**3 * fracture_density * params['connectivity'] / 12.0
            )
            
            return max(1e-18, fracture_permeability)
            
        except Exception as e:
            logger.warning(f"Failed to calculate fracture permeability: {e}")
            return 1e-15
    
    def _calculate_effective_permeability(self, matrix_perm: float, 
                                        fracture_perm: float, porosity: float) -> float:
        """Calculate effective permeability combining matrix and fracture contributions"""
        try:
            # Parallel flow model for fractured media
            # k_eff = k_matrix × φ_matrix + k_fracture × φ_fracture
            
            # Estimate fracture porosity (typically much smaller than matrix porosity)
            fracture_porosity = min(0.01, porosity * 0.1)  # Assume 10% of total porosity
            matrix_porosity = porosity - fracture_porosity
            
            # Weighted effective permeability
            effective_permeability = (
                matrix_perm * matrix_porosity + fracture_perm * fracture_porosity
            ) / porosity
            
            # Apply connectivity correction
            connectivity_factor = 0.5 + 0.5 * (fracture_porosity / porosity)
            
            return effective_permeability * connectivity_factor
            
        except Exception as e:
            logger.warning(f"Failed to calculate effective permeability: {e}")
            return max(matrix_perm, fracture_perm)
    
    def _calculate_geochemical_reactivity(self, rock_unit: Dict[str, Any]) -> float:
        """Calculate geochemical reactivity affecting gas-rock interactions"""
        try:
            # Extract geochemical composition
            sio2 = rock_unit.get('sio2_content', 60.0)  # Default intermediate composition
            al2o3 = rock_unit.get('al2o3_content', 15.0)
            fe2o3 = rock_unit.get('fe2o3_content', 5.0)
            cao = rock_unit.get('cao_content', 5.0)
            mgo = rock_unit.get('mgo_content', 3.0)
            k2o = rock_unit.get('k2o_content', 3.0)
            na2o = rock_unit.get('na2o_content', 4.0)
            
            # Calculate reactivity indices
            
            # 1. Mafic index (higher = more reactive with CO2)
            mafic_index = (fe2o3 + mgo + cao) / (sio2 + al2o3)
            
            # 2. Alkali index (affects pH buffering)
            alkali_index = (k2o + na2o) / (sio2 + al2o3)
            
            # 3. Weathering susceptibility
            weathering_index = (cao + mgo + na2o + k2o) / (sio2 + al2o3 + fe2o3)
            
            # Combined reactivity score
            reactivity_score = (
                0.5 * min(mafic_index / 0.3, 1.0) +      # Normalize to typical basalt
                0.3 * min(alkali_index / 0.2, 1.0) +     # Normalize to typical granite
                0.2 * min(weathering_index / 0.4, 1.0)   # Normalize to typical values
            )
            
            return max(0.0, min(1.0, reactivity_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate geochemical reactivity: {e}")
            return 0.5
    
    def _calculate_depth_dependent_properties(self, pathway_array: np.ndarray, 
                                            rock_unit: Dict[str, Any], 
                                            distance: float) -> Dict[str, float]:
        """Calculate depth-dependent evolution of rock properties"""
        try:
            # Average pathway depth
            avg_depth = np.mean(pathway_array[:, 2])
            
            # Pressure effects on porosity (mechanical compaction)
            effective_stress = 2700 * 9.81 * avg_depth  # Pa
            initial_porosity = rock_unit.get('porosity', 0.1)
            
            # Athy's law for porosity-depth relationship
            compaction_coefficient = 3e-4  # m⁻¹ (typical for sedimentary rocks)
            depth_corrected_porosity = initial_porosity * np.exp(-compaction_coefficient * avg_depth)
            
            # Temperature effects
            temperature = 15.0 + 25.0 * avg_depth / 1000.0  # °C
            
            # Thermal effects on mineral stability and alteration
            thermal_alteration_rate = np.exp((temperature - 15.0) / 50.0) - 1.0
            
            # Pressure solution effects on permeability
            pressure_solution_factor = np.exp(-effective_stress / 50e6)  # 50 MPa characteristic stress
            
            return {
                'depth_corrected_porosity': depth_corrected_porosity,
                'thermal_alteration_rate': thermal_alteration_rate,
                'pressure_solution_factor': pressure_solution_factor,
                'effective_stress': effective_stress,
                'temperature': temperature
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate depth-dependent properties: {e}")
            return {}
    
    def _calculate_permeability_anisotropy(self, rock_type: str, grain_size: str, 
                                         sorting: str, fracture_density: float) -> float:
        """Calculate permeability anisotropy factor"""
        try:
            # Base anisotropy from rock type
            anisotropy_factors = {
                'shale': 0.1,      # Highly anisotropic
                'slate': 0.2,
                'schist': 0.3,
                'sandstone': 0.7,  # Moderately anisotropic
                'limestone': 0.8,
                'granite': 0.9,    # Nearly isotropic
                'basalt': 0.8
            }
            
            base_anisotropy = anisotropy_factors.get(rock_type, 0.6)
            
            # Grain size effects
            grain_size_factors = {
                'very_fine': 0.8,
                'fine': 0.9,
                'medium': 1.0,
                'coarse': 1.1,
                'very_coarse': 1.2
            }
            
            grain_factor = grain_size_factors.get(grain_size, 1.0)
            
            # Sorting effects
            sorting_factors = {
                'very_poor': 0.7,
                'poor': 0.8,
                'moderate': 1.0,
                'good': 1.1,
                'very_good': 1.2
            }
            
            sorting_factor = sorting_factors.get(sorting, 1.0)
            
            # Fracture density effects (fractures can increase anisotropy)
            fracture_factor = 1.0 - 0.3 * min(fracture_density / 10.0, 1.0)
            
            # Combined anisotropy
            anisotropy = base_anisotropy * grain_factor * sorting_factor * fracture_factor
            
            return max(0.1, min(1.0, anisotropy))
            
        except Exception as e:
            logger.warning(f"Failed to calculate permeability anisotropy: {e}")
            return 0.6
    
    def _calculate_capillary_effects(self, porosity: float, permeability: float, 
                                   grain_size: str, sorting: str) -> float:
        """Calculate capillary pressure effects on gas migration"""
        try:
            # Estimate pore throat radius from Kozeny-Carman relationship
            grain_size_values = {
                'very_fine': 0.1e-3,   # mm
                'fine': 0.25e-3,
                'medium': 0.5e-3,
                'coarse': 1.0e-3,
                'very_coarse': 2.0e-3
            }
            
            grain_diameter = grain_size_values.get(grain_size, 0.5e-3)  # m
            
            # Pore throat radius estimation
            pore_throat_radius = 0.1 * grain_diameter * (porosity / 0.2)**0.5
            
            # Capillary pressure for gas-water system
            surface_tension = 0.072  # N/m (water-air at 20°C)
            contact_angle = 30.0     # degrees (typical for gas-water-rock)
            
            capillary_pressure = 2 * surface_tension * np.cos(np.radians(contact_angle)) / pore_throat_radius
            
            # Capillary number (ratio of viscous to capillary forces)
            gas_velocity = 1e-6  # m/s (typical)
            gas_viscosity = 2e-5  # Pa·s
            
            capillary_number = gas_viscosity * gas_velocity / surface_tension
            
            # Capillary effects on flow efficiency
            if capillary_number > 1e-6:
                capillary_efficiency = 1.0  # Viscous forces dominate
            else:
                capillary_efficiency = 0.5 + 0.5 * (capillary_number / 1e-6)
            
            return max(0.1, min(1.0, capillary_efficiency))
            
        except Exception as e:
            logger.warning(f"Failed to calculate capillary effects: {e}")
            return 0.7
    
    def _calculate_thermal_rock_effects(self, pathway_array: np.ndarray, 
                                      rock_unit: Dict[str, Any]) -> float:
        """Calculate thermal effects on rock properties"""
        try:
            avg_depth = np.mean(pathway_array[:, 2])
            temperature = 15.0 + 25.0 * avg_depth / 1000.0  # °C
            
            rock_type = rock_unit.get('rock_type', 'unknown')
            
            # Thermal expansion coefficients (per °C)
            thermal_expansion_coeffs = {
                'granite': 8e-6,
                'basalt': 6e-6,
                'limestone': 5e-6,
                'sandstone': 12e-6,
                'shale': 10e-6
            }
            
            expansion_coeff = thermal_expansion_coeffs.get(rock_type, 8e-6)
            
            # Thermal expansion effects on porosity and permeability
            temp_diff = temperature - 15.0  # Reference temperature
            thermal_strain = expansion_coeff * temp_diff
            
            # Thermal effects on mineral stability
            alteration_rate = np.exp(temp_diff / 50.0) - 1.0  # Arrhenius-type relationship
            
            # Combined thermal effects
            thermal_enhancement = 1.0 + 0.5 * thermal_strain + 0.2 * alteration_rate
            
            return max(0.5, min(2.0, thermal_enhancement))
            
        except Exception as e:
            logger.warning(f"Failed to calculate thermal rock effects: {e}")
            return 1.0
    
    def _normalize_permeability_score(self, permeability: float) -> float:
        """Normalize permeability to 0-1 score"""
        try:
            log_perm = np.log10(max(permeability, 1e-20))
            # Scale from 1e-18 (score=0) to 1e-10 (score=1)
            score = (log_perm + 18.0) / 8.0
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_porosity_score(self, porosity: float) -> float:
        """Calculate porosity favorability score"""
        try:
            if porosity < 0.02:
                return porosity / 0.02
            elif porosity < 0.3:
                return 1.0
            else:
                return max(0.3, 1.0 - (porosity - 0.3) / 0.2)
        except:
            return 0.5
    
    def _calculate_alteration_score(self, alteration: float, rock_type: str) -> float:
        """Calculate alteration favorability score"""
        try:
            # Different rock types respond differently to alteration
            if rock_type in ['granite', 'basalt', 'volcanic']:
                # Igneous rocks: moderate alteration enhances permeability
                if alteration < 0.5:
                    return 0.5 + alteration
                else:
                    return max(0.2, 1.0 - (alteration - 0.5) / 0.5)
            else:
                # Sedimentary rocks: less alteration generally better
                return max(0.2, 1.0 - alteration)
        except:
            return 0.5
    
    def _calculate_kozeny_carman_permeability(self, porosity: float, 
                                            grain_size: str, sorting: str) -> float:
        """Calculate permeability using Kozeny-Carman equation"""
        try:
            # Grain size to diameter mapping
            grain_diameters = {
                'very_fine': 0.1e-3,
                'fine': 0.25e-3, 
                'medium': 0.5e-3,
                'coarse': 1.0e-3,
                'very_coarse': 2.0e-3
            }
            
            d = grain_diameters.get(grain_size, 0.5e-3)  # m
            
            # Sorting factor (affects Kozeny constant)
            sorting_factors = {
                'very_poor': 150,
                'poor': 120,
                'moderate': 100,
                'good': 80,
                'very_good': 60
            }
            
            kozeny_constant = sorting_factors.get(sorting, 100)
            
            # Kozeny-Carman equation: k = d² × φ³ / (kozeny_constant × (1-φ)²)
            if porosity >= 0.99:
                porosity = 0.99  # Avoid division by zero
            
            permeability = (d**2 * porosity**3) / (kozeny_constant * (1 - porosity)**2)
            
            return max(1e-20, permeability)
            
        except Exception as e:
            logger.warning(f"Failed to calculate Kozeny-Carman permeability: {e}")
            return 1e-15
    
    def _calculate_archie_factor(self, porosity: float, cementation: str) -> float:
        """Calculate Archie's law factor for pore connectivity"""
        try:
            # Cementation exponent
            cementation_exponents = {
                'very_weak': 1.3,
                'weak': 1.5,
                'moderate': 1.8,
                'strong': 2.0,
                'very_strong': 2.5
            }
            
            m = cementation_exponents.get(cementation, 1.8)
            
            # Archie factor: F = φ^(-m)
            archie_factor = porosity**(-m) if porosity > 0 else 1000
            
            # Normalize to reasonable range
            return min(archie_factor / 100.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate Archie factor: {e}")
            return 0.5
    
    def _analyze_pore_throat_distribution(self, porosity: float, 
                                        permeability: float, grain_size: str) -> Dict[str, float]:
        """Analyze pore throat size distribution"""
        try:
            # Estimate mean pore throat size from permeability-porosity relationship
            if porosity > 0 and permeability > 0:
                # Simplified relationship: r = sqrt(8k/φ)
                mean_throat_size = np.sqrt(8 * permeability / porosity)
            else:
                mean_throat_size = 1e-6  # Default 1 micron
            
            # Heterogeneity based on grain size uniformity
            grain_heterogeneity = {
                'very_fine': 0.2,
                'fine': 0.3,
                'medium': 0.5,
                'coarse': 0.7,
                'very_coarse': 0.9
            }
            
            heterogeneity = grain_heterogeneity.get(grain_size, 0.5)
            
            return {
                'mean_size': mean_throat_size,
                'heterogeneity': heterogeneity
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze pore throat distribution: {e}")
            return {'mean_size': 1e-6, 'heterogeneity': 0.5}
    
    def _calculate_tortuosity(self, porosity: float, rock_type: str, cementation: str) -> float:
        """Calculate pore network tortuosity"""
        try:
            # Base tortuosity from rock type
            base_tortuosity = {
                'sandstone': 2.0,
                'limestone': 2.5,
                'granite': 3.0,
                'basalt': 2.8,
                'shale': 5.0
            }
            
            tau_base = base_tortuosity.get(rock_type, 2.5)
            
            # Porosity effects (higher porosity = lower tortuosity)
            porosity_factor = (0.3 / max(porosity, 0.01))**0.5
            
            # Cementation effects
            cementation_factors = {
                'very_weak': 0.8,
                'weak': 0.9,
                'moderate': 1.0,
                'strong': 1.2,
                'very_strong': 1.5
            }
            
            cement_factor = cementation_factors.get(cementation, 1.0)
            
            tortuosity = tau_base * porosity_factor * cement_factor
            
            return max(1.0, tortuosity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate tortuosity: {e}")
            return 2.5
    
    def _calculate_specific_surface_area(self, porosity: float, 
                                       grain_size: str, rock_type: str) -> float:
        """Calculate specific surface area"""
        try:
            # Grain size to surface area mapping
            grain_surface_areas = {
                'very_fine': 10.0,    # m²/g
                'fine': 5.0,
                'medium': 2.0,
                'coarse': 1.0,
                'very_coarse': 0.5
            }
            
            base_ssa = grain_surface_areas.get(grain_size, 2.0)
            
            # Rock type corrections
            rock_corrections = {
                'shale': 2.0,      # High clay content
                'limestone': 0.8,   # Lower surface area
                'sandstone': 1.0,   # Reference
                'granite': 0.6,     # Crystalline, low surface area
                'basalt': 1.2      # Vesicular texture
            }
            
            correction = rock_corrections.get(rock_type, 1.0)
            
            # Porosity effects
            porosity_factor = 1.0 + porosity
            
            specific_surface_area = base_ssa * correction * porosity_factor
            
            return max(0.1, specific_surface_area)
            
        except Exception as e:
            logger.warning(f"Failed to calculate specific surface area: {e}")
            return 2.0
    
    def _calculate_lithological_uncertainty(self, rock_unit: Dict[str, Any], 
                                          petrophysical_analysis: Dict[str, float],
                                          distance: float) -> Dict[str, float]:
        """Calculate uncertainty in lithological analysis"""
        try:
            # Distance-based uncertainty
            distance_uncertainty = min(distance / 200.0, 1.0)
            
            # Property measurement uncertainty
            porosity_uncertainty = 0.1  # +/-10% typical
            permeability_uncertainty = 0.5  # +/-50% typical (log scale)
            
            # Spatial extrapolation uncertainty
            spatial_uncertainty = 0.2 + 0.3 * distance_uncertainty
            
            # Model uncertainty
            model_uncertainty = 0.15  # Inherent model limitations
            
            # Combined uncertainty
            total_uncertainty = np.sqrt(
                porosity_uncertainty**2 + 
                permeability_uncertainty**2 + 
                spatial_uncertainty**2 + 
                model_uncertainty**2
            )
            
            return {
                'distance_uncertainty': distance_uncertainty,
                'porosity_uncertainty': porosity_uncertainty,
                'permeability_uncertainty': permeability_uncertainty,
                'spatial_uncertainty': spatial_uncertainty,
                'model_uncertainty': model_uncertainty,
                'total_uncertainty': total_uncertainty
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate lithological uncertainty: {e}")
            return {'total_uncertainty': 0.3}
    
    def _identify_rock_specific_controls(self, rock_type: str, 
                                       petrophysical_analysis: Dict[str, float],
                                       favorability: float) -> List[str]:
        """Identify rock-type specific controls"""
        try:
            controls = []
            
            if rock_type == 'sandstone' and favorability > 0.5:
                controls.append(f"sandstone_high_matrix_permeability_favorable")
                
            elif rock_type == 'limestone' and favorability > 0.4:
                controls.append(f"limestone_dissolution_enhanced_permeability")
                
            elif rock_type == 'granite' and favorability > 0.3:
                controls.append(f"granite_fracture_controlled_permeability")
                
            elif rock_type == 'basalt' and favorability > 0.4:
                controls.append(f"basalt_vesicular_porosity_enhanced")
                
            elif rock_type == 'shale' and favorability > 0.2:
                controls.append(f"shale_fracture_network_critical")
            
            return controls
            
        except Exception as e:
            logger.warning(f"Failed to identify rock-specific controls: {e}")
            return []
    
    async def _identify_regional_lithological_trends(self, pathway_array: np.ndarray,
                                                   lithology_data: List[Dict]) -> List[str]:
        """Identify regional lithological trends affecting gas migration"""
        try:
            controls = []
            
            if not lithology_data:
                return controls
            
            # Analyze stratigraphic trends
            rock_types = [unit.get('rock_type', 'unknown') for unit in lithology_data]
            formations = [unit.get('formation_name', 'unknown') for unit in lithology_data]
            
            # Identify dominant lithologies
            from collections import Counter
            rock_counts = Counter(rock_types)
            formation_counts = Counter(formations)
            
            dominant_rock = rock_counts.most_common(1)[0][0] if rock_counts else 'unknown'
            dominant_formation = formation_counts.most_common(1)[0][0] if formation_counts else 'unknown'
            
            if rock_counts[dominant_rock] > len(lithology_data) * 0.6:
                controls.append(f"regional_lithology_dominated_by_{dominant_rock}")
            
            if formation_counts[dominant_formation] > len(lithology_data) * 0.5:
                controls.append(f"regional_formation_{dominant_formation}_control")
            
            # Analyze permeability trends with depth
            permeabilities = [unit.get('permeability', 1e-15) for unit in lithology_data]
            if len(permeabilities) > 3:
                # Simple trend analysis
                log_perms = [np.log10(p) for p in permeabilities]
                trend_slope = (log_perms[-1] - log_perms[0]) / len(log_perms)
                
                if trend_slope > 0.5:
                    controls.append("regional_permeability_increases_with_depth")
                elif trend_slope < -0.5:
                    controls.append("regional_permeability_decreases_with_depth")
            
            return controls
            
        except Exception as e:
            logger.warning(f"Failed to identify regional lithological trends: {e}")
            return []
    
    async def _identify_stress_field_controls(self, pathway_array: np.ndarray) -> List[str]:
        """Identify stress field controls on gas migration using advanced geomechanical analysis"""
        controls = []
        
        try:
            # Comprehensive stress field database query
            stress_query = """
            SELECT sf.stress_regime, sf.shmax_azimuth, sf.shmin_azimuth, sf.sv_gradient,
                   sf.stress_magnitude_ratio, sf.pore_pressure_ratio, sf.depth_range,
                   sf.confidence, sf.measurement_method, sf.data_quality,
                   eq.magnitude, eq.depth as eq_depth, eq.focal_mechanism,
                   ST_AsText(sf.geometry) as geometry_wkt,
                   ST_Distance(sf.geometry, ST_MakeLine(ARRAY[%s])) as distance
            FROM stress_field_data sf
            LEFT JOIN earthquake_data eq ON ST_DWithin(sf.geometry, eq.geometry, 1000)
            WHERE ST_DWithin(sf.geometry, ST_MakeLine(ARRAY[%s]), %s)
            ORDER BY sf.confidence DESC, distance ASC
            """
            
            pathway_linestring = f"LINESTRING({', '.join([f'{x} {y} {z}' for x, y, z in pathway_array])})"
            
            try:
                stress_data = await self.db_manager.execute_query(
                    stress_query, (pathway_linestring, pathway_linestring, 1000.0)  # 1km buffer
                )
            except RuntimeError:
                stress_data = []
            
            # Advanced stress field analysis
            for stress_field in stress_data:
                # Comprehensive stress field influence calculation
                stress_analysis = await self._perform_advanced_stress_analysis(
                    stress_field, pathway_array
                )
                
                if stress_analysis['total_influence'] > 0.3:  # Significant stress control
                    # Detailed stress control description
                    control_desc = (
                        f"stress_regime_{stress_field['stress_regime']}_"
                        f"shmax_azimuth{stress_field['shmax_azimuth']:.0f}_"
                        f"stress_ratio{stress_field['stress_magnitude_ratio']:.2f}_"
                        f"pore_pressure_ratio{stress_field.get('pore_pressure_ratio', 1.0):.2f}_"
                        f"fracture_susceptibility{stress_analysis['fracture_susceptibility']:.2f}_"
                        f"permeability_enhancement{stress_analysis['permeability_enhancement']:.2f}_"
                        f"pathway_alignment{stress_analysis['pathway_alignment']:.2f}_"
                        f"total_influence{stress_analysis['total_influence']:.3f}_"
                        f"confidence{stress_field['confidence']:.2f}_"
                        f"distance{stress_field.get('distance', 0):.0f}m"
                    )
                    controls.append(control_desc)
                    
                    # Add specific stress-related controls
                    specific_controls = self._identify_specific_stress_controls(
                        stress_field, stress_analysis, pathway_array
                    )
                    controls.extend(specific_controls)
            
            # Add regional stress pattern analysis
            regional_stress_controls = await self._analyze_regional_stress_patterns(
                pathway_array, stress_data
            )
            controls.extend(regional_stress_controls)
            
            # Add seismic stress indicators
            seismic_stress_controls = await self._analyze_seismic_stress_indicators(
                pathway_array
            )
            controls.extend(seismic_stress_controls)
            
        except Exception as e:
            logger.warning(f"Failed to identify stress field controls: {e}")
        
        return controls
    
    async def _perform_advanced_stress_analysis(self, stress_field: Dict[str, Any], 
                                               pathway_array: np.ndarray) -> Dict[str, float]:
        """Perform comprehensive stress field analysis for gas migration"""
        try:
            # Extract stress field parameters
            stress_regime = stress_field.get('stress_regime', 'unknown')
            shmax_azimuth = np.radians(stress_field.get('shmax_azimuth', 0.0))
            shmin_azimuth = np.radians(stress_field.get('shmin_azimuth', shmax_azimuth + np.pi/2))
            stress_ratio = stress_field.get('stress_magnitude_ratio', 1.0)
            pore_pressure_ratio = stress_field.get('pore_pressure_ratio', 1.0)
            confidence = stress_field.get('confidence', 0.5)
            
            # Calculate 3D stress tensor components
            stress_tensor = self._calculate_3d_stress_tensor(
                stress_regime, shmax_azimuth, shmin_azimuth, stress_ratio, pathway_array
            )
            
            # Analyze pathway orientation relative to principal stress directions
            pathway_alignment = self._calculate_pathway_stress_alignment(
                pathway_array, stress_tensor
            )
            
            # Calculate fracture susceptibility using Mohr-Coulomb criteria
            fracture_susceptibility = self._calculate_fracture_susceptibility(
                stress_tensor, pore_pressure_ratio
            )
            
            # Stress-dependent permeability enhancement
            permeability_enhancement = self._calculate_stress_permeability_enhancement(
                stress_tensor, pathway_alignment, fracture_susceptibility
            )
            
            # Critical stress analysis for fracture reactivation
            reactivation_potential = self._calculate_fracture_reactivation_potential(
                stress_tensor, pathway_array
            )
            
            # Stress concentration effects near pathway
            stress_concentration = self._calculate_pathway_stress_concentration(
                pathway_array, stress_tensor
            )
            
            # Pore pressure effects on effective stress
            effective_stress_analysis = self._analyze_effective_stress_effects(
                stress_tensor, pore_pressure_ratio, pathway_array
            )
            
            # Temporal stress variations (if seismic data available)
            temporal_stress_variations = self._analyze_temporal_stress_variations(
                stress_field, pathway_array
            )
            
            # Combined stress influence calculation
            influence_components = {
                'pathway_alignment': pathway_alignment,
                'fracture_susceptibility': fracture_susceptibility,
                'permeability_enhancement': permeability_enhancement,
                'reactivation_potential': reactivation_potential,
                'stress_concentration': stress_concentration,
                'effective_stress_factor': effective_stress_analysis,
                'temporal_variations': temporal_stress_variations,
                'confidence_factor': confidence
            }
            
            # Weighted combination
            weights = {
                'pathway_alignment': 0.25,
                'fracture_susceptibility': 0.20,
                'permeability_enhancement': 0.20,
                'reactivation_potential': 0.15,
                'stress_concentration': 0.10,
                'effective_stress_factor': 0.05,
                'temporal_variations': 0.03,
                'confidence_factor': 0.02
            }
            
            total_influence = sum(
                influence_components[component] * weights[component]
                for component in influence_components
            )
            
            # Add all components to return dictionary
            result = influence_components.copy()
            result['total_influence'] = total_influence
            result['stress_regime'] = stress_regime
            result['stress_tensor'] = stress_tensor
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to perform advanced stress analysis: {e}")
            return {'total_influence': 0.5}
    
    def _calculate_3d_stress_tensor(self, stress_regime: str, shmax_azimuth: float,
                                   shmin_azimuth: float, stress_ratio: float,
                                   pathway_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate 3D stress tensor from stress field parameters"""
        try:
            # Average depth for stress calculations
            avg_depth = np.mean(pathway_array[:, 2])
            
            # Vertical stress (lithostatic)
            rock_density = 2700  # kg/m³
            g = 9.81  # m/s²
            sv = rock_density * g * avg_depth  # Pa
            
            # Horizontal stresses based on stress regime
            if stress_regime.lower() == 'normal':
                # σv > σHmax > σhmin
                shmax = sv * 0.7  # Typical ratio for normal faulting
                shmin = sv * 0.5
            elif stress_regime.lower() == 'reverse':
                # σHmax > σhmin > σv
                shmax = sv * 1.5  # Typical ratio for reverse faulting
                shmin = sv * 1.2
            elif stress_regime.lower() == 'strike_slip':
                # σHmax > σv > σhmin
                shmax = sv * 1.2  # Typical ratio for strike-slip
                shmin = sv * 0.8
            else:
                # Unknown regime - assume intermediate
                shmax = sv * 1.0
                shmin = sv * 0.8
            
            # Apply stress magnitude ratio
            shmax *= stress_ratio
            
            # Create stress tensor in principal coordinates
            principal_stresses = np.array([shmax, shmin, sv])
            
            # Rotation matrix to geographic coordinates
            cos_shmax = np.cos(shmax_azimuth)
            sin_shmax = np.sin(shmax_azimuth)
            cos_shmin = np.cos(shmin_azimuth)
            sin_shmin = np.sin(shmin_azimuth)
            
            # Stress tensor components in geographic coordinates
            stress_tensor = {
                'sigma_xx': shmax * cos_shmax**2 + shmin * sin_shmax**2,
                'sigma_yy': shmax * sin_shmax**2 + shmin * cos_shmax**2,
                'sigma_zz': sv,
                'sigma_xy': (shmax - shmin) * cos_shmax * sin_shmax,
                'sigma_xz': 0.0,  # Assume no shear in vertical plane
                'sigma_yz': 0.0,
                'principal_stresses': principal_stresses,
                'shmax_azimuth': shmax_azimuth,
                'shmin_azimuth': shmin_azimuth
            }
            
            return stress_tensor
            
        except Exception as e:
            logger.warning(f"Failed to calculate 3D stress tensor: {e}")
            return {}
    
    def _calculate_pathway_stress_alignment(self, pathway_array: np.ndarray,
                                          stress_tensor: Dict[str, Any]) -> float:
        """Calculate pathway alignment with principal stress directions"""
        try:
            if len(pathway_array) < 2 or not stress_tensor:
                return 0.5
            
            # Calculate pathway direction vectors
            pathway_vectors = np.diff(pathway_array, axis=0)
            
            # Principal stress directions
            shmax_direction = np.array([
                np.cos(stress_tensor['shmax_azimuth']),
                np.sin(stress_tensor['shmax_azimuth']),
                0.0
            ])
            
            shmin_direction = np.array([
                np.cos(stress_tensor['shmin_azimuth']),
                np.sin(stress_tensor['shmin_azimuth']),
                0.0
            ])
            
            vertical_direction = np.array([0.0, 0.0, 1.0])
            
            # Calculate alignment with each principal direction
            alignments = []
            
            for segment in pathway_vectors:
                segment_length = np.linalg.norm(segment)
                if segment_length > 0:
                    segment_unit = segment / segment_length
                    
                    # Calculate dot products with principal directions
                    shmax_alignment = abs(np.dot(segment_unit, shmax_direction))
                    shmin_alignment = abs(np.dot(segment_unit, shmin_direction))
                    vertical_alignment = abs(np.dot(segment_unit, vertical_direction))
                    
                    # Best alignment with any principal direction
                    best_alignment = max(shmax_alignment, shmin_alignment, vertical_alignment)
                    alignments.append(best_alignment)
            
            # Average alignment across all segments
            if alignments:
                return np.mean(alignments)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Failed to calculate pathway stress alignment: {e}")
            return 0.5
    
    def _calculate_fracture_susceptibility(self, stress_tensor: Dict[str, Any],
                                         pore_pressure_ratio: float) -> float:
        """Calculate fracture susceptibility using Mohr-Coulomb failure criteria"""
        try:
            if not stress_tensor:
                return 0.5
            
            principal_stresses = stress_tensor.get('principal_stresses', np.array([1e6, 0.8e6, 1.2e6]))
            sigma1, sigma2, sigma3 = np.sort(principal_stresses)[::-1]  # Sort in descending order
            
            # Effective stresses (subtract pore pressure)
            pore_pressure = pore_pressure_ratio * sigma3  # Assume hydrostatic pore pressure
            sigma1_eff = sigma1 - pore_pressure
            sigma3_eff = sigma3 - pore_pressure
            
            # Mohr-Coulomb failure criteria
            # τ = c + σn × tan(φ)
            # For tensile failure: σ3_eff < -T (tensile strength)
            # For shear failure: (σ1_eff - σ3_eff) > 2c×cos(φ) + (σ1_eff + σ3_eff)×sin(φ)
            
            # Typical rock properties
            cohesion = 10e6  # Pa (10 MPa)
            friction_angle = np.radians(30)  # 30 degrees
            tensile_strength = 5e6  # Pa (5 MPa)
            
            # Tensile failure criterion
            tensile_failure_ratio = max(0, -sigma3_eff / tensile_strength)
            
            # Shear failure criterion
            shear_stress = (sigma1_eff - sigma3_eff) / 2.0
            normal_stress = (sigma1_eff + sigma3_eff) / 2.0
            shear_strength = cohesion + normal_stress * np.tan(friction_angle)
            shear_failure_ratio = shear_stress / shear_strength if shear_strength > 0 else 0
            
            # Combined fracture susceptibility
            fracture_susceptibility = max(tensile_failure_ratio, shear_failure_ratio)
            
            return min(1.0, fracture_susceptibility)
            
        except Exception as e:
            logger.warning(f"Failed to calculate fracture susceptibility: {e}")
            return 0.5
    
    def _calculate_stress_permeability_enhancement(self, stress_tensor: Dict[str, Any],
                                                 pathway_alignment: float,
                                                 fracture_susceptibility: float) -> float:
        """Calculate stress-induced permeability enhancement"""
        try:
            if not stress_tensor:
                return 1.0
            
            # Stress-dependent permeability enhancement mechanisms:
            # 1. Dilatancy due to shear stress
            # 2. Fracture opening due to tensile stress
            # 3. Pathway alignment with favorable stress directions
            
            principal_stresses = stress_tensor.get('principal_stresses', np.array([1e6, 0.8e6, 1.2e6]))
            sigma1, sigma2, sigma3 = np.sort(principal_stresses)[::-1]
            
            # Differential stress (deviatoric stress)
            differential_stress = sigma1 - sigma3
            mean_stress = (sigma1 + sigma2 + sigma3) / 3.0
            
            # Dilatancy factor (higher differential stress = more dilatancy)
            dilatancy_factor = 1.0 + 0.5 * min(differential_stress / mean_stress, 2.0)
            
            # Fracture opening factor
            opening_factor = 1.0 + 2.0 * fracture_susceptibility
            
            # Alignment factor (better alignment = higher enhancement)
            alignment_factor = 1.0 + pathway_alignment
            
            # Combined enhancement
            permeability_enhancement = dilatancy_factor * opening_factor * alignment_factor
            
            return min(5.0, permeability_enhancement)  # Cap at 5x enhancement
            
        except Exception as e:
            logger.warning(f"Failed to calculate stress permeability enhancement: {e}")
            return 1.0
    
    def _calculate_fracture_reactivation_potential(self, stress_tensor: Dict[str, Any],
                                                 pathway_array: np.ndarray) -> float:
        """Calculate potential for fracture reactivation under current stress"""
        try:
            if not stress_tensor or len(pathway_array) < 2:
                return 0.5
            
            # Assume existing fractures along pathway
            pathway_vectors = np.diff(pathway_array, axis=0)
            
            reactivation_potentials = []
            
            for segment in pathway_vectors:
                segment_length = np.linalg.norm(segment)
                if segment_length > 0:
                    segment_unit = segment / segment_length
                    
                    # Calculate resolved shear and normal stress on fracture plane
                    # Fracture normal perpendicular to segment
                    fracture_normal = np.array([-segment_unit[1], segment_unit[0], 0])
                    fracture_normal /= np.linalg.norm(fracture_normal)
                    
                    # Stress components
                    sigma_xx = stress_tensor.get('sigma_xx', 1e6)
                    sigma_yy = stress_tensor.get('sigma_yy', 1e6)
                    sigma_xy = stress_tensor.get('sigma_xy', 0)
                    
                    # Normal stress on fracture plane
                    normal_stress = (
                        sigma_xx * fracture_normal[0]**2 +
                        sigma_yy * fracture_normal[1]**2 +
                        2 * sigma_xy * fracture_normal[0] * fracture_normal[1]
                    )
                    
                    # Shear stress on fracture plane
                    shear_stress = abs(
                        (sigma_xx - sigma_yy) * fracture_normal[0] * fracture_normal[1] +
                        sigma_xy * (fracture_normal[0]**2 - fracture_normal[1]**2)
                    )
                    
                    # Reactivation criterion (simplified Coulomb friction)
                    friction_coefficient = 0.6  # Typical value
                    cohesion = 1e6  # Pa
                    
                    critical_shear_stress = cohesion + friction_coefficient * normal_stress
                    reactivation_ratio = shear_stress / critical_shear_stress if critical_shear_stress > 0 else 0
                    
                    reactivation_potentials.append(min(1.0, reactivation_ratio))
            
            return np.mean(reactivation_potentials) if reactivation_potentials else 0.5
            
        except Exception as e:
            logger.warning(f"Failed to calculate fracture reactivation potential: {e}")
            return 0.5
    
    def _calculate_pathway_stress_concentration(self, pathway_array: np.ndarray,
                                              stress_tensor: Dict[str, Any]) -> float:
        """Calculate stress concentration effects near pathway"""
        try:
            # Simplified stress concentration analysis
            # Higher stress concentrations can enhance fracture propagation
            
            if not stress_tensor:
                return 0.5
            
            principal_stresses = stress_tensor.get('principal_stresses', np.array([1e6, 0.8e6, 1.2e6]))
            sigma1, sigma2, sigma3 = np.sort(principal_stresses)[::-1]
            
            # Stress concentration factor based on geometry and stress state
            # Higher differential stress = higher concentration potential
            stress_concentration_factor = (sigma1 - sigma3) / sigma3 if sigma3 > 0 else 1.0
            
            # Normalize to 0-1 range
            concentration_score = min(stress_concentration_factor / 3.0, 1.0)
            
            return concentration_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate pathway stress concentration: {e}")
            return 0.5
    
    def _analyze_effective_stress_effects(self, stress_tensor: Dict[str, Any],
                                        pore_pressure_ratio: float,
                                        pathway_array: np.ndarray) -> float:
        """Analyze effective stress effects on gas migration"""
        try:
            if not stress_tensor:
                return 0.5
            
            # Calculate effective stress
            principal_stresses = stress_tensor.get('principal_stresses', np.array([1e6, 0.8e6, 1.2e6]))
            avg_depth = np.mean(pathway_array[:, 2])
            
            # Pore pressure
            hydrostatic_pressure = 1000 * 9.81 * avg_depth  # Water density × g × depth
            pore_pressure = pore_pressure_ratio * hydrostatic_pressure
            
            # Effective stresses
            effective_stresses = principal_stresses - pore_pressure
            
            # Effective stress ratio
            if np.min(principal_stresses) > 0:
                effective_stress_ratio = np.mean(effective_stresses) / np.mean(principal_stresses)
            else:
                effective_stress_ratio = 0.5
            
            # Higher pore pressure (lower effective stress) generally favors gas migration
            return max(0.1, 1.0 - effective_stress_ratio)
            
        except Exception as e:
            logger.warning(f"Failed to analyze effective stress effects: {e}")
            return 0.5
    
    def _analyze_temporal_stress_variations(self, stress_field: Dict[str, Any],
                                          pathway_array: np.ndarray) -> float:
        """Analyze temporal variations in stress field"""
        try:
            # Check for seismic activity indicators
            magnitude = stress_field.get('magnitude', 0.0)
            eq_depth = stress_field.get('eq_depth', 0.0)
            
            if magnitude > 0:
                # Seismic activity can temporarily alter stress field
                # Larger earthquakes have more influence
                seismic_influence = min(magnitude / 7.0, 1.0)  # Normalize to M7
                
                # Depth effects - shallower earthquakes have more surface influence
                depth_factor = np.exp(-eq_depth / 10000.0)  # 10km characteristic depth
                
                temporal_variation = seismic_influence * depth_factor
            else:
                temporal_variation = 0.1  # Background tectonic loading
            
            return min(1.0, temporal_variation)
            
        except Exception as e:
            logger.warning(f"Failed to analyze temporal stress variations: {e}")
            return 0.1
    
    def _identify_specific_stress_controls(self, stress_field: Dict[str, Any],
                                         stress_analysis: Dict[str, float],
                                         pathway_array: np.ndarray) -> List[str]:
        """Identify specific stress-related controls"""
        try:
            controls = []
            
            stress_regime = stress_field.get('stress_regime', 'unknown')
            
            # High fracture susceptibility
            if stress_analysis.get('fracture_susceptibility', 0) > 0.7:
                controls.append(f"high_fracture_susceptibility_{stress_regime}")
            
            # Strong permeability enhancement
            if stress_analysis.get('permeability_enhancement', 1) > 2.0:
                controls.append(f"strong_stress_permeability_enhancement")
            
            # High reactivation potential
            if stress_analysis.get('reactivation_potential', 0) > 0.6:
                controls.append(f"high_fracture_reactivation_potential")
            
            # Favorable pathway alignment
            if stress_analysis.get('pathway_alignment', 0) > 0.8:
                controls.append(f"optimal_pathway_stress_alignment")
            
            return controls
            
        except Exception as e:
            logger.warning(f"Failed to identify specific stress controls: {e}")
            return []
    
    async def _analyze_regional_stress_patterns(self, pathway_array: np.ndarray,
                                              stress_data: List[Dict]) -> List[str]:
        """Analyze regional stress patterns"""
        try:
            controls = []
            
            if not stress_data:
                return controls
            
            # Analyze stress regime consistency
            regimes = [sf.get('stress_regime', 'unknown') for sf in stress_data]
            from collections import Counter
            regime_counts = Counter(regimes)
            
            dominant_regime = regime_counts.most_common(1)[0][0] if regime_counts else 'unknown'
            
            if regime_counts[dominant_regime] > len(stress_data) * 0.7:
                controls.append(f"regional_stress_regime_{dominant_regime}_dominant")
            
            # Analyze stress orientation consistency
            azimuths = [sf.get('shmax_azimuth', 0) for sf in stress_data if sf.get('shmax_azimuth') is not None]
            if len(azimuths) > 2:
                azimuth_std = np.std(azimuths)
                if azimuth_std < 15.0:  # Low variability
                    controls.append(f"consistent_regional_stress_orientation_std{azimuth_std:.1f}")
                elif azimuth_std > 45.0:  # High variability
                    controls.append(f"variable_regional_stress_orientation_std{azimuth_std:.1f}")
            
            return controls
            
        except Exception as e:
            logger.warning(f"Failed to analyze regional stress patterns: {e}")
            return []
    
    async def _analyze_seismic_stress_indicators(self, pathway_array: np.ndarray) -> List[str]:
        """Analyze seismic indicators of stress field"""
        try:
            controls = []
            
            # Query recent seismic activity
            seismic_query = """
            SELECT magnitude, depth, focal_mechanism, event_time,
                   ST_Distance(geometry, ST_MakeLine(ARRAY[%s])) as distance
            FROM earthquake_data 
            WHERE ST_DWithin(geometry, ST_MakeLine(ARRAY[%s]), %s)
            AND event_time > NOW() - INTERVAL '1 year'
            ORDER BY magnitude DESC
            """
            
            pathway_linestring = f"LINESTRING({', '.join([f'{x} {y} {z}' for x, y, z in pathway_array])})"
            
            try:
                seismic_data = await self.db_manager.execute_query(
                    seismic_query, (pathway_linestring, pathway_linestring, 10000.0)  # 10km radius
                )
            except RuntimeError:
                seismic_data = []
            
            if seismic_data:
                # Analyze seismic activity patterns
                magnitudes = [eq.get('magnitude', 0) for eq in seismic_data]
                max_magnitude = max(magnitudes) if magnitudes else 0
                
                if max_magnitude > 4.0:
                    controls.append(f"significant_seismic_activity_max_M{max_magnitude:.1f}")
                
                # Analyze focal mechanisms if available
                focal_mechanisms = [eq.get('focal_mechanism', 'unknown') for eq in seismic_data]
                mechanism_counts = Counter(focal_mechanisms)
                
                if mechanism_counts:
                    dominant_mechanism = mechanism_counts.most_common(1)[0][0]
                    if dominant_mechanism != 'unknown':
                        controls.append(f"dominant_focal_mechanism_{dominant_mechanism}")
            
            return controls
            
        except Exception as e:
            logger.warning(f"Failed to analyze seismic stress indicators: {e}")
            return []
            
            return max(0.0, min(1.0, stress_influence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate stress field influence: {e}")
            return 0.5
    
    async def perform_temporal_pathway_analysis(self, volcanic_system: VolcanicSystem,
                                              time_range: Tuple[datetime, datetime],
                                              time_resolution: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Perform temporal analysis of gas migration pathways"""
        try:
            start_time, end_time = time_range
            current_time = start_time
            temporal_results = {
                'timestamps': [],
                'pathway_evolution': {},
                'gas_flux_evolution': {},
                'composition_evolution': {},
                'efficiency_evolution': {}
            }
            
            while current_time <= end_time:
                # Get time-specific data
                time_specific_data = await self._get_time_specific_data(
                    volcanic_system, current_time
                )
                
                # Update fracture network based on seismic activity
                await self._update_fracture_network_temporal(
                    volcanic_system, current_time, time_specific_data
                )
                
                # Recalculate pathways for current time
                current_pathways = await self._calculate_pathways_at_time(
                    volcanic_system, current_time, time_specific_data
                )
                
                # Store temporal results
                temporal_results['timestamps'].append(current_time)
                temporal_results['pathway_evolution'][current_time.isoformat()] = current_pathways
                
                # Calculate temporal metrics
                flux_metrics = self._calculate_temporal_flux_metrics(current_pathways)
                temporal_results['gas_flux_evolution'][current_time.isoformat()] = flux_metrics
                
                composition_metrics = await self._calculate_temporal_composition_metrics(
                    current_pathways, time_specific_data
                )
                temporal_results['composition_evolution'][current_time.isoformat()] = composition_metrics
                
                efficiency_metrics = self._calculate_temporal_efficiency_metrics(current_pathways)
                temporal_results['efficiency_evolution'][current_time.isoformat()] = efficiency_metrics
                
                current_time += time_resolution
            
            # Analyze temporal trends
            trend_analysis = self._analyze_temporal_trends(temporal_results)
            temporal_results['trend_analysis'] = trend_analysis
            
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error performing temporal pathway analysis: {str(e)}")
            raise
    
    async def _get_time_specific_data(self, volcanic_system: VolcanicSystem, 
                                    timestamp: datetime) -> Dict[str, Any]:
        """Get time-specific volcanic and geological data"""
        try:
            # Query seismic activity
            seismic_query = """
            SELECT event_id, magnitude, depth, location, event_type
            FROM geological_events 
            WHERE volcanic_system_id = %s 
            AND timestamp BETWEEN %s AND %s
            AND event_type IN ('earthquake', 'tremor', 'explosion')
            ORDER BY timestamp DESC
            """
            
            time_window = timedelta(hours=24)  # 24-hour window
            start_window = timestamp - time_window
            end_window = timestamp + time_window
            
            seismic_events = await self.db_manager.execute_query(
                seismic_query, (volcanic_system.system_id, start_window, end_window)
            )
            
            # Query deformation data
            deformation_query = """
            SELECT station_id, displacement_x, displacement_y, displacement_z,
                   uncertainty_x, uncertainty_y, uncertainty_z
            FROM deformation_measurements 
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
            """
            
            deformation_data = await self.db_manager.execute_query(
                deformation_query, (start_window, end_window)
            )
            
            # Query gas emission data
            gas_query = """
            SELECT measurement_id, co2_flux, so2_flux, h2s_concentration,
                   temperature, location
            FROM gas_measurements 
            WHERE timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
            """
            
            gas_data = await self.db_manager.execute_query(
                gas_query, (start_window, end_window)
            )
            
            return {
                'seismic_events': seismic_events,
                'deformation_data': deformation_data,
                'gas_data': gas_data,
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.warning(f"Failed to get time-specific data: {e}")
            return {'timestamp': timestamp}
    
    def _calculate_linear_trend(self, x_values: List[float], y_values: List[float]) -> Dict[str, float]:
        """Calculate linear trend using least squares regression"""
        try:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
            
            x = np.array(x_values)
            y = np.array(y_values)
            
            # Calculate slope and intercept
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            intercept = (np.sum(y) - slope * np.sum(x)) / n
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate linear trend: {e}")
            return {'slope': 0.0, 'intercept': 0.0, 'r_squared': 0.0}
    
    async def integrate_with_real_time_monitoring(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Integrate gas migration modeling with real-time monitoring systems"""
        try:
            # Initialize real-time data streams
            monitoring_integration = {
                'seismic_stream': await self._setup_seismic_data_stream(volcanic_system),
                'gas_stream': await self._setup_gas_monitoring_stream(volcanic_system),
                'deformation_stream': await self._setup_deformation_stream(volcanic_system),
                'thermal_stream': await self._setup_thermal_monitoring_stream(volcanic_system)
            }
            
            # Setup automated pathway updates
            update_scheduler = await self._setup_automated_pathway_updates(
                volcanic_system, monitoring_integration
            )
            
            # Initialize alert system
            alert_system = await self._setup_pathway_alert_system(volcanic_system)
            
            return {
                'monitoring_integration': monitoring_integration,
                'update_scheduler': update_scheduler,
                'alert_system': alert_system,
                'status': 'active',
                'integration_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error integrating with real-time monitoring: {str(e)}")
            raise
    
    async def _setup_seismic_data_stream(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Setup real-time seismic data stream integration"""
        try:
            # Query active seismic stations
            station_query = """
            SELECT station_id, name, location, network, sampling_rate, active
            FROM seismic_stations 
            WHERE ST_DWithin(location, ST_GeomFromText(%s), %s) 
            AND active = true
            """
            
            # Create bounding geometry for volcanic system
            system_center = f"POINT({volcanic_system.longitude} {volcanic_system.latitude})"
            search_radius = 50000.0  # 50 km radius
            
            active_stations = await self.db_manager.execute_query(
                station_query, (system_center, search_radius)
            )
            
            # Setup data stream configuration
            stream_config = {
                'stations': active_stations,
                'data_types': ['continuous', 'triggered', 'events'],
                'processing_parameters': {
                    'frequency_bands': [(0.5, 5.0), (5.0, 15.0), (15.0, 50.0)],
                    'detection_thresholds': {'sta_lta': 3.0, 'amplitude': 1e-6},
                    'event_classification': True,
                    'real_time_location': True
                },
                'update_interval': 60,  # seconds
                'buffer_size': 3600  # 1 hour of data
            }
            
            return stream_config
            
        except Exception as e:
            logger.warning(f"Failed to setup seismic data stream: {e}")
            return {}
    
    async def _setup_gas_monitoring_stream(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Setup real-time gas monitoring stream integration"""
        try:
            # Query active gas monitoring stations
            gas_station_query = """
            SELECT station_id, location, measurement_types, sampling_interval,
                   calibration_date, active
            FROM gas_monitoring_stations 
            WHERE ST_DWithin(location, ST_GeomFromText(%s), %s) 
            AND active = true
            """
            
            system_center = f"POINT({volcanic_system.longitude} {volcanic_system.latitude})"
            search_radius = 25000.0  # 25 km radius
            
            gas_stations = await self.db_manager.execute_query(
                gas_station_query, (system_center, search_radius)
            )
            
            # Setup gas stream configuration
            stream_config = {
                'stations': gas_stations,
                'measured_species': ['CO2', 'SO2', 'H2S', 'H2O', 'CH4', 'He', 'Rn'],
                'measurement_parameters': {
                    'flux_measurements': True,
                    'concentration_measurements': True,
                    'isotopic_ratios': True,
                    'temperature_measurements': True
                },
                'quality_control': {
                    'outlier_detection': True,
                    'calibration_drift_correction': True,
                    'atmospheric_correction': True
                },
                'update_interval': 300,  # 5 minutes
                'alert_thresholds': {
                    'CO2_flux': 1000.0,  # tonnes/day
                    'SO2_flux': 100.0,   # tonnes/day
                    'H2S_concentration': 10.0  # ppm
                }
            }
            
            return stream_config
            
        except Exception as e:
            logger.warning(f"Failed to setup gas monitoring stream: {e}")
            return {}
    
    async def _setup_deformation_stream(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Setup real-time deformation monitoring stream"""
        try:
            # Query GPS and InSAR monitoring systems
            deformation_query = """
            SELECT station_id, station_type, location, processing_method,
                   update_frequency, precision, active
            FROM deformation_monitoring_stations 
            WHERE ST_DWithin(location, ST_GeomFromText(%s), %s) 
            AND active = true
            """
            
            system_center = f"POINT({volcanic_system.longitude} {volcanic_system.latitude})"
            search_radius = 100000.0  # 100 km radius for deformation
            
            deformation_stations = await self.db_manager.execute_query(
                deformation_query, (system_center, search_radius)
            )
            
            # Setup deformation stream configuration
            stream_config = {
                'stations': deformation_stations,
                'monitoring_types': ['GPS', 'InSAR', 'tiltmeter', 'strainmeter'],
                'processing_parameters': {
                    'reference_frame': 'ITRF2014',
                    'atmospheric_correction': True,
                    'orbital_correction': True,
                    'temporal_filtering': True,
                    'spatial_filtering': True
                },
                'analysis_methods': {
                    'time_series_analysis': True,
                    'velocity_estimation': True,
                    'acceleration_detection': True,
                    'source_modeling': True
                },
                'update_interval': 3600,  # 1 hour
                'alert_thresholds': {
                    'displacement_rate': 10.0,  # mm/day
                    'acceleration': 5.0,  # mm/day²
                    'coherence_loss': 0.3
                }
            }
            
            return stream_config
            
        except Exception as e:
            logger.warning(f"Failed to setup deformation stream: {e}")
            return {}
    
    async def _setup_thermal_monitoring_stream(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Setup real-time thermal monitoring stream"""
        try:
            # Query thermal monitoring systems
            thermal_query = """
            SELECT sensor_id, sensor_type, location, spectral_bands,
                   spatial_resolution, temporal_resolution, active
            FROM thermal_monitoring_sensors 
            WHERE ST_DWithin(location, ST_GeomFromText(%s), %s) 
            AND active = true
            """
            
            system_center = f"POINT({volcanic_system.longitude} {volcanic_system.latitude})"
            search_radius = 30000.0  # 30 km radius
            
            thermal_sensors = await self.db_manager.execute_query(
                thermal_query, (system_center, search_radius)
            )
            
            # Setup thermal stream configuration
            stream_config = {
                'sensors': thermal_sensors,
                'data_sources': ['MODIS', 'VIIRS', 'ASTER', 'Landsat', 'ground_based'],
                'processing_parameters': {
                    'atmospheric_correction': True,
                    'cloud_masking': True,
                    'fire_pixel_detection': True,
                    'temperature_retrieval': True,
                    'radiant_flux_calculation': True
                },
                'analysis_methods': {
                    'anomaly_detection': True,
                    'trend_analysis': True,
                    'hotspot_tracking': True,
                    'thermal_evolution': True
                },
                'update_interval': 1800,  # 30 minutes
                'alert_thresholds': {
                    'temperature_anomaly': 50.0,  # K above background
                    'radiant_flux': 1e6,  # W
                    'hotspot_area': 1000.0  # m²
                }
            }
            
            return stream_config
            
        except Exception as e:
            logger.warning(f"Failed to setup thermal monitoring stream: {e}")
            return {}
    
    async def _setup_automated_pathway_updates(self, volcanic_system: VolcanicSystem,
                                             monitoring_integration: Dict[str, Any]) -> Dict[str, Any]:
        """Setup automated pathway updates based on monitoring data"""
        try:
            update_scheduler = {
                'update_triggers': {
                    'seismic_threshold': 3.0,  # Magnitude threshold
                    'gas_flux_change': 0.5,   # Relative change threshold
                    'deformation_rate': 5.0,  # mm/day threshold
                    'thermal_anomaly': 20.0   # K threshold
                },
                'update_intervals': {
                    'routine_update': 3600,    # 1 hour
                    'triggered_update': 300,   # 5 minutes
                    'emergency_update': 60     # 1 minute
                },
                'processing_queue': {
                    'max_concurrent_updates': 3,
                    'priority_levels': ['emergency', 'triggered', 'routine'],
                    'timeout_seconds': 1800
                },
                'validation_criteria': {
                    'minimum_data_quality': 0.7,
                    'maximum_uncertainty': 0.3,
                    'convergence_tolerance': 0.05
                }
            }
            
            return update_scheduler
            
        except Exception as e:
            logger.warning(f"Failed to setup automated pathway updates: {e}")
            return {}
    
    async def _setup_pathway_alert_system(self, volcanic_system: VolcanicSystem) -> Dict[str, Any]:
        """Setup alert system for gas migration pathway changes"""
        try:
            alert_system = {
                'alert_levels': {
                    'green': {'flux_change': 0.2, 'pathway_change': 0.1},
                    'yellow': {'flux_change': 0.5, 'pathway_change': 0.3},
                    'orange': {'flux_change': 1.0, 'pathway_change': 0.5},
                    'red': {'flux_change': 2.0, 'pathway_change': 0.8}
                },
                'notification_channels': {
                    'email': True,
                    'sms': True,
                    'dashboard': True,
                    'api_webhook': True
                },
                'escalation_procedures': {
                    'yellow_alert': {'delay_minutes': 15, 'recipients': ['operators']},
                    'orange_alert': {'delay_minutes': 5, 'recipients': ['operators', 'scientists']},
                    'red_alert': {'delay_minutes': 1, 'recipients': ['all', 'emergency_contacts']}
                },
                'alert_suppression': {
                    'minimum_interval': 300,  # 5 minutes between similar alerts
                    'maximum_frequency': 10   # Max 10 alerts per hour
                }
            }
            
            return alert_system
            
        except Exception as e:
            logger.warning(f"Failed to setup pathway alert system: {e}")
            return {}
    
    async def generate_comprehensive_pathway_report(self, volcanic_system: VolcanicSystem,
                                                  pathways: List[GasMigrationPathway],
                                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive gas migration pathway analysis report"""
        try:
            report = {
                'executive_summary': await self._generate_executive_summary(
                    volcanic_system, pathways, analysis_results
                ),
                'pathway_analysis': await self._generate_pathway_analysis_section(pathways),
                'temporal_analysis': analysis_results.get('temporal_analysis', {}),
                'uncertainty_analysis': analysis_results.get('uncertainty_analysis', {}),
                'structural_controls': await self._generate_structural_controls_section(pathways),
                'gas_composition': await self._generate_gas_composition_section(pathways),
                'hazard_assessment': await self._generate_hazard_assessment_section(
                    volcanic_system, pathways
                ),
                'recommendations': await self._generate_recommendations_section(
                    volcanic_system, pathways, analysis_results
                ),
                'technical_appendix': await self._generate_technical_appendix(
                    pathways, analysis_results
                ),
                'metadata': {
                    'report_id': f"gas_pathway_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'volcanic_system': volcanic_system.system_id,
                    'generation_timestamp': datetime.now().isoformat(),
                    'analysis_version': '2.0',
                    'data_sources': self._get_data_sources_summary()
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive pathway report: {str(e)}")
            raise
    
    async def _generate_executive_summary(self, volcanic_system: VolcanicSystem,
                                        pathways: List[GasMigrationPathway],
                                        analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of gas migration analysis"""
        try:
            # Calculate key metrics
            total_pathways = len(pathways)
            high_efficiency_pathways = sum(1 for p in pathways if p.efficiency_score > 0.7)
            
            # Calculate total gas flux
            total_flux = sum(self._calculate_gas_flux(p) for p in pathways)
            
            # Identify primary structural controls
            all_controls = []
            for pathway in pathways:
                all_controls.extend(pathway.structural_controls)
            
            primary_controls = self._identify_primary_controls(all_controls)
            
            # Assess overall hazard level
            hazard_level = self._assess_overall_hazard_level(pathways, analysis_results)
            
            summary = {
                'key_findings': {
                    'total_active_pathways': total_pathways,
                    'high_efficiency_pathways': high_efficiency_pathways,
                    'total_gas_flux_m3_per_day': total_flux * 86400,  # Convert to daily
                    'primary_structural_controls': primary_controls,
                    'overall_hazard_level': hazard_level
                },
                'critical_observations': await self._identify_critical_observations(
                    pathways, analysis_results
                ),
                'immediate_concerns': await self._identify_immediate_concerns(
                    pathways, analysis_results
                ),
                'monitoring_recommendations': await self._generate_monitoring_recommendations(
                    volcanic_system, pathways
                )
            }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate executive summary: {e}")
            return {}
    
    def _identify_primary_controls(self, all_controls: List[str]) -> List[str]:
        """Identify primary structural controls from all pathway controls"""
        try:
            # Count control types
            control_counts = {}
            for control in all_controls:
                control_type = control.split('_')[0]  # Get control type prefix
                control_counts[control_type] = control_counts.get(control_type, 0) + 1
            
            # Sort by frequency and return top 3
            sorted_controls = sorted(control_counts.items(), key=lambda x: x[1], reverse=True)
            return [control[0] for control in sorted_controls[:3]]
            
        except Exception as e:
            logger.warning(f"Failed to identify primary controls: {e}")
            return []
    
    def _assess_overall_hazard_level(self, pathways: List[GasMigrationPathway],
                                   analysis_results: Dict[str, Any]) -> str:
        """Assess overall gas migration hazard level"""
        try:
            if not pathways:
                return 'minimal'
            
            # Calculate hazard metrics
            max_flux = max(self._calculate_gas_flux(p) for p in pathways)
            avg_efficiency = np.mean([p.efficiency_score for p in pathways])
            pathway_density = len(pathways) / 100.0  # Normalize by area
            
            # Assess temporal trends
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            flux_trends = temporal_analysis.get('flux_trends', {})
            flux_trend_slope = flux_trends.get('total_flux_trend', {}).get('slope', 0.0)
            
            # Calculate composite hazard score
            hazard_score = (
                0.4 * min(max_flux / 1e-3, 1.0) +      # Flux component
                0.3 * avg_efficiency +                   # Efficiency component
                0.2 * min(pathway_density, 1.0) +       # Density component
                0.1 * min(abs(flux_trend_slope), 1.0)   # Trend component
            )
            
            # Classify hazard level
            if hazard_score < 0.3:
                return 'minimal'
            elif hazard_score < 0.5:
                return 'low'
            elif hazard_score < 0.7:
                return 'moderate'
            elif hazard_score < 0.9:
                return 'high'
            else:
                return 'critical'
                
        except Exception as e:
            logger.warning(f"Failed to assess overall hazard level: {e}")
            return 'unknown'
    
    def _get_data_sources_summary(self) -> List[str]:
        """Get summary of data sources used in analysis"""
        return [
            'PostgreSQL geological database',
            'Real-time seismic monitoring network',
            'GPS/InSAR deformation measurements',
            'Gas emission monitoring stations',
            'Structural geology database',
            'Rock property measurements',
            'Stress field analysis',
            'Geochemical reaction database'
        ]
    
    async def _update_fracture_network_temporal(self, volcanic_system: VolcanicSystem,
                                              timestamp: datetime, 
                                              time_specific_data: Dict[str, Any]) -> None:
        """Update fracture network based on temporal seismic activity"""
        try:
            seismic_events = time_specific_data.get('seismic_events', [])
            
            # Analyze seismic events for new fracture generation
            for event in seismic_events:
                if event.get('magnitude', 0) > 2.0:  # Significant events only
                    # Calculate stress field changes from earthquake
                    stress_changes = self._calculate_coseismic_stress_changes(event)
                    
                    # Update fracture permeabilities based on stress changes
                    self._update_fracture_permeabilities(stress_changes)
                    
                    # Generate new fractures if stress exceeds threshold
                    if stress_changes.get('max_stress', 0) > 1e6:  # 1 MPa threshold
                        new_fractures = self._generate_new_fractures(event, stress_changes)
                        self._add_fractures_to_network(new_fractures)
            
            # Update fracture connectivity based on deformation
            deformation_data = time_specific_data.get('deformation_data', [])
            if deformation_data:
                self._update_fracture_connectivity(deformation_data)
                
        except Exception as e:
            logger.warning(f"Failed to update fracture network temporally: {e}")
    
    def _calculate_coseismic_stress_changes(self, seismic_event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stress field changes from seismic event"""
        try:
            magnitude = seismic_event.get('magnitude', 0.0)
            depth = seismic_event.get('depth', 5000.0)  # meters
            
            # Empirical stress drop relationship
            stress_drop = 3e6 * (10**(1.5 * magnitude - 9.1))  # Pa
            
            # Calculate stress field using Okada model (simplified)
            max_stress = stress_drop * np.exp(-depth / 10000.0)  # Depth decay
            
            return {
                'max_stress': max_stress,
                'stress_drop': stress_drop,
                'affected_radius': depth * 2.0,  # Simplified affected zone
                'event_location': seismic_event.get('location', (0, 0, depth))
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate coseismic stress changes: {e}")
            return {'max_stress': 0.0}
    
    def _update_fracture_permeabilities(self, stress_changes: Dict[str, float]) -> None:
        """Update fracture permeabilities based on stress changes"""
        try:
            if not self.fracture_network:
                return
                
            max_stress = stress_changes.get('max_stress', 0.0)
            event_location = stress_changes.get('event_location', (0, 0, 0))
            affected_radius = stress_changes.get('affected_radius', 1000.0)
            
            for node in self.fracture_network.nodes():
                fracture_data = self.fracture_network.nodes[node]
                fracture_pos = fracture_data['position']
                
                # Calculate distance from event
                distance = np.linalg.norm(np.array(fracture_pos) - np.array(event_location))
                
                if distance < affected_radius:
                    # Calculate stress influence (exponential decay)
                    stress_factor = max_stress * np.exp(-distance / (affected_radius / 3))
                    
                    # Update permeability based on stress (can increase or decrease)
                    current_perm = fracture_data['permeability']
                    
                    # Stress-permeability relationship (simplified)
                    if stress_factor > 1e5:  # Compression threshold
                        perm_factor = 0.5 + 0.5 * np.exp(-stress_factor / 1e6)
                    else:  # Extension enhances permeability
                        perm_factor = 1.0 + stress_factor / 1e6
                    
                    new_permeability = current_perm * perm_factor
                    self.fracture_network.nodes[node]['permeability'] = new_permeability
                    
        except Exception as e:
            logger.warning(f"Failed to update fracture permeabilities: {e}")
    
    def _generate_new_fractures(self, seismic_event: Dict[str, Any], 
                               stress_changes: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate new fractures from seismic event"""
        try:
            new_fractures = []
            magnitude = seismic_event.get('magnitude', 0.0)
            event_location = stress_changes.get('event_location', (0, 0, 0))
            
            # Number of new fractures based on magnitude
            n_fractures = max(0, int((magnitude - 2.0) * 2))
            
            for i in range(n_fractures):
                # Random fracture orientation with some clustering around event
                strike = np.random.uniform(0, 360)
                dip = np.random.uniform(45, 90)
                
                # Fracture location near event
                offset_x = np.random.normal(0, 500)  # 500m standard deviation
                offset_y = np.random.normal(0, 500)
                offset_z = np.random.normal(0, 200)
                
                fracture_location = (
                    event_location[0] + offset_x,
                    event_location[1] + offset_y,
                    event_location[2] + offset_z
                )
                
                # Fracture properties based on magnitude
                length = 100 * (10**(0.5 * magnitude - 1.5))  # Empirical scaling
                aperture = length / 10000.0  # Aperture-length scaling
                permeability = aperture**2 / 12.0  # Cubic law
                
                new_fracture = {
                    'fracture_id': f"seismic_{seismic_event.get('event_id', i)}_{i}",
                    'position': fracture_location,
                    'strike': strike,
                    'dip': dip,
                    'length': length,
                    'aperture': aperture,
                    'permeability': permeability,
                    'connectivity': 0.5,  # Initial connectivity
                    'age': 0.0,  # Newly formed
                    'origin': 'seismic_event'
                }
                
                new_fractures.append(new_fracture)
            
            return new_fractures
            
        except Exception as e:
            logger.warning(f"Failed to generate new fractures: {e}")
            return []
    
    def _add_fractures_to_network(self, new_fractures: List[Dict[str, Any]]) -> None:
        """Add new fractures to the fracture network"""
        try:
            if not self.fracture_network:
                return
                
            for fracture in new_fractures:
                node_id = fracture['fracture_id']
                
                # Add fracture as new node
                self.fracture_network.add_node(
                    node_id,
                    position=fracture['position'],
                    strike=fracture['strike'],
                    dip=fracture['dip'],
                    length=fracture['length'],
                    aperture=fracture['aperture'],
                    permeability=fracture['permeability'],
                    connectivity=fracture['connectivity']
                )
                
                # Connect to nearby existing fractures
                fracture_pos = np.array(fracture['position'])
                
                for existing_node in self.fracture_network.nodes():
                    if existing_node == node_id:
                        continue
                        
                    existing_pos = np.array(self.fracture_network.nodes[existing_node]['position'])
                    distance = np.linalg.norm(fracture_pos - existing_pos)
                    
                    if distance < 1000.0:  # 1km connection radius
                        connection_prob = self._calculate_fracture_connection_probability(
                            fracture, self.fracture_network.nodes[existing_node]
                        )
                        
                        if connection_prob > 0.3:
                            flow_capacity = self._calculate_flow_capacity(
                                fracture, self.fracture_network.nodes[existing_node]
                            )
                            
                            self.fracture_network.add_edge(
                                node_id, existing_node,
                                distance=distance,
                                connection_probability=connection_prob,
                                flow_capacity=flow_capacity
                            )
                            
        except Exception as e:
            logger.warning(f"Failed to add fractures to network: {e}")
    
    def _update_fracture_connectivity(self, deformation_data: List[Dict[str, Any]]) -> None:
        """Update fracture connectivity based on deformation measurements"""
        try:
            if not self.fracture_network or not deformation_data:
                return
                
            # Calculate average deformation rate
            total_displacement = 0.0
            for measurement in deformation_data:
                displacement = np.sqrt(
                    measurement.get('displacement_x', 0)**2 +
                    measurement.get('displacement_y', 0)**2 +
                    measurement.get('displacement_z', 0)**2
                )
                total_displacement += displacement
            
            avg_displacement = total_displacement / len(deformation_data)
            
            # Update connectivity based on deformation
            connectivity_factor = 1.0 + avg_displacement / 0.01  # 1cm reference
            
            for edge in self.fracture_network.edges():
                current_capacity = self.fracture_network.edges[edge].get('flow_capacity', 0)
                new_capacity = current_capacity * connectivity_factor
                self.fracture_network.edges[edge]['flow_capacity'] = new_capacity
                
        except Exception as e:
            logger.warning(f"Failed to update fracture connectivity: {e}")
    
    async def _calculate_pathways_at_time(self, volcanic_system: VolcanicSystem,
                                        timestamp: datetime,
                                        time_specific_data: Dict[str, Any]) -> List[GasMigrationPathway]:
        """Calculate gas migration pathways for specific time"""
        try:
            # Update permeability field based on current conditions
            await self.model_permeability_field(volcanic_system)
            
            # Get gas source locations and pressures from current data
            gas_data = time_specific_data.get('gas_data', [])
            
            source_locations = []
            source_pressures = []
            
            for gas_measurement in gas_data:
                location = gas_measurement.get('location', (0, 0, 1000))  # Default 1km depth
                
                # Estimate source pressure from gas flux
                co2_flux = gas_measurement.get('co2_flux', 0.0)
                estimated_pressure = 101325.0 + co2_flux * 1e6  # Empirical relationship
                
                source_locations.append(location)
                source_pressures.append(estimated_pressure)
            
            # Use default sources if no gas data available
            if not source_locations:
                source_locations = [(0, 0, 5000), (1000, 1000, 3000)]  # Default deep sources
                source_pressures = [5e6, 3e6]  # 5 MPa and 3 MPa
            
            # Simulate pressure-driven flow
            flow_results = await self.simulate_pressure_driven_flow(
                source_locations, source_pressures
            )
            
            pathways = flow_results.get('pathways', [])
            
            # Update pathway properties with time-specific data
            for pathway in pathways:
                # Update gas composition based on current measurements
                if gas_data:
                    pathway.gas_composition_evolution = self._extract_current_gas_composition(
                        gas_data, pathway
                    )
                
                # Update efficiency based on current fracture network state
                pathway.efficiency_score = self._calculate_pathway_efficiency(
                    pathway.pathway_nodes,
                    flow_results.get('velocity_field', {}),
                    100.0, 100.0, 100.0
                )
            
            return pathways
            
        except Exception as e:
            logger.warning(f"Failed to calculate pathways at time: {e}")
            return []
    
    def _extract_current_gas_composition(self, gas_data: List[Dict[str, Any]], 
                                       pathway: GasMigrationPathway) -> Dict[str, np.ndarray]:
        """Extract current gas composition for pathway"""
        try:
            # Find gas measurement closest to pathway surface location
            pathway_surface = pathway.surface_location
            closest_measurement = None
            min_distance = float('inf')
            
            for measurement in gas_data:
                measurement_loc = measurement.get('location', (0, 0))
                distance = np.sqrt(
                    (pathway_surface[0] - measurement_loc[0])**2 +
                    (pathway_surface[1] - measurement_loc[1])**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_measurement = measurement
            
            if closest_measurement:
                # Extract composition ratios
                co2_flux = closest_measurement.get('co2_flux', 100.0)
                so2_flux = closest_measurement.get('so2_flux', 10.0)
                h2s_conc = closest_measurement.get('h2s_concentration', 1.0)
                
                # Normalize to fractions
                total = co2_flux + so2_flux + h2s_conc + 10.0  # Add H2O estimate
                
                n_nodes = len(pathway.pathway_nodes)
                composition = {
                    'CO2': np.full(n_nodes, co2_flux / total),
                    'SO2': np.full(n_nodes, so2_flux / total),
                    'H2S': np.full(n_nodes, h2s_conc / total),
                    'H2O': np.full(n_nodes, 10.0 / total)
                }
                
                return composition
            else:
                # Default composition
                n_nodes = len(pathway.pathway_nodes)
                return {
                    'CO2': np.full(n_nodes, 0.8),
                    'H2O': np.full(n_nodes, 0.15),
                    'SO2': np.full(n_nodes, 0.04),
                    'H2S': np.full(n_nodes, 0.01)
                }
                
        except Exception as e:
            logger.warning(f"Failed to extract current gas composition: {e}")
            n_nodes = len(pathway.pathway_nodes)
            return {
                'CO2': np.full(n_nodes, 0.8),
                'H2O': np.full(n_nodes, 0.2)
            }
    
    def _calculate_temporal_flux_metrics(self, pathways: List[GasMigrationPathway]) -> Dict[str, float]:
        """Calculate temporal gas flux metrics"""
        try:
            if not pathways:
                return {'total_flux': 0.0, 'max_flux': 0.0, 'avg_flux': 0.0}
            
            fluxes = [self._calculate_gas_flux(pathway) for pathway in pathways]
            
            return {
                'total_flux': sum(fluxes),
                'max_flux': max(fluxes),
                'avg_flux': np.mean(fluxes),
                'flux_std': np.std(fluxes),
                'active_pathways': len([f for f in fluxes if f > 1e-8])
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate temporal flux metrics: {e}")
            return {'total_flux': 0.0}
    
    async def _calculate_temporal_composition_metrics(self, pathways: List[GasMigrationPathway],
                                                   time_specific_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate temporal gas composition metrics"""
        try:
            if not pathways:
                return {}
            
            # Aggregate composition data from all pathways
            all_compositions = {}
            
            for pathway in pathways:
                for species, evolution in pathway.gas_composition_evolution.items():
                    if species not in all_compositions:
                        all_compositions[species] = []
                    
                    # Use surface composition (last node)
                    if len(evolution) > 0:
                        all_compositions[species].append(evolution[-1])
            
            # Calculate metrics for each species
            composition_metrics = {}
            for species, values in all_compositions.items():
                if values:
                    composition_metrics[f'{species}_mean'] = np.mean(values)
                    composition_metrics[f'{species}_std'] = np.std(values)
                    composition_metrics[f'{species}_max'] = np.max(values)
                    composition_metrics[f'{species}_min'] = np.min(values)
            
            return composition_metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate temporal composition metrics: {e}")
            return {}
    
    def _calculate_temporal_efficiency_metrics(self, pathways: List[GasMigrationPathway]) -> Dict[str, float]:
        """Calculate temporal pathway efficiency metrics"""
        try:
            if not pathways:
                return {'avg_efficiency': 0.0}
            
            efficiencies = [pathway.efficiency_score for pathway in pathways]
            
            return {
                'avg_efficiency': np.mean(efficiencies),
                'max_efficiency': np.max(efficiencies),
                'min_efficiency': np.min(efficiencies),
                'efficiency_std': np.std(efficiencies),
                'high_efficiency_count': len([e for e in efficiencies if e > 0.7])
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate temporal efficiency metrics: {e}")
            return {'avg_efficiency': 0.0}
    
    def _analyze_temporal_trends(self, temporal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in gas migration pathways"""
        try:
            timestamps = temporal_results.get('timestamps', [])
            if len(timestamps) < 2:
                return {}
            
            # Convert timestamps to numerical values for trend analysis
            time_values = [(t - timestamps[0]).total_seconds() / 3600.0 for t in timestamps]  # Hours
            
            trend_analysis = {}
            
            # Analyze flux trends
            flux_evolution = temporal_results.get('gas_flux_evolution', {})
            if flux_evolution:
                total_fluxes = []
                for timestamp_str in flux_evolution.keys():
                    flux_data = flux_evolution[timestamp_str]
                    total_fluxes.append(flux_data.get('total_flux', 0.0))
                
                if len(total_fluxes) == len(time_values):
                    flux_trend = self._calculate_linear_trend(time_values, total_fluxes)
                    trend_analysis['total_flux_trend'] = flux_trend
            
            # Analyze efficiency trends
            efficiency_evolution = temporal_results.get('efficiency_evolution', {})
            if efficiency_evolution:
                avg_efficiencies = []
                for timestamp_str in efficiency_evolution.keys():
                    eff_data = efficiency_evolution[timestamp_str]
                    avg_efficiencies.append(eff_data.get('avg_efficiency', 0.0))
                
                if len(avg_efficiencies) == len(time_values):
                    efficiency_trend = self._calculate_linear_trend(time_values, avg_efficiencies)
                    trend_analysis['efficiency_trend'] = efficiency_trend
            
            # Analyze composition trends
            composition_evolution = temporal_results.get('composition_evolution', {})
            if composition_evolution:
                co2_means = []
                for timestamp_str in composition_evolution.keys():
                    comp_data = composition_evolution[timestamp_str]
                    co2_means.append(comp_data.get('CO2_mean', 0.0))
                
                if len(co2_means) == len(time_values):
                    co2_trend = self._calculate_linear_trend(time_values, co2_means)
                    trend_analysis['CO2_trend'] = co2_trend
            
            return trend_analysis
            
        except Exception as e:
            logger.warning(f"Failed to analyze temporal trends: {e}")
            return {}
    
    async def _generate_pathway_analysis_section(self, pathways: List[GasMigrationPathway]) -> Dict[str, Any]:
        """Generate pathway analysis section for report"""
        try:
            if not pathways:
                return {'summary': 'No active pathways identified'}
            
            # Calculate pathway statistics
            total_pathways = len(pathways)
            efficiency_scores = [p.efficiency_score for p in pathways]
            flux_values = [self._calculate_gas_flux(p) for p in pathways]
            
            # Identify most significant pathways
            sorted_pathways = sorted(pathways, key=lambda p: p.efficiency_score, reverse=True)
            top_pathways = sorted_pathways[:5]  # Top 5 pathways
            
            # Analyze pathway characteristics
            pathway_lengths = []
            pathway_depths = []
            
            for pathway in pathways:
                if pathway.pathway_nodes:
                    # Calculate total pathway length
                    total_length = 0.0
                    for i in range(1, len(pathway.pathway_nodes)):
                        segment_length = np.linalg.norm(
                            np.array(pathway.pathway_nodes[i]) - np.array(pathway.pathway_nodes[i-1])
                        )
                        total_length += segment_length
                    pathway_lengths.append(total_length)
                    
                    # Maximum depth
                    depths = [node[2] for node in pathway.pathway_nodes]
                    pathway_depths.append(max(depths))
            
            analysis = {
                'summary_statistics': {
                    'total_pathways': total_pathways,
                    'avg_efficiency': np.mean(efficiency_scores),
                    'total_gas_flux': sum(flux_values),
                    'avg_pathway_length': np.mean(pathway_lengths) if pathway_lengths else 0,
                    'avg_pathway_depth': np.mean(pathway_depths) if pathway_depths else 0
                },
                'top_pathways': [
                    {
                        'pathway_id': p.pathway_id,
                        'efficiency_score': p.efficiency_score,
                        'gas_flux': self._calculate_gas_flux(p),
                        'source_depth': p.source_location[2],
                        'surface_location': p.surface_location
                    }
                    for p in top_pathways
                ],
                'pathway_distribution': {
                    'high_efficiency': len([e for e in efficiency_scores if e > 0.7]),
                    'medium_efficiency': len([e for e in efficiency_scores if 0.3 < e <= 0.7]),
                    'low_efficiency': len([e for e in efficiency_scores if e <= 0.3])
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to generate pathway analysis section: {e}")
            return {}
    
    async def _generate_structural_controls_section(self, pathways: List[GasMigrationPathway]) -> Dict[str, Any]:
        """Generate structural controls analysis section"""
        try:
            if not pathways:
                return {'summary': 'No structural controls identified'}
            
            # Aggregate all structural controls
            all_controls = []
            for pathway in pathways:
                all_controls.extend(pathway.structural_controls)
            
            # Analyze control types
            control_types = {}
            for control in all_controls:
                control_type = control.split('_')[0]
                control_types[control_type] = control_types.get(control_type, 0) + 1
            
            # Sort by frequency
            sorted_controls = sorted(control_types.items(), key=lambda x: x[1], reverse=True)
            
            # Analyze fracture controls specifically
            fracture_controls = [c for c in all_controls if c.startswith('fracture')]
            fault_controls = [c for c in all_controls if 'fault' in c.lower()]
            lithology_controls = [c for c in all_controls if c.startswith('lithology')]
            
            structural_analysis = {
                'control_frequency': dict(sorted_controls),
                'primary_controls': [control[0] for control in sorted_controls[:3]],
                'fracture_analysis': {
                    'total_fracture_controls': len(fracture_controls),
                    'fracture_influence_range': self._analyze_fracture_influence_range(fracture_controls)
                },
                'fault_analysis': {
                    'total_fault_controls': len(fault_controls),
                    'fault_types': self._classify_fault_types(fault_controls)
                },
                'lithological_analysis': {
                    'total_lithology_controls': len(lithology_controls),
                    'rock_type_distribution': self._analyze_rock_type_distribution(lithology_controls)
                }
            }
            
            return structural_analysis
            
        except Exception as e:
            logger.warning(f"Failed to generate structural controls section: {e}")
            return {}
    
    def _analyze_fracture_influence_range(self, fracture_controls: List[str]) -> Dict[str, float]:
        """Analyze fracture influence range from control descriptions"""
        try:
            distances = []
            influences = []
            
            for control in fracture_controls:
                # Extract distance and influence from control string
                parts = control.split('_')
                for part in parts:
                    if part.endswith('m'):  # Distance in meters
                        try:
                            distance = float(part[:-1])
                            distances.append(distance)
                        except ValueError:
                            pass
                    elif part.startswith('influence'):
                        try:
                            influence = float(part.replace('influence', ''))
                            influences.append(influence)
                        except ValueError:
                            pass
            
            return {
                'avg_influence_distance': np.mean(distances) if distances else 0.0,
                'max_influence_distance': np.max(distances) if distances else 0.0,
                'avg_influence_score': np.mean(influences) if influences else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze fracture influence range: {e}")
            return {}
    
    def _classify_fault_types(self, fault_controls: List[str]) -> Dict[str, int]:
        """Classify fault types from control descriptions"""
        try:
            fault_types = {'normal': 0, 'reverse': 0, 'strike_slip': 0, 'unknown': 0}
            
            for control in fault_controls:
                control_lower = control.lower()
                if 'normal' in control_lower:
                    fault_types['normal'] += 1
                elif 'reverse' in control_lower or 'thrust' in control_lower:
                    fault_types['reverse'] += 1
                elif 'strike' in control_lower or 'slip' in control_lower:
                    fault_types['strike_slip'] += 1
                else:
                    fault_types['unknown'] += 1
            
            return fault_types
            
        except Exception as e:
            logger.warning(f"Failed to classify fault types: {e}")
            return {}
    
    def _analyze_rock_type_distribution(self, lithology_controls: List[str]) -> Dict[str, int]:
        """Analyze rock type distribution from lithological controls"""
        try:
            rock_types = {}
            
            for control in lithology_controls:
                # Extract rock type from control string
                parts = control.split('_')
                if len(parts) > 1:
                    rock_type = parts[1]  # Assuming format: lithology_rocktype_...
                    rock_types[rock_type] = rock_types.get(rock_type, 0) + 1
            
            return rock_types
            
        except Exception as e:
            logger.warning(f"Failed to analyze rock type distribution: {e}")
            return {}
    
    async def _generate_gas_composition_section(self, pathways: List[GasMigrationPathway]) -> Dict[str, Any]:
        """Generate gas composition analysis section"""
        try:
            if not pathways:
                return {'summary': 'No gas composition data available'}
            
            # Aggregate composition data from all pathways
            composition_data = {}
            
            for pathway in pathways:
                for species, evolution in pathway.gas_composition_evolution.items():
                    if species not in composition_data:
                        composition_data[species] = []
                    
                    if len(evolution) > 0:
                        # Use surface composition (final values)
                        composition_data[species].append(evolution[-1])
            
            # Calculate composition statistics
            composition_stats = {}
            for species, values in composition_data.items():
                if values:
                    composition_stats[species] = {
                        'mean_fraction': np.mean(values),
                        'std_fraction': np.std(values),
                        'min_fraction': np.min(values),
                        'max_fraction': np.max(values),
                        'pathways_present': len(values)
                    }
            
            # Identify dominant gas species
            if composition_stats:
                dominant_species = max(composition_stats.keys(), 
                                     key=lambda s: composition_stats[s]['mean_fraction'])
            else:
                dominant_species = 'unknown'
            
            # Analyze composition evolution patterns
            evolution_patterns = self._analyze_composition_evolution_patterns(pathways)
            
            gas_analysis = {
                'composition_statistics': composition_stats,
                'dominant_species': dominant_species,
                'total_pathways_analyzed': len(pathways),
                'evolution_patterns': evolution_patterns,
                'hazardous_species_assessment': self._assess_hazardous_species(composition_stats)
            }
            
            return gas_analysis
            
        except Exception as e:
            logger.warning(f"Failed to generate gas composition section: {e}")
            return {}
    
    def _analyze_composition_evolution_patterns(self, pathways: List[GasMigrationPathway]) -> Dict[str, Any]:
        """Analyze gas composition evolution patterns along pathways"""
        try:
            evolution_patterns = {}
            
            for pathway in pathways:
                for species, evolution in pathway.gas_composition_evolution.items():
                    if len(evolution) > 1:
                        # Calculate evolution trend
                        initial_fraction = evolution[0]
                        final_fraction = evolution[-1]
                        change = final_fraction - initial_fraction
                        
                        if species not in evolution_patterns:
                            evolution_patterns[species] = {'increases': 0, 'decreases': 0, 'stable': 0}
                        
                        if abs(change) < 0.01:  # Stable (< 1% change)
                            evolution_patterns[species]['stable'] += 1
                        elif change > 0:
                            evolution_patterns[species]['increases'] += 1
                        else:
                            evolution_patterns[species]['decreases'] += 1
            
            return evolution_patterns
            
        except Exception as e:
            logger.warning(f"Failed to analyze composition evolution patterns: {e}")
            return {}
    
    def _assess_hazardous_species(self, composition_stats: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Assess hazardous gas species concentrations"""
        try:
            hazard_assessment = {}
            
            # Define hazard thresholds (volume fractions)
            hazard_thresholds = {
                'CO2': {'low': 0.1, 'moderate': 0.3, 'high': 0.5},
                'SO2': {'low': 0.01, 'moderate': 0.05, 'high': 0.1},
                'H2S': {'low': 0.001, 'moderate': 0.01, 'high': 0.05},
                'CO': {'low': 0.001, 'moderate': 0.01, 'high': 0.05}
            }
            
            for species, stats in composition_stats.items():
                if species in hazard_thresholds:
                    mean_fraction = stats['mean_fraction']
                    thresholds = hazard_thresholds[species]
                    
                    if mean_fraction < thresholds['low']:
                        hazard_level = 'minimal'
                    elif mean_fraction < thresholds['moderate']:
                        hazard_level = 'low'
                    elif mean_fraction < thresholds['high']:
                        hazard_level = 'moderate'
                    else:
                        hazard_level = 'high'
                    
                    hazard_assessment[species] = hazard_level
            
            return hazard_assessment
            
        except Exception as e:
            logger.warning(f"Failed to assess hazardous species: {e}")
            return {}
    
    async def _generate_hazard_assessment_section(self, volcanic_system: VolcanicSystem,
                                                pathways: List[GasMigrationPathway]) -> Dict[str, Any]:
        """Generate hazard assessment section"""
        try:
            if not pathways:
                return {'overall_hazard_level': 'minimal', 'summary': 'No active pathways'}
            
            # Calculate gas flux hazards
            total_flux = sum(self._calculate_gas_flux(p) for p in pathways)
            flux_hazard = self._assess_flux_hazard_level(total_flux)
            
            # Assess pathway proximity to populated areas
            proximity_hazard = await self._assess_proximity_hazard(volcanic_system, pathways)
            
            # Assess gas composition hazards
            composition_hazard = self._assess_composition_hazard(pathways)
            
            # Calculate overall hazard level
            hazard_scores = {
                'flux': self._hazard_level_to_score(flux_hazard),
                'proximity': self._hazard_level_to_score(proximity_hazard),
                'composition': self._hazard_level_to_score(composition_hazard)
            }
            
            overall_score = np.mean(list(hazard_scores.values()))
            overall_hazard = self._score_to_hazard_level(overall_score)
            
            hazard_assessment = {
                'overall_hazard_level': overall_hazard,
                'component_hazards': {
                    'gas_flux_hazard': flux_hazard,
                    'proximity_hazard': proximity_hazard,
                    'composition_hazard': composition_hazard
                },
                'hazard_scores': hazard_scores,
                'risk_factors': self._identify_risk_factors(pathways),
                'mitigation_recommendations': self._generate_mitigation_recommendations(overall_hazard)
            }
            
            return hazard_assessment
            
        except Exception as e:
            logger.warning(f"Failed to generate hazard assessment section: {e}")
            return {'overall_hazard_level': 'unknown'}
    
    def _assess_flux_hazard_level(self, total_flux: float) -> str:
        """Assess hazard level based on total gas flux"""
        try:
            # Flux thresholds in m³/s
            if total_flux < 1e-6:
                return 'minimal'
            elif total_flux < 1e-4:
                return 'low'
            elif total_flux < 1e-2:
                return 'moderate'
            elif total_flux < 1e-1:
                return 'high'
            else:
                return 'critical'
                
        except Exception as e:
            logger.warning(f"Failed to assess flux hazard level: {e}")
            return 'unknown'
    
    async def _assess_proximity_hazard(self, volcanic_system: VolcanicSystem,
                                     pathways: List[GasMigrationPathway]) -> str:
        """Assess hazard based on proximity to populated areas"""
        try:
            # Query populated areas near volcanic system
            population_query = """
            SELECT name, population, ST_AsText(geometry) as geometry_wkt
            FROM populated_places 
            WHERE ST_DWithin(geometry, ST_GeomFromText(%s), %s)
            ORDER BY population DESC
            """
            
            system_center = f"POINT({volcanic_system.longitude} {volcanic_system.latitude})"
            search_radius = 50000.0  # 50 km
            
            populated_areas = await self.db_manager.execute_query(
                population_query, (system_center, search_radius)
            )
            
            if not populated_areas:
                return 'minimal'
            
            # Calculate minimum distance from pathways to populated areas
            min_distances = []
            
            for pathway in pathways:
                pathway_surface = pathway.surface_location
                
                for area in populated_areas:
                    # Simplified distance calculation (should use proper geometric distance)
                    area_coords = (0, 0)  # Would extract from geometry_wkt in real implementation
                    distance = np.sqrt(
                        (pathway_surface[0] - area_coords[0])**2 +
                        (pathway_surface[1] - area_coords[1])**2
                    )
                    min_distances.append(distance)
            
            if min_distances:
                min_distance = min(min_distances)
                
                # Distance-based hazard assessment
                if min_distance > 20000:  # > 20 km
                    return 'minimal'
                elif min_distance > 10000:  # > 10 km
                    return 'low'
                elif min_distance > 5000:   # > 5 km
                    return 'moderate'
                elif min_distance > 1000:   # > 1 km
                    return 'high'
                else:
                    return 'critical'
            
            return 'low'
            
        except Exception as e:
            logger.warning(f"Failed to assess proximity hazard: {e}")
            return 'unknown'
    
    def _assess_composition_hazard(self, pathways: List[GasMigrationPathway]) -> str:
        """Assess hazard based on gas composition"""
        try:
            max_hazard_score = 0.0
            
            for pathway in pathways:
                for species, evolution in pathway.gas_composition_evolution.items():
                    if len(evolution) > 0:
                        surface_fraction = evolution[-1]  # Surface concentration
                        
                        # Species-specific hazard scoring
                        if species == 'CO2' and surface_fraction > 0.3:
                            max_hazard_score = max(max_hazard_score, 0.8)
                        elif species == 'SO2' and surface_fraction > 0.05:
                            max_hazard_score = max(max_hazard_score, 0.9)
                        elif species == 'H2S' and surface_fraction > 0.01:
                            max_hazard_score = max(max_hazard_score, 1.0)
                        elif species == 'CO' and surface_fraction > 0.01:
                            max_hazard_score = max(max_hazard_score, 0.7)
            
            return self._score_to_hazard_level(max_hazard_score)
            
        except Exception as e:
            logger.warning(f"Failed to assess composition hazard: {e}")
            return 'unknown'
    
    def _hazard_level_to_score(self, hazard_level: str) -> float:
        """Convert hazard level to numerical score"""
        hazard_scores = {
            'minimal': 0.1,
            'low': 0.3,
            'moderate': 0.5,
            'high': 0.8,
            'critical': 1.0,
            'unknown': 0.5
        }
        return hazard_scores.get(hazard_level, 0.5)
    
    def _score_to_hazard_level(self, score: float) -> str:
        """Convert numerical score to hazard level"""
        if score < 0.2:
            return 'minimal'
        elif score < 0.4:
            return 'low'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.9:
            return 'high'
        else:
            return 'critical'
    
    def _identify_risk_factors(self, pathways: List[GasMigrationPathway]) -> List[str]:
        """Identify key risk factors from pathway analysis"""
        try:
            risk_factors = []
            
            # High efficiency pathways
            high_eff_count = len([p for p in pathways if p.efficiency_score > 0.7])
            if high_eff_count > 0:
                risk_factors.append(f"{high_eff_count} high-efficiency gas migration pathways")
            
            # Multiple active pathways
            if len(pathways) > 5:
                risk_factors.append(f"Multiple active pathways ({len(pathways)} total)")
            
            # Shallow source depths
            shallow_sources = [p for p in pathways if p.source_location[2] < 2000]
            if shallow_sources:
                risk_factors.append(f"{len(shallow_sources)} pathways from shallow sources (<2km)")
            
            # High gas flux pathways
            high_flux_count = len([p for p in pathways if self._calculate_gas_flux(p) > 1e-4])
            if high_flux_count > 0:
                risk_factors.append(f"{high_flux_count} high gas flux pathways")
            
            return risk_factors
            
        except Exception as e:
            logger.warning(f"Failed to identify risk factors: {e}")
            return []
    
    def _generate_mitigation_recommendations(self, hazard_level: str) -> List[str]:
        """Generate mitigation recommendations based on hazard level"""
        try:
            recommendations = []
            
            if hazard_level in ['high', 'critical']:
                recommendations.extend([
                    "Implement continuous gas monitoring at surface emission points",
                    "Establish evacuation procedures for nearby populated areas",
                    "Install automated gas detection and alert systems",
                    "Conduct regular air quality assessments"
                ])
            
            if hazard_level in ['moderate', 'high', 'critical']:
                recommendations.extend([
                    "Increase monitoring frequency of gas emissions",
                    "Map gas accumulation zones and restrict access",
                    "Develop emergency response protocols"
                ])
            
            if hazard_level != 'minimal':
                recommendations.extend([
                    "Maintain baseline gas monitoring network",
                    "Regular pathway model updates with new data",
                    "Public awareness and education programs"
                ])
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate mitigation recommendations: {e}")
            return []
    
    async def _generate_recommendations_section(self, volcanic_system: VolcanicSystem,
                                             pathways: List[GasMigrationPathway],
                                             analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations section"""
        try:
            # Monitoring recommendations
            monitoring_recs = await self._generate_monitoring_recommendations(volcanic_system, pathways)
            
            # Research recommendations
            research_recs = self._generate_research_recommendations(pathways, analysis_results)
            
            # Operational recommendations
            operational_recs = self._generate_operational_recommendations(pathways)
            
            recommendations = {
                'monitoring_recommendations': monitoring_recs,
                'research_recommendations': research_recs,
                'operational_recommendations': operational_recs,
                'priority_actions': self._identify_priority_actions(pathways, analysis_results),
                'timeline': self._generate_implementation_timeline()
            }
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations section: {e}")
            return {}
    
    async def _generate_monitoring_recommendations(self, volcanic_system: VolcanicSystem,
                                                pathways: List[GasMigrationPathway]) -> List[str]:
        """Generate monitoring recommendations"""
        try:
            recommendations = []
            
            if pathways:
                # Gas monitoring recommendations
                recommendations.append("Install additional gas monitoring stations at pathway surface locations")
                recommendations.append("Implement real-time CO2 and SO2 flux measurements")
                
                # Seismic monitoring
                recommendations.append("Enhance seismic monitoring network density")
                recommendations.append("Deploy broadband seismometers for fracture detection")
                
                # Deformation monitoring
                recommendations.append("Install GPS stations near high-efficiency pathways")
                recommendations.append("Increase InSAR monitoring frequency")
                
                # Geochemical monitoring
                recommendations.append("Conduct regular isotopic analysis of gas emissions")
                recommendations.append("Monitor groundwater chemistry changes")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate monitoring recommendations: {e}")
            return []
    
    def _generate_research_recommendations(self, pathways: List[GasMigrationPathway],
                                         analysis_results: Dict[str, Any]) -> List[str]:
        """Generate research recommendations"""
        try:
            recommendations = []
            
            # Pathway modeling improvements
            recommendations.append("Refine fracture network models with high-resolution geophysical data")
            recommendations.append("Develop machine learning models for pathway prediction")
            
            # Uncertainty reduction
            uncertainty_analysis = analysis_results.get('uncertainty_analysis', {})
            if uncertainty_analysis:
                recommendations.append("Reduce parameter uncertainties through targeted field studies")
                recommendations.append("Validate pathway models with tracer gas experiments")
            
            # Temporal analysis
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            if temporal_analysis:
                recommendations.append("Investigate temporal pathway evolution mechanisms")
                recommendations.append("Study correlation between seismic activity and pathway changes")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate research recommendations: {e}")
            return []
    
    def _generate_operational_recommendations(self, pathways: List[GasMigrationPathway]) -> List[str]:
        """Generate operational recommendations"""
        try:
            recommendations = []
            
            if pathways:
                # Model updates
                recommendations.append("Update pathway models monthly with new monitoring data")
                recommendations.append("Recalibrate models after significant seismic events")
                
                # Data management
                recommendations.append("Implement automated data quality control procedures")
                recommendations.append("Establish data backup and archival protocols")
                
                # Communication
                recommendations.append("Develop stakeholder communication protocols")
                recommendations.append("Create public information dashboards")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate operational recommendations: {e}")
            return []
    
    def _identify_priority_actions(self, pathways: List[GasMigrationPathway],
                                 analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify priority actions with urgency levels"""
        try:
            priority_actions = []
            
            # High-priority actions based on pathway characteristics
            high_eff_pathways = [p for p in pathways if p.efficiency_score > 0.8]
            if high_eff_pathways:
                priority_actions.append({
                    'action': f"Install gas monitoring at {len(high_eff_pathways)} high-efficiency pathway locations",
                    'urgency': 'high',
                    'timeline': '1-2 weeks'
                })
            
            # Medium-priority actions
            if len(pathways) > 3:
                priority_actions.append({
                    'action': "Enhance seismic monitoring network",
                    'urgency': 'medium',
                    'timeline': '1-3 months'
                })
            
            # Low-priority actions
            priority_actions.append({
                'action': "Conduct comprehensive geochemical survey",
                'urgency': 'low',
                'timeline': '6-12 months'
            })
            
            return priority_actions
            
        except Exception as e:
            logger.warning(f"Failed to identify priority actions: {e}")
            return []
    
    def _generate_implementation_timeline(self) -> Dict[str, List[str]]:
        """Generate implementation timeline for recommendations"""
        try:
            timeline = {
                'immediate (1-2 weeks)': [
                    "Deploy emergency gas monitoring equipment",
                    "Activate enhanced monitoring protocols"
                ],
                'short-term (1-3 months)': [
                    "Install permanent monitoring stations",
                    "Implement automated alert systems",
                    "Conduct stakeholder meetings"
                ],
                'medium-term (3-12 months)': [
                    "Expand monitoring network",
                    "Develop predictive models",
                    "Conduct validation studies"
                ],
                'long-term (1-3 years)': [
                    "Establish comprehensive monitoring infrastructure",
                    "Develop advanced forecasting capabilities",
                    "Implement community preparedness programs"
                ]
            }
            
            return timeline
            
        except Exception as e:
            logger.warning(f"Failed to generate implementation timeline: {e}")
            return {}
    
    async def _generate_technical_appendix(self, pathways: List[GasMigrationPathway],
                                         analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical appendix with detailed methodology and data"""
        try:
            appendix = {
                'methodology': {
                    'fracture_network_modeling': "3D fracture network constructed from structural geology database using geometric compatibility algorithms",
                    'permeability_field_modeling': "RBF interpolation of rock property measurements with geological constraints",
                    'pressure_flow_simulation': "Finite difference solution of steady-state pressure equation with Darcy velocity calculation",
                    'streamline_tracing': "4th-order Runge-Kutta integration for pathway determination",
                    'uncertainty_quantification': "Monte Carlo simulation with parameter uncertainty distributions"
                },
                'model_parameters': {
                    'grid_resolution': '100x100x50 cells',
                    'grid_spacing': '100m x 100m x 100m',
                    'convergence_tolerance': '1 Pa pressure difference',
                    'monte_carlo_samples': '1000 realizations',
                    'fracture_connection_threshold': '0.3 probability'
                },
                'data_sources': {
                    'structural_geology': 'PostgreSQL geological database',
                    'rock_properties': 'Laboratory measurements and field surveys',
                    'seismic_data': 'Real-time monitoring network',
                    'gas_measurements': 'Multi-component gas analyzers',
                    'deformation_data': 'GPS and InSAR measurements'
                },
                'validation_metrics': self._calculate_validation_metrics(pathways, analysis_results),
                'computational_performance': self._get_computational_performance_metrics()
            }
            
            return appendix
            
        except Exception as e:
            logger.warning(f"Failed to generate technical appendix: {e}")
            return {}
    
    def _calculate_validation_metrics(self, pathways: List[GasMigrationPathway],
                                    analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model validation metrics"""
        try:
            validation_metrics = {}
            
            if pathways:
                # Pathway consistency metrics
                efficiency_scores = [p.efficiency_score for p in pathways]
                validation_metrics['avg_pathway_efficiency'] = np.mean(efficiency_scores)
                validation_metrics['efficiency_std'] = np.std(efficiency_scores)
                
                # Flux conservation check
                total_source_flux = len(pathways) * 1e-4  # Assumed source flux
                total_surface_flux = sum(self._calculate_gas_flux(p) for p in pathways)
                validation_metrics['flux_conservation_ratio'] = total_surface_flux / total_source_flux if total_source_flux > 0 else 0
                
                # Uncertainty metrics
                uncertainty_analysis = analysis_results.get('uncertainty_analysis', {})
                if uncertainty_analysis:
                    # Average coefficient of variation across all pathways
                    cv_values = []
                    for pathway_id, uncertainty in uncertainty_analysis.items():
                        for metric, stats in uncertainty.items():
                            if isinstance(stats, dict) and 'coefficient_of_variation' in stats:
                                cv_values.append(stats['coefficient_of_variation'])
                    
                    if cv_values:
                        validation_metrics['avg_uncertainty_cv'] = np.mean(cv_values)
            
            return validation_metrics
            
        except Exception as e:
            logger.warning(f"Failed to calculate validation metrics: {e}")
            return {}
    
    def _get_computational_performance_metrics(self) -> Dict[str, str]:
        """Get computational performance metrics"""
        try:
            return {
                'fracture_network_nodes': str(len(self.fracture_network.nodes()) if self.fracture_network else 0),
                'fracture_network_edges': str(len(self.fracture_network.edges()) if self.fracture_network else 0),
                'permeability_field_size': f"{self.permeability_field.shape}" if self.permeability_field is not None else "Not computed",
                'pressure_field_size': f"{self.pressure_field.shape}" if self.pressure_field is not None else "Not computed",
                'typical_computation_time': "5-15 minutes for standard analysis",
                'memory_usage': "~2-8 GB depending on grid resolution"
            }
            
        except Exception as e:
            logger.warning(f"Failed to get computational performance metrics: {e}")
            return {}
    
    async def _identify_critical_observations(self, pathways: List[GasMigrationPathway],
                                           analysis_results: Dict[str, Any]) -> List[str]:
        """Identify critical observations from pathway analysis"""
        try:
            observations = []
            
            if not pathways:
                observations.append("No active gas migration pathways detected")
                return observations
            
            # High-efficiency pathways
            high_eff_pathways = [p for p in pathways if p.efficiency_score > 0.8]
            if high_eff_pathways:
                observations.append(f"{len(high_eff_pathways)} pathways show very high migration efficiency (>80%)")
            
            # Shallow source pathways
            shallow_pathways = [p for p in pathways if p.source_location[2] < 1000]
            if shallow_pathways:
                observations.append(f"{len(shallow_pathways)} pathways originate from shallow sources (<1km depth)")
            
            # Temporal trends
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            if temporal_analysis:
                flux_trend = temporal_analysis.get('total_flux_trend', {})
                if flux_trend.get('slope', 0) > 0.1:
                    observations.append("Gas flux shows significant increasing trend over time")
                elif flux_trend.get('slope', 0) < -0.1:
                    observations.append("Gas flux shows significant decreasing trend over time")
            
            # High uncertainty pathways
            uncertainty_analysis = analysis_results.get('uncertainty_analysis', {})
            if uncertainty_analysis:
                high_uncertainty_count = 0
                for pathway_id, uncertainty in uncertainty_analysis.items():
                    for metric, stats in uncertainty.items():
                        if isinstance(stats, dict) and stats.get('coefficient_of_variation', 0) > 0.5:
                            high_uncertainty_count += 1
                            break
                
                if high_uncertainty_count > 0:
                    observations.append(f"{high_uncertainty_count} pathways show high parameter uncertainty")
            
            return observations
            
        except Exception as e:
            logger.warning(f"Failed to identify critical observations: {e}")
            return []
    
    async def _identify_immediate_concerns(self, pathways: List[GasMigrationPathway],
                                         analysis_results: Dict[str, Any]) -> List[str]:
        """Identify immediate concerns requiring attention"""
        try:
            concerns = []
            
            if not pathways:
                return concerns
            
            # Very high gas flux
            high_flux_pathways = [p for p in pathways if self._calculate_gas_flux(p) > 1e-3]
            if high_flux_pathways:
                concerns.append(f"Extremely high gas flux detected in {len(high_flux_pathways)} pathways - immediate monitoring required")
            
            # Hazardous gas concentrations
            for pathway in pathways:
                for species, evolution in pathway.gas_composition_evolution.items():
                    if len(evolution) > 0:
                        surface_conc = evolution[-1]
                        if species == 'H2S' and surface_conc > 0.05:
                            concerns.append(f"Dangerous H2S concentrations (>{surface_conc:.1%}) detected at pathway {pathway.pathway_id}")
                        elif species == 'CO2' and surface_conc > 0.5:
                            concerns.append(f"Lethal CO2 concentrations (>{surface_conc:.1%}) detected at pathway {pathway.pathway_id}")
            
            # Rapid temporal changes
            temporal_analysis = analysis_results.get('temporal_analysis', {})
            if temporal_analysis:
                flux_trend = temporal_analysis.get('total_flux_trend', {})
                if abs(flux_trend.get('slope', 0)) > 0.5:
                    concerns.append("Rapid changes in gas flux detected - enhanced monitoring recommended")
            
            # Multiple converging pathways
            surface_locations = [p.surface_location for p in pathways]
            if len(surface_locations) > 1:
                # Check for clustering of surface locations
                distances = []
                for i in range(len(surface_locations)):
                    for j in range(i+1, len(surface_locations)):
                        dist = np.sqrt(
                            (surface_locations[i][0] - surface_locations[j][0])**2 +
                            (surface_locations[i][1] - surface_locations[j][1])**2
                        )
                        distances.append(dist)
                
                if distances and min(distances) < 500:  # Within 500m
                    concerns.append("Multiple pathways converging in small area - potential gas accumulation zone")
            
            return concerns
            
        except Exception as e:
            logger.warning(f"Failed to identify immediate concerns: {e}")
            return []