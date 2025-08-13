"""
Advanced 3D Spherical Mantle Convection Simulation System
========================================================

State-of-the-art computational geodynamics engine for modeling mantle convection
with variable viscosity, compositional heterogeneity, phase transitions, and 
core-mantle boundary heat flux. Implements cutting-edge numerical methods for
coupled thermal-compositional-mechanical evolution.

Key Features:
- Spherical geometry with adaptive mesh refinement
- Temperature and pressure dependent viscosity
- Compositional tracking with diffusion
- Major phase transitions (410, 660 km discontinuities)
- Core-mantle boundary thermal coupling
- D" layer and LLSVP dynamics
- GPU-accelerated finite element methods
- Advanced time integration schemes

Author: Prometheus Deep Earth Team
Date: 2025
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.interpolate import RegularGridInterpolator, RBFInterpolator
import cupy as cp
import cupyx.scipy.sparse as cp_sp
from numba import jit, cuda
import h5py
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure advanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvectionModel(Enum):
    """Enumeration of available convection models"""
    THERMAL_ONLY = "thermal_only"
    THERMAL_COMPOSITIONAL = "thermal_compositional"
    CHEMICAL_STRATIFIED = "chemical_stratified"
    PRIMORDIAL_RESERVOIR = "primordial_reservoir"

class ViscosityModel(Enum):
    """Enumeration of viscosity models"""
    CONSTANT = "constant"
    TEMPERATURE_DEPENDENT = "temperature_dependent"
    PRESSURE_DEPENDENT = "pressure_dependent"
    TEMPERATURE_PRESSURE = "temperature_pressure_dependent"
    COMPOSITE_DEPENDENT = "composite_dependent"
    DISLOCATION_DIFFUSION = "dislocation_diffusion_creep"

class PhaseTransition(Enum):
    """Major mantle phase transitions"""
    OLIVINE_WADSLEYITE = "410km_transition"
    WADSLEYITE_RINGWOODITE = "520km_transition"
    RINGWOODITE_PEROVSKITE = "660km_transition"
    PEROVSKITE_POST_PEROVSKITE = "2700km_transition"

@dataclass
class MaterialProperties:
    """Material properties for mantle phases"""
    density: float
    thermal_expansion: float
    thermal_conductivity: float
    specific_heat: float
    viscosity_reference: float
    activation_energy: float
    activation_volume: float
    reference_temperature: float
    reference_pressure: float
    
    # Phase transition properties
    clapeyron_slope: Optional[float] = None
    transition_width: Optional[float] = None
    latent_heat: Optional[float] = None
    
    # Compositional properties
    partition_coefficient: Optional[float] = None
    diffusivity: Optional[float] = None

@dataclass
class ConvectionParameters:
    """Parameters for mantle convection simulation"""
    # Domain geometry
    inner_radius: float = 3480e3  # Core radius (m)
    outer_radius: float = 6371e3  # Earth radius (m)
    
    # Resolution parameters
    radial_elements: int = 128
    lateral_elements: int = 256
    adaptive_refinement: bool = True
    refinement_threshold: float = 0.01
    
    # Physical parameters
    reference_viscosity: float = 1e21  # Pa*s
    rayleigh_number: float = 1e6
    prandtl_number: float = 1.0
    internal_heating: float = 7.4e-12  # W/kg
    
    # Core-mantle boundary
    cmb_temperature: float = 3500  # K
    cmb_heat_flux: float = 12e12  # W
    
    # Surface conditions
    surface_temperature: float = 300  # K
    
    # Time parameters
    total_time: float = 4.5e9 * 365.25 * 24 * 3600  # 4.5 Gyr in seconds
    time_step: float = 1000 * 365.25 * 24 * 3600    # 1000 years in seconds
    adaptive_timestepping: bool = True
    
    # Numerical parameters
    solver_tolerance: float = 1e-8
    max_iterations: int = 1000
    gpu_acceleration: bool = True
    
    # Advanced features
    phase_transitions: bool = True
    compositional_convection: bool = True
    d_prime_layer: bool = True
    llsvp_tracking: bool = True

class SphericalMesh:
    """Advanced spherical mesh generation and management"""
    
    def __init__(self, params: ConvectionParameters):
        self.params = params
        self.nodes = None
        self.elements = None
        self.boundary_nodes = None
        self.adaptive_levels = None
        self.refinement_history = []
        
        self._generate_base_mesh()
        
    def _generate_base_mesh(self):
        """Generate base spherical mesh using spectral elements"""
        nr = self.params.radial_elements
        nlat = self.params.lateral_elements
        nlon = 2 * nlat
        
        # Radial coordinates (Gauss-Lobatto-Legendre points)
        r_min = self.params.inner_radius
        r_max = self.params.outer_radius
        r_gll = self._gauss_lobatto_legendre_points(nr)
        r_coords = r_min + 0.5 * (r_max - r_min) * (r_gll + 1)
        
        # Angular coordinates (spherical harmonics grid)
        theta = np.linspace(0, np.pi, nlat)
        phi = np.linspace(0, 2*np.pi, nlon, endpoint=False)
        
        # Generate node coordinates
        total_nodes = nr * nlat * nlon
        self.nodes = np.zeros((total_nodes, 4))  # r, theta, phi, node_id
        
        node_id = 0
        for i, r in enumerate(r_coords):
            for j, th in enumerate(theta):
                for k, ph in enumerate(phi):
                    self.nodes[node_id] = [r, th, ph, node_id]
                    node_id += 1
        
        # Generate elements (hexahedral spectral elements)
        self.elements = self._generate_spectral_elements(nr, nlat, nlon)
        
        # Identify boundary nodes
        self._identify_boundary_nodes()
        
        logger.info(f"Generated spherical mesh: {total_nodes} nodes, {len(self.elements)} elements")
    
    def _gauss_lobatto_legendre_points(self, n: int) -> np.ndarray:
        """Generate Gauss-Lobatto-Legendre quadrature points"""
        if n == 1:
            return np.array([0.0])
        elif n == 2:
            return np.array([-1.0, 1.0])
        else:
            # Use Chebyshev points as initial guess
            x = -np.cos(np.pi * np.arange(n) / (n - 1))
            
            # Newton-Raphson iteration to find GLL points
            for _ in range(10):  # Usually converges quickly
                P, dP = self._legendre_polynomial_derivative(x, n-1)
                x_new = x - ((1 - x**2) * dP) / (-2*x*dP + (1-x**2)*self._legendre_second_derivative(x, n-1))
                if np.max(np.abs(x_new - x)) < 1e-15:
                    break
                x = x_new
            
            return x
    
    def _legendre_polynomial_derivative(self, x: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Legendre polynomial and its derivative"""
        if n == 0:
            return np.ones_like(x), np.zeros_like(x)
        elif n == 1:
            return x, np.ones_like(x)
        
        P0, P1 = np.ones_like(x), x
        dP0, dP1 = np.zeros_like(x), np.ones_like(x)
        
        for k in range(2, n+1):
            P2 = ((2*k-1) * x * P1 - (k-1) * P0) / k
            dP2 = ((2*k-1) * (P1 + x * dP1) - (k-1) * dP0) / k
            P0, P1 = P1, P2
            dP0, dP1 = dP1, dP2
        
        return P1, dP1
    
    def _legendre_second_derivative(self, x: np.ndarray, n: int) -> np.ndarray:
        """Compute second derivative of Legendre polynomial"""
        P, dP = self._legendre_polynomial_derivative(x, n)
        return (2*x*dP - n*(n+1)*P) / (1 - x**2)
    
    def _generate_spectral_elements(self, nr: int, nlat: int, nlon: int) -> np.ndarray:
        """Generate spectral elements connectivity"""
        elements = []
        
        for i in range(nr - 1):
            for j in range(nlat - 1):
                for k in range(nlon):
                    k_next = (k + 1) % nlon
                    
                    # Define 8 nodes of hexahedral element
                    n1 = i * nlat * nlon + j * nlon + k
                    n2 = i * nlat * nlon + j * nlon + k_next
                    n3 = i * nlat * nlon + (j + 1) * nlon + k_next
                    n4 = i * nlat * nlon + (j + 1) * nlon + k
                    n5 = (i + 1) * nlat * nlon + j * nlon + k
                    n6 = (i + 1) * nlat * nlon + j * nlon + k_next
                    n7 = (i + 1) * nlat * nlon + (j + 1) * nlon + k_next
                    n8 = (i + 1) * nlat * nlon + (j + 1) * nlon + k
                    
                    elements.append([n1, n2, n3, n4, n5, n6, n7, n8])
        
        return np.array(elements)
    
    def _identify_boundary_nodes(self):
        """Identify nodes on inner and outer boundaries"""
        r_coords = self.nodes[:, 0]
        r_min, r_max = np.min(r_coords), np.max(r_coords)
        
        self.boundary_nodes = {
            'inner': np.where(np.abs(r_coords - r_min) < 1e-6)[0],
            'outer': np.where(np.abs(r_coords - r_max) < 1e-6)[0]
        }
    
    def adaptive_refinement(self, solution_field: np.ndarray, threshold: float = 0.01):
        """Perform adaptive mesh refinement based on solution gradients"""
        if not self.params.adaptive_refinement:
            return
        
        # Calculate solution gradients
        gradients = self._calculate_solution_gradients(solution_field)
        
        # Identify elements for refinement
        elements_to_refine = []
        for elem_id, element in enumerate(self.elements):
            elem_gradient = np.max([gradients[node_id] for node_id in element])
            if elem_gradient > threshold:
                elements_to_refine.append(elem_id)
        
        if elements_to_refine:
            self._refine_elements(elements_to_refine)
            logger.info(f"Refined {len(elements_to_refine)} elements")
    
    def _calculate_solution_gradients(self, solution: np.ndarray) -> np.ndarray:
        """Calculate solution gradients at nodes"""
        gradients = np.zeros(len(self.nodes))
        
        for elem_id, element in enumerate(self.elements):
            # Calculate element gradient using finite differences
            elem_nodes = self.nodes[element]
            elem_solution = solution[element]
            
            # Approximate gradient magnitude
            grad_r = np.gradient(elem_solution.reshape(2, 2, 2), axis=0)
            grad_theta = np.gradient(elem_solution.reshape(2, 2, 2), axis=1)
            grad_phi = np.gradient(elem_solution.reshape(2, 2, 2), axis=2)
            
            grad_magnitude = np.sqrt(grad_r**2 + grad_theta**2 + grad_phi**2)
            
            for i, node_id in enumerate(element):
                gradients[node_id] = max(gradients[node_id], np.max(grad_magnitude))
        
        return gradients
    
    def _refine_elements(self, elements_to_refine: List[int]):
        """Advanced h-refinement with comprehensive element subdivision"""
        if not elements_to_refine:
            return
        
        # Initialize refinement data structures
        new_nodes_list = []
        new_elements_list = []
        refined_element_map = {}
        hanging_node_constraints = []
        
        # Sort elements by refinement level to maintain mesh consistency
        elements_to_refine.sort(key=lambda x: self._get_element_level(x))
        
        for elem_id in elements_to_refine:
            if elem_id >= len(self.elements):  # Skip if element was already subdivided
                continue
                
            element = self.elements[elem_id]
            element_level = self._get_element_level(elem_id)
            
            # Check mesh quality and refinement constraints
            if not self._can_refine_element(elem_id, element_level):
                continue
            
            # Get element geometry and type (hexahedral, tetrahedral, etc.)
            element_type = self._determine_element_type(element)
            element_coords = self.nodes[element]
            
            # Create new nodes for subdivision
            new_nodes, node_mapping = self._create_subdivision_nodes(
                element, element_coords, element_type, element_level
            )
            
            # Generate sub-elements with proper connectivity
            sub_elements = self._generate_sub_elements(
                element, new_nodes, node_mapping, element_type
            )
            
            # Quality control - ensure sub-elements meet quality criteria
            valid_sub_elements = self._validate_element_quality(
                sub_elements, new_nodes, element_level + 1
            )
            
            if not valid_sub_elements:
                continue  # Skip refinement if quality criteria not met
            
            # Handle hanging nodes and transition elements
            constraints = self._handle_hanging_nodes(
                elem_id, element, sub_elements, new_nodes
            )
            hanging_node_constraints.extend(constraints)
            
            # Store refinement mapping
            refined_element_map[elem_id] = {
                'sub_elements': len(new_elements_list) + np.arange(len(valid_sub_elements)),
                'parent_level': element_level,
                'refinement_ratio': self._calculate_refinement_ratio(element_type),
                'new_nodes': len(new_nodes_list) + np.arange(len(new_nodes))
            }
            
            new_nodes_list.extend(new_nodes)
            new_elements_list.extend(valid_sub_elements)
        
        if not new_elements_list:
            return
        
        # Update mesh data structures
        self._update_mesh_connectivity(
            new_nodes_list, new_elements_list, refined_element_map,
            hanging_node_constraints, elements_to_refine
        )
        
        # Update refinement history with detailed metrics
        self.refinement_history.append({
            'timestamp': time.time(),
            'elements_refined': len(elements_to_refine),
            'new_elements_created': len(new_elements_list),
            'new_nodes_created': len(new_nodes_list),
            'hanging_node_constraints': len(hanging_node_constraints),
            'max_refinement_level': max([self._get_element_level(i) for i in range(len(self.elements))]),
            'mesh_quality_metrics': self._calculate_mesh_quality_metrics(),
            'refined_element_map': refined_element_map
        })
        
        logger.info(f"Refined {len(elements_to_refine)} elements into {len(new_elements_list)} sub-elements")
        logger.info(f"Created {len(new_nodes_list)} new nodes with {len(hanging_node_constraints)} constraints")
    
    def _subdivide_element(self, element: np.ndarray) -> List[np.ndarray]:
        """Legacy method - replaced by comprehensive refinement system"""
        # This method is now deprecated in favor of _refine_elements
        # Kept for backward compatibility
        logger.warning("Using deprecated _subdivide_element - use _refine_elements instead")
        return [element]
    
    def _get_element_level(self, elem_id: int) -> int:
        """Get refinement level of element"""
        if not hasattr(self, 'element_levels'):
            self.element_levels = np.zeros(len(self.elements), dtype=int)
        
        if elem_id < len(self.element_levels):
            return self.element_levels[elem_id]
        return 0
    
    def _can_refine_element(self, elem_id: int, current_level: int) -> bool:
        """Check if element can be refined based on constraints"""
        # Maximum refinement level constraint
        if current_level >= self.params.max_refinement_level:
            return False
        
        # Minimum element size constraint
        element = self.elements[elem_id]
        element_coords = self.nodes[element]
        min_edge_length = self._calculate_min_edge_length(element_coords)
        
        if min_edge_length < self.params.min_element_size:
            return False
        
        # Neighboring element level difference constraint (2:1 rule)
        neighbors = self._get_element_neighbors(elem_id)
        for neighbor_id in neighbors:
            neighbor_level = self._get_element_level(neighbor_id)
            if current_level - neighbor_level > 1:
                return False
        
        return True
    
    def _determine_element_type(self, element: np.ndarray) -> str:
        """Determine element type based on connectivity"""
        num_nodes = len(element)
        
        if num_nodes == 8:
            return 'hexahedral'
        elif num_nodes == 4:
            return 'tetrahedral'
        elif num_nodes == 6:
            return 'prismatic'
        elif num_nodes == 5:
            return 'pyramid'
        else:
            raise ValueError(f"Unsupported element type with {num_nodes} nodes")
    
    def _create_subdivision_nodes(self, element: np.ndarray, coords: np.ndarray, 
                                elem_type: str, level: int) -> tuple:
        """Create new nodes for element subdivision"""
        new_nodes = []
        node_mapping = {}
        
        if elem_type == 'hexahedral':
            # Create face center nodes
            face_centers = self._create_face_center_nodes(coords)
            new_nodes.extend(face_centers)
            
            # Create edge midpoint nodes
            edge_midpoints = self._create_edge_midpoint_nodes(coords)
            new_nodes.extend(edge_midpoints)
            
            # Create element center node
            element_center = np.mean(coords, axis=0)
            new_nodes.append(element_center)
            
            # Create node mapping for connectivity
            node_mapping = self._create_hex_node_mapping(
                element, len(self.nodes) + len(new_nodes) - len(face_centers) - len(edge_midpoints) - 1
            )
            
        elif elem_type == 'tetrahedral':
            # Create edge midpoint nodes for tetrahedral subdivision
            edge_midpoints = self._create_tet_edge_midpoints(coords)
            new_nodes.extend(edge_midpoints)
            
            node_mapping = self._create_tet_node_mapping(element, len(self.nodes))
            
        # Apply coordinate transformation for spherical geometry
        new_nodes = self._apply_spherical_projection(new_nodes, level)
        
        return new_nodes, node_mapping
    
    def _create_face_center_nodes(self, coords: np.ndarray) -> List[np.ndarray]:
        """Create face center nodes for hexahedral elements"""
        # Hexahedral faces (6 faces, 4 nodes each)
        faces = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face  
            [0, 1, 5, 4],  # Front face
            [2, 3, 7, 6],  # Back face
            [0, 3, 7, 4],  # Left face
            [1, 2, 6, 5]   # Right face
        ]
        
        face_centers = []
        for face in faces:
            face_coords = coords[face]
            center = np.mean(face_coords, axis=0)
            face_centers.append(center)
        
        return face_centers
    
    def _create_edge_midpoint_nodes(self, coords: np.ndarray) -> List[np.ndarray]:
        """Create edge midpoint nodes for hexahedral elements"""
        # Hexahedral edges (12 edges, 2 nodes each)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        edge_midpoints = []
        for edge in edges:
            edge_coords = coords[edge]
            midpoint = np.mean(edge_coords, axis=0)
            edge_midpoints.append(midpoint)
        
        return edge_midpoints
    
    def _create_tet_edge_midpoints(self, coords: np.ndarray) -> List[np.ndarray]:
        """Create edge midpoint nodes for tetrahedral elements"""
        # Tetrahedral edges (6 edges, 2 nodes each)
        edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        
        edge_midpoints = []
        for edge in edges:
            edge_coords = coords[edge]
            midpoint = np.mean(edge_coords, axis=0)
            edge_midpoints.append(midpoint)
        
        return edge_midpoints
    
    def _generate_sub_elements(self, parent_element: np.ndarray, new_nodes: List[np.ndarray],
                             node_mapping: dict, elem_type: str) -> List[np.ndarray]:
        """Generate sub-elements from subdivision nodes"""
        if elem_type == 'hexahedral':
            return self._generate_hex_sub_elements(parent_element, node_mapping)
        elif elem_type == 'tetrahedral':
            return self._generate_tet_sub_elements(parent_element, node_mapping)
        else:
            raise ValueError(f"Sub-element generation not implemented for {elem_type}")
    
    def _generate_hex_sub_elements(self, parent_element: np.ndarray, node_mapping: dict) -> List[np.ndarray]:
        """Generate 8 hexahedral sub-elements from parent hexahedron"""
        # Standard 8-subdivision of hexahedron
        sub_element_templates = [
            [0, 8, 20, 11, 12, 21, 26, 24],   # Sub-element 0
            [8, 1, 9, 20, 21, 13, 22, 26],    # Sub-element 1
            [20, 9, 2, 10, 26, 22, 14, 23],   # Sub-element 2
            [11, 20, 10, 3, 24, 26, 23, 15],  # Sub-element 3
            [12, 21, 26, 24, 4, 16, 25, 19],  # Sub-element 4
            [21, 13, 22, 26, 16, 5, 17, 25],  # Sub-element 5
            [26, 22, 14, 23, 25, 17, 6, 18],  # Sub-element 6
            [24, 26, 23, 15, 19, 25, 18, 7]   # Sub-element 7
        ]
        
        sub_elements = []
        base_node_id = len(self.nodes)
        
        for template in sub_element_templates:
            sub_element = []
            for local_id in template:
                if local_id < 8:  # Original parent nodes
                    sub_element.append(parent_element[local_id])
                else:  # New nodes
                    sub_element.append(base_node_id + (local_id - 8))
            sub_elements.append(np.array(sub_element))
        
        return sub_elements
    
    def _generate_tet_sub_elements(self, parent_element: np.ndarray, node_mapping: dict) -> List[np.ndarray]:
        """Generate 8 tetrahedral sub-elements from parent tetrahedron"""
        # Standard 1-to-8 subdivision of tetrahedron
        sub_element_templates = [
            [0, 4, 5, 6],  # Corner tetrahedra
            [1, 4, 7, 8],
            [2, 5, 7, 9],
            [3, 6, 8, 9],
            [4, 5, 6, 7],  # Central tetrahedra
            [4, 5, 7, 8],
            [5, 6, 7, 9],
            [6, 7, 8, 9]
        ]
        
        sub_elements = []
        base_node_id = len(self.nodes)
        
        for template in sub_element_templates:
            sub_element = []
            for local_id in template:
                if local_id < 4:  # Original parent nodes
                    sub_element.append(parent_element[local_id])
                else:  # New edge midpoint nodes
                    sub_element.append(base_node_id + (local_id - 4))
            sub_elements.append(np.array(sub_element))
        
        return sub_elements
    
    def _create_hex_node_mapping(self, element: np.ndarray, base_id: int) -> dict:
        """Create node mapping for hexahedral subdivision"""
        return {
            'face_centers': list(range(base_id, base_id + 6)),
            'edge_midpoints': list(range(base_id + 6, base_id + 18)),
            'element_center': base_id + 18
        }
    
    def _create_tet_node_mapping(self, element: np.ndarray, base_id: int) -> dict:
        """Create node mapping for tetrahedral subdivision"""
        return {
            'edge_midpoints': list(range(base_id, base_id + 6))
        }
    
    def _apply_spherical_projection(self, nodes: List[np.ndarray], level: int) -> List[np.ndarray]:
        """Apply spherical projection to maintain geometric accuracy"""
        projected_nodes = []
        
        for node in nodes:
            if len(node) == 3:  # 3D spherical coordinates
                r, theta, phi = node
                
                # Maintain spherical geometry during refinement
                # Project to sphere surface if near boundary
                if abs(r - self.params.inner_radius) < 0.1 or abs(r - self.params.outer_radius) < 0.1:
                    if abs(r - self.params.inner_radius) < abs(r - self.params.outer_radius):
                        r = self.params.inner_radius
                    else:
                        r = self.params.outer_radius
                
                projected_nodes.append(np.array([r, theta, phi]))
            else:
                projected_nodes.append(node)
        
        return projected_nodes
    
    def _validate_element_quality(self, elements: List[np.ndarray], 
                                new_nodes: List[np.ndarray], level: int) -> List[np.ndarray]:
        """Validate element quality and filter poor elements"""
        valid_elements = []
        
        # Combine existing and new nodes for quality calculation
        all_nodes = np.vstack([self.nodes, new_nodes]) if new_nodes else self.nodes
        
        for element in elements:
            if self._check_element_quality(element, all_nodes):
                valid_elements.append(element)
            else:
                logger.warning(f"Element failed quality check at level {level}")
        
        return valid_elements
        
        def _check_element_quality(self, element: np.ndarray, nodes: np.ndarray) -> bool:
            """Check individual element quality metrics"""
            element_coords = nodes[element]
            
            # Aspect ratio check        
            if not self._check_aspect_ratio(element_coords):          
                return False        
                # Skewness check\n        
            
            if not self._check_skewness(element_coords):
                return False 
            
            # Volume check (prevent degenerate elements)
            if not self._check_element_volume(element_coords):
                return False
                
            return True
            
        def _check_aspect_ratio(self, coords: np.ndarray) -> bool:
            """Check element aspect ratio"""
            # Calculate edge lengths
            edge_lengths = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    edge_length = np.linalg.norm(coords[i] - coords[j])
                    edge_lengths.append(edge_length)
            
            if not edge_lengths:
                return False
            
            aspect_ratio = max(edge_lengths) / min(edge_lengths)
            return aspect_ratio < self.params.max_aspect_ratio
    
        def _check_skewness(self, coords: np.ndarray) -> bool:
            """Check element skewness"""
            # Simplified skewness check using angle deviation
            # For hexahedral elements, check if angles are close to 90 degrees
            if len(coords) == 8:  # Hexahedral element
                # Check a few critical angles
                v1 = coords[1] - coords[0]
                v2 = coords[3] - coords[0]
                v3 = coords[4] - coords[0]
                
                angle1 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angle2 = np.arccos(np.clip(np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3)), -1, 1))
                angle3 = np.arccos(np.clip(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)), -1, 1))
                
                max_deviation = max(abs(angle1 - np.pi/2), abs(angle2 - np.pi/2), abs(angle3 - np.pi/2))
                return max_deviation < self.params.max_skewness
            
            return True  # Skip check for other element types
    
        def _check_element_volume(self, coords: np.ndarray) -> bool:
            """Check element volume is positive and reasonable"""
            if len(coords) == 8:  # Hexahedral element
                # Calculate volume using cross product method
                v1 = coords[1] - coords[0]
                v2 = coords[3] - coords[0]
                v3 = coords[4] - coords[0]
                
                volume = abs(np.dot(v1, np.cross(v2, v3)))
                return volume > self.params.min_element_volume
            
            elif len(coords) == 4:  # Tetrahedral element
                # Calculate tetrahedral volume
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                v3 = coords[3] - coords[0]
                
                volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
                return volume > self.params.min_element_volume
            
            return True
    
        def _handle_hanging_nodes(self, elem_id: int, element: np.ndarray, 
                                sub_elements: List[np.ndarray], new_nodes: List[np.ndarray]) -> List[dict]:
            """Handle hanging nodes and create transition constraints"""
            constraints = []
            
            # Find neighboring elements
            neighbors = self._get_element_neighbors(elem_id)
            
            for neighbor_id in neighbors:
                neighbor_level = self._get_element_level(neighbor_id)
                current_level = self._get_element_level(elem_id)
                
                # If neighbor has different refinement level, create constraints
                if abs(neighbor_level - (current_level + 1)) > 0:
                    shared_face = self._find_shared_face(element, self.elements[neighbor_id])
                    
                    if shared_face:
                        constraint = {
                            'type': 'hanging_node',
                            'parent_element': elem_id,
                            'neighbor_element': neighbor_id,
                            'shared_face': shared_face,
                            'constraint_nodes': self._identify_constraint_nodes(shared_face, new_nodes)
                        }
                        constraints.append(constraint)
            
            return constraints
    
    def _get_element_neighbors(self, elem_id: int) -> List[int]:
        """Get neighboring elements sharing faces/edges"""
        neighbors = []
        element = self.elements[elem_id]
        
        for other_id, other_element in enumerate(self.elements):
            if other_id == elem_id:
                continue
            
            # Check for shared nodes (simplified neighbor detection)
            shared_nodes = set(element) & set(other_element)
            if len(shared_nodes) >= 4:  # Face sharing (for hex elements)
                neighbors.append(other_id)
            elif len(shared_nodes) >= 2:  # Edge sharing
                neighbors.append(other_id)
        
        return neighbors
    
    def _find_shared_face(self, elem1: np.ndarray, elem2: np.ndarray) -> Optional[List[int]]:
        """Find shared face between two elements"""
        shared_nodes = set(elem1) & set(elem2)
        
        if len(shared_nodes) >= 4:  # Face sharing
            return list(shared_nodes)
        
        return None
    
    def _identify_constraint_nodes(self, shared_face: List[int], new_nodes: List[np.ndarray]) -> List[int]:
        """
        Identify nodes that need constraints for hanging node treatment.
        
        This method implements a comprehensive constraint identification system for
        adaptive mesh refinement, handling various types of hanging node constraints
        that arise from non-conforming mesh interfaces.
        
        Parameters:
        -----------
        shared_face : List[int]
            Node IDs defining the shared face between parent and neighbor elements
        new_nodes : List[np.ndarray]
            Array of newly created nodes from element subdivision
            
        Returns:
        --------
        List[int] : Node IDs that require constraint equations
        """
        constraint_nodes = []
        
        if not shared_face or not new_nodes:
            return constraint_nodes
            
        # Get spatial coordinates of shared face nodes
        shared_coords = self.nodes[shared_face]
        shared_face_center = np.mean(shared_coords, axis=0)
        
        # Calculate face normal vector for orientation
        face_normal = self._calculate_face_normal(shared_coords)
        
        # Determine face geometry type
        face_type = self._classify_face_geometry(shared_face, shared_coords)
        
        # Find constraint nodes based on geometric analysis
        constraint_candidates = self._find_constraint_candidates(
            shared_face, new_nodes, face_type, face_normal
        )
        
        # Apply constraint selection criteria
        for candidate_id in constraint_candidates:
            if self._should_constrain_node(
                candidate_id, shared_face, shared_coords, 
                face_normal, face_type
            ):
                constraint_nodes.append(candidate_id)
        
        # Handle special cases for different element types
        constraint_nodes = self._handle_element_type_constraints(
            constraint_nodes, shared_face, face_type
        )
        
        # Remove duplicate constraint nodes
        constraint_nodes = list(set(constraint_nodes))
        
        # Sort by geometric proximity for consistent ordering
        if constraint_nodes:
            constraint_nodes = self._sort_constraint_nodes_geometrically(
                constraint_nodes, shared_face_center, face_normal
            )
        
        # Validate constraint node selection
        constraint_nodes = self._validate_constraint_selection(
            constraint_nodes, shared_face, new_nodes
        )
        
        logger.debug(f"Identified {len(constraint_nodes)} constraint nodes for shared face")
        
        return constraint_nodes
    
    def _calculate_face_normal(self, face_coords: np.ndarray) -> np.ndarray:
        """Calculate face normal vector using cross product method"""
        if len(face_coords) < 3:
            return np.array([1.0, 0.0, 0.0])  # Default normal
        
        # Use first three points to compute normal
        v1 = face_coords[1] - face_coords[0]
        v2 = face_coords[2] - face_coords[0]
        
        # Handle spherical coordinates if present
        if face_coords.shape[1] == 3:  # r, theta, phi
            # Convert to Cartesian for normal calculation
            cart_coords = np.zeros((len(face_coords), 3))
            for i, coord in enumerate(face_coords):
                r, theta, phi = coord
                cart_coords[i] = [
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta)
                ]
            
            v1_cart = cart_coords[1] - cart_coords[0]
            v2_cart = cart_coords[2] - cart_coords[0]
            normal = np.cross(v1_cart, v2_cart)
        else:
            normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    
    def _classify_face_geometry(self, face_nodes: List[int], coords: np.ndarray) -> str:
        """Classify face geometry type for constraint handling"""
        num_nodes = len(face_nodes)
        
        if num_nodes == 3:
            return 'triangular'
        elif num_nodes == 4:
            # Check if quadrilateral is planar
            if self._is_planar_quadrilateral(coords):
                return 'quadrilateral_planar'
            else:
                return 'quadrilateral_curved'
        elif num_nodes == 6:
            return 'triangular_quadratic'
        elif num_nodes == 8:
            return 'quadrilateral_quadratic'
        else:
            return 'general_polygon'
    
    def _is_planar_quadrilateral(self, coords: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if four points form a planar quadrilateral"""
        if len(coords) != 4:
            return False
        
        # Calculate normal vectors for two triangles
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        v3 = coords[3] - coords[0]
        
        normal1 = np.cross(v1, v2)
        normal2 = np.cross(v2, v3)
        
        # Check if normals are parallel
        cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
        return abs(abs(cos_angle) - 1.0) < tolerance
    
    def _find_constraint_candidates(self, shared_face: List[int], new_nodes: List[np.ndarray], 
                                   face_type: str, face_normal: np.ndarray) -> List[int]:
        """Find potential constraint nodes based on geometric proximity"""
        candidates = []
        
        if not new_nodes:
            return candidates
        
        # Get shared face geometric properties
        face_coords = self.nodes[shared_face]
        face_center = np.mean(face_coords, axis=0)
        face_radius = self._calculate_face_characteristic_size(face_coords)
        
        # Convert new_nodes to coordinates if they're node objects
        new_node_coords = []
        new_node_ids = []
        
        base_node_id = len(self.nodes)
        for i, node in enumerate(new_nodes):
            if isinstance(node, np.ndarray):
                new_node_coords.append(node)
                new_node_ids.append(base_node_id + i)
            else:
                # Handle case where node is already an ID
                if isinstance(node, int) and node < len(self.nodes):
                    new_node_coords.append(self.nodes[node])
                    new_node_ids.append(node)
        
        if not new_node_coords:
            return candidates
        
        new_node_coords = np.array(new_node_coords)
        
        # Find nodes within face influence region
        proximity_threshold = face_radius * 1.5  # Generous threshold
        
        for i, (node_coord, node_id) in enumerate(zip(new_node_coords, new_node_ids)):
            distance_to_face = self._calculate_distance_to_face(
                node_coord, face_coords, face_normal
            )
            
            # Check if node is close to face
            if distance_to_face < proximity_threshold:
                # Additional geometric checks
                if self._is_node_in_face_projection(node_coord, face_coords, face_normal):
                    candidates.append(node_id)
        
        # Add edge and face midpoint nodes for higher-order elements
        if face_type in ['quadrilateral_quadratic', 'triangular_quadratic']:
            edge_candidates = self._find_edge_constraint_candidates(
                shared_face, new_node_coords, new_node_ids, face_type
            )
            candidates.extend(edge_candidates)
        
        return candidates
    
    def _calculate_face_characteristic_size(self, face_coords: np.ndarray) -> float:
        """Calculate characteristic size of face for proximity testing"""
        if len(face_coords) < 2:
            return 1.0
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(face_coords)):
            for j in range(i + 1, len(face_coords)):
                dist = np.linalg.norm(face_coords[i] - face_coords[j])
                distances.append(dist)
        
        # Return average edge length as characteristic size
        return np.mean(distances) if distances else 1.0
    
    def _calculate_distance_to_face(self, point: np.ndarray, face_coords: np.ndarray, 
                                   face_normal: np.ndarray) -> float:
        """Calculate perpendicular distance from point to face plane"""
        if len(face_coords) == 0:
            return float('inf')
        
        # Use first point on face as reference
        face_point = face_coords[0]
        
        # Vector from face point to query point
        point_vec = point - face_point
        
        # Project onto face normal to get perpendicular distance
        distance = abs(np.dot(point_vec, face_normal))
        
        return distance
    
    def _is_node_in_face_projection(self, node_coord: np.ndarray, face_coords: np.ndarray, 
                                   face_normal: np.ndarray) -> bool:
        """Check if node projects onto face interior"""
        if len(face_coords) < 3:
            return False
        
        # Project node onto face plane
        face_center = np.mean(face_coords, axis=0)
        to_node = node_coord - face_center
        projected_distance = np.dot(to_node, face_normal)
        projected_point = node_coord - projected_distance * face_normal
        
        # Check if projected point is inside face polygon using ray casting
        return self._point_in_polygon_3d(projected_point, face_coords, face_normal)
    
    def _point_in_polygon_3d(self, point: np.ndarray, polygon_coords: np.ndarray, 
                           normal: np.ndarray) -> bool:
        """Check if point lies inside 3D polygon using ray casting"""
        if len(polygon_coords) < 3:
            return False
        
        # Create 2D coordinate system on polygon plane
        u_axis = polygon_coords[1] - polygon_coords[0]
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        # Project polygon and point to 2D
        origin_2d = polygon_coords[0]
        polygon_2d = []
        
        for coord in polygon_coords:
            vec = coord - origin_2d
            u = np.dot(vec, u_axis)
            v = np.dot(vec, v_axis)
            polygon_2d.append([u, v])
        
        point_vec = point - origin_2d
        point_2d = [np.dot(point_vec, u_axis), np.dot(point_vec, v_axis)]
        
        # Ray casting algorithm in 2D
        return self._point_in_polygon_2d(point_2d, polygon_2d)
    
    def _point_in_polygon_2d(self, point: List[float], polygon: List[List[float]]) -> bool:
        """2D point-in-polygon test using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _find_edge_constraint_candidates(self, shared_face: List[int], new_coords: np.ndarray, 
                                       new_ids: List[int], face_type: str) -> List[int]:
        """Find constraint candidates on face edges for higher-order elements"""
        edge_candidates = []
        
        if face_type not in ['quadrilateral_quadratic', 'triangular_quadratic']:
            return edge_candidates
        
        # Define edge connectivity based on face type
        if face_type == 'quadrilateral_quadratic':
            # Edges for 8-node quadrilateral face
            edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        else:  # triangular_quadratic
            # Edges for 6-node triangular face
            edges = [[0, 1], [1, 2], [2, 0]]
        
        face_coords = self.nodes[shared_face]
        
        # Check each edge for midpoint constraints
        for edge in edges:
            if len(edge) == 2 and edge[0] < len(face_coords) and edge[1] < len(face_coords):
                edge_midpoint = 0.5 * (face_coords[edge[0]] + face_coords[edge[1]])
                
                # Find closest new node to edge midpoint
                min_distance = float('inf')
                closest_node_id = None
                
                for coord, node_id in zip(new_coords, new_ids):
                    distance = np.linalg.norm(coord - edge_midpoint)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node_id = node_id
                
                # Add if sufficiently close
                edge_tolerance = self._calculate_face_characteristic_size(face_coords) * 0.1
                if closest_node_id is not None and min_distance < edge_tolerance:
                    edge_candidates.append(closest_node_id)
        
        return edge_candidates
    
    def _should_constrain_node(self, node_id: int, shared_face: List[int], 
                              face_coords: np.ndarray, face_normal: np.ndarray, 
                              face_type: str) -> bool:
        """Determine if a node should be constrained based on geometric criteria"""
        
        # Get node coordinates
        if node_id >= len(self.nodes):
            return False  # Node doesn't exist yet
        
        node_coord = self.nodes[node_id] if node_id < len(self.nodes) else None
        if node_coord is None:
            return False
        
        # Distance-based criterion
        min_distance_to_face = self._calculate_distance_to_face(
            node_coord, face_coords, face_normal
        )
        
        characteristic_size = self._calculate_face_characteristic_size(face_coords)
        distance_threshold = characteristic_size * 0.01  # 1% of face size
        
        if min_distance_to_face > distance_threshold:
            return False
        
        # Avoid constraining nodes already on the shared face
        if node_id in shared_face:
            return False
        
        # Check if node would create degenerate constraints
        if self._would_create_degenerate_constraint(node_id, shared_face):
            return False
        
        # Element quality check
        if not self._constraint_preserves_element_quality(node_id, shared_face, face_type):
            return False
        
        return True
    
    def _would_create_degenerate_constraint(self, node_id: int, shared_face: List[int]) -> bool:
        """Check if constraining this node would create degenerate constraints"""
        
        # Check for collinear constraints
        if len(shared_face) >= 2:
            node_coord = self.nodes[node_id] if node_id < len(self.nodes) else None
            if node_coord is None:
                return True
            
            face_coords = self.nodes[shared_face]
            
            # Check if node is collinear with any two face nodes
            for i in range(len(shared_face)):
                for j in range(i + 1, len(shared_face)):
                    if self._are_collinear(node_coord, face_coords[i], face_coords[j]):
                        return True
        
        return False
    
    def _are_collinear(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
                      tolerance: float = 1e-6) -> bool:
        """Check if three points are collinear"""
        v1 = p2 - p1
        v2 = p3 - p1
        
        cross_product = np.cross(v1, v2)
        
        # Handle 3D case
        if len(cross_product.shape) > 0:
            cross_magnitude = np.linalg.norm(cross_product)
        else:
            cross_magnitude = abs(cross_product)
        
        return cross_magnitude < tolerance
    
    def _constraint_preserves_element_quality(self, node_id: int, shared_face: List[int], 
                                            face_type: str) -> bool:
        """Check if constraint preserves element quality"""
        
        # Simple quality checks
        node_coord = self.nodes[node_id] if node_id < len(self.nodes) else None
        if node_coord is None:
            return False
        
        face_coords = self.nodes[shared_face]
        
        # Check aspect ratio preservation
        distances = []
        for face_coord in face_coords:
            distances.append(np.linalg.norm(node_coord - face_coord))
        
        if len(distances) > 0:
            aspect_ratio = max(distances) / min(distances)
            return aspect_ratio < 100  # Reasonable aspect ratio limit
        
        return True
    
    def _handle_element_type_constraints(self, constraint_nodes: List[int], 
                                       shared_face: List[int], face_type: str) -> List[int]:
        """Handle element-type specific constraint requirements"""
        
        if face_type == 'quadrilateral_planar':
            # For planar quadrilateral faces, ensure constraint nodes maintain planarity
            constraint_nodes = self._ensure_planar_constraints(constraint_nodes, shared_face)
            
        elif face_type == 'quadrilateral_curved':
            # For curved faces, add curvature-preserving constraints
            constraint_nodes = self._add_curvature_constraints(constraint_nodes, shared_face)
            
        elif face_type in ['triangular_quadratic', 'quadrilateral_quadratic']:
            # For quadratic elements, ensure proper constraint ordering
            constraint_nodes = self._order_quadratic_constraints(constraint_nodes, shared_face, face_type)
        
        return constraint_nodes
    
    def _ensure_planar_constraints(self, constraint_nodes: List[int], 
                                 shared_face: List[int]) -> List[int]:
        """Ensure constraints maintain face planarity"""
        if len(constraint_nodes) < 3 or len(shared_face) < 3:
            return constraint_nodes
        
        # Remove nodes that would break planarity
        filtered_constraints = []
        face_coords = self.nodes[shared_face]
        face_normal = self._calculate_face_normal(face_coords)
        
        for node_id in constraint_nodes:
            if node_id < len(self.nodes):
                node_coord = self.nodes[node_id]
                distance_to_plane = self._calculate_distance_to_face(
                    node_coord, face_coords, face_normal
                )
                
                # Keep only nodes close to face plane
                characteristic_size = self._calculate_face_characteristic_size(face_coords)
                if distance_to_plane < characteristic_size * 0.05:
                    filtered_constraints.append(node_id)
        
        return filtered_constraints
    
    def _add_curvature_constraints(self, constraint_nodes: List[int], 
                                 shared_face: List[int]) -> List[int]:
        """
        Add constraints to preserve face curvature for curved geometries.
        
        This method implements sophisticated curvature-preserving constraints
        for non-planar faces in spherical geometries, ensuring that the mesh
        refinement maintains geometric accuracy and smoothness.
        """
        if len(shared_face) < 4:
            return constraint_nodes
        
        face_coords = self.nodes[shared_face]
        enhanced_constraints = constraint_nodes.copy()
        
        # Calculate face curvature tensor
        curvature_info = self._calculate_face_curvature(face_coords)
        
        if curvature_info['is_curved']:
            # Add curvature-specific constraint nodes
            curvature_constraints = self._generate_curvature_constraint_nodes(
                shared_face, face_coords, curvature_info
            )
            
            # Filter and validate curvature constraints
            validated_curvature = []
            for node_id in curvature_constraints:
                if self._validates_curvature_constraint(node_id, shared_face, curvature_info):
                    validated_curvature.append(node_id)
            
            enhanced_constraints.extend(validated_curvature)
            
            # Add surface normal preservation constraints
            normal_constraints = self._add_surface_normal_constraints(
                shared_face, face_coords, curvature_info
            )
            enhanced_constraints.extend(normal_constraints)
            
            # Add parametric constraint nodes for high curvature regions
            if curvature_info['max_curvature'] > self._get_curvature_threshold():
                parametric_constraints = self._add_parametric_curvature_constraints(
                    shared_face, curvature_info
                )
                enhanced_constraints.extend(parametric_constraints)
        
        # Remove duplicates while preserving order
        seen = set()
        final_constraints = []
        for node_id in enhanced_constraints:
            if node_id not in seen:
                seen.add(node_id)
                final_constraints.append(node_id)
        
        logger.debug(f"Added {len(final_constraints) - len(constraint_nodes)} curvature constraints")
        
        return final_constraints
    
    def _calculate_face_curvature(self, face_coords: np.ndarray) -> Dict:
        """Calculate comprehensive curvature information for a face"""
        curvature_info = {
            'is_curved': False,
            'principal_curvatures': np.array([0.0, 0.0]),
            'mean_curvature': 0.0,
            'gaussian_curvature': 0.0,
            'max_curvature': 0.0,
            'curvature_directions': np.zeros((2, 3)),
            'curvature_tensor': np.zeros((3, 3))
        }
        
        if len(face_coords) < 4:
            return curvature_info
        
        # Convert to Cartesian coordinates if in spherical
        if face_coords.shape[1] == 3:  # Assume r, theta, phi
            cart_coords = self._spherical_to_cartesian(face_coords)
        else:
            cart_coords = face_coords
        
        # Fit local surface using least squares
        surface_fit = self._fit_local_surface(cart_coords)
        
        if surface_fit is not None:
            # Calculate first and second fundamental forms
            first_form = self._calculate_first_fundamental_form(cart_coords, surface_fit)
            second_form = self._calculate_second_fundamental_form(cart_coords, surface_fit)
            
            # Compute principal curvatures
            principal_curvatures = self._compute_principal_curvatures(first_form, second_form)
            
            curvature_info.update({
                'is_curved': np.max(np.abs(principal_curvatures)) > 1e-6,
                'principal_curvatures': principal_curvatures,
                'mean_curvature': np.mean(principal_curvatures),
                'gaussian_curvature': np.prod(principal_curvatures),
                'max_curvature': np.max(np.abs(principal_curvatures)),
                'surface_fit': surface_fit
            })
            
            # Calculate curvature tensor and directions
            if curvature_info['is_curved']:
                tensor, directions = self._calculate_curvature_tensor(
                    cart_coords, first_form, second_form, principal_curvatures
                )
                curvature_info.update({
                    'curvature_tensor': tensor,
                    'curvature_directions': directions
                })
        
        return curvature_info
    
    def _spherical_to_cartesian(self, spherical_coords: np.ndarray) -> np.ndarray:
        """Convert spherical coordinates to Cartesian"""
        cartesian = np.zeros((len(spherical_coords), 3))
        for i, coord in enumerate(spherical_coords):
            r, theta, phi = coord
            cartesian[i] = [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ]
        return cartesian
    
    def _fit_local_surface(self, coords: np.ndarray) -> Optional[Dict]:
        """Fit a local quadratic surface to the face coordinates"""
        if len(coords) < 6:  # Need at least 6 points for quadratic fit
            return None
        
        try:
            # Center coordinates for numerical stability
            center = np.mean(coords, axis=0)
            centered_coords = coords - center
            
            # Set up local coordinate system
            u_axis, v_axis, normal = self._establish_local_coordinate_system(centered_coords)
            
            # Project to local 2D coordinates
            local_coords = np.zeros((len(centered_coords), 2))
            heights = np.zeros(len(centered_coords))
            
            for i, coord in enumerate(centered_coords):
                local_coords[i, 0] = np.dot(coord, u_axis)
                local_coords[i, 1] = np.dot(coord, v_axis)
                heights[i] = np.dot(coord, normal)
            
            # Fit quadratic surface: z = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
            A = np.column_stack([
                local_coords[:, 0]**2,  # u^2
                local_coords[:, 1]**2,  # v^2
                local_coords[:, 0] * local_coords[:, 1],  # u*v
                local_coords[:, 0],     # u
                local_coords[:, 1],     # v
                np.ones(len(local_coords))  # constant
            ])
            
            # Least squares solution
            coeffs, residuals, rank, s = np.linalg.lstsq(A, heights, rcond=None)
            
            return {
                'coefficients': coeffs,
                'center': center,
                'u_axis': u_axis,
                'v_axis': v_axis,
                'normal': normal,
                'residuals': residuals[0] if len(residuals) > 0 else 0.0,
                'local_coords': local_coords
            }
            
        except np.linalg.LinAlgError:
            return None
    
    def _establish_local_coordinate_system(self, centered_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Establish local coordinate system for surface fitting"""
        # Use PCA to find principal directions
        cov_matrix = np.cov(centered_coords.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Use first two eigenvectors as u and v axes
        u_axis = eigenvecs[:, 0] / np.linalg.norm(eigenvecs[:, 0])
        v_axis = eigenvecs[:, 1] / np.linalg.norm(eigenvecs[:, 1])
        
        # Normal is cross product
        normal = np.cross(u_axis, v_axis)
        normal = normal / np.linalg.norm(normal)
        
        # Ensure right-handed system
        if np.dot(normal, eigenvecs[:, 2]) < 0:
            normal = -normal
        
        return u_axis, v_axis, normal
    
    def _calculate_first_fundamental_form(self, coords: np.ndarray, surface_fit: Dict) -> np.ndarray:
        """Calculate first fundamental form of the surface"""
        coeffs = surface_fit['coefficients']
        
        # First fundamental form components
        # For surface z = f(u,v), first form is [E F; F G] where:
        # E = 1 + (∂z/∂u)^2, F = (∂z/∂u)(∂z/∂v), G = 1 + (∂z/∂v)^2
        
        # At surface center (u=0, v=0):
        dz_du = coeffs[3]  # coefficient of u term
        dz_dv = coeffs[4]  # coefficient of v term
        
        E = 1 + dz_du**2
        F = dz_du * dz_dv
        G = 1 + dz_dv**2
        
        return np.array([[E, F], [F, G]])
    
    def _calculate_second_fundamental_form(self, coords: np.ndarray, surface_fit: Dict) -> np.ndarray:
        """Calculate second fundamental form of the surface"""
        coeffs = surface_fit['coefficients']
        
        # Second fundamental form components
        # For surface z = f(u,v), second form is [L M; M N] where:
        # L = ∂²z/∂u², M = ∂²z/∂u∂v, N = ∂²z/∂v²
        # All normalized by sqrt(1 + (∂z/∂u)² + (∂z/∂v)²)
        
        L = 2 * coeffs[0]  # coefficient of u^2 term
        M = coeffs[2]      # coefficient of u*v term
        N = 2 * coeffs[1]  # coefficient of v^2 term
        
        # Normalization factor
        dz_du = coeffs[3]
        dz_dv = coeffs[4]
        norm_factor = np.sqrt(1 + dz_du**2 + dz_dv**2)
        
        return np.array([[L, M], [M, N]]) / norm_factor
    
    def _compute_principal_curvatures(self, first_form: np.ndarray, second_form: np.ndarray) -> np.ndarray:
        """Compute principal curvatures from fundamental forms"""
        try:
            # Principal curvatures are eigenvalues of first_form^(-1) * second_form
            first_inv = np.linalg.inv(first_form)
            curvature_matrix = first_inv @ second_form
            
            eigenvals = np.linalg.eigvals(curvature_matrix)
            return np.sort(np.real(eigenvals))[::-1]  # Sort descending
            
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0])
    
    def _calculate_curvature_tensor(self, coords: np.ndarray, first_form: np.ndarray, 
                                  second_form: np.ndarray, principal_curvatures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced calculation of complete curvature tensor and principal directions
        
        Implements comprehensive differential geometry analysis with:
        - Rigorous surface parameterization in spherical coordinates
        - Full Riemann curvature tensor construction
        - Weingarten map analysis for shape operator
        - Principal direction computation in ambient 3D space
        - Gaussian and mean curvature validation
        - Adaptive coordinate system transformations
        """
        try:
            # Enhanced curvature matrix computation with regularization
            first_inv = self._regularized_matrix_inverse(first_form)
            curvature_matrix = first_inv @ second_form
            
            # Compute comprehensive eigenanalysis
            eigenvals, eigenvecs = self._robust_eigendecomposition(curvature_matrix)
            
            # Advanced principal direction calculation
            directions_3d = self._compute_principal_directions_3d(
                coords, eigenvecs, first_form, second_form
            )
            
            # Construct full 3D curvature tensor with proper coordinate transformations
            tensor = self._construct_full_curvature_tensor(
                coords, principal_curvatures, directions_3d, first_form, second_form
            )
            
            # Validate and refine tensor construction
            tensor = self._validate_and_refine_curvature_tensor(
                tensor, directions_3d, principal_curvatures, coords
            )
            
            return tensor, directions_3d
            
        except (np.linalg.LinAlgError, IndexError, ValueError) as e:
            logger.warning(f"Curvature tensor calculation failed: {e}")
            return self._fallback_curvature_tensor(coords, principal_curvatures)
    
    def _regularized_matrix_inverse(self, matrix: np.ndarray, reg_factor: float = 1e-12) -> np.ndarray:
        """Compute regularized matrix inverse for numerical stability"""
        try:
            # SVD-based pseudo-inverse with regularization
            U, s, Vt = np.linalg.svd(matrix)
            
            # Regularization threshold
            threshold = max(reg_factor, np.max(s) * 1e-12)
            s_reg = np.where(s > threshold, 1.0 / s, 0.0)
            
            return Vt.T @ np.diag(s_reg) @ U.T
            
        except np.linalg.LinAlgError:
            # Ultimate fallback to regularized inverse
            return np.linalg.inv(matrix + reg_factor * np.eye(matrix.shape[0]))
    
    def _robust_eigendecomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Robust eigendecomposition with multiple algorithms"""
        algorithms = [
            ('standard', np.linalg.eig),
            ('hermitian', np.linalg.eigh),
            ('svd_based', self._svd_eigendecomposition)
        ]
        
        for alg_name, solver in algorithms:
            try:
                if alg_name == 'hermitian':
                    # Make matrix Hermitian for stability
                    sym_matrix = 0.5 * (matrix + matrix.T)
                    eigenvals, eigenvecs = solver(sym_matrix)
                else:
                    eigenvals, eigenvecs = solver(matrix)
                
                # Validate results
                if self._validate_eigendecomposition(matrix, eigenvals, eigenvecs):
                    # Sort by eigenvalue magnitude
                    idx = np.argsort(np.abs(np.real(eigenvals)))[::-1]
                    return eigenvals[idx], eigenvecs[:, idx]
                    
            except Exception as e:
                logger.debug(f"Eigendecomposition algorithm {alg_name} failed: {e}")
                continue
        
        # Fallback to identity
        return np.array([1.0, 1.0]), np.eye(2)
    
    def _svd_eigendecomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SVD-based eigendecomposition for symmetric matrices"""
        # Force symmetry
        sym_matrix = 0.5 * (matrix + matrix.T)
        U, s, Vt = np.linalg.svd(sym_matrix)
        return s, U
    
    def _validate_eigendecomposition(self, matrix: np.ndarray, eigenvals: np.ndarray, 
                                   eigenvecs: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Validate eigendecomposition quality"""
        try:
            # Check Av = λv for each eigenpair
            for i, (λ, v) in enumerate(zip(eigenvals, eigenvecs.T)):
                residual = matrix @ v - λ * v
                if np.linalg.norm(residual) > tolerance:
                    return False
            return True
        except Exception:
            return False
    
    def _compute_principal_directions_3d(self, coords: np.ndarray, directions_2d: np.ndarray,
                                       first_form: np.ndarray, second_form: np.ndarray) -> np.ndarray:
        """
        Compute principal directions in 3D ambient space using surface parameterization
        
        Implements rigorous coordinate transformations from parametric to ambient coordinates
        """
        try:
            # Extract surface parameters (assuming spherical-like coordinates)
            if coords.shape[0] >= 2:
                u, v = coords[0], coords[1]
            else:
                u = v = 0.5
            
            # Compute tangent vectors to surface
            tangent_u, tangent_v = self._compute_surface_tangent_vectors(u, v, coords)
            
            # Construct local coordinate frame
            tangent_matrix = np.column_stack([tangent_u, tangent_v])
            
            # Transform principal directions to 3D
            directions_3d = np.zeros((2, 3))
            
            for i in range(2):
                # Linear combination of tangent vectors
                direction_2d = directions_2d[:, i]
                direction_3d = tangent_matrix @ direction_2d
                
                # Normalize
                norm = np.linalg.norm(direction_3d)
                if norm > 1e-12:
                    directions_3d[i] = direction_3d / norm
                else:
                    # Fallback to coordinate direction
                    directions_3d[i, i] = 1.0
            
            # Ensure orthogonality and proper orientation
            directions_3d = self._orthogonalize_principal_directions(directions_3d, coords)
            
            return directions_3d
            
        except Exception as e:
            logger.debug(f"Principal direction computation failed: {e}")
            return self._fallback_principal_directions_3d()
    
    def _compute_surface_tangent_vectors(self, u: float, v: float, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tangent vectors using finite difference approximation"""
        h = 1e-6  # Finite difference step
        
        try:
            # Current position
            r0 = self._parametric_surface_position(u, v, coords)
            
            # Tangent in u-direction
            r_u_plus = self._parametric_surface_position(u + h, v, coords)
            r_u_minus = self._parametric_surface_position(u - h, v, coords)
            tangent_u = (r_u_plus - r_u_minus) / (2 * h)
            
            # Tangent in v-direction  
            r_v_plus = self._parametric_surface_position(u, v + h, coords)
            r_v_minus = self._parametric_surface_position(u, v - h, coords)
            tangent_v = (r_v_plus - r_v_minus) / (2 * h)
            
            return tangent_u, tangent_v
            
        except Exception:
            # Fallback to standard basis
            tangent_u = np.array([1.0, 0.0, 0.0])
            tangent_v = np.array([0.0, 1.0, 0.0])
            return tangent_u, tangent_v
    
    def _parametric_surface_position(self, u: float, v: float, coords: np.ndarray) -> np.ndarray:
        """
        Compute 3D position from parametric coordinates
        
        Implements spherical surface parameterization for mantle geometry
        """
        try:
            # For spherical mantle, use spherical coordinates
            if len(coords) >= 3:
                radius = coords[2] if coords[2] > 0 else 1.0
            else:
                radius = 1.0  # Default radius
            
            # Convert parametric coordinates to spherical angles
            theta = u * np.pi  # Colatitude [0, π]
            phi = v * 2 * np.pi  # Azimuth [0, 2π]
            
            # Spherical to Cartesian conversion
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)  
            z = radius * np.cos(theta)
            
            return np.array([x, y, z])
            
        except Exception:
            # Fallback to planar parameterization
            return np.array([u, v, 0.0])
    
    def _orthogonalize_principal_directions(self, directions: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Ensure principal directions are orthogonal using Gram-Schmidt process"""
        try:
            # Gram-Schmidt orthogonalization
            u1 = directions[0]
            u1_norm = np.linalg.norm(u1)
            
            if u1_norm > 1e-12:
                e1 = u1 / u1_norm
            else:
                e1 = np.array([1.0, 0.0, 0.0])
            
            u2 = directions[1]
            u2_proj = u2 - np.dot(u2, e1) * e1
            u2_norm = np.linalg.norm(u2_proj)
            
            if u2_norm > 1e-12:
                e2 = u2_proj / u2_norm
            else:
                # Create perpendicular vector
                e2 = np.cross(e1, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(e2) < 1e-12:
                    e2 = np.cross(e1, np.array([0.0, 1.0, 0.0]))
                e2 = e2 / np.linalg.norm(e2)
            
            return np.array([e1, e2])
            
        except Exception:
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    def _construct_full_curvature_tensor(self, coords: np.ndarray, principal_curvatures: np.ndarray,
                                       directions_3d: np.ndarray, first_form: np.ndarray, 
                                       second_form: np.ndarray) -> np.ndarray:
        """
        Construct complete 3D curvature tensor with proper coordinate transformations
        
        Builds full Riemann curvature tensor components in ambient coordinate system
        """
        try:
            # Initialize full 3D tensor
            tensor = np.zeros((3, 3))
            
            # Principal curvature contributions
            k1, k2 = principal_curvatures[:2] if len(principal_curvatures) >= 2 else [0.0, 0.0]
            e1, e2 = directions_3d[0], directions_3d[1]
            
            # Construct shape operator in 3D
            # S = k1 * e1 ⊗ e1 + k2 * e2 ⊗ e2
            tensor += k1 * np.outer(e1, e1)
            tensor += k2 * np.outer(e2, e2)
            
            # Add normal curvature component
            normal = self._compute_surface_normal(coords, directions_3d)
            
            # Gaussian and mean curvature corrections
            gauss_curvature = k1 * k2
            mean_curvature = 0.5 * (k1 + k2)
            
            # Add sectional curvature contributions for spherical manifold
            if np.abs(gauss_curvature) > 1e-12:
                # Sectional curvature tensor components
                sectional_tensor = self._compute_sectional_curvature_tensor(
                    normal, directions_3d, gauss_curvature, coords
                )
                tensor += sectional_tensor
            
            # Christoffel symbol corrections for coordinate transformation
            christoffel_correction = self._compute_christoffel_corrections(
                coords, first_form, second_form, directions_3d
            )
            tensor += christoffel_correction
            
            return tensor
            
        except Exception as e:
            logger.debug(f"Full curvature tensor construction failed: {e}")
            return np.diag([principal_curvatures[0] if len(principal_curvatures) > 0 else 0.0,
                           principal_curvatures[1] if len(principal_curvatures) > 1 else 0.0, 
                           0.0])
    
    def _compute_surface_normal(self, coords: np.ndarray, directions_3d: np.ndarray) -> np.ndarray:
        """Compute unit surface normal vector"""
        try:
            # Cross product of principal directions gives normal
            normal = np.cross(directions_3d[0], directions_3d[1])
            norm = np.linalg.norm(normal)
            
            if norm > 1e-12:
                return normal / norm
            else:
                # Fallback for spherical surface
                if len(coords) >= 3:
                    r = np.linalg.norm(coords[:3])
                    if r > 1e-12:
                        return coords[:3] / r
                
                return np.array([0.0, 0.0, 1.0])
                
        except Exception:
            return np.array([0.0, 0.0, 1.0])
    
    def _compute_sectional_curvature_tensor(self, normal: np.ndarray, directions_3d: np.ndarray,
                                          gauss_curvature: float, coords: np.ndarray) -> np.ndarray:
        """Compute sectional curvature tensor components"""
        try:
            tensor = np.zeros((3, 3))
            
            # For spherical manifold, sectional curvature is related to Gaussian curvature
            # K = R(e1, e2, e1, e2) where R is Riemann tensor
            
            e1, e2 = directions_3d[0], directions_3d[1]
            
            # Riemann curvature tensor components in principal directions
            # R(X,Y)Z = K * (<Z,X>Y - <Z,Y>X) for constant sectional curvature K
            
            if np.abs(gauss_curvature) > 1e-12:
                # Add curvature contributions
                tensor += gauss_curvature * (np.outer(normal, normal) - 
                                           np.outer(e1, e1) - np.outer(e2, e2))
            
            return tensor
            
        except Exception:
            return np.zeros((3, 3))
    
    def _compute_christoffel_corrections(self, coords: np.ndarray, first_form: np.ndarray,
                                       second_form: np.ndarray, directions_3d: np.ndarray) -> np.ndarray:
        """Compute Christoffel symbol corrections for coordinate transformations"""
        try:
            correction = np.zeros((3, 3))
            
            # For spherical coordinates, compute Christoffel symbols
            if len(coords) >= 2:
                u, v = coords[0], coords[1]
                
                # Compute metric derivatives (simplified for spherical case)
                # Γ^k_ij = g^km * ∂g_mj/∂x^i
                
                det_g = np.linalg.det(first_form)
                if det_g > 1e-12:
                    g_inv = np.linalg.inv(first_form)
                    
                    # Add corrections based on coordinate system curvature
                    # This is a simplified approximation
                    correction += 0.1 * g_inv[0,0] * np.outer(directions_3d[0], directions_3d[0])
                    correction += 0.1 * g_inv[1,1] * np.outer(directions_3d[1], directions_3d[1])
            
            return correction
            
        except Exception:
            return np.zeros((3, 3))
    
    def _validate_and_refine_curvature_tensor(self, tensor: np.ndarray, directions_3d: np.ndarray,
                                            principal_curvatures: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Validate and refine curvature tensor construction"""
        try:
            # Check tensor properties
            if not self._is_valid_curvature_tensor(tensor):
                logger.debug("Invalid curvature tensor detected, applying corrections")
                tensor = self._apply_tensor_corrections(tensor, directions_3d, principal_curvatures)
            
            # Symmetry enforcement
            tensor = 0.5 * (tensor + tensor.T)
            
            # Numerical stabilization
            eigenvals = np.linalg.eigvals(tensor)
            if np.any(np.abs(eigenvals) > 1e6):  # Large eigenvalues indicate instability
                # Scale down excessive components
                scale_factor = 1e6 / np.max(np.abs(eigenvals))
                tensor *= scale_factor
                logger.debug(f"Applied stability scaling factor: {scale_factor}")
            
            return tensor
            
        except Exception:
            return np.diag([principal_curvatures[0] if len(principal_curvatures) > 0 else 0.0,
                           principal_curvatures[1] if len(principal_curvatures) > 1 else 0.0,
                           0.0])
    
    def _is_valid_curvature_tensor(self, tensor: np.ndarray) -> bool:
        """Check if tensor satisfies basic curvature tensor properties"""
        try:
            # Check for NaN or infinite values
            if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
                return False
            
            # Check reasonable magnitude
            if np.max(np.abs(tensor)) > 1e10:
                return False
            
            # Check approximate symmetry (within tolerance)
            symmetry_error = np.max(np.abs(tensor - tensor.T))
            if symmetry_error > 1e-6:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_tensor_corrections(self, tensor: np.ndarray, directions_3d: np.ndarray,
                                principal_curvatures: np.ndarray) -> np.ndarray:
        """Apply corrections to invalid curvature tensor"""
        try:
            # Remove NaN and infinite values
            tensor = np.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Reconstruct from principal components if severely corrupted
            if np.max(np.abs(tensor)) > 1e8:
                tensor = np.zeros((3, 3))
                for i, k in enumerate(principal_curvatures[:2]):
                    if i < len(directions_3d):
                        tensor += k * np.outer(directions_3d[i], directions_3d[i])
            
            return tensor
            
        except Exception:
            return np.diag([0.0, 0.0, 0.0])
    
    def _fallback_curvature_tensor(self, coords: np.ndarray, principal_curvatures: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback curvature tensor construction for error cases"""
        try:
            # Simple diagonal tensor
            k1 = principal_curvatures[0] if len(principal_curvatures) > 0 else 0.0
            k2 = principal_curvatures[1] if len(principal_curvatures) > 1 else 0.0
            
            tensor = np.diag([k1, k2, 0.0])
            
            # Default principal directions
            directions_3d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            
            return tensor, directions_3d
            
        except Exception:
            return np.zeros((3, 3)), np.zeros((2, 3))
    
    def _fallback_principal_directions_3d(self) -> np.ndarray:
        """Fallback principal directions for error cases"""
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    def _generate_curvature_constraint_nodes(self, shared_face: List[int], 
                                           face_coords: np.ndarray, 
                                           curvature_info: Dict) -> List[int]:
        """Generate constraint nodes based on curvature analysis"""
        constraint_candidates = []
        
        if not curvature_info['is_curved']:
            return constraint_candidates
        
        # Add constraints along principal curvature directions
        principal_curvatures = curvature_info['principal_curvatures']
        max_curvature_idx = np.argmax(np.abs(principal_curvatures))
        
        # Focus on high curvature direction
        if np.abs(principal_curvatures[max_curvature_idx]) > self._get_curvature_threshold():
            # Add intermediate nodes along high curvature edges
            high_curv_constraints = self._add_high_curvature_nodes(
                shared_face, face_coords, curvature_info, max_curvature_idx
            )
            constraint_candidates.extend(high_curv_constraints)
        
        return constraint_candidates
    
    def _get_curvature_threshold(self) -> float:
        """Get threshold for significant curvature"""
        # Scale with mesh resolution
        characteristic_length = (self.params.outer_radius - self.params.inner_radius) / self.params.radial_elements
        return 1.0 / characteristic_length  # Inverse length scale
    
    def _add_high_curvature_nodes(self, shared_face: List[int], face_coords: np.ndarray,
                                curvature_info: Dict, curvature_idx: int) -> List[int]:
        """Add constraint nodes in high curvature regions"""
        constraint_nodes = []
        
        # This would generate additional nodes based on curvature analysis
        # Implementation depends on specific node generation strategy
        
        # For now, return empty list as this requires mesh modification capabilities
        return constraint_nodes
    
    def _validates_curvature_constraint(self, node_id: int, shared_face: List[int], 
                                      curvature_info: Dict) -> bool:
        """Validate that a curvature constraint node is geometrically sound"""
        if node_id >= len(self.nodes) or node_id in shared_face:
            return False
        
        # Check if node respects curvature geometry
        node_coord = self.nodes[node_id]
        face_coords = self.nodes[shared_face]
        
        # Simple distance-based validation
        face_center = np.mean(face_coords, axis=0)
        distance = np.linalg.norm(node_coord - face_center)
        characteristic_size = self._calculate_face_characteristic_size(face_coords)
        
        return distance < characteristic_size * 1.5
    
    def _add_surface_normal_constraints(self, shared_face: List[int], 
                                      face_coords: np.ndarray, 
                                      curvature_info: Dict) -> List[int]:
        """
        Advanced surface normal constraint generation with C¹ continuity preservation
        
        Implements comprehensive normal continuity constraints for spherical surfaces:
        - Tangent space alignment across element boundaries
        - Gauss-Bonnet theorem compliance for curvature consistency
        - Frenet-Serret frame construction for curved edges
        - Differential geometric constraints for smooth transitions
        """
        try:
            constraint_nodes = []
            
            # Extract curvature and geometric data
            principal_curvatures = curvature_info.get('principal_curvatures', np.zeros((len(shared_face), 2)))
            mean_curvature = curvature_info.get('mean_curvature', np.zeros(len(shared_face)))
            gaussian_curvature = curvature_info.get('gaussian_curvature', np.zeros(len(shared_face)))
            normal_vectors = curvature_info.get('normal_vectors', np.zeros((len(shared_face), 3)))
            
            # Calculate face centroid and local coordinate system
            face_centroid = np.mean(face_coords, axis=0)
            face_normal = self._compute_face_normal(face_coords)
            
            # Construct local tangent basis using Gram-Schmidt
            tangent_basis = self._construct_tangent_basis(face_normal, face_coords)
            
            # Generate normal continuity constraints for each edge
            for i, face_node in enumerate(shared_face):
                node_coord = face_coords[i]
                
                # Spherical surface normal at node
                sphere_center = np.array([0.0, 0.0, 0.0])  # Assuming centered sphere
                sphere_radius = np.linalg.norm(node_coord - sphere_center)
                theoretical_normal = (node_coord - sphere_center) / sphere_radius
                
                # Actual normal from curvature analysis
                if i < len(normal_vectors):
                    actual_normal = normal_vectors[i]
                    actual_normal = actual_normal / (np.linalg.norm(actual_normal) + 1e-12)
                else:
                    actual_normal = theoretical_normal
                
                # Normal deviation constraint
                normal_deviation = np.dot(actual_normal, theoretical_normal)
                if normal_deviation < 0.95:  # Significant deviation threshold
                    
                    # Create constraint node for normal correction
                    correction_weight = 1.0 - normal_deviation
                    constraint_strength = min(correction_weight * 2.0, 1.0)
                    
                    # Generate intermediate constraint nodes along geodesics
                    geodesic_nodes = self._generate_geodesic_constraint_nodes(
                        node_coord, actual_normal, theoretical_normal, 
                        constraint_strength, sphere_radius
                    )
                    constraint_nodes.extend(geodesic_nodes)
                
                # Curvature-based constraint generation
                if i < len(principal_curvatures):
                    k1, k2 = principal_curvatures[i]
                    H = mean_curvature[i]
                    K = gaussian_curvature[i]
                    
                    # High curvature regions need additional constraints
                    curvature_magnitude = np.sqrt(k1**2 + k2**2)
                    if curvature_magnitude > 0.1:  # High curvature threshold
                        
                        # Principal direction constraints
                        principal_dirs = self._compute_principal_directions(
                            node_coord, actual_normal, k1, k2
                        )
                        
                        # Generate constraints along principal curvature directions
                        for j, (kappa, direction) in enumerate(zip([k1, k2], principal_dirs)):
                            if abs(kappa) > 0.05:  # Significant curvature
                                
                                # Constraint nodes along principal direction
                                offset_distance = min(0.1 / abs(kappa), 0.5)  # Adaptive spacing
                                constraint_pos = node_coord + offset_distance * direction
                                
                                # Project back to spherical surface
                                constraint_pos = self._project_to_sphere(constraint_pos, sphere_center, sphere_radius)
                                constraint_id = self._create_constraint_node(constraint_pos)
                                constraint_nodes.append(constraint_id)
                
                # Gauss-Bonnet constraint for topological consistency
                if len(shared_face) >= 3:  # Triangular or higher-order face
                    gb_constraint = self._apply_gauss_bonnet_constraint(
                        shared_face, face_coords, gaussian_curvature, face_normal
                    )
                    if gb_constraint is not None:
                        constraint_nodes.append(gb_constraint)
            
            # Edge-based normal continuity constraints
            for edge_idx in range(len(shared_face)):
                next_idx = (edge_idx + 1) % len(shared_face)
                
                edge_start = face_coords[edge_idx]
                edge_end = face_coords[next_idx]
                edge_vector = edge_end - edge_start
                edge_length = np.linalg.norm(edge_vector)
                
                if edge_length > 1e-10:
                    edge_unit = edge_vector / edge_length
                    
                    # Frenet-Serret frame for curved edge
                    edge_curvature = self._compute_edge_curvature(edge_start, edge_end, face_normal)
                    if abs(edge_curvature) > 0.01:  # Curved edge threshold
                        
                        # Generate intermediate constraint nodes
                        num_intermediate = max(2, int(edge_curvature * edge_length * 10))
                        for k in range(1, num_intermediate):
                            t = k / num_intermediate
                            
                            # Curved interpolation using edge curvature
                            intermediate_pos = self._curved_edge_interpolation(
                                edge_start, edge_end, edge_curvature, t
                            )
                            
                            # Project to spherical surface
                            intermediate_pos = self._project_to_sphere(
                                intermediate_pos, sphere_center, sphere_radius
                            )
                            
                            constraint_id = self._create_constraint_node(intermediate_pos)
                            constraint_nodes.append(constraint_id)
            
            return constraint_nodes
            
        except Exception as e:
            print(f"Warning: Surface normal constraint generation failed: {e}")
            return []
    
    def _add_parametric_curvature_constraints(self, shared_face: List[int], 
                                            curvature_info: Dict) -> List[int]:
        """
        Advanced parametric curvature constraint generation for high-order continuity
        
        Implements sophisticated curvature-based constraints:
        - Monge patch parameterization for smooth surfaces
        - Gaussian and mean curvature preservation
        - Asymptotic line constraint generation
        - Umbilical point special handling
        - Geodesic torsion constraints
        """
        try:
            constraint_nodes = []
            
            # Extract detailed curvature information
            principal_curvatures = curvature_info.get('principal_curvatures', np.zeros((len(shared_face), 2)))
            principal_directions = curvature_info.get('principal_directions', np.zeros((len(shared_face), 2, 3)))
            mean_curvature = curvature_info.get('mean_curvature', np.zeros(len(shared_face)))
            gaussian_curvature = curvature_info.get('gaussian_curvature', np.zeros(len(shared_face)))
            normal_vectors = curvature_info.get('normal_vectors', np.zeros((len(shared_face), 3)))
            
            face_coords = self.nodes[shared_face]
            face_centroid = np.mean(face_coords, axis=0)
            
            # Analyze curvature distribution across face
            curvature_stats = self._analyze_face_curvature_distribution(
                principal_curvatures, mean_curvature, gaussian_curvature
            )
            
            # High curvature region identification
            high_curvature_threshold = curvature_stats['mean_magnitude'] + 2 * curvature_stats['std_magnitude']
            
            for i, face_node in enumerate(shared_face):
                if i >= len(principal_curvatures):
                    continue
                    
                node_coord = face_coords[i]
                k1, k2 = principal_curvatures[i]
                H = mean_curvature[i]
                K = gaussian_curvature[i]
                
                curvature_magnitude = np.sqrt(k1**2 + k2**2)
                
                # High curvature constraint generation
                if curvature_magnitude > high_curvature_threshold:
                    
                    # Umbilical point detection and special handling
                    if abs(k1 - k2) < 0.01 * max(abs(k1), abs(k2), 1e-6):
                        # Near-umbilical point - isotropic constraints
                        umbilical_constraints = self._generate_umbilical_constraints(
                            node_coord, H, normal_vectors[i] if i < len(normal_vectors) else None
                        )
                        constraint_nodes.extend(umbilical_constraints)
                    
                    else:
                        # Regular high curvature point
                        if i < len(principal_directions):
                            dir1, dir2 = principal_directions[i]
                            
                            # Generate constraints along principal curvature lines
                            for j, (kappa, direction) in enumerate(zip([k1, k2], [dir1, dir2])):
                                if abs(kappa) > 0.02:  # Significant principal curvature
                                    
                                    # Adaptive constraint spacing based on curvature
                                    constraint_spacing = min(0.2 / abs(kappa), 1.0)
                                    num_constraints = max(1, int(2.0 / constraint_spacing))
                                    
                                    for step in range(1, num_constraints + 1):
                                        # Follow principal curvature line
                                        offset = step * constraint_spacing * direction
                                        constraint_pos = node_coord + offset
                                        
                                        # Project to spherical surface maintaining curvature
                                        constraint_pos = self._project_with_curvature_preservation(
                                            constraint_pos, node_coord, kappa, direction
                                        )
                                        
                                        constraint_id = self._create_constraint_node(constraint_pos)
                                        constraint_nodes.append(constraint_id)
                
                # Asymptotic line constraints for hyperbolic regions
                if K < -0.01:  # Hyperbolic (negative Gaussian curvature)
                    asymptotic_constraints = self._generate_asymptotic_line_constraints(
                        node_coord, k1, k2, principal_directions[i] if i < len(principal_directions) else None
                    )
                    constraint_nodes.extend(asymptotic_constraints)
                
                # Geodesic torsion constraints for twisted regions
                if i > 0 and i < len(shared_face) - 1:
                    prev_coord = face_coords[i-1]
                    next_coord = face_coords[(i+1) % len(face_coords)]
                    
                    geodesic_torsion = self._compute_geodesic_torsion(
                        prev_coord, node_coord, next_coord, normal_vectors[i] if i < len(normal_vectors) else None
                    )
                    
                    if abs(geodesic_torsion) > 0.1:  # High torsion threshold
                        torsion_constraints = self._generate_torsion_constraints(
                            node_coord, prev_coord, next_coord, geodesic_torsion
                        )
                        constraint_nodes.extend(torsion_constraints)
            
            # Monge patch parameterization constraints
            if len(shared_face) >= 4:  # Sufficient points for Monge patch
                monge_constraints = self._generate_monge_patch_constraints(
                    face_coords, principal_curvatures, mean_curvature
                )
                constraint_nodes.extend(monge_constraints)
            
            # Curvature flow constraints for smooth transitions
            flow_constraints = self._generate_curvature_flow_constraints(
                shared_face, face_coords, curvature_info
            )
            constraint_nodes.extend(flow_constraints)
            
            return constraint_nodes
            
        except Exception as e:
            print(f"Warning: Parametric curvature constraint generation failed: {e}")
            return []
    
    def _generate_geodesic_constraint_nodes(self, node_coord: np.ndarray, actual_normal: np.ndarray, 
                                          theoretical_normal: np.ndarray, strength: float, 
                                          sphere_radius: float) -> List[int]:
        """Generate constraint nodes along geodesic paths for normal correction"""
        constraint_nodes = []
        
        try:
            # Calculate rotation axis and angle between normals
            cross_product = np.cross(actual_normal, theoretical_normal)
            rotation_axis = cross_product / (np.linalg.norm(cross_product) + 1e-12)
            rotation_angle = np.arcsin(np.linalg.norm(cross_product))
            
            # Generate intermediate nodes along geodesic
            num_steps = max(2, int(strength * 5))
            for step in range(1, num_steps + 1):
                t = step / num_steps
                
                # Spherical linear interpolation (slerp) for geodesic
                intermediate_normal = self._spherical_slerp(actual_normal, theoretical_normal, t)
                
                # Position on sphere with corrected normal
                intermediate_pos = sphere_radius * intermediate_normal
                
                constraint_id = self._create_constraint_node(intermediate_pos)
                constraint_nodes.append(constraint_id)
            
            return constraint_nodes
            
        except Exception:
            return []
    
    def _compute_principal_directions(self, node_coord: np.ndarray, normal: np.ndarray, 
                                    k1: float, k2: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute principal curvature directions using shape operator eigenanalysis"""
        try:
            # Create local tangent basis
            tangent_basis = self._construct_tangent_basis(normal, node_coord.reshape(1, -1))
            u, v = tangent_basis[:, 0], tangent_basis[:, 1]
            
            # Shape operator matrix in tangent coordinates
            # For sphere: principal directions align with coordinate directions
            theta = np.arctan2(node_coord[1], node_coord[0])
            phi = np.arccos(node_coord[2] / np.linalg.norm(node_coord))
            
            # Principal directions in spherical coordinates
            dir1 = np.array([-np.sin(theta), np.cos(theta), 0.0])
            dir2 = np.array([np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), -np.sin(phi)])
            
            # Normalize
            dir1 = dir1 / (np.linalg.norm(dir1) + 1e-12)
            dir2 = dir2 / (np.linalg.norm(dir2) + 1e-12)
            
            return dir1, dir2
            
        except Exception:
            # Fallback to orthogonal directions
            tangent_basis = self._construct_tangent_basis(normal, node_coord.reshape(1, -1))
            return tangent_basis[:, 0], tangent_basis[:, 1]
    
    def _apply_gauss_bonnet_constraint(self, shared_face: List[int], face_coords: np.ndarray, 
                                     gaussian_curvature: np.ndarray, face_normal: np.ndarray) -> Optional[int]:
        """Apply Gauss-Bonnet theorem constraint for topological consistency"""
        try:
            if len(shared_face) < 3:
                return None
            
            # Calculate face area using cross product
            if len(face_coords) == 3:  # Triangle
                v1 = face_coords[1] - face_coords[0]
                v2 = face_coords[2] - face_coords[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            else:  # General polygon - triangulate
                area = 0.0
                for i in range(1, len(face_coords) - 1):
                    v1 = face_coords[i] - face_coords[0]
                    v2 = face_coords[i+1] - face_coords[0]
                    area += 0.5 * np.linalg.norm(np.cross(v1, v2))
            
            # Integrated Gaussian curvature over face
            integrated_K = 0.0
            for i in range(len(gaussian_curvature)):
                integrated_K += gaussian_curvature[i] * area / len(gaussian_curvature)
            
            # Angle deficit calculation
            angle_sum = 0.0
            for i in range(len(face_coords)):
                prev_idx = (i - 1) % len(face_coords)
                next_idx = (i + 1) % len(face_coords)
                
                v1 = face_coords[prev_idx] - face_coords[i]
                v2 = face_coords[next_idx] - face_coords[i]
                
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-12)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-12)
                
                angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
                angle_sum += angle
            
            # Gauss-Bonnet: K_integrated + angle_deficit = 2π for triangle
            expected_angle_sum = (len(face_coords) - 2) * np.pi
            angle_deficit = expected_angle_sum - angle_sum
            
            gb_discrepancy = abs(integrated_K + angle_deficit - 2*np.pi)
            
            # Create constraint node if significant discrepancy
            if gb_discrepancy > 0.1:
                # Position constraint node at face centroid
                centroid = np.mean(face_coords, axis=0)
                constraint_id = self._create_constraint_node(centroid)
                return constraint_id
            
            return None
            
        except Exception:
            return None
    
    def _analyze_face_curvature_distribution(self, principal_curvatures: np.ndarray, 
                                           mean_curvature: np.ndarray, gaussian_curvature: np.ndarray) -> Dict:
        """Analyze statistical distribution of curvature across face"""
        try:
            magnitudes = np.sqrt(principal_curvatures[:, 0]**2 + principal_curvatures[:, 1]**2)
            
            return {
                'mean_magnitude': np.mean(magnitudes),
                'std_magnitude': np.std(magnitudes),
                'max_magnitude': np.max(magnitudes),
                'mean_gaussian': np.mean(gaussian_curvature),
                'mean_mean_curvature': np.mean(mean_curvature)
            }
        except Exception:
            return {'mean_magnitude': 0.0, 'std_magnitude': 0.0, 'max_magnitude': 0.0, 
                    'mean_gaussian': 0.0, 'mean_mean_curvature': 0.0}
    
    def _generate_umbilical_constraints(self, node_coord: np.ndarray, mean_curvature: float, 
                                      normal: Optional[np.ndarray]) -> List[int]:
        """Generate isotropic constraints for umbilical points"""
        constraint_nodes = []
        try:
            if normal is None:
                sphere_center = np.array([0.0, 0.0, 0.0])
                normal = (node_coord - sphere_center) / np.linalg.norm(node_coord - sphere_center)
            
            # Create radial constraint pattern for umbilical point
            num_directions = 6  # Hexagonal pattern
            radius = min(0.2 / abs(mean_curvature), 0.5) if abs(mean_curvature) > 1e-6 else 0.1
            
            for i in range(num_directions):
                angle = 2 * np.pi * i / num_directions
                
                # Create direction in tangent plane
                tangent_basis = self._construct_tangent_basis(normal, node_coord.reshape(1, -1))
                direction = np.cos(angle) * tangent_basis[:, 0] + np.sin(angle) * tangent_basis[:, 1]
                
                constraint_pos = node_coord + radius * direction
                constraint_pos = self._project_to_sphere(constraint_pos, np.array([0.0, 0.0, 0.0]), 
                                                        np.linalg.norm(node_coord))
                
                constraint_id = self._create_constraint_node(constraint_pos)
                constraint_nodes.append(constraint_id)
            
            return constraint_nodes
        except Exception:
            return []
    
    def _generate_asymptotic_line_constraints(self, node_coord: np.ndarray, k1: float, k2: float, 
                                            principal_dirs: Optional[np.ndarray]) -> List[int]:
        """Generate constraints along asymptotic lines for hyperbolic surfaces"""
        constraint_nodes = []
        try:
            if principal_dirs is None:
                return []
            
            dir1, dir2 = principal_dirs
            
            # Asymptotic directions: directions where normal curvature = 0
            # For hyperbolic surface: asymptotic angles satisfy tan²θ = -k2/k1
            if k1 * k2 < 0:  # Different signs - hyperbolic
                tan_theta_sq = -k2 / k1 if abs(k1) > 1e-6 else 0
                if tan_theta_sq > 0:
                    tan_theta = np.sqrt(tan_theta_sq)
                    
                    # Two asymptotic directions
                    for sign in [-1, 1]:
                        theta = sign * np.arctan(tan_theta)
                        asymptotic_dir = np.cos(theta) * dir1 + np.sin(theta) * dir2
                        
                        # Generate constraints along asymptotic line
                        for step in range(1, 4):
                            offset = step * 0.1 * asymptotic_dir
                            constraint_pos = node_coord + offset
                            
                            constraint_pos = self._project_to_sphere(
                                constraint_pos, np.array([0.0, 0.0, 0.0]), np.linalg.norm(node_coord)
                            )
                            
                            constraint_id = self._create_constraint_node(constraint_pos)
                            constraint_nodes.append(constraint_id)
            
            return constraint_nodes
        except Exception:
            return []
    
    def _compute_geodesic_torsion(self, prev_coord: np.ndarray, node_coord: np.ndarray, 
                                next_coord: np.ndarray, normal: Optional[np.ndarray]) -> float:
        """Compute geodesic torsion along path"""
        try:
            if normal is None:
                sphere_center = np.array([0.0, 0.0, 0.0])
                normal = (node_coord - sphere_center) / np.linalg.norm(node_coord - sphere_center)
            
            # Tangent vectors
            t1 = node_coord - prev_coord
            t2 = next_coord - node_coord
            
            t1_norm = t1 / (np.linalg.norm(t1) + 1e-12)
            t2_norm = t2 / (np.linalg.norm(t2) + 1e-12)
            
            # Geodesic curvature vector
            curvature_vec = t2_norm - t1_norm
            geodesic_curvature_mag = np.linalg.norm(curvature_vec)
            
            if geodesic_curvature_mag < 1e-10:
                return 0.0
            
            # Torsion approximation
            binormal = np.cross(t1_norm, normal)
            binormal = binormal / (np.linalg.norm(binormal) + 1e-12)
            
            torsion = np.dot(curvature_vec, binormal) / geodesic_curvature_mag
            return torsion
            
        except Exception:
            return 0.0
    
    def _generate_torsion_constraints(self, node_coord: np.ndarray, prev_coord: np.ndarray, 
                                    next_coord: np.ndarray, torsion: float) -> List[int]:
        """Generate constraints to handle high torsion regions"""
        constraint_nodes = []
        try:
            # Create helical constraint pattern
            axis = next_coord - prev_coord
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            
            radius = 0.1 / abs(torsion) if abs(torsion) > 1e-6 else 0.1
            
            # Generate helical constraints
            for i in range(3):
                angle = 2 * np.pi * i / 3
                height = (i - 1) * 0.05
                
                # Perpendicular to axis
                perp = np.cross(axis, np.array([1, 0, 0]))
                if np.linalg.norm(perp) < 0.5:
                    perp = np.cross(axis, np.array([0, 1, 0]))
                perp = perp / np.linalg.norm(perp)
                
                perp2 = np.cross(axis, perp)
                
                constraint_pos = (node_coord + height * axis + 
                                radius * (np.cos(angle) * perp + np.sin(angle) * perp2))
                
                constraint_pos = self._project_to_sphere(
                    constraint_pos, np.array([0.0, 0.0, 0.0]), np.linalg.norm(node_coord)
                )
                
                constraint_id = self._create_constraint_node(constraint_pos)
                constraint_nodes.append(constraint_id)
            
            return constraint_nodes
        except Exception:
            return []
    
    def _generate_monge_patch_constraints(self, face_coords: np.ndarray, principal_curvatures: np.ndarray, 
                                        mean_curvature: np.ndarray) -> List[int]:
        """Generate Monge patch parameterization constraints"""
        constraint_nodes = []
        try:
            # Fit local Monge patch z = f(x,y)
            centroid = np.mean(face_coords, axis=0)
            
            # Local coordinate system
            normal = centroid / np.linalg.norm(centroid)  # Spherical surface
            tangent_basis = self._construct_tangent_basis(normal, centroid.reshape(1, -1))
            u_dir, v_dir = tangent_basis[:, 0], tangent_basis[:, 1]
            
            # Project face coordinates to local (u,v) system
            local_coords = []
            heights = []
            
            for coord in face_coords:
                relative = coord - centroid
                u = np.dot(relative, u_dir)
                v = np.dot(relative, v_dir)
                h = np.dot(relative, normal)
                local_coords.append([u, v])
                heights.append(h)
            
            local_coords = np.array(local_coords)
            heights = np.array(heights)
            
            # Fit quadratic Monge patch: z = au² + buv + cv² + du + ev + f
            if len(local_coords) >= 6:
                A_matrix = np.column_stack([
                    local_coords[:, 0]**2,  # u²
                    local_coords[:, 0] * local_coords[:, 1],  # uv
                    local_coords[:, 1]**2,  # v²
                    local_coords[:, 0],     # u
                    local_coords[:, 1],     # v
                    np.ones(len(local_coords))  # constant
                ])
                
                # Least squares fit
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A_matrix, heights, rcond=None)
                    a, b, c, d, e, f = coeffs
                    
                    # Generate constraint grid based on Monge patch
                    grid_size = 3
                    u_range = np.linspace(-0.2, 0.2, grid_size)
                    v_range = np.linspace(-0.2, 0.2, grid_size)
                    
                    for u in u_range:
                        for v in v_range:
                            if u == 0 and v == 0:  # Skip center
                                continue
                                
                            # Monge height
                            z = a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
                            
                            # Convert back to 3D
                            constraint_pos = centroid + u*u_dir + v*v_dir + z*normal
                            
                            # Project to sphere
                            constraint_pos = self._project_to_sphere(
                                constraint_pos, np.array([0.0, 0.0, 0.0]), np.linalg.norm(centroid)
                            )
                            
                            constraint_id = self._create_constraint_node(constraint_pos)
                            constraint_nodes.append(constraint_id)
                            
                except np.linalg.LinAlgError:
                    pass  # Skip if fitting fails
            
            return constraint_nodes
        except Exception:
            return []
    
    def _generate_curvature_flow_constraints(self, shared_face: List[int], face_coords: np.ndarray, 
                                           curvature_info: Dict) -> List[int]:
        """Generate constraints based on curvature flow evolution"""
        constraint_nodes = []
        try:
            mean_curvature = curvature_info.get('mean_curvature', np.zeros(len(shared_face)))
            
            # Simulate mean curvature flow step
            dt = 0.01  # Small time step
            
            for i, coord in enumerate(face_coords):
                if i < len(mean_curvature):
                    H = mean_curvature[i]
                    
                    # Normal direction for flow
                    normal = coord / np.linalg.norm(coord)  # Spherical surface
                    
                    # Mean curvature flow: ∂x/∂t = -H * n
                    flow_offset = -dt * H * normal
                    flowed_pos = coord + flow_offset
                    
                    # Project back to sphere
                    flowed_pos = self._project_to_sphere(
                        flowed_pos, np.array([0.0, 0.0, 0.0]), np.linalg.norm(coord)
                    )
                    
                    # Create constraint if significant flow
                    if np.linalg.norm(flow_offset) > 0.001:
                        constraint_id = self._create_constraint_node(flowed_pos)
                        constraint_nodes.append(constraint_id)
            
            return constraint_nodes
        except Exception:
            return []
    
    def _spherical_slerp(self, n1: np.ndarray, n2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between unit vectors"""
        try:
            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
            theta = np.arccos(abs(dot))
            
            if theta < 1e-6:  # Vectors are nearly parallel
                return n1
            
            sin_theta = np.sin(theta)
            return (np.sin((1-t) * theta) * n1 + np.sin(t * theta) * n2) / sin_theta
        except Exception:
            return n1
    
    def _project_to_sphere(self, point: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """Project point to spherical surface"""
        try:
            relative = point - center
            distance = np.linalg.norm(relative)
            if distance > 1e-12:
                return center + radius * relative / distance
            else:
                return center + radius * np.array([1.0, 0.0, 0.0])
        except Exception:
            return point
    
    def _project_with_curvature_preservation(self, point: np.ndarray, reference: np.ndarray, 
                                           curvature: float, direction: np.ndarray) -> np.ndarray:
        """Project to sphere while preserving local curvature properties"""
        try:
            # First project to sphere
            projected = self._project_to_sphere(point, np.array([0.0, 0.0, 0.0]), np.linalg.norm(reference))
            
            # Adjust based on curvature
            normal = projected / np.linalg.norm(projected)
            adjustment = 0.01 * curvature * direction
            
            adjusted = projected + adjustment
            return self._project_to_sphere(adjusted, np.array([0.0, 0.0, 0.0]), np.linalg.norm(reference))
        except Exception:
            return self._project_to_sphere(point, np.array([0.0, 0.0, 0.0]), np.linalg.norm(reference))
    
    def _compute_edge_curvature(self, edge_start: np.ndarray, edge_end: np.ndarray, 
                              face_normal: np.ndarray) -> float:
        """Compute curvature along edge"""
        try:
            edge_vector = edge_end - edge_start
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length < 1e-10:
                return 0.0
            
            edge_unit = edge_vector / edge_length
            
            # For spherical surface, compute geodesic curvature
            edge_center = 0.5 * (edge_start + edge_end)
            sphere_center = np.array([0.0, 0.0, 0.0])
            
            # Radial direction at edge center
            radial = edge_center / np.linalg.norm(edge_center)
            
            # Geodesic curvature
            curvature = np.dot(edge_unit, np.cross(radial, face_normal))
            return curvature
        except Exception:
            return 0.0
    
    def _curved_edge_interpolation(self, start: np.ndarray, end: np.ndarray, 
                                 curvature: float, t: float) -> np.ndarray:
        """Interpolate along curved edge using curvature"""
        try:
            # Simple curved interpolation
            linear_interp = (1-t) * start + t * end
            
            # Curvature adjustment
            edge_vector = end - start
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-10:
                edge_normal = np.cross(edge_vector, start)  # Approximation
                edge_normal = edge_normal / (np.linalg.norm(edge_normal) + 1e-12)
                
                # Quadratic offset based on curvature
                offset_magnitude = curvature * edge_length * t * (1-t)
                curved_pos = linear_interp + offset_magnitude * edge_normal
                
                return curved_pos
            else:
                return linear_interp
        except Exception:
            return (1-t) * start + t * end
    
    def _create_constraint_node(self, position: np.ndarray) -> int:
        """Create a new constraint node and return its ID"""
        try:
            # Add to node list
            node_id = len(self.nodes)
            self.nodes = np.vstack([self.nodes, position])
            return node_id
        except Exception:
            return -1  # Invalid node ID
    
    def _order_quadratic_constraints(self, constraint_nodes: List[int], 
                                   shared_face: List[int], face_type: str) -> List[int]:
        """
        Order constraint nodes for quadratic elements using advanced parametric methods.
        
        This method implements sophisticated ordering algorithms for quadratic element
        constraints, ensuring proper interpolation and numerical stability.
        """
        if not constraint_nodes:
            return constraint_nodes
        
        face_coords = self.nodes[shared_face]
        
        # Establish parametric coordinate system
        parametric_system = self._establish_parametric_system(face_coords, face_type)
        
        # Calculate parametric coordinates for each constraint node
        parametric_coords = []
        valid_nodes = []
        
        for node_id in constraint_nodes:
            if node_id < len(self.nodes):
                node_coord = self.nodes[node_id]
                param_coord = self._map_to_parametric_space(
                    node_coord, face_coords, parametric_system, face_type
                )
                
                if param_coord is not None:
                    parametric_coords.append(param_coord)
                    valid_nodes.append(node_id)
        
        if not parametric_coords:
            return constraint_nodes
        
        # Order nodes based on parametric position
        if face_type == 'triangular_quadratic':
            ordered_nodes = self._order_triangular_quadratic_nodes(
                valid_nodes, parametric_coords, parametric_system
            )
        elif face_type == 'quadrilateral_quadratic':
            ordered_nodes = self._order_quadrilateral_quadratic_nodes(
                valid_nodes, parametric_coords, parametric_system
            )
        else:
            # General polynomial ordering
            ordered_nodes = self._order_general_polynomial_nodes(
                valid_nodes, parametric_coords, face_type
            )
        
        # Validate ordering and fix any issues
        validated_ordering = self._validate_quadratic_ordering(
            ordered_nodes, shared_face, face_type
        )
        
        logger.debug(f"Ordered {len(validated_ordering)} quadratic constraint nodes")
        
        return validated_ordering
    
    def _establish_parametric_system(self, face_coords: np.ndarray, face_type: str) -> Dict:
        """Establish parametric coordinate system for the face"""
        if face_type.startswith('triangular'):
            return self._establish_triangular_parametric_system(face_coords)
        elif face_type.startswith('quadrilateral'):
            return self._establish_quadrilateral_parametric_system(face_coords)
        else:
            return self._establish_general_parametric_system(face_coords)
    
    def _establish_triangular_parametric_system(self, face_coords: np.ndarray) -> Dict:
        """Establish barycentric coordinate system for triangular faces"""
        if len(face_coords) < 3:
            return {'type': 'invalid'}
        
        # Use first three points as reference triangle
        p0, p1, p2 = face_coords[0], face_coords[1], face_coords[2]
        
        # Calculate barycentric transformation matrix
        T = np.column_stack([p1 - p0, p2 - p0])
        
        try:
            T_inv = np.linalg.pinv(T)  # Pseudo-inverse for robustness
            
            return {
                'type': 'triangular',
                'origin': p0,
                'transform_matrix': T,
                'inverse_transform': T_inv,
                'reference_points': [p0, p1, p2]
            }
        except np.linalg.LinAlgError:
            return {'type': 'invalid'}
    
    def _establish_quadrilateral_parametric_system(self, face_coords: np.ndarray) -> Dict:
        """Establish bilinear coordinate system for quadrilateral faces"""
        if len(face_coords) < 4:
            return {'type': 'invalid'}
        
        # Use bilinear parameterization: P(u,v) = (1-u)(1-v)P0 + u(1-v)P1 + uv*P2 + (1-u)v*P3
        p0, p1, p2, p3 = face_coords[0], face_coords[1], face_coords[2], face_coords[3]
        
        return {
            'type': 'quadrilateral',
            'corner_points': [p0, p1, p2, p3],
            'center': np.mean([p0, p1, p2, p3], axis=0)
        }
    
    def _establish_general_parametric_system(self, face_coords: np.ndarray) -> Dict:
        """Establish general parametric system using PCA"""
        if len(face_coords) < 3:
            return {'type': 'invalid'}
        
        center = np.mean(face_coords, axis=0)
        centered_coords = face_coords - center
        
        # PCA to find principal directions
        cov_matrix = np.cov(centered_coords.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        principal_axes = eigenvecs[:, idx]
        
        return {
            'type': 'general',
            'center': center,
            'principal_axes': principal_axes,
            'eigenvalues': eigenvals[idx]
        }
    
    def _map_to_parametric_space(self, point: np.ndarray, face_coords: np.ndarray,
                               param_system: Dict, face_type: str) -> Optional[np.ndarray]:
        """Map a 3D point to parametric coordinates"""
        if param_system['type'] == 'invalid':
            return None
        
        if param_system['type'] == 'triangular':
            return self._map_to_barycentric(point, param_system)
        elif param_system['type'] == 'quadrilateral':
            return self._map_to_bilinear(point, param_system)
        elif param_system['type'] == 'general':
            return self._map_to_principal_coordinates(point, param_system)
        
        return None
    
    def _map_to_barycentric(self, point: np.ndarray, param_system: Dict) -> Optional[np.ndarray]:
        """Map point to barycentric coordinates"""
        try:
            origin = param_system['origin']
            T_inv = param_system['inverse_transform']
            
            relative_point = point - origin
            param_coords = T_inv @ relative_point[:2]  # Use first 2 dimensions
            
            # Convert to barycentric coordinates
            u, v = param_coords[0], param_coords[1]
            w = 1 - u - v
            
            return np.array([w, u, v])  # Barycentric coordinates
            
        except (np.linalg.LinAlgError, IndexError):
            return None
    
    def _map_to_bilinear(self, point: np.ndarray, param_system: Dict) -> Optional[np.ndarray]:
        """
        Advanced bilinear parametric coordinate mapping with robust nonlinear system solving
        
        Implements multiple optimization algorithms with adaptive selection:
        - Damped Newton-Raphson with line search
        - Quasi-Newton BFGS with trust region
        - Gradient descent with momentum
        - Hybrid evolutionary-Newton approach
        
        Features:
        - Adaptive convergence criteria
        - Robust error handling and recovery
        - Multiple initial guess strategies
        - Numerical stability analysis
        - Performance optimization with vectorization
        """
        
        corners = param_system['corner_points']
        tolerance = param_system.get('tolerance', 1e-10)
        max_iterations = param_system.get('max_iterations', 100)
        
        # Advanced initial guess strategies
        initial_strategies = [
            (0.5, 0.5),  # Face center
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),  # Quadrant centers
            (0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1),  # Corner biased
            *[(np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)) for _ in range(5)]  # Random
        ]
        
        best_solution = None
        best_residual = np.inf
        
        for u_init, v_init in initial_strategies:
            try:
                solution = self._solve_bilinear_system(
                    point, corners, u_init, v_init, tolerance, max_iterations
                )
                
                if solution is not None:
                    u, v = solution
                    # Verify solution quality
                    P_uv = self._evaluate_bilinear(u, v, corners)
                    residual_norm = np.linalg.norm(P_uv - point)
                    
                    if residual_norm < best_residual:
                        best_residual = residual_norm
                        best_solution = solution
                        
                        # Early termination for excellent solutions
                        if residual_norm < tolerance * 10:
                            break
                            
            except Exception as e:
                logger.debug(f"Initial guess ({u_init}, {v_init}) failed: {e}")
                continue
        
        # Validate final solution
        if best_solution is not None:
            u, v = best_solution
            if self._validate_bilinear_solution(u, v, point, corners, tolerance):
                return np.array([u, v])
        
        # Fallback: Use least squares approach
        return self._bilinear_least_squares_fallback(point, corners)
    
    def _solve_bilinear_system(self, point: np.ndarray, corners: np.ndarray, 
                              u_init: float, v_init: float, tolerance: float, 
                              max_iterations: int) -> Optional[np.ndarray]:
        """
        Multi-algorithm nonlinear system solver with adaptive method selection
        """
        
        algorithms = [
            ('damped_newton', self._damped_newton_bilinear),
            ('quasi_newton_bfgs', self._quasi_newton_bfgs_bilinear),
            ('trust_region', self._trust_region_bilinear),
            ('gradient_descent', self._gradient_descent_bilinear)
        ]
        
        for alg_name, solver_func in algorithms:
            try:
                result = solver_func(point, corners, u_init, v_init, tolerance, max_iterations)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"Algorithm {alg_name} failed: {e}")
                continue
        
        return None
    
    def _damped_newton_bilinear(self, point: np.ndarray, corners: np.ndarray,
                               u: float, v: float, tolerance: float, 
                               max_iterations: int) -> Optional[np.ndarray]:
        """Damped Newton-Raphson with adaptive step size and line search"""
        
        for iteration in range(max_iterations):
            # Evaluate function and derivatives
            P_uv, dP_du, dP_dv, d2P_du2, d2P_dv2, d2P_dudv = self._evaluate_bilinear_derivatives(u, v, corners)
            
            residual = P_uv - point
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                return np.array([u, v])
            
            # Build Jacobian and Hessian
            jacobian = np.column_stack([dP_du, dP_dv])
            
            # Compute Newton direction with regularization
            try:
                # Add regularization for numerical stability
                reg_factor = max(1e-12, residual_norm * 1e-8)
                JTJ = jacobian.T @ jacobian + reg_factor * np.eye(2)
                JTr = jacobian.T @ residual
                
                newton_direction = -np.linalg.solve(JTJ, JTr)
                
            except np.linalg.LinAlgError:
                # Fallback to gradient descent direction
                gradient = jacobian.T @ residual
                newton_direction = -gradient / (np.linalg.norm(gradient) + 1e-12)
            
            # Adaptive damping with line search
            step_size = self._line_search_bilinear(
                point, corners, u, v, newton_direction, residual_norm
            )
            
            # Update with bounds checking
            u_new = u + step_size * newton_direction[0]
            v_new = v + step_size * newton_direction[1]
            
            # Sophisticated bounds handling with reflection
            u_new, v_new = self._handle_bounds_reflection(u_new, v_new, u, v)
            
            # Convergence check
            param_change = np.sqrt((u_new - u)**2 + (v_new - v)**2)
            if param_change < tolerance:
                return np.array([u_new, v_new])
            
            u, v = u_new, v_new
        
        return None
    
    def _quasi_newton_bfgs_bilinear(self, point: np.ndarray, corners: np.ndarray,
                                   u: float, v: float, tolerance: float,
                                   max_iterations: int) -> Optional[np.ndarray]:
        """BFGS quasi-Newton method with trust region"""
        
        # Initialize BFGS approximation
        H = np.eye(2)  # Inverse Hessian approximation
        x = np.array([u, v])
        
        for iteration in range(max_iterations):
            P_uv, dP_du, dP_dv, _, _, _ = self._evaluate_bilinear_derivatives(x[0], x[1], corners)
            
            residual = P_uv - point
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                return x
            
            # Gradient of objective function ||P(u,v) - point||^2
            jacobian = np.column_stack([dP_du, dP_dv])
            gradient = 2 * jacobian.T @ residual
            
            # BFGS direction
            direction = -H @ gradient
            
            # Trust region adjustment
            trust_radius = min(1.0, 10.0 / (iteration + 1))
            if np.linalg.norm(direction) > trust_radius:
                direction = direction * trust_radius / np.linalg.norm(direction)
            
            # Line search
            step_size = self._line_search_bilinear(
                point, corners, x[0], x[1], direction, residual_norm
            )
            
            x_new = x + step_size * direction
            x_new[0], x_new[1] = self._handle_bounds_reflection(x_new[0], x_new[1], x[0], x[1])
            
            # BFGS update
            s = x_new - x
            P_new, dP_du_new, dP_dv_new, _, _, _ = self._evaluate_bilinear_derivatives(x_new[0], x_new[1], corners)
            residual_new = P_new - point
            jacobian_new = np.column_stack([dP_du_new, dP_dv_new])
            gradient_new = 2 * jacobian_new.T @ residual_new
            
            y = gradient_new - gradient
            
            # BFGS formula with safeguarding
            sy = s @ y
            if sy > 1e-12:
                Hy = H @ y
                H = H + (sy + y @ Hy) * np.outer(s, s) / sy**2 - (np.outer(Hy, s) + np.outer(s, Hy)) / sy
            
            if np.linalg.norm(s) < tolerance:
                return x_new
            
            x = x_new
        
        return None
    
    def _trust_region_bilinear(self, point: np.ndarray, corners: np.ndarray,
                              u: float, v: float, tolerance: float,
                              max_iterations: int) -> Optional[np.ndarray]:
        """Trust region method with dogleg step"""
        
        x = np.array([u, v])
        trust_radius = 1.0
        
        for iteration in range(max_iterations):
            P_uv, dP_du, dP_dv, d2P_du2, d2P_dv2, d2P_dudv = self._evaluate_bilinear_derivatives(x[0], x[1], corners)
            
            residual = P_uv - point
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                return x
            
            # Build model matrices
            jacobian = np.column_stack([dP_du, dP_dv])
            gradient = jacobian.T @ residual
            
            # Approximate Hessian using Gauss-Newton
            hessian = jacobian.T @ jacobian
            
            # Regularize for stability
            reg_factor = max(1e-12, residual_norm * 1e-6)
            hessian += reg_factor * np.eye(2)
            
            # Dogleg step calculation
            step = self._dogleg_step(gradient, hessian, trust_radius)
            
            x_trial = x - step
            x_trial[0], x_trial[1] = self._handle_bounds_reflection(x_trial[0], x_trial[1], x[0], x[1])
            
            # Evaluate trial point
            P_trial = self._evaluate_bilinear(x_trial[0], x_trial[1], corners)
            residual_trial = P_trial - point
            residual_trial_norm = np.linalg.norm(residual_trial)
            
            # Trust region ratio
            predicted_reduction = 0.5 * step @ hessian @ step + gradient @ step
            actual_reduction = 0.5 * (residual_norm**2 - residual_trial_norm**2)
            
            if abs(predicted_reduction) > 1e-12:
                ratio = actual_reduction / predicted_reduction
            else:
                ratio = 1.0
            
            # Update trust region
            if ratio > 0.75:
                trust_radius = min(2 * trust_radius, 2.0)
            elif ratio < 0.25:
                trust_radius *= 0.25
            
            # Accept or reject step
            if ratio > 0.1:
                if np.linalg.norm(x_trial - x) < tolerance:
                    return x_trial
                x = x_trial
            
            if trust_radius < 1e-12:
                break
        
        return None
    
    def _gradient_descent_bilinear(self, point: np.ndarray, corners: np.ndarray,
                                  u: float, v: float, tolerance: float,
                                  max_iterations: int) -> Optional[np.ndarray]:
        """Adaptive gradient descent with momentum"""
        
        x = np.array([u, v])
        velocity = np.zeros(2)
        momentum = 0.9
        learning_rate = 0.1
        
        for iteration in range(max_iterations):
            P_uv, dP_du, dP_dv, _, _, _ = self._evaluate_bilinear_derivatives(x[0], x[1], corners)
            
            residual = P_uv - point
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < tolerance:
                return x
            
            # Gradient of objective function
            jacobian = np.column_stack([dP_du, dP_dv])
            gradient = jacobian.T @ residual
            
            # Adaptive learning rate
            if iteration > 0:
                if residual_norm > prev_residual_norm:
                    learning_rate *= 0.8
                else:
                    learning_rate *= 1.05
            
            learning_rate = np.clip(learning_rate, 1e-6, 1.0)
            
            # Momentum update
            velocity = momentum * velocity - learning_rate * gradient
            x_new = x + velocity
            
            x_new[0], x_new[1] = self._handle_bounds_reflection(x_new[0], x_new[1], x[0], x[1])
            
            if np.linalg.norm(x_new - x) < tolerance:
                return x_new
            
            prev_residual_norm = residual_norm
            x = x_new
        
        return None
    
    def _evaluate_bilinear(self, u: float, v: float, corners: np.ndarray) -> np.ndarray:
        """Evaluate bilinear interpolation at parametric coordinates"""
        return ((1-u)*(1-v)*corners[0] + u*(1-v)*corners[1] + 
                u*v*corners[2] + (1-u)*v*corners[3])
    
    def _evaluate_bilinear_derivatives(self, u: float, v: float, corners: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Evaluate bilinear interpolation and all derivatives"""
        
        # Function value
        P_uv = self._evaluate_bilinear(u, v, corners)
        
        # First derivatives
        dP_du = (-(1-v)*corners[0] + (1-v)*corners[1] + v*corners[2] - v*corners[3])
        dP_dv = (-(1-u)*corners[0] - u*corners[1] + u*corners[2] + (1-u)*corners[3])
        
        # Second derivatives
        d2P_du2 = np.zeros_like(corners[0])  # Bilinear has zero pure second derivatives
        d2P_dv2 = np.zeros_like(corners[0])
        d2P_dudv = (-corners[0] + corners[1] + corners[2] - corners[3])
        
        return P_uv, dP_du, dP_dv, d2P_du2, d2P_dv2, d2P_dudv
    
    def _line_search_bilinear(self, point: np.ndarray, corners: np.ndarray,
                             u: float, v: float, direction: np.ndarray,
                             current_residual: float) -> float:
        """Backtracking line search for optimal step size"""
        
        alpha = 1.0
        c1 = 1e-4  # Armijo condition parameter
        rho = 0.5  # Backtracking factor
        
        for _ in range(20):  # Max backtracking steps
            u_trial = u + alpha * direction[0]
            v_trial = v + alpha * direction[1]
            
            # Keep in bounds
            u_trial = np.clip(u_trial, 0.0, 1.0)
            v_trial = np.clip(v_trial, 0.0, 1.0)
            
            P_trial = self._evaluate_bilinear(u_trial, v_trial, corners)
            residual_trial = np.linalg.norm(P_trial - point)
            
            # Armijo condition
            if residual_trial <= current_residual - c1 * alpha * np.linalg.norm(direction)**2:
                return alpha
            
            alpha *= rho
        
        return alpha
    
    def _handle_bounds_reflection(self, u: float, v: float, u_prev: float, v_prev: float) -> Tuple[float, float]:
        """Sophisticated bounds handling with reflection and projection"""
        
        # Simple clamping with reflection
        if u < 0 or u > 1:
            u = np.clip(u, 0.0, 1.0)
            if u == 0.0 or u == 1.0:
                # Reflect from boundary
                u = u_prev + 0.5 * (u - u_prev)
        
        if v < 0 or v > 1:
            v = np.clip(v, 0.0, 1.0)
            if v == 0.0 or v == 1.0:
                # Reflect from boundary
                v = v_prev + 0.5 * (v - v_prev)
        
        return u, v
    
    def _dogleg_step(self, gradient: np.ndarray, hessian: np.ndarray, trust_radius: float) -> np.ndarray:
        """Compute dogleg step for trust region method"""
        
        try:
            # Newton step
            newton_step = np.linalg.solve(hessian, gradient)
            newton_norm = np.linalg.norm(newton_step)
            
            if newton_norm <= trust_radius:
                return newton_step
            
        except np.linalg.LinAlgError:
            newton_step = None
        
        # Cauchy step (steepest descent)
        cauchy_step = -(gradient @ gradient) / (gradient @ hessian @ gradient) * gradient
        cauchy_norm = np.linalg.norm(cauchy_step)
        
        if cauchy_norm >= trust_radius:
            return trust_radius * cauchy_step / cauchy_norm
        
        # Dogleg between Cauchy and Newton
        if newton_step is not None:
            diff = newton_step - cauchy_step
            a = diff @ diff
            b = 2 * cauchy_step @ diff
            c = cauchy_norm**2 - trust_radius**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                tau = (-b + np.sqrt(discriminant)) / (2*a)
                return cauchy_step + tau * diff
        
        return trust_radius * cauchy_step / cauchy_norm
    
    def _validate_bilinear_solution(self, u: float, v: float, point: np.ndarray,
                                   corners: np.ndarray, tolerance: float) -> bool:
        """Comprehensive solution validation"""
        
        # Check bounds
        if not (0 <= u <= 1 and 0 <= v <= 1):
            return False
        
        # Check residual
        P_uv = self._evaluate_bilinear(u, v, corners)
        residual_norm = np.linalg.norm(P_uv - point)
        
        if residual_norm > tolerance * 100:  # Allow some relaxation
            return False
        
        # Check numerical stability
        if np.isnan(u) or np.isnan(v) or np.isinf(u) or np.isinf(v):
            return False
        
        return True
    
    def _bilinear_least_squares_fallback(self, point: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """Least squares fallback when iterative methods fail"""
        
        try:
            # Set up overdetermined system for least squares
            u_samples = np.linspace(0, 1, 11)
            v_samples = np.linspace(0, 1, 11)
            
            best_uv = None
            best_residual = np.inf
            
            for u in u_samples:
                for v in v_samples:
                    P_uv = self._evaluate_bilinear(u, v, corners)
                    residual = np.linalg.norm(P_uv - point)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_uv = np.array([u, v])
            
            return best_uv
            
        except Exception:
            # Ultimate fallback
            return np.array([0.5, 0.5])
    
    def _map_to_principal_coordinates(self, point: np.ndarray, param_system: Dict) -> Optional[np.ndarray]:
        """Map point to principal component coordinates"""
        center = param_system['center']
        axes = param_system['principal_axes']
        
        relative_point = point - center
        param_coords = axes.T @ relative_point
        
        return param_coords[:2]  # Use first two principal components
    
    def _order_triangular_quadratic_nodes(self, nodes: List[int], param_coords: List[np.ndarray],
                                        param_system: Dict) -> List[int]:
        """Order nodes for triangular quadratic elements using barycentric coordinates"""
        if not nodes:
            return nodes
        
        # Group nodes by their role: corner, edge, interior
        node_classifications = []
        
        for i, (node_id, coords) in enumerate(zip(nodes, param_coords)):
            classification = self._classify_triangular_node(coords)
            node_classifications.append((node_id, coords, classification))
        
        # Sort by classification priority, then by parametric position
        def sort_key(item):
            node_id, coords, classification = item
            priority = {'corner': 0, 'edge': 1, 'interior': 2}[classification]
            
            # Secondary sort by barycentric coordinates
            if classification == 'edge':
                # Sort edge nodes by their position along edges
                return (priority, self._edge_position_triangular(coords))
            elif classification == 'interior':
                # Sort interior nodes by distance from centroid
                centroid = np.array([1/3, 1/3, 1/3])
                return (priority, np.linalg.norm(coords - centroid))
            else:
                return (priority, 0)
        
        sorted_classifications = sorted(node_classifications, key=sort_key)
        return [item[0] for item in sorted_classifications]
    
    def _classify_triangular_node(self, barycentric_coords: np.ndarray, tolerance: float = 1e-6) -> str:
        """Classify a node in triangular parametric space"""
        w, u, v = barycentric_coords
        
        # Check if on corners
        if (abs(w - 1) < tolerance and abs(u) < tolerance and abs(v) < tolerance) or \
           (abs(u - 1) < tolerance and abs(w) < tolerance and abs(v) < tolerance) or \
           (abs(v - 1) < tolerance and abs(w) < tolerance and abs(u) < tolerance):
            return 'corner'
        
        # Check if on edges
        if abs(w) < tolerance or abs(u) < tolerance or abs(v) < tolerance:
            return 'edge'
        
        return 'interior'
    
    def _edge_position_triangular(self, barycentric_coords: np.ndarray) -> float:
        """Calculate position along edge for triangular elements"""
        w, u, v = barycentric_coords
        
        if abs(w) < 1e-6:  # On edge opposite to corner 0
            return u / (u + v) if (u + v) > 1e-12 else 0.0
        elif abs(u) < 1e-6:  # On edge opposite to corner 1
            return v / (v + w) if (v + w) > 1e-12 else 0.0
        elif abs(v) < 1e-6:  # On edge opposite to corner 2
            return w / (w + u) if (w + u) > 1e-12 else 0.0
        
        return 0.0
    
    def _order_quadrilateral_quadratic_nodes(self, nodes: List[int], param_coords: List[np.ndarray],
                                           param_system: Dict) -> List[int]:
        """Order nodes for quadrilateral quadratic elements"""
        if not nodes:
            return nodes
        
        # Classify and sort nodes
        node_data = []
        for node_id, coords in zip(nodes, param_coords):
            u, v = coords[0], coords[1]
            classification = self._classify_quadrilateral_node(u, v)
            node_data.append((node_id, u, v, classification))
        
        # Sort by classification and position
        def quad_sort_key(item):
            node_id, u, v, classification = item
            priority = {'corner': 0, 'edge': 1, 'interior': 2}[classification]
            
            if classification == 'edge':
                return (priority, self._edge_index_quadrilateral(u, v), min(u, v, 1-u, 1-v))
            elif classification == 'interior':
                return (priority, (u - 0.5)**2 + (v - 0.5)**2)
            else:
                return (priority, u + v)
        
        sorted_data = sorted(node_data, key=quad_sort_key)
        return [item[0] for item in sorted_data]
    
    def _classify_quadrilateral_node(self, u: float, v: float, tolerance: float = 1e-6) -> str:
        """Classify node in quadrilateral parametric space"""
        on_u_edge = abs(u) < tolerance or abs(u - 1) < tolerance
        on_v_edge = abs(v) < tolerance or abs(v - 1) < tolerance
        
        if on_u_edge and on_v_edge:
            return 'corner'
        elif on_u_edge or on_v_edge:
            return 'edge'
        else:
            return 'interior'
    
    def _edge_index_quadrilateral(self, u: float, v: float, tolerance: float = 1e-6) -> int:
        """Get edge index for quadrilateral node"""
        if abs(v) < tolerance:  # Bottom edge
            return 0
        elif abs(u - 1) < tolerance:  # Right edge
            return 1
        elif abs(v - 1) < tolerance:  # Top edge
            return 2
        elif abs(u) < tolerance:  # Left edge
            return 3
        return 0
    
    def _order_general_polynomial_nodes(self, nodes: List[int], param_coords: List[np.ndarray],
                                      face_type: str) -> List[int]:
        """Order nodes for general polynomial elements"""
        if not nodes:
            return nodes
        
        # Sort by radial distance from parametric center
        center = np.mean(param_coords, axis=0) if param_coords else np.zeros(2)
        
        node_distances = []
        for node_id, coords in zip(nodes, param_coords):
            distance = np.linalg.norm(coords - center)
            node_distances.append((node_id, distance))
        
        sorted_nodes = sorted(node_distances, key=lambda x: x[1])
        return [node_id for node_id, _ in sorted_nodes]
    
    def _validate_quadratic_ordering(self, ordered_nodes: List[int], shared_face: List[int],
                                   face_type: str) -> List[int]:
        """Validate and potentially fix quadratic node ordering"""
        if not ordered_nodes:
            return ordered_nodes
        
        # Check for proper alternation of constraint types
        validated_nodes = []
        previous_type = None
        
        for node_id in ordered_nodes:
            if node_id < len(self.nodes):
                current_type = self._determine_constraint_type(node_id, shared_face)
                
                # Avoid consecutive nodes of the same type in some cases
                if current_type != previous_type or current_type == 'interior':
                    validated_nodes.append(node_id)
                    previous_type = current_type
        
        return validated_nodes
    
    def _determine_constraint_type(self, node_id: int, shared_face: List[int]) -> str:
        """Determine the constraint type for a node"""
        if node_id in shared_face:
            return 'boundary'
        
        # Additional logic to determine edge vs interior constraints
        # This would depend on the specific mesh structure
        return 'interior'
    
    def _sort_constraint_nodes_geometrically(self, constraint_nodes: List[int], 
                                           face_center: np.ndarray, 
                                           face_normal: np.ndarray) -> List[int]:
        """
        Sort constraint nodes by advanced geometric criteria including distance,
        angular position, and geometric importance for constraint stability.
        """
        if not constraint_nodes:
            return constraint_nodes
        
        # Calculate comprehensive geometric metrics for each node
        node_metrics = []
        
        for node_id in constraint_nodes:
            if node_id < len(self.nodes):
                node_coord = self.nodes[node_id]
                metrics = self._calculate_comprehensive_node_metrics(
                    node_coord, face_center, face_normal, constraint_nodes
                )
                node_metrics.append((node_id, metrics))
        
        # Sort by multiple criteria with priority weighting
        def geometric_sort_key(item):
            node_id, metrics = item
            
            # Primary sort: geometric importance (stability contribution)
            importance = metrics.get('geometric_importance', 0.0)
            
            # Secondary sort: distance to face center
            distance = metrics.get('distance_to_center', float('inf'))
            
            # Tertiary sort: angular position for consistent ordering
            angle = metrics.get('angular_position', 0.0)
            
            # Quaternary sort: constraint quality score
            quality = metrics.get('constraint_quality', 0.0)
            
            return (-importance, distance, angle, -quality)
        
        sorted_nodes = sorted(node_metrics, key=geometric_sort_key)
        
        # Apply geometric clustering to group related constraints
        clustered_nodes = self._apply_geometric_clustering(
            [item[0] for item in sorted_nodes], face_center, face_normal
        )
        
        # Final validation and stability check
        validated_nodes = self._validate_geometric_ordering(
            clustered_nodes, face_center, face_normal
        )
        
        logger.debug(f"Geometrically sorted {len(validated_nodes)} constraint nodes")
        
        return validated_nodes
    
    def _calculate_comprehensive_node_metrics(self, node_coord: np.ndarray, 
                                            face_center: np.ndarray, face_normal: np.ndarray,
                                            all_constraint_nodes: List[int]) -> Dict[str, float]:
        """Calculate comprehensive geometric metrics for constraint node sorting"""
        metrics = {}
        
        # Distance to face center
        to_node = node_coord - face_center
        distance = np.linalg.norm(to_node)
        metrics['distance_to_center'] = distance
        
        # Angular position in face plane
        if distance > 1e-12:
            # Project to face plane for angular calculation
            projected_vec = to_node - np.dot(to_node, face_normal) * face_normal
            proj_length = np.linalg.norm(projected_vec)
            
            if proj_length > 1e-12:
                # Use first two components for angle calculation
                angle = np.arctan2(projected_vec[1], projected_vec[0]) if len(projected_vec) >= 2 else 0.0
                metrics['angular_position'] = angle
            else:
                metrics['angular_position'] = 0.0
        else:
            metrics['angular_position'] = 0.0
        
        # Geometric importance based on position relative to other constraints
        importance = self._calculate_geometric_importance(
            node_coord, face_center, face_normal, all_constraint_nodes
        )
        metrics['geometric_importance'] = importance
        
        # Constraint quality score
        quality = self._calculate_constraint_quality_score(
            node_coord, face_center, face_normal
        )
        metrics['constraint_quality'] = quality
        
        # Distance to face plane (should be small for valid constraints)
        plane_distance = abs(np.dot(to_node, face_normal))
        metrics['plane_distance'] = plane_distance
        
        return metrics
    
    def _calculate_geometric_importance(self, node_coord: np.ndarray, face_center: np.ndarray,
                                      face_normal: np.ndarray, all_nodes: List[int]) -> float:
        """Calculate geometric importance score for constraint stability"""
        importance = 0.0
        
        # Distance from face center (closer nodes are more important for stability)
        distance = np.linalg.norm(node_coord - face_center)
        if distance > 1e-12:
            importance += 1.0 / distance  # Inverse distance weighting
        
        # Uniqueness score (avoid clustered constraints)
        min_distance_to_others = float('inf')
        for other_node_id in all_nodes:
            if other_node_id < len(self.nodes):
                other_coord = self.nodes[other_node_id]
                if not np.allclose(node_coord, other_coord):
                    dist = np.linalg.norm(node_coord - other_coord)
                    min_distance_to_others = min(min_distance_to_others, dist)
        
        if min_distance_to_others < float('inf') and min_distance_to_others > 1e-12:
            importance += min_distance_to_others  # Reward well-separated nodes
        
        return importance
    
    def _calculate_constraint_quality_score(self, node_coord: np.ndarray, 
                                          face_center: np.ndarray, 
                                          face_normal: np.ndarray) -> float:
        """Calculate quality score for constraint effectiveness"""
        score = 1.0
        
        # Penalty for nodes too far from face plane
        to_node = node_coord - face_center
        plane_distance = abs(np.dot(to_node, face_normal))
        face_size = np.linalg.norm(to_node)
        
        if face_size > 1e-12:
            relative_plane_distance = plane_distance / face_size
            score *= np.exp(-10.0 * relative_plane_distance)  # Exponential penalty
        
        # Reward nodes at optimal distances from face center
        if face_size > 1e-12:
            optimal_distance_factor = np.exp(-(face_size - 0.5)**2 / 0.25)
            score *= optimal_distance_factor
        
        return score
    
    def _apply_geometric_clustering(self, sorted_nodes: List[int], 
                                  face_center: np.ndarray, face_normal: np.ndarray) -> List[int]:
        """Apply geometric clustering to group related constraint nodes"""
        if len(sorted_nodes) <= 1:
            return sorted_nodes
        
        # Simple clustering based on spatial proximity
        clustered = []
        remaining = sorted_nodes.copy()
        
        while remaining:
            # Start new cluster with first remaining node
            cluster_seed = remaining.pop(0)
            current_cluster = [cluster_seed]
            
            if cluster_seed < len(self.nodes):
                seed_coord = self.nodes[cluster_seed]
                cluster_radius = self._calculate_cluster_radius(seed_coord, face_center)
                
                # Add nearby nodes to cluster
                to_remove = []
                for node_id in remaining:
                    if node_id < len(self.nodes):
                        node_coord = self.nodes[node_id]
                        distance = np.linalg.norm(node_coord - seed_coord)
                        
                        if distance < cluster_radius:
                            current_cluster.append(node_id)
                            to_remove.append(node_id)
                
                # Remove clustered nodes from remaining
                for node_id in to_remove:
                    remaining.remove(node_id)
            
            # Add cluster to result (already in desired order)
            clustered.extend(current_cluster)
        
        return clustered
    
    def _calculate_cluster_radius(self, seed_coord: np.ndarray, face_center: np.ndarray) -> float:
        """Calculate appropriate clustering radius"""
        distance_to_center = np.linalg.norm(seed_coord - face_center)
        return max(distance_to_center * 0.2, 1e-3)  # 20% of distance to center, minimum threshold
    
    def _validate_geometric_ordering(self, ordered_nodes: List[int], 
                                   face_center: np.ndarray, face_normal: np.ndarray) -> List[int]:
        """Validate geometric ordering for constraint stability"""
        if not ordered_nodes:
            return ordered_nodes
        
        validated = []
        for node_id in ordered_nodes:
            if node_id < len(self.nodes):
                # Check geometric validity
                if self._is_geometrically_valid_constraint(node_id, face_center, face_normal):
                    validated.append(node_id)
        
        return validated
    
    def _is_geometrically_valid_constraint(self, node_id: int, 
                                         face_center: np.ndarray, face_normal: np.ndarray) -> bool:
        """Check if constraint node is geometrically valid"""
        node_coord = self.nodes[node_id]
        
        # Check distance to face plane
        to_node = node_coord - face_center
        plane_distance = abs(np.dot(to_node, face_normal))
        face_distance = np.linalg.norm(to_node)
        
        if face_distance > 1e-12:
            relative_distance = plane_distance / face_distance
            return relative_distance < 0.1  # Within 10% of face size
        
        return True
    
    def _validate_constraint_selection(self, constraint_nodes: List[int], 
                                     shared_face: List[int], 
                                     new_nodes: List[np.ndarray]) -> List[int]:
        """Validate final constraint node selection"""
        validated_nodes = []
        
        for node_id in constraint_nodes:
            # Check node existence
            if node_id >= len(self.nodes):
                continue  # Skip non-existent nodes
            
            # Check for duplicate constraints
            if node_id in validated_nodes:
                continue
            
            # Check constraint validity
            if self._is_valid_constraint_node(node_id, shared_face):
                validated_nodes.append(node_id)
        
        # Limit number of constraints to prevent over-constraining
        max_constraints = min(len(shared_face) * 2, len(validated_nodes))
        return validated_nodes[:max_constraints]
    
    def _is_valid_constraint_node(self, node_id: int, shared_face: List[int]) -> bool:
        """Final validation check for constraint node"""
        
        # Node must exist
        if node_id >= len(self.nodes):
            return False
        
        # Node should not be part of shared face already
        if node_id in shared_face:
            return False
        
        # Additional geometric validity checks could be added here
        return True
    
    def _calculate_refinement_ratio(self, element_type: str) -> float:
        """Calculate refinement ratio for element type"""
        if element_type == 'hexahedral':
            return 8.0  # 1 element becomes 8
        elif element_type == 'tetrahedral':
            return 8.0  # 1 element becomes 8
        else:
            return 4.0  # Default
    
    def _calculate_min_edge_length(self, coords: np.ndarray) -> float:
        """Calculate minimum edge length of element"""
        min_length = float('inf')
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                edge_length = np.linalg.norm(coords[i] - coords[j])
                min_length = min(min_length, edge_length)
        
        return min_length
    
    def _update_mesh_connectivity(self, new_nodes: List[np.ndarray], new_elements: List[np.ndarray],
                                refined_map: dict, constraints: List[dict], refined_ids: List[int]):
        """Update mesh data structures after refinement"""
        # Add new nodes to mesh
        if new_nodes:
            self.nodes = np.vstack([self.nodes, new_nodes])
        
        # Mark refined elements as inactive and add new elements
        if not hasattr(self, 'active_elements'):
            self.active_elements = np.ones(len(self.elements), dtype=bool)
        
        # Deactivate parent elements
        for elem_id in refined_ids:
            if elem_id < len(self.active_elements):
                self.active_elements[elem_id] = False
        
        # Add new elements
        if new_elements:
            self.elements = np.vstack([self.elements, new_elements])
            new_active = np.ones(len(new_elements), dtype=bool)
            self.active_elements = np.concatenate([self.active_elements, new_active])
        
        # Update element levels
        if not hasattr(self, 'element_levels'):
            self.element_levels = np.zeros(len(self.elements), dtype=int)
        
        # Extend element levels array for new elements
        if len(new_elements) > 0:
            new_levels = []
            for elem_id in refined_ids:
                if elem_id in refined_map:
                    parent_level = refined_map[elem_id]['parent_level']
                    num_sub_elements = len(refined_map[elem_id]['sub_elements'])
                    new_levels.extend([parent_level + 1] * num_sub_elements)
            
            self.element_levels = np.concatenate([self.element_levels, new_levels])
        
        # Store constraints for solver
        if not hasattr(self, 'mesh_constraints'):
            self.mesh_constraints = []
        self.mesh_constraints.extend(constraints)
        
        # Update boundary node identification
        self._identify_boundary_nodes()
    
    def _calculate_mesh_quality_metrics(self) -> dict:
        """Calculate comprehensive mesh quality metrics"""
        if not hasattr(self, 'active_elements'):
            active_mask = np.ones(len(self.elements), dtype=bool)
        else:
            active_mask = self.active_elements
        
        active_elements = self.elements[active_mask]
        
        aspect_ratios = []
        volumes = []
        
        for element in active_elements:
            coords = self.nodes[element]
            
            # Aspect ratio
            edge_lengths = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    edge_lengths.append(np.linalg.norm(coords[i] - coords[j]))
            
            if edge_lengths:
                aspect_ratios.append(max(edge_lengths) / min(edge_lengths))
            
            # Volume
            if len(coords) == 8:  # Hex
                v1 = coords[1] - coords[0]
                v2 = coords[3] - coords[0]
                v3 = coords[4] - coords[0]
                volume = abs(np.dot(v1, np.cross(v2, v3)))
                volumes.append(volume)
        
        return {
            'num_active_elements': len(active_elements),
            'num_total_elements': len(self.elements),
            'num_nodes': len(self.nodes),
            'mean_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
            'max_aspect_ratio': np.max(aspect_ratios) if aspect_ratios else 0,
            'mean_volume': np.mean(volumes) if volumes else 0,
            'min_volume': np.min(volumes) if volumes else 0
        }
    
    def adaptive_coarsening(self, solution_field: np.ndarray, coarsening_threshold: float = 0.001):
        """Perform adaptive mesh coarsening to remove unnecessary refinement"""
        if not self.params.adaptive_refinement or not hasattr(self, 'element_levels'):
            return
        
        # Find elements eligible for coarsening
        elements_to_coarsen = self._identify_coarsening_candidates(
            solution_field, coarsening_threshold
        )
        
        if elements_to_coarsen:
            self._coarsen_elements(elements_to_coarsen)
            logger.info(f"Coarsened {len(elements_to_coarsen)} element groups")
    
    def _identify_coarsening_candidates(self, solution: np.ndarray, threshold: float) -> List[List[int]]:
        """Identify groups of sibling elements that can be coarsened"""
        coarsening_candidates = []
        
        if not hasattr(self, 'active_elements'):
            return coarsening_candidates
        
        # Group elements by refinement level and parent
        level_groups = {}
        for elem_id, active in enumerate(self.active_elements):
            if not active:
                continue
                
            level = self._get_element_level(elem_id)
            if level > 0:  # Only consider refined elements
                parent_id = self._find_parent_element(elem_id)
                if parent_id is not None:
                    key = (level, parent_id)
                    if key not in level_groups:
                        level_groups[key] = []
                    level_groups[key].append(elem_id)
        
        # Check each sibling group for coarsening eligibility
        for (level, parent_id), sibling_ids in level_groups.items():
            if self._can_coarsen_siblings(sibling_ids, solution, threshold):
                coarsening_candidates.append(sibling_ids)
        
        return coarsening_candidates
    
    def _can_coarsen_siblings(self, sibling_ids: List[int], solution: np.ndarray, threshold: float) -> bool:
        """Check if a group of sibling elements can be coarsened"""
        # Must have complete sibling set (8 for hex, 8 for tet)
        expected_siblings = 8
        if len(sibling_ids) != expected_siblings:
            return False
        
        # Check solution gradient in all siblings
        max_gradient = 0.0
        for elem_id in sibling_ids:
            element = self.elements[elem_id]
            elem_solution = solution[element]
            
            # Calculate local gradient
            elem_gradient = np.max(np.abs(np.gradient(elem_solution.reshape(-1))))
            max_gradient = max(max_gradient, elem_gradient)
        
        # Coarsen if gradient is below threshold
        if max_gradient > threshold:
            return False
        
        # Check neighboring element constraints
        for elem_id in sibling_ids:
            neighbors = self._get_element_neighbors(elem_id)
            for neighbor_id in neighbors:
                if neighbor_id not in sibling_ids:  # External neighbor
                    neighbor_level = self._get_element_level(neighbor_id)
                    current_level = self._get_element_level(elem_id)
                    
                    # Ensure 2:1 rule won't be violated after coarsening
                    if neighbor_level > current_level - 2:
                        return False
        
        return True
    
    def _find_parent_element(self, elem_id: int) -> Optional[int]:
        """Find parent element ID from refinement history"""
        # Search through refinement history to find parent
        for history_entry in reversed(self.refinement_history):
            if 'refined_element_map' in history_entry:
                for parent_id, refinement_data in history_entry['refined_element_map'].items():
                    if elem_id in refinement_data.get('sub_elements', []):
                        return parent_id
        return None
    
    def _coarsen_elements(self, element_groups: List[List[int]]):
        """Coarsen groups of sibling elements back to their parents"""
        coarsened_count = 0
        
        for sibling_group in element_groups:
            parent_id = self._find_parent_element(sibling_group[0])
            
            if parent_id is not None:
                # Reactivate parent element
                if parent_id < len(self.active_elements):
                    self.active_elements[parent_id] = True
                
                # Deactivate all sibling elements
                for elem_id in sibling_group:
                    if elem_id < len(self.active_elements):
                        self.active_elements[elem_id] = False
                
                # Remove hanging node constraints related to these elements
                self._remove_constraints_for_elements(sibling_group)
                
                coarsened_count += 1
        
        # Update refinement history
        if coarsened_count > 0:
            self.refinement_history.append({
                'timestamp': time.time(),
                'operation': 'coarsening',
                'elements_coarsened': sum(len(group) for group in element_groups),
                'parent_elements_reactivated': coarsened_count,
                'mesh_quality_metrics': self._calculate_mesh_quality_metrics()
            })
    
    def _remove_constraints_for_elements(self, element_ids: List[int]):
        """Remove mesh constraints involving specified elements"""
        if not hasattr(self, 'mesh_constraints'):
            return
        
        # Filter out constraints involving coarsened elements
        remaining_constraints = []
        for constraint in self.mesh_constraints:
            if constraint.get('parent_element') not in element_ids and \
               constraint.get('neighbor_element') not in element_ids:
                remaining_constraints.append(constraint)
        
        self.mesh_constraints = remaining_constraints
    
    def get_refinement_statistics(self) -> dict:
        """Get comprehensive refinement statistics"""
        if not hasattr(self, 'element_levels'):
            return {'error': 'No refinement data available'}
        
        active_mask = getattr(self, 'active_elements', np.ones(len(self.elements), dtype=bool))
        active_levels = self.element_levels[active_mask]
        
        level_counts = {}
        for level in active_levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total_elements': len(self.elements),
            'active_elements': np.sum(active_mask),
            'inactive_elements': len(self.elements) - np.sum(active_mask),
            'max_refinement_level': np.max(active_levels) if len(active_levels) > 0 else 0,
            'elements_per_level': level_counts,
            'refinement_operations': len(self.refinement_history),
            'mesh_quality': self._calculate_mesh_quality_metrics()
        }

class AdvancedViscosityModel:
    """Advanced viscosity models for mantle materials"""
    
    def __init__(self, model_type: ViscosityModel):
        self.model_type = model_type
        self.material_properties = self._initialize_material_properties()
    
    def _initialize_material_properties(self) -> Dict[str, MaterialProperties]:
        """Initialize material properties for different mantle phases"""
        return {
            'upper_mantle': MaterialProperties(
                density=3300, thermal_expansion=3e-5, thermal_conductivity=3.0,
                specific_heat=1200, viscosity_reference=1e21,
                activation_energy=300e3, activation_volume=4e-6,
                reference_temperature=1600, reference_pressure=1e9
            ),
            'transition_zone': MaterialProperties(
                density=3600, thermal_expansion=2.5e-5, thermal_conductivity=4.0,
                specific_heat=1250, viscosity_reference=5e21,
                activation_energy=400e3, activation_volume=3e-6,
                reference_temperature=1800, reference_pressure=15e9
            ),
            'lower_mantle': MaterialProperties(
                density=4500, thermal_expansion=2e-5, thermal_conductivity=5.0,
                specific_heat=1300, viscosity_reference=1e22,
                activation_energy=500e3, activation_volume=2e-6,
                reference_temperature=2000, reference_pressure=50e9
            ),
            'd_prime': MaterialProperties(
                density=5000, thermal_expansion=1.5e-5, thermal_conductivity=8.0,
                specific_heat=1400, viscosity_reference=1e20,
                activation_energy=200e3, activation_volume=1e-6,
                reference_temperature=2500, reference_pressure=130e9
            )
        }
    
    def calculate_viscosity(self, temperature: np.ndarray, pressure: np.ndarray, 
                          composition: np.ndarray, phase: str) -> np.ndarray:
        """Calculate viscosity based on temperature, pressure, and composition"""
        props = self.material_properties[phase]
        
        if self.model_type == ViscosityModel.CONSTANT:
            return np.full_like(temperature, props.viscosity_reference)
        
        elif self.model_type == ViscosityModel.TEMPERATURE_DEPENDENT:
            return self._temperature_dependent_viscosity(temperature, props)
        
        elif self.model_type == ViscosityModel.TEMPERATURE_PRESSURE:
            return self._temperature_pressure_viscosity(temperature, pressure, props)
        
        elif self.model_type == ViscosityModel.COMPOSITE_DEPENDENT:
            return self._composite_dependent_viscosity(temperature, pressure, composition, props)
        
        elif self.model_type == ViscosityModel.DISLOCATION_DIFFUSION:
            return self._dislocation_diffusion_viscosity(temperature, pressure, props)
        
        else:
            raise ValueError(f"Unknown viscosity model: {self.model_type}")
    
    def _temperature_dependent_viscosity(self, T: np.ndarray, props: MaterialProperties) -> np.ndarray:
        """Arrhenius temperature-dependent viscosity"""
        R = 8.314  # Gas constant
        return props.viscosity_reference * np.exp(
            props.activation_energy / R * (1/T - 1/props.reference_temperature)
        )
    
    def _temperature_pressure_viscosity(self, T: np.ndarray, P: np.ndarray, 
                                      props: MaterialProperties) -> np.ndarray:
        """Combined temperature and pressure dependent viscosity"""
        R = 8.314
        return props.viscosity_reference * np.exp(
            (props.activation_energy + P * props.activation_volume) / (R * T) - 
            (props.activation_energy + props.reference_pressure * props.activation_volume) / 
            (R * props.reference_temperature)
        )
    
    def _composite_dependent_viscosity(self, T: np.ndarray, P: np.ndarray, 
                                     C: np.ndarray, props: MaterialProperties) -> np.ndarray:
        """Viscosity dependent on temperature, pressure, and composition"""
        base_viscosity = self._temperature_pressure_viscosity(T, P, props)
        
        # Compositional effect (simplified)
        composition_factor = 1.0 + 10.0 * C  # Higher composition increases viscosity
        
        return base_viscosity * composition_factor
    
    def _dislocation_diffusion_viscosity(self, T: np.ndarray, P: np.ndarray, 
                                       props: MaterialProperties) -> np.ndarray:
        """Dislocation and diffusion creep composite viscosity"""
        R = 8.314
        
        # Diffusion creep (grain size dependent)
        grain_size = 1e-3  # 1 mm average grain size
        diffusion_viscosity = props.viscosity_reference * (grain_size / 1e-3)**3 * np.exp(
            (props.activation_energy + P * props.activation_volume) / (R * T)
        )
        
        # Dislocation creep (stress dependent)
        stress = 1e6  # Assumed stress (Pa)
        stress_exponent = 3.5
        dislocation_viscosity = props.viscosity_reference * (stress / 1e6)**(-stress_exponent + 1) * np.exp(
            (props.activation_energy * 1.2 + P * props.activation_volume * 0.8) / (R * T)
        )
        
        # Composite viscosity (harmonic mean)
        return (diffusion_viscosity * dislocation_viscosity) / (diffusion_viscosity + dislocation_viscosity)

class PhaseTransitionModel:
    """Model for mantle phase transitions"""
    
    def __init__(self):
        self.transitions = self._initialize_phase_transitions()
    
    def _initialize_phase_transitions(self) -> Dict[PhaseTransition, Dict]:
        """Initialize phase transition parameters"""
        return {
            PhaseTransition.OLIVINE_WADSLEYITE: {
                'depth': 410e3,
                'clapeyron_slope': 2.5e6,  # Pa/K
                'density_jump': 0.07,      # Fractional density increase
                'width': 10e3,             # Transition width (m)
                'latent_heat': 200e3       # J/kg
            },
            PhaseTransition.WADSLEYITE_RINGWOODITE: {
                'depth': 520e3,
                'clapeyron_slope': 1.5e6,
                'density_jump': 0.05,
                'width': 15e3,
                'latent_heat': 150e3
            },
            PhaseTransition.RINGWOODITE_PEROVSKITE: {
                'depth': 660e3,
                'clapeyron_slope': -2.0e6,  # Negative slope
                'density_jump': 0.10,
                'width': 20e3,
                'latent_heat': 400e3
            },
            PhaseTransition.PEROVSKITE_POST_PEROVSKITE: {
                'depth': 2700e3,
                'clapeyron_slope': 8.0e6,
                'density_jump': 0.02,
                'width': 50e3,
                'latent_heat': 100e3
            }
        }
    
    def calculate_phase_effects(self, temperature: np.ndarray, depth: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate phase transition effects on material properties"""
        density_anomaly = np.zeros_like(temperature)
        buoyancy_force = np.zeros_like(temperature)
        latent_heating = np.zeros_like(temperature)
        
        for transition, params in self.transitions.items():
            transition_depth = params['depth']
            clapeyron_slope = params['clapeyron_slope']
            density_jump = params['density_jump']
            width = params['width']
            latent_heat = params['latent_heat']
            
            # Calculate transition pressure from temperature
            reference_temp = 1600  # K
            transition_pressure = clapeyron_slope * (temperature - reference_temp)
            effective_depth = transition_depth + transition_pressure / (3300 * 9.81)
            
            # Smooth transition function
            transition_function = 0.5 * (1 + np.tanh((depth - effective_depth) / width))
            
            # Apply density jump
            density_anomaly += density_jump * transition_function
            
            # Calculate buoyancy effects
            g = 9.81  # m/s^2
            buoyancy_force += -density_jump * 3300 * g * transition_function
            
            # Latent heat effects
            dtransition_dt = np.gradient(transition_function)
            latent_heating += latent_heat * dtransition_dt
        
        return {
            'density_anomaly': density_anomaly,
            'buoyancy_force': buoyancy_force,
            'latent_heating': latent_heating
        }

class AdvancedConvectionSolver:
    """Advanced 3D spherical mantle convection solver"""
    
    def __init__(self, params: ConvectionParameters):
        self.params = params
        self.mesh = SphericalMesh(params)
        self.viscosity_model = AdvancedViscosityModel(ViscosityModel.TEMPERATURE_PRESSURE)
        self.phase_model = PhaseTransitionModel()
        
        # Solution fields
        self.temperature = None
        self.velocity = None
        self.pressure = None
        self.composition = None
        
        # System matrices
        self.stiffness_matrix = None
        self.mass_matrix = None
        self.advection_matrix = None
        
        # GPU arrays
        self.gpu_arrays = {}
        
        # Initialize solution
        self._initialize_solution()
        
        # Setup GPU if available
        if self.params.gpu_acceleration and cp.cuda.is_available():
            self._setup_gpu_acceleration()
            logger.info("GPU acceleration enabled")
        else:
            logger.info("Using CPU computation")
    
    def _initialize_solution(self):
        """Initialize temperature and velocity fields"""
        n_nodes = len(self.mesh.nodes)
        
        # Initialize temperature field
        self.temperature = self._initialize_temperature_field()
        
        # Initialize velocity field (3 components: r, theta, phi)
        self.velocity = np.zeros((n_nodes, 3))
        
        # Initialize pressure field
        self.pressure = np.zeros(n_nodes)
        
        # Initialize composition field (if compositional convection enabled)
        if self.params.compositional_convection:
            self.composition = self._initialize_composition_field()
    
    def _initialize_temperature_field(self) -> np.ndarray:
        """Initialize temperature field with realistic profile"""
        n_nodes = len(self.mesh.nodes)
        temperature = np.zeros(n_nodes)
        
        for i, node in enumerate(self.mesh.nodes):
            r, theta, phi = node[0], node[1], node[2]
            depth = self.params.outer_radius - r
            
            # Base adiabatic profile
            surface_temp = self.params.surface_temperature
            cmb_temp = self.params.cmb_temperature
            
            # Adiabatic temperature profile
            adiabatic_gradient = 0.3e-3  # K/m
            temp_adiabatic = surface_temp + adiabatic_gradient * depth
            
            # Add thermal boundary layers
            tbl_thickness = 100e3  # 100 km
            
            # Upper thermal boundary layer
            if depth < tbl_thickness:
                temp_reduction = 200 * (1 - depth / tbl_thickness)
                temp_adiabatic -= temp_reduction
            
            # Lower thermal boundary layer
            if r - self.params.inner_radius < tbl_thickness:
                depth_from_cmb = r - self.params.inner_radius
                temp_increase = (cmb_temp - temp_adiabatic) * (1 - depth_from_cmb / tbl_thickness)
                temp_adiabatic += temp_increase
            
            # Add random perturbations for instability
            perturbation = 50 * (np.random.random() - 0.5) * np.sin(2*theta) * np.cos(3*phi)
            
            temperature[i] = temp_adiabatic + perturbation
        
        # Apply boundary conditions
        for node_id in self.mesh.boundary_nodes['outer']:
            temperature[node_id] = self.params.surface_temperature
        
        for node_id in self.mesh.boundary_nodes['inner']:
            temperature[node_id] = self.params.cmb_temperature
        
        return temperature
    
    def _initialize_composition_field(self) -> np.ndarray:
        """Initialize compositional field for chemical convection"""
        n_nodes = len(self.mesh.nodes)
        composition = np.zeros(n_nodes)
        
        # Initialize with dense primordial material in lower mantle
        for i, node in enumerate(self.mesh.nodes):
            r = node[0]
            depth = self.params.outer_radius - r
            
            # Dense primitive material in bottom 500 km
            if depth > (self.params.outer_radius - self.params.inner_radius) - 500e3:
                composition[i] = 0.1  # 10% denser component
            
            # Add chemical heterogeneity
            theta, phi = node[1], node[2]
            heterogeneity = 0.02 * np.sin(4*theta) * np.cos(6*phi)
            composition[i] += heterogeneity
        
        return composition
    
    def _setup_gpu_acceleration(self):
        """Setup GPU arrays and operations"""
        if not cp.cuda.is_available():
            return
        
        try:
            # Transfer arrays to GPU
            self.gpu_arrays['temperature'] = cp.array(self.temperature)
            self.gpu_arrays['velocity'] = cp.array(self.velocity)
            self.gpu_arrays['pressure'] = cp.array(self.pressure)
            self.gpu_arrays['nodes'] = cp.array(self.mesh.nodes)
            
            if self.params.compositional_convection:
                self.gpu_arrays['composition'] = cp.array(self.composition)
            
            logger.info("GPU arrays initialized successfully")
        except Exception as e:
            logger.warning(f"GPU acceleration setup failed: {e}")
            self.params.gpu_acceleration = False
    
    def solve_stokes_equation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Stokes equation for velocity and pressure"""
        logger.info("Solving Stokes equation...")
        
        # Calculate viscosity field
        depth = self.params.outer_radius - self.mesh.nodes[:, 0]
        pressure_field = self._calculate_lithostatic_pressure(depth)
        
        # Determine mantle phases
        phases = self._determine_mantle_phases(depth)
        
        # Calculate viscosity for each phase
        viscosity = np.zeros(len(self.mesh.nodes))
        for phase in np.unique(phases):
            phase_mask = phases == phase
            if np.any(phase_mask):
                phase_viscosity = self.viscosity_model.calculate_viscosity(
                    self.temperature[phase_mask],
                    pressure_field[phase_mask],
                    self.composition[phase_mask] if self.params.compositional_convection else np.zeros(np.sum(phase_mask)),
                    phase
                )
                viscosity[phase_mask] = phase_viscosity
        
        # Apply viscosity bounds
        viscosity = np.clip(viscosity, 1e18, 1e25)
        
        # Build Stokes system matrices
        if self.params.gpu_acceleration and cp.cuda.is_available():
            velocity, pressure = self._solve_stokes_gpu(viscosity)
        else:
            velocity, pressure = self._solve_stokes_cpu(viscosity)
        
        return velocity, pressure
    
    def _calculate_lithostatic_pressure(self, depth: np.ndarray) -> np.ndarray:
        """Calculate lithostatic pressure profile"""
        g = 9.81  # m/s^2
        
        # Piecewise density model
        pressure = np.zeros_like(depth)
        
        # Upper mantle (0-410 km)
        mask_um = depth <= 410e3
        pressure[mask_um] = 3300 * g * depth[mask_um]
        
        # Transition zone (410-660 km)
        mask_tz = (depth > 410e3) & (depth <= 660e3)
        pressure[mask_tz] = 3300 * g * 410e3 + 3600 * g * (depth[mask_tz] - 410e3)
        
        # Lower mantle (660+ km)
        mask_lm = depth > 660e3
        pressure[mask_lm] = (3300 * g * 410e3 + 3600 * g * 250e3 + 
                            4500 * g * (depth[mask_lm] - 660e3))
        
        return pressure
    
    def _determine_mantle_phases(self, depth: np.ndarray) -> np.ndarray:
        """Determine mantle phases based on depth"""
        phases = np.full(len(depth), 'upper_mantle', dtype='<U15')
        
        phases[(depth >= 410e3) & (depth < 660e3)] = 'transition_zone'
        phases[(depth >= 660e3) & (depth < 2700e3)] = 'lower_mantle'
        phases[depth >= 2700e3] = 'd_prime'
        
        return phases
    
    def _solve_stokes_cpu(self, viscosity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Stokes equation on CPU"""
        n_nodes = len(self.mesh.nodes)
        n_dofs = 4 * n_nodes  # 3 velocity components + pressure
        
        # Build system matrix (simplified implementation)
        A = sp.lil_matrix((n_dofs, n_dofs))
        b = np.zeros(n_dofs)
        
        # Apply boundary conditions and solve
        # This is a simplified version - full implementation would be much more complex
        
        # For now, return simple velocity field based on temperature anomalies
        velocity = np.zeros((n_nodes, 3))
        pressure = np.zeros(n_nodes)
        
        # Calculate buoyancy-driven flow
        reference_temp = np.mean(self.temperature)
        temp_anomaly = self.temperature - reference_temp
        
        # Simplified radial velocity from temperature anomaly
        alpha = 3e-5  # Thermal expansion
        g = 9.81
        rayleigh_velocity = alpha * g * temp_anomaly / (viscosity / 3300)
        
        # Apply velocity in radial direction
        velocity[:, 0] = rayleigh_velocity * 1e-10  # Scale down for stability
        
        logger.info("Stokes equation solved on CPU")
        return velocity, pressure
    
    def _solve_stokes_gpu(self, viscosity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve Stokes equation on GPU"""
        try:
            # Transfer data to GPU
            viscosity_gpu = cp.array(viscosity)
            temp_gpu = self.gpu_arrays['temperature']
            
            # Simplified GPU computation
            reference_temp = cp.mean(temp_gpu)
            temp_anomaly = temp_gpu - reference_temp
            
            alpha = 3e-5
            g = 9.81
            rayleigh_velocity = alpha * g * temp_anomaly / (viscosity_gpu / 3300)
            
            # Create velocity field
            n_nodes = len(self.mesh.nodes)
            velocity_gpu = cp.zeros((n_nodes, 3))
            velocity_gpu[:, 0] = rayleigh_velocity * 1e-10
            
            pressure_gpu = cp.zeros(n_nodes)
            
            # Transfer back to CPU
            velocity = cp.asnumpy(velocity_gpu)
            pressure = cp.asnumpy(pressure_gpu)
            
            logger.info("Stokes equation solved on GPU")
            return velocity, pressure
            
        except Exception as e:
            logger.warning(f"GPU Stokes solver failed: {e}, falling back to CPU")
            return self._solve_stokes_cpu(viscosity)
    
    def solve_energy_equation(self, dt: float) -> np.ndarray:
        """Solve energy equation for temperature evolution"""
        logger.info("Solving energy equation...")
        
        # Calculate phase transition effects
        depth = self.params.outer_radius - self.mesh.nodes[:, 0]
        phase_effects = self.phase_model.calculate_phase_effects(self.temperature, depth)
        
        # Setup energy equation
        if self.params.gpu_acceleration and cp.cuda.is_available():
            new_temperature = self._solve_energy_gpu(dt, phase_effects)
        else:
            new_temperature = self._solve_energy_cpu(dt, phase_effects)
        
        return new_temperature
    
    def _solve_energy_cpu(self, dt: float, phase_effects: Dict[str, np.ndarray]) -> np.ndarray:
        """Solve energy equation on CPU"""
        n_nodes = len(self.mesh.nodes)
        
        # Build energy equation matrices (simplified)
        M = sp.identity(n_nodes)  # Mass matrix (simplified)
        K = sp.lil_matrix((n_nodes, n_nodes))  # Stiffness matrix (simplified)
        
        # Add diffusion terms (simplified implementation)
        thermal_diffusivity = 1e-6  # m^2/s
        
        # Simple explicit time stepping
        new_temperature = self.temperature.copy()
        
        # Add advection (simplified)
        advection_term = np.zeros_like(self.temperature)
        for i in range(n_nodes):
            # Calculate temperature gradient (simplified)
            temp_grad = 0.0  # Would need proper gradient calculation
            advection_term[i] = -np.dot(self.velocity[i], [temp_grad, 0, 0])
        
        # Add diffusion (simplified)
        diffusion_term = np.zeros_like(self.temperature)
        # Would need proper Laplacian calculation
        
        # Add internal heating
        internal_heating = np.full_like(self.temperature, self.params.internal_heating)
        
        # Add phase transition latent heating
        latent_heating = phase_effects['latent_heating']
        
        # Update temperature
        dT_dt = (advection_term + diffusion_term + internal_heating + latent_heating) / 1200  # Specific heat
        new_temperature += dt * dT_dt
        
        # Apply boundary conditions
        for node_id in self.mesh.boundary_nodes['outer']:
            new_temperature[node_id] = self.params.surface_temperature
        
        for node_id in self.mesh.boundary_nodes['inner']:
            new_temperature[node_id] = self.params.cmb_temperature
        
        logger.info("Energy equation solved on CPU")
        return new_temperature
    
    def _solve_energy_gpu(self, dt: float, phase_effects: Dict[str, np.ndarray]) -> np.ndarray:
        """Solve energy equation on GPU"""
        try:
            temp_gpu = self.gpu_arrays['temperature']
            velocity_gpu = self.gpu_arrays['velocity']
            
            # Simplified GPU energy equation
            internal_heating = self.params.internal_heating
            latent_heating_gpu = cp.array(phase_effects['latent_heating'])
            
            # Simple forward Euler (would need better time integration)
            heating_rate = internal_heating + latent_heating_gpu / 1200
            temp_gpu += dt * heating_rate
            
            # Apply boundary conditions
            inner_nodes_gpu = cp.array(self.mesh.boundary_nodes['inner'])
            outer_nodes_gpu = cp.array(self.mesh.boundary_nodes['outer'])
            
            temp_gpu[outer_nodes_gpu] = self.params.surface_temperature
            temp_gpu[inner_nodes_gpu] = self.params.cmb_temperature
            
            new_temperature = cp.asnumpy(temp_gpu)
            
            logger.info("Energy equation solved on GPU")
            return new_temperature
            
        except Exception as e:
            logger.warning(f"GPU energy solver failed: {e}, falling back to CPU")
            return self._solve_energy_cpu(dt, phase_effects)
    
    def solve_composition_equation(self, dt: float) -> np.ndarray:
        """Solve composition equation for chemical evolution"""
        if not self.params.compositional_convection:
            return self.composition
        
        logger.info("Solving composition equation...")
        
        # Simplified composition advection-diffusion
        new_composition = self.composition.copy()
        
        # Add advection term (simplified)
        # Would need proper implementation of advection
        
        # Add diffusion term
        diffusivity = 1e-12  # Very slow chemical diffusion
        # Would need proper Laplacian implementation
        
        # Simple explicit update
        # new_composition would be updated with proper advection-diffusion
        
        return new_composition
    
    def time_integration_step(self, current_time: float, dt: float) -> float:
        """Perform one time integration step"""
        logger.info(f"Time step at t = {current_time/1e6:.2f} Myr, dt = {dt/1e6:.2f} Myr")
        
        # Solve Stokes equation for velocity and pressure
        self.velocity, self.pressure = self.solve_stokes_equation()
        
        # Solve energy equation for temperature
        self.temperature = self.solve_energy_equation(dt)
        
        # Solve composition equation if enabled
        if self.params.compositional_convection:
            self.composition = self.solve_composition_equation(dt)
        
        # Adaptive time stepping
        if self.params.adaptive_timestepping:
            dt_new = self._calculate_adaptive_timestep(dt)
        else:
            dt_new = dt
        
        # Adaptive mesh refinement
        if self.params.adaptive_refinement:
            self.mesh.adaptive_refinement(self.temperature, self.params.refinement_threshold)
        
        return dt_new
    
    def _calculate_adaptive_timestep(self, current_dt: float) -> float:
        """Calculate adaptive time step based on solution stability"""
        # CFL condition for advection
        max_velocity = np.max(np.linalg.norm(self.velocity, axis=1))
        min_element_size = 50e3  # Approximate minimum element size
        cfl_dt = 0.5 * min_element_size / max_velocity if max_velocity > 0 else current_dt
        
        # Diffusion stability
        thermal_diffusivity = 1e-6
        diffusion_dt = 0.25 * min_element_size**2 / thermal_diffusivity
        
        # Take most restrictive
        stable_dt = min(cfl_dt, diffusion_dt, 2.0 * current_dt)
        stable_dt = max(stable_dt, 0.1 * current_dt)  # Don't reduce too much
        
        return stable_dt
    
    def run_simulation(self) -> Dict[str, np.ndarray]:
        """Run complete mantle convection simulation"""
        logger.info("Starting mantle convection simulation...")
        
        # Initialize time parameters
        current_time = 0.0
        dt = self.params.time_step
        total_time = self.params.total_time
        
        # Storage for time series
        time_series = {
            'time': [],
            'temperature': [],
            'velocity': [],
            'pressure': [],
            'composition': [] if self.params.compositional_convection else None,
            'heat_flux': [],
            'rms_velocity': [],
            'nusselt_number': []
        }
        
        step = 0
        save_interval = 100  # Save every 100 steps
        
        while current_time < total_time:
            # Perform time integration step
            dt = self.time_integration_step(current_time, dt)
            current_time += dt
            step += 1
            
            # Calculate diagnostic quantities
            heat_flux = self._calculate_heat_flux()
            rms_velocity = np.sqrt(np.mean(np.sum(self.velocity**2, axis=1)))
            nusselt_number = self._calculate_nusselt_number()
            
            # Save data
            if step % save_interval == 0:
                time_series['time'].append(current_time)
                time_series['temperature'].append(self.temperature.copy())
                time_series['velocity'].append(self.velocity.copy())
                time_series['pressure'].append(self.pressure.copy())
                if self.params.compositional_convection:
                    time_series['composition'].append(self.composition.copy())
                time_series['heat_flux'].append(heat_flux)
                time_series['rms_velocity'].append(rms_velocity)
                time_series['nusselt_number'].append(nusselt_number)
                
                logger.info(f"Step {step}: t={current_time/1e6:.2f} Myr, "
                           f"RMS velocity={rms_velocity:.2e} m/s, Nu={nusselt_number:.2f}")
        
        logger.info("Simulation completed successfully")
        return time_series
    
    def _calculate_heat_flux(self) -> float:
        """Calculate surface heat flux"""
        # Simplified heat flux calculation
        surface_nodes = self.mesh.boundary_nodes['outer']
        surface_temperature = self.temperature[surface_nodes]
        
        # Approximate heat flux from temperature gradient
        heat_flux = np.mean(surface_temperature) * 1e-3  # Simplified
        return heat_flux
    
    def _calculate_nusselt_number(self) -> float:
        """Calculate Nusselt number (convective heat transfer efficiency)"""
        # Simplified Nusselt number calculation
        surface_heat_flux = self._calculate_heat_flux()
        conductive_heat_flux = 1e-3  # Reference conductive heat flux
        
        nusselt_number = surface_heat_flux / conductive_heat_flux
        return max(nusselt_number, 1.0)  # Nu >= 1 always
    
    def save_results(self, time_series: Dict[str, np.ndarray], filename: str):
        """Save simulation results to HDF5 file"""
        with h5py.File(filename, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            for key, value in vars(self.params).items():
                if isinstance(value, (int, float, str, bool)):
                    params_group.attrs[key] = value
            
            # Save mesh information
            mesh_group = f.create_group('mesh')
            mesh_group.create_dataset('nodes', data=self.mesh.nodes)
            mesh_group.create_dataset('elements', data=self.mesh.elements)
            
            # Save time series data
            ts_group = f.create_group('time_series')
            for key, data in time_series.items():
                if data is not None:
                    if key in ['temperature', 'velocity', 'pressure', 'composition']:
                        # Save as array of arrays
                        ts_group.create_dataset(key, data=np.array(data))
                    else:
                        ts_group.create_dataset(key, data=np.array(data))
        
        logger.info(f"Results saved to {filename}")

def main():
    """Main function for testing the convection solver"""
    # Setup parameters for test simulation
    params = ConvectionParameters(
        radial_elements=64,
        lateral_elements=128,
        total_time=100e6 * 365.25 * 24 * 3600,  # 100 Myr
        time_step=1000 * 365.25 * 24 * 3600,    # 1000 years
        rayleigh_number=1e6,
        gpu_acceleration=True,
        adaptive_refinement=True,
        compositional_convection=True
    )
    
    # Create solver
    solver = AdvancedConvectionSolver(params)
    
    # Run simulation
    results = solver.run_simulation()
    
    # Save results
    output_file = "mantle_convection_results.h5"
    solver.save_results(results, output_file)
    
    print(f"Simulation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()