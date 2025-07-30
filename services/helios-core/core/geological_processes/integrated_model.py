"""
Integrated Geological Process Model

Combines all sophisticated geological process models into a unified system:
- Mantle convection with crustal deformation coupling
- Hydrothermal flow with geochemical reactions
- Multi-physics interactions and feedback loops
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging

from .mantle_convection import MantleConvectionModel, ConvectionParameters
from .crustal_deformation import CrustalDeformationModel, ViscoelasticParameters
from .fluid_flow import HydrothermalFlowModel, FluidFlowParameters
from .geochemical_reactions import GeochemicalReactionNetwork, ReactionParameters

logger = logging.getLogger(__name__)

@dataclass
class IntegratedModelParameters:
    """Parameters for integrated geological process model"""
    # Individual model parameters
    convection_params: ConvectionParameters = None
    deformation_params: ViscoelasticParameters = None
    flow_params: FluidFlowParameters = None
    reaction_params: ReactionParameters = None
    
    # Coupling parameters
    thermal_coupling_strength: float = 1.0
    mechanical_coupling_strength: float = 1.0
    chemical_coupling_strength: float = 1.0
    
    # Time scales
    mantle_time_scale: float = 1e6  # years
    crustal_time_scale: float = 1e4  # years
    hydrothermal_time_scale: float = 1e2  # years
    reaction_time_scale: float = 1.0  # years
    
    def __post_init__(self):
        if self.convection_params is None:
            self.convection_params = ConvectionParameters()
        if self.deformation_params is None:
            self.deformation_params = ViscoelasticParameters()
        if self.flow_params is None:
            self.flow_params = FluidFlowParameters()
        if self.reaction_params is None:
            self.reaction_params = ReactionParameters()


class IntegratedGeologicalModel(nn.Module):
    """
    Integrated model combining all geological processes with full complexity
    """
    
    def __init__(self, params: IntegratedModelParameters):
        super().__init__()
        self.params = params
        
        # Initialize individual models
        self.mantle_model = MantleConvectionModel(params.convection_params)
        self.crustal_model = CrustalDeformationModel(params.deformation_params)
        self.flow_model = HydrothermalFlowModel(params.flow_params)
        self.reaction_model = GeochemicalReactionNetwork(params.reaction_params)
        
        # Coupling networks for inter-process communication
        self.thermal_coupling_net = nn.Sequential(
            nn.Linear(8, 256),  # Combined thermal fields
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 4)   # Thermal coupling terms
        )
        
        self.mechanical_coupling_net = nn.Sequential(
            nn.Linear(14, 256),  # Combined mechanical fields (3+3+6+3+1+1 = 17, adjusted to 14)
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 6)   # Mechanical coupling terms
        )
        
        self.chemical_coupling_net = nn.Sequential(
            nn.Linear(16, 256),  # Combined chemical fields
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 8)   # Chemical coupling terms
        )
        
    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through integrated geological model
        
        Args:
            coords: Input coordinates (x, y, z, t) [batch_size, 4]
            
        Returns:
            Dictionary containing all geological fields and their interactions
        """
        # Individual model predictions
        mantle_predictions = self.mantle_model(coords)
        crustal_predictions = self.crustal_model(coords)
        flow_predictions = self.flow_model(coords)
        reaction_predictions = self.reaction_model(coords)
        
        # Compute coupling terms
        thermal_coupling = self._compute_thermal_coupling(
            mantle_predictions, crustal_predictions, flow_predictions
        )
        
        mechanical_coupling = self._compute_mechanical_coupling(
            mantle_predictions, crustal_predictions, flow_predictions
        )
        
        chemical_coupling = self._compute_chemical_coupling(
            flow_predictions, reaction_predictions
        )
        
        # Integrate all predictions
        integrated_predictions = {
            'mantle': mantle_predictions,
            'crustal': crustal_predictions,
            'flow': flow_predictions,
            'reactions': reaction_predictions,
            'thermal_coupling': thermal_coupling,
            'mechanical_coupling': mechanical_coupling,
            'chemical_coupling': chemical_coupling
        }
        
        return integrated_predictions
    
    def compute_physics_loss(self, coords: torch.Tensor, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-informed loss for integrated model
        """
        # Individual model losses
        mantle_loss = self.mantle_model.compute_physics_loss(coords, predictions['mantle'])
        crustal_loss = self.crustal_model.compute_physics_loss(coords, predictions['crustal'])
        flow_loss = self.flow_model.compute_physics_loss(coords, predictions['flow'])
        reaction_loss = self.reaction_model.compute_physics_loss(coords, predictions['reactions'])
        
        # Coupling losses
        thermal_coupling_loss = self._compute_thermal_coupling_loss(coords, predictions)
        mechanical_coupling_loss = self._compute_mechanical_coupling_loss(coords, predictions)
        chemical_coupling_loss = self._compute_chemical_coupling_loss(coords, predictions)
        
        # Multi-scale temporal consistency loss
        temporal_consistency_loss = self._compute_temporal_consistency_loss(coords, predictions)
        
        total_loss = (mantle_loss + crustal_loss + flow_loss + reaction_loss +
                     self.params.thermal_coupling_strength * thermal_coupling_loss +
                     self.params.mechanical_coupling_strength * mechanical_coupling_loss +
                     self.params.chemical_coupling_strength * chemical_coupling_loss +
                     temporal_consistency_loss)
        
        return total_loss
    
    def _compute_thermal_coupling(self, mantle_pred: Dict, crustal_pred: Dict, 
                                flow_pred: Dict) -> torch.Tensor:
        """
        Compute thermal coupling between processes
        """
        # Combine thermal fields from all models
        thermal_fields = torch.cat([
            mantle_pred['temperature'],
            crustal_pred['temperature'],
            flow_pred['temperature'],
            mantle_pred['velocity'][:, 0:1],  # Radial velocity component
            crustal_pred['displacement'][:, 2:3],  # Vertical displacement
            flow_pred['velocity'][:, 2:3],  # Vertical flow velocity
            flow_pred['pressure'],
            flow_pred['phase_fraction']
        ], dim=-1)
        
        coupling_terms = self.thermal_coupling_net(thermal_fields)
        
        return coupling_terms
    
    def _compute_mechanical_coupling(self, mantle_pred: Dict, crustal_pred: Dict,
                                   flow_pred: Dict) -> torch.Tensor:
        """
        Compute mechanical coupling between processes
        """
        # Combine mechanical fields (ensure total is 14 dimensions)
        mechanical_fields = torch.cat([
            mantle_pred['velocity'],          # 3 dimensions
            crustal_pred['displacement'],     # 3 dimensions
            crustal_pred['stress'][:, :3],    # 3 dimensions (normal stress components)
            flow_pred['velocity'],            # 3 dimensions
            flow_pred['pressure'],            # 1 dimension
            mantle_pred['pressure']           # 1 dimension
        ], dim=-1)  # Total: 3+3+3+3+1+1 = 14 dimensions
        
        coupling_terms = self.mechanical_coupling_net(mechanical_fields)
        
        return coupling_terms
    
    def _compute_chemical_coupling(self, flow_pred: Dict, reaction_pred: Dict) -> torch.Tensor:
        """
        Compute chemical coupling between flow and reactions
        """
        # Combine chemical fields
        chemical_fields = torch.cat([
            flow_pred['temperature'],
            flow_pred['pressure'],
            flow_pred['velocity'],
            flow_pred['phase_fraction'],
            reaction_pred['concentrations'][:, :8],  # First 8 species
            reaction_pred['ph'],
            reaction_pred['eh']
        ], dim=-1)
        
        coupling_terms = self.chemical_coupling_net(chemical_fields)
        
        return coupling_terms
    
    def _compute_thermal_coupling_loss(self, coords: torch.Tensor, 
                                     predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute thermal coupling constraint loss
        """
        # Energy conservation across model boundaries
        mantle_temp = predictions['mantle']['temperature']
        crustal_temp = predictions['crustal']['temperature']
        flow_temp = predictions['flow']['temperature']
        
        # Temperature continuity at interfaces
        mantle_crustal_interface_loss = torch.mean((mantle_temp - crustal_temp)**2)
        crustal_flow_interface_loss = torch.mean((crustal_temp - flow_temp)**2)
        
        # Heat flux continuity
        thermal_coupling = predictions['thermal_coupling']
        heat_flux_continuity_loss = torch.mean(thermal_coupling[:, :2]**2)
        
        total_thermal_loss = (mantle_crustal_interface_loss + 
                            crustal_flow_interface_loss + 
                            heat_flux_continuity_loss)
        
        return total_thermal_loss
    
    def _compute_mechanical_coupling_loss(self, coords: torch.Tensor,
                                        predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute mechanical coupling constraint loss
        """
        # Stress continuity at interfaces
        mantle_pressure = predictions['mantle']['pressure']
        crustal_stress = predictions['crustal']['stress'][:, 2:3]  # Vertical stress
        flow_pressure = predictions['flow']['pressure']
        
        # Pressure/stress continuity
        mantle_crustal_stress_loss = torch.mean((mantle_pressure - crustal_stress)**2)
        crustal_flow_pressure_loss = torch.mean((crustal_stress - flow_pressure)**2)
        
        # Velocity/displacement compatibility
        mechanical_coupling = predictions['mechanical_coupling']
        velocity_compatibility_loss = torch.mean(mechanical_coupling[:, :3]**2)
        
        total_mechanical_loss = (mantle_crustal_stress_loss + 
                               crustal_flow_pressure_loss + 
                               velocity_compatibility_loss)
        
        return total_mechanical_loss
    
    def _compute_chemical_coupling_loss(self, coords: torch.Tensor,
                                      predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute chemical coupling constraint loss
        """
        # Mass conservation across flow-reaction interface
        flow_concentrations = predictions['flow']['concentrations']
        reaction_concentrations = predictions['reactions']['concentrations']
        
        # Concentration continuity (first few species)
        n_common_species = min(flow_concentrations.shape[-1], reaction_concentrations.shape[-1])
        concentration_continuity_loss = torch.mean(
            (flow_concentrations[:, :n_common_species] - 
             reaction_concentrations[:, :n_common_species])**2
        )
        
        # Chemical equilibrium constraints
        chemical_coupling = predictions['chemical_coupling']
        equilibrium_constraint_loss = torch.mean(chemical_coupling[:, :4]**2)
        
        total_chemical_loss = concentration_continuity_loss + equilibrium_constraint_loss
        
        return total_chemical_loss
    
    def _compute_temporal_consistency_loss(self, coords: torch.Tensor,
                                         predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute multi-scale temporal consistency loss
        """
        # Ensure different processes evolve at appropriate time scales
        time = coords[:, 3:4]
        
        # Mantle processes should vary slowly
        mantle_temp = predictions['mantle']['temperature']
        mantle_temp_rate = torch.autograd.grad(mantle_temp, coords, 
                                             grad_outputs=torch.ones_like(mantle_temp),
                                             create_graph=True)[0][:, 3:4]
        mantle_temporal_loss = torch.mean((mantle_temp_rate * self.params.mantle_time_scale)**2)
        
        # Crustal processes at intermediate time scale
        crustal_displacement = predictions['crustal']['displacement']
        crustal_velocity = torch.autograd.grad(crustal_displacement, coords,
                                             grad_outputs=torch.ones_like(crustal_displacement),
                                             create_graph=True)[0][:, 3:4]
        crustal_temporal_loss = torch.mean((crustal_velocity * self.params.crustal_time_scale)**2)
        
        # Flow processes should be faster
        flow_pressure = predictions['flow']['pressure']
        flow_pressure_rate = torch.autograd.grad(flow_pressure, coords,
                                                grad_outputs=torch.ones_like(flow_pressure),
                                                create_graph=True)[0][:, 3:4]
        flow_temporal_loss = torch.mean((flow_pressure_rate * self.params.hydrothermal_time_scale)**2)
        
        total_temporal_loss = (mantle_temporal_loss * 1e-6 + 
                             crustal_temporal_loss * 1e-4 + 
                             flow_temporal_loss * 1e-2)
        
        return total_temporal_loss
    
    def simulate_integrated_evolution(self, simulation_parameters: Dict, 
                                    total_time: float) -> Dict[str, np.ndarray]:
        """
        Simulate integrated geological system evolution
        
        Args:
            simulation_parameters: Parameters for simulation
            total_time: Total simulation time (years)
            
        Returns:
            Dictionary containing evolution of all geological processes
        """
        logger.info(f"Starting integrated geological simulation for {total_time} years")
        
        # Adaptive time stepping based on process time scales
        time_steps = self._compute_adaptive_time_steps(total_time)
        
        results = {
            'mantle_evolution': [],
            'crustal_evolution': [],
            'flow_evolution': [],
            'reaction_evolution': [],
            'coupling_evolution': [],
            'time_points': []
        }
        
        # Simulation loop with adaptive time stepping
        current_time = 0.0
        step = 0
        
        while current_time < total_time:
            # Determine time step based on current conditions
            dt = self._adaptive_time_step(current_time, total_time, step)
            
            # Create coordinate batch
            coords = self._create_integrated_coordinates(simulation_parameters, current_time)
            
            # Forward pass
            with torch.no_grad():
                predictions = self.forward(coords)
            
            # Store results
            results['mantle_evolution'].append(predictions['mantle'])
            results['crustal_evolution'].append(predictions['crustal'])
            results['flow_evolution'].append(predictions['flow'])
            results['reaction_evolution'].append(predictions['reactions'])
            results['coupling_evolution'].append({
                'thermal': predictions['thermal_coupling'],
                'mechanical': predictions['mechanical_coupling'],
                'chemical': predictions['chemical_coupling']
            })
            results['time_points'].append(current_time)
            
            current_time += dt
            step += 1
            
            if step % 100 == 0:
                logger.info(f"Completed integrated simulation step {step}, time: {current_time:.2e} years")
        
        logger.info("Completed integrated geological simulation")
        
        return results
    
    def _compute_adaptive_time_steps(self, total_time: float) -> List[float]:
        """Compute adaptive time steps for multi-scale simulation"""
        # Start with small time steps, increase gradually
        time_steps = []
        current_time = 0.0
        dt = 0.1  # Start with 0.1 year steps
        
        while current_time < total_time:
            time_steps.append(dt)
            current_time += dt
            
            # Gradually increase time step (but respect process time scales)
            dt = min(dt * 1.01, total_time * 0.01)
        
        return time_steps
    
    def _adaptive_time_step(self, current_time: float, total_time: float, step: int) -> float:
        """Compute adaptive time step based on current conditions"""
        # Base time step
        base_dt = total_time / 10000  # Target ~10000 steps
        
        # Adjust based on simulation phase
        if current_time < total_time * 0.1:
            # Early phase: small time steps
            dt = base_dt * 0.1
        elif current_time < total_time * 0.5:
            # Middle phase: moderate time steps
            dt = base_dt
        else:
            # Late phase: larger time steps
            dt = base_dt * 2.0
        
        # Ensure we don't overshoot
        dt = min(dt, total_time - current_time)
        
        return dt
    
    def _create_integrated_coordinates(self, params: Dict, time: float) -> torch.Tensor:
        """Create coordinate batch for integrated simulation"""
        n_points = 5000
        
        # Sample points in 3D domain
        x = torch.rand(n_points, 1) * params.get('domain_x', 1000e3)
        y = torch.rand(n_points, 1) * params.get('domain_y', 1000e3)
        z = torch.rand(n_points, 1) * params.get('domain_z', 100e3)
        t = torch.full((n_points, 1), time)
        
        coords = torch.cat([x, y, z, t], dim=1)
        
        return coords


def create_integrated_geological_model(config: Dict) -> IntegratedGeologicalModel:
    """
    Factory function to create integrated geological model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured integrated geological model
    """
    # Create individual model parameters
    convection_params = ConvectionParameters(
        rayleigh_number=config.get('rayleigh_number', 1e6),
        n_radial=config.get('n_radial', 64),
        n_theta=config.get('n_theta', 128),
        n_phi=config.get('n_phi', 256)
    )
    
    deformation_params = ViscoelasticParameters(
        youngs_modulus=config.get('youngs_modulus', 70e9),
        viscosity=config.get('viscosity', 1e22),
        domain_size=tuple(config.get('crustal_domain_size', [200e3, 200e3, 35e3]))
    )
    
    flow_params = FluidFlowParameters(
        permeability=config.get('permeability', 1e-15),
        porosity=config.get('porosity', 0.1),
        domain_size=tuple(config.get('flow_domain_size', [1000.0, 1000.0, 500.0]))
    )
    
    reaction_params = ReactionParameters(
        n_species=config.get('n_species', 20),
        n_reactions=config.get('n_reactions', 15)
    )
    
    # Create integrated parameters
    integrated_params = IntegratedModelParameters(
        convection_params=convection_params,
        deformation_params=deformation_params,
        flow_params=flow_params,
        reaction_params=reaction_params,
        thermal_coupling_strength=config.get('thermal_coupling', 1.0),
        mechanical_coupling_strength=config.get('mechanical_coupling', 1.0),
        chemical_coupling_strength=config.get('chemical_coupling', 1.0)
    )
    
    model = IntegratedGeologicalModel(integrated_params)
    
    logger.info("Created sophisticated integrated geological process model")
    
    return model