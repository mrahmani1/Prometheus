"""
Test suite for Multi-Physics Coupling Engine

Tests all components of the advanced multi-physics coupling system including:
    - Individual physics domain solvers
- Coupling term computation
- Adaptive time stepping
- Stability analysis
- Convergence monitoring
- Mesh refinement
"""

import numpy as np
import pytest
import torch
from core.multi_physics_coupling import (
    AdaptiveTimeStepConfig,
    ChemicalSolver,
    ConvergenceConfig,
    ConvergenceStatus,
    ElectromagneticSolver,
    MechanicalSolver,
    MeshRefinementConfig,
    MultiPhysicsCouplingEngine,
    PhysicsDomain,
    ThermalSolver,
    create_chemical_thermal_coupling,
    create_electromagnetic_thermal_coupling,
    create_multi_physics_engine,
    create_thermal_mechanical_coupling,
)
from core.pinn_engine import GeologicalProcess

class TestPhysicsDomainSolvers:
    """Test individual physics domain solvers"""

    def test_thermal_solver_initialization(self):
        """Test thermal solver initialization"""
        solver = ThermalSolver(thermal_diffusivity=1e-6, heat_capacity=1000.0)

        assert solver.thermal_diffusivity == 1e-6
        assert solver.heat_capacity == 1000.0

    def test_thermal_solver_step(self):
        """Test thermal solver time step"""
        solver = ThermalSolver(thermal_diffusivity=1e-6, heat_capacity=1000.0)

        # Create test temperature field
        temperature = torch.ones(10, 10) * 300.0  # 300K
        dt = 0.01
        coupling_terms = [torch.zeros_like(temperature)]

        # Solve step
        temperature_new = solver.solve_step(temperature, dt, coupling_terms)

        assert temperature_new.shape == temperature.shape
        assert torch.all(torch.isfinite(temperature_new))

    def test_mechanical_solver_initialization(self):
        """Test mechanical solver initialization"""
        solver = MechanicalSolver(
            youngs_modulus=70e9, poisson_ratio=0.3, density=2700.0
        )

        assert solver.youngs_modulus == 70e9
        assert solver.poisson_ratio == 0.3
        assert solver.density == 2700.0

        # Check Lamé parameters
        expected_mu = 70e9 / (2 * (1 + 0.3))
        expected_lambda = (70e9 * 0.3) / ((1 + 0.3) * (1 - 2 * 0.3))

        assert abs(solver.mu - expected_mu) < 1e6
        assert abs(solver.lambda_param - expected_lambda) < 1e6

    def test_mechanical_solver_step(self):
        """Test mechanical solver time step"""
        solver = MechanicalSolver(
            youngs_modulus=70e9, poisson_ratio=0.3, density=2700.0
        )

        # Create test displacement field
        displacement = torch.zeros(10, 10, 3)  # 3D displacement
        dt = 1e-6
        coupling_terms = [torch.zeros_like(displacement)]

        # Solve step
        displacement_new = solver.solve_step(displacement, dt, coupling_terms)

        assert displacement_new.shape == displacement.shape
        assert torch.all(torch.isfinite(displacement_new))

    def test_chemical_solver_initialization(self):
        """Test chemical solver initialization"""
        solver = ChemicalSolver(diffusion_coefficient=1e-9, reaction_rate=1e-3)

        assert solver.diffusion_coefficient == 1e-9
        assert solver.reaction_rate == 1e-3

    def test_chemical_solver_step(self):
        """Test chemical solver time step"""
        solver = ChemicalSolver(diffusion_coefficient=1e-9, reaction_rate=1e-3)

        # Create test concentration field
        concentration = torch.ones(10, 10) * 0.1  # 0.1 mol/L
        dt = 0.01
        coupling_terms = [torch.zeros_like(concentration)]

        # Solve step
        concentration_new = solver.solve_step(concentration, dt, coupling_terms)

        assert concentration_new.shape == concentration.shape
        assert torch.all(torch.isfinite(concentration_new))
        assert torch.all(concentration_new >= 0)  # Non-negative concentrations

    def test_electromagnetic_solver_initialization(self):
        """Test electromagnetic solver initialization"""
        solver = ElectromagneticSolver(
            permittivity=8.854e-12, permeability=4e-7 * np.pi
        )

        assert solver.permittivity == 8.854e-12
        assert solver.permeability == 4e-7 * np.pi

        # Check speed of light calculation
        expected_c = 1.0 / np.sqrt(8.854e-12 * 4e-7 * np.pi)
        assert abs(solver.c - expected_c) < 1e6

    def test_electromagnetic_solver_step(self):
        """Test electromagnetic solver time step"""
        solver = ElectromagneticSolver(
            permittivity=8.854e-12, permeability=4e-7 * np.pi
        )

        # Create test electromagnetic field
        em_field = torch.zeros(10, 10, 6)  # [E_x, E_y, E_z, B_x, B_y, B_z]
        dt = 1e-12
        coupling_terms = [torch.zeros_like(em_field)]

        # Solve step
        em_field_new = solver.solve_step(em_field, dt, coupling_terms)

        assert em_field_new.shape == em_field.shape
        assert torch.all(torch.isfinite(em_field_new))

class TestCouplingTerms:
    """Test coupling term implementations"""

    def test_thermal_mechanical_coupling_creation(self):
        """Test creation of thermal-mechanical coupling terms"""
        coupling_terms = create_thermal_mechanical_coupling(
            thermal_expansion_coefficient=1e-5, reference_temperature=293.15
        )

        assert len(coupling_terms) == 2
        assert coupling_terms[0].source_domain == PhysicsDomain.THERMAL
        assert coupling_terms[0].target_domain == PhysicsDomain.MECHANICAL
        assert coupling_terms[1].source_domain == PhysicsDomain.MECHANICAL
        assert coupling_terms[1].target_domain == PhysicsDomain.THERMAL

    def test_thermal_mechanical_coupling_function(self):
        """Test thermal-mechanical coupling function"""
        coupling_terms = create_thermal_mechanical_coupling()

        # Test thermal to mechanical coupling
        temperature = torch.ones(10, 10) * 350.0  # 350K
        displacement = torch.zeros(10, 10, 3)

        thermal_force = coupling_terms[0].coupling_function(temperature, displacement)

        assert thermal_force.shape == displacement.shape
        assert torch.all(torch.isfinite(thermal_force))
        assert torch.any(thermal_force != 0)  # Should produce non-zero force

    def test_chemical_thermal_coupling_creation(self):
        """Test creation of chemical-thermal coupling terms"""
        coupling_terms = create_chemical_thermal_coupling(
            reaction_heat=1e5, activation_energy=5e4, gas_constant=8.314
        )

        assert len(coupling_terms) == 2
        assert coupling_terms[0].source_domain == PhysicsDomain.CHEMICAL
        assert coupling_terms[0].target_domain == PhysicsDomain.THERMAL
        assert coupling_terms[1].source_domain == PhysicsDomain.THERMAL
        assert coupling_terms[1].target_domain == PhysicsDomain.CHEMICAL

    def test_chemical_thermal_coupling_function(self):
        """Test chemical-thermal coupling function"""
        coupling_terms = create_chemical_thermal_coupling()

        # Test chemical to thermal coupling
        concentration = torch.ones(10, 10) * 0.1
        temperature = torch.ones(10, 10) * 300.0

        heat_source = coupling_terms[0].coupling_function(concentration, temperature)

        assert heat_source.shape == concentration.shape
        assert torch.all(torch.isfinite(heat_source))

    def test_electromagnetic_thermal_coupling_creation(self):
        """Test creation of electromagnetic-thermal coupling terms"""
        coupling_terms = create_electromagnetic_thermal_coupling(
            electrical_conductivity=1e6, joule_heating_coefficient=1.0
        )

        assert len(coupling_terms) == 2
        assert coupling_terms[0].source_domain == PhysicsDomain.ELECTROMAGNETIC
        assert coupling_terms[0].target_domain == PhysicsDomain.THERMAL
        assert coupling_terms[1].source_domain == PhysicsDomain.THERMAL
        assert coupling_terms[1].target_domain == PhysicsDomain.ELECTROMAGNETIC

class TestMultiPhysicsCouplingEngine:
    """Test the main multi-physics coupling engine"""

    def create_test_engine(self):
        """Create a test multi-physics coupling engine"""
        # Create domain solvers
        domains = {
            PhysicsDomain.THERMAL: ThermalSolver(
                thermal_diffusivity=1e-6, heat_capacity=1000.0
            ),
            PhysicsDomain.MECHANICAL: MechanicalSolver(
                youngs_modulus=70e9, poisson_ratio=0.3, density=2700.0
            ),
        }

        # Create coupling terms
        coupling_terms = create_thermal_mechanical_coupling()

        # Create configurations
        time_step_config = AdaptiveTimeStepConfig(
            initial_dt=1e-3, min_dt=1e-6, max_dt=1e-2
        )

        convergence_config = ConvergenceConfig(
            max_iterations=10, absolute_tolerance=1e-6
        )

        mesh_refinement_config = MeshRefinementConfig(max_refinement_levels=2)

        # Create engine
        engine = MultiPhysicsCouplingEngine(
            domains=domains,
            coupling_terms=coupling_terms,
            time_step_config=time_step_config,
            convergence_config=convergence_config,
            mesh_refinement_config=mesh_refinement_config,
        )

        return engine

    def test_engine_initialization(self):
        """Test multi-physics engine initialization"""
        engine = self.create_test_engine()

        assert len(engine.domains) == 2
        assert len(engine.coupling_terms) == 2
        assert engine.current_time == 0.0
        assert engine.refinement_level == 0

    def test_state_initialization(self):
        """Test domain state initialization"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }

        engine.initialize_states(initial_states)

        assert PhysicsDomain.THERMAL in engine.domain_states
        assert PhysicsDomain.MECHANICAL in engine.domain_states
        assert engine.domain_states[PhysicsDomain.THERMAL].shape == (5, 5)
        assert engine.domain_states[PhysicsDomain.MECHANICAL].shape == (5, 5, 3)

    def test_coupling_term_computation(self):
        """Test coupling term computation"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 350.0,  # Higher temperature
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Compute coupling terms for mechanical domain
        coupling_terms = engine.compute_coupling_terms(PhysicsDomain.MECHANICAL)

        assert len(coupling_terms) == 1  # One thermal->mechanical coupling
        assert coupling_terms[0].shape == (5, 5, 3)
        assert torch.all(torch.isfinite(coupling_terms[0]))

    def test_coupled_step_solution(self):
        """Test solving one coupled time step"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Solve one step
        dt = 1e-3
        convergence_result = engine.solve_coupled_step(dt)

        assert convergence_result.status in [
            ConvergenceStatus.CONVERGED,
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.MAX_ITERATIONS,
        ]
        assert convergence_result.iterations > 0
        assert convergence_result.final_residual >= 0
        assert len(convergence_result.residual_history) > 0

    def test_stability_analysis(self):
        """Test numerical stability analysis"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Perform stability analysis
        stability_result = engine.analyze_stability()

        assert stability_result.stability_margin >= 0
        assert stability_result.critical_time_step > 0
        assert isinstance(stability_result.is_stable, bool)
        assert isinstance(stability_result.recommendations, list)

    def test_time_step_adaptation(self):
        """Test adaptive time stepping"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Create mock convergence and stability results
        from core.multi_physics_coupling import (
            ConvergenceResult,
            StabilityAnalysisResult,
        )
        from core.multi_physics_coupling import torch.nn as nn
        from core.multi_physics_coupling import torch.nn.functional as F
        from core.multi_physics_coupling import numpy as np

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        convergence_result = ConvergenceResult(
            status=ConvergenceStatus.CONVERGED,
            iterations=5,
            final_residual=1e-7,
            convergence_rate=-0.1,
            time_to_convergence=0.01,
        )

        stability_result = StabilityAnalysisResult(
            is_stable=True, stability_margin=0.1, critical_time_step=1e-2
        )

        initial_dt = engine.current_dt
        engine.adapt_time_step(convergence_result, stability_result)

        # Time step should be adapted
        assert (
            engine.current_dt != initial_dt
            or initial_dt == engine.time_step_config.max_dt
        )
        assert engine.current_dt >= engine.time_step_config.min_dt
        assert engine.current_dt <= engine.time_step_config.max_dt

    def test_mesh_refinement_criteria(self):
        """Test mesh refinement criteria checking"""
        engine = self.create_test_engine()

        # Initialize states with high gradients
        x = torch.linspace(0, 1, 10)
        y = torch.linspace(0, 1, 10)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Create field with high gradient
        temperature = 300.0 + 100.0 * torch.exp(
            -((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01
        )
        displacement = torch.zeros(10, 10, 3)

        initial_states = {
            PhysicsDomain.THERMAL: temperature,
            PhysicsDomain.MECHANICAL: displacement,
        }
        engine.initialize_states(initial_states)

        # Check refinement criteria
        needs_refinement = engine.check_mesh_refinement_criteria()

        assert isinstance(needs_refinement, bool)

    def test_simulation_run(self):
        """Test running a complete simulation"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(3, 3) * 300.0,  # Small grid for fast test
            PhysicsDomain.MECHANICAL: torch.zeros(3, 3, 3),
        }
        engine.initialize_states(initial_states)

        # Run short simulation
        end_time = 1e-2
        results = engine.simulate(end_time, output_frequency=5e-3)

        assert "times" in results
        assert "states" in results
        assert "time_history" in results
        assert "convergence_history" in results
        assert "stability_history" in results

        assert len(results["times"]) > 0
        assert (
            results["final_time"] <= end_time + 1e-6
        )  # Allow small numerical precision error
        assert results["total_steps"] > 0

        # Check that states were saved
        for domain in engine.domains:
            assert domain in results["states"]
            assert len(results["states"][domain]) > 0

    def test_diagnostics(self):
        """Test simulation diagnostics"""
        engine = self.create_test_engine()

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Get diagnostics
        diagnostics = engine.get_diagnostics()

        assert "current_time" in diagnostics
        assert "current_dt" in diagnostics
        assert "total_steps" in diagnostics
        assert "refinement_level" in diagnostics
        assert "domain_states_norm" in diagnostics
        assert "coupling_terms_active" in diagnostics

        assert diagnostics["current_time"] >= 0
        assert diagnostics["current_dt"] > 0
        assert diagnostics["total_steps"] >= 0
        assert diagnostics["refinement_level"] >= 0

class TestFactoryFunctions:
    """Test factory functions for creating multi-physics engines"""

    def test_create_multi_physics_engine_default(self):
        """Test creating multi-physics engine with default parameters"""
        engine = create_multi_physics_engine(GeologicalProcess.MAGMA_FLOW)

        assert len(engine.domains) == 4  # All default domains
        assert PhysicsDomain.THERMAL in engine.domains
        assert PhysicsDomain.MECHANICAL in engine.domains
        assert PhysicsDomain.CHEMICAL in engine.domains
        assert PhysicsDomain.ELECTROMAGNETIC in engine.domains

        assert len(engine.coupling_terms) > 0  # Should have coupling terms

    def test_create_multi_physics_engine_custom(self):
        """Test creating multi-physics engine with custom parameters"""
        domain_configs = {
            PhysicsDomain.THERMAL: {
                "thermal_diffusivity": 2e-6,
                "heat_capacity": 2000.0,
            },
            PhysicsDomain.MECHANICAL: {
                "youngs_modulus": 100e9,
                "poisson_ratio": 0.25,
                "density": 3000.0,
            },
        }

        engine = create_multi_physics_engine(
            GeologicalProcess.CRUSTAL_DEFORMATION, domain_configs=domain_configs
        )

        assert len(engine.domains) == 2
        assert PhysicsDomain.THERMAL in engine.domains
        assert PhysicsDomain.MECHANICAL in engine.domains

        # Check that custom parameters were applied
        thermal_solver = engine.domains[PhysicsDomain.THERMAL]
        assert thermal_solver.thermal_diffusivity == 2e-6
        assert thermal_solver.heat_capacity == 2000.0

        mechanical_solver = engine.domains[PhysicsDomain.MECHANICAL]
        assert mechanical_solver.youngs_modulus == 100e9
        assert mechanical_solver.poisson_ratio == 0.25
        assert mechanical_solver.density == 3000.0

class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_domain_initialization(self):
        """Test error handling for invalid domain initialization"""
        engine = TestMultiPhysicsCouplingEngine().create_test_engine()

        # Try to initialize with invalid domain
        invalid_states = {
            PhysicsDomain.FLUID_DYNAMICS: torch.ones(5, 5)  # Domain not in engine
        }

        with pytest.raises(ValueError, match="Domain .* not registered"):
            engine.initialize_states(invalid_states)

    def test_stability_analysis_failure(self):
        """Test stability analysis with problematic states"""
        engine = TestMultiPhysicsCouplingEngine().create_test_engine()

        # Initialize with problematic states (NaN values)
        initial_states = {
            PhysicsDomain.THERMAL: torch.full((5, 5), float("nan")),
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Stability analysis should handle this gracefully
        stability_result = engine.analyze_stability()

        assert not stability_result.is_stable
        assert len(stability_result.recommendations) > 0

    def test_convergence_failure_handling(self):
        """Test handling of convergence failures"""
        engine = TestMultiPhysicsCouplingEngine().create_test_engine()

        # Set very strict convergence criteria
        engine.convergence_config.max_iterations = 1
        engine.convergence_config.absolute_tolerance = 1e-15

        # Initialize states
        initial_states = {
            PhysicsDomain.THERMAL: torch.ones(5, 5) * 300.0,
            PhysicsDomain.MECHANICAL: torch.zeros(5, 5, 3),
        }
        engine.initialize_states(initial_states)

        # Solve step (should not converge)
        convergence_result = engine.solve_coupled_step(1e-3)

        assert convergence_result.status in [
            ConvergenceStatus.NOT_CONVERGED,
            ConvergenceStatus.MAX_ITERATIONS,
        ]
        assert convergence_result.iterations <= engine.convergence_config.max_iterations

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
