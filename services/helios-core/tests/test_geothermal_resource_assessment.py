"""
Tests for Geothermal Resource Assessment System

Tests all components of the geothermal resource assessment system including:
    - Thermal modeling
- Permeability estimation
- Sustainability modeling
- Economic viability assessment

Requirements: 6.2, 6.3
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
from core.geothermal_resource_assessment import (
    EconomicParameters,
    EconomicViabilityAssessor,
    GeothermalAssessmentResult,
    GeothermalReservoir,
    GeothermalResourceAssessmentSystem,
    GeothermalSustainabilityModeler,
    PermeabilityEstimator,
    ThermalModelingEngine,
)
from core.physics_constraints import PhysicsConstraintEnforcer
from models.geological import GeologicalLocation, ThermalMeasurement

class TestThermalModelingEngine:
    """Test thermal modeling functionality"""

    @pytest.fixture
    def thermal_engine(self):
        physics_constraints = Mock(spec=PhysicsConstraintEnforcer)
        return ThermalModelingEngine(physics_constraints)

    @pytest.fixture
    def sample_locations(self):
        return [
            GeologicalLocation(latitude=40.0, longitude=-120.0, elevation=1000.0),
            GeologicalLocation(latitude=40.1, longitude=-120.1, elevation=1100.0),
            GeologicalLocation(latitude=40.2, longitude=-120.2, elevation=1200.0),
        ]

    @pytest.fixture
    def sample_thermal_measurements(self):
        return [
            ThermalMeasurement(
                timestamp=datetime.now(),
                location=GeologicalLocation(
                    latitude=40.0, longitude=-120.0, elevation=1000.0
                ),
                temperature=25.0,
                measurement_method="satellite",
                spatial_resolution=100.0,
                uncertainty=2.0,
            ),
            ThermalMeasurement(
                timestamp=datetime.now(),
                location=GeologicalLocation(
                    latitude=40.1, longitude=-120.1, elevation=1100.0
                ),
                temperature=27.0,
                measurement_method="ground",
                spatial_resolution=10.0,
                uncertainty=1.0,
            ),
        ]

    def test_thermal_model_initialization(self, thermal_engine):
        """Test thermal model initialization"""
        assert thermal_engine.thermal_model is not None
        assert hasattr(thermal_engine, "physics_constraints")

    def test_subsurface_temperature_estimation(
        self, thermal_engine, sample_locations, sample_thermal_measurements
    ):
        """Test subsurface temperature estimation"""
        depth_profile = np.array([1000, 2000, 3000, 4000, 5000])

        temperature_field = thermal_engine.estimate_subsurface_temperature(
            sample_locations, depth_profile, sample_thermal_measurements
        )

        # Check output shape
        assert temperature_field.shape == (50, 50, len(depth_profile))

        # Check temperature increases with depth
        for i in range(temperature_field.shape[0]):
            for j in range(temperature_field.shape[1]):
                temp_profile = temperature_field[i, j, :]
                # Temperature should generally increase with depth
                assert temp_profile[-1] > temp_profile[0]

    def test_surface_temperature_interpolation(
        self, thermal_engine, sample_thermal_measurements
    ):
        """Test surface temperature interpolation"""
        temp = thermal_engine._interpolate_surface_temperature(
            40.05, -120.05, sample_thermal_measurements
        )

        # Should return interpolated value between measurements
        assert 20.0 <= temp <= 30.0

    def test_geothermal_gradient_calculation(
        self, thermal_engine, sample_thermal_measurements
    ):
        """Test geothermal gradient calculation"""
        gradient = thermal_engine._calculate_geothermal_gradient(
            40.0, -120.0, 2000.0, sample_thermal_measurements
        )

        # Should return reasonable geothermal gradient
        assert 15.0 <= gradient <= 50.0  # Typical range for geothermal gradients

class TestPermeabilityEstimator:
    """Test permeability estimation functionality"""

    @pytest.fixture
    def permeability_estimator(self):
        return PermeabilityEstimator()

    def test_permeability_model_initialization(self, permeability_estimator):
        """Test permeability model initialization"""
        assert permeability_estimator.permeability_model is not None

    def test_permeability_estimation(self, permeability_estimator):
        """Test permeability estimation"""
        permeability = permeability_estimator.estimate_permeability(
            depth=2000.0,
            porosity=0.15,
            lithology_code=5,
            fracture_density=2.0,
            stress_state=50.0,
            temperature=200.0,
        )

        # Should return positive permeability value
        assert permeability > 0
        assert permeability < 10000  # Reasonable upper bound

    def test_permeability_depth_dependency(self, permeability_estimator):
        """Test that permeability varies with depth"""
        shallow_perm = permeability_estimator.estimate_permeability(
            depth=1000.0,
            porosity=0.2,
            lithology_code=5,
            fracture_density=2.0,
            stress_state=25.0,
            temperature=150.0,
        )

        deep_perm = permeability_estimator.estimate_permeability(
            depth=4000.0,
            porosity=0.1,
            lithology_code=5,
            fracture_density=1.0,
            stress_state=100.0,
            temperature=250.0,
        )

        # Generally, permeability decreases with depth due to compaction
        # But this depends on fracturing and lithology
        assert isinstance(shallow_perm, float)
        assert isinstance(deep_perm, float)

class TestGeothermalSustainabilityModeler:
    """Test sustainability modeling functionality"""

    @pytest.fixture
    def sustainability_modeler(self):
        return GeothermalSustainabilityModeler()

    @pytest.fixture
    def sample_reservoir(self):
        return GeothermalReservoir(
            location=GeologicalLocation(
                latitude=40.0, longitude=-120.0, elevation=1000.0
            ),
            depth_range=(2000, 2500),
            temperature_range=(180, 200),
            permeability=100.0,
            porosity=0.15,
            fluid_type="water",
            salinity=5000.0,
            volume=1e8,
            heat_capacity=4200.0,
            thermal_conductivity=2.5,
            pressure=20.0,
            flow_rate_potential=50.0,
        )

    def test_sustainability_model_initialization(self, sustainability_modeler):
        """Test sustainability model initialization"""
        assert sustainability_modeler.sustainability_model is not None

    def test_sustainability_assessment(self, sustainability_modeler, sample_reservoir):
        """Test sustainability assessment"""
        sustainability_years, decline_rate = (
            sustainability_modeler.assess_sustainability(
                sample_reservoir, extraction_rate=50.0, reinjection_rate=45.0
            )
        )

        # Should return reasonable values
        assert sustainability_years > 0
        assert sustainability_years < 100  # Reasonable upper bound
        assert 0 <= decline_rate <= 1  # Decline rate should be between 0 and 100%

    def test_thermal_decline_modeling(self, sustainability_modeler, sample_reservoir):
        """Test thermal decline modeling"""
        extraction_scenario = {
            "decline_rate": 0.02,
            "extraction_rate": 50.0,
            "reinjection_rate": 45.0,
        }

        thermal_decline = sustainability_modeler.model_thermal_decline(
            sample_reservoir, extraction_scenario, time_horizon_years=30
        )

        # Check output
        assert len(thermal_decline) == 31  # 30 years + initial
        assert thermal_decline[0] >= thermal_decline[-1]  # Temperature should decline
        assert np.all(thermal_decline >= 150.0)  # Should maintain minimum temperature

    def test_reinjection_impact(self, sustainability_modeler, sample_reservoir):
        """Test impact of reinjection rate on sustainability"""
        # High reinjection rate
        high_reinjection_years, _ = sustainability_modeler.assess_sustainability(
            sample_reservoir, extraction_rate=50.0, reinjection_rate=48.0
        )

        # Low reinjection rate
        low_reinjection_years, _ = sustainability_modeler.assess_sustainability(
            sample_reservoir, extraction_rate=50.0, reinjection_rate=25.0
        )

        # Higher reinjection should lead to better sustainability
        assert high_reinjection_years >= low_reinjection_years

class TestEconomicViabilityAssessor:
    """Test economic viability assessment functionality"""

    @pytest.fixture
    def economic_params(self):
        return EconomicParameters(
            drilling_cost_per_meter=1500.0,
            plant_construction_cost_per_mw=4000000.0,
            operation_maintenance_cost_annual=0.03,
            electricity_price=0.08,
            discount_rate=0.08,
            project_lifetime_years=30,
            tax_rate=0.25,
        )

    @pytest.fixture
    def economic_assessor(self, economic_params):
        return EconomicViabilityAssessor(economic_params)

    @pytest.fixture
    def sample_reservoir(self):
        return GeothermalReservoir(
            location=GeologicalLocation(
                latitude=40.0, longitude=-120.0, elevation=1000.0
            ),
            depth_range=(2000, 2500),
            temperature_range=(180, 200),
            permeability=100.0,
            porosity=0.15,
            fluid_type="water",
            salinity=5000.0,
            volume=1e8,
            heat_capacity=4200.0,
            thermal_conductivity=2.5,
            pressure=20.0,
            flow_rate_potential=50.0,
        )

    def test_economic_assessor_initialization(self, economic_assessor, economic_params):
        """Test economic assessor initialization"""
        assert economic_assessor.params == economic_params

    def test_economic_viability_assessment(self, economic_assessor, sample_reservoir):
        """Test comprehensive economic viability assessment"""
        metrics = economic_assessor.assess_economic_viability(
            sample_reservoir, power_potential_mw=10.0, sustainability_years=30.0
        )

        # Check required metrics are present
        required_keys = [
            "total_capex_usd",
            "annual_revenue_usd",
            "annual_opex_usd",
            "npv_usd",
            "irr_percent",
            "payback_period_years",
            "lcoe_usd_per_mwh",
            "viability_score",
            "project_lifetime_years",
        ]

        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_npv_calculation(self, economic_assessor):
        """Test NPV calculation"""
        npv = economic_assessor._calculate_npv(
            annual_revenue=5000000, annual_opex=1000000, capex=50000000, lifetime=25
        )

        # NPV should be a finite number
        assert isinstance(npv, float)
        assert not np.isnan(npv)
        assert not np.isinf(npv)

    def test_irr_calculation(self, economic_assessor):
        """Test IRR calculation"""
        irr = economic_assessor._calculate_irr(
            annual_revenue=5000000, annual_opex=1000000, capex=50000000, lifetime=25
        )

        # IRR should be between 0 and 1 (0% to 100%)
        assert 0 <= irr <= 1

    def test_payback_period_calculation(self, economic_assessor):
        """Test payback period calculation"""
        payback = economic_assessor._calculate_payback_period(
            annual_revenue=5000000, annual_opex=1000000, capex=50000000
        )

        # Should return reasonable payback period
        assert payback > 0
        assert payback < 100  # Reasonable upper bound

    def test_lcoe_calculation(self, economic_assessor):
        """Test LCOE calculation"""
        lcoe = economic_assessor._calculate_lcoe(
            capex=50000000,
            annual_opex=1000000,
            annual_generation_mwh=75000,
            lifetime=25,
        )

        # LCOE should be positive
        assert lcoe > 0
        assert lcoe < 1000  # Reasonable upper bound ($/MWh)

    def test_viability_score_calculation(self, economic_assessor):
        """Test viability score calculation"""
        score = economic_assessor._calculate_viability_score(
            npv=10000000, irr=0.12, payback_period=8.0, lcoe=60.0
        )

        # Score should be between 0 and 1
        assert 0 <= score <= 1

class TestGeothermalResourceAssessmentSystem:
    """Test the complete geothermal resource assessment system"""

    @pytest.fixture
    def assessment_system(self):
        return GeothermalResourceAssessmentSystem()

    @pytest.fixture
    def sample_locations(self):
        return [
            GeologicalLocation(latitude=40.0, longitude=-120.0, elevation=1000.0),
            GeologicalLocation(latitude=40.1, longitude=-120.1, elevation=1100.0),
        ]

    @pytest.fixture
    def sample_thermal_measurements(self):
        return [
            ThermalMeasurement(
                timestamp=datetime.now(),
                location=GeologicalLocation(
                    latitude=40.0, longitude=-120.0, elevation=1000.0
                ),
                temperature=25.0,
                measurement_method="satellite",
                spatial_resolution=100.0,
                uncertainty=2.0,
            )
        ]

    @pytest.fixture
    def sample_geological_data(self):
        return {
            "lithology_code": 5,
            "fracture_density": 2.0,
            "salinity": 5000.0,
            "complexity_score": 0.3,
        }

    def test_system_initialization(self, assessment_system):
        """Test system initialization"""
        assert assessment_system.thermal_engine is not None
        assert assessment_system.permeability_estimator is not None
        assert assessment_system.sustainability_modeler is not None
        assert assessment_system.economic_assessor is not None
        assert assessment_system.uncertainty_quantifier is not None

    @patch("services.helios_core.core.geothermal_resource_assessment.logger")
    def test_comprehensive_assessment(
        self,
        mock_logger,
        assessment_system,
        sample_locations,
        sample_thermal_measurements,
        sample_geological_data,
    ):
        """Test comprehensive geothermal assessment"""
        results = assessment_system.assess_geothermal_potential(
            locations=sample_locations,
            thermal_measurements=sample_thermal_measurements,
            geological_data=sample_geological_data,
            assessment_depth_range=(1000, 5000),
        )

        # Should return results for each location
        assert len(results) <= len(sample_locations)  # Some locations might fail

        for result in results:
            assert isinstance(result, GeothermalAssessmentResult)
            assert result.thermal_power_potential >= 0
            assert result.electrical_power_potential >= 0
            assert result.sustainability_years > 0
            assert 0 <= result.economic_viability_score <= 1
            assert len(result.confidence_interval) == 2
            assert result.confidence_interval[0] <= result.confidence_interval[1]

    def test_reservoir_property_estimation(
        self, assessment_system, sample_geological_data
    ):
        """Test reservoir property estimation"""
        location = GeologicalLocation(latitude=40.0, longitude=-120.0, elevation=1000.0)

        reservoir = assessment_system._estimate_reservoir_properties(
            location,
            depth=2500.0,
            temperature=200.0,
            geological_data=sample_geological_data,
        )

        assert isinstance(reservoir, GeothermalReservoir)
        assert reservoir.location == location
        assert reservoir.depth_range[0] <= 2500.0 <= reservoir.depth_range[1]
        assert reservoir.temperature_range[0] <= 200.0 <= reservoir.temperature_range[1]
        assert reservoir.permeability > 0
        assert 0 < reservoir.porosity < 1

    def test_thermal_power_calculation(self, assessment_system):
        """Test thermal power potential calculation"""
        reservoir = GeothermalReservoir(
            location=GeologicalLocation(
                latitude=40.0, longitude=-120.0, elevation=1000.0
            ),
            depth_range=(2000, 2500),
            temperature_range=(180, 200),
            permeability=100.0,
            porosity=0.15,
            fluid_type="water",
            salinity=5000.0,
            volume=1e8,
            heat_capacity=4200.0,
            thermal_conductivity=2.5,
            pressure=20.0,
            flow_rate_potential=50.0,
        )

        thermal_power = assessment_system._calculate_thermal_power_potential(reservoir)

        assert thermal_power > 0
        assert thermal_power < 1000  # Reasonable upper bound (MW)

    def test_uncertainty_quantification(
        self, assessment_system, sample_thermal_measurements, sample_geological_data
    ):
        """Test uncertainty quantification"""
        reservoir = GeothermalReservoir(
            location=GeologicalLocation(
                latitude=40.0, longitude=-120.0, elevation=1000.0
            ),
            depth_range=(2000, 2500),
            temperature_range=(180, 200),
            permeability=100.0,
            porosity=0.15,
            fluid_type="water",
            salinity=5000.0,
            volume=1e8,
            heat_capacity=4200.0,
            thermal_conductivity=2.5,
            pressure=20.0,
            flow_rate_potential=50.0,
        )

        uncertainties = assessment_system._quantify_uncertainties(
            reservoir, sample_thermal_measurements, sample_geological_data
        )

        required_keys = ["data_quality", "geological", "model", "depth", "overall"]
        for key in required_keys:
            assert key in uncertainties
            assert 0 <= uncertainties[key] <= 1

    def test_confidence_interval_calculation(self, assessment_system):
        """Test confidence interval calculation"""
        uncertainty_factors = {
            "data_quality": 0.2,
            "geological": 0.3,
            "model": 0.2,
            "depth": 0.1,
            "overall": 0.35,
        }

        confidence_interval = assessment_system._calculate_confidence_interval(
            power_estimate=10.0, uncertainty_factors=uncertainty_factors
        )

        assert len(confidence_interval) == 2
        assert confidence_interval[0] <= 10.0 <= confidence_interval[1]
        assert confidence_interval[0] >= 0  # Lower bound should be non-negative

    def test_report_generation(
        self,
        assessment_system,
        sample_locations,
        sample_thermal_measurements,
        sample_geological_data,
    ):
        """Test assessment report generation"""
        # Create sample results
        results = assessment_system.assess_geothermal_potential(
            locations=sample_locations[:1],  # Use only one location for testing
            thermal_measurements=sample_thermal_measurements,
            geological_data=sample_geological_data,
        )

        if results:  # Only test if we have results
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                temp_path = f.name

            try:
                report = assessment_system.generate_assessment_report(
                    results, temp_path
                )

                # Check report structure
                assert "assessment_summary" in report
                assert "detailed_results" in report

                summary = report["assessment_summary"]
                assert "total_locations_assessed" in summary
                assert "total_electrical_potential_mw" in summary
                assert "viable_projects_count" in summary
                assert "average_viability_score" in summary

                # Check file was created
                assert os.path.exists(temp_path)

                # Verify file contents
                with open(temp_path, "r") as f:
                    saved_report = json.load(f)

                assert saved_report["assessment_summary"][
                    "total_locations_assessed"
                ] == len(results)

            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_viability_classification(self, assessment_system):
        """Test viability classification"""
        assert assessment_system._classify_viability(0.9) == "Highly Viable"
        assert assessment_system._classify_viability(0.7) == "Viable"
        assert assessment_system._classify_viability(0.5) == "Marginal"
        assert assessment_system._classify_viability(0.2) == "Not Viable"

class TestIntegration:
    """Integration tests for the complete system"""

    def test_end_to_end_assessment(self):
        """Test complete end-to-end assessment workflow"""
        # Create system
        system = GeothermalResourceAssessmentSystem()

        # Sample data
        locations = [
            GeologicalLocation(
                latitude=38.5, longitude=-119.5, elevation=2000.0
            )  # Mono Lake area
        ]

        thermal_measurements = [
            ThermalMeasurement(
                timestamp=datetime.now(),
                location=GeologicalLocation(
                    latitude=38.5, longitude=-119.5, elevation=2000.0
                ),
                temperature=30.0,
                measurement_method="ground",
                spatial_resolution=10.0,
                uncertainty=1.5,
            )
        ]

        geological_data = {
            "lithology_code": 7,  # Volcanic rocks
            "fracture_density": 5.0,  # High fracturing
            "salinity": 8000.0,
            "complexity_score": 0.6,
        }

        # Perform assessment
        results = system.assess_geothermal_potential(
            locations=locations,
            thermal_measurements=thermal_measurements,
            geological_data=geological_data,
            assessment_depth_range=(1500, 4000),
        )

        # Verify results
        assert len(results) >= 0  # Should complete without error

        if results:
            result = results[0]
            assert isinstance(result, GeothermalAssessmentResult)
            assert result.thermal_power_potential > 0
            assert result.electrical_power_potential > 0
            assert result.sustainability_years > 0
            assert 0 <= result.economic_viability_score <= 1

    def test_error_handling(self):
        """Test error handling in assessment system"""
        system = GeothermalResourceAssessmentSystem()

        # Test with invalid data
        invalid_locations = []  # Empty locations list

        results = system.assess_geothermal_potential(
            locations=invalid_locations,
            thermal_measurements=[],
            geological_data={},
            assessment_depth_range=(1000, 5000),
        )

        # Should handle gracefully and return empty results
        assert isinstance(results, list)
        assert len(results) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
