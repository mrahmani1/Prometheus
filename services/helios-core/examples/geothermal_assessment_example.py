"""
Geothermal Resource Assessment Example

This example demonstrates the comprehensive geothermal resource assessment capabilities
including thermal modeling, sustainability analysis, and economic viability assessment.

Requirements: 6.2, 6.3
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geothermal_resource_assessment import (
    EconomicParameters,
    GeothermalReservoir,
    GeothermalResourceAssessmentSystem,
)
from models.geological import GeologicalLocation, ThermalMeasurement


def create_sample_data():
    """Create sample data for geothermal assessment"""

    # Sample locations in geothermally active regions
    locations = [
        GeologicalLocation(
            latitude=38.5, longitude=-119.5, elevation=2000.0
        ),  # Mono Lake, CA
        GeologicalLocation(
            latitude=44.5, longitude=-110.8, elevation=2400.0
        ),  # Yellowstone, WY
        GeologicalLocation(
            latitude=40.8, longitude=-121.5, elevation=1800.0
        ),  # Lassen, CA
        GeologicalLocation(
            latitude=42.2, longitude=-122.2, elevation=1200.0
        ),  # Crater Lake, OR
        GeologicalLocation(
            latitude=46.2, longitude=-122.2, elevation=1400.0
        ),  # Mount St. Helens, WA
    ]

    # Sample thermal measurements
    thermal_measurements = [
        ThermalMeasurement(
            timestamp=datetime.now(),
            location=GeologicalLocation(
                latitude=38.5, longitude=-119.5, elevation=2000.0
            ),
            temperature=35.0,
            measurement_method="ground",
            spatial_resolution=10.0,
            uncertainty=1.5,
        ),
        ThermalMeasurement(
            timestamp=datetime.now(),
            location=GeologicalLocation(
                latitude=44.5, longitude=-110.8, elevation=2400.0
            ),
            temperature=45.0,
            measurement_method="satellite",
            spatial_resolution=100.0,
            uncertainty=3.0,
        ),
        ThermalMeasurement(
            timestamp=datetime.now(),
            location=GeologicalLocation(
                latitude=40.8, longitude=-121.5, elevation=1800.0
            ),
            temperature=32.0,
            measurement_method="ground",
            spatial_resolution=5.0,
            uncertainty=1.0,
        ),
        ThermalMeasurement(
            timestamp=datetime.now(),
            location=GeologicalLocation(
                latitude=42.2, longitude=-122.2, elevation=1200.0
            ),
            temperature=28.0,
            measurement_method="borehole",
            spatial_resolution=1.0,
            uncertainty=0.5,
        ),
        ThermalMeasurement(
            timestamp=datetime.now(),
            location=GeologicalLocation(
                latitude=46.2, longitude=-122.2, elevation=1400.0
            ),
            temperature=30.0,
            measurement_method="ground",
            spatial_resolution=15.0,
            uncertainty=2.0,
        ),
    ]

    # Sample geological data
    geological_data = {
        "lithology_code": 7,  # Volcanic rocks
        "fracture_density": 4.0,  # High fracturing in volcanic areas
        "salinity": 6000.0,  # Moderate salinity
        "complexity_score": 0.7,  # High geological complexity
    }

    return locations, thermal_measurements, geological_data


def demonstrate_thermal_modeling():
    """Demonstrate thermal modeling capabilities"""
    print("\n=== Thermal Modeling Demonstration ===")

    # Create assessment system
    system = GeothermalResourceAssessmentSystem()

    # Sample data
    locations, thermal_measurements, _ = create_sample_data()

    # Create depth profile
    depth_profile = np.linspace(1000, 5000, 20)

    print(f"Modeling subsurface temperatures for {len(locations)} locations")
    print(f"Depth range: {depth_profile[0]:.0f} - {depth_profile[-1]:.0f} meters")

    # Perform thermal modeling
    temperature_field = system.thermal_engine.estimate_subsurface_temperature(
        locations[:2], depth_profile, thermal_measurements[:2]  # Use subset for demo
    )

    print(f"Temperature field shape: {temperature_field.shape}")
    print(
        f"Temperature range: {np.min(temperature_field):.1f} - {np.max(temperature_field):.1f} °C"
    )

    # Plot temperature profiles
    plt.figure(figsize=(12, 8))

    for i in range(min(2, temperature_field.shape[0])):
        temp_profile = temperature_field[i, i, :]
        plt.subplot(2, 2, i + 1)
        plt.plot(temp_profile, depth_profile, "b-", linewidth=2)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Depth (m)")
        plt.title(f"Location {i+1}: Thermal Profile")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()

    # Plot temperature field at specific depth
    plt.subplot(2, 2, 3)
    depth_idx = len(depth_profile) // 2  # Middle depth
    temp_slice = temperature_field[:, :, depth_idx]
    plt.imshow(temp_slice, cmap="hot", aspect="auto")
    plt.colorbar(label="Temperature (°C)")
    plt.title(f"Temperature at {depth_profile[depth_idx]:.0f}m depth")
    plt.xlabel("Longitude index")
    plt.ylabel("Latitude index")

    # Plot geothermal gradient
    plt.subplot(2, 2, 4)
    gradients = []
    for i in range(min(2, temperature_field.shape[0])):
        temp_profile = temperature_field[i, i, :]
        gradient = np.gradient(temp_profile, depth_profile / 1000)  # °C/km
        gradients.append(gradient)
        plt.plot(depth_profile, gradient, label=f"Location {i+1}")

    plt.xlabel("Depth (m)")
    plt.ylabel("Geothermal Gradient (°C/km)")
    plt.title("Geothermal Gradient Profiles")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thermal_modeling_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return temperature_field


def demonstrate_sustainability_analysis():
    """Demonstrate sustainability analysis capabilities"""
    print("\n=== Sustainability Analysis Demonstration ===")

    system = GeothermalResourceAssessmentSystem()

    # Create sample reservoir
    reservoir = GeothermalReservoir(
        location=GeologicalLocation(latitude=38.5, longitude=-119.5, elevation=2000.0),
        depth_range=(2000, 2500),
        temperature_range=(180, 220),
        permeability=150.0,
        porosity=0.18,
        fluid_type="water",
        salinity=6000.0,
        volume=2e8,
        heat_capacity=4200.0,
        thermal_conductivity=2.8,
        pressure=22.5,
        flow_rate_potential=75.0,
    )

    print(
        f"Analyzing reservoir at depth {reservoir.depth_range[0]}-{reservoir.depth_range[1]}m"
    )
    print(
        f"Temperature range: {reservoir.temperature_range[0]}-{reservoir.temperature_range[1]}°C"
    )
    print(f"Permeability: {reservoir.permeability:.1f} mD")

    # Test different extraction scenarios
    scenarios = [
        {"name": "Conservative", "extraction": 30.0, "reinjection": 28.0},
        {"name": "Moderate", "extraction": 50.0, "reinjection": 45.0},
        {"name": "Aggressive", "extraction": 80.0, "reinjection": 70.0},
    ]

    results = {}

    plt.figure(figsize=(15, 10))

    for i, scenario in enumerate(scenarios):
        print(f"\nScenario: {scenario['name']}")
        print(f"Extraction rate: {scenario['extraction']} kg/s")
        print(f"Reinjection rate: {scenario['reinjection']} kg/s")

        # Assess sustainability
        sustainability_years, decline_rate = (
            system.sustainability_modeler.assess_sustainability(
                reservoir, scenario["extraction"], scenario["reinjection"]
            )
        )

        print(f"Expected lifetime: {sustainability_years:.1f} years")
        print(f"Annual decline rate: {decline_rate:.3f}")

        # Model thermal decline
        extraction_scenario = {
            "decline_rate": decline_rate,
            "extraction_rate": scenario["extraction"],
            "reinjection_rate": scenario["reinjection"],
        }

        thermal_decline = system.sustainability_modeler.model_thermal_decline(
            reservoir, extraction_scenario, time_horizon_years=50
        )

        results[scenario["name"]] = {
            "sustainability_years": sustainability_years,
            "decline_rate": decline_rate,
            "thermal_decline": thermal_decline,
        }

        # Plot thermal decline
        plt.subplot(2, 3, i + 1)
        time_years = np.arange(len(thermal_decline))
        plt.plot(time_years, thermal_decline, "b-", linewidth=2)
        plt.xlabel("Time (years)")
        plt.ylabel("Temperature (°C)")
        plt.title(f'{scenario["name"]} Scenario')
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=150, color="r", linestyle="--", alpha=0.7, label="Min viable temp"
        )
        plt.legend()

    # Compare scenarios
    plt.subplot(2, 3, 4)
    for scenario_name, result in results.items():
        time_years = np.arange(len(result["thermal_decline"]))
        plt.plot(
            time_years, result["thermal_decline"], linewidth=2, label=scenario_name
        )

    plt.xlabel("Time (years)")
    plt.ylabel("Temperature (°C)")
    plt.title("Scenario Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=150, color="r", linestyle="--", alpha=0.7)

    # Sustainability comparison
    plt.subplot(2, 3, 5)
    scenario_names = list(results.keys())
    lifetimes = [results[name]["sustainability_years"] for name in scenario_names]
    decline_rates = [results[name]["decline_rate"] * 100 for name in scenario_names]

    x = np.arange(len(scenario_names))
    width = 0.35

    plt.bar(x - width / 2, lifetimes, width, label="Lifetime (years)", alpha=0.8)
    plt.bar(
        x + width / 2,
        [dr * 10 for dr in decline_rates],
        width,
        label="Decline rate (% × 10)",
        alpha=0.8,
    )

    plt.xlabel("Scenario")
    plt.ylabel("Value")
    plt.title("Sustainability Metrics")
    plt.xticks(x, scenario_names)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Reinjection efficiency analysis
    plt.subplot(2, 3, 6)
    reinjection_ratios = [s["reinjection"] / s["extraction"] for s in scenarios]
    plt.scatter(reinjection_ratios, lifetimes, s=100, alpha=0.7)

    for i, name in enumerate(scenario_names):
        plt.annotate(
            name,
            (reinjection_ratios[i], lifetimes[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Reinjection Ratio")
    plt.ylabel("Sustainability (years)")
    plt.title("Reinjection Impact on Sustainability")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sustainability_analysis_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


def demonstrate_economic_assessment():
    """Demonstrate economic viability assessment"""
    print("\n=== Economic Viability Assessment Demonstration ===")

    # Create different economic scenarios
    economic_scenarios = [
        {
            "name": "Optimistic",
            "params": EconomicParameters(
                drilling_cost_per_meter=1200.0,
                plant_construction_cost_per_mw=3500000.0,
                electricity_price=0.10,
                discount_rate=0.06,
            ),
        },
        {
            "name": "Base Case",
            "params": EconomicParameters(
                drilling_cost_per_meter=1500.0,
                plant_construction_cost_per_mw=4000000.0,
                electricity_price=0.08,
                discount_rate=0.08,
            ),
        },
        {
            "name": "Conservative",
            "params": EconomicParameters(
                drilling_cost_per_meter=2000.0,
                plant_construction_cost_per_mw=5000000.0,
                electricity_price=0.06,
                discount_rate=0.10,
            ),
        },
    ]

    # Sample reservoir properties
    reservoir = GeothermalReservoir(
        location=GeologicalLocation(latitude=38.5, longitude=-119.5, elevation=2000.0),
        depth_range=(2000, 2500),
        temperature_range=(180, 220),
        permeability=150.0,
        porosity=0.18,
        fluid_type="water",
        salinity=6000.0,
        volume=2e8,
        heat_capacity=4200.0,
        thermal_conductivity=2.8,
        pressure=22.5,
        flow_rate_potential=75.0,
    )

    # Test different power capacities
    power_capacities = [5, 10, 20, 50]  # MW
    sustainability_years = 35.0

    results = {}

    plt.figure(figsize=(16, 12))

    for i, scenario in enumerate(economic_scenarios):
        print(f"\n{scenario['name']} Economic Scenario:")

        assessor = system.economic_assessor.__class__(scenario["params"])
        scenario_results = []

        for power_mw in power_capacities:
            metrics = assessor.assess_economic_viability(
                reservoir, power_mw, sustainability_years
            )

            scenario_results.append({"power_mw": power_mw, "metrics": metrics})

            print(
                f"  {power_mw} MW: NPV=${metrics['npv_usd']:,.0f}, "
                f"IRR={metrics['irr_percent']:.1f}%, "
                f"Viability={metrics['viability_score']:.2f}"
            )

        results[scenario["name"]] = scenario_results

        # Plot NPV vs Power Capacity
        plt.subplot(3, 4, i + 1)
        powers = [r["power_mw"] for r in scenario_results]
        npvs = [
            r["metrics"]["npv_usd"] / 1e6 for r in scenario_results
        ]  # Convert to millions
        plt.plot(powers, npvs, "o-", linewidth=2, markersize=8)
        plt.xlabel("Power Capacity (MW)")
        plt.ylabel("NPV (Million USD)")
        plt.title(f'{scenario["name"]}: NPV vs Capacity')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.7)

        # Plot IRR vs Power Capacity
        plt.subplot(3, 4, i + 5)
        irrs = [r["metrics"]["irr_percent"] for r in scenario_results]
        plt.plot(powers, irrs, "s-", linewidth=2, markersize=8, color="orange")
        plt.xlabel("Power Capacity (MW)")
        plt.ylabel("IRR (%)")
        plt.title(f'{scenario["name"]}: IRR vs Capacity')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=8, color="r", linestyle="--", alpha=0.7, label="Hurdle rate")
        plt.legend()

        # Plot Viability Score vs Power Capacity
        plt.subplot(3, 4, i + 9)
        viability_scores = [r["metrics"]["viability_score"] for r in scenario_results]
        plt.plot(
            powers, viability_scores, "^-", linewidth=2, markersize=8, color="green"
        )
        plt.xlabel("Power Capacity (MW)")
        plt.ylabel("Viability Score")
        plt.title(f'{scenario["name"]}: Viability vs Capacity')
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=0.6, color="r", linestyle="--", alpha=0.7, label="Viable threshold"
        )
        plt.legend()

    # Comparison plot
    plt.subplot(3, 4, 4)
    for scenario_name, scenario_results in results.items():
        powers = [r["power_mw"] for r in scenario_results]
        viability_scores = [r["metrics"]["viability_score"] for r in scenario_results]
        plt.plot(powers, viability_scores, "o-", linewidth=2, label=scenario_name)

    plt.xlabel("Power Capacity (MW)")
    plt.ylabel("Viability Score")
    plt.title("Scenario Comparison: Viability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.6, color="r", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("economic_assessment_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


def demonstrate_comprehensive_assessment():
    """Demonstrate comprehensive geothermal resource assessment"""
    print("\n=== Comprehensive Geothermal Resource Assessment ===")

    # Create assessment system
    system = GeothermalResourceAssessmentSystem()

    # Get sample data
    locations, thermal_measurements, geological_data = create_sample_data()

    print(f"Assessing {len(locations)} potential geothermal sites")
    print("Locations:")
    for i, loc in enumerate(locations):
        print(
            f"  {i+1}. Lat: {loc.latitude:.2f}, Lon: {loc.longitude:.2f}, "
            f"Elev: {loc.elevation:.0f}m"
        )

    # Perform comprehensive assessment
    results = system.assess_geothermal_potential(
        locations=locations,
        thermal_measurements=thermal_measurements,
        geological_data=geological_data,
        assessment_depth_range=(1500, 4000),
    )

    print(f"\nAssessment completed for {len(results)} locations")

    if not results:
        print("No viable results found")
        return

    # Analyze results
    total_potential = sum(r.electrical_power_potential for r in results)
    viable_sites = [r for r in results if r.economic_viability_score > 0.6]

    print(f"Total electrical potential: {total_potential:.1f} MW")
    print(f"Viable sites: {len(viable_sites)}/{len(results)}")

    # Create comprehensive visualization
    plt.figure(figsize=(20, 15))

    # Site locations and potential
    plt.subplot(3, 4, 1)
    lats = [r.reservoir.location.latitude for r in results]
    lons = [r.reservoir.location.longitude for r in results]
    potentials = [r.electrical_power_potential for r in results]

    scatter = plt.scatter(lons, lats, c=potentials, s=200, cmap="hot", alpha=0.7)
    plt.colorbar(scatter, label="Electrical Potential (MW)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Site Locations and Power Potential")
    plt.grid(True, alpha=0.3)

    # Power potential distribution
    plt.subplot(3, 4, 2)
    plt.hist(potentials, bins=10, alpha=0.7, edgecolor="black")
    plt.xlabel("Electrical Power Potential (MW)")
    plt.ylabel("Number of Sites")
    plt.title("Power Potential Distribution")
    plt.grid(True, alpha=0.3)

    # Economic viability scores
    plt.subplot(3, 4, 3)
    viability_scores = [r.economic_viability_score for r in results]
    plt.hist(viability_scores, bins=10, alpha=0.7, edgecolor="black", color="green")
    plt.xlabel("Economic Viability Score")
    plt.ylabel("Number of Sites")
    plt.title("Economic Viability Distribution")
    plt.axvline(x=0.6, color="r", linestyle="--", label="Viable threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sustainability vs Viability
    plt.subplot(3, 4, 4)
    sustainability_years = [r.sustainability_years for r in results]
    plt.scatter(sustainability_years, viability_scores, s=100, alpha=0.7)
    plt.xlabel("Sustainability (years)")
    plt.ylabel("Economic Viability Score")
    plt.title("Sustainability vs Economic Viability")
    plt.grid(True, alpha=0.3)

    # Reservoir properties
    plt.subplot(3, 4, 5)
    depths = [
        (r.reservoir.depth_range[0] + r.reservoir.depth_range[1]) / 2 for r in results
    ]
    temperatures = [
        (r.reservoir.temperature_range[0] + r.reservoir.temperature_range[1]) / 2
        for r in results
    ]
    plt.scatter(depths, temperatures, c=potentials, s=100, cmap="hot", alpha=0.7)
    plt.colorbar(label="Power Potential (MW)")
    plt.xlabel("Reservoir Depth (m)")
    plt.ylabel("Reservoir Temperature (°C)")
    plt.title("Reservoir Properties vs Potential")
    plt.grid(True, alpha=0.3)

    # Permeability vs Power
    plt.subplot(3, 4, 6)
    permeabilities = [r.reservoir.permeability for r in results]
    plt.scatter(permeabilities, potentials, s=100, alpha=0.7, color="blue")
    plt.xlabel("Permeability (mD)")
    plt.ylabel("Electrical Power Potential (MW)")
    plt.title("Permeability vs Power Potential")
    plt.grid(True, alpha=0.3)

    # Uncertainty analysis
    plt.subplot(3, 4, 7)
    overall_uncertainties = [r.uncertainty_factors["overall"] for r in results]
    confidence_widths = [
        r.confidence_interval[1] - r.confidence_interval[0] for r in results
    ]
    plt.scatter(
        overall_uncertainties, confidence_widths, s=100, alpha=0.7, color="purple"
    )
    plt.xlabel("Overall Uncertainty")
    plt.ylabel("Confidence Interval Width (MW)")
    plt.title("Uncertainty vs Confidence Interval")
    plt.grid(True, alpha=0.3)

    # Top sites ranking
    plt.subplot(3, 4, 8)
    # Sort by viability score
    sorted_results = sorted(
        results, key=lambda x: x.economic_viability_score, reverse=True
    )
    top_5 = sorted_results[: min(5, len(sorted_results))]

    site_names = [f"Site {i+1}" for i in range(len(top_5))]
    scores = [r.economic_viability_score for r in top_5]

    bars = plt.bar(site_names, scores, alpha=0.7, color="gold")
    plt.ylabel("Economic Viability Score")
    plt.title("Top Sites by Viability")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center",
            va="bottom",
        )

    # Detailed results for top sites
    for i in range(4):
        if i < len(top_5):
            result = top_5[i]
            plt.subplot(3, 4, 9 + i)

            # Create radar chart data
            categories = [
                "Power\n(MW)",
                "Viability\nScore",
                "Sustainability\n(years/50)",
                "Temperature\n(°C/300)",
                "Permeability\n(mD/1000)",
            ]

            values = [
                result.electrical_power_potential / 50,  # Normalize to 0-1
                result.economic_viability_score,
                result.sustainability_years / 50,
                (
                    (
                        result.reservoir.temperature_range[0]
                        + result.reservoir.temperature_range[1]
                    )
                    / 2
                )
                / 300,
                result.reservoir.permeability / 1000,
            ]

            # Simple bar chart instead of radar
            plt.bar(range(len(categories)), values, alpha=0.7)
            plt.xticks(range(len(categories)), categories, rotation=45, ha="right")
            plt.ylabel("Normalized Value")
            plt.title(f"Site {i+1} Profile")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comprehensive_assessment_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Generate detailed report
    report = system.generate_assessment_report(
        results, "geothermal_assessment_report.json"
    )

    print("\nDetailed report saved to: geothermal_assessment_report.json")
    print("Report summary:")
    print(
        f"  Total locations assessed: {report['assessment_summary']['total_locations_assessed']}"
    )
    print(
        f"  Total electrical potential: {report['assessment_summary']['total_electrical_potential_mw']:.1f} MW"
    )
    print(f"  Viable projects: {report['assessment_summary']['viable_projects_count']}")
    print(
        f"  Average viability score: {report['assessment_summary']['average_viability_score']:.2f}"
    )

    return results, report


def main():
    """Main demonstration function"""
    print("Geothermal Resource Assessment System Demonstration")
    print("=" * 60)

    try:
        # Demonstrate individual components
        print("1. Thermal Modeling")
        demonstrate_thermal_modeling()

        print("\n2. Sustainability Analysis")
        demonstrate_sustainability_analysis()

        print("\n3. Economic Assessment")
        demonstrate_economic_assessment()

        print("\n4. Comprehensive Assessment")
        assessment_results, report = demonstrate_comprehensive_assessment()

        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("\nGenerated files:")
        print("  - thermal_modeling_results.png")
        print("  - sustainability_analysis_results.png")
        print("  - economic_assessment_results.png")
        print("  - comprehensive_assessment_results.png")
        print("  - geothermal_assessment_report.json")

    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
