"""
Earthquake Impact Modeling Example

Demonstrates the comprehensive earthquake impact modeling capabilities including:
    - Ground shaking simulation using physics-based models
- Liquefaction susceptibility mapping and modeling
- Landslide triggering and runout simulation
- Tsunami generation and propagation modeling

This example shows how to use the earthquake impact modeling system
for a hypothetical M7.5 earthquake scenario.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.nn.functional as F
from core.earthquake_impact_modeling import (
    EarthquakeImpactModelingEngine,
    GroundMotionModel,
    SoilType,
)
from models.geological import EventType, GeologicalLocation, SeismicEvent

def create_scenario_earthquake():
    """Create a scenario earthquake for impact modeling"""
    return SeismicEvent(
        timestamp=datetime.utcnow(),
        location=GeologicalLocation(
            latitude=37.7749,  # San Francisco Bay Area
            longitude=-122.4194,
            elevation=-500.0,  # Offshore for tsunami potential
        ),
        magnitude=7.5,
        depth=12.0,  # Shallow for maximum impact
        event_type=EventType.EARTHQUAKE,
        event_id="scenario_earthquake_001",
    )

def create_target_locations():
    """Create a grid of target locations for impact assessment"""
    locations = []

    # Create a grid around the San Francisco Bay Area
    lat_range = np.linspace(37.4, 38.1, 8)  # ~80 km north-south
    lon_range = np.linspace(-122.8, -122.0, 9)  # ~90 km east-west

    for i, lat in enumerate(lat_range):
        for j, lon in enumerate(lon_range):
            # Vary elevation based on distance from coast
            elevation = 50.0 + (j * 20.0)  # Higher elevation inland

            location = GeologicalLocation(
                latitude=lat, longitude=lon, elevation=elevation
            )
            locations.append(location)

    return locations

def create_soil_conditions(num_locations):
    """Create soil condition data for target locations"""
    soil_conditions = {}

    for i in range(num_locations):
        # Vary soil conditions based on location
        if i % 4 == 0:
            soil_conditions[str(i)] = SoilType.ROCK
        elif i % 4 == 1:
            soil_conditions[str(i)] = SoilType.STIFF_SOIL
        elif i % 4 == 2:
            soil_conditions[str(i)] = SoilType.SOFT_SOIL
        else:
            soil_conditions[str(i)] = SoilType.VERY_SOFT_SOIL

    return soil_conditions

def create_soil_properties(num_locations):
    """Create soil properties for liquefaction analysis"""
    soil_properties = {}

    for i in range(num_locations):
        # Generate realistic soil properties with some variation
        base_n_value = 15 + np.random.normal(0, 5)
        base_fines = 25 + np.random.normal(0, 10)

        soil_properties[str(i)] = {
            "spt_n_value": max(5, min(50, base_n_value)),
            "fines_content": max(5, min(60, base_fines)),
            "total_stress": 100000 + i * 5000,  # Increases with depth/location
            "effective_stress": 50000 + i * 2500,
        }

    return soil_properties

def create_groundwater_data(num_locations):
    """Create groundwater depth data"""
    groundwater_data = {}

    for i in range(num_locations):
        # Vary groundwater depth (shallower near coast)
        depth = 2.0 + (i % 10) * 1.5  # 2-15 meters
        groundwater_data[str(i)] = depth

    return groundwater_data

def create_slope_data(num_locations):
    """Create slope angle data for landslide analysis"""
    slope_data = {}

    for i in range(num_locations):
        # Generate realistic slope angles
        if i % 5 == 0:
            slope = 5.0 + np.random.uniform(0, 10)  # Gentle slopes
        elif i % 5 == 1:
            slope = 15.0 + np.random.uniform(0, 15)  # Moderate slopes
        elif i % 5 == 2:
            slope = 30.0 + np.random.uniform(0, 15)  # Steep slopes
        else:
            slope = 45.0 + np.random.uniform(0, 10)  # Very steep slopes

        slope_data[str(i)] = min(slope, 60.0)  # Cap at 60 degrees

    return slope_data

def create_soil_strength_data(num_locations):
    """Create soil strength parameters for landslide analysis"""
    soil_strength = {}

    for i in range(num_locations):
        # Generate realistic soil strength parameters
        cohesion = 3000 + np.random.uniform(0, 7000)  # 3-10 kPa
        friction_angle = 25 + np.random.uniform(0, 15)  # 25-40 degrees
        unit_weight = 16000 + np.random.uniform(0, 4000)  # 16-20 kN/m³
        thickness = 1.5 + np.random.uniform(0, 3.5)  # 1.5-5 m

        soil_strength[str(i)] = {
            "cohesion": cohesion,
            "friction_angle": friction_angle,
            "unit_weight": unit_weight,
            "thickness": thickness,
        }

    return soil_strength

def create_coastal_locations():
    """Create coastal locations for tsunami analysis"""
    coastal_locations = []

    # Create locations along the coast
    coastal_lats = np.linspace(37.5, 38.0, 6)
    coastal_lons = np.linspace(-122.7, -122.3, 5)

    for lat in coastal_lats:
        for lon in coastal_lons:
            location = GeologicalLocation(
                latitude=lat, longitude=lon, elevation=5.0  # Near sea level
            )
            coastal_locations.append(location)

    return coastal_locations

def run_ground_shaking_analysis(engine, earthquake, locations, soil_conditions):
    """Run ground shaking analysis"""
    print("Running ground shaking analysis...")

    # Test different ground motion models
    models = [
        GroundMotionModel.BOORE_ATKINSON_2008,
        GroundMotionModel.CAMPBELL_BOZORGNIA_2014,
        GroundMotionModel.CHIOU_YOUNGS_2014,
    ]

    results = {}

    for model in models:
        print(f"  Using {model.value} model...")

        ground_motion = engine.simulate_ground_shaking(
            earthquake=earthquake,
            target_locations=locations,
            soil_conditions=soil_conditions,
            model=model,
        )

        results[model.value] = ground_motion

        # Calculate statistics
        pga_values = [gm.peak_ground_acceleration for gm in ground_motion]
        pgv_values = [gm.peak_ground_velocity for gm in ground_motion]

        print(f"    PGA range: {min(pga_values):.3f} - {max(pga_values):.3f} g")
        print(f"    PGV range: {min(pgv_values):.1f} - {max(pgv_values):.1f} cm/s")

    return results

def run_liquefaction_analysis(engine, ground_motion, soil_properties, groundwater_data):
    """Run liquefaction susceptibility analysis"""
    print("Running liquefaction analysis...")

    liquefaction_results = engine.assess_liquefaction_susceptibility(
        ground_motion=ground_motion,
        soil_properties=soil_properties,
        groundwater_data=groundwater_data,
    )

    # Analyze results
    susceptibility_counts = {}
    high_risk_locations = []

    for result in liquefaction_results:
        susc = result.susceptibility.value
        susceptibility_counts[susc] = susceptibility_counts.get(susc, 0) + 1

        if result.susceptibility.value in ["high", "very_high"]:
            high_risk_locations.append(result)

    print(f"  Total locations analyzed: {len(liquefaction_results)}")
    print("  Susceptibility distribution:")
    for susc, count in susceptibility_counts.items():
        print(f"    {susc}: {count} locations")

    print(f"  High-risk locations: {len(high_risk_locations)}")

    return liquefaction_results

def run_landslide_analysis(engine, ground_motion, slope_data, soil_strength):
    """Run landslide triggering analysis"""
    print("Running landslide analysis...")

    landslide_results = engine.model_landslide_triggering(
        ground_motion=ground_motion, slope_data=slope_data, soil_strength=soil_strength
    )

    # Analyze results
    susceptibility_counts = {}
    high_risk_landslides = []
    total_volume = 0

    for result in landslide_results:
        susc = result.susceptibility.value
        susceptibility_counts[susc] = susceptibility_counts.get(susc, 0) + 1
        total_volume += result.volume_estimate

        if result.susceptibility.value in ["high", "very_high"]:
            high_risk_landslides.append(result)

    print(f"  Locations with landslide potential: {len(landslide_results)}")
    print("  Susceptibility distribution:")
    for susc, count in susceptibility_counts.items():
        print(f"    {susc}: {count} locations")

    print(f"  High-risk landslides: {len(high_risk_landslides)}")
    print(f"  Total estimated volume: {total_volume:,.0f} m³")

    if high_risk_landslides:
        max_runout = max(ls.runout_distance for ls in high_risk_landslides)
        print(f"  Maximum runout distance: {max_runout:.0f} m")

    return landslide_results

def run_tsunami_analysis(engine, earthquake, coastal_locations):
    """Run tsunami generation and propagation analysis"""
    print("Running tsunami analysis...")

    tsunami_result = engine.model_tsunami_generation(
        earthquake=earthquake, coastal_locations=coastal_locations
    )

    if tsunami_result is None:
        print("  No significant tsunami generated")
        return None

    print("  Tsunami generated!")
    print(f"  Source type: {tsunami_result.source_type.value}")
    print(f"  Initial wave height: {tsunami_result.initial_wave_height:.2f} m")
    print(f"  Wave period: {tsunami_result.wave_period:.0f} seconds")
    print(f"  Propagation speed: {tsunami_result.propagation_speed:.0f} m/s")
    print(f"  Affected coastline: {tsunami_result.affected_coastline:.0f} km")

    # Analyze arrival times and wave heights
    arrival_times = list(tsunami_result.arrival_times.values())
    wave_heights = list(tsunami_result.wave_heights.values())
    inundation_distances = list(tsunami_result.inundation_distances.values())

    print(
        f"  Arrival time range: {min(arrival_times):.1f} - {max(arrival_times):.1f} minutes"
    )
    print(f"  Wave height range: {min(wave_heights):.2f} - {max(wave_heights):.2f} m")
    print(
        f"  Inundation range: {min(inundation_distances):.0f} - {max(inundation_distances):.0f} m"
    )

    return tsunami_result

def create_summary_report(
    earthquake,
    ground_motion_results,
    liquefaction_results,
    landslide_results,
    tsunami_result,
):
    """Create a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("EARTHQUAKE IMPACT ASSESSMENT SUMMARY")
    print("=" * 60)

    print("\nScenario Earthquake:")
    print(f"  Magnitude: M{earthquake.magnitude}")
    print(f"  Depth: {earthquake.depth} km")
    print(
        f"  Location: {earthquake.location.latitude:.4f}°N, {earthquake.location.longitude:.4f}°W"
    )

    # Ground motion summary
    print("\nGround Motion Analysis:")
    ba_results = ground_motion_results[GroundMotionModel.BOORE_ATKINSON_2008.value]
    pga_values = [gm.peak_ground_acceleration for gm in ba_results]
    pgv_values = [gm.peak_ground_velocity for gm in ba_results]

    print(f"  Locations analyzed: {len(ba_results)}")
    print(
        f"  Peak Ground Acceleration: {min(pga_values):.3f} - {max(pga_values):.3f} g"
    )
    print(f"  Peak Ground Velocity: {min(pgv_values):.1f} - {max(pgv_values):.1f} cm/s")

    # Liquefaction summary
    print("\nLiquefaction Analysis:")
    print(f"  Locations analyzed: {len(liquefaction_results)}")
    high_liq_risk = sum(
        1
        for lr in liquefaction_results
        if lr.susceptibility.value in ["high", "very_high"]
    )
    print(f"  High-risk locations: {high_liq_risk}")

    # Landslide summary
    print("\nLandslide Analysis:")
    print(f"  Locations with potential: {len(landslide_results)}")
    high_ls_risk = sum(
        1
        for lr in landslide_results
        if lr.susceptibility.value in ["high", "very_high"]
    )
    print(f"  High-risk locations: {high_ls_risk}")

    if landslide_results:
        total_volume = sum(lr.volume_estimate for lr in landslide_results)
        print(f"  Total estimated volume: {total_volume:,.0f} m³")

    # Tsunami summary
    print("\nTsunami Analysis:")
    if tsunami_result:
        print("  Tsunami generated: Yes")
        print(f"  Initial wave height: {tsunami_result.initial_wave_height:.2f} m")
        print(f"  Affected coastline: {tsunami_result.affected_coastline:.0f} km")
    else:
        print("  Tsunami generated: No")

    print("\n" + "=" * 60)

def save_results_to_json(results, filename="earthquake_impact_results.json"):
    """Save results to JSON file for further analysis"""
    # Convert results to serializable format
    serializable_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "earthquake": {
            "magnitude": results["earthquake"].magnitude,
            "depth": results["earthquake"].depth,
            "latitude": results["earthquake"].location.latitude,
            "longitude": results["earthquake"].location.longitude,
        },
        "ground_motion_summary": {
            "num_locations": len(results["ground_motion"]),
            "pga_range": [
                min(gm.peak_ground_acceleration for gm in results["ground_motion"]),
                max(gm.peak_ground_acceleration for gm in results["ground_motion"]),
            ],
            "pgv_range": [
                min(gm.peak_ground_velocity for gm in results["ground_motion"]),
                max(gm.peak_ground_velocity for gm in results["ground_motion"]),
            ],
        },
        "liquefaction_summary": {
            "num_locations": len(results["liquefaction"]),
            "high_risk_count": sum(
                1
                for lr in results["liquefaction"]
                if lr.susceptibility.value in ["high", "very_high"]
            ),
        },
        "landslide_summary": {
            "num_locations": len(results["landslide"]),
            "high_risk_count": sum(
                1
                for lr in results["landslide"]
                if lr.susceptibility.value in ["high", "very_high"]
            ),
            "total_volume": sum(lr.volume_estimate for lr in results["landslide"]),
        },
        "tsunami_summary": {
            "generated": results["tsunami"] is not None,
            "initial_height": (
                results["tsunami"].initial_wave_height if results["tsunami"] else 0
            ),
            "affected_coastline": (
                results["tsunami"].affected_coastline if results["tsunami"] else 0
            ),
        },
    }

    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {filename}")

def main():
    """Main function to run the earthquake impact modeling example"""
    print("Earthquake Impact Modeling Example")
    print("=" * 50)

    # Initialize the modeling engine
    engine = EarthquakeImpactModelingEngine()

    # Create scenario
    earthquake = create_scenario_earthquake()
    locations = create_target_locations()
    coastal_locations = create_coastal_locations()

    print(f"Created scenario: M{earthquake.magnitude} earthquake")
    print(f"Target locations: {len(locations)}")
    print(f"Coastal locations: {len(coastal_locations)}")

    # Create input data
    soil_conditions = create_soil_conditions(len(locations))
    soil_properties = create_soil_properties(len(locations))
    groundwater_data = create_groundwater_data(len(locations))
    slope_data = create_slope_data(len(locations))
    soil_strength = create_soil_strength_data(len(locations))

    print("\nRunning comprehensive impact analysis...")
    print("-" * 50)

    # Run analyses
    ground_motion_results = run_ground_shaking_analysis(
        engine, earthquake, locations, soil_conditions
    )

    # Use Boore-Atkinson results for subsequent analyses
    ba_ground_motion = ground_motion_results[
        GroundMotionModel.BOORE_ATKINSON_2008.value
    ]

    liquefaction_results = run_liquefaction_analysis(
        engine, ba_ground_motion, soil_properties, groundwater_data
    )

    landslide_results = run_landslide_analysis(
        engine, ba_ground_motion, slope_data, soil_strength
    )

    tsunami_result = run_tsunami_analysis(engine, earthquake, coastal_locations)

    # Create summary report
    create_summary_report(
        earthquake,
        ground_motion_results,
        liquefaction_results,
        landslide_results,
        tsunami_result,
    )

    # Save results
    results = {
        "earthquake": earthquake,
        "ground_motion": ba_ground_motion,
        "liquefaction": liquefaction_results,
        "landslide": landslide_results,
        "tsunami": tsunami_result,
    }

    save_results_to_json(results)

    print("\nEarthquake impact modeling example completed successfully!")

if __name__ == "__main__":
    main()
