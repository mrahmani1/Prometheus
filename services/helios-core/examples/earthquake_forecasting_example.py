"""
Earthquake Forecasting Example

Demonstrates the use of the probabilistic earthquake forecasting system
including machine learning models, statistical seismicity models,
temporal clustering analysis, and probabilistic hazard assessment.

Requirements: 8.1, 8.7
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.earthquake_forecasting import EarthquakeForecastingEngine, HazardLevel
from models.geological import SeismicEvent, GeologicalLocation, EventType


def generate_synthetic_earthquake_catalog(n_events=100, region_bounds=None):
    """
    Generate synthetic earthquake catalog for demonstration.
    
    Args:
        n_events: Number of events to generate
        region_bounds: Geographic bounds for events
        
    Returns:
        List of SeismicEvent objects
    """
    if region_bounds is None:
        region_bounds = {
            "lat_min": 34.0, "lat_max": 36.0,
            "lon_min": -119.0, "lon_max": -117.0
        }
    
    events = []
    base_time = datetime(2020, 1, 1)
    
    print(f"Generating {n_events} synthetic earthquake events...")
    
    # Generate background seismicity
    for i in range(n_events):
        # Random time with clustering tendency
        if i > 0 and np.random.random() < 0.3:  # 30% chance of clustering
            # Cluster around previous event
            time_offset = np.random.exponential(2)  # Days
            lat_offset = np.random.normal(0, 0.05)
            lon_offset = np.random.normal(0, 0.05)
            prev_event = events[-1]
            latitude = np.clip(
                prev_event.location.latitude + lat_offset,
                region_bounds["lat_min"], region_bounds["lat_max"]
            )
            longitude = np.clip(
                prev_event.location.longitude + lon_offset,
                region_bounds["lon_min"], region_bounds["lon_max"]
            )
        else:
            # Random background event
            time_offset = np.random.exponential(5)  # Days
            latitude = np.random.uniform(region_bounds["lat_min"], region_bounds["lat_max"])
            longitude = np.random.uniform(region_bounds["lon_min"], region_bounds["lon_max"])
        
        timestamp = base_time + timedelta(days=sum(np.random.exponential(5) for _ in range(i)))
        
        # Generate magnitude following Gutenberg-Richter law
        # P(M >= m) = 10^(a - b*m), with b ≈ 1.0
        magnitude = 2.5 + np.random.exponential(1.0)  # Minimum magnitude 2.5
        magnitude = min(magnitude, 8.0)  # Cap at magnitude 8.0
        
        # Depth varies with magnitude (deeper events tend to be larger)
        depth = np.random.uniform(5, 30) + magnitude * 2
        depth = min(depth, 50)  # Cap at 50 km
        
        event = SeismicEvent(
            timestamp=timestamp,
            location=GeologicalLocation(
                latitude=latitude,
                longitude=longitude,
                elevation=0.0
            ),
            magnitude=magnitude,
            depth=depth,
            event_type=EventType.EARTHQUAKE,
            confidence=np.random.uniform(0.8, 1.0),
            agency="SYNTHETIC",
            event_id=f"synth_{i:04d}"
        )
        
        events.append(event)
    
    # Sort events by time
    events.sort(key=lambda e: e.timestamp)
    
    print(f"Generated catalog spanning {(events[-1].timestamp - events[0].timestamp).days} days")
    print(f"Magnitude range: {min(e.magnitude for e in events):.1f} - {max(e.magnitude for e in events):.1f}")
    
    return events


def demonstrate_pattern_analysis(engine, events):
    """Demonstrate earthquake pattern analysis"""
    print("\n" + "="*60)
    print("EARTHQUAKE PATTERN ANALYSIS")
    print("="*60)
    
    try:
        results = engine.analyze_earthquake_patterns(events)
        
        print("\n📊 SPATIAL PATTERNS:")
        spatial = results["spatial_patterns"]
        print(f"  • Mean inter-event distance: {spatial['mean_inter_event_distance_km']:.1f} km")
        print(f"  • Spatial extent: {spatial['spatial_extent_km2']:.1f} km²")
        print(f"  • Centroid: {spatial['centroid'][0]:.3f}°N, {spatial['centroid'][1]:.3f}°W")
        
        print("\n⏰ TEMPORAL PATTERNS:")
        temporal = results["temporal_patterns"]
        print(f"  • Mean inter-event time: {temporal['mean_inter_event_time_hours']:.1f} hours")
        print(f"  • Event rate: {temporal['event_rate_per_day']:.2f} events/day")
        print(f"  • Total time span: {temporal['total_time_span_days']} days")
        
        print("\n📈 MAGNITUDE PATTERNS:")
        magnitude = results["magnitude_patterns"]
        mag_stats = magnitude["magnitude_statistics"]
        print(f"  • Mean magnitude: {mag_stats['mean']:.2f}")
        print(f"  • Magnitude range: {mag_stats['min']:.1f} - {mag_stats['max']:.1f}")
        print(f"  • Standard deviation: {mag_stats['std']:.2f}")
        
        # Gutenberg-Richter parameters
        gr = magnitude["gutenberg_richter"]
        if "error" not in gr:
            print(f"  • Gutenberg-Richter b-value: {gr['b_value']:.2f} ± {gr['b_uncertainty']:.2f}")
            print(f"  • Gutenberg-Richter a-value: {gr['a_value']:.2f} ± {gr['a_uncertainty']:.2f}")
        
        print("\n🤖 MACHINE LEARNING RESULTS:")
        ml_results = results["ml_model_results"]
        for model_name, model_result in ml_results.items():
            print(f"  • {model_name.replace('_', ' ').title()}: CV Score = {model_result['cross_val_score']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in pattern analysis: {e}")
        return None


def demonstrate_statistical_modeling(engine, events):
    """Demonstrate statistical seismicity modeling"""
    print("\n" + "="*60)
    print("STATISTICAL SEISMICITY MODELING")
    print("="*60)
    
    try:
        results = engine.create_statistical_seismicity_model(events)
        
        print("\n📊 GUTENBERG-RICHTER PARAMETERS:")
        gr = results["gutenberg_richter"]
        if "error" not in gr:
            print(f"  • b-value: {gr['b_value']:.3f} ± {gr['b_uncertainty']:.3f}")
            print(f"  • a-value: {gr['a_value']:.3f} ± {gr['a_uncertainty']:.3f}")
            print(f"  • Completeness magnitude: {gr['completeness_magnitude']}")
            print(f"  • Events used: {gr['n_events_used']}")
        else:
            print(f"  ❌ {gr['error']}")
        
        print("\n📈 SEISMICITY RATES (events/year):")
        rates = results["seismicity_rates"]
        for mag_range, rate in sorted(rates.items()):
            if rate > 0:
                print(f"  • {mag_range}: {rate:.3f}")
        
        print("\n⏳ RECURRENCE INTERVALS (years):")
        intervals = results["recurrence_intervals"]
        for mag_range, interval in sorted(intervals.items()):
            if interval != float('inf') and interval < 1000:
                print(f"  • {mag_range}: {interval:.1f}")
        
        print("\n🔍 MODEL VALIDATION:")
        validation = results["model_validation"]
        if "validation_error" not in validation:
            print(f"  • Kolmogorov-Smirnov p-value: {validation['p_value']:.3f}")
            print(f"  • Goodness of fit: {validation['goodness_of_fit']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in statistical modeling: {e}")
        return None


def demonstrate_temporal_clustering(engine, events):
    """Demonstrate temporal clustering analysis"""
    print("\n" + "="*60)
    print("TEMPORAL CLUSTERING ANALYSIS")
    print("="*60)
    
    try:
        clusters = engine.perform_temporal_clustering_analysis(events)
        
        print(f"\n🔍 IDENTIFIED {len(clusters)} EARTHQUAKE CLUSTERS:")
        
        if clusters:
            for i, cluster in enumerate(clusters):
                print(f"\n  Cluster {i+1} ({cluster.cluster_type}):")
                print(f"    • Events: {len(cluster.events)}")
                print(f"    • Spatial extent: {cluster.spatial_extent:.1f} km")
                print(f"    • Temporal extent: {cluster.temporal_extent:.1f} days")
                print(f"    • Magnitude range: {cluster.magnitude_range[0]:.1f} - {cluster.magnitude_range[1]:.1f}")
                print(f"    • Centroid: {cluster.centroid_location.latitude:.3f}°N, {cluster.centroid_location.longitude:.3f}°W")
                
                # Show largest event in cluster
                largest_event = max(cluster.events, key=lambda e: e.magnitude)
                print(f"    • Largest event: M{largest_event.magnitude:.1f} on {largest_event.timestamp.strftime('%Y-%m-%d')}")
        else:
            print("  No significant clusters detected with current parameters.")
            print("  This could indicate:")
            print("    • Events are too dispersed in space/time")
            print("    • Clustering parameters need adjustment")
            print("    • Background seismicity dominates")
        
        return clusters
        
    except Exception as e:
        print(f"❌ Error in clustering analysis: {e}")
        return []


def demonstrate_probabilistic_forecasting(engine, events, target_region):
    """Demonstrate probabilistic hazard assessment"""
    print("\n" + "="*60)
    print("PROBABILISTIC HAZARD ASSESSMENT")
    print("="*60)
    
    try:
        # Generate forecasts for different time periods
        forecast_periods = [7, 30, 365]  # days
        
        for period in forecast_periods:
            print(f"\n🔮 {period}-DAY FORECAST:")
            
            forecast = engine.generate_probabilistic_hazard_assessment(
                events, target_region, forecast_period_days=period
            )
            
            print(f"  • Forecast generated: {forecast.forecast_time.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"  • Hazard level: {forecast.hazard_level.value.upper()}")
            print(f"  • Primary model: {forecast.model_used.value}")
            
            print(f"\n  📊 PROBABILITY ESTIMATES:")
            for mag_range, prob in sorted(forecast.probability_estimates.items()):
                lower, upper = forecast.confidence_intervals[mag_range]
                print(f"    • {mag_range}: {prob:.1%} (95% CI: {lower:.1%} - {upper:.1%})")
            
            print(f"\n  🎯 UNCERTAINTY METRICS:")
            for metric, value in forecast.uncertainty_metrics.items():
                if not metric.endswith('_std') and not metric.endswith('_range'):
                    print(f"    • {metric.replace('_', ' ').title()}: {value:.3f}")
            
            if forecast.contributing_factors:
                print(f"\n  🔍 CONTRIBUTING FACTORS:")
                for factor in forecast.contributing_factors:
                    print(f"    • {factor.replace('_', ' ').title()}")
            
            # Hazard level interpretation
            print(f"\n  💡 INTERPRETATION:")
            if forecast.hazard_level == HazardLevel.LOW:
                print("    • Low seismic hazard - normal background activity expected")
            elif forecast.hazard_level == HazardLevel.MODERATE:
                print("    • Moderate seismic hazard - slightly elevated activity possible")
            elif forecast.hazard_level == HazardLevel.HIGH:
                print("    • High seismic hazard - increased earthquake likelihood")
            elif forecast.hazard_level == HazardLevel.VERY_HIGH:
                print("    • Very high seismic hazard - significant earthquake risk")
            elif forecast.hazard_level == HazardLevel.EXTREME:
                print("    • Extreme seismic hazard - major earthquake possible")
        
        return forecast
        
    except Exception as e:
        print(f"❌ Error in probabilistic forecasting: {e}")
        return None


def create_visualization_plots(events, pattern_results, clusters, forecast):
    """Create visualization plots for the analysis results"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION PLOTS")
    print("="*60)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Earthquake Forecasting Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Magnitude vs Time
        ax1 = axes[0, 0]
        times = [(e.timestamp - events[0].timestamp).days for e in events]
        magnitudes = [e.magnitude for e in events]
        
        scatter = ax1.scatter(times, magnitudes, c=magnitudes, cmap='viridis', alpha=0.7, s=30)
        ax1.set_xlabel('Days since first event')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Earthquake Magnitude vs Time')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Magnitude')
        
        # Plot 2: Spatial distribution
        ax2 = axes[0, 1]
        lats = [e.location.latitude for e in events]
        lons = [e.location.longitude for e in events]
        
        scatter2 = ax2.scatter(lons, lats, c=magnitudes, cmap='viridis', alpha=0.7, s=30)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Spatial Distribution of Earthquakes')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Magnitude')
        
        # Plot 3: Magnitude distribution
        ax3 = axes[1, 0]
        ax3.hist(magnitudes, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Magnitude')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Magnitude Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add Gutenberg-Richter fit if available
        if pattern_results and "magnitude_patterns" in pattern_results:
            gr = pattern_results["magnitude_patterns"]["gutenberg_richter"]
            if "error" not in gr:
                mag_range = np.linspace(min(magnitudes), max(magnitudes), 100)
                gr_fit = 10**(gr["a_value"] - gr["b_value"] * mag_range)
                ax3_twin = ax3.twinx()
                ax3_twin.semilogy(mag_range, gr_fit, 'r-', label=f'G-R fit (b={gr["b_value"]:.2f})')
                ax3_twin.set_ylabel('Cumulative frequency (log scale)', color='r')
                ax3_twin.legend()
        
        # Plot 4: Forecast probabilities
        ax4 = axes[1, 1]
        if forecast:
            mag_ranges = list(forecast.probability_estimates.keys())
            probabilities = list(forecast.probability_estimates.values())
            
            bars = ax4.bar(range(len(mag_ranges)), probabilities, alpha=0.7)
            ax4.set_xlabel('Magnitude Range')
            ax4.set_ylabel('Probability')
            ax4.set_title(f'Forecast Probabilities ({forecast.hazard_level.value.title()} Hazard)')
            ax4.set_xticks(range(len(mag_ranges)))
            ax4.set_xticklabels(mag_ranges, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add confidence intervals
            if forecast.confidence_intervals:
                lower_bounds = [forecast.confidence_intervals[mr][0] for mr in mag_ranges]
                upper_bounds = [forecast.confidence_intervals[mr][1] for mr in mag_ranges]
                errors = [[p - l for p, l in zip(probabilities, lower_bounds)],
                         [u - p for p, u in zip(probabilities, upper_bounds)]]
                ax4.errorbar(range(len(mag_ranges)), probabilities, yerr=errors, 
                           fmt='none', color='black', capsize=5)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = 'earthquake_forecasting_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {plot_filename}")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            print("📊 Plot created (display not available in this environment)")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def save_results_to_json(events, pattern_results, clusters, forecast, filename="earthquake_analysis_results.json"):
    """Save analysis results to JSON file"""
    print(f"\n💾 Saving results to {filename}...")
    
    try:
        # Prepare data for JSON serialization
        results = {
            "metadata": {
                "analysis_date": datetime.utcnow().isoformat(),
                "total_events": len(events),
                "time_span_days": (events[-1].timestamp - events[0].timestamp).days,
                "magnitude_range": [min(e.magnitude for e in events), max(e.magnitude for e in events)]
            },
            "pattern_analysis": pattern_results if pattern_results else {},
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_type": cluster.cluster_type,
                    "n_events": len(cluster.events),
                    "spatial_extent_km": cluster.spatial_extent,
                    "temporal_extent_days": cluster.temporal_extent,
                    "magnitude_range": cluster.magnitude_range,
                    "centroid_lat": cluster.centroid_location.latitude,
                    "centroid_lon": cluster.centroid_location.longitude
                }
                for cluster in clusters
            ],
            "forecast": {
                "forecast_time": forecast.forecast_time.isoformat(),
                "hazard_level": forecast.hazard_level.value,
                "model_used": forecast.model_used.value,
                "probability_estimates": forecast.probability_estimates,
                "confidence_intervals": {
                    k: {"lower": v[0], "upper": v[1]} 
                    for k, v in forecast.confidence_intervals.items()
                },
                "uncertainty_metrics": forecast.uncertainty_metrics,
                "contributing_factors": forecast.contributing_factors
            } if forecast else {}
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main demonstration function"""
    print("🌍 PROMETHEUS AI EARTHQUAKE FORECASTING DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the probabilistic earthquake forecasting system")
    print("including machine learning models, statistical analysis, and hazard assessment.")
    print("=" * 80)
    
    # Initialize the forecasting engine
    print("\n🚀 Initializing Earthquake Forecasting Engine...")
    engine = EarthquakeForecastingEngine()
    
    # Define target region (Southern California example)
    target_region = {
        "lat_min": 34.0,
        "lat_max": 36.0,
        "lon_min": -119.0,
        "lon_max": -117.0
    }
    
    print(f"📍 Target region: {target_region['lat_min']}°N to {target_region['lat_max']}°N, "
          f"{target_region['lon_min']}°W to {target_region['lon_max']}°W")
    
    # Generate synthetic earthquake catalog
    events = generate_synthetic_earthquake_catalog(n_events=150, region_bounds=target_region)
    
    # Demonstrate pattern analysis
    pattern_results = demonstrate_pattern_analysis(engine, events)
    
    # Demonstrate statistical modeling
    statistical_results = demonstrate_statistical_modeling(engine, events)
    
    # Demonstrate temporal clustering
    clusters = demonstrate_temporal_clustering(engine, events)
    
    # Demonstrate probabilistic forecasting
    forecast = demonstrate_probabilistic_forecasting(engine, events, target_region)
    
    # Create visualizations
    create_visualization_plots(events, pattern_results, clusters, forecast)
    
    # Save results
    save_results_to_json(events, pattern_results, clusters, forecast)
    
    print("\n" + "="*80)
    print("🎉 EARTHQUAKE FORECASTING DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nKey capabilities demonstrated:")
    print("✅ Machine learning pattern analysis")
    print("✅ Statistical seismicity modeling (Gutenberg-Richter)")
    print("✅ Temporal clustering analysis")
    print("✅ Probabilistic hazard assessment")
    print("✅ Uncertainty quantification")
    print("✅ Multi-model ensemble forecasting")
    print("\nThis system provides comprehensive earthquake forecasting capabilities")
    print("for disaster preparedness and seismic hazard assessment.")


if __name__ == "__main__":
    main()