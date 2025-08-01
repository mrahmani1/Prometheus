#!/usr/bin/env python3
"""
Simple InSAR Deformation Analysis Demo

A simplified demonstration of the InSAR analysis capabilities.

Requirements addressed: 3.3, 1.2
"""

import os
import sys

import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.insar_analysis import InSARDeformationAnalyzer, load_insar_data

def main():
    """Simple demo of InSAR analysis"""
    print("InSAR Deformation Analysis - Simple Demo")
    print("=" * 50)

    # Initialize analyzer
    print("Initializing InSAR analyzer...")
    analyzer = InSARDeformationAnalyzer()
    print(f"✓ Analyzer initialized on device: {analyzer.device}")

    # Create synthetic InSAR data
    print("\nCreating synthetic InSAR data...")
    metadata = {"sensor": "Sentinel-1", "track": "T064", "demo": True}
    insar_data = load_insar_data("demo.ti", metadata)
    print(f"✓ Data shape: {insar_data.interferogram.shape}")
    print(f"✓ Wavelength: {insar_data.wavelength} m")
    print(
        f"✓ Acquisition period: {insar_data.acquisition_dates[0]} to {insar_data.acquisition_dates[1]}"
    )

    # Process the data
    print("\nProcessing InSAR data...")
    print("  - Applying noise reduction...")
    print("  - Correcting atmospheric effects...")
    print("  - Converting phase to displacement...")
    print("  - Estimating uncertainties...")
    print("  - Classifying deformation patterns...")

    result = analyzer.process_insar_data(insar_data)

    # Display results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)

    print(f"Processing Date: {result.processing_metadata['processing_date']}")
    print(f"Device Used: {result.processing_metadata['device_used']}")

    print("\nDisplacement Analysis:")
    print(f"  Max Displacement: {np.nanmax(np.abs(result.displacement_map)):.3f} m")
    print(f"  Min Displacement: {np.nanmin(result.displacement_map):.3f} m")
    print(f"  Mean Displacement: {np.nanmean(result.displacement_map):.3f} m")
    print(
        f"  RMS Displacement: {np.sqrt(np.nanmean(result.displacement_map**2)):.3f} m"
    )

    print("\nUncertainty Analysis:")
    print(f"  Mean Uncertainty: {np.nanmean(result.uncertainty_map):.3f} m")
    print(f"  Max Uncertainty: {np.nanmax(result.uncertainty_map):.3f} m")
    print(f"  Signal-to-Noise Ratio: {result.quality_metrics['displacement_snr']:.2f}")

    print("\nData Quality Metrics:")
    print(f"  Mean Coherence: {result.quality_metrics['mean_coherence']:.3f}")
    print(
        f"  High Coherence Fraction: {result.quality_metrics['high_coherence_fraction']:.3f}"
    )
    print(
        f"  Overall Quality Score: {result.quality_metrics['overall_quality_score']:.3f}"
    )
    print(f"  Spatial Coverage: {result.quality_metrics['spatial_coverage']:.3f}")

    print(f"\nDeformation Patterns Detected: {len(result.deformation_patterns)}")

    # Show only significant patterns
    significant_patterns = [
        p for p in result.deformation_patterns if p.get("confidence", 0) > 0.7
    ]

    if significant_patterns:
        print("\nSignificant Patterns (confidence > 0.7):")
        for i, pattern in enumerate(significant_patterns[:5]):  # Show max 5
            print(
                f"  {i+1}. {pattern.get('pattern_type', 'unknown').replace('_', ' ').title()}"
            )
            print(f"     Confidence: {pattern.get('confidence', 0.0):.3f}")
            if "magnitude_m" in pattern:
                print(f"     Magnitude: {pattern['magnitude_m']:.3f} m")
            if "location" in pattern:
                print(
                    f"     Location: ({pattern['location'][0]:.1f}, {pattern['location'][1]:.1f})"
                )
    else:
        print("  No high-confidence patterns detected")

    print("\n" + "=" * 50)
    print("CAPABILITIES DEMONSTRATED")
    print("=" * 50)
    print("✓ Deep learning-based atmospheric correction")
    print("✓ Noise reduction using residual networks")
    print("✓ Phase to displacement conversion")
    print("✓ Uncertainty quantification")
    print("✓ Deformation pattern recognition")
    print("✓ Quality assessment")
    print("✓ Geometric analysis")

    print("\nDemo completed successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

import numpy as np
import torch.nn.functional as F

        traceback.print_exc()
        exit(1)
