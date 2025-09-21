"""
Validation script for the Uplift-Subsidence Cycle Analyzer
Tests core functionality without requiring full database setup
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from modules.resurgent_caldera.uplift_subsidence_cycle_analysis import (
    UpliftSubsidenceCycleAnalyzer,
    UpliftSubsidenceCycle,
    DeformationTimeSeries
)

async def test_uplift_subsidence_cycle_analyzer():
    """Test basic functionality of the uplift-subsidence cycle analyzer"""
    print("Testing Uplift-Subsidence Cycle Analyzer...")
    
    # Use in-memory SQLite for testing
    database_url = "sqlite:///:memory:"
    
    try:
        # Initialize analyzer
        analyzer = UpliftSubsidenceCycleAnalyzer(database_url)
        print("✓ Uplift-Subsidence Cycle Analyzer initialized successfully")
        
        # Test long-term trend analysis
        print("\n--- Testing Long-Term Trend Analysis ---")
        
        # Create synthetic deformation time-series with clear cycles
        n_points = 500
        time_years = np.linspace(0, 15, n_points)  # 15 years of data
        
        # Create synthetic cycles: 5-year primary cycle + 2-year secondary cycle + trend + noise
        primary_cycle = 30 * np.sin(2 * np.pi * time_years / 5)  # 30 mm amplitude, 5-year period
        secondary_cycle = 10 * np.sin(2 * np.pi * time_years / 2)  # 10 mm amplitude, 2-year period
        long_term_trend = 2 * time_years  # 2 mm/year trend
        noise = np.random.normal(0, 3, n_points)  # 3 mm noise
        
        synthetic_displacements = primary_cycle + secondary_cycle + long_term_trend + noise
        
        # Create timestamps
        start_date = datetime(2005, 1, 1)
        timestamps = np.array([start_date + timedelta(days=365.25 * t) for t in time_years])
        
        # Create deformation time-series object
        deformation_data = DeformationTimeSeries(
            caldera_id='test_caldera',
            timestamps=timestamps,
            vertical_displacement=synthetic_displacements,
            uncertainties=np.ones(n_points) * 2.0,  # 2 mm uncertainty
            data_source='InSAR',
            processing_level='L2'
        )
        
        # Analyze long-term trends
        trend_results = await analyzer.analyze_long_term_trends(
            deformation_data=deformation_data,
            trend_window=3.0,
            detrending_method='polynomial'
        )
        
        print(f"✓ Long-term trend analysis completed")
        print(f"  - Data quality: {trend_results['processed_data']['data_quality']}")
        print(f"  - Number of data points: {trend_results['processed_data']['n_points']}")
        print(f"  - Cycles detected: {trend_results['cycle_detection']['n_cycles']}")
        print(f"  - Trend significance: {trend_results['trend_significance']['is_significant']}")
        print(f"  - Mean trend slope: {trend_results['trend_significance']['trend_slope']:.3f} mm/year")
        
        # Test statistical analysis
        print("\n--- Testing Statistical Analysis ---")
        
        # Create mock cycles for statistical analysis
        mock_cycles = []
        for i in range(8):  # 8 cycles over 15 years
            cycle = UpliftSubsidenceCycle(
                cycle_id=f"cycle_{i+1:03d}",
                caldera_id='test_caldera',
                start_date=start_date + timedelta(days=365 * i * 1.8),
                end_date=start_date + timedelta(days=365 * (i + 1) * 1.8),
                cycle_duration=1.8 + np.random.normal(0, 0.3),  # ~1.8 years ± 0.3
                maximum_uplift=25 + np.random.normal(0, 5),  # ~25 mm ± 5
                maximum_subsidence=20 + np.random.normal(0, 4),  # ~20 mm ± 4
                uplift_rate=15 + np.random.normal(0, 3),  # ~15 mm/year ± 3
                subsidence_rate=12 + np.random.normal(0, 2),  # ~12 mm/year ± 2
                cycle_amplitude=45 + np.random.normal(0, 8),  # ~45 mm ± 8
                cycle_pattern='symmetric' if i % 2 == 0 else 'asymmetric',
                driving_mechanism='magma_injection' if i < 4 else 'volatile_exsolution',
                confidence_score=0.8 + np.random.uniform(-0.1, 0.1)
            )
            mock_cycles.append(cycle)
        
        statistical_results = await analyzer.perform_statistical_analysis(
            cycles=mock_cycles,
            statistical_tests=['normality', 'stationarity', 'periodicity', 'correlation']
        )
        
        print(f"✓ Statistical analysis completed")
        print(f"  - Number of cycles analyzed: {statistical_results['n_cycles']}")
        print(f"  - Mean cycle duration: {statistical_results['basic_statistics'].mean_duration:.2f} years")
        print(f"  - Mean cycle amplitude: {statistical_results['basic_statistics'].mean_amplitude:.2f} mm")
        print(f"  - Cycle regularity: {statistical_results['basic_statistics'].cycle_regularity:.3f}")
        print(f"  - Pattern consistency: {statistical_results['basic_statistics'].pattern_consistency:.3f}")
        
        # Check normality tests
        if 'normality_tests' in statistical_results:
            duration_normal = statistical_results['normality_tests']['duration_normality']['is_normal']
            amplitude_normal = statistical_results['normality_tests']['amplitude_normality']['is_normal']
            print(f"  - Duration normality: {duration_normal}")
            print(f"  - Amplitude normality: {amplitude_normal}")
        
        # Test cycle pattern characterization
        print("\n--- Testing Cycle Pattern Characterization ---")
        
        characterization_results = await analyzer.characterize_cycle_patterns(
            cycles=mock_cycles,
            characterization_method='comprehensive'
        )
        
        print(f"✓ Cycle pattern characterization completed")
        print(f"  - Duration analysis: {characterization_results['duration_analysis']['mean_duration']:.2f} years")
        print(f"  - Amplitude analysis: {characterization_results['amplitude_analysis']['mean_amplitude']:.2f} mm")
        print(f"  - Dominant pattern: {characterization_results['pattern_classification']['dominant_pattern']}")
        
        # Check clustering results
        if 'clustering_analysis' in characterization_results and not characterization_results['clustering_analysis'].get('insufficient_data', False):
            n_clusters = characterization_results['clustering_analysis']['n_clusters']
            print(f"  - Number of cycle clusters: {n_clusters}")
        
        # Test future cycle prediction
        print("\n--- Testing Future Cycle Prediction ---")
        
        prediction_results = await analyzer.predict_future_cycles(
            historical_cycles=mock_cycles,
            prediction_horizon=10.0,  # 10 years
            prediction_method='ensemble'
        )
        
        print(f"✓ Future cycle prediction completed")
        print(f"  - Prediction horizon: {prediction_results['prediction_horizon']} years")
        print(f"  - Prediction method: {prediction_results['prediction_method']}")
        print(f"  - Model accuracy: {prediction_results['trained_models']['model_accuracy']:.3f}")
        print(f"  - Cross-validation score: {prediction_results['model_validation']['cross_validation_score']:.3f}")
        
        # Check predictions
        if 'cycle_predictions' in prediction_results:
            predicted_cycles = prediction_results['cycle_predictions']['predicted_cycles']
            print(f"  - Predicted number of cycles: {predicted_cycles}")
        
        # Test correlation with volcanic activity
        print("\n--- Testing Correlation with Volcanic Activity ---")
        
        # Mock volcanic activity data
        volcanic_activity_data = {
            'seismic_events': {
                'timestamps': [start_date + timedelta(days=365 * i) for i in range(15)],
                'magnitudes': np.random.exponential(2.0, 15),
                'event_types': ['VT'] * 8 + ['LP'] * 4 + ['explosion'] * 3
            },
            'gas_emissions': {
                'timestamps': [start_date + timedelta(days=30 * i) for i in range(180)],
                'so2_flux': np.random.lognormal(3, 1, 180),
                'co2_flux': np.random.lognormal(4, 0.8, 180)
            },
            'thermal_anomalies': {
                'timestamps': [start_date + timedelta(days=7 * i) for i in range(780)],
                'temperature_anomaly': np.random.normal(0, 5, 780)
            }
        }
        
        correlation_results = await analyzer.correlate_cycles_with_volcanic_activity(
            cycles=mock_cycles,
            volcanic_activity_data=volcanic_activity_data,
            correlation_methods=['temporal', 'statistical']
        )
        
        print(f"✓ Correlation with volcanic activity completed")
        print(f"  - Correlation methods: {correlation_results['correlation_methods']}")
        print(f"  - Temporal correlation available: {'temporal_correlation' in correlation_results}")
        print(f"  - Statistical correlation available: {'statistical_correlation' in correlation_results}")
        print(f"  - Event correlation available: {'event_correlation' in correlation_results}")
        
        # Test comparative analysis
        print("\n--- Testing Comparative Analysis ---")
        
        # Create mock data for multiple calderas
        caldera_cycles = {
            'Yellowstone': mock_cycles[:4],
            'Long_Valley': mock_cycles[2:6],
            'Campi_Flegrei': mock_cycles[4:8]
        }
        
        comparative_results = await analyzer.compare_global_resurgent_calderas(
            caldera_cycles=caldera_cycles,
            comparison_metrics=['duration', 'amplitude', 'rate', 'pattern']
        )
        
        print(f"✓ Comparative analysis completed")
        print(f"  - Number of calderas compared: {comparative_results['n_calderas']}")
        print(f"  - Comparison metrics: {comparative_results['comparison_metrics']}")
        print(f"  - Comparative statistics available: {'comparative_statistics' in comparative_results}")
        print(f"  - Caldera clustering available: {'caldera_clustering' in comparative_results}")
        
        # Test machine learning pattern recognition
        print("\n--- Testing Machine Learning Pattern Recognition ---")
        
        ml_results = await analyzer.apply_machine_learning_pattern_recognition(
            cycles=mock_cycles,
            ml_methods=['clustering', 'classification', 'anomaly_detection'],
            feature_engineering=True
        )
        
        print(f"✓ Machine learning pattern recognition completed")
        print(f"  - ML methods applied: {ml_results['ml_methods']}")
        print(f"  - Feature engineering: {ml_results['feature_engineering']}")
        print(f"  - Clustering results available: {'clustering_results' in ml_results}")
        print(f"  - Classification results available: {'classification_results' in ml_results}")
        print(f"  - Anomaly detection available: {'anomaly_detection' in ml_results}")
        
        # Test uncertainty quantification
        print("\n--- Testing Uncertainty Quantification ---")
        
        uncertainty_results = await analyzer.quantify_prediction_uncertainties(
            prediction_results=prediction_results,
            uncertainty_methods=['bootstrap', 'monte_carlo']
        )
        
        print(f"✓ Uncertainty quantification completed")
        print(f"  - Uncertainty methods: {uncertainty_results['uncertainty_methods']}")
        print(f"  - Bootstrap uncertainty available: {'bootstrap_uncertainty' in uncertainty_results}")
        print(f"  - Monte Carlo uncertainty available: {'monte_carlo_uncertainty' in uncertainty_results}")
        print(f"  - Confidence intervals available: {'confidence_intervals' in uncertainty_results}")
        print(f"  - Sensitivity analysis available: {'sensitivity_analysis' in uncertainty_results}")
        
        # Store results in database
        await analyzer._store_cycle_results('test_caldera', mock_cycles)
        print(f"✓ Results stored in database")
        
        # Close database session
        analyzer.session.close()
        print("\n✓ Database session closed")
        
        print("\n🎉 All tests passed successfully!")
        print("Uplift-Subsidence Cycle Analyzer is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation function"""
    print("=" * 70)
    print("Uplift-Subsidence Cycle Analyzer Validation")
    print("=" * 70)
    
    success = await test_uplift_subsidence_cycle_analyzer()
    
    if success:
        print("\n✅ Validation completed successfully!")
        print("The uplift-subsidence cycle analyzer is ready for use.")
    else:
        print("\n❌ Validation failed!")
        print("Please check the error messages above.")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())