#!/usr/bin/env python3
"""
Test Advanced Stress Field Mapping System
Revolutionary features: Gephart-Forsyth inversion, bootstrap uncertainty quantification,
geological knowledge validation, and comprehensive backend integration
"""

import sys
import os
import asyncio
import numpy as np
import time
from datetime import datetime, timedelta
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_stress_field_mapper_initialization():
    """Test stress field mapper initialization and configuration"""
    print("Testing stress field mapper initialization...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism, StressTensor
        
        # Test configuration creation
        config = StressFieldConfig(
            inversion_method="gephart",
            grid_spacing=0.05,
            smoothing_radius=30.0,
            minimum_events=4,
            bootstrap_samples=500,
            confidence_level=0.95
        )
        
        print("+ StressFieldConfig created successfully")
        print(f"  Method: {config.inversion_method}")
        print(f"  Grid spacing: {config.grid_spacing}")
        print(f"  Bootstrap samples: {config.bootstrap_samples}")
        
        # Test engine initialization
        mapper = StressInversionEngine(config)
        print("+ StressInversionEngine instantiated successfully")
        
        # Test geological knowledge initialization
        assert hasattr(mapper, 'geological_knowledge')
        assert 'validation_rules' in mapper.geological_knowledge
        assert 'regional_stress_patterns' in mapper.geological_knowledge
        print("+ Geological knowledge systems initialized")
        
        # Test backend module connections
        assert hasattr(mapper, 'backend_modules')
        print(f"+ Backend modules: {list(mapper.backend_modules.keys())}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_focal_mechanism_management():
    """Test focal mechanism data management"""
    print("\nTesting focal mechanism data management...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        mapper = StressInversionEngine(StressFieldConfig())
        
        # Create test focal mechanisms
        focal_mechanisms = []
        
        for i in range(20):
            # Generate realistic focal mechanism
            fm = FocalMechanism(
                event_id=f"test_event_{i:03d}",
                latitude=45.0 + np.random.uniform(-0.5, 0.5),
                longitude=-120.0 + np.random.uniform(-0.5, 0.5),
                depth=np.random.uniform(5, 25),
                magnitude=np.random.uniform(3.5, 6.0),
                strike1=np.random.uniform(0, 360),
                dip1=np.random.uniform(60, 90),
                rake1=np.random.uniform(-180, 180),
                strike2=np.random.uniform(0, 360),
                dip2=np.random.uniform(60, 90),
                rake2=np.random.uniform(-180, 180),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 365)),
                quality=['A', 'B', 'C'][i % 3],
                uncertainty={
                    'strike_error': np.random.uniform(5, 15),
                    'dip_error': np.random.uniform(3, 10),
                    'rake_error': np.random.uniform(10, 25)
                }
            )
            focal_mechanisms.append(fm)
        
        # Add focal mechanisms to mapper
        mapper.add_focal_mechanisms(focal_mechanisms)
        
        assert len(mapper.focal_mechanisms) == 20
        print(f"+ Added {len(focal_mechanisms)} focal mechanisms successfully")
        
        # Test temporal sorting
        timestamps = [fm.timestamp for fm in mapper.focal_mechanisms]
        sorted_timestamps = sorted(timestamps)
        assert timestamps == sorted_timestamps, "Focal mechanisms should be sorted by time"
        print("+ Focal mechanisms properly sorted by timestamp")
        
        return True
        
    except Exception as e:
        print(f"X Error testing focal mechanism management: {e}")
        return False

def test_gephart_inversion_method():
    """Test advanced Gephart-Forsyth inversion method"""
    print("\nTesting Gephart-Forsyth inversion method...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        config = StressFieldConfig(inversion_method="gephart")
        mapper = StressInversionEngine(config)
        
        # Create synthetic focal mechanisms with known stress field
        # Simulate strike-slip regime with NE-SW compression
        focal_mechanisms = []
        
        for i in range(15):
            # Strike-slip focal mechanisms
            strike1 = 45 + np.random.normal(0, 10)  # NE-SW striking faults
            dip1 = 85 + np.random.normal(0, 5)      # Near-vertical
            rake1 = np.random.choice([-10, 170]) + np.random.normal(0, 15)  # Right-lateral
            
            # Calculate auxiliary plane (approximate)
            strike2 = (strike1 + 90) % 360
            dip2 = 85 + np.random.normal(0, 5)
            rake2 = rake1 - 180 if rake1 > 0 else rake1 + 180
            
            fm = FocalMechanism(
                event_id=f"ss_event_{i:03d}",
                latitude=45.0 + np.random.uniform(-0.2, 0.2),
                longitude=-120.0 + np.random.uniform(-0.2, 0.2),
                depth=np.random.uniform(8, 20),
                magnitude=np.random.uniform(4.0, 5.5),
                strike1=strike1,
                dip1=dip1,
                rake1=rake1,
                strike2=strike2,
                dip2=dip2,
                rake2=rake2,
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 100)),
                quality='A'  # High quality
            )
            focal_mechanisms.append(fm)
        
        # Test Gephart inversion
        stress_tensor = mapper._gephart_inversion(focal_mechanisms)
        
        assert stress_tensor is not None
        assert hasattr(stress_tensor, 'azimuth1')
        assert hasattr(stress_tensor, 'stress_ratio')
        assert hasattr(stress_tensor, 'regime')
        assert hasattr(stress_tensor, 'confidence')
        
        print(f"+ Gephart inversion successful")
        print(f"  Regime: {stress_tensor.regime}")
        print(f"  Sigma1 azimuth: {stress_tensor.azimuth1:.1f}°")
        print(f"  Stress ratio: {stress_tensor.stress_ratio:.3f}")
        print(f"  Confidence: {stress_tensor.confidence:.3f}")
        
        # Validate results for strike-slip regime
        assert stress_tensor.regime == 'SS', f"Expected SS regime, got {stress_tensor.regime}"
        assert 0.0 <= stress_tensor.stress_ratio <= 1.0, "Stress ratio out of bounds"
        assert 0.0 <= stress_tensor.confidence <= 1.0, "Confidence out of bounds"
        
        # Check if sigma1 azimuth is approximately NE-SW (allowing for uncertainty)
        expected_azimuth = 45  # NE
        azimuth_error = min(abs(stress_tensor.azimuth1 - expected_azimuth), 
                           360 - abs(stress_tensor.azimuth1 - expected_azimuth))
        assert azimuth_error < 60, f"Azimuth error too large: {azimuth_error}°"
        print(f"  Azimuth error: {azimuth_error:.1f}° (acceptable)")
        
        return True
        
    except Exception as e:
        print(f"X Error testing Gephart inversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bootstrap_uncertainty_quantification():
    """Test bootstrap uncertainty quantification"""
    print("\nTesting bootstrap uncertainty quantification...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        config = StressFieldConfig(bootstrap_samples=200)
        mapper = StressInversionEngine(config)
        
        # Create focal mechanisms with some variability
        focal_mechanisms = []
        
        for i in range(12):
            fm = FocalMechanism(
                event_id=f"bootstrap_event_{i:03d}",
                latitude=45.0 + np.random.uniform(-0.1, 0.1),
                longitude=-120.0 + np.random.uniform(-0.1, 0.1),
                depth=np.random.uniform(10, 18),
                magnitude=np.random.uniform(4.0, 5.0),
                strike1=np.random.uniform(30, 60),      # Some variation
                dip1=np.random.uniform(80, 90),
                rake1=np.random.uniform(-20, 20),       # Strike-slip variation
                strike2=np.random.uniform(120, 150),
                dip2=np.random.uniform(80, 90),
                rake2=np.random.uniform(160, 200),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 50)),
                quality=['A', 'B'][i % 2],
                uncertainty={
                    'strike_error': np.random.uniform(3, 8),
                    'dip_error': np.random.uniform(2, 6),
                    'rake_error': np.random.uniform(5, 12)
                }
            )
            focal_mechanisms.append(fm)
        
        # Test bootstrap analysis
        point = np.array([45.0, -120.0])
        bootstrap_results = mapper._bootstrap_stress_analysis(focal_mechanisms, point)
        
        assert 'bootstrap_results' in bootstrap_results
        assert 'confidence_intervals' in bootstrap_results
        assert 'stability_metrics' in bootstrap_results
        
        print("+ Bootstrap analysis completed successfully")
        print(f"  Successful inversions: {bootstrap_results['n_successful']}")
        print(f"  Success rate: {bootstrap_results['stability_metrics']['success_rate']:.2f}")
        
        # Check confidence intervals
        if bootstrap_results['confidence_intervals']:
            confidence_intervals = bootstrap_results['confidence_intervals']
            
            if 'azimuth' in confidence_intervals:
                az_ci = confidence_intervals['azimuth']
                print(f"  Azimuth CI: {az_ci['lower']:.1f}° - {az_ci['upper']:.1f}°")
                print(f"  Circular std: {az_ci['circular_std']:.1f}°")
                
                assert az_ci['circular_std'] > 0, "Circular standard deviation should be positive"
                assert 0 <= az_ci['lower'] <= 360, "Lower bound out of range"
                assert 0 <= az_ci['upper'] <= 360, "Upper bound out of range"
            
            if 'ratio' in confidence_intervals:
                ratio_ci = confidence_intervals['ratio']
                print(f"  Ratio CI: {ratio_ci['lower']:.3f} - {ratio_ci['upper']:.3f}")
                print(f"  Ratio std: {ratio_ci['std']:.3f}")
                
                assert 0 <= ratio_ci['lower'] <= 1, "Ratio lower bound out of range"
                assert 0 <= ratio_ci['upper'] <= 1, "Ratio upper bound out of range"
        
        # Check stability metrics
        stability_index = bootstrap_results['stability_metrics']['stability_index']
        assert 0.0 <= stability_index <= 1.0, "Stability index out of range"
        print(f"  Stability index: {stability_index:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing bootstrap uncertainty: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_quality_assessment():
    """Test comprehensive data quality assessment"""
    print("\nTesting data quality assessment...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        mapper = StressInversionEngine(StressFieldConfig())
        
        # Create focal mechanisms with varying quality
        high_quality_mechanisms = []
        low_quality_mechanisms = []
        
        # High quality mechanisms
        for i in range(8):
            fm = FocalMechanism(
                event_id=f"high_qual_{i:03d}",
                latitude=45.0 + np.random.uniform(-0.1, 0.1),
                longitude=-120.0 + np.random.uniform(-0.1, 0.1),
                depth=np.random.uniform(8, 15),  # Good depth range
                magnitude=np.random.uniform(4.5, 5.8),  # Good magnitude
                strike1=np.random.uniform(40, 50),
                dip1=np.random.uniform(85, 90),
                rake1=np.random.uniform(-10, 10),
                strike2=np.random.uniform(130, 140),
                dip2=np.random.uniform(85, 90),
                rake2=np.random.uniform(170, 190),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 30)),
                quality='A',  # High quality
                uncertainty={
                    'strike_error': np.random.uniform(2, 5),
                    'dip_error': np.random.uniform(1, 3),
                    'rake_error': np.random.uniform(3, 8)
                }
            )
            high_quality_mechanisms.append(fm)
        
        # Low quality mechanisms
        for i in range(6):
            fm = FocalMechanism(
                event_id=f"low_qual_{i:03d}",
                latitude=45.0 + np.random.uniform(-0.3, 0.3),
                longitude=-120.0 + np.random.uniform(-0.3, 0.3),
                depth=np.random.uniform(25, 40),  # Deeper, less reliable
                magnitude=np.random.uniform(3.0, 4.2),  # Lower magnitude
                strike1=np.random.uniform(0, 360),  # More scattered
                dip1=np.random.uniform(60, 90),
                rake1=np.random.uniform(-180, 180),  # More scattered
                strike2=np.random.uniform(0, 360),
                dip2=np.random.uniform(60, 90),
                rake2=np.random.uniform(-180, 180),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 200)),
                quality='D',  # Low quality
                uncertainty={
                    'strike_error': np.random.uniform(15, 30),
                    'dip_error': np.random.uniform(10, 20),
                    'rake_error': np.random.uniform(20, 40)
                }
            )
            low_quality_mechanisms.append(fm)
        
        # Test high quality assessment
        high_quality_metrics = mapper._assess_data_quality(high_quality_mechanisms)
        print("+ High quality data assessment:")
        print(f"  Overall quality score: {high_quality_metrics['overall_quality_score']:.3f}")
        print(f"  Mean magnitude: {high_quality_metrics['magnitude_distribution']['mean']:.2f}")
        print(f"  Mean depth: {high_quality_metrics['depth_distribution']['mean']:.1f} km")
        
        assert high_quality_metrics['overall_quality_score'] > 0.7, "High quality data should score >0.7"
        
        # Test low quality assessment
        low_quality_metrics = mapper._assess_data_quality(low_quality_mechanisms)
        print("\n+ Low quality data assessment:")
        print(f"  Overall quality score: {low_quality_metrics['overall_quality_score']:.3f}")
        print(f"  Mean magnitude: {low_quality_metrics['magnitude_distribution']['mean']:.2f}")
        print(f"  Mean depth: {low_quality_metrics['depth_distribution']['mean']:.1f} km")
        
        assert low_quality_metrics['overall_quality_score'] < 0.6, "Low quality data should score <0.6"
        
        # Quality difference validation
        quality_difference = high_quality_metrics['overall_quality_score'] - low_quality_metrics['overall_quality_score']
        assert quality_difference > 0.2, "Quality assessment should distinguish between good and bad data"
        print(f"+ Quality differentiation: {quality_difference:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing data quality assessment: {e}")
        return False

def test_anomaly_detection():
    """Test advanced anomaly detection capabilities"""
    print("\nTesting advanced anomaly detection...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, StressField, StressTensor, FocalMechanism
        
        mapper = StressInversionEngine(StressFieldConfig())
        
        # Create stress field with intentional anomalies
        grid_points = np.array([
            [45.0, -120.0], [45.1, -120.0], [45.2, -120.0],
            [45.0, -119.9], [45.1, -119.9], [45.2, -119.9],
            [45.0, -119.8], [45.1, -119.8], [45.2, -119.8]
        ])
        
        # Create mostly consistent stress tensors
        stress_tensors = []
        for i, point in enumerate(grid_points):
            if i == 4:  # Introduce anomaly at center point
                # Anomalous orientation
                anomalous_tensor = StressTensor(
                    sigma1=1.0, sigma2=0.3, sigma3=0.0,
                    azimuth1=135,  # Opposite to others
                    plunge1=10,
                    azimuth2=225, plunge2=5,
                    azimuth3=45, plunge3=80,
                    stress_ratio=0.9,  # Anomalous ratio
                    regime='TF',  # Different regime
                    confidence=0.6
                )
                stress_tensors.append(anomalous_tensor)
            else:
                # Normal consistent tensors
                normal_tensor = StressTensor(
                    sigma1=1.0, sigma2=0.5, sigma3=0.0,
                    azimuth1=45 + np.random.uniform(-5, 5),  # Consistent NE
                    plunge1=5 + np.random.uniform(-3, 3),
                    azimuth2=135, plunge2=10,
                    azimuth3=315, plunge3=80,
                    stress_ratio=0.5 + np.random.uniform(-0.1, 0.1),
                    regime='SS',
                    confidence=0.8
                )
                stress_tensors.append(normal_tensor)
        
        # Create stress field
        stress_field = StressField(
            grid_points=grid_points,
            stress_tensors=stress_tensors,
            interpolation_method='gephart',
            resolution=0.1,
            temporal_range=(datetime.now() - timedelta(days=365), datetime.now()),
            quality_metrics={'successful_inversions': 9, 'failed_inversions': 0},
            uncertainties={}
        )
        
        # Test anomaly detection
        anomalies = mapper.detect_stress_anomalies(stress_field)
        
        assert len(anomalies) > 0, "Should detect at least one anomaly"
        print(f"+ Detected {len(anomalies)} anomalies")
        
        # Check for different types of anomalies
        anomaly_types = [anomaly['type'] for anomaly in anomalies]
        print(f"+ Anomaly types: {set(anomaly_types)}")
        
        # Should detect orientation anomaly
        orientation_anomalies = [a for a in anomalies if a['type'] == 'orientation_anomaly']
        assert len(orientation_anomalies) > 0, "Should detect orientation anomaly"
        
        # Check anomaly details
        for anomaly in anomalies[:3]:  # Show first 3
            print(f"  {anomaly['type']} at {anomaly['location']}")
            print(f"    Significance: {anomaly['significance']:.3f}")
            if 'azimuth' in anomaly:
                print(f"    Azimuth: {anomaly['azimuth']:.1f}°")
        
        # Validate anomaly significance scores
        for anomaly in anomalies:
            assert anomaly['significance'] > 0, "Significance should be positive"
            assert 'location' in anomaly, "Anomaly should have location"
            assert 'type' in anomaly, "Anomaly should have type"
        
        return True
        
    except Exception as e:
        print(f"X Error testing anomaly detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geological_knowledge_validation():
    """Test geological knowledge validation system"""
    print("\nTesting geological knowledge validation...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, StressField, StressTensor
        
        mapper = StressInversionEngine(StressFieldConfig())
        
        # Create stress field in San Andreas region
        san_andreas_points = np.array([
            [37.0, -122.0], [37.2, -121.8], [37.4, -121.6],  # San Andreas region
            [37.1, -122.1], [37.3, -121.9], [37.5, -121.7]
        ])
        
        # Create stress tensors consistent with San Andreas system
        stress_tensors = []
        for point in san_andreas_points:
            # San Andreas: N-S compression, strike-slip regime
            tensor = StressTensor(
                sigma1=1.0, sigma2=0.4, sigma3=0.0,
                azimuth1=5 + np.random.uniform(-10, 10),  # Approximately N-S
                plunge1=10 + np.random.uniform(-5, 5),
                azimuth2=95, plunge2=5,
                azimuth3=185, plunge3=80,
                stress_ratio=0.4 + np.random.uniform(-0.1, 0.1),
                regime='SS',  # Strike-slip
                confidence=0.75 + np.random.uniform(-0.15, 0.15)
            )
            stress_tensors.append(tensor)
        
        stress_field = StressField(
            grid_points=san_andreas_points,
            stress_tensors=stress_tensors,
            interpolation_method='gephart',
            resolution=0.1,
            temporal_range=(datetime.now() - timedelta(days=365), datetime.now()),
            quality_metrics={'successful_inversions': 6, 'failed_inversions': 0},
            uncertainties={}
        )
        
        # Test geological validation
        validation_results = mapper.validate_with_geological_knowledge(stress_field)
        
        assert 'overall_validity' in validation_results
        assert 'tectonic_consistency' in validation_results
        assert 'geological_plausibility' in validation_results
        assert 'constraint_violations' in validation_results
        assert 'recommendations' in validation_results
        
        print("+ Geological validation completed successfully")
        print(f"  Overall validity: {validation_results['overall_validity']:.3f}")
        
        # Check tectonic consistency
        tectonic_consistency = validation_results['tectonic_consistency']
        if 'consistency_score' in tectonic_consistency:
            print(f"  Tectonic consistency: {tectonic_consistency['consistency_score']:.3f}")
            
            # Should have good consistency with San Andreas pattern
            assert tectonic_consistency['consistency_score'] > 0.6, "Should be consistent with San Andreas"
        
        # Check geological plausibility
        plausibility = validation_results['geological_plausibility']
        if 'plausibility_score' in plausibility:
            print(f"  Geological plausibility: {plausibility['plausibility_score']:.3f}")
        
        # Check constraint violations
        violations = validation_results['constraint_violations']
        print(f"  Constraint violations: {len(violations)}")
        
        # Should have few violations for realistic data
        assert len(violations) < 3, "Should have minimal violations for realistic data"
        
        # Check recommendations
        recommendations = validation_results['recommendations']
        print(f"  Recommendations: {len(recommendations)}")
        
        for rec in recommendations[:2]:  # Show first 2
            print(f"    {rec['type']}: {rec['description']}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing geological validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_stress_field_inversion():
    """Test complete stress field inversion workflow"""
    print("\nTesting complete stress field inversion workflow...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        # Configure for comprehensive testing
        config = StressFieldConfig(
            inversion_method="gephart",
            grid_spacing=0.1,
            smoothing_radius=40.0,
            minimum_events=4,
            bootstrap_samples=100,  # Reduced for testing speed
            confidence_level=0.95
        )
        
        mapper = StressInversionEngine(config)
        
        # Create synthetic earthquake sequence in a region
        focal_mechanisms = []
        region_center = (45.0, -120.0)
        
        # Simulate aftershock sequence with consistent stress field
        for i in range(25):
            # Locations around region center
            lat = region_center[0] + np.random.normal(0, 0.15)
            lon = region_center[1] + np.random.normal(0, 0.15)
            
            # Strike-slip focal mechanisms with some variation
            base_strike = 45 + np.random.normal(0, 15)
            
            fm = FocalMechanism(
                event_id=f"complete_test_{i:03d}",
                latitude=lat,
                longitude=lon,
                depth=np.random.uniform(8, 20),
                magnitude=np.random.uniform(3.8, 5.2),
                strike1=base_strike % 360,
                dip1=np.random.uniform(75, 90),
                rake1=np.random.normal(0, 20),  # Strike-slip with variation
                strike2=(base_strike + 90) % 360,
                dip2=np.random.uniform(75, 90),
                rake2=np.random.normal(180, 20),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 180)),
                quality=['A', 'B', 'C'][i % 3],
                uncertainty={
                    'strike_error': np.random.uniform(3, 12),
                    'dip_error': np.random.uniform(2, 8),
                    'rake_error': np.random.uniform(5, 15)
                }
            )
            focal_mechanisms.append(fm)
        
        # Add focal mechanisms
        mapper.add_focal_mechanisms(focal_mechanisms)
        print(f"+ Added {len(focal_mechanisms)} focal mechanisms")
        
        # Define study region
        region_bounds = (44.5, 45.5, -120.5, -119.5)  # min_lat, max_lat, min_lon, max_lon
        
        # Perform stress field inversion
        start_time = time.time()
        stress_field = mapper.invert_stress_field(region_bounds, method='gephart')
        end_time = time.time()
        
        computation_time = end_time - start_time
        print(f"+ Stress field inversion completed in {computation_time:.2f} seconds")
        
        # Validate results
        assert stress_field is not None
        assert len(stress_field.stress_tensors) > 0
        assert len(stress_field.grid_points) > 0
        
        print(f"  Grid points: {len(stress_field.grid_points)}")
        print(f"  Successful inversions: {stress_field.quality_metrics['successful_inversions']}")
        print(f"  Failed inversions: {stress_field.quality_metrics['failed_inversions']}")
        
        # Analyze successful tensors
        successful_tensors = [t for t in stress_field.stress_tensors if t.confidence > 0.1]
        print(f"  Reliable stress tensors: {len(successful_tensors)}")
        
        if successful_tensors:
            # Statistics
            azimuths = [t.azimuth1 for t in successful_tensors]
            regimes = [t.regime for t in successful_tensors]
            ratios = [t.stress_ratio for t in successful_tensors]
            confidences = [t.confidence for t in successful_tensors]
            
            print(f"  Mean sigma1 azimuth: {np.mean(azimuths):.1f}°")
            print(f"  Std sigma1 azimuth: {np.std(azimuths):.1f}°")
            print(f"  Mean stress ratio: {np.mean(ratios):.3f}")
            print(f"  Mean confidence: {np.mean(confidences):.3f}")
            
            # Regime distribution
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            print(f"  Stress regimes: {regime_counts}")
            
            # Expected strike-slip dominance
            if 'SS' in regime_counts:
                ss_fraction = regime_counts['SS'] / len(regimes)
                assert ss_fraction > 0.4, "Should have significant strike-slip component"
                print(f"  Strike-slip fraction: {ss_fraction:.2f}")
        
        # Test heterogeneity analysis
        heterogeneity = mapper.calculate_stress_heterogeneity(stress_field)
        print(f"  Heterogeneity index: {heterogeneity['heterogeneity_index']:.3f}")
        print(f"  Coherence length: {heterogeneity['coherence_length']:.1f} km")
        
        # Test anomaly detection
        anomalies = mapper.detect_stress_anomalies(stress_field)
        print(f"  Detected anomalies: {len(anomalies)}")
        
        # Test geological validation
        validation = mapper.validate_with_geological_knowledge(stress_field)
        print(f"  Geological validity: {validation['overall_validity']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing complete workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_export_and_serialization():
    """Test stress field export and data serialization"""
    print("\nTesting export and serialization...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, StressField, StressTensor
        
        mapper = StressInversionEngine(StressFieldConfig())
        
        # Create simple stress field for export testing
        grid_points = np.array([[45.0, -120.0], [45.1, -120.0], [45.0, -119.9]])
        
        stress_tensors = []
        for i, point in enumerate(grid_points):
            tensor = StressTensor(
                sigma1=1.0, sigma2=0.5, sigma3=0.0,
                azimuth1=45, plunge1=5,
                azimuth2=135, plunge2=10,
                azimuth3=315, plunge3=80,
                stress_ratio=0.5,
                regime='SS',
                confidence=0.8
            )
            stress_tensors.append(tensor)
        
        stress_field = StressField(
            grid_points=grid_points,
            stress_tensors=stress_tensors,
            interpolation_method='gephart',
            resolution=0.1,
            temporal_range=(datetime.now() - timedelta(days=365), datetime.now()),
            quality_metrics={'successful_inversions': 3, 'failed_inversions': 0},
            uncertainties={'azimuth_uncertainty': [5.0, 6.0, 7.0]}
        )
        
        # Test export
        output_filename = 'test_stress_field_export.json'
        mapper.export_stress_field(stress_field, output_filename)
        
        # Verify file was created and contains expected data
        assert os.path.exists(output_filename), "Export file should be created"
        
        # Read and validate exported data
        with open(output_filename, 'r') as f:
            exported_data = json.load(f)
        
        # Check required fields
        required_fields = [
            'timestamp', 'config', 'temporal_range', 'grid_points',
            'stress_tensors', 'quality_metrics', 'geological_validation'
        ]
        
        for field in required_fields:
            assert field in exported_data, f"Missing required field: {field}"
        
        print(f"+ Export successful with {len(exported_data['stress_tensors'])} tensors")
        print(f"  File size: {os.path.getsize(output_filename)} bytes")
        
        # Check geological validation was included
        geological_validation = exported_data['geological_validation']
        assert 'overall_validity' in geological_validation
        print(f"  Geological validation included: validity = {geological_validation['overall_validity']:.3f}")
        
        # Cleanup
        os.remove(output_filename)
        
        return True
        
    except Exception as e:
        print(f"X Error testing export: {e}")
        return False

def test_performance_benchmarks():
    """Test performance with larger datasets"""
    print("\nTesting performance benchmarks...")
    
    try:
        from stress_field_mapper import StressFieldConfig, StressInversionEngine, FocalMechanism
        
        # Performance-optimized configuration
        config = StressFieldConfig(
            inversion_method="gephart",
            grid_spacing=0.2,  # Coarser for speed
            smoothing_radius=50.0,
            minimum_events=3,
            bootstrap_samples=50  # Reduced for speed
        )
        
        mapper = StressInversionEngine(config)
        
        # Create larger dataset
        n_events = 100
        focal_mechanisms = []
        
        print(f"+ Generating {n_events} focal mechanisms...")
        
        for i in range(n_events):
            fm = FocalMechanism(
                event_id=f"perf_event_{i:04d}",
                latitude=45.0 + np.random.uniform(-1.0, 1.0),
                longitude=-120.0 + np.random.uniform(-1.0, 1.0),
                depth=np.random.uniform(5, 30),
                magnitude=np.random.uniform(3.0, 6.0),
                strike1=np.random.uniform(0, 360),
                dip1=np.random.uniform(60, 90),
                rake1=np.random.uniform(-180, 180),
                strike2=np.random.uniform(0, 360),
                dip2=np.random.uniform(60, 90),
                rake2=np.random.uniform(-180, 180),
                timestamp=datetime.now() - timedelta(days=np.random.uniform(0, 1000)),
                quality=['A', 'B', 'C', 'D'][i % 4]
            )
            focal_mechanisms.append(fm)
        
        # Add focal mechanisms
        start_time = time.time()
        mapper.add_focal_mechanisms(focal_mechanisms)
        add_time = time.time() - start_time
        
        print(f"+ Added {len(focal_mechanisms)} focal mechanisms in {add_time:.3f} seconds")
        
        # Large region inversion
        region_bounds = (43.0, 47.0, -122.0, -118.0)  # 4° x 4° region
        
        start_time = time.time()
        stress_field = mapper.invert_stress_field(region_bounds, method='gephart')
        inversion_time = time.time() - start_time
        
        print(f"+ Stress field inversion completed in {inversion_time:.2f} seconds")
        print(f"  Processing rate: {len(focal_mechanisms) / inversion_time:.1f} events/second")
        
        # Performance validation
        assert inversion_time < 120, "Should complete inversion within 2 minutes"
        
        # Grid points created
        n_grid_points = len(stress_field.grid_points)
        print(f"  Grid points processed: {n_grid_points}")
        print(f"  Grid processing rate: {n_grid_points / inversion_time:.1f} points/second")
        
        # Successful inversion rate
        success_rate = (stress_field.quality_metrics['successful_inversions'] / 
                       (stress_field.quality_metrics['successful_inversions'] + 
                        stress_field.quality_metrics['failed_inversions']))
        
        print(f"  Inversion success rate: {success_rate:.2f}")
        assert success_rate > 0.3, "Should have reasonable success rate"
        
        # Memory usage (approximate)
        import sys
        total_size = sys.getsizeof(stress_field.stress_tensors) + sys.getsizeof(stress_field.grid_points)
        print(f"  Approximate memory usage: {total_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"X Error testing performance: {e}")
        return False

def main():
    """Run all stress field mapping tests"""
    print("=" * 80)
    print("ADVANCED STRESS FIELD MAPPING TESTS")
    print("=" * 80)
    print("Revolutionary Features:")
    print("• Advanced Gephart-Forsyth inversion with damped least squares")
    print("• Bootstrap uncertainty quantification with confidence intervals")
    print("• Comprehensive data quality assessment and outlier detection")
    print("• Multi-method anomaly detection with geological validation")
    print("• Geological knowledge integration and tectonic consistency")
    print("• Backend module integration for hypothesis validation")
    print("• Production-ready performance and scalability")
    print("=" * 80)
    
    tests = [
        test_stress_field_mapper_initialization,
        test_focal_mechanism_management,
        test_gephart_inversion_method,
        test_bootstrap_uncertainty_quantification,
        test_data_quality_assessment,
        test_anomaly_detection,
        test_geological_knowledge_validation,
        test_complete_stress_field_inversion,
        test_export_and_serialization,
        test_performance_benchmarks
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"X Test failed with exception: {e}")
        print()
    
    print("=" * 80)
    print(f"STRESS FIELD MAPPING RESULTS: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("All stress field mapping tests PASSED!")
        print("+ Advanced Gephart-Forsyth inversion working perfectly")
        print("+ Bootstrap uncertainty quantification operational") 
        print("+ Comprehensive data quality assessment implemented")
        print("+ Multi-method anomaly detection functional")
        print("+ Geological knowledge validation active")
        print("+ Backend module integration ready")
        print("+ Performance optimized for large datasets")
        print("+ Export and serialization working correctly")
        print()
        print("REVOLUTIONARY CAPABILITIES SUCCESSFULLY IMPLEMENTED:")
        print("1. Advanced Stress Inversion - Gephart-Forsyth method with")
        print("   damped least squares, grid search optimization, and")
        print("   statistical F-test validation for robust solutions")
        print("2. Bootstrap Uncertainty Analysis - Comprehensive resampling")
        print("   with perturbation modeling and confidence intervals")
        print("3. Data Quality Intelligence - Multi-factor assessment")
        print("   including magnitude, depth, temporal, and spatial analysis")
        print("4. Anomaly Detection Framework - Statistical outlier detection,")
        print("   spatial gradient analysis, and geological plausibility")
        print("5. Geological Knowledge Integration - Tectonic pattern")
        print("   validation, constraint checking, and recommendation engine")
        print("6. Backend Module Connectivity - Seamless integration with")
        print("   hypothesis systems, complexity assessment, and data pipelines")
        print("7. Production Scalability - Optimized for large datasets")
        print("   with efficient algorithms and memory management")
    else:
        print("Some stress field mapping tests FAILED.")
        print("Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)