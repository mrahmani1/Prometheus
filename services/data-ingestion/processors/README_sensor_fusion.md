# Multi-Sensor Data Fusion for Volcanic Monitoring

This module implements advanced multi-sensor data fusion algorithms for the Prometheus AI Geodynamic Platform's Volcano Watchtower real-time monitoring system.

## Overview

The sensor fusion system combines data from multiple sensor types to provide improved accuracy, reduced uncertainty, and enhanced detection capabilities for volcanic monitoring. It implements both Kalman filter-based fusion and weighted average fusion methods.

## Features

### Core Algorithms

1. **Kalman Filter Sensor Fusion** (`KalmanSensorFusion`)
   - Physics-informed state estimation
   - Automatic outlier detection and rejection
   - Temporal alignment of asynchronous data streams
   - Spatial clustering for multi-location analysis
   - Quality-based weighting of sensor data
   - Uncertainty quantification and reduction

2. **Weighted Average Fusion** (`WeightedSensorFusion`)
   - Distance-based weight decay
   - Quality-based sensor weighting
   - Fast processing for high-throughput scenarios
   - Configurable sensor type priorities

3. **Integration Layer** (`VolcanoWatchtowerFusion`)
   - Real-time stream processing integration
   - Background fusion processing
   - Alert generation based on fusion results
   - Statistics and performance monitoring

### Supported Sensor Types

- **Seismic Events**: Earthquake detection and magnitude estimation
- **Thermal Measurements**: Temperature anomaly detection
- **Deformation Measurements**: Ground displacement tracking (GPS, InSAR)
- **Strain Measurements**: Distributed Acoustic Sensing (DAS) data
- **Atmospheric Measurements**: Pressure, temperature, humidity, gas concentrations

## Implementation Details

### Task 7.1 Requirements Implementation

This implementation fulfills all requirements specified in task 7.1:

#### ✅ Kalman Filtering for Sensor Data Integration
- Implemented full Kalman filter with prediction and update steps
- Handles multi-dimensional state vectors for different sensor types
- Automatic covariance matrix management and numerical stability
- Physics-informed constraints through process noise modeling

#### ✅ Weighted Fusion Algorithms for Different Sensor Types
- Configurable sensor type weights based on reliability
- Distance-based weight decay for spatial correlation
- Quality-based weighting using data quality indicators
- Uncertainty-based weighting (inverse uncertainty weighting)

#### ✅ Temporal Alignment Systems for Asynchronous Data Streams
- Configurable temporal windows for data alignment
- Automatic reference time calculation
- Temporal offset tracking and compensation
- Support for different sampling rates across sensor types

#### ✅ Data Quality Weighting and Outlier Rejection Methods
- Mahalanobis distance-based outlier detection
- Chi-square statistical testing for outlier significance
- Quality-based uncertainty adjustment
- Configurable outlier detection thresholds
- Graceful handling of singular covariance matrices

## Usage Examples

### Basic Kalman Fusion

```python
from processors.sensor_fusion import MultiSensorDataFusion
from models.geological_data import SeismicEvent, ThermalMeasurement

# Configure fusion system
config = {
    'temporal_window': 300,    # 5 minutes
    'spatial_threshold': 50.0, # 50 km
    'outlier_threshold': 3.0   # 3-sigma outlier detection
}

fusion_system = MultiSensorDataFusion(config)

# Prepare geological data
geological_data = [seismic_events, thermal_measurements, deformation_data]

# Perform fusion
fusion_results = fusion_system.fuse_sensors(
    geological_data,
    fusion_method='kalman',
    reference_location=volcano_location,
    reference_time=datetime.now()
)

# Analyze results
for result in fusion_results:
    print(f"Fusion quality: {result.fusion_quality:.3f}")
    print(f"Confidence level: {result.confidence_level:.3f}")
    print(f"Contributing sensors: {result.contributing_sensors}")
```

### Real-time Integration

```python
from processors.fusion_integration import VolcanoWatchtowerFusion

# Initialize real-time fusion
fusion_config = {
    'fusion_temporal_window': 300,
    'fusion_spatial_threshold': 25.0,
    'fusion_interval': 60,  # Fusion every 60 seconds
    'min_sensors_for_fusion': 2
}

watchtower_fusion = VolcanoWatchtowerFusion(fusion_config)

# Start background processing
await watchtower_fusion.start_fusion_processing()

# Add sensor data as it arrives
await watchtower_fusion.add_seismic_event(seismic_data)
await watchtower_fusion.add_thermal_measurement(thermal_data)

# Get fusion statistics
stats = watchtower_fusion.get_fusion_statistics()
print(f"Success rate: {stats['success_rate']:.2f}")
```

## Configuration Parameters

### Kalman Filter Parameters

- `temporal_window`: Time window for data alignment (seconds)
- `spatial_threshold`: Maximum distance for spatial clustering (km)
- `outlier_threshold`: Mahalanobis distance threshold for outlier detection
- `process_noise`: Process noise parameters by sensor type
- `measurement_noise`: Measurement noise parameters by sensor type

### Weighted Fusion Parameters

- `sensor_weights`: Base weights for different sensor types
- `distance_decay`: Distance-based weight decay rates
- `quality_weights`: Weights for different data quality levels

### Integration Parameters

- `fusion_interval`: Background fusion processing interval (seconds)
- `min_sensors_for_fusion`: Minimum sensors required for fusion
- `fusion_method`: 'kalman' or 'weighted'
- `buffer_sizes`: Buffer sizes for different sensor types

## Performance Characteristics

### Kalman Filter Fusion
- **Accuracy**: High - optimal state estimation under Gaussian assumptions
- **Computational Cost**: Moderate - O(n³) for covariance updates
- **Memory Usage**: Moderate - stores full covariance matrices
- **Outlier Handling**: Excellent - statistical outlier detection
- **Real-time Capability**: Good - suitable for monitoring applications

### Weighted Average Fusion
- **Accuracy**: Good - simple but effective for high-quality data
- **Computational Cost**: Low - O(n) linear operations
- **Memory Usage**: Low - minimal state storage
- **Outlier Handling**: Limited - relies on quality indicators
- **Real-time Capability**: Excellent - very fast processing

## Testing

The implementation includes comprehensive tests covering:

- Unit tests for individual fusion algorithms
- Integration tests with geological data models
- Performance tests with large datasets
- Volcanic monitoring scenario simulations
- Error handling and edge cases

Run tests with:
```bash
python -m pytest tests/test_sensor_fusion.py -v
```

## Demonstration

A complete demonstration is available:
```bash
python examples/sensor_fusion_demo.py
```

This demonstrates:
- Volcanic monitoring scenario simulation
- Both Kalman and weighted fusion methods
- Sensor contribution analysis
- Alert generation based on fusion results

## Scientific Validation

The algorithms are validated against:
- Known volcanic eruption sequences
- Synthetic data with ground truth
- Cross-validation with independent monitoring data
- Comparison with published research results

## Future Enhancements

Potential improvements include:
- Extended Kalman Filter (EKF) for nonlinear systems
- Particle filter implementation for non-Gaussian distributions
- Machine learning-based sensor reliability assessment
- Adaptive fusion parameter tuning
- Integration with physics-informed neural networks

## References

1. Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. Bar-Shalom, Y., Li, X.R., Kirubarajan, T. (2001). "Estimation with Applications to Tracking and Navigation"
3. Hall, D.L., Llinas, J. (2001). "Handbook of Multisensor Data Fusion"
4. Volcanic monitoring best practices from USGS Volcano Hazards Program
5. InSAR and GPS integration techniques for volcanic deformation monitoring