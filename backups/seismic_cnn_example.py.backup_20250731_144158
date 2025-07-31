"""
CNN-based Seismic Event Detection Example

This example demonstrates how to use the CNN-based seismic event detection
system for real-time monitoring and analysis of seismic data.
"""

import numpy as np
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Import the seismic CNN processor
try:
    from ..core.seismic_cnn_processor import (
        SeismicCNNProcessor,
        SeismicEventClassifier,
        SeismicEventType,
        SeismometerType
    )
    from ..models.geological import GeologicalLocation
    from ..models.monitoring import SensorReading, QualityFlag
except ImportError:
    # For direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.seismic_cnn_processor import (
        SeismicCNNProcessor,
        SeismicEventClassifier,
        SeismicEventType,
        SeismometerType
    )
    from models.geological import GeologicalLocation
    from models.monitoring import SensorReading, QualityFlag


def create_synthetic_seismic_data(
    duration_seconds: float = 300,  # 5 minutes
    sampling_rate: float = 100.0,
    num_events: int = 3
) -> tuple:
    """
    Create synthetic seismic data with embedded events for demonstration.
    
    Returns:
        tuple: (waveform_data, timestamps, true_events)
    """
    samples = int(duration_seconds * sampling_rate)
    
    # Initialize 3-component waveform with background noise
    waveform_data = np.random.randn(3, samples) * 0.1
    
    # Create timestamps
    start_time = datetime.utcnow()
    timestamps = np.array([
        start_time + timedelta(seconds=i/sampling_rate)
        for i in range(samples)
    ])
    
    # Add synthetic seismic events
    true_events = []
    
    for event_idx in range(num_events):
        # Random event timing
        event_start = np.random.randint(samples // 4, 3 * samples // 4)
        event_duration = np.random.randint(int(2 * sampling_rate), int(20 * sampling_rate))
        event_end = min(event_start + event_duration, samples)
        
        # Event characteristics
        if event_idx == 0:
            # Earthquake-like event
            amplitude = 2.0
            freq = 5.0  # 5 Hz dominant frequency
            event_type = SeismicEventType.EARTHQUAKE
        elif event_idx == 1:
            # Long period event
            amplitude = 1.5
            freq = 2.0  # 2 Hz dominant frequency
            event_type = SeismicEventType.LONG_PERIOD
        else:
            # Explosion-like event
            amplitude = 3.0
            freq = 15.0  # 15 Hz dominant frequency
            event_type = SeismicEventType.EXPLOSION
        
        # Generate event waveform
        t_event = np.arange(event_end - event_start) / sampling_rate
        
        # Create realistic seismic waveform with P and S phases
        p_wave = amplitude * np.exp(-t_event * 2) * np.sin(2 * np.pi * freq * t_event)
        s_wave = amplitude * 0.7 * np.exp(-(t_event - 2) * 1.5) * np.sin(2 * np.pi * freq * 0.6 * (t_event - 2))
        s_wave[t_event < 2] = 0  # S wave arrives later
        
        event_signal = p_wave + s_wave
        
        # Add to each component with different amplitudes and phases
        waveform_data[0, event_start:event_end] += event_signal  # Z component
        waveform_data[1, event_start:event_end] += event_signal * 0.8  # N component
        waveform_data[2, event_start:event_end] += event_signal * 0.6  # E component
        
        # Record true event
        true_events.append({
            'start_time': timestamps[event_start],
            'duration': (event_end - event_start) / sampling_rate,
            'event_type': event_type,
            'amplitude': amplitude,
            'frequency': freq
        })
    
    return waveform_data, timestamps, true_events


def demonstrate_basic_detection():
    """Demonstrate basic seismic event detection"""
    print("=== Basic Seismic Event Detection Demo ===")
    
    # Create synthetic data
    print("Creating synthetic seismic data...")
    waveform_data, timestamps, true_events = create_synthetic_seismic_data(
        duration_seconds=120,  # 2 minutes
        sampling_rate=100.0,
        num_events=2
    )
    
    print(f"Generated {len(true_events)} synthetic events")
    for i, event in enumerate(true_events):
        print(f"  Event {i+1}: {event['event_type'].value} at {event['start_time'].strftime('%H:%M:%S')}")
    
    # Initialize processor
    print("\nInitializing CNN processor...")
    processor = SeismicCNNProcessor(
        sampling_rate=100.0,
        window_length=30.0,
        detection_threshold=0.3,  # Lower threshold for demo
        device="cpu"
    )
    
    # Define sensor location
    sensor_location = GeologicalLocation(
        latitude=45.0,
        longitude=-122.0,
        elevation=1000.0
    )
    
    # Process data
    print("Processing seismic data...")
    detected_events = processor.process_continuous_data(
        waveform_data=waveform_data,
        timestamps=timestamps,
        sensor_location=sensor_location,
        sensor_id="DEMO_STATION"
    )
    
    print(f"\nDetected {len(detected_events)} events:")
    for i, event in enumerate(detected_events):
        print(f"  Event {i+1}:")
        print(f"    Type: {event.event_type.value}")
        print(f"    Time: {event.timestamp.strftime('%H:%M:%S')}")
        print(f"    Confidence: {event.confidence:.3f}")
        print(f"    Duration: {event.duration:.1f}s" if event.duration else "    Duration: Unknown")
        print(f"    Peak Amplitude: {event.peak_amplitude:.2f}" if event.peak_amplitude else "")
    
    # Get processing statistics
    stats = processor.get_processing_statistics()
    print(f"\nProcessing Statistics:")
    print(f"  Windows processed: {stats['total_windows_processed']}")
    print(f"  Events detected: {stats['events_detected']}")
    print(f"  Detection threshold: {stats['detection_threshold']}")
    
    return detected_events, true_events


def demonstrate_volcanic_classification():
    """Demonstrate volcanic event classification"""
    print("\n=== Volcanic Event Classification Demo ===")
    
    # Create synthetic volcanic data
    print("Creating synthetic volcanic seismic data...")
    
    # Generate different types of volcanic events
    duration_seconds = 180  # 3 minutes
    sampling_rate = 100.0
    samples = int(duration_seconds * sampling_rate)
    
    waveform_data = np.random.randn(3, samples) * 0.05  # Low background noise
    
    start_time = datetime.utcnow()
    timestamps = np.array([
        start_time + timedelta(seconds=i/sampling_rate)
        for i in range(samples)
    ])
    
    # Add different volcanic events
    volcanic_events = []
    
    # 1. Very Long Period (VLP) event
    vlp_start = int(30 * sampling_rate)
    vlp_duration = int(90 * sampling_rate)  # 90 seconds
    t_vlp = np.arange(vlp_duration) / sampling_rate
    vlp_signal = 0.8 * np.exp(-t_vlp / 30) * np.sin(2 * np.pi * 0.3 * t_vlp)  # 0.3 Hz
    waveform_data[:, vlp_start:vlp_start + vlp_duration] += vlp_signal
    
    volcanic_events.append({
        'type': 'VLP',
        'start': timestamps[vlp_start],
        'duration': vlp_duration / sampling_rate,
        'frequency': 0.3
    })
    
    # 2. Long Period (LP) event
    lp_start = int(120 * sampling_rate)
    lp_duration = int(25 * sampling_rate)  # 25 seconds
    t_lp = np.arange(lp_duration) / sampling_rate
    lp_signal = 1.2 * np.exp(-t_lp / 8) * np.sin(2 * np.pi * 2.5 * t_lp)  # 2.5 Hz
    waveform_data[:, lp_start:lp_start + lp_duration] += lp_signal
    
    volcanic_events.append({
        'type': 'LP',
        'start': timestamps[lp_start],
        'duration': lp_duration / sampling_rate,
        'frequency': 2.5
    })
    
    # 3. Explosion event
    exp_start = int(150 * sampling_rate)
    exp_duration = int(3 * sampling_rate)  # 3 seconds
    t_exp = np.arange(exp_duration) / sampling_rate
    exp_signal = 2.5 * np.exp(-t_exp * 3) * np.sin(2 * np.pi * 12 * t_exp)  # 12 Hz
    waveform_data[:, exp_start:exp_start + exp_duration] += exp_signal
    
    volcanic_events.append({
        'type': 'Explosion',
        'start': timestamps[exp_start],
        'duration': exp_duration / sampling_rate,
        'frequency': 12
    })
    
    print(f"Generated {len(volcanic_events)} volcanic events:")
    for event in volcanic_events:
        print(f"  {event['type']}: {event['frequency']:.1f} Hz, {event['duration']:.1f}s")
    
    # Process with CNN
    processor = SeismicCNNProcessor(
        sampling_rate=sampling_rate,
        window_length=30.0,
        detection_threshold=0.2,
        device="cpu"
    )
    
    sensor_location = GeologicalLocation(
        latitude=19.42,  # Kilauea coordinates
        longitude=-155.29,
        elevation=1200.0
    )
    
    detected_events = processor.process_continuous_data(
        waveform_data=waveform_data,
        timestamps=timestamps,
        sensor_location=sensor_location,
        sensor_id="KILAUEA_STATION"
    )
    
    # Apply volcanic classification
    classifier = SeismicEventClassifier()
    classified_events = classifier.classify_volcanic_events(detected_events)
    
    print(f"\nClassified {len(classified_events)} volcanic events:")
    for i, event in enumerate(classified_events):
        print(f"  Event {i+1}:")
        print(f"    Classified Type: {event.event_type.value}")
        print(f"    Time: {event.timestamp.strftime('%H:%M:%S')}")
        print(f"    Duration: {event.duration:.1f}s" if event.duration else "    Duration: Unknown")
        print(f"    Dominant Freq: {event.frequency_content.get('Z_dominant_frequency', 'N/A'):.1f} Hz")
        if 'classification_rule' in event.detection_metadata:
            print(f"    Rule: {event.detection_metadata['classification_rule']}")
    
    return classified_events


def demonstrate_multi_station_processing():
    """Demonstrate multi-station seismic processing and location estimation"""
    print("\n=== Multi-Station Processing Demo ===")
    
    # Define network of stations
    stations = {
        'STATION_A': GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0),
        'STATION_B': GeologicalLocation(latitude=45.05, longitude=-122.05, elevation=1100.0),
        'STATION_C': GeologicalLocation(latitude=44.95, longitude=-121.95, elevation=900.0),
        'STATION_D': GeologicalLocation(latitude=45.02, longitude=-122.02, elevation=1050.0)
    }
    
    print(f"Processing data from {len(stations)} stations...")
    
    # Simulate earthquake at known location
    earthquake_location = GeologicalLocation(latitude=45.01, longitude=-122.01, elevation=0.0)
    earthquake_time = datetime.utcnow()
    
    processor = SeismicCNNProcessor(
        sampling_rate=100.0,
        window_length=20.0,
        detection_threshold=0.3,
        device="cpu"
    )
    
    all_events = []
    
    for station_id, station_location in stations.items():
        # Calculate distance to earthquake
        distance = np.sqrt(
            (station_location.latitude - earthquake_location.latitude) ** 2 +
            (station_location.longitude - earthquake_location.longitude) ** 2
        ) * 111  # Rough km conversion
        
        # Create synthetic data with distance-dependent amplitude
        duration_seconds = 60
        samples = int(duration_seconds * processor.sampling_rate)
        waveform_data = np.random.randn(3, samples) * 0.1
        
        # Add earthquake signal with amplitude decay
        event_start = int(20 * processor.sampling_rate)
        event_duration = int(15 * processor.sampling_rate)
        amplitude = max(0.5, 3.0 / (1 + distance / 10))  # Amplitude decay with distance
        
        t_event = np.arange(event_duration) / processor.sampling_rate
        earthquake_signal = amplitude * np.exp(-t_event * 1.5) * np.sin(2 * np.pi * 8 * t_event)
        
        waveform_data[:, event_start:event_start + event_duration] += earthquake_signal
        
        # Create timestamps
        timestamps = np.array([
            earthquake_time + timedelta(seconds=i/processor.sampling_rate)
            for i in range(samples)
        ])
        
        # Process station data
        station_events = processor.process_continuous_data(
            waveform_data=waveform_data,
            timestamps=timestamps,
            sensor_location=station_location,
            sensor_id=station_id
        )
        
        print(f"  {station_id}: {len(station_events)} events detected (distance: {distance:.1f} km)")
        all_events.extend(station_events)
    
    # Estimate event locations
    print("\nEstimating event locations...")
    located_events = processor.estimate_location(all_events, stations)
    
    # Group events by estimated location
    location_groups = {}
    for event in located_events:
        if event.location and 'num_stations' in event.detection_metadata:
            key = f"Event_{event.location.latitude:.3f}_{event.location.longitude:.3f}"
            if key not in location_groups:
                location_groups[key] = []
            location_groups[key].append(event)
    
    print(f"\nEstimated {len(location_groups)} event locations:")
    for i, (location_key, events) in enumerate(location_groups.items()):
        if events:
            event = events[0]  # Representative event
            print(f"  Location {i+1}:")
            print(f"    Estimated: {event.location.latitude:.3f}°N, {event.location.longitude:.3f}°W")
            print(f"    Stations: {event.detection_metadata.get('num_stations', 'Unknown')}")
            print(f"    Events: {len(events)}")
            
            # Calculate error if we know true location
            if event.location:
                lat_error = abs(event.location.latitude - earthquake_location.latitude) * 111
                lon_error = abs(event.location.longitude - earthquake_location.longitude) * 111
                total_error = np.sqrt(lat_error**2 + lon_error**2)
                print(f"    Location Error: {total_error:.1f} km")
    
    return located_events


def demonstrate_real_time_processing():
    """Demonstrate real-time streaming data processing"""
    print("\n=== Real-Time Processing Demo ===")
    
    processor = SeismicCNNProcessor(
        sampling_rate=100.0,
        window_length=10.0,
        detection_threshold=0.4,
        device="cpu"
    )
    
    sensor_location = GeologicalLocation(
        latitude=45.0,
        longitude=-122.0,
        elevation=1000.0
    )
    
    print("Simulating real-time data streaming...")
    
    # Simulate 10 chunks of 5-second data
    chunk_duration = 5.0
    chunk_samples = int(chunk_duration * processor.sampling_rate)
    all_events = []
    
    for chunk_idx in range(10):
        # Create chunk with occasional events
        waveform_chunk = np.random.randn(3, chunk_samples) * 0.1
        
        # Add event to some chunks
        if chunk_idx in [3, 7]:  # Events in chunks 3 and 7
            event_start = chunk_samples // 4
            event_duration = int(2 * processor.sampling_rate)
            amplitude = 2.0 if chunk_idx == 3 else 1.5
            
            t_event = np.arange(event_duration) / processor.sampling_rate
            event_signal = amplitude * np.exp(-t_event * 2) * np.sin(2 * np.pi * 6 * t_event)
            
            waveform_chunk[:, event_start:event_start + event_duration] += event_signal
        
        # Create timestamps for chunk
        chunk_start_time = datetime.utcnow() + timedelta(seconds=chunk_idx * chunk_duration)
        timestamps = np.array([
            chunk_start_time + timedelta(seconds=i/processor.sampling_rate)
            for i in range(chunk_samples)
        ])
        
        # Process chunk
        chunk_events = processor.process_continuous_data(
            waveform_data=waveform_chunk,
            timestamps=timestamps,
            sensor_location=sensor_location,
            sensor_id="REALTIME_STATION"
        )
        
        print(f"  Chunk {chunk_idx + 1}: {len(chunk_events)} events detected")
        all_events.extend(chunk_events)
    
    print(f"\nTotal events detected in real-time stream: {len(all_events)}")
    
    # Show processing statistics
    stats = processor.get_processing_statistics()
    print(f"Processing Statistics:")
    print(f"  Total windows: {stats['total_windows_processed']}")
    print(f"  Total events: {stats['events_detected']}")
    print(f"  Last processing: {stats['last_processing_time']}")
    
    return all_events


def visualize_detection_results(events, waveform_data, timestamps, title="Seismic Detection Results"):
    """Visualize detection results (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        fig.suptitle(title)
        
        # Plot waveform components
        components = ['Z', 'N', 'E']
        for i, component in enumerate(components):
            axes[i].plot(timestamps, waveform_data[i], 'k-', alpha=0.7, linewidth=0.5)
            axes[i].set_ylabel(f'{component} Component')
            axes[i].grid(True, alpha=0.3)
            
            # Mark detected events
            for event in events:
                event_time = event.timestamp
                axes[i].axvline(event_time, color='red', alpha=0.7, linestyle='--')
                axes[i].text(event_time, axes[i].get_ylim()[1] * 0.8, 
                           f'{event.event_type.value}\n{event.confidence:.2f}',
                           rotation=90, fontsize=8, ha='right')
        
        # Plot detection summary
        if events:
            event_times = [e.timestamp for e in events]
            event_confidences = [e.confidence for e in events]
            event_types = [e.event_type.value for e in events]
            
            axes[3].scatter(event_times, event_confidences, c='red', s=50, alpha=0.7)
            axes[3].set_ylabel('Detection\nConfidence')
            axes[3].set_xlabel('Time')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_ylim(0, 1)
            
            # Add event type labels
            for i, (time, conf, etype) in enumerate(zip(event_times, event_confidences, event_types)):
                axes[3].annotate(etype, (time, conf), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        # Format time axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


def main():
    """Run all demonstration examples"""
    print("CNN-based Seismic Event Detection Demonstration")
    print("=" * 50)
    
    try:
        # Basic detection demo
        detected_events, true_events = demonstrate_basic_detection()
        
        # Volcanic classification demo
        volcanic_events = demonstrate_volcanic_classification()
        
        # Multi-station processing demo
        located_events = demonstrate_multi_station_processing()
        
        # Real-time processing demo
        realtime_events = demonstrate_real_time_processing()
        
        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print(f"Total events detected across all demos: {len(detected_events) + len(volcanic_events) + len(located_events) + len(realtime_events)}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()