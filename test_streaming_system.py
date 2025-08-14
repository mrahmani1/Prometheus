#!/usr/bin/env python3
"""
Test real-time data streaming system
"""

import sys
import os
import asyncio
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_streaming_system_instantiation():
    """Test basic instantiation of streaming system"""
    print("Testing RealTimeDataStreamer instantiation...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        # Create streaming system with custom config
        config = {
            'streams': {
                'usgs_earthquakes': {
                    'enabled': True,
                    'interval': 60,  # 1 minute for testing
                    'min_magnitude': 4.0
                },
                'nasa_fires': {
                    'enabled': False  # Disable for testing
                }
            },
            'buffer_settings': {
                'max_buffer_size': 100,
                'retention_hours': 1
            }
        }
        
        streamer = RealTimeDataStreamer(config=config)
        print("+ RealTimeDataStreamer instantiated successfully")
        
        # Test configuration
        assert 'streams' in streamer.config
        assert 'buffer_settings' in streamer.config
        print("+ Configuration loaded correctly")
        
        # Test stream configuration
        assert streamer.config['streams']['usgs_earthquakes']['enabled'] == True
        assert streamer.config['streams']['usgs_earthquakes']['interval'] == 60
        print("+ Stream configuration customized correctly")
        
        # Test API initialization
        assert hasattr(streamer, 'usgs_api')
        assert hasattr(streamer, 'nasa_api')
        assert hasattr(streamer, 'asf_api')
        assert hasattr(streamer, 'iris_api')
        print("+ External APIs initialized")
        
        # Test ontology initialization
        assert hasattr(streamer, 'geological_ontology')
        print("+ Geological ontology initialized")
        
        return True
        
    except Exception as e:
        print(f"X Error testing streaming system instantiation: {e}")
        return False

def test_stream_configuration():
    """Test stream configuration and management"""
    print("\nTesting stream configuration...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test default configuration
        default_config = streamer._get_default_config()
        print(f"+ Default configuration has {len(default_config['streams'])} streams")
        
        # Test stream types
        expected_streams = [
            'usgs_earthquakes', 'usgs_volcanoes', 'nasa_fires', 
            'nasa_thermal_anomalies', 'asf_insar_alerts', 'iris_seismic_stations'
        ]
        
        for stream_name in expected_streams:
            assert stream_name in default_config['streams']
            print(f"+ Stream configured: {stream_name}")
        
        # Test configuration sections
        assert 'buffer_settings' in default_config
        assert 'performance' in default_config
        assert 'data_quality' in default_config
        print("+ All configuration sections present")
        
        return True
        
    except Exception as e:
        print(f"X Error testing stream configuration: {e}")
        return False

def test_buffer_management():
    """Test data buffer management"""
    print("\nTesting buffer management...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test buffer data method
        test_data = [
            {'id': '1', 'type': 'earthquake', 'magnitude': 5.2},
            {'id': '2', 'type': 'earthquake', 'magnitude': 4.8},
            {'id': '3', 'type': 'volcano', 'alert_level': 'watch'}
        ]
        
        # Test adding data to buffer
        asyncio.run(streamer._buffer_data('test_stream', test_data))
        print("+ Data added to buffer successfully")
        
        # Test retrieving buffered data
        buffered_data = streamer.get_buffered_data('test_stream')
        assert len(buffered_data) == 3
        print(f"+ Retrieved {len(buffered_data)} items from buffer")
        
        # Test buffer limit
        buffered_data_limited = streamer.get_buffered_data('test_stream', limit=2)
        assert len(buffered_data_limited) == 2
        print("+ Buffer limit functionality works")
        
        # Test latest data retrieval
        latest_data = asyncio.run(streamer.get_latest_data('test_stream', count=1))
        assert len(latest_data) == 1
        print("+ Latest data retrieval works")
        
        # Test buffer timestamps
        for item in buffered_data:
            assert 'buffer_timestamp' in item
        print("+ Buffer timestamps added correctly")
        
        return True
        
    except Exception as e:
        print(f"X Error testing buffer management: {e}")
        return False

def test_subscription_system():
    """Test subscription and notification system"""
    print("\nTesting subscription system...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test subscription
        notifications_received = []
        
        def test_callback(stream_name, data):
            notifications_received.append((stream_name, len(data)))
        
        # Subscribe to stream
        streamer.subscribe('test_stream', test_callback)
        print("+ Subscribed to test stream")
        
        # Test notification
        test_data = [{'id': '1', 'type': 'test'}]
        asyncio.run(streamer._notify_subscribers('test_stream', test_data))
        
        # Check if notification was received
        assert len(notifications_received) == 1
        assert notifications_received[0][0] == 'test_stream'
        assert notifications_received[0][1] == 1
        print("+ Notification system works")
        
        # Test unsubscription
        streamer.unsubscribe('test_stream', test_callback)
        
        # Test notification after unsubscription
        asyncio.run(streamer._notify_subscribers('test_stream', test_data))
        assert len(notifications_received) == 1  # No new notifications
        print("+ Unsubscription works")
        
        return True
        
    except Exception as e:
        print(f"X Error testing subscription system: {e}")
        return False

def test_stream_statistics():
    """Test stream statistics tracking"""
    print("\nTesting stream statistics...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test statistics update
        streamer._update_stream_stats('test_stream', 5)
        streamer._update_stream_stats('test_stream', 3)
        streamer._update_stream_stats('test_stream', 7)
        
        # Get statistics
        stats = streamer.get_stream_stats()
        assert 'test_stream' in stats
        
        stream_stats = stats['test_stream']
        assert stream_stats['total_records'] == 15
        assert stream_stats['update_count'] == 3
        assert stream_stats['average_records_per_update'] == 5.0
        print("+ Stream statistics calculated correctly")
        
        # Test required fields
        required_fields = ['total_records', 'update_count', 'average_records_per_update', 
                          'last_update', 'start_time', 'uptime_seconds']
        for field in required_fields:
            assert field in stream_stats
        print("+ All required statistical fields present")
        
        return True
        
    except Exception as e:
        print(f"X Error testing stream statistics: {e}")
        return False

def test_data_enhancement():
    """Test data enhancement with geological ontology"""
    print("\nTesting data enhancement...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test geological context extraction
        test_item = {
            'title': 'Magnitude 5.8 earthquake near volcanic region',
            'ontology': {
                'matched_concepts': ['earthquake', 'volcanic_eruption', 'seismic_wave'],
                'related_concepts': ['p_wave', 's_wave', 'fault']
            },
            'properties': {
                'mag': 5.8
            }
        }
        
        geological_context = streamer._extract_geological_context(test_item)
        print("+ Geological context extracted")
        
        # Check context fields
        assert 'phenomena_type' in geological_context
        assert 'urgency_level' in geological_context
        assert 'impact_assessment' in geological_context
        assert 'monitoring_priority' in geological_context
        print("+ All context fields present")
        
        # Check phenomena type detection
        assert 'seismic' in geological_context['phenomena_type']
        assert 'volcanic' in geological_context['phenomena_type']
        print("+ Phenomena types correctly identified")
        
        # Check urgency assessment
        assert geological_context['urgency_level'] == 'medium'  # For magnitude 5.8
        assert geological_context['impact_assessment'] == 'medium'
        print("+ Urgency assessment correct for magnitude 5.8")
        
        return True
        
    except Exception as e:
        print(f"X Error testing data enhancement: {e}")
        return False

async def test_stream_data_fetching():
    """Test stream data fetching methods"""
    print("\nTesting stream data fetching...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test USGS earthquake fetching (using fallback)
        config = {'feed_type': 'all', 'min_magnitude': 4.0}
        earthquakes = await streamer._fetch_usgs_earthquakes(config)
        print(f"+ USGS earthquakes fetched: {len(earthquakes)} events")
        
        # Check earthquake data structure
        if earthquakes:
            eq = earthquakes[0]
            assert 'stream_source' in eq
            assert 'stream_timestamp' in eq
            assert 'data_type' in eq
            assert eq['stream_source'] == 'usgs_earthquakes'
            assert eq['data_type'] == 'seismic_event'
            print("+ Earthquake data structure correct")
        
        # Test USGS volcano fetching
        volcano_config = {'alert_levels': ['watch', 'advisory']}
        volcanoes = await streamer._fetch_usgs_volcanoes(volcano_config)
        print(f"+ USGS volcanoes fetched: {len(volcanoes)} alerts")
        
        # Test NASA fire fetching
        fire_config = {'confidence_threshold': 75}
        fires = await streamer._fetch_nasa_fires(fire_config)
        print(f"+ NASA fires fetched: {len(fires)} detections")
        
        return True
        
    except Exception as e:
        print(f"X Error testing stream data fetching: {e}")
        return False

def test_stream_enhancement():
    """Test stream data enhancement"""
    print("\nTesting stream data enhancement...")
    
    try:
        from scientific_knowledge_graph import RealTimeDataStreamer
        
        streamer = RealTimeDataStreamer()
        
        # Test data enhancement
        test_data = [
            {
                'title': 'Earthquake magnitude 4.5',
                'type': 'earthquake',
                'properties': {'mag': 4.5}
            }
        ]
        
        enhanced_data = asyncio.run(streamer._enhance_stream_data('usgs_earthquakes', test_data))
        
        assert len(enhanced_data) == 1
        enhanced_item = enhanced_data[0]
        
        # Check real-time context
        assert 'real_time_context' in enhanced_item
        rt_context = enhanced_item['real_time_context']
        assert rt_context['stream_name'] == 'usgs_earthquakes'
        assert rt_context['data_freshness'] == 'real_time'
        print("+ Real-time context added correctly")
        
        # Check ontology enhancement (if available)
        if 'ontology' in enhanced_item:
            assert 'geological_context' in enhanced_item
            print("+ Geological context added")
        
        return True
        
    except Exception as e:
        print(f"X Error testing stream enhancement: {e}")
        return False

def main():
    """Run all streaming system tests"""
    print("=" * 60)
    print("REAL-TIME DATA STREAMING SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_streaming_system_instantiation,
        test_stream_configuration,
        test_buffer_management,
        test_subscription_system,
        test_stream_statistics,
        test_data_enhancement,
        lambda: asyncio.run(test_stream_data_fetching()),
        test_stream_enhancement
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
    
    print("=" * 60)
    print(f"STREAMING SYSTEM RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("All streaming system tests PASSED! System is ready for production.")
    else:
        print("Some streaming system tests FAILED. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)