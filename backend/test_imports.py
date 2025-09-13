#!/usr/bin/env python3

try:
    import obspy
    print("✓ obspy available")
except ImportError:
    print("✗ obspy not available - needed for IRIS connector")

try:
    import numpy
    print("✓ numpy available")
except ImportError:
    print("✗ numpy not available")

try:
    from core.real_data_connectors.base_connector import BaseDataConnector
    print("✓ BaseDataConnector import successful")
except ImportError as e:
    print(f"✗ BaseDataConnector import failed: {e}")

try:
    from core.real_data_connectors import ConnectorFactory
    print("✓ ConnectorFactory import successful")
except ImportError as e:
    print(f"✗ ConnectorFactory import failed: {e}")