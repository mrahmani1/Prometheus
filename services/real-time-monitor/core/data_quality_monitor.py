#!/usr/bin/env python3
"""
Advanced Data Quality Monitoring System

This module provides comprehensive data quality monitoring and validation
for real-time geological data streams with automated quality assessment,
anomaly detection, and data integrity verification.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import json

logger = logging.getLogger(__name__)

class QualityFlag(str, Enum):
    GOOD = "good"
    QUESTIONABLE = "questionable"
    BAD = "bad"
    MISSING = "missing"
    OUT_OF_RANGE = "out_of_range"
    SPIKE = "spike"
    DRIFT = "drift"
    FROZEN = "frozen"
    INCONSISTENT = "inconsistent"

class DataSource(str, Enum):
    SEISMIC_NETWORK = "seismic_network"
    GPS_STATION = "gps_station"
    SATELLITE_SAR = "satellite_sar"
    GAS_SENSOR = "gas_sensor"
    THERMAL_CAMERA = "thermal_camera"
    TILTMETER = "tiltmeter"
    MAGNETOMETER = "magnetometer"
    GRAVIMETER = "gravimeter"

@dataclass
class QualityMetrics:
    completeness: float  # 0-1, percentage of expected data received
    accuracy: float      # 0-1, accurac