"""
Disaster Recovery Monitoring System
Advanced monitoring and metrics collection for disaster recovery operations,
RTO/RPO tracking, and comprehensive health monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import asdict
import statistics

import asyncpg
import aioredis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

from models.disaster_recovery_models import *

logger = logging.getLogger(__name__)

class DisasterRecoveryMonitoring:
    """Advanced monitoring system for disaster recovery operations."""
    
    def __init__(self):
        self.metrics_registry = CollectorReg