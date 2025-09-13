"""
Real Data Connectors Framework

This module provides authenticated connectors to real geological data sources,
replacing all placeholder and mock data implementations with authentic data streams.
"""

from .base_connector import BaseDataConnector, DataSourceConfig, ConnectionStatus
from .iris_connector import IRISConnector
from .usgs_connector import USGSVolcanoConnector
from .nasa_connector import NASAEarthDataConnector
from .copernicus_connector import CopernicusConnector
from .asf_connector import ASFHyp3Connector
from .geochem_connector import GeochemConnector
from .geological_survey_connector import GeologicalSurveyConnector
from .health_monitor import DataSourceHealthMonitor
from .connector_factory import ConnectorFactory

__all__ = [
    'BaseDataConnector',
    'DataSourceConfig', 
    'ConnectionStatus',
    'IRISConnector',
    'USGSVolcanoConnector',
    'NASAEarthDataConnector',
    'CopernicusConnector',
    'ASFHyp3Connector',
    'GeochemConnector',
    'GeologicalSurveyConnector',
    'DataSourceHealthMonitor',
    'ConnectorFactory'
]