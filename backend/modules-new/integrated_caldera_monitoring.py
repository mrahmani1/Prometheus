"""
Integrated Caldera Monitoring System
Orchestrates satellite data acquisition, ground-based monitoring,
and real-time analysis for comprehensive resurgent caldera surveillance
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import aiohttp
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Prometheus modules
try:
    from .resurgent_caldera_monitoring import ResurgentCalderaMonitoringSystem
    from .satellite_scheduler import (
        SatelliteScheduler, AcquisitionPlan, GeographicBounds, 
        SatelliteMission, ProcessingLevel
    )
    from .3d_magmatic_system_modeler import ThreeDMagmaticSystemModeler
    from .cap_rock_stability_analyzer import CapRockStabilityAnalyzer
    from .data_integration_pipeline import DataIntegrationPipeline
    from .supereruption_analysis import SupereruptionPrecursorAnalyzer
    PROMETHEUS_MODULES = True
except ImportError:
    # Fallbacks for testing
    class ResurgentCalderaMonitoringSystem: pass
    class SatelliteScheduler: pass
    class ThreeDMagmaticSystemModeler: pass
    class CapRockStabilityAnalyzer: pass
    class DataIntegrationPipeline: pass
    class SupereruptionPrecursorAnalyzer: pass
    PROMETHEUS_MODULES = False

warnings.filterwarnings('ignore')

# Database models
Base = declarative_base()

class MonitoringSession(Base):
    """Database model for monitoring sessions"""
    __tablename__ = 'monitoring_sessions'
    
    id = Column(Integer, primary_key=True)
    caldera_id = Column(String(50), nullable=False)
    session_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    monitoring_type = Column(String(30), nullable=False)  # continuous, campaign, emergency
    status = Column(String(20), default='active')
    configuration = Column(JSON)
    results_summary = Column(JSON)
    alert_level = Column(String(20), default='normal')
    created_date = Column(DateTime, default=datetime.utcnow)

class DataProduct(Base):
    """Database model for data products"""
    __tablename__ = 'data_products'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(String(100), unique=True, nullable=False)
    caldera_id = Column(String(50), nullable=False)
    data_type = Column(String(50), nullable=False)  # satellite, seismic, gps, gas
    acquisition_time = Column(DateTime, nullable=False)
    processing_level = Column(String(20), nullable=False)
    file_path = Column(Text, nullable=False)
    metadata = Column(JSON)
    quality_score = Column(Float)
    processing_status = Column(String(20), default='raw')
    integrated = Column(Boolean, default=False)
    created_date = Column(DateTime, default=datetime.utcnow)

class AlertEvent(Base):
    """Database model for alerts"""
    __tablename__ = 'alert_events'
    
    id = Column(Integer, primary_key=True)
    caldera_id = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    trigger_conditions = Column(JSON)
    alert_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    response_actions = Column(JSON)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MonitoringType(Enum):
    CONTINUOUS = "continuous"
    CAMPAIGN = "campaign"
    EMERGENCY = "emergency"
    VALIDATION = "validation"

class DataType(Enum):
    SATELLITE_INSAR = "satellite_insar"
    SATELLITE_OPTICAL = "satellite_optical"
    SEISMIC = "seismic"
    GPS = "gps"
    GAS_EMISSIONS = "gas_emissions"
    THERMAL = "thermal"
    GRAVITY = "gravity"
    MAGNETIC = "magnetic"
    METEOROLOGICAL = "meteorological"

@dataclass
class MonitoringConfiguration:
    """Configuration for integrated monitoring"""
    caldera_id: str
    monitoring_type: MonitoringType
    data_sources: List[DataType]
    acquisition_frequency: Dict[str, int]  # hours between acquisitions
    processing_priorities: Dict[str, int]  # 1=highest, 5=lowest
    alert_thresholds: Dict[str, float]
    quality_requirements: Dict[str, float]
    backup_systems: List[str]
    automated_responses: Dict[str, List[str]]

@dataclass
class IntegratedAnalysisResult:
    """Result of integrated multi-source analysis"""
    caldera_id: str
    analysis_timestamp: datetime
    overall_threat_level: AlertSeverity
    confidence_score: float
    key_findings: List[str]
    data_sources_used: List[str]
    analysis_components: Dict[str, Any]
    recommendations: List[str]
    next_scheduled_analysis: datetime
    alerts_generated: List[Dict]

class IntegratedCalderaMonitoring:
    """Integrated caldera monitoring system orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None, database_url: str = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.database_url = database_url or "sqlite:///integrated_monitoring.db"
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # Initialize subsystems
        self.resurgent_monitor = ResurgentCalderaMonitoringSystem()
        self.satellite_scheduler = SatelliteScheduler()
        self.magma_modeler = ThreeDMagmaticSystemModeler()
        self.cap_rock_analyzer = CapRockStabilityAnalyzer()
        self.data_integration = DataIntegrationPipeline()
        self.supereruption_analyzer = SupereruptionPrecursorAnalyzer()
        
        # Active monitoring sessions
        self.active_sessions: Dict[str, Dict] = {}
        
        # Alert system
        self.alert_handlers = {}
        self.notification_channels = []
        
        # Data processing queue
        self.processing_queue = []
        self.processing_workers = 4
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'monitoring_modes': {
                'continuous': {
                    'satellite_frequency_hours': 12,
                    'ground_frequency_hours': 1,
                    'analysis_frequency_hours': 6
                },
                'campaign': {
                    'satellite_frequency_hours': 6,
                    'ground_frequency_hours': 0.5,
                    'analysis_frequency_hours': 2
                },
                'emergency': {
                    'satellite_frequency_hours': 2,
                    'ground_frequency_hours': 0.1,
                    'analysis_frequency_hours': 0.5
                }
            },
            'alert_thresholds': {
                'deformation_rate_mm_day': 10,
                'seismic_energy_increase': 5.0,
                'gas_emission_increase': 3.0,
                'thermal_anomaly_c': 50,
                'magma_chamber_pressure_increase': 20,
                'cap_rock_failure_probability': 0.3
            },
            'data_quality_requirements': {
                'satellite_coherence_min': 0.3,
                'gps_precision_mm': 5,
                'seismic_completeness_magnitude': 1.5,
                'gas_measurement_accuracy': 0.1
            },
            'integration_weights': {
                'satellite_deformation': 0.25,
                'seismic_activity': 0.25,
                'gas_emissions': 0.20,
                'thermal_activity': 0.15,
                'ground_deformation': 0.15
            },
            'automated_responses': {
                'high_alert': [
                    'increase_monitoring_frequency',
                    'notify_authorities',
                    'activate_backup_systems'
                ],
                'critical_alert': [
                    'emergency_monitoring_mode',
                    'immediate_notification',
                    'prepare_evacuation_support'
                ]
            }
        }
    
    async def initialize_monitoring_session(self, caldera_id: str,
                                          monitoring_config: MonitoringConfiguration) -> str:
        """Initialize integrated monitoring session for a caldera"""
        
        self.logger.info(f"Initializing monitoring session for {caldera_id}")
        
        # Create database record
        session_record = MonitoringSession(
            caldera_id=caldera_id,
            session_name=f"{caldera_id}_monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}",
            start_time=datetime.now(),
            monitoring_type=monitoring_config.monitoring_type.value,
            configuration=monitoring_config.__dict__,
            status='initializing'
        )
        
        self.db_session.add(session_record)
        self.db_session.commit()
        
        session_id = str(session_record.id)
        
        # Initialize resurgent caldera monitoring
        caldera_bounds = await self._get_caldera_bounds(caldera_id)
        caldera_location = await self._get_caldera_location(caldera_id)
        
        await self.resurgent_monitor.initialize_caldera_monitoring(
            caldera_id, caldera_location, caldera_bounds
        )
        
        # Setup satellite acquisition
        satellite_plan = await self._setup_satellite_acquisition(
            caldera_id, monitoring_config, caldera_bounds
        )
        
        # Initialize ground-based monitoring
        ground_monitoring = await self._initialize_ground_monitoring(
            caldera_id, monitoring_config
        )
        
        # Setup data integration
        integration_config = await self._configure_data_integration(monitoring_config)
        
        # Store session information
        session_info = {
            'session_id': session_id,
            'caldera_id': caldera_id,
            'configuration': monitoring_config,
            'satellite_plan': satellite_plan,
            'ground_monitoring': ground_monitoring,
            'integration_config': integration_config,
            'status': 'active',
            'last_analysis': None,
            'alert_level': AlertSeverity.LOW
        }
        
        self.active_sessions[session_id] = session_info
        
        # Update database
        session_record.status = 'active'
        self.db_session.commit()
        
        # Start monitoring tasks
        await self._start_monitoring_tasks(session_id)
        
        self.logger.info(f"Monitoring session {session_id} initialized for {caldera_id}")
        
        return session_id
    
    async def _get_caldera_bounds(self, caldera_id: str) -> Dict[str, float]:
        """Get geographic bounds for caldera"""
        
        # In practice, would query from caldera database
        # For now, return example bounds
        caldera_bounds_db = {
            'yellowstone': {'north': 44.8, 'south': 44.1, 'east': -110.1, 'west': -111.2, 'max_depth': 15},
            'long_valley': {'north': 37.8, 'south': 37.6, 'east': -118.7, 'west': -119.1, 'max_depth': 12},
            'valles': {'north': 35.95, 'south': 35.75, 'east': -106.4, 'west': -106.75, 'max_depth': 10}
        }
        
        return caldera_bounds_db.get(caldera_id.lower(), {
            'north': 0, 'south': 0, 'east': 0, 'west': 0, 'max_depth': 15
        })
    
    async def _get_caldera_location(self, caldera_id: str) -> Tuple[float, float]:
        """Get center coordinates for caldera"""
        
        caldera_locations = {
            'yellowstone': (-110.65, 44.45),
            'long_valley': (-118.9, 37.7),
            'valles': (-106.575, 35.85)
        }
        
        return caldera_locations.get(caldera_id.lower(), (0, 0))
    
    async def _setup_satellite_acquisition(self, caldera_id: str,
                                         monitoring_config: MonitoringConfiguration,
                                         caldera_bounds: Dict) -> Dict:
        """Setup automated satellite data acquisition"""
        
        # Create geographic bounds object
        geo_bounds = GeographicBounds(
            north=caldera_bounds['north'],
            south=caldera_bounds['south'],
            east=caldera_bounds['east'],
            west=caldera_bounds['west']
        )
        
        # Determine satellite missions based on monitoring type
        if monitoring_config.monitoring_type == MonitoringType.EMERGENCY:
            missions = [SatelliteMission.SENTINEL1, SatelliteMission.ALOS2, SatelliteMission.ICEYE]
            temporal_baseline = 6  # days
        elif monitoring_config.monitoring_type == MonitoringType.CAMPAIGN:
            missions = [SatelliteMission.SENTINEL1, SatelliteMission.ALOS2]
            temporal_baseline = 12
        else:  # continuous
            missions = [SatelliteMission.SENTINEL1]
            temporal_baseline = 24
        
        # Create acquisition plan
        plan = AcquisitionPlan(
            plan_id=f"{caldera_id}_integrated_monitoring",
            name=f"Integrated Monitoring - {caldera_id}",
            description=f"Automated satellite acquisition for {caldera_id} monitoring",
            target_area=geo_bounds,
            priority="HIGH" if monitoring_config.monitoring_type == MonitoringType.EMERGENCY else "MEDIUM",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=365),
            missions=missions,
            processing_levels=[ProcessingLevel.SLC, ProcessingLevel.GRD],
            temporal_baseline_max=temporal_baseline,
            auto_download=True,
            auto_process=True
        )
        
        # Submit to satellite scheduler
        plan_id = await self.satellite_scheduler.create_acquisition_plan(plan)
        
        return {
            'plan_id': plan_id,
            'missions': [m.value for m in missions],
            'temporal_baseline_days': temporal_baseline,
            'auto_processing': True
        }
    
    async def _initialize_ground_monitoring(self, caldera_id: str,
                                          monitoring_config: MonitoringConfiguration) -> Dict:
        """Initialize ground-based monitoring systems"""
        
        # This would interface with actual ground monitoring systems
        # For now, return configuration
        
        ground_config = {
            'seismic_network': {
                'stations_active': 15,
                'detection_threshold': 0.5,
                'real_time_processing': True
            },
            'gps_network': {
                'stations_active': 8,
                'measurement_frequency_min': 15,
                'precision_mm': 3
            },
            'gas_monitoring': {
                'stations_active': 5,
                'species_monitored': ['CO2', 'SO2', 'H2S'],
                'measurement_frequency_min': 60
            },
            'thermal_monitoring': {
                'cameras_active': 3,
                'measurement_frequency_min': 30,
                'temperature_precision_c': 2
            }
        }
        
        return ground_config
    
    async def _configure_data_integration(self, monitoring_config: MonitoringConfiguration) -> Dict:
        """Configure data integration parameters"""
        
        integration_config = {
            'fusion_algorithms': ['weighted_average', 'kalman_filter', 'bayesian_integration'],
            'temporal_windows': {
                'satellite': 24,  # hours
                'seismic': 1,
                'gps': 6,
                'gas': 12
            },
            'quality_weights': {
                'high_quality': 1.0,
                'medium_quality': 0.7,
                'low_quality': 0.3
            },
            'uncertainty_propagation': True,
            'outlier_detection': True
        }
        
        return integration_config
    
    async def _start_monitoring_tasks(self, session_id: str):
        """Start automated monitoring tasks"""
        
        session = self.active_sessions[session_id]
        
        # Start data acquisition monitoring
        asyncio.create_task(self._monitor_data_acquisition(session_id))
        
        # Start data processing tasks
        asyncio.create_task(self._process_data_queue(session_id))
        
        # Start integrated analysis
        asyncio.create_task(self._run_integrated_analysis(session_id))
        
        # Start alert monitoring
        asyncio.create_task(self._monitor_alerts(session_id))
    
    async def _monitor_data_acquisition(self, session_id: str):
        """Monitor incoming data from all sources"""
        
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        while session.get('status') == 'active':
            try:
                # Check satellite data
                await self._check_satellite_data(session_id)
                
                # Check ground-based data
                await self._check_ground_data(session_id)
                
                # Update data inventory
                await self._update_data_inventory(session_id)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in data acquisition monitoring: {e}")
                await asyncio.sleep(60)  # Shorter wait on error
    
    async def _check_satellite_data(self, session_id: str):
        """Check for new satellite data"""
        
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        # Get satellite plan status
        plan_id = session['satellite_plan']['plan_id']
        plan_status = await self.satellite_scheduler.get_plan_status(plan_id)
        
        # Check for new products
        new_products = plan_status.get('new_products_since_last_check', [])
        
        for product in new_products:
            # Add to processing queue
            data_product = DataProduct(
                product_id=product['product_id'],
                caldera_id=caldera_id,
                data_type='satellite_insar',
                acquisition_time=product['acquisition_date'],
                processing_level='raw',
                file_path=product['file_path'],
                metadata=product.get('metadata', {}),
                quality_score=product.get('quality_score', 0.5)
            )
            
            self.db_session.add(data_product)
            self.processing_queue.append({
                'session_id': session_id,
                'product': data_product,
                'priority': 2
            })
        
        self.db_session.commit()
    
    async def _check_ground_data(self, session_id: str):
        """Check for new ground-based monitoring data"""
        
        # In practice, would interface with ground monitoring systems
        # For now, simulate data availability
        
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        # Simulate ground data updates
        ground_data_types = ['seismic', 'gps', 'gas_emissions', 'thermal']
        
        for data_type in ground_data_types:
            # Simulate some new data
            if np.random.random() < 0.3:  # 30% chance of new data
                data_product = DataProduct(
                    product_id=f"{caldera_id}_{data_type}_{int(datetime.now().timestamp())}",
                    caldera_id=caldera_id,
                    data_type=data_type,
                    acquisition_time=datetime.now(),
                    processing_level='processed',
                    file_path=f"/data/{caldera_id}/{data_type}/latest.dat",
                    metadata={'simulated': True},
                    quality_score=0.8
                )
                
                self.db_session.add(data_product)
                self.processing_queue.append({
                    'session_id': session_id,
                    'product': data_product,
                    'priority': 1  # Ground data gets higher priority
                })
        
        self.db_session.commit()
    
    async def _update_data_inventory(self, session_id: str):
        """Update inventory of available data"""
        
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        # Query database for recent data
        recent_data = self.db_session.query(DataProduct).filter(
            DataProduct.caldera_id == caldera_id,
            DataProduct.acquisition_time >= datetime.now() - timedelta(hours=24)
        ).all()
        
        # Update session inventory
        session['data_inventory'] = {
            'total_products': len(recent_data),
            'by_type': {},
            'quality_summary': {},
            'last_updated': datetime.now()
        }
        
        for product in recent_data:
            data_type = product.data_type
            if data_type not in session['data_inventory']['by_type']:
                session['data_inventory']['by_type'][data_type] = 0
            session['data_inventory']['by_type'][data_type] += 1
    
    async def _process_data_queue(self, session_id: str):
        """Process queued data products"""
        
        while self.active_sessions.get(session_id, {}).get('status') == 'active':
            try:
                if self.processing_queue:
                    # Sort by priority
                    self.processing_queue.sort(key=lambda x: x['priority'])
                    
                    # Process highest priority items
                    batch_size = min(self.processing_workers, len(self.processing_queue))
                    current_batch = self.processing_queue[:batch_size]
                    self.processing_queue = self.processing_queue[batch_size:]
                    
                    # Process in parallel
                    tasks = [
                        self._process_data_product(item['session_id'], item['product'])
                        for item in current_batch
                        if item['session_id'] == session_id
                    ]
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(30)  # Check queue every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in data processing queue: {e}")
                await asyncio.sleep(60)
    
    async def _process_data_product(self, session_id: str, product: DataProduct):
        """Process individual data product"""
        
        try:
            if product.data_type == 'satellite_insar':
                # Process InSAR data
                processed_data = await self._process_insar_data(product)
                
            elif product.data_type == 'seismic':
                # Process seismic data
                processed_data = await self._process_seismic_data(product)
                
            elif product.data_type == 'gps':
                # Process GPS data
                processed_data = await self._process_gps_data(product)
                
            elif product.data_type == 'gas_emissions':
                # Process gas data
                processed_data = await self._process_gas_data(product)
                
            else:
                # Generic processing
                processed_data = {'status': 'processed', 'timestamp': datetime.now()}
            
            # Update database
            product.processing_status = 'processed'
            product.metadata = {**product.metadata, **processed_data}
            self.db_session.commit()
            
            self.logger.debug(f"Processed data product {product.product_id}")
            
        except Exception as e:
            product.processing_status = 'failed'
            product.metadata['error'] = str(e)
            self.db_session.commit()
            self.logger.error(f"Failed to process {product.product_id}: {e}")
    
    async def _process_insar_data(self, product: DataProduct) -> Dict:
        """Process InSAR satellite data"""
        
        # Simulate InSAR processing
        processing_result = {
            'deformation_map_generated': True,
            'coherence_mean': np.random.uniform(0.3, 0.8),
            'max_deformation_mm': np.random.uniform(0, 50),
            'processing_algorithm': 'ISCE2',
            'reference_date': '2023-01-01',
            'temporal_baseline_days': 12
        }
        
        return processing_result
    
    async def _process_seismic_data(self, product: DataProduct) -> Dict:
        """Process seismic monitoring data"""
        
        processing_result = {
            'events_detected': np.random.poisson(5),
            'magnitude_range': [0.5, 3.2],
            'location_accuracy_km': 0.8,
            'frequency_content_analyzed': True,
            'b_value': np.random.uniform(0.8, 1.2)
        }
        
        return processing_result
    
    async def _process_gps_data(self, product: DataProduct) -> Dict:
        """Process GPS monitoring data"""
        
        processing_result = {
            'position_precision_mm': np.random.uniform(2, 8),
            'displacement_vectors': {
                'north_mm': np.random.uniform(-10, 10),
                'east_mm': np.random.uniform(-10, 10),
                'up_mm': np.random.uniform(-5, 20)
            },
            'velocity_estimates': True,
            'atmospheric_correction_applied': True
        }
        
        return processing_result
    
    async def _process_gas_data(self, product: DataProduct) -> Dict:
        """Process gas emission monitoring data"""
        
        processing_result = {
            'co2_flux_tons_day': np.random.uniform(100, 1000),
            'so2_flux_tons_day': np.random.uniform(1, 50),
            'he3_he4_ratio': np.random.uniform(6, 8),
            'temperature_c': np.random.uniform(80, 120),
            'wind_correction_applied': True
        }
        
        return processing_result
    
    async def _run_integrated_analysis(self, session_id: str):
        """Run integrated analysis combining all data sources"""
        
        session = self.active_sessions[session_id]
        config = session['configuration']
        
        # Determine analysis frequency
        if config.monitoring_type == MonitoringType.EMERGENCY:
            analysis_interval = 1800  # 30 minutes
        elif config.monitoring_type == MonitoringType.CAMPAIGN:
            analysis_interval = 7200   # 2 hours
        else:  # continuous
            analysis_interval = 21600  # 6 hours
        
        while session.get('status') == 'active':
            try:
                # Run comprehensive analysis
                analysis_result = await self._perform_integrated_analysis(session_id)
                
                # Store results
                session['last_analysis'] = analysis_result
                
                # Check for alerts
                alerts = await self._evaluate_alerts(session_id, analysis_result)
                
                # Update alert level
                if alerts:
                    max_severity = max(AlertSeverity(alert['severity']) for alert in alerts)
                    session['alert_level'] = max_severity
                
                # Log analysis completion
                self.logger.info(f"Integrated analysis completed for {session['caldera_id']}")
                
                await asyncio.sleep(analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in integrated analysis: {e}")
                await asyncio.sleep(300)  # Shorter wait on error
    
    async def _perform_integrated_analysis(self, session_id: str) -> IntegratedAnalysisResult:
        """Perform comprehensive integrated analysis"""
        
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        # Gather recent data
        recent_data = await self._gather_recent_data(caldera_id)
        
        # Initialize analysis components
        analysis_components = {}
        data_sources_used = []
        key_findings = []
        
        # Deformation analysis
        if 'satellite_insar' in recent_data or 'gps' in recent_data:
            deformation_analysis = await self._analyze_deformation(recent_data)
            analysis_components['deformation'] = deformation_analysis
            data_sources_used.extend(['InSAR', 'GPS'])
            
            if deformation_analysis.get('significant_uplift', False):
                key_findings.append("Significant ground deformation detected")
        
        # Seismic analysis
        if 'seismic' in recent_data:
            seismic_analysis = await self._analyze_seismic_activity(recent_data)
            analysis_components['seismic'] = seismic_analysis
            data_sources_used.append('Seismic')
            
            if seismic_analysis.get('increased_activity', False):
                key_findings.append("Elevated seismic activity observed")
        
        # Gas emission analysis
        if 'gas_emissions' in recent_data:
            gas_analysis = await self._analyze_gas_emissions(recent_data)
            analysis_components['gas'] = gas_analysis
            data_sources_used.append('Gas')
            
            if gas_analysis.get('anomalous_emissions', False):
                key_findings.append("Anomalous gas emission patterns detected")
        
        # Thermal analysis
        if 'thermal' in recent_data:
            thermal_analysis = await self._analyze_thermal_activity(recent_data)
            analysis_components['thermal'] = thermal_analysis
            data_sources_used.append('Thermal')
        
        # Integrated risk assessment
        threat_assessment = await self._assess_integrated_threat(analysis_components)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            threat_assessment, analysis_components
        )
        
        # Create result
        analysis_result = IntegratedAnalysisResult(
            caldera_id=caldera_id,
            analysis_timestamp=datetime.now(),
            overall_threat_level=threat_assessment['threat_level'],
            confidence_score=threat_assessment['confidence'],
            key_findings=key_findings,
            data_sources_used=data_sources_used,
            analysis_components=analysis_components,
            recommendations=recommendations,
            next_scheduled_analysis=datetime.now() + timedelta(
                seconds=session['configuration'].acquisition_frequency.get('analysis', 21600)
            ),
            alerts_generated=[]
        )
        
        return analysis_result
    
    async def _gather_recent_data(self, caldera_id: str, hours: int = 24) -> Dict[str, List]:
        """Gather recent processed data from all sources"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_products = self.db_session.query(DataProduct).filter(
            DataProduct.caldera_id == caldera_id,
            DataProduct.acquisition_time >= cutoff_time,
            DataProduct.processing_status == 'processed'
        ).all()
        
        # Organize by data type
        data_by_type = {}
        for product in recent_products:
            data_type = product.data_type
            if data_type not in data_by_type:
                data_by_type[data_type] = []
            
            data_by_type[data_type].append({
                'product_id': product.product_id,
                'acquisition_time': product.acquisition_time,
                'metadata': product.metadata,
                'quality_score': product.quality_score
            })
        
        return data_by_type
    
    async def _analyze_deformation(self, recent_data: Dict) -> Dict:
        """Analyze ground deformation from satellite and GPS data"""
        
        analysis = {
            'max_uplift_mm': 0,
            'max_subsidence_mm': 0,
            'deformation_rate_mm_day': 0,
            'significant_uplift': False,
            'coherent_pattern': False
        }
        
        # Process InSAR data
        if 'satellite_insar' in recent_data:
            insar_products = recent_data['satellite_insar']
            
            for product in insar_products:
                metadata = product['metadata']
                max_deform = metadata.get('max_deformation_mm', 0)
                
                if max_deform > analysis['max_uplift_mm']:
                    analysis['max_uplift_mm'] = max_deform
                
                coherence = metadata.get('coherence_mean', 0)
                if coherence > 0.5:
                    analysis['coherent_pattern'] = True
        
        # Process GPS data
        if 'gps' in recent_data:
            gps_products = recent_data['gps']
            
            for product in gps_products:
                metadata = product['metadata']
                displacement = metadata.get('displacement_vectors', {})
                up_motion = displacement.get('up_mm', 0)
                
                if up_motion > analysis['max_uplift_mm']:
                    analysis['max_uplift_mm'] = up_motion
        
        # Assess significance
        if analysis['max_uplift_mm'] > 20:  # mm
            analysis['significant_uplift'] = True
        
        # Calculate rate (simplified)
        if len(recent_data.get('satellite_insar', [])) > 1:
            analysis['deformation_rate_mm_day'] = analysis['max_uplift_mm'] / 12  # Rough estimate
        
        return analysis
    
    async def _analyze_seismic_activity(self, recent_data: Dict) -> Dict:
        """Analyze seismic activity patterns"""
        
        analysis = {
            'total_events': 0,
            'max_magnitude': 0,
            'b_value': 1.0,
            'increased_activity': False,
            'deep_events_fraction': 0
        }
        
        if 'seismic' in recent_data:
            seismic_products = recent_data['seismic']
            
            total_events = 0
            max_mag = 0
            b_values = []
            
            for product in seismic_products:
                metadata = product['metadata']
                
                events = metadata.get('events_detected', 0)
                total_events += events
                
                mag_range = metadata.get('magnitude_range', [0, 0])
                if len(mag_range) >= 2:
                    max_mag = max(max_mag, mag_range[1])
                
                b_val = metadata.get('b_value', 1.0)
                if b_val > 0:
                    b_values.append(b_val)
            
            analysis['total_events'] = total_events
            analysis['max_magnitude'] = max_mag
            
            if b_values:
                analysis['b_value'] = np.mean(b_values)
            
            # Assess if activity is increased
            if total_events > 50 or max_mag > 3.0:  # Thresholds
                analysis['increased_activity'] = True
        
        return analysis
    
    async def _analyze_gas_emissions(self, recent_data: Dict) -> Dict:
        """Analyze volcanic gas emission patterns"""
        
        analysis = {
            'co2_flux_avg': 0,
            'so2_flux_avg': 0,
            'he3_he4_ratio_avg': 7,
            'anomalous_emissions': False,
            'flux_increase_factor': 1.0
        }
        
        if 'gas_emissions' in recent_data:
            gas_products = recent_data['gas_emissions']
            
            co2_fluxes = []
            so2_fluxes = []
            he_ratios = []
            
            for product in gas_products:
                metadata = product['metadata']
                
                co2_flux = metadata.get('co2_flux_tons_day', 0)
                so2_flux = metadata.get('so2_flux_tons_day', 0)
                he_ratio = metadata.get('he3_he4_ratio', 7)
                
                if co2_flux > 0:
                    co2_fluxes.append(co2_flux)
                if so2_flux > 0:
                    so2_fluxes.append(so2_flux)
                if he_ratio > 0:
                    he_ratios.append(he_ratio)
            
            if co2_fluxes:
                analysis['co2_flux_avg'] = np.mean(co2_fluxes)
            if so2_fluxes:
                analysis['so2_flux_avg'] = np.mean(so2_fluxes)
            if he_ratios:
                analysis['he3_he4_ratio_avg'] = np.mean(he_ratios)
            
            # Check for anomalous emissions
            if analysis['co2_flux_avg'] > 800:  # tons/day threshold
                analysis['anomalous_emissions'] = True
                analysis['flux_increase_factor'] = analysis['co2_flux_avg'] / 300  # Baseline
            
            if analysis['he3_he4_ratio_avg'] > 8:  # High mantle signature
                analysis['anomalous_emissions'] = True
        
        return analysis
    
    async def _analyze_thermal_activity(self, recent_data: Dict) -> Dict:
        """Analyze thermal activity patterns"""
        
        analysis = {
            'max_temperature_c': 0,
            'thermal_anomaly_area_km2': 0,
            'temperature_increase': False
        }
        
        if 'thermal' in recent_data:
            thermal_products = recent_data['thermal']
            
            max_temps = []
            
            for product in thermal_products:
                metadata = product['metadata']
                temp = metadata.get('temperature_c', 0)
                if temp > 0:
                    max_temps.append(temp)
            
            if max_temps:
                analysis['max_temperature_c'] = max(max_temps)
                
                if analysis['max_temperature_c'] > 100:  # Threshold
                    analysis['temperature_increase'] = True
        
        return analysis
    
    async def _assess_integrated_threat(self, analysis_components: Dict) -> Dict:
        """Assess integrated threat level from all analysis components"""
        
        # Initialize threat factors
        threat_factors = []
        
        # Deformation factor
        if 'deformation' in analysis_components:
            deform = analysis_components['deformation']
            if deform.get('significant_uplift', False):
                threat_factors.append(0.7)
            elif deform.get('max_uplift_mm', 0) > 10:
                threat_factors.append(0.4)
            else:
                threat_factors.append(0.1)
        
        # Seismic factor
        if 'seismic' in analysis_components:
            seismic = analysis_components['seismic']
            if seismic.get('increased_activity', False):
                threat_factors.append(0.6)
            elif seismic.get('total_events', 0) > 20:
                threat_factors.append(0.3)
            else:
                threat_factors.append(0.1)
        
        # Gas factor
        if 'gas' in analysis_components:
            gas = analysis_components['gas']
            if gas.get('anomalous_emissions', False):
                threat_factors.append(0.5)
            else:
                threat_factors.append(0.1)
        
        # Thermal factor
        if 'thermal' in analysis_components:
            thermal = analysis_components['thermal']
            if thermal.get('temperature_increase', False):
                threat_factors.append(0.4)
            else:
                threat_factors.append(0.1)
        
        # Calculate overall threat score
        if threat_factors:
            threat_score = np.mean(threat_factors)
        else:
            threat_score = 0.1
        
        # Determine threat level
        if threat_score > 0.7:
            threat_level = AlertSeverity.CRITICAL
        elif threat_score > 0.5:
            threat_level = AlertSeverity.HIGH
        elif threat_score > 0.3:
            threat_level = AlertSeverity.MEDIUM
        else:
            threat_level = AlertSeverity.LOW
        
        # Calculate confidence based on data availability
        data_sources = len(analysis_components)
        confidence = min(1.0, data_sources / 4)  # Max confidence with 4+ sources
        
        return {
            'threat_level': threat_level,
            'threat_score': threat_score,
            'confidence': confidence,
            'contributing_factors': threat_factors
        }
    
    async def _generate_recommendations(self, threat_assessment: Dict,
                                     analysis_components: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        threat_level = threat_assessment['threat_level']
        
        # General recommendations based on threat level
        if threat_level == AlertSeverity.CRITICAL:
            recommendations.extend([
                "URGENT: Activate emergency monitoring protocols",
                "Consider evacuation planning for high-risk zones",
                "Increase monitoring frequency to continuous",
                "Coordinate with emergency management authorities"
            ])
        elif threat_level == AlertSeverity.HIGH:
            recommendations.extend([
                "Increase monitoring frequency",
                "Deploy additional monitoring equipment",
                "Alert local authorities",
                "Review emergency response plans"
            ])
        elif threat_level == AlertSeverity.MEDIUM:
            recommendations.extend([
                "Enhanced monitoring recommended",
                "Conduct detailed analysis of anomalies",
                "Inform relevant stakeholders"
            ])
        else:
            recommendations.extend([
                "Continue routine monitoring",
                "Maintain data quality standards"
            ])
        
        # Specific recommendations based on analysis
        if 'deformation' in analysis_components:
            deform = analysis_components['deformation']
            if deform.get('significant_uplift', False):
                recommendations.append("Focus on deformation source modeling")
        
        if 'seismic' in analysis_components:
            seismic = analysis_components['seismic']
            if seismic.get('increased_activity', False):
                recommendations.append("Deploy additional seismic stations")
        
        if 'gas' in analysis_components:
            gas = analysis_components['gas']
            if gas.get('anomalous_emissions', False):
                recommendations.append("Increase gas monitoring frequency")
        
        return recommendations
    
    async def _evaluate_alerts(self, session_id: str, 
                             analysis_result: IntegratedAnalysisResult) -> List[Dict]:
        """Evaluate analysis results for alert conditions"""
        
        alerts = []
        session = self.active_sessions[session_id]
        caldera_id = session['caldera_id']
        
        # Check threat level alert
        if analysis_result.overall_threat_level in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            alert = {
                'alert_type': 'threat_level',
                'severity': analysis_result.overall_threat_level.value,
                'message': f"Elevated threat level detected for {caldera_id}",
                'trigger_conditions': {
                    'threat_level': analysis_result.overall_threat_level.value,
                    'confidence': analysis_result.confidence_score
                },
                'timestamp': datetime.now()
            }
            alerts.append(alert)
            
            # Store in database
            alert_record = AlertEvent(
                caldera_id=caldera_id,
                alert_type=alert['alert_type'],
                severity=alert['severity'],
                trigger_conditions=alert['trigger_conditions'],
                alert_message=alert['message']
            )
            self.db_session.add(alert_record)
        
        # Check specific parameter alerts
        thresholds = self.config['alert_thresholds']
        
        # Deformation alerts
        if 'deformation' in analysis_result.analysis_components:
            deform = analysis_result.analysis_components['deformation']
            uplift = deform.get('max_uplift_mm', 0)
            
            if uplift > thresholds.get('deformation_rate_mm_day', 10) * 30:  # Monthly threshold
                alert = {
                    'alert_type': 'excessive_deformation',
                    'severity': 'high',
                    'message': f"Excessive ground deformation detected: {uplift:.1f} mm",
                    'trigger_conditions': {'uplift_mm': uplift},
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
        
        # Commit alert records
        self.db_session.commit()
        
        return alerts
    
    async def _monitor_alerts(self, session_id: str):
        """Monitor and handle alerts"""
        
        while self.active_sessions.get(session_id, {}).get('status') == 'active':
            try:
                session = self.active_sessions[session_id]
                caldera_id = session['caldera_id']
                
                # Check for new unacknowledged alerts
                recent_alerts = self.db_session.query(AlertEvent).filter(
                    AlertEvent.caldera_id == caldera_id,
                    AlertEvent.timestamp >= datetime.now() - timedelta(minutes=15),
                    AlertEvent.acknowledged == False
                ).all()
                
                for alert in recent_alerts:
                    await self._handle_alert(session_id, alert)
                    alert.acknowledged = True
                
                self.db_session.commit()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _handle_alert(self, session_id: str, alert: AlertEvent):
        """Handle individual alert"""
        
        session = self.active_sessions[session_id]
        
        self.logger.warning(f"ALERT: {alert.alert_message}")
        
        # Automated responses
        severity = AlertSeverity(alert.severity)
        
        if severity == AlertSeverity.CRITICAL:
            # Critical alert responses
            await self._execute_emergency_response(session_id)
            
        elif severity == AlertSeverity.HIGH:
            # High alert responses
            await self._increase_monitoring_frequency(session_id)
            await self._notify_stakeholders(session_id, alert)
        
        # Log alert handling
        alert.response_actions = {
            'automated_response': True,
            'response_time': datetime.now().isoformat(),
            'actions_taken': ['logged', 'stakeholder_notification']
        }
    
    async def _execute_emergency_response(self, session_id: str):
        """Execute emergency response protocols"""
        
        session = self.active_sessions[session_id]
        
        self.logger.critical(f"EXECUTING EMERGENCY RESPONSE for {session['caldera_id']}")
        
        # Switch to emergency monitoring mode
        session['configuration'].monitoring_type = MonitoringType.EMERGENCY
        
        # Increase data acquisition frequency
        # (In practice, would interface with actual systems)
        
        # Alert all stakeholders
        # (In practice, would send notifications)
    
    async def _increase_monitoring_frequency(self, session_id: str):
        """Increase monitoring frequency"""
        
        session = self.active_sessions[session_id]
        
        # Increase frequencies
        if session['configuration'].monitoring_type == MonitoringType.CONTINUOUS:
            session['configuration'].monitoring_type = MonitoringType.CAMPAIGN
        elif session['configuration'].monitoring_type == MonitoringType.CAMPAIGN:
            session['configuration'].monitoring_type = MonitoringType.EMERGENCY
        
        self.logger.info(f"Increased monitoring frequency to {session['configuration'].monitoring_type.value}")
    
    async def _notify_stakeholders(self, session_id: str, alert: AlertEvent):
        """Notify relevant stakeholders"""
        
        # In practice, would send emails/SMS/push notifications
        self.logger.info(f"Stakeholder notification sent for alert: {alert.alert_message}")
    
    def create_monitoring_dashboard(self, session_id: str) -> go.Figure:
        """Create comprehensive monitoring dashboard"""
        
        session = self.active_sessions[session_id]
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=[['xy', 'xy'], ['xy', 'xy'], ['xy', 'xy']],
            subplot_titles=[
                'Alert Level Timeline',
                'Data Acquisition Status',
                'Threat Assessment',
                'Processing Queue',
                'System Status',
                'Recent Activity'
            ]
        )
        
        # Alert level timeline (placeholder)
        time_points = pd.date_range(start=datetime.now()-timedelta(hours=24), 
                                  end=datetime.now(), freq='1H')
        alert_levels = np.random.choice([1, 2, 3, 4], size=len(time_points), p=[0.6, 0.2, 0.15, 0.05])
        
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=alert_levels,
                mode='lines+markers',
                name='Alert Level',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        # Data acquisition status
        data_types = ['Satellite', 'Seismic', 'GPS', 'Gas', 'Thermal']
        acquisition_status = [0.9, 0.95, 0.8, 0.7, 0.85]  # Uptime fractions
        
        fig.add_trace(
            go.Bar(
                x=data_types,
                y=acquisition_status,
                name='Data Uptime',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Threat assessment pie chart
        threat_components = ['Deformation', 'Seismic', 'Gas', 'Thermal', 'Other']
        threat_contributions = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        fig.add_trace(
            go.Pie(
                labels=threat_components,
                values=threat_contributions,
                name="Threat Factors"
            ),
            row=2, col=1
        )
        
        # Processing queue status
        queue_times = ['<1min', '1-5min', '5-15min', '>15min']
        queue_counts = [5, 12, 3, 1]
        
        fig.add_trace(
            go.Bar(
                x=queue_times,
                y=queue_counts,
                name='Processing Queue',
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Integrated Monitoring Dashboard - {session['caldera_id']}",
            showlegend=True,
            height=900
        )
        
        return fig

# Example usage
async def example_integrated_monitoring():
    """Example of integrated caldera monitoring system"""
    
    print("Integrated Caldera Monitoring System Example")
    
    # Initialize system
    monitor = IntegratedCalderaMonitoring()
    
    # Create monitoring configuration
    config = MonitoringConfiguration(
        caldera_id="yellowstone",
        monitoring_type=MonitoringType.CONTINUOUS,
        data_sources=[
            DataType.SATELLITE_INSAR,
            DataType.SEISMIC,
            DataType.GPS,
            DataType.GAS_EMISSIONS
        ],
        acquisition_frequency={
            'satellite': 12,
            'ground': 1,
            'analysis': 6
        },
        processing_priorities={
            'emergency_data': 1,
            'deformation': 2,
            'seismic': 2,
            'gas': 3
        },
        alert_thresholds={
            'deformation_rate': 10.0,
            'seismic_energy': 5.0,
            'gas_flux': 3.0
        },
        quality_requirements={
            'coherence_min': 0.3,
            'precision_mm': 5
        },
        backup_systems=['manual_processing'],
        automated_responses={
            'high_alert': ['increase_frequency', 'notify_authorities'],
            'critical_alert': ['emergency_mode', 'evacuation_support']
        }
    )
    
    # Initialize monitoring session
    print("Initializing monitoring session...")
    session_id = await monitor.initialize_monitoring_session("yellowstone", config)
    print(f"Monitoring session {session_id} started")
    
    # Simulate some monitoring time
    print("Monitoring active... (simulating 30 seconds)")
    await asyncio.sleep(30)
    
    # Check session status
    session = monitor.active_sessions[session_id]
    print(f"Session status: {session['status']}")
    print(f"Alert level: {session['alert_level'].value}")
    
    if session.get('last_analysis'):
        analysis = session['last_analysis']
        print(f"Last analysis threat level: {analysis.overall_threat_level.value}")
        print(f"Key findings: {', '.join(analysis.key_findings) if analysis.key_findings else 'None'}")
    
    # Create dashboard
    print("Generating monitoring dashboard...")
    dashboard = monitor.create_monitoring_dashboard(session_id)
    dashboard.write_html("integrated_monitoring_dashboard.html")
    print("Dashboard saved to integrated_monitoring_dashboard.html")
    
    print("Integrated monitoring example completed!")

if __name__ == "__main__":
    asyncio.run(example_integrated_monitoring())