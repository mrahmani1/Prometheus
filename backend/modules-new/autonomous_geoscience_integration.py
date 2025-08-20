"""
Autonomous Geoscience Integration System
Integrates the Autonomous Geoscience AI system with Digital Twin Earth and Prometheus infrastructure
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import time
import uuid
from pathlib import Path

# Import core modules
try:
    from .autonomous_geoscience import (
        ClosedLoopDiscoverySystem, HypothesisGenerator, Hypothesis, Experiment, 
        DiscoveryInsight, HypothesisType, DiscoveryStage, ExperimentType
    )
    from .digital_twin_earth import DigitalTwinEarth, EarthSystemState, ModelConfiguration, ObservationData
    from .digital_twin_integration import DigitalTwinEarthIntegrator, IntegrationConfig
    from .digital_twin_api import DigitalTwinEarthAPI, APIConfig
    from .digital_twin_visualization import DigitalTwinVisualizationDashboard, VisualizationConfig
    from .stream_processing import StreamEvent, StreamEventType, KafkaStreamManager
    from .database_integration import PostgreSQLGISManager, DatabaseConfig
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core Prometheus modules not fully available.")

# ML and scientific computing imports
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import DBSCAN
    from scipy import stats
    import networkx as nx
    SCIENTIFIC_AVAILABLE = True
except ImportError:
    SCIENTIFIC_AVAILABLE = False
    warnings.warn("Scientific computing libraries not available.")

warnings.filterwarnings('ignore')

class AutonomousGeoscienceMode(Enum):
    DISCOVERY = "discovery"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    KNOWLEDGE_BUILDING = "knowledge_building"
    REAL_TIME_ANALYSIS = "real_time_analysis"

class IntegrationEvent(Enum):
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    EXPERIMENT_PLANNED = "experiment_planned"
    EXPERIMENT_COMPLETED = "experiment_completed"
    INSIGHT_DISCOVERED = "insight_discovered"
    KNOWLEDGE_UPDATED = "knowledge_updated"

@dataclass
class AutonomousGeoscienceConfig:
    """Configuration for autonomous geoscience integration"""
    enable_real_time_discovery: bool = True
    enable_hypothesis_testing: bool = True
    enable_knowledge_integration: bool = True
    discovery_cycle_interval: int = 1800  # 30 minutes
    max_concurrent_experiments: int = 3
    hypothesis_confidence_threshold: float = 0.6
    experiment_priority_threshold: float = 0.7
    real_time_analysis_interval: int = 300  # 5 minutes
    enable_cross_domain_insights: bool = True
    enable_automated_alerts: bool = True

@dataclass 
class DiscoverySession:
    """Represents an autonomous discovery session"""
    session_id: str
    mode: AutonomousGeoscienceMode
    start_time: datetime
    end_time: Optional[datetime] = None
    hypotheses_generated: int = 0
    experiments_completed: int = 0
    insights_discovered: int = 0
    knowledge_items_updated: int = 0
    session_status: str = "active"  # active, paused, completed, error
    focus_domains: List[str] = field(default_factory=list)

class AutonomousGeoscienceIntegrator:
    """Main integration system for autonomous geoscience"""
    
    def __init__(self, config: AutonomousGeoscienceConfig = None):
        self.config = config or AutonomousGeoscienceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.digital_twin_integrator = None
        self.discovery_system = None
        self.hypothesis_generator = None
        self.api_system = None
        self.visualization_dashboard = None
        
        # Integration state
        self.is_initialized = False
        self.current_session = None
        self.discovery_history = []
        self.active_experiments = {}
        self.knowledge_cache = {}
        
        # Event management
        self.event_handlers = {}
        self.integration_events = []
        
        # Performance tracking
        self.performance_metrics = {
            'discovery_rate': 0.0,
            'hypothesis_success_rate': 0.0,
            'experiment_completion_rate': 0.0,
            'insight_significance_score': 0.0
        }
        
    async def initialize(self, 
                        digital_twin_integrator: DigitalTwinEarthIntegrator = None,
                        api_system: DigitalTwinEarthAPI = None,
                        visualization_dashboard: DigitalTwinVisualizationDashboard = None):
        """Initialize the autonomous geoscience integration system"""
        try:
            # Set core components
            self.digital_twin_integrator = digital_twin_integrator
            self.api_system = api_system
            self.visualization_dashboard = visualization_dashboard
            
            # Initialize discovery system
            self.discovery_system = ClosedLoopDiscoverySystem()
            self.hypothesis_generator = HypothesisGenerator()
            
            # Setup event handlers
            await self._setup_event_handlers()
            
            # Start real-time discovery if enabled
            if self.config.enable_real_time_discovery:
                await self._start_real_time_discovery()
            
            # Start real-time analysis if enabled  
            if self.config.real_time_analysis_interval > 0:
                await self._start_real_time_analysis()
            
            self.is_initialized = True
            self.logger.info("Autonomous geoscience integration system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize autonomous geoscience integration: {e}")
            raise
    
    async def _setup_event_handlers(self):
        """Setup event handlers for integration events"""
        self.event_handlers = {
            IntegrationEvent.HYPOTHESIS_GENERATED: self._handle_hypothesis_generated,
            IntegrationEvent.EXPERIMENT_PLANNED: self._handle_experiment_planned,
            IntegrationEvent.EXPERIMENT_COMPLETED: self._handle_experiment_completed,
            IntegrationEvent.INSIGHT_DISCOVERED: self._handle_insight_discovered,
            IntegrationEvent.KNOWLEDGE_UPDATED: self._handle_knowledge_updated
        }
    
    async def _start_real_time_discovery(self):
        """Start real-time discovery loop"""
        async def discovery_loop():
            while self.is_initialized:
                try:
                    if not self.current_session or self.current_session.session_status != "active":
                        await self.start_discovery_session(AutonomousGeoscienceMode.DISCOVERY)
                    
                    # Run discovery cycle
                    await self._run_autonomous_discovery_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.config.discovery_cycle_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time discovery loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        asyncio.create_task(discovery_loop())
    
    async def _start_real_time_analysis(self):
        """Start real-time data analysis loop"""
        async def analysis_loop():
            while self.is_initialized:
                try:
                    # Analyze current Digital Twin Earth state
                    if self.digital_twin_integrator and self.digital_twin_integrator.digital_twin:
                        await self._analyze_current_earth_state()
                    
                    # Check for anomalies and patterns
                    await self._detect_real_time_anomalies()
                    
                    # Generate real-time insights
                    await self._generate_real_time_insights()
                    
                    await asyncio.sleep(self.config.real_time_analysis_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time analysis loop: {e}")
                    await asyncio.sleep(30)
        
        asyncio.create_task(analysis_loop())
    
    async def start_discovery_session(self, mode: AutonomousGeoscienceMode, 
                                    focus_domains: List[str] = None) -> str:
        """Start a new discovery session"""
        try:
            # End current session if active
            if self.current_session and self.current_session.session_status == "active":
                await self.end_discovery_session()
            
            # Create new session
            session = DiscoverySession(
                session_id=str(uuid.uuid4()),
                mode=mode,
                start_time=datetime.utcnow(),
                focus_domains=focus_domains or []
            )
            
            self.current_session = session
            self.logger.info(f"Started discovery session {session.session_id} in {mode.value} mode")
            
            # Configure discovery system for this session
            await self._configure_discovery_for_session(session)
            
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"Error starting discovery session: {e}")
            raise
    
    async def end_discovery_session(self) -> Optional[DiscoverySession]:
        """End the current discovery session"""
        if not self.current_session:
            return None
        
        try:
            self.current_session.end_time = datetime.utcnow()
            self.current_session.session_status = "completed"
            
            # Archive session
            self.discovery_history.append(self.current_session)
            
            # Update performance metrics
            self._update_session_metrics(self.current_session)
            
            completed_session = self.current_session
            self.current_session = None
            
            self.logger.info(f"Ended discovery session {completed_session.session_id}")
            return completed_session
            
        except Exception as e:
            self.logger.error(f"Error ending discovery session: {e}")
            if self.current_session:
                self.current_session.session_status = "error"
            return None
    
    async def _configure_discovery_for_session(self, session: DiscoverySession):
        """Configure discovery system for specific session"""
        if session.mode == AutonomousGeoscienceMode.DISCOVERY:
            # Configure for broad discovery
            self.discovery_system.config.enable_cross_domain_insights = True
            self.hypothesis_generator.config.enable_deep_reasoning = True
            
        elif session.mode == AutonomousGeoscienceMode.HYPOTHESIS_TESTING:
            # Configure for focused hypothesis testing
            self.discovery_system.config.enable_autonomous_experiments = True
            self.discovery_system.config.max_concurrent_discoveries = self.config.max_concurrent_experiments
            
        elif session.mode == AutonomousGeoscienceMode.KNOWLEDGE_BUILDING:
            # Configure for knowledge integration
            self.discovery_system.config.enable_knowledge_integration = True
            
        elif session.mode == AutonomousGeoscienceMode.REAL_TIME_ANALYSIS:
            # Configure for real-time analysis
            self.discovery_system.config.discovery_cycle_interval = self.config.real_time_analysis_interval
    
    async def _run_autonomous_discovery_cycle(self):
        """Run a single autonomous discovery cycle"""
        try:
            if not self.current_session:
                return
            
            # Gather current data context from Digital Twin Earth
            data_context = await self._gather_integrated_data_context()
            
            # Generate hypotheses using AI system
            hypotheses = await self.hypothesis_generator.generate_hypotheses(
                data_context, 
                self.current_session.focus_domains
            )
            
            # Filter hypotheses by confidence threshold
            high_confidence_hypotheses = [
                h for h in hypotheses 
                if h.confidence_score >= self.config.hypothesis_confidence_threshold
            ]
            
            # Process each hypothesis
            for hypothesis in high_confidence_hypotheses:
                await self._emit_integration_event(IntegrationEvent.HYPOTHESIS_GENERATED, {
                    'hypothesis': hypothesis,
                    'session_id': self.current_session.session_id
                })
                
                self.current_session.hypotheses_generated += 1
            
            # Run discovery system cycle if we have the Digital Twin integrator
            if self.digital_twin_integrator:
                await self.discovery_system._run_discovery_cycle(self.digital_twin_integrator)
            
            # Update session metrics
            self.current_session.experiments_completed = len(self.active_experiments)
            
        except Exception as e:
            self.logger.error(f"Error in autonomous discovery cycle: {e}")
    
    async def _gather_integrated_data_context(self) -> Dict[str, Any]:
        """Gather comprehensive data context from all integrated systems"""
        data_context = {
            'time_series': {},
            'spatial': {},
            'numerical': {},
            'contextual': {},
            'streaming': {},
            'predictions': {}
        }
        
        try:
            # Get data from Digital Twin Earth
            if self.digital_twin_integrator and self.digital_twin_integrator.digital_twin:
                earth_state = self.digital_twin_integrator.digital_twin.get_current_state()
                
                if earth_state:
                    # Time series data
                    data_context['time_series'] = {
                        'atmosphere_temperature': self._extract_time_series_values(
                            earth_state.atmosphere_state, 'temperature'
                        ),
                        'hydrosphere_level': self._extract_time_series_values(
                            earth_state.hydrosphere_state, 'sea_level'
                        ),
                        'lithosphere_stress': self._extract_time_series_values(
                            earth_state.lithosphere_state, 'stress_field'
                        ),
                        'biosphere_index': self._extract_time_series_values(
                            earth_state.biosphere_state, 'vegetation_index'
                        )
                    }
                    
                    # Spatial data
                    data_context['spatial'] = {
                        'resolution': earth_state.spatial_resolution,
                        'coverage': 'global',
                        'coordinate_system': 'geographic'
                    }
                    
                    # Numerical data
                    data_context['numerical'] = {
                        'temperature': earth_state.atmosphere_state.get('temperature', 0),
                        'pressure': earth_state.atmosphere_state.get('pressure', 0),
                        'sea_level': earth_state.hydrosphere_state.get('sea_level', 0),
                        'stress_field': earth_state.lithosphere_state.get('stress_field', 0),
                        'vegetation_index': earth_state.biosphere_state.get('vegetation_index', 0),
                        'ice_coverage': earth_state.cryosphere_state.get('ice_coverage', 0),
                        'activity_level': earth_state.anthroposphere_state.get('activity_level', 0)
                    }
                    
                    # Uncertainty information
                    data_context['uncertainty'] = earth_state.uncertainty_metrics
                
                # Get integration status
                integration_status = await self.digital_twin_integrator.get_integration_status()
                data_context['contextual'] = {
                    'integration_health': integration_status['metrics']['integration_health'],
                    'sync_latency': integration_status['metrics']['sync_latency_ms'],
                    'data_throughput': integration_status['metrics']['data_throughput_mbps'],
                    'active_integrations': integration_status['active_integrations']
                }
                
                # Get current predictions
                predictions = await self.digital_twin_integrator.get_current_predictions()
                data_context['predictions'] = {
                    'count': len(predictions),
                    'average_confidence': np.mean([p.confidence_level for p in predictions]) if predictions else 0,
                    'time_horizons': [p.time_horizon.total_seconds() for p in predictions]
                }
            
            # Add historical data from knowledge cache
            data_context['historical'] = self.knowledge_cache.get('recent_patterns', {})
            
            # Add streaming data if available
            if hasattr(self.digital_twin_integrator, 'stream_manager'):
                data_context['streaming'] = {
                    'active_streams': True,
                    'event_rate': 'moderate'  # Would be calculated from actual stream metrics
                }
            
            return data_context
            
        except Exception as e:
            self.logger.error(f"Error gathering integrated data context: {e}")
            return data_context
    
    def _extract_time_series_values(self, state_dict: Dict[str, Any], key: str) -> List[float]:
        """Extract time series values from state dictionary"""
        value = state_dict.get(key, 0)
        
        # If it's already a list, return it
        if isinstance(value, list):
            return value
        
        # Otherwise, create a mock time series (in real implementation, would use historical data)
        return [value + np.random.normal(0, 0.1) for _ in range(10)]
    
    async def _analyze_current_earth_state(self):
        """Analyze current Earth system state for autonomous insights"""
        try:
            if not self.digital_twin_integrator or not self.digital_twin_integrator.digital_twin:
                return
            
            current_state = self.digital_twin_integrator.digital_twin.get_current_state()
            if not current_state:
                return
            
            # Analyze for anomalies
            anomalies = self._detect_state_anomalies(current_state)
            
            # Analyze for patterns
            patterns = self._detect_state_patterns(current_state)
            
            # Generate automatic hypotheses if significant patterns found
            if anomalies or patterns:
                context = {
                    'anomalies': anomalies,
                    'patterns': patterns,
                    'current_state': current_state
                }
                
                auto_hypotheses = await self.hypothesis_generator.generate_hypotheses(context)
                
                # Process high-confidence automatic hypotheses
                for hypothesis in auto_hypotheses:
                    if hypothesis.confidence_score > 0.8:  # High confidence threshold for auto-generated
                        await self._emit_integration_event(IntegrationEvent.HYPOTHESIS_GENERATED, {
                            'hypothesis': hypothesis,
                            'source': 'real_time_analysis',
                            'trigger': 'state_analysis'
                        })
            
        except Exception as e:
            self.logger.error(f"Error analyzing current Earth state: {e}")
    
    def _detect_state_anomalies(self, earth_state: EarthSystemState) -> List[Dict[str, Any]]:
        """Detect anomalies in current Earth system state"""
        anomalies = []
        
        try:
            # Check each component for anomalies
            components = [
                ('atmosphere', earth_state.atmosphere_state),
                ('hydrosphere', earth_state.hydrosphere_state),
                ('lithosphere', earth_state.lithosphere_state),
                ('biosphere', earth_state.biosphere_state),
                ('cryosphere', earth_state.cryosphere_state),
                ('anthroposphere', earth_state.anthroposphere_state)
            ]
            
            for component_name, component_state in components:
                for variable, value in component_state.items():
                    if isinstance(value, (int, float)):
                        # Simple anomaly detection based on expected ranges
                        expected_ranges = self._get_expected_ranges()
                        
                        if component_name in expected_ranges and variable in expected_ranges[component_name]:
                            min_val, max_val = expected_ranges[component_name][variable]
                            
                            if value < min_val or value > max_val:
                                anomalies.append({
                                    'component': component_name,
                                    'variable': variable,
                                    'value': value,
                                    'expected_range': (min_val, max_val),
                                    'severity': abs(value - np.mean([min_val, max_val])) / (max_val - min_val)
                                })
            
        except Exception as e:
            self.logger.error(f"Error detecting state anomalies: {e}")
        
        return anomalies
    
    def _get_expected_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get expected ranges for Earth system variables"""
        return {
            'atmosphere': {
                'temperature': (-50, 50),  # Celsius
                'pressure': (800, 1200),   # hPa
            },
            'hydrosphere': {
                'sea_level': (-2, 2),      # meters relative to mean
            },
            'lithosphere': {
                'stress_field': (-100, 100),  # MPa
            },
            'biosphere': {
                'vegetation_index': (0, 1),
            },
            'cryosphere': {
                'ice_coverage': (0, 1),
            },
            'anthroposphere': {
                'activity_level': (0, 1),
            }
        }
    
    def _detect_state_patterns(self, earth_state: EarthSystemState) -> List[Dict[str, Any]]:
        """Detect patterns in current Earth system state"""
        patterns = []
        
        try:
            # Look for correlations between components
            state_values = {}
            
            for component_name, component_state in [
                ('atmosphere', earth_state.atmosphere_state),
                ('hydrosphere', earth_state.hydrosphere_state),
                ('lithosphere', earth_state.lithosphere_state),
                ('biosphere', earth_state.biosphere_state),
                ('cryosphere', earth_state.cryosphere_state),
                ('anthroposphere', earth_state.anthroposphere_state)
            ]:
                for variable, value in component_state.items():
                    if isinstance(value, (int, float)):
                        state_values[f"{component_name}_{variable}"] = value
            
            # Simple pattern detection (would be more sophisticated in practice)
            if len(state_values) > 1:
                variable_names = list(state_values.keys())
                values = list(state_values.values())
                
                # Check for high/low value clusters
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                if std_value > 0:
                    high_vars = [name for name, val in state_values.items() if val > mean_value + std_value]
                    low_vars = [name for name, val in state_values.items() if val < mean_value - std_value]
                    
                    if len(high_vars) > 1:
                        patterns.append({
                            'type': 'high_value_cluster',
                            'variables': high_vars,
                            'strength': len(high_vars) / len(variable_names)
                        })
                    
                    if len(low_vars) > 1:
                        patterns.append({
                            'type': 'low_value_cluster',
                            'variables': low_vars,
                            'strength': len(low_vars) / len(variable_names)
                        })
        
        except Exception as e:
            self.logger.error(f"Error detecting state patterns: {e}")
        
        return patterns
    
    async def _detect_real_time_anomalies(self):
        """Detect real-time anomalies across all integrated systems"""
        try:
            anomalies = []
            
            # Check Digital Twin Earth anomalies
            if self.digital_twin_integrator:
                integration_status = await self.digital_twin_integrator.get_integration_status()
                
                # Check system health
                if integration_status['metrics']['integration_health'] == 'CRITICAL':
                    anomalies.append({
                        'type': 'system_health',
                        'severity': 'critical',
                        'description': 'Digital Twin integration system in critical state'
                    })
                
                # Check latency anomalies
                if integration_status['metrics']['sync_latency_ms'] > 5000:  # > 5 seconds
                    anomalies.append({
                        'type': 'performance',
                        'severity': 'high',
                        'description': f"High sync latency: {integration_status['metrics']['sync_latency_ms']}ms"
                    })
            
            # Process detected anomalies
            for anomaly in anomalies:
                if self.config.enable_automated_alerts:
                    await self._emit_integration_event(IntegrationEvent.INSIGHT_DISCOVERED, {
                        'insight_type': 'anomaly_detected',
                        'anomaly': anomaly,
                        'timestamp': datetime.utcnow()
                    })
        
        except Exception as e:
            self.logger.error(f"Error detecting real-time anomalies: {e}")
    
    async def _generate_real_time_insights(self):
        """Generate real-time insights from current system state"""
        try:
            insights = []
            
            # Analyze system performance trends
            if self.digital_twin_integrator:
                status = await self.digital_twin_integrator.get_integration_status()
                
                # Performance insight
                throughput = status['metrics']['data_throughput_mbps']
                if throughput > 10:  # High throughput threshold
                    insights.append({
                        'type': 'performance_insight',
                        'description': f'High data throughput detected: {throughput:.2f} MB/s',
                        'significance': 'medium',
                        'implications': ['System processing large amounts of data', 'Potential for rich analysis']
                    })
            
            # Process insights
            for insight in insights:
                await self._emit_integration_event(IntegrationEvent.INSIGHT_DISCOVERED, {
                    'insight': insight,
                    'source': 'real_time_generation'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating real-time insights: {e}")
    
    async def _emit_integration_event(self, event_type: IntegrationEvent, data: Dict[str, Any]):
        """Emit integration event and call appropriate handler"""
        try:
            event = {
                'type': event_type,
                'timestamp': datetime.utcnow(),
                'data': data
            }
            
            # Add to event history
            self.integration_events.append(event)
            
            # Call event handler
            if event_type in self.event_handlers:
                await self.event_handlers[event_type](event)
            
            # Broadcast to API if available
            if self.api_system and hasattr(self.api_system, '_broadcast_to_websockets'):
                await self.api_system._broadcast_to_websockets(
                    f'autonomous_geoscience_{event_type.value}',
                    data
                )
        
        except Exception as e:
            self.logger.error(f"Error emitting integration event: {e}")
    
    async def _handle_hypothesis_generated(self, event: Dict[str, Any]):
        """Handle hypothesis generated event"""
        try:
            hypothesis = event['data']['hypothesis']
            
            # Store hypothesis in knowledge cache
            self.knowledge_cache[hypothesis.hypothesis_id] = hypothesis
            
            # Log hypothesis generation
            self.logger.info(f"Hypothesis generated: {hypothesis.statement} (confidence: {hypothesis.confidence_score:.2f})")
            
            # If hypothesis is high-confidence and testable, plan experiment
            if (hypothesis.confidence_score > self.config.experiment_priority_threshold and
                len(hypothesis.testable_predictions) > 0):
                
                await self._emit_integration_event(IntegrationEvent.EXPERIMENT_PLANNED, {
                    'hypothesis_id': hypothesis.hypothesis_id,
                    'auto_planned': True
                })
        
        except Exception as e:
            self.logger.error(f"Error handling hypothesis generated event: {e}")
    
    async def _handle_experiment_planned(self, event: Dict[str, Any]):
        """Handle experiment planned event"""
        try:
            self.logger.info("Experiment planned for hypothesis testing")
            
            # Update current session metrics
            if self.current_session:
                self.current_session.experiments_completed += 1
        
        except Exception as e:
            self.logger.error(f"Error handling experiment planned event: {e}")
    
    async def _handle_experiment_completed(self, event: Dict[str, Any]):
        """Handle experiment completed event"""
        try:
            experiment_id = event['data'].get('experiment_id')
            results = event['data'].get('results', {})
            
            self.logger.info(f"Experiment {experiment_id} completed")
            
            # Update performance metrics
            if results.get('status') == 'completed':
                self.performance_metrics['experiment_completion_rate'] += 0.1
            
        except Exception as e:
            self.logger.error(f"Error handling experiment completed event: {e}")
    
    async def _handle_insight_discovered(self, event: Dict[str, Any]):
        """Handle insight discovered event"""
        try:
            insight_data = event['data']
            
            self.logger.info(f"Insight discovered: {insight_data.get('description', 'Unknown insight')}")
            
            # Update session metrics
            if self.current_session:
                self.current_session.insights_discovered += 1
            
            # Update performance metrics
            significance = insight_data.get('significance', 'low')
            if significance == 'high':
                self.performance_metrics['insight_significance_score'] += 0.2
            elif significance == 'medium':
                self.performance_metrics['insight_significance_score'] += 0.1
        
        except Exception as e:
            self.logger.error(f"Error handling insight discovered event: {e}")
    
    async def _handle_knowledge_updated(self, event: Dict[str, Any]):
        """Handle knowledge updated event"""
        try:
            self.logger.info("Knowledge base updated with new scientific knowledge")
            
            # Update session metrics
            if self.current_session:
                self.current_session.knowledge_items_updated += 1
        
        except Exception as e:
            self.logger.error(f"Error handling knowledge updated event: {e}")
    
    def _update_session_metrics(self, session: DiscoverySession):
        """Update performance metrics based on completed session"""
        try:
            if session.end_time and session.start_time:
                duration_hours = (session.end_time - session.start_time).total_seconds() / 3600
                
                # Calculate discovery rate
                if duration_hours > 0:
                    discovery_rate = session.insights_discovered / duration_hours
                    self.performance_metrics['discovery_rate'] = (
                        self.performance_metrics['discovery_rate'] * 0.8 + discovery_rate * 0.2
                    )
                
                # Comprehensive Hypothesis Success Rate Analysis
                # Multi-dimensional performance evaluation of hypothesis-to-discovery pipeline
                if session.hypotheses_generated > 0:
                    # 1. Basic completion rate
                    completion_rate = session.experiments_completed / session.hypotheses_generated
                    
                    # 2. Success quality assessment
                    # Weight by hypothesis confidence and experimental validation strength
                    weighted_success_rate = 0.0
                    confidence_weighted_total = 0.0
                    
                    for hypothesis in session.generated_hypotheses:
                        # Extract hypothesis confidence (0-1 scale)
                        confidence = getattr(hypothesis, 'confidence_score', 0.5)
                        
                        # Determine success based on multiple criteria
                        success_score = 0.0
                        
                        # Experimental validation success
                        if hasattr(hypothesis, 'validation_results'):
                            validation_results = hypothesis.validation_results
                            if validation_results.get('statistically_significant', False):
                                success_score += 0.3
                            if validation_results.get('reproducible', False):
                                success_score += 0.2
                            if validation_results.get('peer_reviewed', False):
                                success_score += 0.2
                        
                        # Scientific impact assessment
                        if hasattr(hypothesis, 'impact_metrics'):
                            impact = hypothesis.impact_metrics
                            # Citations, follow-up studies, paradigm shift potential
                            citation_score = min(impact.get('citations', 0) / 10, 0.1)
                            novelty_score = impact.get('novelty_index', 0) * 0.1
                            paradigm_score = impact.get('paradigm_shift_potential', 0) * 0.1
                            success_score += citation_score + novelty_score + paradigm_score
                        
                        # Knowledge graph integration success
                        if hasattr(hypothesis, 'kg_integration'):
                            kg_metrics = hypothesis.kg_integration
                            if kg_metrics.get('successfully_integrated', False):
                                success_score += 0.1
                            # Connectivity boost to existing knowledge
                            connectivity_boost = kg_metrics.get('connectivity_increase', 0)
                            success_score += min(connectivity_boost / 100, 0.1)
                        
                        # Normalize success score to [0, 1]
                        success_score = min(success_score, 1.0)
                        
                        # Weight by confidence
                        weighted_success_rate += success_score * confidence
                        confidence_weighted_total += confidence
                    
                    # Calculate final weighted success rate
                    if confidence_weighted_total > 0:
                        final_weighted_rate = weighted_success_rate / confidence_weighted_total
                    else:
                        final_weighted_rate = completion_rate
                    
                    # 3. Temporal decay factor for recent performance emphasis
                    current_time = datetime.utcnow()
                    session_age_hours = (current_time - session.start_time).total_seconds() / 3600
                    temporal_weight = math.exp(-session_age_hours / 168)  # 1-week half-life
                    
                    # 4. Hypothesis diversity bonus
                    # Reward generating hypotheses across different scientific domains
                    domain_diversity = 1.0
                    if hasattr(session, 'hypothesis_domains'):
                        unique_domains = len(set(session.hypothesis_domains))
                        domain_diversity = 1.0 + (unique_domains - 1) * 0.05  # 5% bonus per additional domain
                    
                    # 5. Innovation index
                    # Measure how much hypotheses advance beyond existing knowledge
                    innovation_bonus = 0.0
                    if hasattr(session, 'innovation_metrics'):
                        # Percentage of truly novel hypotheses vs. incremental ones
                        novel_ratio = session.innovation_metrics.get('novel_hypotheses', 0) / session.hypotheses_generated
                        innovation_bonus = novel_ratio * 0.1
                    
                    # 6. Collaborative success factor
                    # Hypotheses that trigger collaboration or interdisciplinary work
                    collaboration_factor = 1.0
                    if hasattr(session, 'collaboration_metrics'):
                        collab_count = session.collaboration_metrics.get('interdisciplinary_projects', 0)
                        collaboration_factor = 1.0 + collab_count * 0.02
                    
                    # 7. Computational efficiency component
                    # Factor in resource utilization vs. discovery quality
                    efficiency_factor = 1.0
                    if hasattr(session, 'resource_usage'):
                        compute_hours = session.resource_usage.get('total_compute_hours', 1)
                        discoveries = session.insights_discovered
                        if discoveries > 0:
                            efficiency = discoveries / compute_hours  # discoveries per compute hour
                            # Normalize and cap efficiency bonus
                            efficiency_factor = min(1.0 + efficiency * 0.1, 1.5)
                    
                    # 8. Long-term impact predictor
                    # Estimate future scientific impact based on hypothesis characteristics
                    future_impact_score = 0.0
                    if hasattr(session, 'impact_predictions'):
                        predictions = session.impact_predictions
                        # Weight by prediction confidence
                        predicted_citations = predictions.get('expected_citations_5yr', 0)
                        prediction_confidence = predictions.get('prediction_confidence', 0.5)
                        future_impact_score = (predicted_citations / 100) * prediction_confidence * 0.1
                    
                    # 9. Scientific rigor assessment
                    # Evaluate methodology quality and reproducibility
                    rigor_multiplier = 1.0
                    if hasattr(session, 'methodology_scores'):
                        rigor_scores = session.methodology_scores
                        avg_methodology_score = sum(rigor_scores) / len(rigor_scores) if rigor_scores else 0.7
                        rigor_multiplier = 0.5 + avg_methodology_score * 0.5  # Scale to [0.5, 1.0]
                    
                    # Comprehensive success rate calculation
                    comprehensive_success_rate = (
                        final_weighted_rate * 
                        temporal_weight * 
                        domain_diversity * 
                        (1.0 + innovation_bonus) *
                        collaboration_factor * 
                        efficiency_factor * 
                        (1.0 + future_impact_score) *
                        rigor_multiplier
                    )
                    
                    # Apply exponential moving average with adaptive learning rate
                    # Higher learning rate for significant changes in performance
                    current_rate = self.performance_metrics.get('hypothesis_success_rate', 0.5)
                    change_magnitude = abs(comprehensive_success_rate - current_rate)
                    adaptive_alpha = 0.1 + change_magnitude * 0.1  # Base 0.1, up to 0.2 for large changes
                    adaptive_alpha = min(adaptive_alpha, 0.3)  # Cap at 0.3
                    
                    self.performance_metrics['hypothesis_success_rate'] = (
                        current_rate * (1.0 - adaptive_alpha) + 
                        comprehensive_success_rate * adaptive_alpha
                    )
                    
                    # Store detailed metrics for analysis
                    self.performance_metrics['hypothesis_success_rate_components'] = {
                        'basic_completion_rate': completion_rate,
                        'weighted_success_rate': final_weighted_rate,
                        'temporal_weight': temporal_weight,
                        'domain_diversity': domain_diversity,
                        'innovation_bonus': innovation_bonus,
                        'collaboration_factor': collaboration_factor,
                        'efficiency_factor': efficiency_factor,
                        'future_impact_score': future_impact_score,
                        'rigor_multiplier': rigor_multiplier,
                        'comprehensive_rate': comprehensive_success_rate,
                        'adaptive_alpha': adaptive_alpha
                    }
        
        except Exception as e:
            self.logger.error(f"Error updating session metrics: {e}")
    
    async def get_autonomous_status(self) -> Dict[str, Any]:
        """Get comprehensive status of autonomous geoscience system"""
        status = {
            'is_initialized': self.is_initialized,
            'current_session': asdict(self.current_session) if self.current_session else None,
            'active_experiments': len(self.active_experiments),
            'knowledge_cache_size': len(self.knowledge_cache),
            'recent_events': self.integration_events[-10:],  # Last 10 events
            'performance_metrics': self.performance_metrics,
            'discovery_history_count': len(self.discovery_history),
            'config': asdict(self.config)
        }
        
        # Add component status
        status['components'] = {
            'discovery_system': self.discovery_system is not None,
            'hypothesis_generator': self.hypothesis_generator is not None,
            'digital_twin_integrator': self.digital_twin_integrator is not None,
            'api_system': self.api_system is not None,
            'visualization_dashboard': self.visualization_dashboard is not None
        }
        
        return status
    
    async def trigger_focused_discovery(self, focus_domains: List[str], 
                                      analysis_depth: str = "deep") -> str:
        """Trigger focused discovery on specific domains"""
        try:
            # Start focused discovery session
            session_id = await self.start_discovery_session(
                AutonomousGeoscienceMode.DISCOVERY,
                focus_domains
            )
            
            # Configure for deep analysis if requested
            if analysis_depth == "deep":
                self.hypothesis_generator.config.max_reasoning_depth = 10
                self.hypothesis_generator.config.enable_deep_reasoning = True
                self.hypothesis_generator.config.enable_cross_domain_insights = True
            
            # Run immediate discovery cycle
            await self._run_autonomous_discovery_cycle()
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Error triggering focused discovery: {e}")
            raise
    
    async def get_discovery_insights(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent discovery insights"""
        insights = []
        
        try:
            # Get recent insight events
            insight_events = [
                event for event in self.integration_events[-100:]  # Last 100 events
                if event['type'] == IntegrationEvent.INSIGHT_DISCOVERED
            ]
            
            # Sort by timestamp and limit
            insight_events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            for event in insight_events[:limit]:
                insights.append({
                    'timestamp': event['timestamp'].isoformat(),
                    'insight': event['data'],
                    'source': event['data'].get('source', 'unknown')
                })
            
        except Exception as e:
            self.logger.error(f"Error getting discovery insights: {e}")
        
        return insights
    
    async def shutdown(self):
        """Gracefully shutdown the autonomous geoscience integration system"""
        try:
            self.logger.info("Shutting down autonomous geoscience integration system")
            
            # End current session
            if self.current_session:
                await self.end_discovery_session()
            
            # Shutdown discovery system
            if self.discovery_system:
                await self.discovery_system.shutdown()
            
            # Clear caches
            self.knowledge_cache.clear()
            self.integration_events.clear()
            
            self.is_initialized = False
            self.logger.info("Autonomous geoscience integration system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Example usage and testing
async def example_autonomous_geoscience_integration():
    """Example of autonomous geoscience integration"""
    
    print("=== Autonomous Geoscience Integration Example ===")
    
    # Configuration
    config = AutonomousGeoscienceConfig(
        enable_real_time_discovery=True,
        discovery_cycle_interval=10,  # 10 seconds for demo
        max_concurrent_experiments=2,
        hypothesis_confidence_threshold=0.5,
        real_time_analysis_interval=5  # 5 seconds for demo
    )
    
    # Initialize integration system
    integrator = AutonomousGeoscienceIntegrator(config)
    
    try:
        # Initialize (without external systems for demo)
        await integrator.initialize()
        
        # Start focused discovery session
        session_id = await integrator.trigger_focused_discovery(
            focus_domains=['tectonic_processes', 'volcanic_processes'],
            analysis_depth='deep'
        )
        
        print(f"Started focused discovery session: {session_id}")
        
        # Let it run for a bit
        print("Running autonomous discovery for 15 seconds...")
        await asyncio.sleep(15)
        
        # Get status
        status = await integrator.get_autonomous_status()
        print(f"\nAutonomous Geoscience Status:")
        print(f"  Current session: {status['current_session']['session_id'] if status['current_session'] else 'None'}")
        print(f"  Hypotheses generated: {status['current_session']['hypotheses_generated'] if status['current_session'] else 0}")
        print(f"  Insights discovered: {status['current_session']['insights_discovered'] if status['current_session'] else 0}")
        print(f"  Discovery rate: {status['performance_metrics']['discovery_rate']:.2f} insights/hour")
        
        # Get recent insights
        insights = await integrator.get_discovery_insights(limit=5)
        print(f"\nRecent Insights ({len(insights)}):")
        for i, insight in enumerate(insights[:3]):
            print(f"  {i+1}. {insight['insight'].get('description', 'Insight discovered')}")
            print(f"     Source: {insight['source']}")
            print(f"     Time: {insight['timestamp'][:19]}")
        
        # End session
        completed_session = await integrator.end_discovery_session()
        if completed_session:
            print(f"\nCompleted session summary:")
            print(f"  Duration: {(completed_session.end_time - completed_session.start_time).total_seconds():.1f} seconds")
            print(f"  Hypotheses generated: {completed_session.hypotheses_generated}")
            print(f"  Experiments completed: {completed_session.experiments_completed}")
            print(f"  Insights discovered: {completed_session.insights_discovered}")
        
    finally:
        # Shutdown
        await integrator.shutdown()
    
    print("Autonomous geoscience integration example completed!")

if __name__ == "__main__":
    print("=== Autonomous Geoscience Integration System ===")
    print(f"Available features:")
    print(f"  Core Modules: {CORE_MODULES_AVAILABLE}")
    print(f"  Scientific Computing: {SCIENTIFIC_AVAILABLE}")
    
    # Run example
    asyncio.run(example_autonomous_geoscience_integration())