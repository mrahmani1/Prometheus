"""
Digital Twin Earth Visualization Dashboard
Advanced real-time visualization and monitoring interface for Digital Twin Earth system
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import base64
import io
from pathlib import Path

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_file
    from flask_socketio import SocketIO, emit, join_room, leave_room
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    warnings.warn("Web frameworks not available. Visualization will be limited.")

# Scientific plotting imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import seaborn as sns
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/Cartopy not available. Some visualizations will be limited.")

# 3D visualization imports
try:
    import vtk
    from vtk.util import numpy_support
    import pyvista as pv
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK/PyVista not available. 3D visualizations will be limited.")

# Image processing imports
try:
    from PIL import Image
    import cv2
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    warnings.warn("Image processing libraries not available.")

# Import Digital Twin Earth modules
try:
    from .digital_twin_earth import DigitalTwinEarth, EarthSystemState, ModelConfiguration
    from .digital_twin_integration import DigitalTwinEarthIntegrator, IntegrationConfig
    TWIN_MODULES_AVAILABLE = True
except ImportError:
    TWIN_MODULES_AVAILABLE = False
    warnings.warn("Digital Twin modules not available.")

warnings.filterwarnings('ignore')

class VisualizationType(Enum):
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    EARTH_SYSTEM_GLOBE = "earth_system_globe"
    TIME_SERIES_PLOTS = "time_series_plots"
    UNCERTAINTY_MAPS = "uncertainty_maps"
    PREDICTION_SCENARIOS = "prediction_scenarios"
    PERFORMANCE_METRICS = "performance_metrics"
    ANOMALY_DETECTION = "anomaly_detection"

class ViewMode(Enum):
    GLOBAL = "global"
    REGIONAL = "regional"
    LOCAL = "local"
    MULTI_SCALE = "multi_scale"

@dataclass
class VisualizationConfig:
    """Configuration for Digital Twin visualization system"""
    enable_real_time_updates: bool = True
    update_interval_ms: int = 1000
    enable_3d_visualization: bool = True
    enable_webgl_acceleration: bool = True
    max_time_series_points: int = 1000
    enable_uncertainty_visualization: bool = True
    default_view_mode: ViewMode = ViewMode.GLOBAL
    color_scheme: str = "viridis"
    enable_animations: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "html"])

class DigitalTwinVisualizationDashboard:
    """Main visualization dashboard for Digital Twin Earth"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Dashboard components
        self.flask_app = None
        self.socketio = None
        self.dash_app = None
        
        # Data management
        self.integrator = None
        self.current_earth_state = None
        self.time_series_data = {}
        self.visualization_cache = {}
        
        # Visualization state
        self.active_visualizations = set()
        self.connected_clients = set()
        self.update_tasks = {}
        
        # Performance tracking
        self.render_times = {}
        self.update_counts = {}
    
    async def initialize(self, integrator: DigitalTwinEarthIntegrator = None):
        """Initialize the visualization dashboard"""
        try:
            self.integrator = integrator
            
            if WEB_AVAILABLE:
                await self._setup_flask_app()
                await self._setup_dash_app()
                await self._setup_socketio()
            
            # Initialize visualization components
            await self._setup_visualization_components()
            
            # Start real-time updates if enabled
            if self.config.enable_real_time_updates:
                await self._start_real_time_updates()
            
            self.logger.info("Digital Twin visualization dashboard initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize visualization dashboard: {e}")
            raise
    
    async def _setup_flask_app(self):
        """Setup Flask application for web interface"""
        self.flask_app = Flask(__name__)
        self.flask_app.config['SECRET_KEY'] = 'digital_twin_earth_secret'
        
        # Main dashboard route
        @self.flask_app.route('/')
        def dashboard():
            return render_template('digital_twin_dashboard.html')
        
        # API routes
        @self.flask_app.route('/api/earth_state')
        def get_earth_state():
            if self.current_earth_state:
                return jsonify(self._serialize_earth_state(self.current_earth_state))
            return jsonify({'error': 'No earth state available'})
        
        @self.flask_app.route('/api/time_series/<component>')
        def get_time_series(component):
            if component in self.time_series_data:
                return jsonify(self.time_series_data[component])
            return jsonify({'error': f'No time series data for {component}'})
        
        @self.flask_app.route('/api/visualization/<viz_type>')
        def get_visualization(viz_type):
            try:
                viz_data = await self._generate_visualization(VisualizationType(viz_type))
                return jsonify(viz_data)
            except Exception as e:
                return jsonify({'error': str(e)})
    
    async def _setup_dash_app(self):
        """Setup Dash application for interactive plots"""
        if not WEB_AVAILABLE:
            return
        
        self.dash_app = dash.Dash(__name__, server=self.flask_app, url_base_pathname='/dash/')
        
        # Dashboard layout
        self.dash_app.layout = html.Div([
            html.H1("Digital Twin Earth Dashboard", className="header"),
            
            # Control panel
            html.Div([
                html.Label("View Mode:"),
                dcc.Dropdown(
                    id='view-mode-dropdown',
                    options=[{'label': mode.value, 'value': mode.value} for mode in ViewMode],
                    value=self.config.default_view_mode.value
                ),
                
                html.Label("Update Interval (ms):"),
                dcc.Slider(
                    id='update-interval-slider',
                    min=100,
                    max=10000,
                    step=100,
                    value=self.config.update_interval_ms,
                    marks={i: str(i) for i in range(1000, 11000, 2000)}
                ),
                
                html.Button('Export Visualization', id='export-button', n_clicks=0)
            ], className="control-panel"),
            
            # Main visualization area
            html.Div([
                # Earth system overview
                html.Div([
                    html.H3("Earth System State"),
                    dcc.Graph(id='earth-system-globe')
                ], className="viz-panel", style={'width': '50%', 'display': 'inline-block'}),
                
                # Time series plots
                html.Div([
                    html.H3("Component Time Series"),
                    dcc.Graph(id='component-time-series')
                ], className="viz-panel", style={'width': '50%', 'display': 'inline-block'}),
                
                # Uncertainty visualization
                html.Div([
                    html.H3("Uncertainty Maps"),
                    dcc.Graph(id='uncertainty-maps')
                ], className="viz-panel", style={'width': '50%', 'display': 'inline-block'}),
                
                # Prediction scenarios
                html.Div([
                    html.H3("Prediction Scenarios"),
                    dcc.Graph(id='prediction-scenarios')
                ], className="viz-panel", style={'width': '50%', 'display': 'inline-block'})
            ], className="main-viz-area"),
            
            # Performance metrics
            html.Div([
                html.H3("System Performance"),
                dcc.Graph(id='performance-metrics')
            ], className="performance-panel"),
            
            # Auto-update component
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval_ms,
                n_intervals=0
            )
        ])
        
        # Callbacks
        self._setup_dash_callbacks()
    
    def _setup_dash_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        if not self.dash_app:
            return
        
        @self.dash_app.callback(
            [Output('earth-system-globe', 'figure'),
             Output('component-time-series', 'figure'),
             Output('uncertainty-maps', 'figure'),
             Output('prediction-scenarios', 'figure'),
             Output('performance-metrics', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('view-mode-dropdown', 'value')]
        )
        def update_visualizations(n_intervals, view_mode):
            try:
                # Generate all visualizations
                globe_fig = self._create_earth_system_globe(ViewMode(view_mode))
                time_series_fig = self._create_component_time_series()
                uncertainty_fig = self._create_uncertainty_maps()
                scenarios_fig = self._create_prediction_scenarios()
                performance_fig = self._create_performance_metrics()
                
                return globe_fig, time_series_fig, uncertainty_fig, scenarios_fig, performance_fig
                
            except Exception as e:
                self.logger.error(f"Error updating visualizations: {e}")
                # Return empty figures
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        @self.dash_app.callback(
            Output('interval-component', 'interval'),
            [Input('update-interval-slider', 'value')]
        )
        def update_interval(interval_value):
            return interval_value
    
    async def _setup_socketio(self):
        """Setup SocketIO for real-time communication"""
        if not WEB_AVAILABLE:
            return
        
        self.socketio = SocketIO(self.flask_app, cors_allowed_origins="*")
        
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            self.connected_clients.add(client_id)
            self.logger.info(f"Client {client_id} connected")
            
            # Send current state to new client
            if self.current_earth_state:
                emit('earth_state_update', self._serialize_earth_state(self.current_earth_state))
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            self.connected_clients.discard(client_id)
            self.logger.info(f"Client {client_id} disconnected")
        
        @self.socketio.on('request_visualization')
        def handle_visualization_request(data):
            try:
                viz_type = VisualizationType(data.get('type'))
                viz_data = asyncio.run(self._generate_visualization(viz_type))
                emit('visualization_data', viz_data)
            except Exception as e:
                emit('error', {'message': str(e)})
    
    async def _setup_visualization_components(self):
        """Setup visualization-specific components"""
        # Initialize time series data storage
        self.time_series_data = {
            'atmosphere': {'timestamps': [], 'values': []},
            'hydrosphere': {'timestamps': [], 'values': []},
            'lithosphere': {'timestamps': [], 'values': []},
            'biosphere': {'timestamps': [], 'values': []},
            'cryosphere': {'timestamps': [], 'values': []},
            'anthroposphere': {'timestamps': [], 'values': []}
        }
        
        # Initialize visualization cache
        self.visualization_cache = {}
        
        # Setup matplotlib style if available
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            sns.set_palette(self.config.color_scheme)
    
    async def _start_real_time_updates(self):
        """Start real-time visualization updates"""
        async def update_loop():
            while True:
                try:
                    # Get current earth state from integrator
                    if self.integrator:
                        status = await self.integrator.get_integration_status()
                        
                        # Update time series data
                        await self._update_time_series_data(status)
                        
                        # Broadcast updates to connected clients
                        if self.socketio and self.connected_clients:
                            self.socketio.emit('system_update', status)
                    
                    await asyncio.sleep(self.config.update_interval_ms / 1000.0)
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time update loop: {e}")
                    await asyncio.sleep(5)
        
        self.update_tasks['main_update'] = asyncio.create_task(update_loop())
    
    async def _update_time_series_data(self, status: Dict[str, Any]):
        """Update time series data from system status"""
        try:
            current_time = datetime.utcnow()
            
            # Mock Earth system component data (in real implementation, would come from integrator)
            components = ['atmosphere', 'hydrosphere', 'lithosphere', 'biosphere', 'cryosphere', 'anthroposphere']
            
            for component in components:
                # Generate mock data (replace with actual data from integrator)
                value = np.random.normal(0, 1) + np.sin(time.time() / 3600) * 2
                
                # Add to time series
                self.time_series_data[component]['timestamps'].append(current_time.isoformat())
                self.time_series_data[component]['values'].append(value)
                
                # Limit time series length
                if len(self.time_series_data[component]['values']) > self.config.max_time_series_points:
                    self.time_series_data[component]['timestamps'].pop(0)
                    self.time_series_data[component]['values'].pop(0)
        
        except Exception as e:
            self.logger.error(f"Error updating time series data: {e}")
    
    def _create_earth_system_globe(self, view_mode: ViewMode = ViewMode.GLOBAL) -> go.Figure:
        """Create Earth system globe visualization"""
        try:
            fig = go.Figure()
            
            # Create base globe
            if view_mode == ViewMode.GLOBAL:
                # Global view with Earth surface
                fig.add_trace(go.Scattergeo(
                    lon=[],
                    lat=[],
                    mode='markers',
                    marker=dict(size=1, color='blue'),
                    name='Earth System State'
                ))
                
                # Add atmosphere layer (simplified)
                atmosphere_data = self._generate_mock_atmosphere_data()
                fig.add_trace(go.Scattergeo(
                    lon=atmosphere_data['lon'],
                    lat=atmosphere_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=atmosphere_data['temperature'],
                        color=atmosphere_data['temperature'],
                        colorscale=self.config.color_scheme,
                        showscale=True,
                        colorbar=dict(title="Temperature (°C)")
                    ),
                    name='Atmosphere'
                ))
            
            # Configure layout
            fig.update_layout(
                title="Digital Twin Earth - Global System State",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='lightgray',
                    showocean=True,
                    oceancolor='lightblue',
                    showlakes=True,
                    lakecolor='blue',
                    showcountries=True,
                    countrycolor='darkgray'
                ),
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating Earth system globe: {e}")
            return go.Figure()
    
    def _create_component_time_series(self) -> go.Figure:
        """Create component time series visualization"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Atmosphere', 'Hydrosphere', 'Lithosphere', 
                               'Biosphere', 'Cryosphere', 'Anthroposphere'),
                vertical_spacing=0.1
            )
            
            components = ['atmosphere', 'hydrosphere', 'lithosphere', 'biosphere', 'cryosphere', 'anthroposphere']
            positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
            
            for i, component in enumerate(components):
                if component in self.time_series_data and self.time_series_data[component]['values']:
                    row, col = positions[i]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=self.time_series_data[component]['timestamps'],
                            y=self.time_series_data[component]['values'],
                            mode='lines',
                            name=component.capitalize(),
                            line=dict(width=2)
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Earth System Component Time Series",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating component time series: {e}")
            return go.Figure()
    
    def _create_uncertainty_maps(self) -> go.Figure:
        """Create uncertainty visualization maps"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Model Uncertainty', 'Prediction Uncertainty'),
                specs=[[{"type": "geo"}, {"type": "geo"}]]
            )
            
            # Generate mock uncertainty data
            uncertainty_data = self._generate_mock_uncertainty_data()
            
            # Model uncertainty
            fig.add_trace(
                go.Scattergeo(
                    lon=uncertainty_data['lon'],
                    lat=uncertainty_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=uncertainty_data['model_uncertainty'] * 20,
                        color=uncertainty_data['model_uncertainty'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Model Uncertainty", x=0.45)
                    ),
                    name='Model Uncertainty'
                ),
                row=1, col=1
            )
            
            # Prediction uncertainty
            fig.add_trace(
                go.Scattergeo(
                    lon=uncertainty_data['lon'],
                    lat=uncertainty_data['lat'],
                    mode='markers',
                    marker=dict(
                        size=uncertainty_data['prediction_uncertainty'] * 20,
                        color=uncertainty_data['prediction_uncertainty'],
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="Prediction Uncertainty", x=1.02)
                    ),
                    name='Prediction Uncertainty'
                ),
                row=1, col=2
            )
            
            # Update geo layouts
            for i in range(1, 3):
                fig.update_geos(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='lightgray',
                    showocean=True,
                    oceancolor='lightblue',
                    row=1, col=i
                )
            
            fig.update_layout(
                title="Uncertainty Visualization",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating uncertainty maps: {e}")
            return go.Figure()
    
    def _create_prediction_scenarios(self) -> go.Figure:
        """Create prediction scenarios visualization"""
        try:
            fig = go.Figure()
            
            # Generate mock prediction scenarios
            scenarios = self._generate_mock_prediction_scenarios()
            
            for i, scenario in enumerate(scenarios):
                fig.add_trace(
                    go.Scatter(
                        x=scenario['time_horizon'],
                        y=scenario['predicted_value'],
                        mode='lines+markers',
                        name=f'Scenario {i+1}',
                        line=dict(width=2),
                        opacity=0.7
                    )
                )
                
                # Add uncertainty bands
                fig.add_trace(
                    go.Scatter(
                        x=scenario['time_horizon'] + scenario['time_horizon'][::-1],
                        y=scenario['upper_bound'] + scenario['lower_bound'][::-1],
                        fill='tonexty' if i > 0 else 'tozeroy',
                        fillcolor=f'rgba({i*50}, {100+i*30}, {200-i*20}, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'Scenario {i+1} Uncertainty',
                        showlegend=False
                    )
                )
            
            fig.update_layout(
                title="Prediction Scenarios with Uncertainty Bounds",
                xaxis_title="Time Horizon (hours)",
                yaxis_title="Predicted Value",
                height=400
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating prediction scenarios: {e}")
            return go.Figure()
    
    def _create_performance_metrics(self) -> go.Figure:
        """Create system performance metrics visualization"""
        try:
            # Generate mock performance data
            perf_data = self._generate_mock_performance_data()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Network I/O', 'Integration Health'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "indicator"}]]
            )
            
            # CPU Usage
            fig.add_trace(
                go.Scatter(
                    x=perf_data['timestamps'],
                    y=perf_data['cpu_usage'],
                    mode='lines',
                    name='CPU %',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Memory Usage
            fig.add_trace(
                go.Scatter(
                    x=perf_data['timestamps'],
                    y=perf_data['memory_usage'],
                    mode='lines',
                    name='Memory %',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            # Network I/O
            fig.add_trace(
                go.Scatter(
                    x=perf_data['timestamps'],
                    y=perf_data['network_io'],
                    mode='lines',
                    name='Network MB/s',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            # Integration Health Gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=perf_data['health_score'],
                    title={'text': "Health Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="System Performance Metrics",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance metrics: {e}")
            return go.Figure()
    
    def _generate_mock_atmosphere_data(self) -> Dict[str, List]:
        """Generate mock atmosphere data for visualization"""
        n_points = 50
        lon = np.random.uniform(-180, 180, n_points)
        lat = np.random.uniform(-90, 90, n_points)
        temperature = np.random.normal(15, 20, n_points)
        
        return {
            'lon': lon.tolist(),
            'lat': lat.tolist(),
            'temperature': temperature.tolist()
        }
    
    def _generate_mock_uncertainty_data(self) -> Dict[str, List]:
        """Generate mock uncertainty data"""
        n_points = 30
        lon = np.random.uniform(-180, 180, n_points)
        lat = np.random.uniform(-90, 90, n_points)
        model_uncertainty = np.random.uniform(0, 1, n_points)
        prediction_uncertainty = np.random.uniform(0, 1, n_points)
        
        return {
            'lon': lon.tolist(),
            'lat': lat.tolist(),
            'model_uncertainty': model_uncertainty.tolist(),
            'prediction_uncertainty': prediction_uncertainty.tolist()
        }
    
    def _generate_mock_prediction_scenarios(self) -> List[Dict[str, List]]:
        """Generate mock prediction scenarios"""
        scenarios = []
        time_horizon = np.arange(0, 24, 0.5)  # 24 hours with 30-min intervals
        
        for i in range(3):
            base_trend = np.sin(time_horizon / 12 * np.pi) * (i + 1)
            noise = np.random.normal(0, 0.5, len(time_horizon))
            predicted_value = base_trend + noise
            
            uncertainty = np.abs(np.random.normal(0, 1, len(time_horizon)))
            upper_bound = predicted_value + uncertainty
            lower_bound = predicted_value - uncertainty
            
            scenarios.append({
                'time_horizon': time_horizon.tolist(),
                'predicted_value': predicted_value.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist()
            })
        
        return scenarios
    
    def _generate_mock_performance_data(self) -> Dict[str, List]:
        """Generate mock performance data"""
        n_points = 50
        timestamps = [(datetime.utcnow() - timedelta(minutes=i)).isoformat() for i in range(n_points, 0, -1)]
        
        cpu_usage = 20 + 30 * np.sin(np.arange(n_points) / 10) + np.random.normal(0, 5, n_points)
        memory_usage = 40 + 20 * np.cos(np.arange(n_points) / 15) + np.random.normal(0, 3, n_points)
        network_io = 10 + 15 * np.random.random(n_points)
        health_score = 85 + 10 * np.sin(time.time() / 100)
        
        return {
            'timestamps': timestamps,
            'cpu_usage': np.clip(cpu_usage, 0, 100).tolist(),
            'memory_usage': np.clip(memory_usage, 0, 100).tolist(),
            'network_io': np.clip(network_io, 0, 100).tolist(),
            'health_score': np.clip(health_score, 0, 100)
        }
    
    async def _generate_visualization(self, viz_type: VisualizationType) -> Dict[str, Any]:
        """Generate specific visualization data"""
        try:
            if viz_type == VisualizationType.REAL_TIME_DASHBOARD:
                return await self._generate_dashboard_data()
            elif viz_type == VisualizationType.EARTH_SYSTEM_GLOBE:
                return self._create_earth_system_globe()
            elif viz_type == VisualizationType.TIME_SERIES_PLOTS:
                return self._create_component_time_series()
            elif viz_type == VisualizationType.UNCERTAINTY_MAPS:
                return self._create_uncertainty_maps()
            elif viz_type == VisualizationType.PREDICTION_SCENARIOS:
                return self._create_prediction_scenarios()
            elif viz_type == VisualizationType.PERFORMANCE_METRICS:
                return self._create_performance_metrics()
            else:
                return {'error': f'Visualization type {viz_type} not supported'}
        
        except Exception as e:
            self.logger.error(f"Error generating visualization {viz_type}: {e}")
            return {'error': str(e)}
    
    async def _generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        return {
            'earth_state': self._serialize_earth_state(self.current_earth_state) if self.current_earth_state else None,
            'time_series': self.time_series_data,
            'performance_metrics': self._generate_mock_performance_data(),
            'integration_status': await self.integrator.get_integration_status() if self.integrator else None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _serialize_earth_state(self, earth_state: Any) -> Dict[str, Any]:
        """Serialize Earth system state for JSON transmission"""
        if not earth_state:
            return {}
        
        return {
            'timestamp': earth_state.timestamp.isoformat() if hasattr(earth_state, 'timestamp') else datetime.utcnow().isoformat(),
            'atmosphere_state': getattr(earth_state, 'atmosphere_state', {}),
            'hydrosphere_state': getattr(earth_state, 'hydrosphere_state', {}),
            'lithosphere_state': getattr(earth_state, 'lithosphere_state', {}),
            'biosphere_state': getattr(earth_state, 'biosphere_state', {}),
            'cryosphere_state': getattr(earth_state, 'cryosphere_state', {}),
            'anthroposphere_state': getattr(earth_state, 'anthroposphere_state', {}),
            'uncertainty_metrics': getattr(earth_state, 'uncertainty_metrics', {})
        }
    
    async def export_visualization(self, viz_type: VisualizationType, format: str = "png") -> str:
        """Export visualization to file"""
        try:
            viz_data = await self._generate_visualization(viz_type)
            
            if format in self.config.export_formats:
                filename = f"digital_twin_{viz_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
                filepath = Path(f"exports/{filename}")
                filepath.parent.mkdir(exist_ok=True)
                
                # Export logic would depend on visualization type and format
                # For now, return the filename
                return str(filepath)
            else:
                raise ValueError(f"Export format {format} not supported")
        
        except Exception as e:
            self.logger.error(f"Error exporting visualization: {e}")
            raise
    
    async def run_dashboard(self, host: str = "localhost", port: int = 5000, debug: bool = False):
        """Run the visualization dashboard"""
        try:
            if not WEB_AVAILABLE:
                self.logger.error("Web frameworks not available. Cannot run dashboard.")
                return
            
            self.logger.info(f"Starting Digital Twin visualization dashboard on {host}:{port}")
            
            # Run SocketIO app
            self.socketio.run(
                self.flask_app,
                host=host,
                port=port,
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the visualization dashboard"""
        try:
            # Cancel update tasks
            for task_name, task in self.update_tasks.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Clear data
            self.time_series_data.clear()
            self.visualization_cache.clear()
            self.connected_clients.clear()
            
            self.logger.info("Digital Twin visualization dashboard shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during dashboard shutdown: {e}")

# Example usage and testing
async def example_digital_twin_visualization():
    """Example of Digital Twin visualization dashboard"""
    
    print("=== Digital Twin Earth Visualization Dashboard Example ===")
    
    # Configuration
    viz_config = VisualizationConfig(
        enable_real_time_updates=True,
        update_interval_ms=2000,
        enable_3d_visualization=True,
        max_time_series_points=100
    )
    
    # Initialize dashboard
    dashboard = DigitalTwinVisualizationDashboard(viz_config)
    
    try:
        # Initialize without integrator for demo
        await dashboard.initialize()
        
        print("Dashboard initialized successfully")
        print("Available visualizations:")
        for viz_type in VisualizationType:
            print(f"  - {viz_type.value}")
        
        # Generate sample visualizations
        for viz_type in [VisualizationType.EARTH_SYSTEM_GLOBE, VisualizationType.TIME_SERIES_PLOTS]:
            viz_data = await dashboard._generate_visualization(viz_type)
            print(f"Generated {viz_type.value} visualization")
        
        # Show dashboard status
        dashboard_data = await dashboard._generate_dashboard_data()
        print(f"Dashboard data keys: {list(dashboard_data.keys())}")
        
        print("To run the full web dashboard, use: dashboard.run_dashboard()")
        
    finally:
        await dashboard.shutdown()
    
    print("Digital Twin visualization example completed!")

if __name__ == "__main__":
    print("=== Digital Twin Earth Visualization System ===")
    print(f"Available features:")
    print(f"  Web Frameworks: {WEB_AVAILABLE}")
    print(f"  Matplotlib: {MATPLOTLIB_AVAILABLE}")
    print(f"  VTK/PyVista: {VTK_AVAILABLE}")
    print(f"  Image Processing: {IMAGE_PROCESSING_AVAILABLE}")
    print(f"  Twin Modules: {TWIN_MODULES_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_visualization())