"""
Prometheus Backend API
Main Flask API server for resurgent caldera monitoring system
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
import logging
from datetime import datetime
import json
from typing import Dict, Any, Optional

# Import our resurgent caldera monitoring modules
try:
    from modules.resurgent_caldera_monitoring import ResurgentCalderaMonitoringSystem
    from modules.integrated_caldera_monitoring import IntegratedCalderaMonitoring
    from modules.supereruption_analysis import *  # Import all classes from supereruption_analysis
    from modules.thermodynamic_engine import ThermodynamicEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import monitoring modules: {e}")
    MODULES_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
monitoring_system = None
integrated_monitoring = None
thermodynamic_engine = None

def init_systems():
    """Initialize monitoring systems"""
    global monitoring_system, integrated_monitoring, thermodynamic_engine
    
    if MODULES_AVAILABLE:
        monitoring_system = ResurgentCalderaMonitoringSystem()
        integrated_monitoring = IntegratedCalderaMonitoring()
        thermodynamic_engine = ThermodynamicEngine()
        logger.info("Monitoring and thermodynamic systems initialized")
    else:
        logger.warning("Modules not available - using mock responses")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "prometheus-caldera-monitoring",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "modules_available": MODULES_AVAILABLE
    })

@app.route('/api/calderas', methods=['GET'])
def get_calderas():
    """Get list of available calderas for monitoring"""
    calderas = [
        {
            "id": "yellowstone",
            "name": "Yellowstone Caldera",
            "location": {"latitude": 44.4280, "longitude": -110.5885},
            "status": "monitored",
            "last_eruption": "640,000 years ago",
            "country": "USA",
            "type": "resurgent_caldera"
        },
        {
            "id": "long_valley",
            "name": "Long Valley Caldera",
            "location": {"latitude": 37.7, "longitude": -118.87},
            "status": "monitored", 
            "last_eruption": "760,000 years ago",
            "country": "USA",
            "type": "resurgent_caldera"
        },
        {
            "id": "laguna_del_maule",
            "name": "Laguna del Maule",
            "location": {"latitude": -36.02, "longitude": -70.58},
            "status": "monitored",
            "last_eruption": "2,560 years ago",
            "country": "Chile",
            "type": "resurgent_caldera"
        },
        {
            "id": "chiles_cerro_negro",
            "name": "Chiles-Cerro Negro",
            "location": {"latitude": 0.82, "longitude": -77.93},
            "status": "monitored",
            "last_eruption": "1936",
            "country": "Ecuador/Colombia",
            "type": "volcanic_complex"
        },
        {
            "id": "valles",
            "name": "Valles Caldera",
            "location": {"latitude": 35.87, "longitude": -106.55},
            "status": "available",
            "last_eruption": "1.25 million years ago",
            "country": "USA",
            "type": "resurgent_caldera"
        },
        {
            "id": "toba",
            "name": "Toba Caldera",
            "location": {"latitude": 2.5, "longitude": 98.8},
            "status": "available",
            "last_eruption": "74,000 years ago",
            "country": "Indonesia",
            "type": "supervolcano"
        },
        {
            "id": "campi_flegrei",
            "name": "Campi Flegrei",
            "location": {"latitude": 40.827, "longitude": 14.139},
            "status": "available",
            "last_eruption": "1538",
            "country": "Italy",
            "type": "resurgent_caldera"
        },
        {
            "id": "santorini",
            "name": "Santorini Caldera",
            "location": {"latitude": 36.404, "longitude": 25.396},
            "status": "available",
            "last_eruption": "1950",
            "country": "Greece",
            "type": "caldera_complex"
        },
        {
            "id": "rabaul",
            "name": "Rabaul Caldera",
            "location": {"latitude": -4.271, "longitude": 152.203},
            "status": "available",
            "last_eruption": "2014",
            "country": "Papua New Guinea",
            "type": "caldera_complex"
        },
        {
            "id": "askja",
            "name": "Askja Caldera",
            "location": {"latitude": 65.03, "longitude": -16.75},
            "status": "available",
            "last_eruption": "1961",
            "country": "Iceland",
            "type": "caldera_complex"
        },
        {
            "id": "aso",
            "name": "Aso Caldera",
            "location": {"latitude": 32.88, "longitude": 131.10},
            "status": "available",
            "last_eruption": "2019",
            "country": "Japan",
            "type": "caldera_complex"
        }
    ]
    return jsonify({"calderas": calderas})

@app.route('/api/caldera-monitoring/initialize/<caldera_id>', methods=['POST'])
def initialize_caldera_monitoring(caldera_id: str):
    """Initialize resurgent caldera monitoring system"""
    try:
        data = request.get_json() or {}
        
        if MODULES_AVAILABLE and monitoring_system:
            location = (data.get('latitude', 0), data.get('longitude', 0))
            bounds = data.get('bounds', {
                'lat_min': location[0] - 0.5, 'lat_max': location[0] + 0.5,
                'lon_min': location[1] - 0.5, 'lon_max': location[1] + 0.5,
                'depth_min': 0, 'depth_max': 15
            })
            
            # Use asyncio to run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            session_id = loop.run_until_complete(
                monitoring_system.initialize_caldera_monitoring(caldera_id, location, bounds)
            )
            loop.close()
            
            return jsonify({
                "session_id": session_id,
                "status": "initialized",
                "caldera_id": caldera_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Mock response
            import uuid
            session_id = str(uuid.uuid4())
            return jsonify({
                "session_id": session_id,
                "status": "initialized",
                "caldera_id": caldera_id,
                "timestamp": datetime.now().isoformat(),
                "mock": True
            })
            
    except Exception as e:
        logger.error(f"Failed to initialize caldera monitoring: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/caldera-monitoring/status/<session_id>', methods=['GET'])
def get_monitoring_status(session_id: str):
    """Get current monitoring status"""
    try:
        # Generate realistic mock data
        import random
        import numpy as np
        
        alert_levels = ['normal', 'medium', 'high', 'critical']
        alert_weights = [0.7, 0.2, 0.08, 0.02]  # Most likely to be normal
        
        return jsonify({
            "session_id": session_id,
            "status": "active",
            "alert_level": np.random.choice(alert_levels, p=alert_weights),
            "last_update": datetime.now().isoformat(),
            "metrics": {
                "deformation_rate": round(random.uniform(0.1, 5.0), 2),
                "seismic_activity": round(random.uniform(0.0, 2.0), 2),
                "gas_emissions": round(random.uniform(0.5, 3.0), 2),
                "thermal_anomaly": round(random.uniform(0.0, 1.5), 2),
                "ground_tilt": round(random.uniform(-0.1, 0.3), 3),
                "so2_flux": round(random.uniform(10, 500), 1),
                "magma_depth": round(random.uniform(3, 12), 1)
            },
            "predictions": {
                "eruption_probability_30d": round(random.uniform(0.001, 0.1), 4),
                "unrest_probability_7d": round(random.uniform(0.01, 0.3), 3),
                "cap_rock_stability": round(random.uniform(0.6, 1.0), 3)
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/caldera-monitoring/3d-data/<session_id>', methods=['GET'])
def get_3d_visualization_data(session_id: str):
    """Get 3D visualization data for caldera"""
    try:
        import numpy as np
        
        # Generate synthetic 3D grid data
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        z = np.linspace(-15, 0, 15)
        
        # Create stress field data
        stress_values = []
        stress_coords = []
        for i, xi in enumerate(x[::2]):  # Subsample for performance
            for j, yj in enumerate(y[::2]):
                for k, zk in enumerate(z[::2]):
                    r = np.sqrt(xi**2 + yj**2 + (zk + 5)**2)
                    stress = max(0.1, 2.0 * np.exp(-r/5) + 0.3 * np.random.random())
                    stress_values.append(stress)
                    stress_coords.append([xi, yj, zk])
        
        # Generate magma chamber geometry
        chambers = [
            {
                "center": [0, 0, -8],
                "radius": 3.5,
                "melt_fraction": round(np.random.uniform(0.3, 0.8), 2),
                "temperature": int(np.random.uniform(800, 1200)),
                "pressure": round(np.random.uniform(100, 500), 1),
                "composition": "rhyolitic"
            },
            {
                "center": [2, -1, -12],
                "radius": 2.0,
                "melt_fraction": round(np.random.uniform(0.1, 0.4), 2),
                "temperature": int(np.random.uniform(700, 900)),
                "pressure": round(np.random.uniform(200, 600), 1),
                "composition": "andesitic"
            }
        ]
        
        # Generate deformation data
        deformation_grid = []
        for xi in x:
            for yj in y:
                r = np.sqrt(xi**2 + yj**2)
                uplift = 0.2 * np.exp(-r/8) + 0.05 * np.random.random() - 0.025
                deformation_grid.append({
                    "x": xi, "y": yj, "uplift": round(uplift, 4)
                })
        
        return jsonify({
            "stress_field": {
                "coordinates": stress_coords,
                "values": stress_values,
                "units": "MPa"
            },
            "magma_chambers": chambers,
            "deformation": {
                "data": deformation_grid,
                "units": "m",
                "type": "vertical_displacement"
            },
            "seismic_tomography": {
                "velocity_grid": {
                    "x_range": [-10, 10],
                    "y_range": [-10, 10], 
                    "z_range": [-15, 0],
                    "resolution": 1.0
                },
                "p_wave_anomaly": np.random.uniform(-15, 10, (20, 20, 15)).tolist(),
                "s_wave_anomaly": np.random.uniform(-20, 15, (20, 20, 15)).tolist()
            },
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        })
        
    except Exception as e:
        logger.error(f"Failed to get 3D data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/caldera-monitoring/alerts/<session_id>', methods=['GET'])
def get_alerts(session_id: str):
    """Get current alerts for monitoring session"""
    try:
        import random
        from datetime import timedelta
        
        # Generate sample alerts
        alert_types = [
            "Seismic swarm detected",
            "Ground deformation increased",
            "Gas emission spike",
            "Thermal anomaly detected",
            "Unusual seismic pattern",
            "Ground tilt change"
        ]
        
        alerts = []
        num_alerts = random.randint(0, 5)
        
        for i in range(num_alerts):
            alert_time = datetime.now() - timedelta(hours=random.randint(1, 48))
            severity = random.choice(['medium', 'high', 'critical'])
            
            alerts.append({
                "id": f"alert_{i+1}",
                "type": random.choice(alert_types),
                "severity": severity,
                "timestamp": alert_time.isoformat(),
                "message": f"Alert {i+1}: Monitoring detected {severity} level activity",
                "location": {
                    "lat": round(random.uniform(-1, 1), 4),
                    "lon": round(random.uniform(-1, 1), 4),
                    "depth": round(random.uniform(1, 10), 1)
                }
            })
        
        return jsonify({
            "alerts": alerts,
            "session_id": session_id,
            "total_alerts": len(alerts),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/caldera-monitoring/export/<session_id>', methods=['GET'])
def export_data(session_id: str):
    """Export monitoring data"""
    try:
        export_format = request.args.get('format', 'json')
        
        if export_format == 'json':
            # Get current status and 3D data
            status = get_monitoring_status(session_id).get_json()
            viz_data = get_3d_visualization_data(session_id).get_json()
            alerts = get_alerts(session_id).get_json()
            
            export_data = {
                "session_id": session_id,
                "export_timestamp": datetime.now().isoformat(),
                "format": "json",
                "data": {
                    "status": status,
                    "visualization": viz_data,
                    "alerts": alerts
                }
            }
            
            return jsonify(export_data)
        else:
            return jsonify({"error": "Unsupported export format"}), 400
            
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        return jsonify({"error": str(e)}), 500

# Thermodynamic Engine API Endpoints
@app.route('/api/thermodynamics/phase-equilibrium', methods=['POST'])
def calculate_phase_equilibrium():
    """Calculate thermodynamic phase equilibrium"""
    try:
        if not thermodynamic_engine:
            return jsonify({"error": "Thermodynamic engine not available"}), 503
            
        data = request.get_json()
        composition = data.get('composition', {})
        temperature = data.get('temperature', 298.15)
        pressure = data.get('pressure', 1.0)
        
        result = thermodynamic_engine.solve_multicomponent_equilibrium(
            composition, temperature, pressure
        )
        
        return jsonify({
            "status": "success",
            "result": result,
            "parameters": {
                "temperature": temperature,
                "pressure": pressure,
                "composition": composition
            }
        })
        
    except Exception as e:
        logger.error(f"Phase equilibrium calculation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/thermodynamics/reaction-path', methods=['POST']) 
def calculate_reaction_path():
    """Calculate thermodynamic reaction path"""
    try:
        if not thermodynamic_engine:
            return jsonify({"error": "Thermodynamic engine not available"}), 503
            
        data = request.get_json()
        initial_composition = data.get('initial_composition', {})
        final_composition = data.get('final_composition', {})
        temperature_range = data.get('temperature_range', [298.15, 1273.15])
        pressure = data.get('pressure', 1.0)
        n_steps = data.get('n_steps', 100)
        
        result = thermodynamic_engine.calculate_reaction_path(
            initial_composition, final_composition, 
            temperature_range, pressure, n_steps
        )
        
        return jsonify({
            "status": "success", 
            "result": result,
            "parameters": {
                "initial_composition": initial_composition,
                "final_composition": final_composition,
                "temperature_range": temperature_range,
                "pressure": pressure,
                "n_steps": n_steps
            }
        })
        
    except Exception as e:
        logger.error(f"Reaction path calculation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/thermodynamics/pt-diagram', methods=['POST'])
def generate_pt_diagram():
    """Generate pressure-temperature diagram"""
    try:
        if not thermodynamic_engine:
            return jsonify({"error": "Thermodynamic engine not available"}), 503
            
        data = request.get_json()
        composition = data.get('composition', {})
        temperature_range = data.get('temperature_range', [298.15, 1273.15])
        pressure_range = data.get('pressure_range', [0.001, 10.0])
        
        result = thermodynamic_engine.generate_pt_diagram(
            composition, temperature_range, pressure_range
        )
        
        return jsonify({
            "status": "success",
            "result": result,
            "parameters": {
                "composition": composition,
                "temperature_range": temperature_range,
                "pressure_range": pressure_range
            }
        })
        
    except Exception as e:
        logger.error(f"P-T diagram generation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize systems on startup
    init_systems()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )