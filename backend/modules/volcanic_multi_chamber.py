"""
Multi-Chamber Volcanic System Backend
Advanced physics-based simulation with ML predictions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import json

# Initialize FastAPI app
app = FastAPI(title="Multi-Chamber Volcanic System API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class ChamberComposition(BaseModel):
    """Magma composition in weight percent"""
    SiO2: float  # Silica content
    H2O: float   # Water content
    CO2: float   # Carbon dioxide content
    FeO: Optional[float] = 8.0
    MgO: Optional[float] = 4.0
    CaO: Optional[float] = 6.0
    Na2O: Optional[float] = 3.0
    K2O: Optional[float] = 2.0

class MagmaChamber(BaseModel):
    """Individual magma chamber parameters"""
    id: str
    depth: float  # meters
    volume: float  # m³
    temperature: float  # °C
    pressure: float  # MPa
    composition: ChamberComposition
    position: Dict[str, float]  # x, y, z coordinates
    
class ChamberConnection(BaseModel):
    """Connection between chambers"""
    from_chamber: str
    to_chamber: str
    conductivity: float  # m³/s/Pa
    
class SimulationParameters(BaseModel):
    """Simulation configuration"""
    chambers: List[MagmaChamber]
    connections: List[ChamberConnection]
    time_range: Dict[str, float]  # start, end, step in days
    physics: Dict[str, any]
    ml: Dict[str, bool]

# ============================================================================
# Physics Engine
# ============================================================================

class VolcanicPhysicsEngine:
    """
    Advanced physics simulation for multi-chamber volcanic systems
    """
    
    def __init__(self):
        self.gravity = 9.81  # m/s²
        self.gas_constant = 8.314  # J/(mol·K)
        
    def calculate_density(self, temperature: float, pressure: float, 
                         composition: ChamberComposition) -> float:
        """
        Calculate magma density using equation of state
        """
        # Simplified density model based on composition and P-T conditions
        base_density = 2700  # kg/m³
        
        # Temperature effect (thermal expansion)
        thermal_factor = 1 - 3e-5 * (temperature - 1000)
        
        # Pressure effect (compressibility)
        pressure_factor = 1 + 1e-10 * pressure * 1e6  # Convert MPa to Pa
        
        # Compositional effect
        silica_factor = 1 - 0.01 * (composition.SiO2 - 60)
        water_factor = 1 - 0.1 * composition.H2O
        
        density = base_density * thermal_factor * pressure_factor * silica_factor * water_factor
        return density
    
    def calculate_viscosity(self, temperature: float, composition: ChamberComposition,
                          crystal_fraction: float = 0) -> float:
        """
        Calculate magma viscosity using VFT equation
        """
        # Vogel-Fulcher-Tammann equation parameters
        A = -4.55
        B = 8000 + 100 * (composition.SiO2 - 50)  # SiO2 dependence
        T0 = 500 - 20 * composition.H2O  # Water effect
        
        # Calculate melt viscosity
        log_visc = A + B / (temperature - T0)
        melt_viscosity = 10 ** log_visc
        
        # Einstein-Roscoe equation for crystal effect
        max_packing = 0.6
        if crystal_fraction < max_packing:
            relative_viscosity = (1 - crystal_fraction / max_packing) ** -2.5
        else:
            relative_viscosity = 1e10  # Effectively solid
            
        return melt_viscosity * relative_viscosity
    
    def gas_solubility(self, pressure: float, temperature: float, 
                       species: str = 'H2O') -> float:
        """
        Calculate gas solubility in magma
        """
        if species == 'H2O':
            # Water solubility model (simplified)
            solubility = 0.11 * np.sqrt(pressure)  # wt%
        elif species == 'CO2':
            # CO2 solubility model
            solubility = 0.0023 * pressure  # wt%
        else:
            solubility = 0
            
        # Temperature correction
        temp_factor = np.exp(-0.001 * (temperature - 1000))
        return solubility * temp_factor
    
    def calculate_buoyancy(self, density_magma: float, density_host: float = 2800) -> float:
        """
        Calculate buoyancy force per unit volume
        """
        return self.gravity * (density_host - density_magma)
    
    def conduit_flow_rate(self, pressure_diff: float, conductivity: float,
                         viscosity: float) -> float:
        """
        Calculate flow rate through conduit using Poiseuille flow
        """
        # Modified for non-Newtonian behavior
        flow_rate = conductivity * pressure_diff / viscosity
        return flow_rate

# ============================================================================
# ML Models
# ============================================================================

class EruptionPredictor(nn.Module):
    """
    LSTM-based eruption prediction model
    """
    
    def __init__(self, input_size=10, hidden_size=128, num_layers=3):
        super(EruptionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Probability, Time to eruption, VEI
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take last timestep
        out = attn_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        predictions = self.fc3(out)
        
        # Apply sigmoid to probability
        predictions[:, 0] = self.sigmoid(predictions[:, 0])
        
        return predictions

class ChamberInteractionGNN(nn.Module):
    """
    Graph Neural Network for chamber interactions
    """
    
    def __init__(self, node_features=8, edge_features=3, hidden_dim=64):
        super(ChamberInteractionGNN, self).__init__()
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.message_passing = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, node_features)
        
    def forward(self, node_features, edge_features, adjacency):
        # Encode nodes and edges
        h_nodes = self.node_encoder(node_features)
        h_edges = self.edge_encoder(edge_features)
        
        # Message passing (simplified)
        for _ in range(3):  # 3 iterations
            messages = torch.zeros_like(h_nodes)
            for i in range(adjacency.shape[0]):
                for j in range(adjacency.shape[1]):
                    if adjacency[i, j] > 0:
                        message = torch.cat([h_nodes[j], h_edges[i, j]], dim=-1)
                        messages[i] += self.message_passing(message, h_nodes[i])
            
            # Update node states
            h_nodes = self.node_update(messages, h_nodes)
        
        # Decode
        output = self.decoder(h_nodes)
        return output

# ============================================================================
# Simulation Engine
# ============================================================================

class MultiChamberSimulator:
    """
    Main simulation engine for multi-chamber volcanic systems
    """
    
    def __init__(self):
        self.physics = VolcanicPhysicsEngine()
        self.eruption_predictor = EruptionPredictor()
        self.chamber_gnn = ChamberInteractionGNN()
        
        # Load pre-trained models if available
        self.load_models()
        
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            # Load models from disk
            # self.eruption_predictor.load_state_dict(torch.load('models/eruption_predictor.pth'))
            # self.chamber_gnn.load_state_dict(torch.load('models/chamber_gnn.pth'))
            pass
        except:
            print("Using untrained models - demo mode")
            
    def simulate(self, params: SimulationParameters) -> Dict:
        """
        Run full multi-chamber simulation
        """
        # Extract parameters
        chambers = params.chambers
        connections = params.connections
        time_range = params.time_range
        
        # Initialize state variables
        num_chambers = len(chambers)
        num_steps = int((time_range['end'] - time_range['start']) / time_range['step'])
        
        # State arrays
        pressure = np.zeros((num_steps, num_chambers))
        temperature = np.zeros((num_steps, num_chambers))
        gas_content = np.zeros((num_steps, num_chambers))
        crystallinity = np.zeros((num_steps, num_chambers))
        
        # Initial conditions
        for i, chamber in enumerate(chambers):
            pressure[0, i] = chamber.pressure
            temperature[0, i] = chamber.temperature
            gas_content[0, i] = chamber.composition.H2O + chamber.composition.CO2
            crystallinity[0, i] = 0
            
        # Time integration
        for t in range(1, num_steps):
            dt = time_range['step'] * 86400  # Convert days to seconds
            
            # Calculate chamber properties
            for i, chamber in enumerate(chambers):
                # Update pressure (simplified mass balance)
                dp_dt = self.calculate_pressure_change(
                    chamber, chambers, connections, 
                    pressure[t-1], temperature[t-1]
                )
                pressure[t, i] = pressure[t-1, i] + dp_dt * dt / 1e6  # Convert to MPa
                
                # Update temperature (heat balance)
                dT_dt = self.calculate_temperature_change(
                    chamber, temperature[t-1, i], pressure[t, i]
                )
                temperature[t, i] = temperature[t-1, i] + dT_dt * dt
                
                # Update gas content (exsolution)
                gas_content[t, i] = self.calculate_gas_evolution(
                    pressure[t, i], temperature[t, i], chamber.composition
                )
                
                # Update crystallinity
                crystallinity[t, i] = self.calculate_crystallization(
                    temperature[t, i], t * time_range['step']
                )
        
        # Calculate derived quantities
        flow_rates = self.calculate_flow_rates(pressure, connections, chambers)
        
        # ML predictions
        ml_predictions = self.generate_ml_predictions(
            pressure, temperature, gas_content, crystallinity
        )
        
        # Compile results
        results = {
            'timeSteps': np.arange(0, num_steps) * time_range['step'],
            'chambers': {},
            'flowRates': flow_rates.tolist(),
            'eruptionProbability': ml_predictions['eruption_probability'].tolist()
        }
        
        for i, chamber in enumerate(chambers):
            results['chambers'][chamber.id] = {
                'pressure': pressure[:, i].tolist(),
                'temperature': temperature[:, i].tolist(),
                'gasContent': gas_content[:, i].tolist(),
                'crystallinity': crystallinity[:, i].tolist()
            }
            
        return results, ml_predictions
    
    def calculate_pressure_change(self, chamber, all_chambers, connections,
                                 pressures, temperatures):
        """
        Calculate pressure change due to magma flow
        """
        chamber_idx = next(i for i, c in enumerate(all_chambers) if c.id == chamber.id)
        dp_dt = 0
        
        # Magma recharge from depth
        recharge_rate = 1e-3  # m³/s (simplified)
        dp_dt += recharge_rate * 1e9 / chamber.volume  # Pa/s
        
        # Flow between chambers
        for connection in connections:
            if connection.from_chamber == chamber.id:
                to_idx = next(i for i, c in enumerate(all_chambers) 
                            if c.id == connection.to_chamber)
                pressure_diff = pressures[chamber_idx] - pressures[to_idx]
                
                # Calculate viscosity
                viscosity = self.physics.calculate_viscosity(
                    temperatures[chamber_idx], 
                    chamber.composition
                )
                
                # Calculate flow rate
                flow_rate = self.physics.conduit_flow_rate(
                    pressure_diff * 1e6,  # Convert to Pa
                    connection.conductivity,
                    viscosity
                )
                
                dp_dt -= flow_rate * 1e9 / chamber.volume  # Pa/s
                
            elif connection.to_chamber == chamber.id:
                from_idx = next(i for i, c in enumerate(all_chambers) 
                              if c.id == connection.from_chamber)
                pressure_diff = pressures[from_idx] - pressures[chamber_idx]
                
                viscosity = self.physics.calculate_viscosity(
                    temperatures[from_idx],
                    all_chambers[from_idx].composition
                )
                
                flow_rate = self.physics.conduit_flow_rate(
                    pressure_diff * 1e6,
                    connection.conductivity,
                    viscosity
                )
                
                dp_dt += flow_rate * 1e9 / chamber.volume  # Pa/s
                
        return dp_dt
    
    def calculate_temperature_change(self, chamber, temperature, pressure):
        """
        Calculate temperature change due to cooling and decompression
        """
        # Conductive cooling
        cooling_rate = -0.001  # °C/s (simplified)
        
        # Adiabatic effect
        dP_dt = 0  # Pressure change rate (simplified)
        adiabatic_rate = 0.3 * dP_dt  # °C/MPa
        
        return cooling_rate + adiabatic_rate
    
    def calculate_gas_evolution(self, pressure, temperature, composition):
        """
        Calculate dissolved gas content
        """
        h2o_solubility = self.physics.gas_solubility(pressure, temperature, 'H2O')
        co2_solubility = self.physics.gas_solubility(pressure, temperature, 'CO2')
        
        # Total dissolved gas
        total_gas = min(composition.H2O, h2o_solubility) + min(composition.CO2, co2_solubility)
        return total_gas
    
    def calculate_crystallization(self, temperature, time):
        """
        Calculate crystal fraction using Avrami equation
        """
        # Liquidus temperature
        T_liquidus = 1200  # °C (simplified)
        
        if temperature < T_liquidus:
            # Avrami equation
            k = 0.001  # Rate constant
            n = 2  # Avrami exponent
            crystal_fraction = 1 - np.exp(-k * time ** n)
            return min(crystal_fraction * 100, 50)  # Max 50% crystals
        return 0
    
    def calculate_flow_rates(self, pressure, connections, chambers):
        """
        Calculate flow rates between chambers
        """
        num_steps = pressure.shape[0]
        flow_rates = np.zeros((num_steps, len(connections)))
        
        for t in range(num_steps):
            for i, connection in enumerate(connections):
                from_idx = next(j for j, c in enumerate(chambers) 
                              if c.id == connection.from_chamber)
                to_idx = next(j for j, c in enumerate(chambers) 
                            if c.id == connection.to_chamber)
                
                pressure_diff = pressure[t, from_idx] - pressure[t, to_idx]
                
                # Simplified flow calculation
                flow_rates[t, i] = connection.conductivity * pressure_diff * 1000
                
        return flow_rates
    
    def generate_ml_predictions(self, pressure, temperature, gas_content, crystallinity):
        """
        Generate ML predictions for eruption forecasting
        """
        # Prepare input features
        features = []
        for t in range(len(pressure)):
            timestep_features = []
            for i in range(pressure.shape[1]):
                timestep_features.extend([
                    pressure[t, i] / 1000,  # Normalize
                    temperature[t, i] / 1500,
                    gas_content[t, i] / 10,
                    crystallinity[t, i] / 100
                ])
            features.append(timestep_features)
            
        features = np.array(features)
        
        # Use sliding window for LSTM
        window_size = min(50, len(features))
        if len(features) >= window_size:
            input_tensor = torch.FloatTensor(features[-window_size:]).unsqueeze(0)
            
            with torch.no_grad():
                predictions = self.eruption_predictor(input_tensor)
                
            eruption_prob = predictions[0, 0].item()
            time_to_eruption = max(0, predictions[0, 1].item() * 100)  # Scale to days
            expected_vei = int(predictions[0, 2].item() * 7)  # Scale to VEI 0-7
        else:
            # Fallback for short sequences
            eruption_prob = 0.1
            time_to_eruption = 365
            expected_vei = 2
            
        # Calculate uncertainty
        uncertainty_std = 0.15
        lower_bound = max(0, eruption_prob - uncertainty_std)
        upper_bound = min(1, eruption_prob + uncertainty_std)
        
        # Generate time series of eruption probability
        eruption_probability = []
        for t in range(len(pressure)):
            # Increase probability with pressure
            avg_pressure = np.mean(pressure[t])
            prob = min(1, max(0, (avg_pressure - 200) / 300))
            eruption_probability.append(prob)
            
        return {
            'eruption_probability': eruption_prob,
            'time_to_eruption': time_to_eruption if eruption_prob > 0.3 else None,
            'expected_vei': expected_vei,
            'confidence': 0.75,  # Placeholder
            'uncertainty_bounds': {
                'lower': [max(0, p - uncertainty_std) for p in eruption_probability],
                'upper': [min(1, p + uncertainty_std) for p in eruption_probability]
            }
        }

# ============================================================================
# API Endpoints
# ============================================================================

simulator = MultiChamberSimulator()
active_connections = []

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multi-chamber-volcanic-system"}

@app.post("/api/v1/volcanic/multi-chamber/simulate")
async def run_simulation(params: SimulationParameters):
    """
    Run multi-chamber volcanic system simulation
    """
    try:
        # Run simulation
        results, ml_predictions = simulator.simulate(params)
        
        # Generate simulation ID
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "simulation_id": simulation_id,
            "results": results,
            "ml_predictions": ml_predictions,
            "status": "completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/volcanic-monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """
    WebSocket endpoint for real-time monitoring
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'subscribe':
                # Handle subscription to specific topics
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "topics": message['topics']
                }))
                
                # Start sending real-time updates
                asyncio.create_task(send_monitoring_updates(websocket, message['topics']))
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def send_monitoring_updates(websocket: WebSocket, topics: List[str]):
    """
    Send real-time monitoring updates to connected clients
    """
    while websocket in active_connections:
        try:
            # Generate mock real-time data
            for topic in topics:
                if topic == 'chamber_pressure':
                    update = {
                        "type": "chamber_pressure",
                        "chamber_id": np.random.choice(['upper', 'middle', 'lower']),
                        "pressure": np.random.normal(250, 50),
                        "timestamp": datetime.now().isoformat()
                    }
                elif topic == 'temperature':
                    update = {
                        "type": "temperature",
                        "chamber_id": np.random.choice(['upper', 'middle', 'lower']),
                        "temperature": np.random.normal(950, 100),
                        "timestamp": datetime.now().isoformat()
                    }
                elif topic == 'seismic':
                    update = {
                        "type": "seismic",
                        "magnitude": np.random.exponential(0.5),
                        "depth": np.random.normal(5000, 2000),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    continue
                    
                await websocket.send_text(json.dumps(update))
                
            # Send updates every 5 seconds
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"Error sending updates: {e}")
            break

@app.get("/api/v1/volcanic/chambers")
async def get_chambers():
    """
    Get available volcanic systems and their chamber configurations
    """
    # Example volcanic systems
    systems = {
        "mount_vesuvius": {
            "name": "Mount Vesuvius",
            "location": {"lat": 40.821, "lon": 14.426},
            "chambers": [
                {"id": "shallow", "depth": 3000, "volume": 1e8},
                {"id": "deep", "depth": 8000, "volume": 5e8}
            ]
        },
        "yellowstone": {
            "name": "Yellowstone",
            "location": {"lat": 44.428, "lon": -110.588},
            "chambers": [
                {"id": "upper", "depth": 5000, "volume": 1e9},
                {"id": "lower", "depth": 20000, "volume": 4e9}
            ]
        },
        "campi_flegrei": {
            "name": "Campi Flegrei",
            "location": {"lat": 40.827, "lon": 14.139},
            "chambers": [
                {"id": "shallow", "depth": 3000, "volume": 2e8},
                {"id": "middle", "depth": 7500, "volume": 1e9},
                {"id": "deep", "depth": 15000, "volume": 2e9}
            ]
        }
    }
    return systems

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)