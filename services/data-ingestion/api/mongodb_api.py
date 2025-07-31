"""MongoDB API endpoints for unstructured geological data"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field
import structlog

from storage.mongodb_storage import MongoDBStorage
from models.geological_data import GeochemicalSample, AtmosphericMeasurement
from models.unstructured_geological_data import (
    RockSample, MineralAnalysis, FluidSample, CoreSample,
    GeologicalSurvey, FieldObservation, LaboratoryResult
)

logger = structlog.get_logger()

router = APIRouter(prefix="/mongodb", tags=["MongoDB Storage"])

# Pydantic models for API requests/responses
class LocationQuery(BaseModel):
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)
    radius_km: float = Field(..., gt=0, le=1000)

class TimeRangeQuery(BaseModel):
    start_time: datetime
    end_time: datetime

class BulkInsertRequest(BaseModel):
    documents: List[Dict[str, Any]]
    collection_name: str

class AggregationRequest(BaseModel):
    collection_name: str
    pipeline: List[Dict[str, Any]]

class DataQualityResponse(BaseModel):
    total_count: List[Dict[str, int]]
    quality_distribution: List[Dict[str, Any]]
    validation_issues: List[Dict[str, Any]]
    temporal_coverage: List[Dict[str, Any]]
    source_distribution: List[Dict[str, Any]]

# Dependency to get MongoDB storage instance
async def get_mongodb_storage() -> MongoDBStorage:
    # This would typically be injected from the main application
    # For now, we'll create a placeholder
    storage = MongoDBStorage("mongodb://localhost:27017")
    if not storage.is_initialized:
        await storage.initialize()
    return storage

@router.post("/geochemical-samples", response_model=Dict[str, str])
async def store_geochemical_sample(
    sample: GeochemicalSample,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store a geochemical sample in MongoDB"""
    try:
        document_id = await storage.store_geochemical_sample(sample)
        
class AdvancedGeochemicalAnalyzer:
    """Advanced geochemical analysis with machine learning and thermodynamic modeling"""
    
    def __init__(self):
        self.element_database = self._load_element_database()
        self.mineral_database = self._load_mineral_database()
        self.ml_models = self._initialize_ml_models()
        
    def _load_element_database(self) -> Dict[str, Dict[str, float]]:
        """Load comprehensive element property database"""
        return {
            'Si': {'atomic_mass': 28.0855, 'ionic_radius': 0.40, 'electronegativity': 1.90, 'oxidation_states': [4, -4]},
            'Al': {'atomic_mass': 26.9815, 'ionic_radius': 0.535, 'electronegativity': 1.61, 'oxidation_states': [3]},
            'Fe': {'atomic_mass': 55.845, 'ionic_radius': 0.645, 'electronegativity': 1.83, 'oxidation_states': [2, 3]},
            'Mg': {'atomic_mass': 24.305, 'ionic_radius': 0.72, 'electronegativity': 1.31, 'oxidation_states': [2]},
            'Ca': {'atomic_mass': 40.078, 'ionic_radius': 1.00, 'electronegativity': 1.00, 'oxidation_states': [2]},
            'Na': {'atomic_mass': 22.990, 'ionic_radius': 1.02, 'electronegativity': 0.93, 'oxidation_states': [1]},
            'K': {'atomic_mass': 39.098, 'ionic_radius': 1.38, 'electronegativity': 0.82, 'oxidation_states': [1]},
            'Ti': {'atomic_mass': 47.867, 'ionic_radius': 0.605, 'electronegativity': 1.54, 'oxidation_states': [4, 3]},
            'Mn': {'atomic_mass': 54.938, 'ionic_radius': 0.67, 'electronegativity': 1.55, 'oxidation_states': [2, 3, 4, 7]},
            'P': {'atomic_mass': 30.974, 'ionic_radius': 0.44, 'electronegativity': 2.19, 'oxidation_states': [5, 3, -3]}
        }
    
    def _load_mineral_database(self) -> Dict[str, Dict[str, Any]]:
        """Load mineral composition and stability database"""
        return {
            'quartz': {
                'formula': 'SiO2',
                'composition': {'Si': 1, 'O': 2},
                'density': 2.65,
                'hardness': 7,
                'stability_temp': 573.0,  # Alpha-beta transition
                'formation_pressure': 1.0
            },
            'feldspar': {
                'formula': 'KAlSi3O8',
                'composition': {'K': 1, 'Al': 1, 'Si': 3, 'O': 8},
                'density': 2.56,
                'hardness': 6,
                'stability_temp': 1150.0,
                'formation_pressure': 1.0
            },
            'olivine': {
                'formula': '(Mg,Fe)2SiO4',
                'composition': {'Mg': 1.5, 'Fe': 0.5, 'Si': 1, 'O': 4},
                'density': 3.3,
                'hardness': 6.5,
                'stability_temp': 1890.0,
                'formation_pressure': 1.0
            },
            'pyroxene': {
                'formula': '(Ca,Mg,Fe)SiO3',
                'composition': {'Ca': 0.5, 'Mg': 0.8, 'Fe': 0.7, 'Si': 1, 'O': 3},
                'density': 3.4,
                'hardness': 6,
                'stability_temp': 1200.0,
                'formation_pressure': 1.0
            }
        }
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for geochemical analysis"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
        from sklearn.neural_network import MLPRegressor
        
        return {
            'composition_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'rock_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'temperature_estimator': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
            'pressure_estimator': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def analyze_rock_composition(self, composition: Dict[str, float], 
                               temperature: float = 1200.0,
                               pressure: float = 1.0) -> Dict[str, Any]:
        """Comprehensive rock composition analysis"""
        
        # Normalize composition to 100%
        total = sum(composition.values())
        normalized_comp = {element: (value/total)*100 for element, value in composition.items()}
        
        # Calculate derived parameters
        analysis_results = {
            'normalized_composition': normalized_comp,
            'rock_classification': self._classify_rock_type(normalized_comp),
            'mineral_assemblage': self._predict_mineral_assemblage(normalized_comp, temperature, pressure),
            'geochemical_indices': self._calculate_geochemical_indices(normalized_comp),
            'thermodynamic_properties': self._calculate_thermodynamic_properties(normalized_comp, temperature, pressure),
            'trace_element_patterns': self._analyze_trace_elements(composition),
            'alteration_assessment': self._assess_alteration(normalized_comp),
            'petrogenetic_indicators': self._calculate_petrogenetic_indicators(normalized_comp)
        }
        
        return analysis_results
    
    def _classify_rock_type(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Classify rock type based on composition"""
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        
        # Total alkalis
        alkalis = na2o + k2o
        
        # Classification logic
        if sio2 > 63:
            if alkalis > 7:
                rock_type = 'alkaline_felsic'
            else:
                rock_type = 'calc_alkaline_felsic'
        elif sio2 > 52:
            if mgo > 6:
                rock_type = 'intermediate_mafic'
            else:
                rock_type = 'intermediate'
        else:
            if mgo > 12:
                rock_type = 'ultramafic'
            else:
                rock_type = 'mafic'
        
        # Tectonic setting inference
        if (k2o/na2o) > 0.5 and sio2 > 60:
            tectonic_setting = 'continental_arc'
        elif alkalis < 3 and mgo > 8:
            tectonic_setting = 'mid_ocean_ridge'
        elif alkalis > 5:
            tectonic_setting = 'within_plate'
        else:
            tectonic_setting = 'island_arc'
        
        return {
            'primary_type': rock_type,
            'silica_content': sio2,
            'alkali_content': alkalis,
            'mg_number': (mgo / (mgo + fe2o3 + 0.1)) * 100,
            'tectonic_setting': tectonic_setting,
            'confidence': 0.85  # Would be calculated from model uncertainty
        }
    
    def _predict_mineral_assemblage(self, composition: Dict[str, float], 
                                  temperature: float, pressure: float) -> Dict[str, Any]:
        """Predict stable mineral assemblage using thermodynamic modeling"""
        
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        stable_minerals = []
        mineral_proportions = {}
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        
        # Quartz stability
        if sio2 > 60:
            stable_minerals.append('quartz')
            mineral_proportions['quartz'] = min(40, (sio2 - 60) * 2)
        
        # Feldspar stability
        if al2o3 > 10:
            stable_minerals.append('feldspar')
            mineral_proportions['feldspar'] = min(50, al2o3 * 3)
        
        # Olivine stability (high temperature, low silica)
        if temperature > 1000 and sio2 < 50 and mgo > 5:
            stable_minerals.append('olivine')
            mineral_proportions['olivine'] = min(30, mgo * 4)
        
        # Pyroxene stability
        if temperature > 800 and (cao > 5 or mgo > 3):
            stable_minerals.append('pyroxene')
            mineral_proportions['pyroxene'] = min(35, (cao + mgo) * 2)
        
        # Normalize proportions
        total_prop = sum(mineral_proportions.values())
        if total_prop > 0:
            mineral_proportions = {k: (v/total_prop)*100 for k, v in mineral_proportions.items()}
        
        return {
            'stable_minerals': stable_minerals,
            'mineral_proportions': mineral_proportions,
            'crystallization_sequence': self._determine_crystallization_sequence(temperature, composition),
            'phase_diagram_position': {
                'temperature': temperature,
                'pressure': pressure,
                'stability_field': 'intermediate' if 800 < temperature < 1200 else 'extreme'
            }
        }
    
    def _calculate_geochemical_indices(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate various geochemical indices"""
        
        sio2 = composition.get('SiO2', 0)
        al2o3 = composition.get('Al2O3', 0)
        fe2o3 = composition.get('Fe2O3', 0) + composition.get('FeO', 0)
        mgo = composition.get('MgO', 0)
        cao = composition.get('CaO', 0)
        na2o = composition.get('Na2O', 0)
        k2o = composition.get('K2O', 0)
        tio2 = composition.get('TiO2', 0)
        
        indices = {}
        
        # Differentiation index
        indices['differentiation_index'] = sio2 + na2o + k2o
        
        # Solidification index
        indices['solidification_index'] = 100 * mgo / (mgo + fe2o3 + na2o + k2o)
        
        # Alumina saturation index
        molecular_al2o3 = al2o3 / 101.96
        molecular_cao = cao / 56.08
        molecular_na2o = na2o / 61.98
        molecular_k2o = k2o / 94.20
        
        indices['alumina_saturation'] = molecular_al2o3 / (molecular_cao + molecular_na2o + molecular_k2o)
        
        # Alkali-lime index
        indices['alkali_lime_index'] = (na2o + k2o) - cao
        
        # Iron enrichment index
        indices['iron_enrichment'] = fe2o3 / (fe2o3 + mgo)
        
        # Titanium enrichment
        indices['titanium_enrichment'] = tio2
        
        return indices
    
    def _calculate_thermodynamic_properties(self, composition: Dict[str, float],
                                          temperature: float, pressure: float) -> Dict[str, float]:
        """Calculate thermodynamic properties"""
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []

        # In practice, would use MELTS or similar thermodynamic software
        
        properties = {}
        
        # Estimate viscosity (simplified Arrhenius model)
        sio2 = composition.get('SiO2', 50)
        al2o3 = composition.get('Al2O3', 15)
        
        # Viscosity increases with SiO2 and Al2O3, decreases with temperature
        log_viscosity = (sio2 * 0.1 + al2o3 * 0.05) - (temperature - 1000) * 0.001
        properties['viscosity_log_pa_s'] = log_viscosity
        
        # Estimate density (simplified mixing model)
        density_components = {
            'SiO2': 2.65, 'Al2O3': 3.95, 'Fe2O3': 5.24, 'FeO': 5.87,
            'MgO': 3.58, 'CaO': 3.34, 'Na2O': 2.27, 'K2O': 2.32, 'TiO2': 4.23
        }
        
        total_density = 0
        total_weight = 0
        for oxide, weight in composition.items():
            if oxide in density_components:
                total_density += weight * density_components[oxide]
                total_weight += weight
        
        properties['density_g_cm3'] = total_density / (total_weight + 1e-8)
        
        # Estimate liquidus temperature
        
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate)
        }

        base_liquidus = 1200  # Base temperature in Celsius
        sio2_effect = (sio2 - 50) * 5  # SiO2 increases liquidus
        mgo_effect = composition.get('MgO', 0) * 10  # MgO increases liquidus
        alkali_effect = -(composition.get('Na2O', 0) + composition.get('K2O', 0)) * 8  # Alkalis decrease liquidus
        
        properties['liquidus_temperature_c'] = base_liquidus + sio2_effect + mgo_effect + alkali_effect
        
        # Heat capacity (simplified)
        properties['heat_capacity_j_mol_k'] = 80 + 0.02 * temperature  
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        
        return properties
    
    def _analyze_trace_elements(self, composition: Dict[str, float]) -> Dict[str, Any]:
        """Analyze trace element patterns"""
        
        # Extract trace elements (typically < 1000 ppm)
        trace_elements = {}
        for element, concentration in composition.items():
            if concentration < 0.1:  # Less than 0.1 wt% = trace element
                trace_elements[element] = concentration * 10000  # Convert to ppm
        
        # Calculate trace element ratios
        ratios = {}
        if 'Rb' in trace_elements and 'Sr' in trace_elements:
            ratios['Rb_Sr'] = trace_elements['Rb'] / (trace_elements['Sr'] + 1e-8)
        
        if 'Ba' in trace_elements and 'Rb' in trace_elements:
            ratios['Ba_Rb'] = trace_elements['Ba'] / (trace_elements['Rb'] + 1e-8)
        
        if 'Zr' in trace_elements and 'Hf' in trace_elements:
            ratios['Zr_Hf'] = trace_elements['Zr'] / (trace_elements['Hf'] + 1e-8)
        
        # REE pattern analysis (if REE data available)
        ree_elements = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
        ree_pattern = {element: trace_elements.get(element, 0) for element in ree_elements}
        
        return {
            'trace_elements_ppm': trace_elements,
            'trace_element_ratios': ratios,
            'ree_pattern': ree_pattern,
            'enrichment_factors': self._calculate_enrichment_factors(trace_elements)
        }
    
    def _calculate_enrichment_factors(self, trace_elements: Dict[str, float]) -> Dict[str, float]:
        """Calculate enrichment factors relative to primitive mantle"""
        
        # Primitive mantle values (McDonough & Sun, 1995)
        primitive_mantle = {
            'Rb': 0.635, 'Ba': 6.989, 'Th': 0.085, 'U': 0.021, 'Nb': 0.713,
            'Ta': 0.041, 'La': 0.687, 'Ce': 1.775, 'Pb': 0.185, 'Pr': 0.276,
            'Sr': 21.1, 'Nd': 1.354, 'Sm': 0.444, 'Zr': 11.2, 'Hf': 0.309,
            'Eu': 0.168, 'Gd': 0.596, 'Tb': 0.108, 'Dy': 0.737, 'Y': 4.55,
            'Ho': 0.164, 'Er': 0.480, 'Tm': 0.074, 'Yb': 0.493, 'Lu': 0.074
        }
        
        enrichment_factors = {}
        for element, concentration in trace_elements.items():
            if element in primitive_mantle:
                enrichment_factors[element] = concentration / primitive_mantle[element]
        
        return enrichment_factors

    except Exception as e:
        logger.error("Failed to store geochemical sample", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/atmospheric-measurements", response_model=Dict[str, str])
async def store_atmospheric_measurement(
    measurement: AtmosphericMeasurement,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store an atmospheric measurement in MongoDB"""
    try:
        document_id = await storage.store_atmospheric_measurement(measurement)
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


    except Exception as e:
        logger.error("Failed to store atmospheric measurement", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unstructured-data", response_model=Dict[str, str])
async def store_unstructured_data(
    data: Dict[str, Any] = Body(...),
    collection_name: str = Query(...),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store arbitrary unstructured geological data"""
    try:
        document_id = await storage.store_unstructured_data(data, collection_name)
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


    except Exception as e:
        logger.error("Failed to store unstructured data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-insert", response_model=Dict[str, Any])
async def bulk_insert_documents(
    request: BulkInsertRequest,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Bulk insert multiple documents"""
    try:
        document_ids = await storage.bulk_insert(request.documents, request.collection_name)
        return {
            "inserted_ids": document_ids,
            "count": len(document_ids),
            "collection": request.collection_name,
            "status": "completed"
        }
    except Exception as e:
        logger.error("Failed bulk insert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/location", response_model=List[Dict[str, Any]])
async def query_by_location(
    query: LocationQuery,
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Query documents by geographic location"""
    try:
        results = await storage.query_by_location(
            query.longitude, query.latitude, query.radius_km, collection_name, limit
        )
        return results
    except Exception as e:
        logger.error("Failed location query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/time-range", response_model=List[Dict[str, Any]])
async def query_by_time_range(
    query: TimeRangeQuery,
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Query documents by time range"""
    try:
        results = await storage.query_by_time_range(
            query.start_time, query.end_time, collection_name, limit
        )
        return results
    except Exception as e:
        logger.error("Failed time range query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[Dict[str, Any]])
async def text_search(
    q: str = Query(..., min_length=1),
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Perform full-text search across documents"""
    try:
        results = await storage.text_search(q, collection_name, limit)
        return results
    except Exception as e:
        logger.error("Failed text search", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/geochemical-composition", response_model=Dict[str, Any])
async def aggregate_geochemical_composition(
    
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""
        
        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights
        
        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                # Feature extraction layers
                self.conv_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        AttentionBlock(64),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        AttentionBlock(128),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        AttentionBlock(256),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        AttentionBlock(512),
                        nn.AdaptiveAvgPool1d(1)
                    )
                ])
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return SeismicCNN(input_channels, num_classes).to(self.device)
    
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate),
            'frequency_content': self._analyze_frequency_content(waveform_data, sampling_rate)
        }
    
    def _preprocess_waveform(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        # Normalize
        normalized_data = (filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)) / \
                         (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))
    
    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds
        
        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window)/sta_window, mode='same')
        lta = np.convolve(energy, np.ones(lta_window)/lta_window, mode='same')
        
        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0
        
        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0
    
    def _analyze_frequency_content(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        from scipy import signal
        
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # Frequency bands of interest
        bands = {
            'low_freq': (0.1, 1.0),
            'mid_freq': (1.0, 10.0),
            'high_freq': (10.0, 25.0)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]
        
        return {
            'dominant_frequency': dominant_freq,
            'band_powers': band_powers,
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd),
            'spectral_bandwidth': np.sqrt(np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd))
        }
 Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate geochemical composition statistics"""
    try:
        time_range = None
        if start_time and end_time:
            time_range = (start_time, end_time)
        
        results = await storage.aggregate_geochemical_composition(sample_type, time_range)
        return results
    except Exception as e:
        logger.error("Failed geochemical composition aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/temporal-trends", response_model=Dict[str, Any])
async def aggregate_temporal_trends(
    collection_name: str = Query(...),
    field_name: str = Query(...),
    time_interval: str = Query("day", regex="^(hour|day|month|year)$"),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate temporal trends for a specific field"""
    try:
        results = await storage.aggregate_temporal_trends(
            collection_name, field_name, time_interval
        )
        return results
    except Exception as e:
        logger.error("Failed temporal trends aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/spatial-distribution", response_model=Dict[str, Any])
async def aggregate_spatial_distribution(
    collection_name: str = Query(...),
    grid_size: float = Query(0.1, gt=0, le=10),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate spatial distribution of data points"""
    try:
        results = await storage.aggregate_spatial_distribution(collection_name, grid_size)
        return results
    except Exception as e:
        logger.error("Failed spatial distribution aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aggregate/custom", response_model=Dict[str, Any])
async def custom_aggregation(
    request: AggregationRequest,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Execute custom aggregation pipeline"""
    try:
        collection = storage.collections.get(request.collection_name, request.collection_name)
        cursor = storage.db[collection].aggregate(request.pipeline)
        results = await cursor.to_list(length=None)
        
        # Convert ObjectId to string for JSON serialization
        for result in results:
            if '_id' in result and hasattr(result['_id'], '__str__'):
                result['_id'] = str(result['_id'])
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("Failed custom aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-report/{collection_name}", response_model=DataQualityResponse)
async def get_data_quality_report(
    collection_name: str,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Generate comprehensive data quality report"""
    try:
        report = await storage.get_data_quality_report(collection_name)
        return DataQualityResponse(**report)
    except Exception as e:
        logger.error("Failed to generate data quality report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup/{collection_name}")
async def cleanup_old_data(
    collection_name: str,
    days_to_keep: int = Query(365, ge=1, le=3650),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Clean up old data based on retention policy"""
    try:
        deleted_count = await storage.cleanup_old_data(collection_name, days_to_keep)
        return {
            "collection": collection_name,
            "deleted_count": deleted_count,
            "days_kept": days_to_keep,
            "status": "completed"
        }
    except Exception as e:
        logger.error("Failed to cleanup old data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections", response_model=Dict[str, List[str]])
async def list_collections(
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """List available collections"""
    try:
        collections = list(storage.collections.keys())
        return {"collections": collections}
    except Exception as e:
        logger.error("Failed to list collections", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Check MongoDB storage health"""
    try:
        is_healthy = await storage.is_healthy()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow(),
            "database": storage.database_name
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{collection_name}")
async def get_collection_stats(
    collection_name: str,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Get collection statistics"""
    try:
        collection = storage.collections.get(collection_name, collection_name)
        stats = await storage.db.command("collStats", collection)
        
        # Clean up the stats for JSON serialization
        clean_stats = {
            "collection": collection_name,
            "count": stats.get("count", 0),
            "size": stats.get("size", 0),
            "avgObjSize": stats.get("avgObjSize", 0),
            "storageSize": stats.get("storageSize", 0),
            "totalIndexSize": stats.get("totalIndexSize", 0),
            "indexSizes": stats.get("indexSizes", {}),
            "timestamp": datetime.utcnow()
        }
        
        return clean_stats
    except Exception as e:
        logger.error("Failed to get collection stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))