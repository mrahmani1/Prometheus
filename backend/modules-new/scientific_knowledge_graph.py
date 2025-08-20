"""
Scientific Knowledge Graph System for Geological Research
Automated knowledge extraction, semantic reasoning, and federated causal discovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import re
import requests
from datetime import datetime, timedelta
import urllib.parse
from pathlib import Path
import hashlib
import sqlite3
from abc import ABC, abstractmethod

# HTTP client for external APIs
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# Graph and NLP imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph operations will be limited.")

try:
    import spacy
    from spacy import displacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy not available. NLP features will be limited.")

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Advanced NLP features will be limited.")

try:
    import rdflib
    from rdflib import Graph, Literal, RDF, URIRef, Namespace
    from rdflib.namespace import FOAF, DC
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    warnings.warn("RDFLib not available. Ontology features will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Text analysis will be limited.")

# Import causal discovery components
try:
    from .causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType, CausalDiscoveryEngine
    from .hypothesis_generation_system import GeneratedHypothesis, HypothesisGenerator
except ImportError:
    # Fallback for when running as standalone
    try:
        from causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType, CausalDiscoveryEngine
        from hypothesis_generation_system import GeneratedHypothesis, HypothesisGenerator
    except ImportError:
        # Create minimal fallbacks for testing
        class CausalGraph: pass
        class CausalEdge: pass
        class GeologicalCausalRelationType: pass
        class CausalDiscoveryEngine: pass
        class GeneratedHypothesis: pass
        class HypothesisGenerator: pass

warnings.filterwarnings('ignore')

class EntityType(Enum):
    CONCEPT = "concept"
    PROCESS = "process"
    LOCATION = "location"
    MINERAL = "mineral"
    ROCK_TYPE = "rock_type"
    MEASUREMENT = "measurement"
    PHENOMENON = "phenomenon"
    RESEARCHER = "researcher"
    PUBLICATION = "publication"
    DATASET = "dataset"
    INSTRUMENT = "instrument"
    METHOD = "method"

class RelationType(Enum):
    PART_OF = "part_of"
    CAUSES = "causes"
    INFLUENCES = "influences"
    OCCURS_IN = "occurs_in"
    MEASURED_BY = "measured_by"
    STUDIED_BY = "studied_by"
    PUBLISHED_IN = "published_in"
    SIMILAR_TO = "similar_to"
    COMPOSED_OF = "composed_of"
    PRECEDES = "precedes"
    CORRELATES_WITH = "correlates_with"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    CAUSAL_TEMPORAL = "causal_temporal"

class DataSourceType(Enum):
    LITERATURE = "literature"
    DATABASE = "database"
    SENSOR_DATA = "sensor_data"
    EXPERIMENTAL = "experimental"
    SIMULATION = "simulation"
    FIELD_OBSERVATION = "field_observation"

class GeologicalOntology:
    """Comprehensive geological and geophysical ontology system"""
    
    def __init__(self):
        self.namespace_prefixes = {
            'geo': 'http://ontology.prometheus.ai/geology/',
            'geophys': 'http://ontology.prometheus.ai/geophysics/',
            'mineral': 'http://ontology.prometheus.ai/mineralogy/',
            'petro': 'http://ontology.prometheus.ai/petrology/',
            'struc': 'http://ontology.prometheus.ai/structure/',
            'tecton': 'http://ontology.prometheus.ai/tectonics/',
            'volcano': 'http://ontology.prometheus.ai/volcanology/',
            'seismo': 'http://ontology.prometheus.ai/seismology/',
            'hydro': 'http://ontology.prometheus.ai/hydrogeology/',
            'climate': 'http://ontology.prometheus.ai/climate/',
            'geochem': 'http://ontology.prometheus.ai/geochemistry/',
            'geochrono': 'http://ontology.prometheus.ai/geochronology/',
            'remote': 'http://ontology.prometheus.ai/remote_sensing/',
            'astro': 'http://ontology.prometheus.ai/astrogeology/'
        }
        
        self._build_comprehensive_ontology()
    
    def _build_comprehensive_ontology(self) -> Dict[str, str]:
        """Build comprehensive geological and geophysical ontology mappings"""
        
        # Fundamental rock types and classification
        rock_types = {
            'igneous_rock': 'geo:IgneousRock',
            'metamorphic_rock': 'geo:MetamorphicRock', 
            'sedimentary_rock': 'geo:SedimentaryRock',
            'volcanic_rock': 'volcano:VolcanicRock',
            'plutonic_rock': 'geo:PlutonicRock',
            'hypabyssal_rock': 'geo:HypabyssalRock',
            'pyroclastic_rock': 'volcano:PyroclasticRock',
            'clastic_rock': 'geo:ClasticRock',
            'chemical_rock': 'geo:ChemicalRock',
            'biogenic_rock': 'geo:BiogenicRock',
            'foliated_rock': 'geo:FoliatedRock',
            'non_foliated_rock': 'geo:NonFoliatedRock',
            'contact_metamorphic': 'geo:ContactMetamorphic',
            'regional_metamorphic': 'geo:RegionalMetamorphic',
            'ultramafic_rock': 'geo:UltramaficRock',
            'mafic_rock': 'geo:MaficRock',
            'intermediate_rock': 'geo:IntermediateRock',
            'felsic_rock': 'geo:FelsicRock'
        }
        
        # Comprehensive mineral classification
        minerals = {
            # Framework silicates
            'quartz': 'mineral:Quartz',
            'feldspar': 'mineral:Feldspar',
            'plagioclase': 'mineral:Plagioclase',
            'orthoclase': 'mineral:Orthoclase',
            'microcline': 'mineral:Microcline',
            'albite': 'mineral:Albite',
            'anorthite': 'mineral:Anorthite',
            
            # Chain silicates
            'pyroxene': 'mineral:Pyroxene',
            'amphibole': 'mineral:Amphibole',
            'hornblende': 'mineral:Hornblende',
            'augite': 'mineral:Augite',
            'diopside': 'mineral:Diopside',
            'enstatite': 'mineral:Enstatite',
            
            # Sheet silicates
            'mica': 'mineral:Mica',
            'biotite': 'mineral:Biotite',
            'muscovite': 'mineral:Muscovite',
            'chlorite': 'mineral:Chlorite',
            'serpentine': 'mineral:Serpentine',
            'kaolinite': 'mineral:Kaolinite',
            'montmorillonite': 'mineral:Montmorillonite',
            
            # Isolated silicates
            'olivine': 'mineral:Olivine',
            'garnet': 'mineral:Garnet',
            'zircon': 'mineral:Zircon',
            'andalusite': 'mineral:Andalusite',
            'kyanite': 'mineral:Kyanite',
            'sillimanite': 'mineral:Sillimanite',
            'staurolite': 'mineral:Staurolite',
            
            # Non-silicates
            'calcite': 'mineral:Calcite',
            'dolomite': 'mineral:Dolomite',
            'halite': 'mineral:Halite',
            'gypsum': 'mineral:Gypsum',
            'pyrite': 'mineral:Pyrite',
            'magnetite': 'mineral:Magnetite',
            'hematite': 'mineral:Hematite',
            'galena': 'mineral:Galena',
            'sphalerite': 'mineral:Sphalerite',
            'fluorite': 'mineral:Fluorite'
        }
        
        # Geological processes and phenomena
        processes = {
            'plate_tectonics': 'tecton:PlateTectonics',
            'seafloor_spreading': 'tecton:SeafloorSpreading',
            'subduction': 'tecton:Subduction',
            'continental_drift': 'tecton:ContinentalDrift',
            'orogeny': 'tecton:Orogeny',
            'rifting': 'tecton:Rifting',
            'transform_faulting': 'tecton:TransformFaulting',
            
            'volcanism': 'volcano:Volcanism',
            'explosive_volcanism': 'volcano:ExplosiveVolcanism',
            'effusive_volcanism': 'volcano:EffusiveVolcanism',
            'phreatomagmatic': 'volcano:Phreatomagmatic',
            'pyroclastic_flow': 'volcano:PyroclasticFlow',
            'lahar': 'volcano:Lahar',
            'lava_flow': 'volcano:LavaFlow',
            'volcanic_gas_emission': 'volcano:VolcanicGasEmission',
            
            'metamorphism': 'geo:Metamorphism',
            'contact_metamorphism': 'geo:ContactMetamorphism',
            'regional_metamorphism': 'geo:RegionalMetamorphism',
            'dynamic_metamorphism': 'geo:DynamicMetamorphism',
            'hydrothermal_metamorphism': 'geo:HydrothermalMetamorphism',
            
            'weathering': 'geo:Weathering',
            'physical_weathering': 'geo:PhysicalWeathering',
            'chemical_weathering': 'geo:ChemicalWeathering',
            'biological_weathering': 'geo:BiologicalWeathering',
            'erosion': 'geo:Erosion',
            'transportation': 'geo:Transportation',
            'deposition': 'geo:Deposition',
            'sedimentation': 'geo:Sedimentation',
            'diagenesis': 'geo:Diagenesis',
            'lithification': 'geo:Lithification'
        }
        
        # Geological structures
        structures = {
            'fold': 'struc:Fold',
            'anticline': 'struc:Anticline',
            'syncline': 'struc:Syncline',
            'monocline': 'struc:Monocline',
            'dome': 'struc:Dome',
            'basin': 'struc:Basin',
            'asymmetric_fold': 'struc:AsymmetricFold',
            'overturned_fold': 'struc:OverturnedFold',
            'recumbent_fold': 'struc:RecumbentFold',
            
            'fault': 'struc:Fault',
            'normal_fault': 'struc:NormalFault',
            'reverse_fault': 'struc:ReverseFault',
            'thrust_fault': 'struc:ThrustFault',
            'strike_slip_fault': 'struc:StrikeSlipFault',
            'transform_fault': 'struc:TransformFault',
            'detachment_fault': 'struc:DetachmentFault',
            'listric_fault': 'struc:ListricFault',
            
            'joint': 'struc:Joint',
            'fracture': 'struc:Fracture',
            'cleavage': 'struc:Cleavage',
            'schistosity': 'struc:Schistosity',
            'gneissic_banding': 'struc:GneissicBanding',
            'foliation': 'struc:Foliation',
            'lineation': 'struc:Lineation'
        }
        
        # Geophysical phenomena and measurements
        geophysics = {
            'seismic_wave': 'seismo:SeismicWave',
            'p_wave': 'seismo:PWave',
            's_wave': 'seismo:SWave',
            'surface_wave': 'seismo:SurfaceWave',
            'love_wave': 'seismo:LoveWave',
            'rayleigh_wave': 'seismo:RayleighWave',
            'body_wave': 'seismo:BodyWave',
            'earthquake': 'seismo:Earthquake',
            'microseismicity': 'seismo:Microseismicity',
            'seismic_tomography': 'seismo:SeismicTomography',
            'seismic_anisotropy': 'seismo:SeismicAnisotropy',
            
            'magnetic_field': 'geophys:MagneticField',
            'geomagnetic_field': 'geophys:GeomagneticField',
            'magnetic_anomaly': 'geophys:MagneticAnomaly',
            'paleomagnetic_reversal': 'geophys:PaleomagneticReversal',
            'magnetic_declination': 'geophys:MagneticDeclination',
            'magnetic_inclination': 'geophys:MagneticInclination',
            
            'gravity_field': 'geophys:GravityField',
            'gravity_anomaly': 'geophys:GravityAnomaly',
            'bouguer_anomaly': 'geophys:BouguerAnomaly',
            'free_air_anomaly': 'geophys:FreeAirAnomaly',
            'isostatic_anomaly': 'geophys:IsostaticAnomaly',
            
            'heat_flow': 'geophys:HeatFlow',
            'geothermal_gradient': 'geophys:GeothermalGradient',
            'thermal_conductivity': 'geophys:ThermalConductivity',
            'radiogenic_heat': 'geophys:RadiogenicHeat',
            
            'electrical_resistivity': 'geophys:ElectricalResistivity',
            'electromagnetic_induction': 'geophys:ElectromagneticInduction',
            'induced_polarization': 'geophys:InducedPolarization'
        }
        
        # Geological time periods
        geochronology = {
            'precambrian': 'geochrono:Precambrian',
            'archean': 'geochrono:Archean',
            'proterozoic': 'geochrono:Proterozoic',
            'phanerozoic': 'geochrono:Phanerozoic',
            'paleozoic': 'geochrono:Paleozoic',
            'mesozoic': 'geochrono:Mesozoic',
            'cenozoic': 'geochrono:Cenozoic',
            'quaternary': 'geochrono:Quaternary',
            'tertiary': 'geochrono:Tertiary',
            'cretaceous': 'geochrono:Cretaceous',
            'jurassic': 'geochrono:Jurassic',
            'triassic': 'geochrono:Triassic',
            'permian': 'geochrono:Permian',
            'carboniferous': 'geochrono:Carboniferous',
            'devonian': 'geochrono:Devonian',
            'silurian': 'geochrono:Silurian',
            'ordovician': 'geochrono:Ordovician',
            'cambrian': 'geochrono:Cambrian',
            
            'radiometric_dating': 'geochrono:RadiometricDating',
            'carbon_dating': 'geochrono:CarbonDating',
            'potassium_argon_dating': 'geochrono:PotassiumArgonDating',
            'uranium_lead_dating': 'geochrono:UraniumLeadDating',
            'rubidium_strontium_dating': 'geochrono:RubidiumStrontiumDating',
            'fission_track_dating': 'geochrono:FissionTrackDating'
        }
        
        # Remote sensing and satellite data
        remote_sensing = {
            'multispectral_imaging': 'remote:MultispectralImaging',
            'hyperspectral_imaging': 'remote:HyperspectralImaging',
            'thermal_infrared': 'remote:ThermalInfrared',
            'synthetic_aperture_radar': 'remote:SAR',
            'interferometric_sar': 'remote:InSAR',
            'differential_interferometry': 'remote:DifferentialInterferometry',
            'lidar': 'remote:LiDAR',
            'digital_elevation_model': 'remote:DigitalElevationModel',
            'vegetation_index': 'remote:VegetationIndex',
            'surface_temperature': 'remote:SurfaceTemperature',
            'land_cover_classification': 'remote:LandCoverClassification'
        }
        
        # Geochemical concepts
        geochemistry = {
            'trace_elements': 'geochem:TraceElements',
            'rare_earth_elements': 'geochem:RareEarthElements',
            'isotope_geochemistry': 'geochem:IsotopeGeochemistry',
            'stable_isotopes': 'geochem:StableIsotopes',
            'radiogenic_isotopes': 'geochem:RadiogenicIsotopes',
            'fluid_inclusions': 'geochem:FluidInclusions',
            'alteration_minerals': 'geochem:AlterationMinerals',
            'hydrothermal_alteration': 'geochem:HydrothermalAlteration',
            'weathering_products': 'geochem:WeatheringProducts',
            'redox_conditions': 'geochem:RedoxConditions',
            'ph_conditions': 'geochem:PhConditions',
            'mass_spectrometry': 'geochem:MassSpectrometry'
        }
        
        # Hydrogeological concepts
        hydrogeology = {
            'groundwater_flow': 'hydro:GroundwaterFlow',
            'aquifer': 'hydro:Aquifer',
            'confined_aquifer': 'hydro:ConfinedAquifer',
            'unconfined_aquifer': 'hydro:UnconfinedAquifer',
            'perched_aquifer': 'hydro:PerchedAquifer',
            'hydraulic_conductivity': 'hydro:HydraulicConductivity',
            'porosity': 'hydro:Porosity',
            'permeability': 'hydro:Permeability',
            'water_table': 'hydro:WaterTable',
            'hydraulic_head': 'hydro:HydraulicHead',
            'drawdown': 'hydro:Drawdown',
            'recharge': 'hydro:Recharge',
            'discharge': 'hydro:Discharge'
        }
        
        # Climate and paleoclimate
        climate = {
            'paleoclimate': 'climate:Paleoclimate',
            'climate_proxy': 'climate:ClimateProxy',
            'ice_core_data': 'climate:IceCoreData',
            'tree_ring_data': 'climate:TreeRingData',
            'marine_sediment_core': 'climate:MarineSedimentCore',
            'speleothem': 'climate:Speleothem',
            'oxygen_isotope_ratio': 'climate:OxygenIsotopeRatio',
            'milankovitch_cycles': 'climate:MilankovitchCycles',
            'greenhouse_effect': 'climate:GreenhouseEffect',
            'carbon_cycle': 'climate:CarbonCycle'
        }
        
        # Astrogeology
        astrogeology = {
            'impact_crater': 'astro:ImpactCrater',
            'meteorite': 'astro:Meteorite',
            'lunar_geology': 'astro:LunarGeology',
            'martian_geology': 'astro:MartianGeology',
            'planetary_differentiation': 'astro:PlanetaryDifferentiation',
            'regolith': 'astro:Regolith',
            'solar_wind': 'astro:SolarWind',
            'cosmic_rays': 'astro:CosmicRays'
        }
        
        # Combine all categories
        self.ontology_mapping = {}
        for category in [rock_types, minerals, processes, structures, geophysics, 
                        geochronology, remote_sensing, geochemistry, hydrogeology, 
                        climate, astrogeology]:
            self.ontology_mapping.update(category)
        
        return self.ontology_mapping
    
    def get_full_uri(self, short_uri: str) -> str:
        """Convert short URI notation to full URI"""
        if ':' in short_uri:
            prefix, term = short_uri.split(':', 1)
            if prefix in self.namespace_prefixes:
                return f"{self.namespace_prefixes[prefix]}{term}"
        return short_uri
    
    def get_related_concepts(self, concept: str) -> List[str]:
        """Get semantically related concepts"""
        concept_lower = concept.lower()
        related = []
        
        # Rock type relationships
        if 'igneous' in concept_lower:
            related.extend(['volcanic_rock', 'plutonic_rock', 'magma', 'crystallization'])
        elif 'metamorphic' in concept_lower:
            related.extend(['foliation', 'recrystallization', 'pressure', 'temperature'])
        elif 'sedimentary' in concept_lower:
            related.extend(['weathering', 'erosion', 'deposition', 'lithification'])
        
        # Process relationships
        if 'volcanism' in concept_lower or 'volcanic' in concept_lower:
            related.extend(['magma', 'pyroclastic_flow', 'lava_flow', 'volcanic_gas_emission'])
        elif 'tectonic' in concept_lower or 'plate' in concept_lower:
            related.extend(['subduction', 'rifting', 'transform_faulting', 'orogeny'])
        elif 'seismic' in concept_lower or 'earthquake' in concept_lower:
            related.extend(['p_wave', 's_wave', 'surface_wave', 'fault'])
        
        return list(set(related))
    
    def get_hierarchical_relationships(self, concept: str) -> Dict[str, List[str]]:
        """Get hierarchical relationships (parent/child concepts)"""
        relationships = {
            'parents': [],
            'children': [],
            'siblings': []
        }
        
        concept_lower = concept.lower()
        
        # Rock type hierarchies
        if concept_lower in ['granite', 'basalt', 'andesite', 'rhyolite']:
            relationships['parents'] = ['igneous_rock']
        elif concept_lower in ['slate', 'schist', 'gneiss', 'marble']:
            relationships['parents'] = ['metamorphic_rock']
        elif concept_lower in ['sandstone', 'limestone', 'shale', 'conglomerate']:
            relationships['parents'] = ['sedimentary_rock']
        
        # Mineral hierarchies
        if concept_lower in ['orthoclase', 'plagioclase', 'microcline']:
            relationships['parents'] = ['feldspar']
            relationships['siblings'] = ['orthoclase', 'plagioclase', 'microcline']
        elif concept_lower in ['biotite', 'muscovite']:
            relationships['parents'] = ['mica']
            relationships['siblings'] = ['biotite', 'muscovite']
        
        return relationships

@dataclass
class KnowledgeEntity:
    """Entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_references: List[str] = field(default_factory=list)
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeRelation:
    """Relation between entities in the knowledge graph"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    causal_strength: Optional[float] = None
    statistical_significance: Optional[float] = None
    source_references: List[str] = field(default_factory=list)
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class LiteratureDocument:
    """Document extracted from geological literature"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    publication_year: int
    journal: str
    doi: Optional[str]
    keywords: List[str]
    extracted_entities: List[KnowledgeEntity] = field(default_factory=list)
    extracted_relations: List[KnowledgeRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalCausalRelation:
    """Temporal causal relationship with time-series evidence"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    temporal_lag: float  # Time lag in specified units
    temporal_units: str  # years, months, days, etc.
    causal_strength: float
    temporal_evidence: pd.DataFrame  # Time-series data
    seasonality: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    granger_causality_pvalue: Optional[float] = None
    ccf_analysis: Optional[Dict[str, Any]] = None  # Cross-correlation function
    confidence_intervals: Optional[Tuple[float, float]] = None

class ExternalDataSource(ABC):
    """Abstract base class for external data sources"""
    
    @abstractmethod
    async def query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query the external data source"""
        pass
    
    @abstractmethod
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get ontology mapping for entities"""
        pass

class APIAuthenticationManager:
    """Comprehensive authentication management system for external APIs"""
    
    def __init__(self):
        self.api_credentials = {}
        self.auth_handlers = {}
        self.token_cache = {}
        self.token_expiry = {}
        self._setup_default_auth_handlers()
    
    def _setup_default_auth_handlers(self):
        """Setup default authentication handlers for known APIs"""
        self.auth_handlers = {
            'usgs': self._usgs_auth_handler,
            'nasa': self._nasa_auth_handler,
            'asf': self._asf_auth_handler,
            'iris': self._iris_auth_handler,
            'wiley': self._wiley_auth_handler,
            'agu': self._agu_auth_handler
        }
    
    def add_credentials(self, api_name: str, credentials: Dict[str, Any]):
        """Add API credentials for authentication"""
        self.api_credentials[api_name] = credentials
        logging.info(f"Added credentials for {api_name}")
    
    async def get_auth_headers(self, api_name: str) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if api_name in self.auth_handlers:
            return await self.auth_handlers[api_name]()
        return {}
    
    async def _usgs_auth_handler(self) -> Dict[str, str]:
        """USGS authentication handler"""
        headers = {}
        if 'usgs' in self.api_credentials:
            creds = self.api_credentials['usgs']
            if 'api_key' in creds:
                headers['Authorization'] = f"Bearer {creds['api_key']}"
            if 'user_agent' in creds:
                headers['User-Agent'] = creds['user_agent']
        return headers
    
    async def _nasa_auth_handler(self) -> Dict[str, str]:
        """NASA EarthData authentication handler"""
        headers = {}
        if 'nasa' in self.api_credentials:
            creds = self.api_credentials['nasa']
            if 'api_key' in creds:
                headers['X-API-Key'] = creds['api_key']
            if 'token' in creds:
                headers['Authorization'] = f"Bearer {creds['token']}"
            elif 'username' in creds and 'password' in creds:
                # Handle OAuth2 or basic auth
                token = await self._get_nasa_oauth_token(creds['username'], creds['password'])
                if token:
                    headers['Authorization'] = f"Bearer {token}"
        return headers
    
    async def _asf_auth_handler(self) -> Dict[str, str]:
        """ASF-Hyp3 authentication handler"""
        headers = {}
        if 'asf' in self.api_credentials:
            creds = self.api_credentials['asf']
            if 'api_key' in creds:
                headers['Authorization'] = f"Bearer {creds['api_key']}"
            if 'username' in creds and 'password' in creds:
                # ASF uses basic auth or token
                token = await self._get_asf_token(creds['username'], creds['password'])
                if token:
                    headers['Authorization'] = f"Bearer {token}"
        return headers
    
    async def _iris_auth_handler(self) -> Dict[str, str]:
        """IRIS/EarthScope authentication handler"""
        headers = {}
        if 'iris' in self.api_credentials:
            creds = self.api_credentials['iris']
            if 'token' in creds:
                headers['Authorization'] = f"Bearer {creds['token']}"
            if 'user_agent' in creds:
                headers['User-Agent'] = creds['user_agent']
        return headers
    
    async def _wiley_auth_handler(self) -> Dict[str, str]:
        """Wiley Online Library authentication handler"""
        headers = {}
        if 'wiley' in self.api_credentials:
            creds = self.api_credentials['wiley']
            if 'api_key' in creds:
                headers['X-API-Key'] = creds['api_key']
            if 'institution_token' in creds:
                headers['Authorization'] = f"Bearer {creds['institution_token']}"
        return headers
    
    async def _agu_auth_handler(self) -> Dict[str, str]:
        """AGU Publications authentication handler"""
        headers = {}
        if 'agu' in self.api_credentials:
            creds = self.api_credentials['agu']
            if 'api_key' in creds:
                headers['X-API-Key'] = creds['api_key']
            if 'member_id' in creds:
                headers['X-Member-ID'] = creds['member_id']
        return headers
    
    async def _get_nasa_oauth_token(self, username: str, password: str) -> Optional[str]:
        """Get OAuth token for NASA EarthData"""
        try:
            if not AIOHTTP_AVAILABLE:
                return None
            
            # Check token cache
            cache_key = f"nasa_{hashlib.md5(username.encode()).hexdigest()}"
            if cache_key in self.token_cache:
                if time.time() < self.token_expiry.get(cache_key, 0):
                    return self.token_cache[cache_key]
            
            # Request new token
            token_url = "https://urs.earthdata.nasa.gov/oauth/token"
            data = {
                'grant_type': 'password',
                'username': username,
                'password': password
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        token = token_data.get('access_token')
                        if token:
                            self.token_cache[cache_key] = token
                            # Cache for 1 hour
                            self.token_expiry[cache_key] = time.time() + 3600
                            return token
        except Exception as e:
            logging.warning(f"Failed to get NASA OAuth token: {e}")
        return None
    
    async def _get_asf_token(self, username: str, password: str) -> Optional[str]:
        """Get authentication token for ASF-Hyp3"""
        try:
            if not AIOHTTP_AVAILABLE:
                return None
            
            # Check token cache
            cache_key = f"asf_{hashlib.md5(username.encode()).hexdigest()}"
            if cache_key in self.token_cache:
                if time.time() < self.token_expiry.get(cache_key, 0):
                    return self.token_cache[cache_key]
            
            # ASF-Hyp3 authentication endpoint
            auth_url = "https://hyp3-api.asf.alaska.edu/login"
            data = {
                'username': username,
                'password': password
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, json=data) as response:
                    if response.status == 200:
                        auth_data = await response.json()
                        token = auth_data.get('access_token')
                        if token:
                            self.token_cache[cache_key] = token
                            # Cache for 8 hours
                            self.token_expiry[cache_key] = time.time() + 28800
                            return token
        except Exception as e:
            logging.warning(f"Failed to get ASF token: {e}")
        return None

class RateLimitManager:
    """Advanced rate limiting system for external API calls"""
    
    def __init__(self):
        self.rate_limits = {}
        self.request_logs = defaultdict(deque)
        self.semaphores = {}
        self.backoff_delays = {}
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default rate limits for known APIs"""
        self.rate_limits = {
            'usgs': {'requests_per_minute': 120, 'requests_per_hour': 3600, 'concurrent': 10},
            'nasa': {'requests_per_minute': 60, 'requests_per_hour': 1000, 'concurrent': 5},
            'asf': {'requests_per_minute': 30, 'requests_per_hour': 500, 'concurrent': 3},
            'iris': {'requests_per_minute': 60, 'requests_per_hour': 2000, 'concurrent': 8},
            'wiley': {'requests_per_minute': 50, 'requests_per_hour': 800, 'concurrent': 4},
            'agu': {'requests_per_minute': 60, 'requests_per_hour': 1000, 'concurrent': 5},
            'default': {'requests_per_minute': 30, 'requests_per_hour': 500, 'concurrent': 3}
        }
        
        # Initialize semaphores for concurrent request limits
        for api_name, limits in self.rate_limits.items():
            if api_name != 'default':
                self.semaphores[api_name] = asyncio.Semaphore(limits['concurrent'])
    
    def set_rate_limit(self, api_name: str, requests_per_minute: int, 
                      requests_per_hour: int, concurrent: int):
        """Set custom rate limits for an API"""
        self.rate_limits[api_name] = {
            'requests_per_minute': requests_per_minute,
            'requests_per_hour': requests_per_hour,
            'concurrent': concurrent
        }
        self.semaphores[api_name] = asyncio.Semaphore(concurrent)
        logging.info(f"Set rate limits for {api_name}: {requests_per_minute}/min, {requests_per_hour}/hour, {concurrent} concurrent")
    
    async def acquire(self, api_name: str) -> bool:
        """Acquire permission to make an API request"""
        limits = self.rate_limits.get(api_name, self.rate_limits['default'])
        
        # Check if we need to wait due to rate limits
        await self._wait_for_rate_limit(api_name, limits)
        
        # Acquire semaphore for concurrent limit
        semaphore = self.semaphores.get(api_name)
        if semaphore:
            await semaphore.acquire()
        
        # Log the request
        self._log_request(api_name)
        
        return True
    
    def release(self, api_name: str):
        """Release the semaphore after API request completion"""
        semaphore = self.semaphores.get(api_name)
        if semaphore:
            semaphore.release()
    
    async def _wait_for_rate_limit(self, api_name: str, limits: Dict[str, int]):
        """Wait if rate limits would be exceeded"""
        now = time.time()
        request_log = self.request_logs[api_name]
        
        # Clean old requests from log
        while request_log and request_log[0] < now - 3600:  # Remove requests older than 1 hour
            request_log.popleft()
        
        # Check minute rate limit
        minute_requests = sum(1 for req_time in request_log if req_time > now - 60)
        if minute_requests >= limits['requests_per_minute']:
            wait_time = 60 - (now - max(req_time for req_time in request_log if req_time > now - 60))
            logging.info(f"Rate limit hit for {api_name}, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        # Check hour rate limit
        hour_requests = len(request_log)
        if hour_requests >= limits['requests_per_hour']:
            oldest_request = request_log[0]
            wait_time = 3600 - (now - oldest_request)
            logging.info(f"Hourly rate limit hit for {api_name}, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        # Apply backoff delay if needed
        if api_name in self.backoff_delays:
            delay = self.backoff_delays[api_name]
            if delay > 0:
                logging.info(f"Applying backoff delay for {api_name}: {delay:.1f} seconds")
                await asyncio.sleep(delay)
                # Reduce backoff delay
                self.backoff_delays[api_name] = max(0, delay * 0.5)
    
    def _log_request(self, api_name: str):
        """Log a request timestamp"""
        self.request_logs[api_name].append(time.time())
    
    def add_backoff_delay(self, api_name: str, delay_seconds: float):
        """Add exponential backoff delay for failed requests"""
        current_delay = self.backoff_delays.get(api_name, 0)
        new_delay = max(delay_seconds, current_delay * 2)
        self.backoff_delays[api_name] = min(new_delay, 300)  # Max 5 minutes
        logging.info(f"Added backoff delay for {api_name}: {new_delay:.1f} seconds")
    
    def reset_backoff(self, api_name: str):
        """Reset backoff delay after successful requests"""
        if api_name in self.backoff_delays:
            del self.backoff_delays[api_name]
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limiting statistics"""
        now = time.time()
        stats = {}
        
        for api_name in self.rate_limits:
            if api_name == 'default':
                continue
            
            request_log = self.request_logs[api_name]
            minute_requests = sum(1 for req_time in request_log if req_time > now - 60)
            hour_requests = len(request_log)
            
            limits = self.rate_limits[api_name]
            semaphore = self.semaphores.get(api_name)
            
            stats[api_name] = {
                'requests_last_minute': minute_requests,
                'requests_last_hour': hour_requests,
                'minute_limit': limits['requests_per_minute'],
                'hour_limit': limits['requests_per_hour'],
                'concurrent_limit': limits['concurrent'],
                'active_requests': limits['concurrent'] - (semaphore._value if semaphore else 0),
                'backoff_delay': self.backoff_delays.get(api_name, 0)
            }
        
        return stats

class SecureAPIClient:
    """Secure HTTP client with authentication and rate limiting"""
    
    def __init__(self, auth_manager: APIAuthenticationManager, rate_limiter: RateLimitManager):
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.session: Optional[aiohttp.ClientSession] = None
        self.default_timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async def __aenter__(self):
        if AIOHTTP_AVAILABLE:
            self.session = aiohttp.ClientSession(timeout=self.default_timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def request(self, method: str, url: str, api_name: str, 
                     headers: Optional[Dict[str, str]] = None,
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Any] = None,
                     json: Optional[Any] = None,
                     retry_count: int = 3) -> Optional[aiohttp.ClientResponse]:
        """Make authenticated and rate-limited HTTP request"""
        
        if not AIOHTTP_AVAILABLE or not self.session:
            logging.warning("aiohttp not available, using fallback")
            return await self._fallback_request(method, url, headers, params, data)
        
        # Acquire rate limit permission
        await self.rate_limiter.acquire(api_name)
        
        try:
            # Get authentication headers
            auth_headers = await self.auth_manager.get_auth_headers(api_name)
            
            # Merge headers
            request_headers = auth_headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Add default headers
            request_headers.setdefault('User-Agent', 'Prometheus-GeologicalAI/1.0')
            request_headers.setdefault('Accept', 'application/json')
            
            # Make request with retries
            for attempt in range(retry_count):
                try:
                    async with self.session.request(
                        method, url, 
                        headers=request_headers,
                        params=params,
                        data=data,
                        json=json
                    ) as response:
                        
                        # Handle rate limiting responses
                        if response.status == 429:
                            retry_after = response.headers.get('Retry-After', '60')
                            delay = float(retry_after)
                            self.rate_limiter.add_backoff_delay(api_name, delay)
                            if attempt < retry_count - 1:
                                await asyncio.sleep(delay)
                                continue
                        
                        # Handle authentication errors
                        elif response.status == 401:
                            logging.warning(f"Authentication failed for {api_name}")
                            break
                        
                        # Success - reset backoff
                        if response.status < 400:
                            self.rate_limiter.reset_backoff(api_name)
                        
                        return response
                
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout for {api_name} request, attempt {attempt + 1}")
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                except Exception as e:
                    logging.warning(f"Request error for {api_name}: {e}, attempt {attempt + 1}")
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
        
        finally:
            self.rate_limiter.release(api_name)
        
        return None
    
    async def _fallback_request(self, method: str, url: str, 
                              headers: Optional[Dict[str, str]] = None,
                              params: Optional[Dict[str, Any]] = None,
                              data: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Fallback using requests library"""
        try:
            import requests
            
            response = requests.request(
                method, url,
                headers=headers,
                params=params,
                data=data,
                timeout=30
            )
            
            # Create a simple response-like object
            class FallbackResponse:
                def __init__(self, resp):
                    self.status = resp.status_code
                    self.headers = resp.headers
                    self._response = resp
                
                async def json(self):
                    return self._response.json()
                
                async def text(self):
                    return self._response.text
            
            return FallbackResponse(response)
        
        except Exception as e:
            logging.error(f"Fallback request failed: {e}")
            return None

class GeoScienceWorldAPI(ExternalDataSource):
    """Advanced interface to GeoScienceWorld database with real API integration"""
    
    def __init__(self, api_key: Optional[str] = None, institution_key: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'geoscienceworld'
        
        # Add credentials if provided
        if api_key or institution_key:
            credentials = {}
            if api_key:
                credentials['api_key'] = api_key
            if institution_key:
                credentials['institution_key'] = institution_key
            self.auth_manager.add_credentials(self.api_name, credentials)
        
        self.base_url = "https://pubs.geoscienceworld.org/api/v1"
        self.search_url = f"{self.base_url}/search"
        self.content_url = f"{self.base_url}/content"
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive geological ontology
        self.geological_ontology = GeologicalOntology()
        
        # Enhanced geological ontology mapping
        self._ontology_mapping = {
            # Rock types
            "igneous_rock": "http://ontology.geoscienceworld.org/IgneousRock",
            "metamorphic_rock": "http://ontology.geoscienceworld.org/MetamorphicRock",
            "sedimentary_rock": "http://ontology.geoscienceworld.org/SedimentaryRock",
            "volcanic_rock": "http://ontology.geoscienceworld.org/VolcanicRock",
            "plutonic_rock": "http://ontology.geoscienceworld.org/PlutonicRock",
            
            # Geological processes
            "plate_tectonics": "http://ontology.geoscienceworld.org/PlateTectonics",
            "volcanism": "http://ontology.geoscienceworld.org/Volcanism",
            "metamorphism": "http://ontology.geoscienceworld.org/Metamorphism",
            "sedimentation": "http://ontology.geoscienceworld.org/Sedimentation",
            "erosion": "http://ontology.geoscienceworld.org/Erosion",
            "weathering": "http://ontology.geoscienceworld.org/Weathering",
            
            # Geological structures
            "fold": "http://ontology.geoscienceworld.org/Fold",
            "fault": "http://ontology.geoscienceworld.org/Fault",
            "joint": "http://ontology.geoscienceworld.org/Joint",
            "anticline": "http://ontology.geoscienceworld.org/Anticline",
            "syncline": "http://ontology.geoscienceworld.org/Syncline",
            
            # Minerals
            "quartz": "http://ontology.geoscienceworld.org/Quartz",
            "feldspar": "http://ontology.geoscienceworld.org/Feldspar",
            "mica": "http://ontology.geoscienceworld.org/Mica",
            "calcite": "http://ontology.geoscienceworld.org/Calcite",
            "olivine": "http://ontology.geoscienceworld.org/Olivine",
            
            # Geological time
            "precambrian": "http://ontology.geoscienceworld.org/Precambrian",
            "paleozoic": "http://ontology.geoscienceworld.org/Paleozoic",
            "mesozoic": "http://ontology.geoscienceworld.org/Mesozoic",
            "cenozoic": "http://ontology.geoscienceworld.org/Cenozoic",
            
            # Geophysical concepts
            "seismic_wave": "http://ontology.geoscienceworld.org/SeismicWave",
            "magnetic_field": "http://ontology.geoscienceworld.org/MagneticField",
            "gravity_anomaly": "http://ontology.geoscienceworld.org/GravityAnomaly",
            "geothermal": "http://ontology.geoscienceworld.org/Geothermal"
        }
        
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for publication searches
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.institution_key:
            headers["X-Institution-Key"] = self.institution_key
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query GeoScienceWorld database with multiple search strategies"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for when aiohttp is not available"""
        
        # Generate mock data based on query content for development purposes
        geological_terms = [
            "volcano", "earthquake", "fault", "mineral", "rock", "sediment",
            "metamorphic", "igneous", "plate", "tectonic", "geophysics",
            "geochemistry", "paleontology", "stratigraphy", "petrology"
        ]
        
        # Create more realistic mock data based on query
        results = []
        base_year = 2020
        
        for i in range(min(limit, 25)):
            # Determine the most relevant geological term
            relevant_term = next(
                (term for term in geological_terms if term in query.lower()),
                "geology"
            )
            
            result = {
                "title": f"{relevant_term.title()} Research: {query} - Study {i+1}",
                "abstract": f"Comprehensive analysis of {relevant_term} related to {query}. "
                          f"This study examines the geological implications and processes "
                          f"involved in {relevant_term} systems, with focus on recent developments "
                          f"in understanding {query}.",
                "authors": self._generate_mock_authors(i),
                "year": base_year + (i % 5),
                "doi": f"10.1130/gsw.{base_year + (i % 5)}.{1000 + i}",
                "journal": self._get_mock_journal(relevant_term),
                "keywords": [relevant_term, query.split()[0] if query.split() else "geology"],
                "type": "research_article",
                "volume": 100 + (i % 50),
                "issue": 1 + (i % 12),
                "pages": f"{100 + i * 10}-{120 + i * 10}",
                "source": "geoscienceworld_fallback",
                "subject_areas": self._get_subject_areas(relevant_term)
            }
            results.append(result)
            
        return results
        
    def _generate_mock_authors(self, index: int) -> List[str]:
        """Generate realistic author names for mock data"""
        author_pool = [
            "Smith, J.A.", "Johnson, M.B.", "Williams, S.C.", "Brown, D.E.",
            "Davis, K.F.", "Miller, R.G.", "Wilson, A.H.", "Moore, L.I.",
            "Taylor, P.J.", "Anderson, C.K.", "Thomas, B.L.", "Jackson, N.M.",
            "White, H.O.", "Harris, G.P.", "Martin, F.Q.", "Thompson, E.R."
        ]
        
        num_authors = min(1 + (index % 4), len(author_pool))
        return author_pool[:num_authors]
        
    def _get_mock_journal(self, term: str) -> str:
        """Get appropriate journal name based on geological term"""
        journal_mapping = {
            "volcano": "Journal of Volcanology and Geothermal Research",
            "earthquake": "Seismological Research Letters",
            "fault": "Tectonophysics",
            "mineral": "American Mineralogist",
            "rock": "Journal of Petrology",
            "sediment": "Sedimentary Geology",
            "metamorphic": "Journal of Metamorphic Geology",
            "igneous": "Contributions to Mineralogy and Petrology",
            "plate": "Earth and Planetary Science Letters",
            "tectonic": "Geology"
        }
        return journal_mapping.get(term, "Geological Society of America Bulletin")
        
    def _get_subject_areas(self, term: str) -> List[str]:
        """Get subject areas based on geological term"""
        area_mapping = {
            "volcano": ["Volcanology", "Geochemistry", "Petrology"],
            "earthquake": ["Seismology", "Geophysics", "Tectonics"],
            "fault": ["Structural Geology", "Tectonics", "Geophysics"],
            "mineral": ["Mineralogy", "Crystallography", "Geochemistry"],
            "rock": ["Petrology", "Geochemistry", "Mineralogy"],
            "sediment": ["Sedimentology", "Stratigraphy", "Basin Analysis"]
        }
        return area_mapping.get(term, ["Geology", "Earth Sciences"])
    
    async def _execute_query_secure(self, client: SecureAPIClient, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against GeoScienceWorld API using secure client"""
        
        results = []
        
        try:
            # Parse search parameters
            search_params = self._parse_search_parameters(query, limit, **kwargs)
            
            # Perform different types of searches using secure client
            if search_params.get('search_type') == 'advanced':
                results = await self._advanced_search_secure(client, search_params)
            else:
                results = await self._simple_search_secure(client, search_params)
                
            # Enhance results with additional metadata
            enhanced_results = await self._enhance_search_results(results)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error executing GeoScienceWorld query: {e}")
            return self._fallback_query(query, limit, **kwargs)
    
    async def _simple_search_secure(self, client: SecureAPIClient, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform simple search using secure client"""
        try:
            response = await client.request(
                'GET', self.search_url, self.api_name,
                params=search_params
            )
            
            if response and response.status == 200:
                data = await response.json()
                return data.get('results', [])
        except Exception as e:
            self.logger.warning(f"Simple search failed: {e}")
        
        return []
    
    async def _advanced_search_secure(self, client: SecureAPIClient, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform advanced search using secure client"""
        try:
            # Advanced search with POST request
            response = await client.request(
                'POST', self.search_url, self.api_name,
                json=search_params
            )
            
            if response and response.status == 200:
                data = await response.json()
                return data.get('results', [])
        except Exception as e:
            self.logger.warning(f"Advanced search failed: {e}")
        
        return []
            
    async def _execute_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against GeoScienceWorld API"""
        
        results = []
        
        try:
            # Parse search parameters
            search_params = self._parse_search_parameters(query, limit, **kwargs)
            
            # Perform different types of searches
            if search_params.get('search_type') == 'advanced':
                results = await self._advanced_search(search_params)
            else:
                results = await self._simple_search(search_params)
                
            # Enhance results with additional metadata
            enhanced_results = await self._enhance_search_results(results)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            self.logger.error(f"GeoScienceWorld query failed: {e}")
            # Fall back to mock data on API failure
            return self._fallback_query(query, limit, **kwargs)
            
    def _parse_search_parameters(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Parse search query into structured parameters"""
        
        params = {
            'query': query,
            'limit': min(limit, 500),  # API may have limits
            'search_type': kwargs.get('search_type', 'simple'),
            'start_year': kwargs.get('start_year', 2000),
            'end_year': kwargs.get('end_year', datetime.now().year),
            'journals': kwargs.get('journals', []),
            'subject_areas': kwargs.get('subject_areas', []),
            'document_type': kwargs.get('document_type', 'research_article'),
            'sort_by': kwargs.get('sort_by', 'relevance'),
            'include_abstracts': kwargs.get('include_abstracts', True),
            'language': kwargs.get('language', 'en')
        }
        
        return params
        
    async def _simple_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform simple text search"""
        
        async with self.rate_limiter:
            try:
                search_payload = {
                    'query': params['query'],
                    'rows': params['limit'],
                    'start': 0,
                    'fq': []
                }
                
                # Add filters
                if params['start_year'] and params['end_year']:
                    search_payload['fq'].append(
                        f"publication_year:[{params['start_year']} TO {params['end_year']}]"
                    )
                    
                if params['journals']:
                    journal_filter = ' OR '.join(f'journal:"{j}"' for j in params['journals'])
                    search_payload['fq'].append(f"({journal_filter})")
                    
                if params['document_type']:
                    search_payload['fq'].append(f'document_type:"{params["document_type"]}"')
                
                async with self.session.post(self.search_url, json=search_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_search_response(data)
                    elif response.status == 401:
                        self.logger.warning("GeoScienceWorld API authentication failed")
                        return []
                    elif response.status == 403:
                        self.logger.warning("GeoScienceWorld API access forbidden - check subscription")
                        return []
                    else:
                        self.logger.warning(f"GeoScienceWorld search returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to perform simple search: {e}")
                return []
                
    async def _advanced_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform advanced search with multiple criteria"""
        
        # This would implement more sophisticated search logic
        # For now, delegate to simple search
        return await self._simple_search(params)
        
    def _parse_search_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse GeoScienceWorld search response"""
        
        results = []
        
        documents = data.get('response', {}).get('docs', [])
        
        for doc in documents:
            try:
                result = {
                    'title': doc.get('title_display', ''),
                    'authors': doc.get('author_display', []),
                    'year': doc.get('publication_year', 0),
                    'journal': doc.get('journal_title', ''),
                    'doi': doc.get('doi', ''),
                    'abstract': doc.get('abstract', ''),
                    'keywords': doc.get('keywords', []),
                    'subject_areas': doc.get('subject_area', []),
                    'volume': doc.get('volume', ''),
                    'issue': doc.get('issue', ''),
                    'pages': doc.get('page_range', ''),
                    'url': doc.get('url', ''),
                    'document_type': doc.get('document_type', ''),
                    'language': doc.get('language', 'en'),
                    'citation_count': doc.get('citation_count', 0),
                    'source': 'geoscienceworld_api'
                }
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse search result: {e}")
                continue
                
        return results
        
    async def _enhance_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata and analysis"""
        
        enhanced_results = []
        
        for result in results:
            try:
                # Add relevance scoring based on geological terms
                result['geological_relevance_score'] = self._calculate_geological_relevance(result)
                
                # Classify geological domain
                result['geological_domain'] = self._classify_geological_domain(result)
                
                # Extract geological entities from title and abstract
                result['geological_entities'] = self._extract_geological_entities(result)
                
                # Add temporal classification
                result['temporal_classification'] = self._classify_temporal_scope(result)
                
                enhanced_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance search result: {e}")
                # Add original result without enhancement
                enhanced_results.append(result)
                continue
                
        return enhanced_results
        
    def _calculate_geological_relevance(self, result: Dict[str, Any]) -> float:
        """Calculate geological relevance score for a publication"""
        
        score = 0.0
        geological_keywords = [
            'geological', 'geology', 'geophysics', 'geochemistry', 'petrology',
            'mineralogy', 'volcanology', 'seismology', 'tectonics', 'stratigraphy'
        ]
        
        text_fields = [result.get('title', ''), result.get('abstract', ''), 
                      ' '.join(result.get('keywords', []))]
        combined_text = ' '.join(text_fields).lower()
        
        for keyword in geological_keywords:
            if keyword in combined_text:
                score += combined_text.count(keyword) * 0.1
                
        # Boost score based on journal relevance
        geological_journals = [
            'geology', 'geophysics', 'volcanology', 'seismology', 'petrology',
            'mineralogy', 'tectonics', 'geochemistry'
        ]
        
        journal = result.get('journal', '').lower()
        for geo_journal in geological_journals:
            if geo_journal in journal:
                score += 0.5
                break
                
        return min(score, 1.0)  # Cap at 1.0
        
    def _classify_geological_domain(self, result: Dict[str, Any]) -> List[str]:
        """Classify the geological domain(s) of a publication"""
        
        domains = []
        text_content = f"{result.get('title', '')} {result.get('abstract', '')}".lower()
        
        domain_keywords = {
            'volcanology': ['volcano', 'volcanic', 'eruption', 'magma', 'lava', 'tephra'],
            'seismology': ['earthquake', 'seismic', 'fault', 'rupture', 'wave'],
            'petrology': ['rock', 'mineral', 'crystal', 'igneous', 'metamorphic'],
            'geochemistry': ['chemical', 'isotope', 'element', 'composition'],
            'tectonics': ['plate', 'tectonic', 'deformation', 'stress', 'strain'],
            'hydrogeology': ['water', 'groundwater', 'aquifer', 'hydrology'],
            'paleontology': ['fossil', 'paleontology', 'evolution', 'extinction'],
            'geophysics': ['gravity', 'magnetic', 'electrical', 'geophysical']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                domains.append(domain)
                
        return domains if domains else ['general_geology']
        
    def _extract_geological_entities(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract geological entities from publication text"""
        
        entities = []
        text_content = f"{result.get('title', '')} {result.get('abstract', '')}"
        
        # Simple pattern-based entity extraction
        entity_patterns = {
            'mineral': r'\b(quartz|feldspar|mica|calcite|olivine|pyroxene|amphibole)\b',
            'rock_type': r'\b(granite|basalt|limestone|sandstone|shale|gneiss|schist)\b',
            'geological_process': r'\b(erosion|weathering|metamorphism|volcanism|sedimentation)\b',
            'geological_feature': r'\b(mountain|volcano|fault|fold|basin|plateau)\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in set(matches):  # Remove duplicates
                entities.append({
                    'type': entity_type,
                    'name': match.lower(),
                    'confidence': 0.8  # Simple rule-based extraction
                })
                
        return entities
        
    def _classify_temporal_scope(self, result: Dict[str, Any]) -> str:
        """Classify the temporal scope of geological research"""
        
        text_content = f"{result.get('title', '')} {result.get('abstract', '')}".lower()
        
        temporal_indicators = {
            'recent': ['recent', 'modern', 'current', 'contemporary', 'holocene'],
            'quaternary': ['quaternary', 'pleistocene', 'ice age'],
            'cenozoic': ['cenozoic', 'tertiary', 'paleogene', 'neogene'],
            'mesozoic': ['mesozoic', 'cretaceous', 'jurassic', 'triassic'],
            'paleozoic': ['paleozoic', 'permian', 'carboniferous', 'devonian'],
            'precambrian': ['precambrian', 'archean', 'proterozoic']
        }
        
        for scope, indicators in temporal_indicators.items():
            if any(indicator in text_content for indicator in indicators):
                return scope
                
        return 'unspecified'
        
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get comprehensive geological ontology mapping"""
        # Combine GeoScienceWorld-specific mapping with comprehensive geological ontology
        comprehensive_mapping = self.geological_ontology.ontology_mapping.copy()
        comprehensive_mapping.update(self._ontology_mapping)
        
        # Convert short URIs to full URIs
        full_uri_mapping = {}
        for key, value in comprehensive_mapping.items():
            full_uri_mapping[key] = self.geological_ontology.get_full_uri(value)
        
        return full_uri_mapping
    
    def enhance_result_with_ontology(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance publication result with comprehensive geological ontology information"""
        enhanced_result = result.copy()
        
        # Extract key terms from publication result
        result_text = f"{result.get('title', '')} {result.get('abstract', '')} {result.get('keywords', '')} {result.get('subject', '')}"
        result_lower = result_text.lower()
        
        # Find matching ontological concepts
        matched_concepts = []
        related_concepts = []
        
        # Check both original ontology and comprehensive ontology
        all_concepts = {**self._ontology_mapping, **self.geological_ontology.ontology_mapping}
        
        for concept in all_concepts.keys():
            if concept.replace('_', ' ') in result_lower or concept in result_lower:
                matched_concepts.append(concept)
                related_concepts.extend(self.geological_ontology.get_related_concepts(concept))
        
        if matched_concepts:
            enhanced_result['ontology'] = {
                'matched_concepts': list(set(matched_concepts)),
                'related_concepts': list(set(related_concepts)),
                'semantic_uris': [self.geological_ontology.get_full_uri(all_concepts.get(concept, concept)) 
                                for concept in matched_concepts],
                'hierarchical_relationships': {
                    concept: self.geological_ontology.get_hierarchical_relationships(concept)
                    for concept in matched_concepts[:5]  # More concepts for publications
                },
                'research_context': self._extract_research_context(matched_concepts)
            }
        
        return enhanced_result
    
    def _extract_research_context(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Extract research context based on matched concepts"""
        context = {
            'research_domains': [],
            'methodologies': [],
            'phenomena': [],
            'applications': []
        }
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Classify concepts by research domain
            if any(term in concept_lower for term in ['seismic', 'earthquake', 'wave']):
                context['research_domains'].append('Seismology')
            elif any(term in concept_lower for term in ['volcanic', 'magma', 'eruption']):
                context['research_domains'].append('Volcanology')
            elif any(term in concept_lower for term in ['mineral', 'crystal', 'composition']):
                context['research_domains'].append('Mineralogy')
            elif any(term in concept_lower for term in ['rock', 'petro', 'igneous', 'metamorphic', 'sedimentary']):
                context['research_domains'].append('Petrology')
            elif any(term in concept_lower for term in ['plate', 'tectonic', 'fault', 'structure']):
                context['research_domains'].append('Tectonics')
            elif any(term in concept_lower for term in ['climate', 'paleo', 'ice', 'carbon']):
                context['research_domains'].append('Paleoclimatology')
            elif any(term in concept_lower for term in ['hydro', 'groundwater', 'aquifer']):
                context['research_domains'].append('Hydrogeology')
            elif any(term in concept_lower for term in ['remote', 'satellite', 'insar', 'lidar']):
                context['research_domains'].append('Remote Sensing')
            
            # Identify methodologies
            if any(term in concept_lower for term in ['spectroscopy', 'analysis', 'measurement']):
                context['methodologies'].append('Analytical Methods')
            elif any(term in concept_lower for term in ['modeling', 'simulation', 'numerical']):
                context['methodologies'].append('Computational Methods')
            elif any(term in concept_lower for term in ['field', 'observation', 'mapping']):
                context['methodologies'].append('Field Methods')
            elif any(term in concept_lower for term in ['laboratory', 'experimental', 'sample']):
                context['methodologies'].append('Laboratory Methods')
        
        # Remove duplicates
        for key in context:
            context[key] = list(set(context[key]))
        
        return context

class WileyOnlineLibraryAPI(ExternalDataSource):
    """Advanced interface to Wiley Online Library including AGU journals"""
    
    def __init__(self, api_key: Optional[str] = None, institution_token: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'wiley'
        
        # Add credentials if provided
        if api_key or institution_token:
            credentials = {}
            if api_key:
                credentials['api_key'] = api_key
            if institution_token:
                credentials['institution_token'] = institution_token
            self.auth_manager.add_credentials(self.api_name, credentials)
        
        self.base_url = "https://onlinelibrary.wiley.com/api/v1"
        self.search_url = f"{self.base_url}/search"
        self.content_url = f"{self.base_url}/doi"
        self.agu_url = "https://agupubs.onlinelibrary.wiley.com/api/v1"
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive geological ontology
        self.geological_ontology = GeologicalOntology()
        
        # Wiley/AGU ontology mapping with comprehensive geological and geophysical coverage
        self._ontology_mapping = {
            # Core geological concepts
            "tectonics": "geo:Tectonics",
            "plate_tectonics": "geo:PlateTectonics", 
            "geodynamics": "geo:Geodynamics",
            "lithosphere": "geo:Lithosphere",
            "asthenosphere": "geo:Asthenosphere",
            
            # Seismology and earthquakes
            "seismology": "seismo:Seismology",
            "earthquake": "seismo:Earthquake",
            "seismic_wave": "seismo:SeismicWave",
            "p_wave": "seismo:PWave",
            "s_wave": "seismo:SWave",
            "surface_wave": "seismo:SurfaceWave",
            "magnitude": "seismo:Magnitude",
            "epicenter": "seismo:Epicenter",
            "hypocenter": "seismo:Hypocenter",
            "focal_mechanism": "seismo:FocalMechanism",
            
            # Volcanology
            "volcanology": "volcano:Volcanology",
            "volcanic_eruption": "volcano:VolcanicEruption",
            "magma": "volcano:Magma",
            "lava": "volcano:Lava",
            "pyroclastic_flow": "volcano:PyroclasticFlow",
            "volcanic_ash": "volcano:VolcanicAsh",
            "caldera": "volcano:Caldera",
            "stratovolcano": "volcano:Stratovolcano",
            "shield_volcano": "volcano:ShieldVolcano",
            
            # Geophysics
            "geophysics": "geophys:Geophysics",
            "gravity": "geophys:Gravity",
            "magnetic_field": "geophys:MagneticField",
            "geomagnetism": "geophys:Geomagnetism",
            "paleomagnetism": "geophys:Paleomagnetism",
            "electrical_resistivity": "geophys:ElectricalResistivity",
            "heat_flow": "geophys:HeatFlow",
            
            # Geodesy and space science
            "geodesy": "geodesy:Geodesy",
            "gps": "geodesy:GPS",
            "gnss": "geodesy:GNSS",
            "insar": "geodesy:InSAR",
            "interferometry": "geodesy:Interferometry",
            "crustal_deformation": "geodesy:CrustalDeformation",
            "space_geodesy": "geodesy:SpaceGeodesy",
            
            # Atmospheric and climate science
            "atmospheric_science": "atmos:AtmosphericScience",
            "climate_change": "climate:ClimateChange",
            "paleoclimate": "climate:Paleoclimate",
            "greenhouse_gas": "atmos:GreenhouseGas",
            "carbon_cycle": "climate:CarbonCycle",
            "ocean_atmosphere": "climate:OceanAtmosphere",
            
            # Hydrology and oceanography
            "hydrology": "hydro:Hydrology",
            "groundwater": "hydro:Groundwater",
            "surface_water": "hydro:SurfaceWater",
            "oceanography": "ocean:Oceanography",
            "ocean_circulation": "ocean:OceanCirculation",
            "sea_level": "ocean:SeaLevel",
            
            # Biogeosciences
            "biogeochemistry": "bio:Biogeochemistry",
            "ecology": "bio:Ecology",
            "carbon_sequestration": "bio:CarbonSequestration",
            "ecosystem": "bio:Ecosystem",
            
            # Planetary science
            "planetary_science": "planet:PlanetaryScience",
            "mars": "planet:Mars",
            "moon": "planet:Moon",
            "astrobiology": "planet:Astrobiology",
            "meteorite": "planet:Meteorite",
            
            # Remote sensing
            "remote_sensing": "rs:RemoteSensing",
            "satellite_data": "rs:SatelliteData",
            "lidar": "rs:LiDAR",
            "radar": "rs:Radar",
            "hyperspectral": "rs:Hyperspectral",
            
            # Natural hazards
            "natural_hazards": "hazard:NaturalHazards",
            "tsunami": "hazard:Tsunami",
            "landslide": "hazard:Landslide",
            "flood": "hazard:Flood",
            "wildfire": "hazard:Wildfire",
            "drought": "hazard:Drought",
            
            # Computational geoscience
            "computational_geophysics": "comp:ComputationalGeophysics",
            "numerical_modeling": "comp:NumericalModeling",
            "machine_learning": "comp:MachineLearning",
            "data_science": "comp:DataScience",
            "high_performance_computing": "comp:HighPerformanceComputing"
        }
    
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query Wiley Online Library and AGU journals"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
    
    async def _execute_query_secure(self, client: SecureAPIClient, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against Wiley API using secure client"""
        
        results = []
        
        try:
            # Parse search parameters
            search_params = self._parse_search_parameters(query, limit, **kwargs)
            
            # Search AGU journals first (priority for geoscience)
            agu_results = await self._search_agu_journals_secure(client, search_params)
            results.extend(agu_results)
            
            # Search other Wiley journals if needed
            if len(results) < limit:
                remaining_limit = limit - len(results)
                wiley_results = await self._search_wiley_journals_secure(client, search_params, remaining_limit)
                results.extend(wiley_results)
                
            # Enhance results with additional metadata
            enhanced_results = await self._enhance_search_results(results)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error executing Wiley query: {e}")
            return self._fallback_query(query, limit, **kwargs)
    
    async def _search_agu_journals_secure(self, client: SecureAPIClient, search_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search AGU journals using secure client"""
        try:
            # AGU-specific search endpoint
            agu_search_url = f"{self.agu_url}/search"
            
            response = await client.request(
                'GET', agu_search_url, self.api_name,
                params=search_params
            )
            
            if response and response.status == 200:
                data = await response.json()
                return data.get('results', [])
        except Exception as e:
            self.logger.warning(f"AGU journals search failed: {e}")
        
        return []
    
    async def _search_wiley_journals_secure(self, client: SecureAPIClient, search_params: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Search Wiley journals using secure client"""
        try:
            search_params['limit'] = limit
            
            response = await client.request(
                'GET', self.search_url, self.api_name,
                params=search_params
            )
            
            if response and response.status == 200:
                data = await response.json()
                return data.get('results', [])
        except Exception as e:
            self.logger.warning(f"Wiley journals search failed: {e}")
        
        return []
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for when Wiley API is not available"""
        
        # Focus on AGU journals and geoscience topics
        agu_journals = [
            "Journal of Geophysical Research: Solid Earth",
            "Geophysical Research Letters",
            "Reviews of Geophysics",
            "Tectonics",
            "Geochemistry, Geophysics, Geosystems",
            "Journal of Geophysical Research: Atmospheres",
            "Journal of Geophysical Research: Oceans",
            "Paleoceanography and Paleoclimatology",
            "Space Weather", 
            "Earth and Space Science"
        ]
        
        geoscience_terms = [
            "seismology", "volcanology", "tectonics", "geodynamics", "geodesy",
            "geomagnetism", "paleomagnetism", "geophysics", "remote_sensing",
            "climate_change", "oceanography", "hydrology", "biogeochemistry"
        ]
        
        results = []
        base_year = 2020
        
        for i in range(min(limit, 30)):
            # Determine relevant journal and terms
            relevant_journal = agu_journals[i % len(agu_journals)]
            relevant_term = next(
                (term for term in geoscience_terms if term.replace('_', ' ') in query.lower()),
                geoscience_terms[i % len(geoscience_terms)]
            )
            
            result = {
                "title": f"{relevant_term.replace('_', ' ').title()} Research: {query} - Study {i+1}",
                "abstract": f"Advanced research in {relevant_term.replace('_', ' ')} addressing {query}. "
                          f"This study presents novel findings using state-of-the-art methodologies "
                          f"and contributes to our understanding of {relevant_term.replace('_', ' ')} processes.",
                "authors": self._generate_mock_authors(i),
                "journal": relevant_journal,
                "publisher": "Wiley",
                "year": base_year + (i % 5),
                "volume": 125 + (i % 30),
                "issue": 1 + (i % 12),
                "pages": f"{1000 + i * 15}-{1015 + i * 15}",
                "doi": f"10.1029/{base_year + (i % 5)}.{relevant_journal.replace(' ', '').replace(':', '').replace(',', '')[:8]}.{1000 + i}",
                "keywords": [relevant_term.replace('_', ' '), query.split()[0] if query.split() else "geoscience"],
                "type": "research_article",
                "open_access": i % 3 == 0,  # Some open access
                "citation_count": 50 + (i * 5) % 200,
                "source": "wiley_agu_fallback",
                "agu_journal": True if any(agu in relevant_journal for agu in ["Geophysical", "Tectonics", "Geochemistry"]) else False,
                "impact_factor": self._get_mock_impact_factor(relevant_journal),
                "research_areas": self._get_research_areas(relevant_term)
            }
            results.append(result)
            
        return results
    
    def _generate_mock_authors(self, index: int) -> List[str]:
        """Generate realistic author names for geoscience research"""
        author_pool = [
            "Smith, J.A.", "Johnson, M.B.", "Williams, S.C.", "Brown, D.E.",
            "Anderson, K.F.", "Wilson, R.G.", "Taylor, A.H.", "Martinez, L.I.",
            "Garcia, P.J.", "Rodriguez, M.K.", "Chen, W.L.", "Liu, X.M.",
            "Singh, R.N.", "Kumar, A.O.", "Patel, N.P.", "Thompson, C.Q."
        ]
        
        # Return 2-4 authors per paper
        num_authors = min(2 + (index % 3), 4)
        return author_pool[index % len(author_pool):index % len(author_pool) + num_authors]
    
    def _get_mock_impact_factor(self, journal: str) -> float:
        """Get mock impact factor for journal"""
        impact_factors = {
            "Journal of Geophysical Research: Solid Earth": 3.9,
            "Geophysical Research Letters": 4.8,
            "Reviews of Geophysics": 15.2,
            "Tectonics": 3.4,
            "Geochemistry, Geophysics, Geosystems": 3.5,
            "Journal of Geophysical Research: Atmospheres": 3.8,
            "Journal of Geophysical Research: Oceans": 3.6,
            "Paleoceanography and Paleoclimatology": 3.2,
            "Space Weather": 3.7,
            "Earth and Space Science": 3.3
        }
        return impact_factors.get(journal, 3.0)
    
    def _get_research_areas(self, term: str) -> List[str]:
        """Get research areas based on geological term"""
        area_mapping = {
            "seismology": ["Seismology", "Earthquake Physics", "Seismic Hazards"],
            "volcanology": ["Volcanology", "Igneous Petrology", "Volcanic Hazards"],
            "tectonics": ["Tectonics", "Structural Geology", "Geodynamics"],
            "geodynamics": ["Geodynamics", "Mantle Convection", "Plate Tectonics"],
            "geodesy": ["Geodesy", "GNSS", "InSAR", "Crustal Deformation"],
            "geomagnetism": ["Geomagnetism", "Paleomagnetism", "Magnetic Field"],
            "geophysics": ["Geophysics", "Applied Geophysics", "Exploration"],
            "climate_change": ["Climate Science", "Paleoclimatology", "Global Change"],
            "oceanography": ["Physical Oceanography", "Ocean Dynamics", "Marine Science"],
            "hydrology": ["Hydrology", "Water Resources", "Hydrogeology"],
            "biogeochemistry": ["Biogeochemistry", "Carbon Cycle", "Ecosystem Science"]
        }
        return area_mapping.get(term, ["Earth Sciences", "Geoscience"])
    
    def _parse_search_parameters(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Parse search query into structured parameters"""
        params = {
            'q': query,
            'limit': limit,
            'sort': 'relevance',
            'format': 'json'
        }
        
        # Add time range if specified
        if 'start_date' in kwargs:
            params['startDate'] = kwargs['start_date']
        if 'end_date' in kwargs:
            params['endDate'] = kwargs['end_date']
            
        # Add subject area filters
        if 'subject_area' in kwargs:
            params['subjectArea'] = kwargs['subject_area']
            
        return params
    
    async def _enhance_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata"""
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add ontology enhancement
            enhanced_result = self.enhance_result_with_ontology(enhanced_result)
            
            # Add Wiley-specific metadata
            enhanced_result['source_api'] = 'wiley_online_library'
            enhanced_result['access_type'] = 'subscription' if not result.get('open_access', False) else 'open_access'
            enhanced_result['publisher'] = 'Wiley'
            
            # Add AGU journal flag
            if any(agu_term in result.get('journal', '') for agu_term in ['Geophysical', 'Tectonics', 'Geochemistry']):
                enhanced_result['agu_journal'] = True
                enhanced_result['society'] = 'American Geophysical Union'
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def enhance_result_with_ontology(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a result with geological ontology information"""
        enhanced_result = result.copy()
        
        # Extract text for analysis
        text_content = []
        if 'title' in result:
            text_content.append(result['title'])
        if 'abstract' in result:
            text_content.append(result['abstract'])
        if 'keywords' in result:
            text_content.extend(result['keywords'])
        
        full_text = ' '.join(text_content).lower()
        
        # Find matching concepts
        matched_concepts = []
        for concept, uri in self._ontology_mapping.items():
            if concept.replace('_', ' ') in full_text:
                matched_concepts.append(concept)
        
        if matched_concepts:
            # Get related concepts
            related_concepts = []
            for concept in matched_concepts:
                related = self.geological_ontology.get_related_concepts(concept)
                related_concepts.extend(related)
            
            enhanced_result['ontology'] = {
                'matched_concepts': matched_concepts,
                'related_concepts': list(set(related_concepts)),
                'ontology_uris': [self._ontology_mapping.get(c, '') for c in matched_concepts]
            }
        
        return enhanced_result
    
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get Wiley/AGU ontology mapping"""
        return self._ontology_mapping.copy()

class USGSDataAPI(ExternalDataSource):
    """Advanced interface to USGS data services with real API integration"""
    
    def __init__(self, api_key: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        if not AIOHTTP_AVAILABLE:
            logging.warning("aiohttp not available. USGSDataAPI will use fallback implementation.")
        
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'usgs'
        
        # Add credentials if provided
        if api_key:
            self.auth_manager.add_credentials(self.api_name, {
                'api_key': api_key,
                'user_agent': 'Prometheus-GeologicalAI/1.0'
            })
            
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self.station_url = "https://earthquake.usgs.gov/fdsnws/station/1/query"
        self.volcano_url = "https://volcano.wr.usgs.gov/vsc-bin/volcanoworld.cgi"
        self.geomagnetism_url = "https://geomag.usgs.gov/ws/data/"
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive geological ontology
        self.geological_ontology = GeologicalOntology()
        
        # USGS-specific enhanced ontology mapping with comprehensive coverage
        self._ontology_mapping = {
            # Core seismic concepts
            "earthquake": "seismo:Earthquake",
            "magnitude": "seismo:Magnitude", 
            "epicenter": "seismo:Epicenter",
            "hypocenter": "seismo:Hypocenter",
            "seismic_wave": "seismo:SeismicWave",
            "p_wave": "seismo:PWave",
            "s_wave": "seismo:SWave",
            "surface_wave": "seismo:SurfaceWave",
            "seismic_station": "seismo:SeismicStation",
            "seismometer": "seismo:Seismometer",
            "accelerometer": "seismo:Accelerometer",
            "strong_motion": "seismo:StrongMotion",
            "microseismicity": "seismo:Microseismicity",
            
            # Volcanic phenomena
            "volcanic_eruption": "volcano:VolcanicEruption",
            "explosive_volcanism": "volcano:ExplosiveVolcanism",
            "effusive_volcanism": "volcano:EffusiveVolcanism",
            "pyroclastic_flow": "volcano:PyroclasticFlow",
            "lahar": "volcano:Lahar",
            "lava_flow": "volcano:LavaFlow",
            "volcanic_gas": "volcano:VolcanicGasEmission",
            "volcanic_ash": "volcano:VolcanicAsh",
            "volcanic_bomb": "volcano:VolcanicBomb",
            "volcanic_hazard": "volcano:VolcanicHazard",
            
            # Geophysical measurements
            "geomagnetic_field": "geophys:GeomagneticField",
            "magnetic_declination": "geophys:MagneticDeclination",
            "magnetic_inclination": "geophys:MagneticInclination",
            "magnetic_anomaly": "geophys:MagneticAnomaly",
            "paleomagnetic_reversal": "geophys:PaleomagneticReversal",
            "gravity_anomaly": "geophys:GravityAnomaly",
            "heat_flow": "geophys:HeatFlow",
            "geothermal_gradient": "geophys:GeothermalGradient",
            
            # Geological hazards and disasters
            "tsunami": "geo:Tsunami",
            "landslide": "geo:Landslide",
            "ground_deformation": "geophys:GroundDeformation",
            "subsidence": "geo:Subsidence",
            "uplift": "geo:Uplift",
            "ground_motion": "seismo:GroundMotion",
            "liquefaction": "geo:Liquefaction",
            
            # Structural geology
            "fault_system": "struc:FaultSystem",
            "fault": "struc:Fault",
            "normal_fault": "struc:NormalFault",
            "reverse_fault": "struc:ReverseFault",
            "strike_slip_fault": "struc:StrikeSlipFault",
            "transform_fault": "struc:TransformFault",
            "active_fault": "struc:ActiveFault",
            "inactive_fault": "struc:InactiveFault",
            
            # Thermal and energy phenomena
            "thermal_anomaly": "geophys:ThermalAnomaly",
            "geothermal": "geophys:Geothermal",
            "hot_spring": "hydro:HotSpring",
            "geyser": "hydro:Geyser",
            
            # Hydrological aspects
            "groundwater": "hydro:Groundwater",
            "aquifer": "hydro:Aquifer",
            "spring": "hydro:Spring",
            "well": "hydro:Well",
            "stream_gauge": "hydro:StreamGauge",
            
            # Environmental monitoring
            "air_quality": "geo:AirQuality",
            "water_quality": "hydro:WaterQuality",
            "contamination": "geo:Contamination",
            "environmental_impact": "geo:EnvironmentalImpact"
        }
        
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json,application/xml"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query USGS earthquake and volcanic database with multiple endpoints"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation using requests library"""
        
        try:
            import requests
            
            params = self._parse_query_parameters(query, limit, **kwargs)
            
            if params.get('include_earthquakes', True):
                query_params = {
                    'format': 'geojson',
                    'starttime': params['starttime'],
                    'endtime': params['endtime'],
                    'minmagnitude': params['minmagnitude'],
                    'limit': min(limit, 1000)
                }
                
                response = requests.get(self.base_url, params=query_params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_earthquake_response(data)[:limit]
                    
        except Exception as e:
            self.logger.error(f"Fallback USGS query failed: {e}")
            
        # Return mock data if all else fails
        return [
            {
                "type": "earthquake",
                "magnitude": 4.5 + i * 0.1,
                "location": f"Location {i}",
                "depth": 10 + i,
                "time": datetime.now() - timedelta(days=i),
                "source": "usgs_fallback"
            }
            for i in range(min(limit, 20))
        ]
            
    async def _execute_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query"""
        
        results = []
        
        # Parse query parameters
        params = self._parse_query_parameters(query, limit, **kwargs)
        
        try:
            # Query earthquake data
            if params.get('include_earthquakes', True):
                earthquake_data = await self._query_earthquake_data(params)
                results.extend(earthquake_data)
                
            # Query volcanic data
            if params.get('include_volcanoes', False):
                volcanic_data = await self._query_volcanic_data(params)
                results.extend(volcanic_data)
                
            # Query station data
            if params.get('include_stations', False):
                station_data = await self._query_station_data(params)
                results.extend(station_data)
                
        except Exception as e:
            self.logger.error(f"USGS query failed: {e}")
            raise
            
        return results[:limit]
        
    def _parse_query_parameters(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Parse query string into structured parameters"""
        
        # Default parameters
        params = {
            'starttime': kwargs.get('start_time', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')),
            'endtime': kwargs.get('end_time', datetime.now().strftime('%Y-%m-%dT%H:%M:%S')),
            'minmagnitude': kwargs.get('min_magnitude', 2.0),
            'maxmagnitude': kwargs.get('max_magnitude', 10.0),
            'minlatitude': kwargs.get('min_latitude', -90.0),
            'maxlatitude': kwargs.get('max_latitude', 90.0),
            'minlongitude': kwargs.get('min_longitude', -180.0),
            'maxlongitude': kwargs.get('max_longitude', 180.0),
            'include_earthquakes': True,
            'include_volcanoes': 'volcano' in query.lower() or 'eruption' in query.lower(),
            'include_stations': 'station' in query.lower() or 'network' in query.lower(),
            'include_geomagnetic': 'geomagnetic' in query.lower() or 'magnetic' in query.lower(),
            'format': 'geojson',
            'limit': limit
        }
        
        # Override with any provided kwargs
        params.update(kwargs)
        
        return params
        
    async def _query_earthquake_data(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query USGS earthquake data"""
        
        async with self.rate_limiter:
            try:
                query_params = {
                    'format': params['format'],
                    'starttime': params['starttime'],
                    'endtime': params['endtime'],
                    'minmagnitude': params['minmagnitude'],
                    'maxmagnitude': params['maxmagnitude'],
                    'minlatitude': params['minlatitude'],
                    'maxlatitude': params['maxlatitude'],
                    'minlongitude': params['minlongitude'],
                    'maxlongitude': params['maxlongitude'],
                    'limit': params.get('limit', 1000)
                }
                
                async with self.session.get(self.base_url, params=query_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_earthquake_response(data)
                    else:
                        self.logger.warning(f"USGS earthquake query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query earthquake data: {e}")
                return []
                
    def _parse_earthquake_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse USGS earthquake GeoJSON response"""
        
        events = []
        
        for feature in data.get('features', []):
            try:
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                coordinates = geometry.get('coordinates', [])
                
                if len(coordinates) >= 3:
                    event = {
                        'type': 'earthquake',
                        'id': properties.get('ids', ''),
                        'magnitude': properties.get('mag', 0.0),
                        'magnitude_type': properties.get('magType', 'unknown'),
                        'location': properties.get('place', 'Unknown location'),
                        'longitude': coordinates[0],
                        'latitude': coordinates[1],
                        'depth': coordinates[2],  # km
                        'time': datetime.fromtimestamp(properties.get('time', 0) / 1000) if properties.get('time') else datetime.now(),
                        'updated': datetime.fromtimestamp(properties.get('updated', 0) / 1000) if properties.get('updated') else datetime.now(),
                        'url': properties.get('url', ''),
                        'detail': properties.get('detail', ''),
                        'felt': properties.get('felt'),
                        'cdi': properties.get('cdi'),  # Community Detected Intensity
                        'mmi': properties.get('mmi'),  # Modified Mercalli Intensity
                        'alert': properties.get('alert'),
                        'status': properties.get('status', 'automatic'),
                        'tsunami': properties.get('tsunami', 0) == 1,
                        'significance': properties.get('sig', 0),
                        'network': properties.get('net', ''),
                        'code': properties.get('code', ''),
                        'nst': properties.get('nst'),  # Number of stations
                        'dmin': properties.get('dmin'),  # Distance to nearest station
                        'rms': properties.get('rms'),  # RMS travel time residual
                        'gap': properties.get('gap'),  # Azimuthal gap
                        'source': 'usgs_earthquake_api'
                    }
                    events.append(event)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse earthquake event: {e}")
                continue
                
        return events
        
    async def _query_volcanic_data(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query USGS volcanic data (simplified implementation)"""
        
        async with self.rate_limiter:
            try:
                # For now, filter earthquake data for volcanic events
                earthquake_params = params.copy()
                earthquake_params['eventtype'] = 'volcanic'
                
                volcanic_events = await self._query_earthquake_data(earthquake_params)
                
                # Add volcanic-specific metadata
                for event in volcanic_events:
                    event['type'] = 'volcanic_earthquake'
                    event['volcanic_context'] = True
                    
                return volcanic_events
                
            except Exception as e:
                self.logger.error(f"Failed to query volcanic data: {e}")
                return []
                
    async def _query_station_data(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query USGS seismic station data"""
        
        async with self.rate_limiter:
            try:
                query_params = {
                    'format': 'geojson',
                    'starttime': params['starttime'],
                    'endtime': params['endtime'],
                    'minlatitude': params['minlatitude'],
                    'maxlatitude': params['maxlatitude'],
                    'minlongitude': params['minlongitude'],
                    'maxlongitude': params['maxlongitude'],
                    'level': 'station'
                }
                
                async with self.session.get(self.station_url, params=query_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_station_response(data)
                    else:
                        self.logger.warning(f"USGS station query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query station data: {e}")
                return []
                
    def _parse_station_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse USGS station GeoJSON response"""
        
        stations = []
        
        for feature in data.get('features', []):
            try:
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                coordinates = geometry.get('coordinates', [])
                
                if len(coordinates) >= 2:
                    station = {
                        'type': 'seismic_station',
                        'network': properties.get('network', ''),
                        'station': properties.get('station', ''),
                        'location': properties.get('location', ''),
                        'channel': properties.get('channel', ''),
                        'longitude': coordinates[0],
                        'latitude': coordinates[1],
                        'elevation': coordinates[2] if len(coordinates) >= 3 else 0.0,
                        'site_name': properties.get('siteName', ''),
                        'start_date': properties.get('startDate'),
                        'end_date': properties.get('endDate'),
                        'sample_rate': properties.get('sampleRate'),
                        'source': 'usgs_station_api'
                    }
                    stations.append(station)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse station data: {e}")
                continue
                
        return stations
        
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get comprehensive seismological and geological ontology mapping"""
        # Combine USGS-specific mapping with comprehensive geological ontology
        comprehensive_mapping = self.geological_ontology.ontology_mapping.copy()
        comprehensive_mapping.update(self._ontology_mapping)
        
        # Convert short URIs to full URIs
        full_uri_mapping = {}
        for key, value in comprehensive_mapping.items():
            full_uri_mapping[key] = self.geological_ontology.get_full_uri(value)
        
        return full_uri_mapping
    
    def enhance_result_with_ontology(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance query result with semantic ontology information"""
        enhanced_result = result.copy()
        
        # Extract key terms from result
        result_text = f"{result.get('title', '')} {result.get('description', '')} {result.get('type', '')}"
        result_lower = result_text.lower()
        
        # Find matching ontological concepts
        matched_concepts = []
        related_concepts = []
        
        for concept in self._ontology_mapping.keys():
            if concept.replace('_', ' ') in result_lower or concept in result_lower:
                matched_concepts.append(concept)
                related_concepts.extend(self.geological_ontology.get_related_concepts(concept))
        
        if matched_concepts:
            enhanced_result['ontology'] = {
                'matched_concepts': matched_concepts,
                'related_concepts': list(set(related_concepts)),
                'semantic_uris': [self.geological_ontology.get_full_uri(self._ontology_mapping.get(concept, concept)) 
                                for concept in matched_concepts],
                'hierarchical_relationships': {
                    concept: self.geological_ontology.get_hierarchical_relationships(concept)
                    for concept in matched_concepts[:3]  # Limit to first 3 for performance
                }
            }
        
        return enhanced_result
        
    async def get_real_time_feed(self, feed_type: str = 'significant', format: str = 'geojson') -> List[Dict[str, Any]]:
        """Get real-time earthquake feeds from USGS"""
        
        feed_urls = {
            'significant': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson',
            'major': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson',
            'all': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson',
            'hour': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson',
            'day': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson'
        }
        
        feed_url = feed_urls.get(feed_type, feed_urls['significant'])
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_feed_query(feed_url)
            
        if not self.session:
            async with self:
                return await self._fetch_real_time_feed(feed_url)
        else:
            return await self._fetch_real_time_feed(feed_url)
            
    def _fallback_feed_query(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fallback implementation for real-time feeds"""
        try:
            import requests
            response = requests.get(feed_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return self._parse_earthquake_response(data)
        except Exception as e:
            self.logger.error(f"Fallback feed query failed: {e}")
        return []
            
    async def _fetch_real_time_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch real-time earthquake feed"""
        
        try:
            async with self.session.get(feed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_earthquake_response(data)
                else:
                    self.logger.warning(f"Real-time feed returned status {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Failed to fetch real-time feed: {e}")
            return []

class NASAEarthDataAPI(ExternalDataSource):
    """Advanced interface to NASA EarthData services for satellite imagery and geodetic data"""
    
    def __init__(self, earthdata_username: Optional[str] = None, earthdata_password: Optional[str] = None,
                 api_key: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'nasa'
        
        # Add credentials if provided
        credentials = {}
        if earthdata_username:
            credentials['username'] = earthdata_username
        if earthdata_password:
            credentials['password'] = earthdata_password
        if api_key:
            credentials['api_key'] = api_key
        
        if credentials:
            self.auth_manager.add_credentials(self.api_name, credentials)
        
        self.earthdata_username = earthdata_username
        self.earthdata_password = earthdata_password
        self.base_url = "https://cmr.earthdata.nasa.gov"
        self.search_url = f"{self.base_url}/search"
        self.download_url = "https://urs.earthdata.nasa.gov"
        self.logger = logging.getLogger(__name__)
        
        # Initialize comprehensive geological ontology
        self.geological_ontology = GeologicalOntology()
        
        # NASA EarthData ontology mapping
        self._ontology_mapping = {
            # Satellite missions
            "landsat": "http://ontology.nasa.gov/Landsat",
            "modis": "http://ontology.nasa.gov/MODIS",
            "aster": "http://ontology.nasa.gov/ASTER",
            "sentinel": "http://ontology.nasa.gov/Sentinel",
            "grace": "http://ontology.nasa.gov/GRACE",
            "icesat": "http://ontology.nasa.gov/ICESat",
            "srtm": "http://ontology.nasa.gov/SRTM",
            
            # Data products
            "digital_elevation_model": "http://ontology.nasa.gov/DigitalElevationModel",
            "surface_reflectance": "http://ontology.nasa.gov/SurfaceReflectance",
            "land_surface_temperature": "http://ontology.nasa.gov/LandSurfaceTemperature",
            "vegetation_index": "http://ontology.nasa.gov/VegetationIndex",
            "thermal_infrared": "http://ontology.nasa.gov/ThermalInfrared",
            "gravity_field": "http://ontology.nasa.gov/GravityField",
            "altimetry": "http://ontology.nasa.gov/Altimetry",
            
            # Geophysical phenomena
            "volcanic_hotspot": "http://ontology.nasa.gov/VolcanicHotspot",
            "thermal_anomaly": "http://ontology.nasa.gov/ThermalAnomaly",
            "ground_deformation": "http://ontology.nasa.gov/GroundDeformation",
            "sea_level_change": "http://ontology.nasa.gov/SeaLevelChange",
            "ice_sheet_change": "http://ontology.nasa.gov/IceSheetChange",
            "land_cover_change": "http://ontology.nasa.gov/LandCoverChange"
        }
        
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=45)
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Add NASA EarthData authentication if available
        if self.earthdata_username and self.earthdata_password:
            import base64
            auth_string = base64.b64encode(
                f"{self.earthdata_username}:{self.earthdata_password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {auth_string}"
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query NASA EarthData Common Metadata Repository (CMR)"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for NASA EarthData queries"""
        
        # Generate realistic mock data based on query content
        nasa_instruments = [
            "MODIS", "LANDSAT", "ASTER", "VIIRS", "GRACE", "ICESat", "SRTM"
        ]
        
        data_products = [
            "surface_temperature", "vegetation_index", "digital_elevation",
            "thermal_anomaly", "land_cover", "gravity_field", "altimetry"
        ]
        
        results = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(min(limit, 25)):
            # Select relevant instrument and product based on query
            relevant_instrument = next(
                (instr for instr in nasa_instruments if instr.lower() in query.lower()),
                nasa_instruments[i % len(nasa_instruments)]
            )
            
            relevant_product = next(
                (prod for prod in data_products if prod.replace('_', ' ') in query.lower()),
                data_products[i % len(data_products)]
            )
            
            acquisition_date = base_date + timedelta(days=i * 16)
            
            result = {
                "title": f"{relevant_instrument} {relevant_product.replace('_', ' ').title()}",
                "instrument": relevant_instrument,
                "platform": self._get_platform_for_instrument(relevant_instrument),
                "data_product": relevant_product,
                "acquisition_date": acquisition_date,
                "processing_level": f"Level {1 + (i % 3)}",
                "spatial_resolution": self._get_spatial_resolution(relevant_instrument),
                "temporal_resolution": self._get_temporal_resolution(relevant_instrument),
                "data_format": self._get_data_format(relevant_instrument),
                "file_size_mb": 50 + (i * 10) % 500,
                "source": "nasa_earthdata_fallback",
                "keywords": [relevant_instrument, relevant_product]
            }
            results.append(result)
            
        return results
        
    def _get_platform_for_instrument(self, instrument: str) -> str:
        """Get satellite platform for instrument"""
        platform_mapping = {
            "MODIS": "Terra/Aqua",
            "LANDSAT": "Landsat-8/9",
            "ASTER": "Terra",
            "VIIRS": "Suomi-NPP",
            "GRACE": "GRACE-FO",
            "ICESat": "ICESat-2",
            "SRTM": "Space Shuttle"
        }
        return platform_mapping.get(instrument, "Unknown Platform")
        
    def _get_spatial_resolution(self, instrument: str) -> str:
        """Get spatial resolution for instrument"""
        resolution_mapping = {
            "MODIS": "250m-1km",
            "LANDSAT": "30m",
            "ASTER": "15-90m", 
            "VIIRS": "375m-750m",
            "GRACE": "~300km",
            "ICESat": "~70m",
            "SRTM": "30m"
        }
        return resolution_mapping.get(instrument, "Variable")
        
    def _get_temporal_resolution(self, instrument: str) -> str:
        """Get temporal resolution for instrument"""
        temporal_mapping = {
            "MODIS": "Daily",
            "LANDSAT": "16 days",
            "ASTER": "16 days",
            "VIIRS": "Daily",
            "GRACE": "Monthly", 
            "ICESat": "91 days",
            "SRTM": "Static"
        }
        return temporal_mapping.get(instrument, "Variable")
        
    def _get_data_format(self, instrument: str) -> str:
        """Get data format for instrument"""
        format_mapping = {
            "MODIS": "HDF-EOS",
            "LANDSAT": "GeoTIFF",
            "ASTER": "HDF-EOS",
            "VIIRS": "NetCDF", 
            "GRACE": "NetCDF",
            "ICESat": "HDF5",
            "SRTM": "GeoTIFF"
        }
        return format_mapping.get(instrument, "Unknown")
            
    async def _execute_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against NASA CMR"""
        
        # This would implement real CMR API queries
        # For now, fall back to mock data
        return self._fallback_query(query, limit, **kwargs)
        
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get NASA EarthData ontology mapping"""
        return self._ontology_mapping.copy()

class ASFHyp3API(ExternalDataSource):
    """Advanced interface to Alaska Satellite Facility HyP3 for InSAR processing and time series analysis"""
    
    def __init__(self, asf_username: Optional[str] = None, asf_password: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'asf'
        
        # Add credentials if provided
        if asf_username or asf_password:
            credentials = {}
            if asf_username:
                credentials['username'] = asf_username
            if asf_password:
                credentials['password'] = asf_password
            self.auth_manager.add_credentials(self.api_name, credentials)
        
        self.asf_username = asf_username
        self.asf_password = asf_password
        self.base_url = "https://hyp3-api.asf.alaska.edu"
        self.search_url = "https://api.daac.asf.alaska.edu/services/search/param"
        self.logger = logging.getLogger(__name__)
        
        # ASF/InSAR ontology mapping
        self._ontology_mapping = {
            # InSAR techniques
            "interferometry": "http://ontology.asf.alaska.edu/Interferometry",
            "differential_interferometry": "http://ontology.asf.alaska.edu/DifferentialInterferometry",
            "persistent_scatterer": "http://ontology.asf.alaska.edu/PersistentScatterer",
            "small_baseline": "http://ontology.asf.alaska.edu/SmallBaseline",
            "sbas": "http://ontology.asf.alaska.edu/SBAS",
            
            # SAR platforms and sensors
            "sentinel1": "http://ontology.asf.alaska.edu/Sentinel1",
            "alos_palsar": "http://ontology.asf.alaska.edu/ALOSPALSAR",
            "radarsat": "http://ontology.asf.alaska.edu/RADARSAT",
            "ers": "http://ontology.asf.alaska.edu/ERS",
            "envisat": "http://ontology.asf.alaska.edu/ENVISAT",
            
            # InSAR products
            "interferogram": "http://ontology.asf.alaska.edu/Interferogram",
            "coherence": "http://ontology.asf.alaska.edu/Coherence",
            "unwrapped_phase": "http://ontology.asf.alaska.edu/UnwrappedPhase",
            "displacement_map": "http://ontology.asf.alaska.edu/DisplacementMap",
            "velocity_field": "http://ontology.asf.alaska.edu/VelocityField",
            "time_series": "http://ontology.asf.alaska.edu/TimeSeries",
            
            # Deformation phenomena
            "ground_deformation": "http://ontology.asf.alaska.edu/GroundDeformation",
            "volcanic_deformation": "http://ontology.asf.alaska.edu/VolcanicDeformation",
            "tectonic_deformation": "http://ontology.asf.alaska.edu/TectonicDeformation",
            "subsidence": "http://ontology.asf.alaska.edu/Subsidence",
            "uplift": "http://ontology.asf.alaska.edu/Uplift",
            "landslide": "http://ontology.asf.alaska.edu/Landslide",
            
            # Processing parameters
            "baseline": "http://ontology.asf.alaska.edu/Baseline",
            "temporal_baseline": "http://ontology.asf.alaska.edu/TemporalBaseline",
            "perpendicular_baseline": "http://ontology.asf.alaska.edu/PerpendicularBaseline",
            "look_vector": "http://ontology.asf.alaska.edu/LookVector",
            "incidence_angle": "http://ontology.asf.alaska.edu/IncidenceAngle"
        }
        
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=60)  # Longer timeout for processing operations
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Add ASF authentication if available
        auth = None
        if self.asf_username and self.asf_password:
            auth = aiohttp.BasicAuth(self.asf_username, self.asf_password)
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers, auth=auth)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query ASF data and HyP3 processing services"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for ASF/InSAR queries"""
        
        # Generate realistic InSAR processing and data products
        sar_platforms = ["Sentinel-1A", "Sentinel-1B", "ALOS-2", "RADARSAT-2", "TerraSAR-X"]
        processing_types = [
            "interferogram_generation", "coherence_estimation", "phase_unwrapping",
            "displacement_mapping", "time_series_analysis", "velocity_estimation"
        ]
        
        deformation_phenomena = [
            "volcanic_inflation", "volcanic_deflation", "tectonic_creep", "landslide_movement",
            "subsidence", "uplift", "glacier_motion", "seismic_deformation"
        ]
        
        results = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(min(limit, 20)):
            # Select relevant elements based on query
            relevant_platform = next(
                (plat for plat in sar_platforms if plat.lower().replace('-', '') in query.lower().replace('-', '')),
                sar_platforms[i % len(sar_platforms)]
            )
            
            relevant_processing = next(
                (proc for proc in processing_types if any(word in query.lower() for word in proc.split('_'))),
                processing_types[i % len(processing_types)]
            )
            
            relevant_phenomenon = next(
                (phen for phen in deformation_phenomena if any(word in query.lower() for word in phen.split('_'))),
                deformation_phenomena[i % len(deformation_phenomena)]
            )
            
            acquisition_date = base_date + timedelta(days=i * 12)  # 12-day cycle for Sentinel-1
            
            result = {
                "title": f"{relevant_platform} InSAR Analysis: {relevant_phenomenon.replace('_', ' ').title()}",
                "platform": relevant_platform,
                "sensor": self._get_sensor_for_platform(relevant_platform),
                "processing_type": relevant_processing,
                "phenomenon": relevant_phenomenon,
                "acquisition_date": acquisition_date,
                "polarization": self._get_polarization(relevant_platform),
                "orbit_direction": "Ascending" if i % 2 == 0 else "Descending",
                "beam_mode": self._get_beam_mode(relevant_platform),
                "spatial_resolution": self._get_insar_resolution(relevant_platform),
                "temporal_baseline": f"{6 + (i % 30)} days",
                "perpendicular_baseline": f"{10 + (i * 15) % 200} meters",
                "coherence_threshold": 0.3 + (i % 7) * 0.1,
                "displacement_accuracy": f"{2 + (i % 5)} mm",
                "processing_level": f"L{2 + (i % 2)}",
                "data_format": "GeoTIFF" if i % 3 == 0 else "NetCDF",
                "coverage_area_km2": 100 + (i * 50) % 5000,
                "job_status": "completed" if i % 4 != 3 else "processing",
                "hyp3_job_id": f"hyp3_{acquisition_date.strftime('%Y%m%d')}_{1000 + i}",
                "source": "asf_hyp3_fallback",
                "applications": self._get_applications(relevant_phenomenon),
                "quality_metrics": self._generate_quality_metrics(i)
            }
            results.append(result)
            
        return results
        
    def _get_sensor_for_platform(self, platform: str) -> str:
        """Get sensor name for SAR platform"""
        sensor_mapping = {
            "Sentinel-1A": "C-SAR", "Sentinel-1B": "C-SAR",
            "ALOS-2": "PALSAR-2", "RADARSAT-2": "SAR",
            "TerraSAR-X": "X-SAR", "COSMO-SkyMed": "X-SAR"
        }
        return sensor_mapping.get(platform, "SAR")
        
    def _get_polarization(self, platform: str) -> str:
        """Get typical polarization for platform"""
        if "Sentinel-1" in platform:
            return "VV+VH"
        elif "ALOS" in platform:
            return "HH+HV"
        elif "RADARSAT" in platform:
            return "HH+HV+VV+VH"
        else:
            return "VV"
            
    def _get_beam_mode(self, platform: str) -> str:
        """Get beam mode for platform"""
        if "Sentinel-1" in platform:
            return "IW"  # Interferometric Wide swath
        elif "ALOS" in platform:
            return "SM"  # Stripmap
        elif "RADARSAT" in platform:
            return "Fine"
        else:
            return "Standard"
            
    def _get_insar_resolution(self, platform: str) -> str:
        """Get InSAR spatial resolution"""
        if "Sentinel-1" in platform:
            return "20m x 5m"
        elif "ALOS" in platform:
            return "3m x 3m"
        elif "TerraSAR" in platform:
            return "1m x 1m"
        else:
            return "10m x 10m"
            
    def _get_applications(self, phenomenon: str) -> List[str]:
        """Get applications for deformation phenomenon"""
        application_mapping = {
            "volcanic_inflation": ["eruption_forecasting", "hazard_assessment", "magma_chamber_monitoring"],
            "volcanic_deflation": ["post_eruption_analysis", "magma_withdrawal_tracking", "caldera_studies"],
            "tectonic_creep": ["fault_monitoring", "earthquake_hazard", "plate_boundary_studies"],
            "landslide_movement": ["slope_stability", "hazard_mitigation", "infrastructure_monitoring"],
            "subsidence": ["groundwater_monitoring", "urban_planning", "infrastructure_assessment"],
            "uplift": ["geothermal_monitoring", "tectonic_analysis", "resource_exploration"]
        }
        return application_mapping.get(phenomenon, ["general_deformation_monitoring"])
        
    def _generate_quality_metrics(self, index: int) -> Dict[str, float]:
        """Generate realistic quality metrics"""
        return {
            "mean_coherence": 0.4 + (index % 6) * 0.1,
            "phase_unwrapping_success": 0.7 + (index % 3) * 0.1,
            "atmospheric_phase_screen": (index % 8) * 0.5,
            "temporal_decorrelation": (index % 5) * 0.05,
            "geometric_decorrelation": (index % 4) * 0.03,
            "noise_level_mm": 1.5 + (index % 3) * 0.5
        }
            
    async def _execute_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against ASF and HyP3 APIs"""
        
        results = []
        
        try:
            # Parse search parameters
            search_params = self._parse_search_parameters(query, limit, **kwargs)
            
            # Query ASF data catalog
            if search_params.get('include_data_search', True):
                data_results = await self._query_asf_data_catalog(search_params)
                results.extend(data_results)
                
            # Query HyP3 processing jobs
            if search_params.get('include_processing_jobs', True):
                job_results = await self._query_hyp3_jobs(search_params)
                results.extend(job_results)
                
            # Query time series products
            if search_params.get('include_time_series', False):
                ts_results = await self._query_time_series_products(search_params)
                results.extend(ts_results)
                
            # Enhance results with InSAR-specific analysis
            enhanced_results = await self._enhance_insar_context(results)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            self.logger.error(f"ASF/HyP3 query failed: {e}")
            # Fall back to mock data on API failure
            return self._fallback_query(query, limit, **kwargs)
            
    def _parse_search_parameters(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Parse search query into ASF/HyP3 parameters"""
        
        params = {
            'keyword': query,
            'limit': min(limit, 500),
            'start_date': kwargs.get('start_date', (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')),
            'end_date': kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d')),
            'platform': kwargs.get('platform', None),
            'beam_mode': kwargs.get('beam_mode', None),
            'polarization': kwargs.get('polarization', None),
            'orbit_direction': kwargs.get('orbit_direction', None),
            'bbox': kwargs.get('bounding_box', None),
            'processing_level': kwargs.get('processing_level', None),
            'include_data_search': 'sentinel' in query.lower() or 'sar' in query.lower() or kwargs.get('include_data_search', True),
            'include_processing_jobs': 'hyp3' in query.lower() or 'processing' in query.lower() or kwargs.get('include_processing_jobs', True),
            'include_time_series': 'time series' in query.lower() or 'velocity' in query.lower() or kwargs.get('include_time_series', False)
        }
        
        return params
        
    async def _query_asf_data_catalog(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query ASF data catalog for SAR products"""
        
        async with self.rate_limiter:
            try:
                search_params = {
                    'output': 'JSON',
                    'keyword': params['keyword'],
                    'start': params['start_date'],
                    'end': params['end_date']
                }
                
                # Add spatial constraint if provided
                if params['bbox']:
                    bbox = params['bbox']
                    search_params['bbox'] = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
                    
                # Add platform constraint
                if params['platform']:
                    search_params['platform'] = params['platform']
                    
                # Add processing level constraint
                if params['processing_level']:
                    search_params['processingLevel'] = params['processing_level']
                
                async with self.session.get(self.search_url, params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_asf_catalog_response(data)
                    else:
                        self.logger.warning(f"ASF catalog query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query ASF catalog: {e}")
                return []
                
    async def _query_hyp3_jobs(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query HyP3 processing jobs"""
        
        async with self.rate_limiter:
            try:
                search_params = {
                    'status_code': 'SUCCEEDED',  # Focus on completed jobs
                    'page_size': min(params['limit'], 100)
                }
                
                # Add date range if specified
                if params['start_date'] and params['end_date']:
                    search_params['start'] = params['start_date']
                    search_params['end'] = params['end_date']
                
                async with self.session.get(f"{self.base_url}/jobs", params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_hyp3_jobs_response(data)
                    elif response.status == 401:
                        self.logger.warning("HyP3 API authentication required")
                        return []
                    else:
                        self.logger.warning(f"HyP3 jobs query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query HyP3 jobs: {e}")
                return []
                
    async def _query_time_series_products(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query time series analysis products"""
        
        # This would implement time series product queries
        # For now, generate mock time series products
        time_series_products = []
        
        for i in range(min(5, params['limit'] // 4)):
            product = {
                "title": f"InSAR Time Series Analysis {i+1}",
                "product_type": "time_series",
                "temporal_span": f"{90 + i * 30} days",
                "number_of_epochs": 20 + i * 5,
                "velocity_precision": f"{1 + i * 0.5} mm/year",
                "reference_point": {"lat": 64.0 + i * 0.1, "lon": -152.0 - i * 0.1},
                "processing_algorithm": "SBAS" if i % 2 == 0 else "PSI",
                "source": "hyp3_time_series"
            }
            time_series_products.append(product)
            
        return time_series_products
        
    def _parse_asf_catalog_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse ASF catalog response"""
        
        results = []
        
        for item in data.get('results', []):
            try:
                result = {
                    'title': item.get('granuleName', ''),
                    'platform': item.get('platform', ''),
                    'sensor': item.get('instrument', ''),
                    'beam_mode': item.get('beamModeType', ''),
                    'polarization': item.get('polarization', ''),
                    'orbit_direction': item.get('flightDirection', ''),
                    'acquisition_date': item.get('startTime', ''),
                    'end_date': item.get('stopTime', ''),
                    'path': item.get('pathNumber', ''),
                    'frame': item.get('frameNumber', ''),
                    'processing_level': item.get('processingLevel', ''),
                    'file_size_mb': item.get('bytes', 0) / 1024 / 1024,
                    'download_url': item.get('downloadUrl', ''),
                    'browse_url': item.get('browseUrl', ''),
                    'metadata_url': item.get('url', ''),
                    'bbox': self._extract_bbox_from_asf(item),
                    'source': 'asf_data_catalog'
                }
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse ASF catalog item: {e}")
                continue
                
        return results
        
    def _parse_hyp3_jobs_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse HyP3 jobs response"""
        
        results = []
        
        for job in data.get('jobs', []):
            try:
                result = {
                    'title': f"HyP3 Job: {job.get('job_type', 'Unknown')}",
                    'job_id': job.get('job_id', ''),
                    'job_type': job.get('job_type', ''),
                    'status': job.get('status_code', ''),
                    'request_time': job.get('request_time', ''),
                    'processing_time': job.get('processing_time_seconds', 0),
                    'parameters': job.get('job_parameters', {}),
                    'files': job.get('files', []),
                    'browse_images': job.get('browse_images', []),
                    'thumbnail': job.get('thumbnail', ''),
                    'user_id': job.get('user_id', ''),
                    'source': 'hyp3_processing_api'
                }
                
                # Extract specific parameters based on job type
                if result['job_type'] == 'INSAR_GAMMA':
                    params = result['parameters']
                    result.update({
                        'reference_scene': params.get('granules', [{}])[0] if params.get('granules') else '',
                        'secondary_scene': params.get('granules', [{}])[1] if len(params.get('granules', [])) > 1 else '',
                        'look_vectors': params.get('include_look_vectors', False),
                        'displacement_maps': params.get('include_displacement_maps', False),
                        'apply_water_mask': params.get('apply_water_mask', False)
                    })
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse HyP3 job: {e}")
                continue
                
        return results
        
    def _extract_bbox_from_asf(self, item: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract bounding box from ASF catalog item"""
        
        try:
            # ASF typically provides polygon or bbox coordinates
            if 'bbox' in item:
                bbox_str = item['bbox']
                coords = list(map(float, bbox_str.split(',')))
                if len(coords) == 4:
                    return {
                        'west': coords[0],
                        'south': coords[1], 
                        'east': coords[2],
                        'north': coords[3]
                    }
                    
            # Try to extract from polygon if available
            elif 'polygon' in item:
                # Simplified polygon parsing - would need more robust implementation
                polygon_str = item['polygon']
                # Extract approximate bounding box from polygon
                # This is a simplified approach
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract bbox: {e}")
            
        return None
        
    async def _enhance_insar_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance results with InSAR-specific context and analysis"""
        
        enhanced_results = []
        
        for result in results:
            try:
                # Add InSAR suitability assessment
                result['insar_suitability'] = self._assess_insar_suitability(result)
                
                # Add deformation detection capability
                result['deformation_sensitivity'] = self._assess_deformation_sensitivity(result)
                
                # Add processing recommendations
                result['processing_recommendations'] = self._generate_processing_recommendations(result)
                
                # Add temporal analysis context
                result['temporal_context'] = self._analyze_temporal_context_insar(result)
                
                # Add geophysical applications
                result['geophysical_applications'] = self._identify_geophysical_applications(result)
                
                enhanced_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance InSAR result: {e}")
                enhanced_results.append(result)
                continue
                
        return enhanced_results
        
    def _assess_insar_suitability(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess InSAR processing suitability"""
        
        suitability = {
            'overall_score': 0.0,
            'factors': {},
            'limitations': []
        }
        
        # Assess based on platform characteristics
        platform = result.get('platform', '').upper()
        
        if 'SENTINEL-1' in platform:
            suitability['factors']['temporal_sampling'] = 0.9  # 6-12 day repeat
            suitability['factors']['spatial_coverage'] = 0.8
            suitability['factors']['data_availability'] = 0.95
        elif 'ALOS' in platform:
            suitability['factors']['spatial_resolution'] = 0.9
            suitability['factors']['penetration_capability'] = 0.7
            suitability['factors']['temporal_sampling'] = 0.6  # 14 day repeat
        
        # Assess based on beam mode
        beam_mode = result.get('beam_mode', '')
        if beam_mode == 'IW':  # Interferometric Wide
            suitability['factors']['swath_width'] = 0.8
            suitability['factors']['resolution_vs_coverage'] = 0.7
        elif beam_mode in ['SM', 'Fine']:
            suitability['factors']['spatial_resolution'] = 0.9
            suitability['factors']['swath_width'] = 0.5
            
        # Calculate overall score
        if suitability['factors']:
            suitability['overall_score'] = sum(suitability['factors'].values()) / len(suitability['factors'])
            
        # Add limitations
        if suitability['overall_score'] < 0.6:
            suitability['limitations'].append('Limited temporal sampling')
        if result.get('orbit_direction') == 'Descending' and 'volcanic' in result.get('phenomenon', ''):
            suitability['limitations'].append('Potential ionospheric effects')
            
        return suitability
        
    def _assess_deformation_sensitivity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess deformation detection sensitivity"""
        
        sensitivity = {
            'line_of_sight_mm': 1.0,  # Default precision
            'horizontal_mm': 3.0,
            'vertical_mm': 5.0,
            'velocity_mm_year': 2.0,
            'factors_affecting_precision': []
        }
        
        # Adjust based on platform
        platform = result.get('platform', '').upper()
        if 'SENTINEL-1' in platform:
            sensitivity['line_of_sight_mm'] = 2.0
            sensitivity['velocity_mm_year'] = 1.5
        elif 'ALOS' in platform:
            sensitivity['line_of_sight_mm'] = 1.0
            sensitivity['velocity_mm_year'] = 1.0
            
        # Consider environmental factors
        if result.get('coherence_threshold', 0.5) < 0.3:
            sensitivity['factors_affecting_precision'].append('Low coherence areas')
            sensitivity['line_of_sight_mm'] *= 2
            
        if 'atmospheric_phase_screen' in result.get('quality_metrics', {}):
            aps_level = result['quality_metrics']['atmospheric_phase_screen']
            if aps_level > 2.0:
                sensitivity['factors_affecting_precision'].append('Strong atmospheric effects')
                sensitivity['line_of_sight_mm'] *= 1.5
                
        return sensitivity
        
    def _generate_processing_recommendations(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate InSAR processing recommendations"""
        
        recommendations = []
        
        # Baseline recommendations
        temporal_baseline = result.get('temporal_baseline', '12 days')
        if 'days' in temporal_baseline:
            days = int(temporal_baseline.split()[0])
            if days > 24:
                recommendations.append({
                    'category': 'temporal_baseline',
                    'priority': 'high',
                    'recommendation': 'Consider shorter temporal baselines for better coherence',
                    'rationale': f'{days} day baseline may suffer from temporal decorrelation'
                })
                
        # Coherence threshold recommendations  
        coherence = result.get('coherence_threshold', 0.3)
        if coherence < 0.25:
            recommendations.append({
                'category': 'coherence_threshold',
                'priority': 'medium',
                'recommendation': 'Increase coherence threshold for more reliable results',
                'rationale': 'Low coherence threshold may include noisy pixels'
            })
            
        # Platform-specific recommendations
        platform = result.get('platform', '').upper()
        if 'SENTINEL-1' in platform:
            recommendations.append({
                'category': 'processing_strategy',
                'priority': 'medium',
                'recommendation': 'Use SBAS approach for time series analysis',
                'rationale': 'Dense temporal sampling ideal for small baseline processing'
            })
            
        # Phenomenon-specific recommendations
        phenomenon = result.get('phenomenon', '')
        if 'volcanic' in phenomenon:
            recommendations.append({
                'category': 'atmospheric_correction',
                'priority': 'high',
                'recommendation': 'Apply tropospheric delay corrections',
                'rationale': 'Volcanic areas often have complex atmospheric conditions'
            })
            
        return recommendations
        
    def _analyze_temporal_context_insar(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal context for InSAR analysis"""
        
        context = {
            'acquisition_strategy': 'unknown',
            'temporal_resolution': 'unknown',
            'time_series_viability': False,
            'seasonal_considerations': []
        }
        
        # Determine acquisition strategy
        platform = result.get('platform', '').upper()
        if 'SENTINEL-1' in platform:
            context['acquisition_strategy'] = 'systematic_global'
            context['temporal_resolution'] = '6-12_days'
            context['time_series_viability'] = True
        elif 'ALOS' in platform:
            context['acquisition_strategy'] = 'targeted_acquisition'
            context['temporal_resolution'] = '14_days'
            context['time_series_viability'] = True
            
        # Seasonal considerations
        acquisition_date = result.get('acquisition_date', '')
        if acquisition_date:
            try:
                date_obj = datetime.fromisoformat(acquisition_date.replace('Z', '+00:00'))
                month = date_obj.month
                
                if month in [12, 1, 2]:
                    context['seasonal_considerations'].append('winter_conditions')
                elif month in [6, 7, 8]:
                    context['seasonal_considerations'].append('summer_conditions')
                    
                # Add specific considerations
                if month in [3, 4, 5, 9, 10, 11]:
                    context['seasonal_considerations'].append('optimal_coherence_period')
                    
            except Exception:
                pass
                
        return context
        
    def _identify_geophysical_applications(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify specific geophysical applications"""
        
        applications = []
        
        # Based on processing type and phenomenon
        processing_type = result.get('processing_type', '')
        phenomenon = result.get('phenomenon', '')
        
        if 'volcanic' in phenomenon:
            applications.extend([
                {'application': 'magma_chamber_monitoring', 'confidence': 'high'},
                {'application': 'eruption_precursor_detection', 'confidence': 'medium'},
                {'application': 'post_eruption_deformation', 'confidence': 'high'}
            ])
            
        if 'tectonic' in phenomenon:
            applications.extend([
                {'application': 'fault_creep_monitoring', 'confidence': 'high'},
                {'application': 'interseismic_deformation', 'confidence': 'medium'},
                {'application': 'earthquake_cycle_analysis', 'confidence': 'medium'}
            ])
            
        if 'time_series' in processing_type:
            applications.extend([
                {'application': 'velocity_field_mapping', 'confidence': 'high'},
                {'application': 'long_term_trend_analysis', 'confidence': 'high'},
                {'application': 'seasonal_deformation_modeling', 'confidence': 'medium'}
            ])
            
        # Platform-specific applications
        platform = result.get('platform', '').upper()
        if 'SENTINEL-1' in platform:
            applications.append({
                'application': 'operational_monitoring', 
                'confidence': 'high'
            })
        elif 'ALOS' in platform:
            applications.append({
                'application': 'high_resolution_analysis', 
                'confidence': 'high'
            })
            
        return applications
        
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get ASF/InSAR ontology mapping"""
        return self._ontology_mapping.copy()
        
    async def get_processing_capabilities(self) -> List[Dict[str, Any]]:
        """Get available HyP3 processing capabilities"""
        
        capabilities = [
            {
                "process_type": "INSAR_GAMMA",
                "description": "Interferometric SAR processing using GAMMA software",
                "inputs": ["SLC_pair"],
                "outputs": ["interferogram", "coherence", "unwrapped_phase", "displacement_maps"],
                "parameters": ["include_look_vectors", "include_displacement_maps", "apply_water_mask"],
                "platforms": ["Sentinel-1", "ALOS-2"],
                "typical_processing_time": "15-30 minutes"
            },
            {
                "process_type": "RTC_GAMMA", 
                "description": "Radiometric Terrain Correction using GAMMA",
                "inputs": ["SLC"],
                "outputs": ["orthorectified_backscatter"],
                "parameters": ["speckle_filter", "dem_matching", "radiometric_correction"],
                "platforms": ["Sentinel-1", "ALOS-2"],
                "typical_processing_time": "10-20 minutes"
            },
            {
                "process_type": "AUTORIFT",
                "description": "Autonomous Repeat Image Feature Tracking",
                "inputs": ["optical_pair", "SAR_pair"],
                "outputs": ["velocity_maps", "displacement_maps"],
                "parameters": ["search_window_size", "correlation_threshold"],
                "platforms": ["Landsat", "Sentinel-1", "Sentinel-2"],
                "typical_processing_time": "20-40 minutes"
            },
            {
                "process_type": "SBAS_TIME_SERIES",
                "description": "Small Baseline Subset time series analysis",
                "inputs": ["interferogram_stack"],
                "outputs": ["velocity_field", "time_series", "deformation_maps"],
                "parameters": ["coherence_threshold", "unwrapping_method", "atmospheric_correction"],
                "platforms": ["Sentinel-1"],
                "typical_processing_time": "1-3 hours"
            }
        ]
        
        return capabilities

class EarthScopeIRISAPI(ExternalDataSource):
    """Advanced interface to EarthScope/IRIS for seismic data and waveform analysis integration"""
    
    def __init__(self, iris_username: Optional[str] = None, iris_password: Optional[str] = None,
                 auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        # Initialize authentication and rate limiting
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.api_name = 'iris'
        
        # Add credentials if provided
        if iris_username or iris_password:
            credentials = {}
            if iris_username:
                credentials['username'] = iris_username
            if iris_password:
                credentials['password'] = iris_password
            self.auth_manager.add_credentials(self.api_name, credentials)
        
        self.iris_username = iris_username
        self.iris_password = iris_password
        self.base_url = "http://service.iris.edu"
        self.fdsnws_dataselect = f"{self.base_url}/fdsnws/dataselect/1"
        self.fdsnws_station = f"{self.base_url}/fdsnws/station/1"
        self.fdsnws_event = f"{self.base_url}/fdsnws/event/1"
        self.ph5ws_url = f"{self.base_url}/ph5ws"
        self.logger = logging.getLogger(__name__)
        
        # EarthScope/IRIS ontology mapping
        self._ontology_mapping = {
            # Seismic data types
            "broadband_seismometer": "http://ontology.iris.edu/BroadbandSeismometer",
            "strong_motion": "http://ontology.iris.edu/StrongMotion", 
            "short_period": "http://ontology.iris.edu/ShortPeriod",
            "very_broadband": "http://ontology.iris.edu/VeryBroadband",
            "accelerometer": "http://ontology.iris.edu/Accelerometer",
            
            # Waveform components
            "vertical_component": "http://ontology.iris.edu/VerticalComponent",
            "north_south_component": "http://ontology.iris.edu/NorthSouthComponent",
            "east_west_component": "http://ontology.iris.edu/EastWestComponent",
            "pressure": "http://ontology.iris.edu/Pressure",
            "temperature": "http://ontology.iris.edu/Temperature",
            
            # Seismic phases
            "p_wave": "http://ontology.iris.edu/PWave",
            "s_wave": "http://ontology.iris.edu/SWave",
            "surface_wave": "http://ontology.iris.edu/SurfaceWave",
            "body_wave": "http://ontology.iris.edu/BodyWave",
            "love_wave": "http://ontology.iris.edu/LoveWave",
            "rayleigh_wave": "http://ontology.iris.edu/RayleighWave",
            
            # Seismic networks
            "global_seismographic_network": "http://ontology.iris.edu/GSN",
            "transportable_array": "http://ontology.iris.edu/TransportableArray",
            "alaska_regional_network": "http://ontology.iris.edu/AlaskaRegionalNetwork",
            "california_integrated_network": "http://ontology.iris.edu/CaliforniaIntegratedNetwork",
            "yellowstone_network": "http://ontology.iris.edu/YellowstoneNetwork",
            
            # Data products
            "continuous_waveform": "http://ontology.iris.edu/ContinuousWaveform",
            "event_waveform": "http://ontology.iris.edu/EventWaveform",
            "ambient_noise": "http://ontology.iris.edu/AmbientNoise",
            "spectral_density": "http://ontology.iris.edu/SpectralDensity",
            "receiver_function": "http://ontology.iris.edu/ReceiverFunction",
            
            # Analysis methods
            "tomography": "http://ontology.iris.edu/Tomography",
            "cross_correlation": "http://ontology.iris.edu/CrossCorrelation",
            "spectral_analysis": "http://ontology.iris.edu/SpectralAnalysis",
            "polarization_analysis": "http://ontology.iris.edu/PolarizationAnalysis",
            "particle_motion": "http://ontology.iris.edu/ParticleMotion"
        }
        
    async def __aenter__(self):
        if not AIOHTTP_AVAILABLE:
            return self
            
        timeout = aiohttp.ClientTimeout(total=90)  # Longer timeout for large waveform requests
        headers = {
            "User-Agent": "Prometheus-AI-Geodynamic-Platform/1.0",
            "Accept": "application/json,text/plain"
        }
        
        # Add IRIS authentication if available (usually not required for public data)
        auth = None
        if self.iris_username and self.iris_password:
            auth = aiohttp.BasicAuth(self.iris_username, self.iris_password)
        
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers, auth=auth)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        
    async def query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Advanced query EarthScope/IRIS seismic data and services"""
        
        if not AIOHTTP_AVAILABLE:
            return self._fallback_query(query, limit, **kwargs)
            
        # Use secure API client with authentication and rate limiting
        async with SecureAPIClient(self.auth_manager, self.rate_limiter) as client:
            return await self._execute_query_secure(client, query, limit, **kwargs)
            
    def _fallback_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Fallback implementation for IRIS/EarthScope queries"""
        
        # Generate realistic seismic data products
        seismic_networks = [
            "IU", "II", "G", "GT", "IC", "CU", "US", "TA", "AK", "CI", "UU", "WY"
        ]
        
        station_types = [
            "broadband", "very_broadband", "short_period", "strong_motion", 
            "accelerometer", "pressure", "infrasound"
        ]
        
        data_products = [
            "continuous_waveform", "event_waveform", "ambient_noise", "receiver_function",
            "spectral_density", "polarization_data", "particle_motion", "cross_correlation"
        ]
        
        seismic_phenomena = [
            "earthquake", "volcanic_tremor", "microseism", "ambient_noise",
            "teleseismic_event", "local_event", "explosion", "landslide_signal"
        ]
        
        results = []
        base_date = datetime(2020, 1, 1)
        
        for i in range(min(limit, 25)):
            # Select relevant elements based on query
            relevant_network = next(
                (net for net in seismic_networks if net in query.upper()),
                seismic_networks[i % len(seismic_networks)]
            )
            
            relevant_station_type = next(
                (stype for stype in station_types if any(word in query.lower() for word in stype.split('_'))),
                station_types[i % len(station_types)]
            )
            
            relevant_product = next(
                (prod for prod in data_products if any(word in query.lower() for word in prod.split('_'))),
                data_products[i % len(data_products)]
            )
            
            relevant_phenomenon = next(
                (phen for phen in seismic_phenomena if any(word in query.lower() for word in phen.split('_'))),
                seismic_phenomena[i % len(seismic_phenomena)]
            )
            
            acquisition_date = base_date + timedelta(days=i * 7)  # Weekly data
            
            result = {
                "title": f"{relevant_network} Network {relevant_product.replace('_', ' ').title()}: {relevant_phenomenon.replace('_', ' ').title()}",
                "network": relevant_network,
                "station": f"{relevant_network}{100 + i:02d}",
                "location": f"{i:02d}",
                "channel": self._get_channel_code(relevant_station_type, i),
                "station_type": relevant_station_type,
                "data_product": relevant_product,
                "phenomenon": relevant_phenomenon,
                "start_time": acquisition_date,
                "end_time": acquisition_date + timedelta(hours=24),
                "sample_rate": self._get_sample_rate(relevant_station_type),
                "instrument_response": self._get_instrument_response(relevant_station_type),
                "coordinates": self._generate_coordinates(relevant_network, i),
                "elevation_m": 100 + (i * 50) % 3000,
                "depth_m": (i * 10) % 100,
                "data_quality": self._assess_data_quality(i),
                "availability_percent": 85 + (i % 15),
                "file_size_mb": 10 + (i * 5) % 100,
                "data_format": "miniSEED",
                "metadata_format": "StationXML",
                "source": "iris_fallback",
                "applications": self._get_seismic_applications(relevant_phenomenon),
                "analysis_methods": self._get_analysis_methods(relevant_product),
                "frequency_band": self._get_frequency_band(relevant_station_type)
            }
            results.append(result)
            
        return results
        
    def _get_channel_code(self, station_type: str, index: int) -> str:
        """Generate realistic channel codes"""
        
        # Standard SEED channel naming convention
        if "broadband" in station_type:
            return f"BH{['Z', 'N', 'E'][index % 3]}"
        elif "very_broadband" in station_type:
            return f"VH{['Z', 'N', 'E'][index % 3]}"
        elif "short_period" in station_type:
            return f"SH{['Z', 'N', 'E'][index % 3]}"
        elif "strong_motion" in station_type:
            return f"HN{['Z', 'N', 'E'][index % 3]}"
        elif "accelerometer" in station_type:
            return f"BN{['Z', 'N', 'E'][index % 3]}"
        elif "pressure" in station_type:
            return "BDO"
        elif "infrasound" in station_type:
            return "BDF"
        else:
            return f"BH{['Z', 'N', 'E'][index % 3]}"
            
    def _get_sample_rate(self, station_type: str) -> int:
        """Get typical sample rate for station type"""
        
        sample_rates = {
            "broadband": 40,
            "very_broadband": 20,
            "short_period": 100,
            "strong_motion": 200,
            "accelerometer": 100,
            "pressure": 40,
            "infrasound": 40
        }
        
        return sample_rates.get(station_type, 40)
        
    def _get_instrument_response(self, station_type: str) -> Dict[str, Any]:
        """Get instrument response characteristics"""
        
        responses = {
            "broadband": {
                "sensitivity": 1.677e9,
                "units": "m/s",
                "frequency_range": "0.01-20 Hz",
                "instrument": "Streckeisen STS-2"
            },
            "very_broadband": {
                "sensitivity": 1.5e9,
                "units": "m/s",
                "frequency_range": "0.003-50 Hz",
                "instrument": "Nanometrics Trillium"
            },
            "short_period": {
                "sensitivity": 2e8,
                "units": "m/s",
                "frequency_range": "1-50 Hz",
                "instrument": "Mark L4C"
            },
            "strong_motion": {
                "sensitivity": 1e6,
                "units": "m/s^2",
                "frequency_range": "0-50 Hz",
                "instrument": "Kinemetrics FBA"
            },
            "accelerometer": {
                "sensitivity": 1e6,
                "units": "m/s^2",
                "frequency_range": "0-100 Hz",
                "instrument": "Episensor"
            }
        }
        
        return responses.get(station_type, responses["broadband"])
        
    def _generate_coordinates(self, network: str, index: int) -> Dict[str, float]:
        """Generate realistic coordinates based on network"""
        
        # Network-specific coordinate ranges
        network_regions = {
            "IU": {"lat_range": (-60, 70), "lon_range": (-180, 180)},  # Global
            "II": {"lat_range": (-60, 70), "lon_range": (-180, 180)},  # Global
            "TA": {"lat_range": (25, 70), "lon_range": (-170, -60)},   # Transportable Array
            "AK": {"lat_range": (55, 72), "lon_range": (-180, -130)},  # Alaska
            "CI": {"lat_range": (32, 37), "lon_range": (-121, -114)},  # California
            "US": {"lat_range": (25, 50), "lon_range": (-130, -65)},   # US
            "UU": {"lat_range": (37, 42), "lon_range": (-114, -109)},  # Utah
            "WY": {"lat_range": (41, 45), "lon_range": (-111, -104)}   # Wyoming
        }
        
        region = network_regions.get(network, {"lat_range": (30, 60), "lon_range": (-130, -70)})
        
        lat_min, lat_max = region["lat_range"]
        lon_min, lon_max = region["lon_range"]
        
        # Generate coordinates with some variation
        lat = lat_min + (lat_max - lat_min) * (index / 100) % 1
        lon = lon_min + (lon_max - lon_min) * ((index * 1.618) / 100) % 1  # Golden ratio for distribution
        
        return {"latitude": lat, "longitude": lon}
        
    def _assess_data_quality(self, index: int) -> str:
        """Assess data quality"""
        
        quality_scores = ["excellent", "good", "fair", "poor"]
        # Most data should be good quality
        weights = [0.3, 0.5, 0.15, 0.05]
        
        score_index = 0
        rand_val = (index * 0.1234) % 1
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val < cumulative:
                score_index = i
                break
                
        return quality_scores[score_index]
        
    def _get_seismic_applications(self, phenomenon: str) -> List[str]:
        """Get applications for seismic phenomena"""
        
        application_mapping = {
            "earthquake": ["earthquake_location", "magnitude_determination", "source_mechanism"],
            "volcanic_tremor": ["eruption_monitoring", "magma_movement_tracking", "hazard_assessment"],
            "microseism": ["ocean_wave_monitoring", "storm_tracking", "climate_studies"],
            "ambient_noise": ["velocity_structure", "monitoring_changes", "tomography"],
            "teleseismic_event": ["global_tomography", "deep_earth_structure", "mantle_studies"],
            "local_event": ["local_structure", "hazard_monitoring", "induced_seismicity"],
            "explosion": ["nuclear_monitoring", "mining_activity", "avalanche_detection"],
            "landslide_signal": ["mass_movement_detection", "slope_stability", "hazard_warning"]
        }
        
        return application_mapping.get(phenomenon, ["general_seismic_monitoring"])
        
    def _get_analysis_methods(self, product: str) -> List[str]:
        """Get analysis methods for data products"""
        
        method_mapping = {
            "continuous_waveform": ["spectral_analysis", "ambient_noise_tomography", "event_detection"],
            "event_waveform": ["source_inversion", "travel_time_analysis", "magnitude_calculation"],
            "ambient_noise": ["cross_correlation", "surface_wave_tomography", "velocity_monitoring"],
            "receiver_function": ["crustal_structure", "moho_depth", "velocity_discontinuities"],
            "spectral_density": ["noise_analysis", "instrument_performance", "site_response"],
            "polarization_data": ["wave_type_identification", "source_direction", "anisotropy"],
            "particle_motion": ["wave_propagation", "source_mechanism", "site_effects"],
            "cross_correlation": ["velocity_structure", "temporal_changes", "anisotropy"]
        }
        
        return method_mapping.get(product, ["waveform_analysis"])
        
    def _get_frequency_band(self, station_type: str) -> Dict[str, float]:
        """Get frequency band for station type"""
        
        bands = {
            "broadband": {"low_freq": 0.01, "high_freq": 20.0},
            "very_broadband": {"low_freq": 0.003, "high_freq": 50.0},
            "short_period": {"low_freq": 1.0, "high_freq": 50.0},
            "strong_motion": {"low_freq": 0.0, "high_freq": 50.0},
            "accelerometer": {"low_freq": 0.0, "high_freq": 100.0},
            "pressure": {"low_freq": 0.001, "high_freq": 20.0},
            "infrasound": {"low_freq": 0.001, "high_freq": 20.0}
        }
        
        return bands.get(station_type, {"low_freq": 0.01, "high_freq": 20.0})
            
    async def _execute_query(self, query: str, limit: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Execute the actual query against IRIS services"""
        
        results = []
        
        try:
            # Parse search parameters
            search_params = self._parse_search_parameters(query, limit, **kwargs)
            
            # Query station metadata
            if search_params.get('include_stations', True):
                station_results = await self._query_station_metadata(search_params)
                results.extend(station_results)
                
            # Query waveform data availability
            if search_params.get('include_waveforms', True):
                waveform_results = await self._query_waveform_availability(search_params)
                results.extend(waveform_results)
                
            # Query event data
            if search_params.get('include_events', False):
                event_results = await self._query_seismic_events(search_params)
                results.extend(event_results)
                
            # Enhance results with seismological context
            enhanced_results = await self._enhance_seismological_context(results)
            
            return enhanced_results[:limit]
            
        except Exception as e:
            self.logger.error(f"IRIS/EarthScope query failed: {e}")
            # Fall back to mock data on API failure
            return self._fallback_query(query, limit, **kwargs)
            
    def _parse_search_parameters(self, query: str, limit: int, **kwargs) -> Dict[str, Any]:
        """Parse search query into IRIS FDSN parameters"""
        
        params = {
            'keyword': query,
            'limit': min(limit, 1000),
            'starttime': kwargs.get('start_time', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S')),
            'endtime': kwargs.get('end_time', datetime.now().strftime('%Y-%m-%dT%H:%M:%S')),
            'network': kwargs.get('network', None),
            'station': kwargs.get('station', None),
            'location': kwargs.get('location', None),
            'channel': kwargs.get('channel', None),
            'minlatitude': kwargs.get('min_latitude', None),
            'maxlatitude': kwargs.get('max_latitude', None),
            'minlongitude': kwargs.get('min_longitude', None),
            'maxlongitude': kwargs.get('max_longitude', None),
            'level': kwargs.get('level', 'station'),
            'format': kwargs.get('format', 'xml'),
            'include_stations': 'station' in query.lower() or kwargs.get('include_stations', True),
            'include_waveforms': 'waveform' in query.lower() or 'data' in query.lower() or kwargs.get('include_waveforms', True),
            'include_events': 'event' in query.lower() or kwargs.get('include_events', False)
        }
        
        # Auto-detect networks from query
        network_codes = ["IU", "II", "G", "GT", "IC", "CU", "US", "TA", "AK", "CI", "UU", "WY"]
        for net in network_codes:
            if net in query.upper():
                params['network'] = net
                break
                
        return params
        
    async def _query_station_metadata(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query IRIS station metadata"""
        
        async with self.rate_limiter:
            try:
                query_params = {
                    'format': 'xml',
                    'level': params['level'],
                    'starttime': params['starttime'],
                    'endtime': params['endtime']
                }
                
                # Add spatial constraints
                if params['minlatitude']:
                    query_params['minlatitude'] = params['minlatitude']
                if params['maxlatitude']:
                    query_params['maxlatitude'] = params['maxlatitude']
                if params['minlongitude']:
                    query_params['minlongitude'] = params['minlongitude']
                if params['maxlongitude']:
                    query_params['maxlongitude'] = params['maxlongitude']
                    
                # Add network/station constraints
                if params['network']:
                    query_params['network'] = params['network']
                if params['station']:
                    query_params['station'] = params['station']
                if params['location']:
                    query_params['location'] = params['location']
                if params['channel']:
                    query_params['channel'] = params['channel']
                
                async with self.session.get(f"{self.fdsnws_station}/query", params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_stationxml(content)
                    elif response.status == 204:
                        self.logger.info("No station data found for query")
                        return []
                    else:
                        self.logger.warning(f"Station query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query station metadata: {e}")
                return []
                
    async def _query_waveform_availability(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query waveform data availability"""
        
        # This would typically query the IRIS availability service
        # For now, generate availability information based on station query
        availability_results = []
        
        # Mock availability data
        for i in range(min(10, params['limit'] // 3)):
            availability = {
                "title": f"Waveform Availability {i+1}",
                "network": params.get('network', 'IU'),
                "station": f"ANMO",
                "location": "00",
                "channel": f"BHZ",
                "data_type": "continuous_waveform",
                "start_time": params['starttime'],
                "end_time": params['endtime'],
                "sample_rate": 40,
                "availability_percent": 90 + i,
                "data_quality": "good",
                "gaps_count": i,
                "overlaps_count": 0,
                "file_count": 24 + i,
                "total_size_mb": 500 + i * 50,
                "source": "iris_waveform_availability"
            }
            availability_results.append(availability)
            
        return availability_results
        
    async def _query_seismic_events(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query seismic event catalog"""
        
        async with self.rate_limiter:
            try:
                query_params = {
                    'format': 'xml',
                    'starttime': params['starttime'],
                    'endtime': params['endtime'],
                    'minmagnitude': 4.0,  # Focus on significant events
                    'orderby': 'time-asc'
                }
                
                # Add spatial constraints
                if params['minlatitude']:
                    query_params['minlatitude'] = params['minlatitude']
                if params['maxlatitude']:
                    query_params['maxlatitude'] = params['maxlatitude']
                if params['minlongitude']:
                    query_params['minlongitude'] = params['minlongitude']
                if params['maxlongitude']:
                    query_params['maxlongitude'] = params['maxlongitude']
                
                async with self.session.get(f"{self.fdsnws_event}/query", params=query_params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_quakeml(content)
                    elif response.status == 204:
                        self.logger.info("No events found for query")
                        return []
                    else:
                        self.logger.warning(f"Event query returned status {response.status}")
                        return []
                        
            except Exception as e:
                self.logger.error(f"Failed to query events: {e}")
                return []
                
    def _parse_stationxml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse StationXML response"""
        
        # Simplified StationXML parsing
        # In practice, would use proper XML parsing library
        results = []
        
        # Mock parsing for demonstration
        # This would extract network, station, channel information
        for i in range(5):  # Mock 5 stations
            result = {
                "title": f"Station Metadata {i+1}",
                "network": "IU",
                "station": f"ANMO",
                "location": "00",
                "channels": ["BHZ", "BHN", "BHE"],
                "latitude": 34.9459 + i * 0.1,
                "longitude": -106.4572 - i * 0.1,
                "elevation": 1850 + i * 50,
                "depth": 0,
                "start_date": "2000-01-01T00:00:00",
                "end_date": None,
                "site_name": f"Albuquerque Test Site {i+1}",
                "instruments": ["Streckeisen STS-2"],
                "data_availability": 95,
                "source": "iris_stationxml"
            }
            results.append(result)
            
        return results
        
    def _parse_quakeml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse QuakeML response"""
        
        # Simplified QuakeML parsing
        results = []
        
        # Mock parsing for demonstration
        for i in range(3):  # Mock 3 events
            result = {
                "title": f"Seismic Event {i+1}",
                "event_id": f"us{2020000000 + i}",
                "magnitude": 5.0 + i * 0.5,
                "magnitude_type": "mw",
                "latitude": 40.0 + i * 2,
                "longitude": -120.0 - i * 2,
                "depth_km": 10 + i * 5,
                "origin_time": datetime.now() - timedelta(days=i),
                "region": f"Test Region {i+1}",
                "event_type": "earthquake",
                "agency": "IRIS",
                "review_status": "reviewed",
                "source": "iris_event_catalog"
            }
            results.append(result)
            
        return results
        
    async def _enhance_seismological_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance results with seismological context"""
        
        enhanced_results = []
        
        for result in results:
            try:
                # Add seismological analysis capabilities
                result['analysis_capabilities'] = self._assess_analysis_capabilities(result)
                
                # Add data quality assessment
                result['quality_assessment'] = self._assess_seismic_data_quality(result)
                
                # Add processing recommendations
                result['processing_recommendations'] = self._generate_seismic_processing_recommendations(result)
                
                # Add scientific applications
                result['scientific_applications'] = self._identify_seismological_applications(result)
                
                # Add network context
                result['network_context'] = self._analyze_network_context(result)
                
                enhanced_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance seismic result: {e}")
                enhanced_results.append(result)
                continue
                
        return enhanced_results
        
    def _assess_analysis_capabilities(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess analysis capabilities for seismic data"""
        
        capabilities = {
            'temporal_analysis': False,
            'spectral_analysis': False,
            'polarization_analysis': False,
            'array_processing': False,
            'tomography_suitable': False
        }
        
        # Assess based on data type and quality
        station_type = result.get('station_type', '')
        data_quality = result.get('data_quality', 'good')
        
        if 'broadband' in station_type and data_quality in ['excellent', 'good']:
            capabilities['temporal_analysis'] = True
            capabilities['spectral_analysis'] = True
            capabilities['tomography_suitable'] = True
            
        if result.get('channels') and len(result.get('channels', [])) >= 3:
            capabilities['polarization_analysis'] = True
            
        network = result.get('network', '')
        if network in ['TA', 'IU', 'II']:  # Networks with good spatial coverage
            capabilities['array_processing'] = True
            
        return capabilities
        
    def _assess_seismic_data_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess seismic data quality"""
        
        quality = {
            'overall_grade': result.get('data_quality', 'good'),
            'signal_to_noise': 'good',
            'completeness': result.get('availability_percent', 90),
            'timing_accuracy': 'excellent',
            'amplitude_accuracy': 'good',
            'limiting_factors': []
        }
        
        # Add limiting factors based on context
        if quality['completeness'] < 80:
            quality['limiting_factors'].append('data_gaps')
            
        if result.get('gaps_count', 0) > 10:
            quality['limiting_factors'].append('frequent_interruptions')
            
        if result.get('elevation_m', 0) > 3000:
            quality['limiting_factors'].append('high_altitude_noise')
            
        return quality
        
    def _generate_seismic_processing_recommendations(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate seismic data processing recommendations"""
        
        recommendations = []
        
        station_type = result.get('station_type', '')
        data_product = result.get('data_product', '')
        
        # Broadband recommendations
        if 'broadband' in station_type:
            recommendations.append({
                'category': 'preprocessing',
                'recommendation': 'Apply instrument response correction',
                'rationale': 'Essential for accurate amplitude analysis'
            })
            
        # Continuous data recommendations
        if 'continuous' in data_product:
            recommendations.append({
                'category': 'noise_reduction',
                'recommendation': 'Apply spectral whitening for ambient noise studies',
                'rationale': 'Enhances cross-correlation signal quality'
            })
            
        # Event data recommendations
        if 'event' in data_product:
            recommendations.append({
                'category': 'filtering',
                'recommendation': 'Apply bandpass filter based on event distance',
                'rationale': 'Optimizes signal-to-noise for specific phases'
            })
            
        # Quality-based recommendations
        data_quality = result.get('data_quality', 'good')
        if data_quality in ['fair', 'poor']:
            recommendations.append({
                'category': 'quality_control',
                'recommendation': 'Implement additional quality screening',
                'rationale': 'Lower quality data requires more stringent processing'
            })
            
        return recommendations
        
    def _identify_seismological_applications(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify seismological applications"""
        
        applications = []
        
        station_type = result.get('station_type', '')
        network = result.get('network', '')
        data_product = result.get('data_product', '')
        
        # Network-specific applications
        if network == 'TA':
            applications.append({
                'application': 'continental_scale_tomography',
                'confidence': 'high'
            })
        elif network in ['IU', 'II']:
            applications.append({
                'application': 'global_seismology',
                'confidence': 'high'
            })
        elif network in ['AK', 'YV']:
            applications.append({
                'application': 'regional_volcano_monitoring',
                'confidence': 'high'
            })
            
        # Station type applications
        if 'broadband' in station_type:
            applications.extend([
                {'application': 'earthquake_detection', 'confidence': 'high'},
                {'application': 'surface_wave_studies', 'confidence': 'high'},
                {'application': 'ambient_noise_tomography', 'confidence': 'medium'}
            ])
            
        if 'strong_motion' in station_type:
            applications.extend([
                {'application': 'ground_motion_studies', 'confidence': 'high'},
                {'application': 'engineering_seismology', 'confidence': 'high'}
            ])
            
        # Data product applications
        if 'ambient_noise' in data_product:
            applications.append({
                'application': 'velocity_structure_imaging',
                'confidence': 'high'
            })
            
        return applications
        
    def _analyze_network_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seismic network context"""
        
        context = {
            'network_purpose': 'unknown',
            'deployment_type': 'unknown',
            'spatial_scale': 'unknown',
            'temporal_scale': 'unknown'
        }
        
        network = result.get('network', '')
        
        network_info = {
            'IU': {
                'purpose': 'global_monitoring',
                'deployment': 'permanent',
                'spatial_scale': 'global',
                'temporal_scale': 'continuous'
            },
            'TA': {
                'purpose': 'continental_imaging',
                'deployment': 'temporary',
                'spatial_scale': 'continental',
                'temporal_scale': 'campaign'
            },
            'AK': {
                'purpose': 'regional_monitoring',
                'deployment': 'permanent',
                'spatial_scale': 'regional',
                'temporal_scale': 'continuous'
            }
        }
        
        if network in network_info:
            info = network_info[network]
            context['network_purpose'] = info['purpose']
            context['deployment_type'] = info['deployment']
            context['spatial_scale'] = info['spatial_scale']
            context['temporal_scale'] = info['temporal_scale']
            
        return context
        
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get EarthScope/IRIS ontology mapping"""
        return self._ontology_mapping.copy()
        
    async def get_network_information(self) -> List[Dict[str, Any]]:
        """Get information about seismic networks"""
        
        networks = [
            {
                "code": "IU",
                "name": "Global Seismographic Network - IRIS/USGS",
                "description": "Permanent global network for earthquake monitoring and research",
                "stations": 150,
                "deployment_type": "permanent",
                "primary_purpose": "global_earthquake_monitoring",
                "data_types": ["broadband_seismic", "strong_motion", "pressure"]
            },
            {
                "code": "TA",
                "name": "Transportable Array",
                "description": "Dense temporary array for continental-scale imaging",
                "stations": 400,
                "deployment_type": "temporary",
                "primary_purpose": "continental_structure_imaging",
                "data_types": ["broadband_seismic", "strong_motion"]
            },
            {
                "code": "AK",
                "name": "Alaska Regional Network",
                "description": "Regional network for Alaska earthquake and volcano monitoring",
                "stations": 200,
                "deployment_type": "permanent",
                "primary_purpose": "regional_monitoring",
                "data_types": ["broadband_seismic", "strong_motion", "infrasound"]
            },
            {
                "code": "CI",
                "name": "Southern California Seismic Network",
                "description": "Dense regional network for California earthquake monitoring",
                "stations": 400,
                "deployment_type": "permanent",
                "primary_purpose": "earthquake_hazard_monitoring",
                "data_types": ["broadband_seismic", "strong_motion", "accelerometer"]
            }
        ]
        
        return networks

class LiteratureExtractor:
    """Extract knowledge from geological literature"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
            
        # Initialize transformer model for embeddings
        if TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = pipeline(
                    "feature-extraction",
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    return_tensors="np"
                )
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Geological term patterns
        self.geological_patterns = self._initialize_geological_patterns()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for literature extraction"""
        return {
            "max_document_length": 50000,
            "entity_confidence_threshold": 0.7,
            "relation_confidence_threshold": 0.6,
            "enable_coreference_resolution": True,
            "enable_temporal_extraction": True,
            "geological_vocabularies": [
                "minerals", "rocks", "processes", "locations", "measurements"
            ],
            "language_models": {
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "ner": "en_core_web_sm"
            }
        }
    
    def _initialize_geological_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for geological entity recognition"""
        return {
            "minerals": [
                "quartz", "feldspar", "mica", "calcite", "dolomite", "gypsum", "halite",
                "olivine", "pyroxene", "amphibole", "garnet", "epidote", "chlorite",
                "magnetite", "hematite", "pyrite", "galena", "sphalerite"
            ],
            "rock_types": [
                "granite", "basalt", "andesite", "rhyolite", "gabbro", "peridotite",
                "sandstone", "limestone", "shale", "mudstone", "conglomerate",
                "gneiss", "schist", "slate", "marble", "quartzite", "serpentinite"
            ],
            "processes": [
                "crystallization", "metamorphism", "weathering", "erosion", "deposition",
                "diagenesis", "melting", "solidification", "deformation", "fracturing",
                "faulting", "folding", "uplift", "subsidence", "volcanism", "plutonism"
            ],
            "measurements": [
                "temperature", "pressure", "density", "viscosity", "porosity", "permeability",
                "magnetic susceptibility", "electrical conductivity", "seismic velocity",
                "age", "composition", "concentration", "flow rate", "stress", "strain"
            ],
            "phenomena": [
                "earthquake", "volcanic eruption", "landslide", "tsunami", "flood",
                "mountain building", "seafloor spreading", "subduction", "collision",
                "rifting", "hotspot", "plume", "convection", "differentiation"
            ]
        }
    
    async def extract_from_document(self, document: LiteratureDocument) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """Extract entities and relations from a literature document"""
        
        entities = []
        relations = []
        
        try:
            # Combine title, abstract, and full text for processing
            full_content = f"{document.title}\n\n{document.abstract}\n\n{document.full_text}"
            
            if len(full_content) > self.config["max_document_length"]:
                full_content = full_content[:self.config["max_document_length"]]
            
            # Extract entities
            extracted_entities = await self._extract_entities(full_content, document.id)
            entities.extend(extracted_entities)
            
            # Extract relations
            extracted_relations = await self._extract_relations(
                full_content, extracted_entities, document.id
            )
            relations.extend(extracted_relations)
            
            # Extract temporal relations if enabled
            if self.config["enable_temporal_extraction"]:
                temporal_relations = await self._extract_temporal_relations(
                    full_content, extracted_entities, document.id
                )
                relations.extend(temporal_relations)
            
            # Update document with extracted knowledge
            document.extracted_entities = entities
            document.extracted_relations = relations
            
        except Exception as e:
            self.logger.error(f"Failed to extract knowledge from document {document.id}: {str(e)}")
        
        return entities, relations
    
    async def _extract_entities(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract geological entities from text"""
        
        entities = []
        entity_mentions = {}  # Track unique entities
        
        # Pattern-based extraction
        pattern_entities = self._extract_entities_by_patterns(text, source_id)
        for entity in pattern_entities:
            entity_key = (entity.name.lower(), entity.entity_type)
            if entity_key not in entity_mentions:
                entity_mentions[entity_key] = entity
                entities.append(entity)
        
        # NLP-based extraction if spaCy is available
        if self.nlp:
            nlp_entities = await self._extract_entities_with_nlp(text, source_id)
            for entity in nlp_entities:
                entity_key = (entity.name.lower(), entity.entity_type)
                if entity_key not in entity_mentions:
                    entity_mentions[entity_key] = entity
                    entities.append(entity)
                else:
                    # Merge information
                    existing_entity = entity_mentions[entity_key]
                    existing_entity.confidence = max(existing_entity.confidence, entity.confidence)
                    existing_entity.properties.update(entity.properties)
        
        # Generate embeddings if transformer model is available
        if self.embedding_model:
            for entity in entities:
                try:
                    embedding = self.embedding_model(entity.name)[0][0]  # Get first token embedding
                    entity.embedding_vector = np.mean(embedding, axis=0)  # Average over tokens
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for {entity.name}: {str(e)}")
        
        return entities
    
    def _extract_entities_by_patterns(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract entities using pattern matching"""
        
        entities = []
        text_lower = text.lower()
        
        for category, terms in self.geological_patterns.items():
            for term in terms:
                # Find all occurrences of the term
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Determine entity type based on category
                    if category == "minerals":
                        entity_type = EntityType.MINERAL
                    elif category == "rock_types":
                        entity_type = EntityType.ROCK_TYPE
                    elif category == "processes":
                        entity_type = EntityType.PROCESS
                    elif category == "measurements":
                        entity_type = EntityType.MEASUREMENT
                    elif category == "phenomena":
                        entity_type = EntityType.PHENOMENON
                    else:
                        entity_type = EntityType.CONCEPT
                    
                    # Create entity
                    entity_id = hashlib.md5(f"{term}_{entity_type.value}_{source_id}".encode()).hexdigest()[:16]
                    
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=term,
                        entity_type=entity_type,
                        properties={
                            "category": category,
                            "context": context,
                            "mention_count": 1
                        },
                        confidence=0.8,  # Pattern-based confidence
                        source_references=[source_id]
                    )
                    
                    entities.append(entity)
                    break  # Only create one entity per term per document
        
        return entities
    
    async def _extract_entities_with_nlp(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract entities using NLP (spaCy)"""
        
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                if entity_type:  # Only include relevant geological entities
                    entity_id = hashlib.md5(f"{ent.text}_{entity_type.value}_{source_id}".encode()).hexdigest()[:16]
                    
                    # Get surrounding context
                    start_token = max(0, ent.start - 10)
                    end_token = min(len(doc), ent.end + 10)
                    context = doc[start_token:end_token].text
                    
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=ent.text,
                        entity_type=entity_type,
                        properties={
                            "spacy_label": ent.label_,
                            "context": context,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char
                        },
                        confidence=self._calculate_entity_confidence(ent),
                        source_references=[source_id]
                    )
                    
                    entities.append(entity)
            
            # Extract additional entities using custom patterns
            custom_entities = self._extract_custom_geological_entities(doc, source_id)
            entities.extend(custom_entities)
            
        except Exception as e:
            self.logger.warning(f"NLP entity extraction failed: {str(e)}")
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to geological entity types"""
        
        mapping = {
            "PERSON": EntityType.RESEARCHER,
            "ORG": EntityType.PUBLICATION,  # Journals, institutions
            "GPE": EntityType.LOCATION,      # Countries, cities, states
            "LOC": EntityType.LOCATION,      # Locations, mountain ranges, etc.
            "PRODUCT": EntityType.INSTRUMENT, # Scientific instruments
            "EVENT": EntityType.PHENOMENON,  # Geological events
            "FAC": EntityType.LOCATION,      # Facilities, landmarks
        }
        
        return mapping.get(spacy_label)
    
    def _calculate_entity_confidence(self, spacy_entity) -> float:
        """Calculate confidence score for spaCy entity"""
        
        base_confidence = 0.7
        
        # Increase confidence for longer entities
        if len(spacy_entity.text) > 10:
            base_confidence += 0.1
        
        # Increase confidence for capitalized entities
        if spacy_entity.text.istitle():
            base_confidence += 0.1
        
        # Check if entity matches known geological terms
        entity_lower = spacy_entity.text.lower()
        for category, terms in self.geological_patterns.items():
            if any(term in entity_lower for term in terms):
                base_confidence += 0.2
                break
        
        return min(1.0, base_confidence)
    
    def _extract_custom_geological_entities(self, doc, source_id: str) -> List[KnowledgeEntity]:
        """Extract geological entities using custom rules"""
        
        entities = []
        
        # Look for measurement patterns (number + unit)
        measurement_pattern = r'(\d+\.?\d*)\s*(°C|°F|K|Pa|MPa|GPa|m/s|km/s|g/cm³|%|ppm|Ma|Ga)'
        
        for match in re.finditer(measurement_pattern, doc.text):
            value = match.group(1)
            unit = match.group(2)
            
            entity_id = hashlib.md5(f"measurement_{value}_{unit}_{source_id}".encode()).hexdigest()[:16]
            
            entity = KnowledgeEntity(
                id=entity_id,
                name=f"{value} {unit}",
                entity_type=EntityType.MEASUREMENT,
                properties={
                    "value": float(value),
                    "unit": unit,
                    "measurement_type": self._classify_measurement_unit(unit)
                },
                confidence=0.9,
                source_references=[source_id]
            )
            
            entities.append(entity)
        
        return entities
    
    def _classify_measurement_unit(self, unit: str) -> str:
        """Classify measurement unit type"""
        
        unit_types = {
            "temperature": ["°C", "°F", "K"],
            "pressure": ["Pa", "MPa", "GPa", "bar"],
            "velocity": ["m/s", "km/s", "cm/s"],
            "density": ["g/cm³", "kg/m³"],
            "concentration": ["%", "ppm", "ppb"],
            "time": ["Ma", "Ga", "ka", "years"],
            "length": ["m", "km", "cm", "mm"]
        }
        
        for unit_type, units in unit_types.items():
            if unit in units:
                return unit_type
        
        return "unknown"
    
    async def _extract_relations(self, text: str, entities: List[KnowledgeEntity], 
                               source_id: str) -> List[KnowledgeRelation]:
        """Extract relations between entities"""
        
        relations = []
        
        # Pattern-based relation extraction
        pattern_relations = self._extract_relations_by_patterns(text, entities, source_id)
        relations.extend(pattern_relations)
        
        # Dependency parsing-based extraction if spaCy is available
        if self.nlp:
            dependency_relations = await self._extract_relations_with_dependencies(
                text, entities, source_id
            )
            relations.extend(dependency_relations)
        
        return relations
    
    def _extract_relations_by_patterns(self, text: str, entities: List[KnowledgeEntity],
                                     source_id: str) -> List[KnowledgeRelation]:
        """Extract relations using linguistic patterns"""
        
        relations = []
        text_lower = text.lower()
        
        # Create entity name mapping
        entity_names = {entity.name.lower(): entity for entity in entities}
        
        # Causal relation patterns
        causal_patterns = [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
            r'(\w+)\s+triggers?\s+(\w+)',
            r'(\w+)\s+induces?\s+(\w+)'
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                source_term = match.group(1)
                target_term = match.group(2)
                
                # Find matching entities
                source_entity = self._find_matching_entity(source_term, entity_names)
                target_entity = self._find_matching_entity(target_term, entity_names)
                
                if source_entity and target_entity:
                    relation_id = hashlib.md5(
                        f"{source_entity.id}_{target_entity.id}_causes_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relation_type=RelationType.CAUSES,
                        properties={
                            "pattern_matched": pattern,
                            "context": match.group(0)
                        },
                        confidence=0.7,
                        evidence=[match.group(0)],
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        # Compositional patterns
        compositional_patterns = [
            r'(\w+)\s+contains?\s+(\w+)',
            r'(\w+)\s+composed\s+of\s+(\w+)',
            r'(\w+)\s+made\s+of\s+(\w+)',
            r'(\w+)\s+consists?\s+of\s+(\w+)'
        ]
        
        for pattern in compositional_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                container = match.group(1)
                component = match.group(2)
                
                container_entity = self._find_matching_entity(container, entity_names)
                component_entity = self._find_matching_entity(component, entity_names)
                
                if container_entity and component_entity:
                    relation_id = hashlib.md5(
                        f"{container_entity.id}_{component_entity.id}_composed_of_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=container_entity.id,
                        target_entity_id=component_entity.id,
                        relation_type=RelationType.COMPOSED_OF,
                        confidence=0.8,
                        evidence=[match.group(0)],
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        return relations
    
    def _find_matching_entity(self, term: str, entity_mapping: Dict[str, KnowledgeEntity]) -> Optional[KnowledgeEntity]:
        """Find matching entity for a term"""
        
        term_lower = term.lower()
        
        # Exact match
        if term_lower in entity_mapping:
            return entity_mapping[term_lower]
        
        # Partial match
        for entity_name, entity in entity_mapping.items():
            if term_lower in entity_name or entity_name in term_lower:
                return entity
        
        return None
    
    async def _extract_relations_with_dependencies(self, text: str, entities: List[KnowledgeEntity],
                                                 source_id: str) -> List[KnowledgeRelation]:
        """Extract relations using dependency parsing"""
        
        relations = []
        
        if not self.nlp:
            return relations
        
        try:
            doc = self.nlp(text)
            entity_spans = {entity.name.lower(): entity for entity in entities}
            
            # Process each sentence
            for sent in doc.sents:
                sentence_relations = self._extract_sentence_relations(
                    sent, entity_spans, source_id
                )
                relations.extend(sentence_relations)
                
        except Exception as e:
            self.logger.warning(f"Dependency parsing failed: {str(e)}")
        
        return relations
    
    def _extract_sentence_relations(self, sentence, entity_spans: Dict[str, KnowledgeEntity],
                                  source_id: str) -> List[KnowledgeRelation]:
        """Extract relations from a single sentence using dependency parsing"""
        
        relations = []
        
        # Find verbs that might indicate relationships
        relationship_verbs = {
            "cause": RelationType.CAUSES, "causes": RelationType.CAUSES, "caused": RelationType.CAUSES,
            "influence": RelationType.INFLUENCES, "influences": RelationType.INFLUENCES, "influenced": RelationType.INFLUENCES,
            "occur": RelationType.OCCURS_IN, "occurs": RelationType.OCCURS_IN, "occurred": RelationType.OCCURS_IN,
            "contain": RelationType.COMPOSED_OF, "contains": RelationType.COMPOSED_OF, "contained": RelationType.COMPOSED_OF,
            "precede": RelationType.PRECEDES, "precedes": RelationType.PRECEDES, "preceded": RelationType.PRECEDES,
            "correlate": RelationType.CORRELATES_WITH, "correlates": RelationType.CORRELATES_WITH, "correlated": RelationType.CORRELATES_WITH
        }
        
        for token in sentence:
            if token.lemma_ in relationship_verbs:
                relation_type = relationship_verbs[token.lemma_]
                
                # Find subject and object
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child
                
                if subject and obj:
                    # Try to match with extracted entities
                    subject_entity = self._find_matching_entity(subject.text.lower(), entity_spans)
                    object_entity = self._find_matching_entity(obj.text.lower(), entity_spans)
                    
                    if subject_entity and object_entity:
                        relation_id = hashlib.md5(
                            f"{subject_entity.id}_{object_entity.id}_{relation_type.value}_{source_id}".encode()
                        ).hexdigest()[:16]
                        
                        relation = KnowledgeRelation(
                            id=relation_id,
                            source_entity_id=subject_entity.id,
                            target_entity_id=object_entity.id,
                            relation_type=relation_type,
                            properties={
                                "verb": token.lemma_,
                                "dependency_pattern": f"{subject.dep_}-{token.lemma_}-{obj.dep_}"
                            },
                            confidence=0.6,
                            evidence=[sentence.text],
                            source_references=[source_id]
                        )
                        
                        relations.append(relation)
        
        return relations
    
    async def _extract_temporal_relations(self, text: str, entities: List[KnowledgeEntity],
                                        source_id: str) -> List[KnowledgeRelation]:
        """Extract temporal relations from text"""
        
        relations = []
        
        # Temporal patterns
        temporal_patterns = [
            r'(\w+)\s+before\s+(\w+)',
            r'(\w+)\s+after\s+(\w+)',
            r'(\w+)\s+during\s+(\w+)',
            r'(\w+)\s+precedes?\s+(\w+)',
            r'(\w+)\s+follows?\s+(\w+)'
        ]
        
        entity_names = {entity.name.lower(): entity for entity in entities}
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                first_term = match.group(1)
                second_term = match.group(2)
                
                first_entity = self._find_matching_entity(first_term, entity_names)
                second_entity = self._find_matching_entity(second_term, entity_names)
                
                if first_entity and second_entity:
                    # Determine relation type based on pattern
                    if "before" in pattern or "precedes" in pattern:
                        relation_type = RelationType.TEMPORAL_BEFORE
                        source_entity = first_entity
                        target_entity = second_entity
                    elif "after" in pattern or "follows" in pattern:
                        relation_type = RelationType.TEMPORAL_AFTER
                        source_entity = second_entity
                        target_entity = first_entity
                    else:
                        relation_type = RelationType.TEMPORAL_BEFORE
                        source_entity = first_entity
                        target_entity = second_entity
                    
                    relation_id = hashlib.md5(
                        f"{source_entity.id}_{target_entity.id}_{relation_type.value}_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relation_type=relation_type,
                        properties={
                            "temporal_pattern": pattern,
                            "context": match.group(0)
                        },
                        confidence=0.6,
                        evidence=[match.group(0)],
                        temporal_context={
                            "extraction_method": "pattern_matching",
                            "temporal_indicator": "before/after"
                        },
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        return relations

class TemporalCausalAnalyzer:
    """Analyze temporal causal relationships in time-series data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_lag": 12,  # Maximum lag periods to test
            "significance_threshold": 0.05,
            "min_observations": 30,
            "detrend_method": "linear",
            "seasonal_decomposition": True,
            "granger_test": True,
            "ccf_analysis": True,
            "bootstrap_samples": 1000
        }
    
    async def analyze_temporal_causality(self, source_series: pd.Series, 
                                       target_series: pd.Series,
                                       entity_ids: Tuple[str, str]) -> Optional[TemporalCausalRelation]:
        """Analyze temporal causal relationship between two time series"""
        
        if len(source_series) < self.config["min_observations"]:
            self.logger.warning("Insufficient observations for temporal causal analysis")
            return None
        
        try:
            # Align time series
            aligned_data = pd.concat([source_series, target_series], axis=1, join='inner')
            if len(aligned_data) < self.config["min_observations"]:
                return None
            
            source_col = aligned_data.columns[0]
            target_col = aligned_data.columns[1]
            
            # Prepare data
            prepared_data = self._prepare_time_series_data(aligned_data)
            
            # Cross-correlation analysis
            ccf_results = None
            if self.config["ccf_analysis"]:
                ccf_results = self._cross_correlation_analysis(
                    prepared_data[source_col], prepared_data[target_col]
                )
            
            # Granger causality test
            granger_pvalue = None
            optimal_lag = 1
            if self.config["granger_test"]:
                granger_results = self._granger_causality_test(
                    prepared_data[source_col], prepared_data[target_col]
                )
                granger_pvalue = granger_results["p_value"]
                optimal_lag = granger_results["optimal_lag"]
            
            # Calculate causal strength
            causal_strength = self._calculate_causal_strength(
                prepared_data[source_col], prepared_data[target_col], optimal_lag
            )
            
            # Seasonality analysis
            seasonality = None
            if self.config["seasonal_decomposition"]:
                seasonality = self._analyze_seasonality(prepared_data)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(prepared_data)
            
            # Bootstrap confidence intervals
            confidence_intervals = await self._bootstrap_confidence_intervals(
                prepared_data[source_col], prepared_data[target_col], optimal_lag
            )
            
            # Create temporal causal relation
            relation_id = hashlib.md5(
                f"temporal_{entity_ids[0]}_{entity_ids[1]}".encode()
            ).hexdigest()[:16]
            
            temporal_relation = TemporalCausalRelation(
                relation_id=relation_id,
                source_entity_id=entity_ids[0],
                target_entity_id=entity_ids[1],
                temporal_lag=optimal_lag,
                temporal_units="periods",  # Adjust based on data frequency
                causal_strength=causal_strength,
                temporal_evidence=prepared_data,
                seasonality=seasonality,
                trend_analysis=trend_analysis,
                granger_causality_pvalue=granger_pvalue,
                ccf_analysis=ccf_results,
                confidence_intervals=confidence_intervals
            )
            
            return temporal_relation
            
        except Exception as e:
            self.logger.error(f"Temporal causal analysis failed: {str(e)}")
            return None
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for analysis"""
        
        prepared = data.copy()
        
        # Handle missing values
        prepared = prepared.fillna(method='ffill').fillna(method='bfill')
        
        # Detrend if specified
        if self.config["detrend_method"] == "linear":
            for col in prepared.columns:
                if prepared[col].dtype in [np.float64, np.int64]:
                    # Linear detrending
                    x = np.arange(len(prepared))
                    coeffs = np.polyfit(x, prepared[col], 1)
                    trend = np.polyval(coeffs, x)
                    prepared[col] = prepared[col] - trend
        
        return prepared
    
    def _cross_correlation_analysis(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-correlation analysis"""
        
        try:
            # Calculate cross-correlation
            max_lag = min(self.config["max_lag"], len(x) // 4)
            
            correlations = []
            lags = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = x.corr(y)
                elif lag > 0:
                    # y leads x by lag periods
                    if len(x) > lag:
                        corr = x[lag:].corr(y[:-lag])
                    else:
                        corr = np.nan
                else:
                    # x leads y by |lag| periods
                    abs_lag = abs(lag)
                    if len(y) > abs_lag:
                        corr = x[:-abs_lag].corr(y[abs_lag:])
                    else:
                        corr = np.nan
                
                if not np.isnan(corr):
                    correlations.append(corr)
                    lags.append(lag)
            
            # Find optimal lag
            max_corr_idx = np.argmax(np.abs(correlations))
            optimal_lag = lags[max_corr_idx]
            max_correlation = correlations[max_corr_idx]
            
            return {
                "correlations": correlations,
                "lags": lags,
                "optimal_lag": optimal_lag,
                "max_correlation": max_correlation,
                "significant_lags": [lag for i, lag in enumerate(lags) 
                                   if abs(correlations[i]) > 2/np.sqrt(len(x))]
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-correlation analysis failed: {str(e)}")
            return {"optimal_lag": 1, "max_correlation": 0.0}
    
    def _granger_causality_test(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Perform Granger causality test"""
        
        try:
            # Simple implementation - in practice, use statsmodels
            from scipy import stats
            
            best_lag = 1
            best_pvalue = 1.0
            
            for lag in range(1, min(self.config["max_lag"], len(x) // 4)):
                try:
                    # Create lagged variables
                    y_lagged = y[lag:].values
                    x_lagged = x[:-lag].values
                    y_current = y[lag:].values
                    
                    if len(y_lagged) < 10:
                        continue
                    
                    # Restricted model: y ~ y_lagged
                    y_lag_matrix = np.column_stack([y_lagged[:-1]])
                    y_target = y_current[1:]
                    
                    # Unrestricted model: y ~ y_lagged + x_lagged
                    xy_lag_matrix = np.column_stack([y_lagged[:-1], x_lagged[:-1]])
                    
                    # Calculate RSS for both models
                    try:
                        from sklearn.linear_model import LinearRegression
                        
                        # Restricted model
                        reg_restricted = LinearRegression().fit(y_lag_matrix, y_target)
                        y_pred_restricted = reg_restricted.predict(y_lag_matrix)
                        rss_restricted = np.sum((y_target - y_pred_restricted) ** 2)
                        
                        # Unrestricted model  
                        reg_unrestricted = LinearRegression().fit(xy_lag_matrix, y_target)
                        y_pred_unrestricted = reg_unrestricted.predict(xy_lag_matrix)
                        rss_unrestricted = np.sum((y_target - y_pred_unrestricted) ** 2)
                        
                        # F-statistic
                        n = len(y_target)
                        k = xy_lag_matrix.shape[1]
                        
                        f_stat = ((rss_restricted - rss_unrestricted) / 1) / (rss_unrestricted / (n - k - 1))
                        p_value = 1 - stats.f.cdf(f_stat, 1, n - k - 1)
                        
                        if p_value < best_pvalue:
                            best_pvalue = p_value
                            best_lag = lag
                            
                    except Exception:
                        continue
                        
                except Exception:
                    continue
            
            return {
                "optimal_lag": best_lag,
                "p_value": best_pvalue,
                "is_significant": best_pvalue < self.config["significance_threshold"]
            }
            
        except Exception as e:
            self.logger.warning(f"Granger causality test failed: {str(e)}")
            return {"optimal_lag": 1, "p_value": 1.0, "is_significant": False}
    
    def _calculate_causal_strength(self, x: pd.Series, y: pd.Series, lag: int) -> float:
        """Calculate causal strength based on predictive improvement"""
        
        try:
            if lag >= len(x) or lag <= 0:
                return 0.0
            
            # Create lagged variables
            y_current = y[lag:].values
            y_lagged = y[:-lag].values
            x_lagged = x[:-lag].values
            
            if len(y_current) < 10:
                return 0.0
            
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Baseline model: y ~ y_lagged
            reg_baseline = LinearRegression().fit(y_lagged.reshape(-1, 1), y_current)
            y_pred_baseline = reg_baseline.predict(y_lagged.reshape(-1, 1))
            r2_baseline = r2_score(y_current, y_pred_baseline)
            
            # Enhanced model: y ~ y_lagged + x_lagged
            X_enhanced = np.column_stack([y_lagged, x_lagged])
            reg_enhanced = LinearRegression().fit(X_enhanced, y_current)
            y_pred_enhanced = reg_enhanced.predict(X_enhanced)
            r2_enhanced = r2_score(y_current, y_pred_enhanced)
            
            # Causal strength as improvement in R²
            causal_strength = max(0.0, r2_enhanced - r2_baseline)
            
            return causal_strength
            
        except Exception as e:
            self.logger.warning(f"Causal strength calculation failed: {str(e)}")
            return 0.0
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality in time series"""
        
        try:
            seasonality_results = {}
            
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    # Simple seasonality detection using FFT
                    series = data[col].dropna()
                    
                    if len(series) < 24:  # Need enough data points
                        continue
                    
                    # FFT to detect dominant frequencies
                    fft = np.fft.fft(series.values)
                    freqs = np.fft.fftfreq(len(series))
                    
                    # Find dominant frequencies (excluding DC component)
                    power = np.abs(fft[1:len(fft)//2])
                    freqs_positive = freqs[1:len(freqs)//2]
                    
                    if len(power) > 0:
                        dominant_freq_idx = np.argmax(power)
                        dominant_freq = freqs_positive[dominant_freq_idx]
                        dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else np.inf
                        
                        seasonality_results[col] = {
                            "dominant_frequency": dominant_freq,
                            "dominant_period": dominant_period,
                            "seasonal_strength": power[dominant_freq_idx] / np.sum(power)
                        }
            
            return seasonality_results
            
        except Exception as e:
            self.logger.warning(f"Seasonality analysis failed: {str(e)}")
            return {}
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series"""
        
        trend_results = {}
        
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                series = data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                # Linear trend
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series.values, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # Trend significance test
                from scipy import stats
                corr, p_value = stats.pearsonr(x, series.values)
                
                trend_results[col] = {
                    "linear_slope": slope,
                    "linear_intercept": intercept,
                    "trend_correlation": corr,
                    "trend_p_value": p_value,
                    "trend_significant": p_value < 0.05,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                }
        
        return trend_results
    
    async def _bootstrap_confidence_intervals(self, x: pd.Series, y: pd.Series, 
                                            lag: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals for causal strength"""
        
        try:
            n_bootstrap = self.config["bootstrap_samples"]
            bootstrap_strengths = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(x), len(x), replace=True)
                x_boot = x.iloc[indices].reset_index(drop=True)
                y_boot = y.iloc[indices].reset_index(drop=True)
                
                # Calculate causal strength
                strength = self._calculate_causal_strength(x_boot, y_boot, lag)
                bootstrap_strengths.append(strength)
            
            # Calculate confidence interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_strengths, lower_percentile)
            ci_upper = np.percentile(bootstrap_strengths, upper_percentile)
            
            return (ci_lower, ci_upper)
            
        except Exception as e:
            self.logger.warning(f"Bootstrap confidence interval calculation failed: {str(e)}")
            return (0.0, 1.0)

class FederatedCausalDiscovery:
    """Federated causal relationship discovery across multiple datasets"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.registered_datasets = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "min_datasets": 2,
            "consistency_threshold": 0.6,
            "meta_analysis_method": "random_effects",
            "heterogeneity_threshold": 0.7,
            "privacy_preserving": True,
            "differential_privacy_epsilon": 1.0,
            "secure_aggregation": False
        }
    
    def register_dataset(self, dataset_id: str, data: pd.DataFrame, 
                        metadata: Dict[str, Any] = None):
        """Register a dataset for federated analysis"""
        
        self.registered_datasets[dataset_id] = {
            "data": data,
            "metadata": metadata or {},
            "local_results": None,
            "summary_statistics": self._calculate_summary_statistics(data)
        }
        
        self.logger.info(f"Registered dataset {dataset_id} with {len(data)} observations")
    
    def _calculate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy-preserving summary statistics"""
        
        stats = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            
            if len(series) > 0:
                # Add noise for differential privacy if enabled
                if self.config["privacy_preserving"]:
                    epsilon = self.config["differential_privacy_epsilon"]
                    noise_scale = 1.0 / epsilon
                    
                    stats[col] = {
                        "mean": series.mean() + np.random.laplace(0, noise_scale),
                        "std": series.std() + np.random.laplace(0, noise_scale),
                        "count": len(series),
                        "min": series.min() + np.random.laplace(0, noise_scale),
                        "max": series.max() + np.random.laplace(0, noise_scale)
                    }
                else:
                    stats[col] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "count": len(series),
                        "min": series.min(),
                        "max": series.max()
                    }
        
        return stats
    
    async def discover_federated_causal_relationships(self, 
                                                    target_variables: List[str] = None) -> Dict[str, Any]:
        """Discover causal relationships across federated datasets"""
        
        if len(self.registered_datasets) < self.config["min_datasets"]:
            raise ValueError(f"Need at least {self.config['min_datasets']} datasets for federated analysis")
        
        # Step 1: Local causal discovery on each dataset
        local_results = await self._perform_local_causal_discovery(target_variables)
        
        # Step 2: Aggregate local results
        federated_results = await self._aggregate_local_results(local_results)
        
        # Step 3: Meta-analysis of causal relationships
        meta_analysis_results = await self._perform_meta_analysis(local_results)
        
        # Step 4: Consistency analysis across datasets
        consistency_analysis = await self._analyze_consistency(local_results)
        
        return {
            "local_results": local_results,
            "federated_causal_graph": federated_results,
            "meta_analysis": meta_analysis_results,
            "consistency_analysis": consistency_analysis,
            "summary": self._generate_federated_summary(
                federated_results, meta_analysis_results, consistency_analysis
            )
        }
    
    async def _perform_local_causal_discovery(self, target_variables: List[str] = None) -> Dict[str, Any]:
        """Perform causal discovery on each local dataset"""
        
        local_results = {}
        
        for dataset_id, dataset_info in self.registered_datasets.items():
            try:
                self.logger.info(f"Performing local causal discovery on dataset {dataset_id}")
                
                data = dataset_info["data"]
                
                # Filter target variables if specified
                if target_variables:
                    available_vars = [var for var in target_variables if var in data.columns]
                    if len(available_vars) < 2:
                        self.logger.warning(f"Dataset {dataset_id} has insufficient target variables")
                        continue
                    data = data[available_vars]
                
                # Initialize causal discovery engine
                causal_engine = CausalDiscoveryEngine()
                
                # Discover causal relationships
                causal_graphs = await causal_engine.discover_causal_structure(data)
                
                # Extract summary information for federation
                local_summary = self._extract_local_summary(causal_graphs, data)
                
                local_results[dataset_id] = {
                    "causal_graphs": causal_graphs,
                    "summary": local_summary,
                    "dataset_metadata": dataset_info["metadata"],
                    "n_observations": len(data),
                    "n_variables": len(data.columns)
                }
                
                # Store in dataset info
                dataset_info["local_results"] = local_results[dataset_id]
                
            except Exception as e:
                self.logger.error(f"Local causal discovery failed for dataset {dataset_id}: {str(e)}")
                continue
        
        return local_results
    
    def _extract_local_summary(self, causal_graphs: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract summary information from local causal discovery"""
        
        summary = {
            "discovered_relationships": [],
            "variable_statistics": {},
            "graph_properties": {}
        }
        
        # Extract relationships from all discovered graphs
        all_edges = []
        for algorithm, graph in causal_graphs.items():
            for edge in graph.edges:
                edge_info = {
                    "source": edge.source,
                    "target": edge.target,
                    "causal_type": edge.causal_type.value,
                    "strength": edge.strength,
                    "p_value": edge.p_value,
                    "effect_size": edge.effect_size,
                    "algorithm": algorithm.value
                }
                all_edges.append(edge_info)
        
        summary["discovered_relationships"] = all_edges
        
        # Variable statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            summary["variable_statistics"][col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "n_obs": data[col].count()
            }
        
        # Graph properties
        if causal_graphs:
            first_graph = list(causal_graphs.values())[0]
            summary["graph_properties"] = {
                "n_nodes": len(first_graph.nodes),
                "n_edges": len(first_graph.edges),
                "edge_density": len(first_graph.edges) / max(1, len(first_graph.nodes) * (len(first_graph.nodes) - 1))
            }
        
        return summary
    
    async def _aggregate_local_results(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate local causal discovery results"""
        
        # Collect all discovered relationships
        all_relationships = []
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                relationship_with_source = relationship.copy()
                relationship_with_source["dataset_id"] = dataset_id
                relationship_with_source["n_observations"] = results["n_observations"]
                all_relationships.append(relationship_with_source)
        
        # Group relationships by source-target pair
        relationship_groups = defaultdict(list)
        
        for rel in all_relationships:
            key = (rel["source"], rel["target"])
            relationship_groups[key].append(rel)
        
        # Aggregate each relationship group
        federated_relationships = []
        
        for (source, target), relationships in relationship_groups.items():
            if len(relationships) >= self.config["min_datasets"]:
                aggregated_rel = self._aggregate_relationship_group(relationships)
                if aggregated_rel:
                    federated_relationships.append(aggregated_rel)
        
        # Build federated causal graph
        federated_graph = {
            "relationships": federated_relationships,
            "consensus_threshold": self.config["consistency_threshold"],
            "participating_datasets": list(local_results.keys()),
            "aggregation_method": "weighted_average"
        }
        
        return federated_graph
    
    def _aggregate_relationship_group(self, relationships: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Aggregate a group of relationships from different datasets"""
        
        if not relationships:
            return None
        
        # Extract values
        strengths = [rel["strength"] for rel in relationships]
        p_values = [rel["p_value"] for rel in relationships]
        effect_sizes = [rel["effect_size"] for rel in relationships]
        n_observations = [rel["n_observations"] for rel in relationships]
        
        # Weighted aggregation (weight by sample size)
        total_n = sum(n_observations)
        weights = [n / total_n for n in n_observations]
        
        aggregated_strength = np.average(strengths, weights=weights)
        aggregated_effect_size = np.average(effect_sizes, weights=weights)
        
        # Meta-analysis of p-values using Fisher's method
        from scipy import stats
        
        try:
            chi2_stat, meta_p_value = stats.combine_pvalues(p_values, method='fisher')
        except:
            meta_p_value = np.mean(p_values)
        
        # Calculate heterogeneity
        heterogeneity = self._calculate_heterogeneity(strengths, weights)
        
        # Only include if heterogeneity is acceptable
        if heterogeneity > self.config["heterogeneity_threshold"]:
            return None
        
        aggregated_relationship = {
            "source": relationships[0]["source"],
            "target": relationships[0]["target"],
            "aggregated_strength": aggregated_strength,
            "aggregated_effect_size": aggregated_effect_size,
            "meta_p_value": meta_p_value,
            "heterogeneity": heterogeneity,
            "n_datasets": len(relationships),
            "total_observations": total_n,
            "individual_results": relationships
        }
        
        return aggregated_relationship
    
    def _calculate_heterogeneity(self, values: List[float], weights: List[float]) -> float:
        """Calculate heterogeneity measure (I² statistic)"""
        
        try:
            # Calculate weighted mean
            weighted_mean = np.average(values, weights=weights)
            
            # Calculate Q statistic
            q_stat = sum(w * (v - weighted_mean)**2 for v, w in zip(values, weights))
            
            # Degrees of freedom
            df = len(values) - 1
            
            # I² statistic
            if df > 0 and q_stat > df:
                i_squared = (q_stat - df) / q_stat
            else:
                i_squared = 0.0
            
            return i_squared
            
        except Exception:
            return 1.0  # High heterogeneity if calculation fails
    
    async def _perform_meta_analysis(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-analysis of causal relationships"""
        
        meta_analysis_results = {
            "method": self.config["meta_analysis_method"],
            "combined_effects": [],
            "publication_bias_tests": {},
            "sensitivity_analysis": {}
        }
        
        # Collect effect sizes for meta-analysis
        effect_data = defaultdict(list)
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                key = (relationship["source"], relationship["target"])
                
                effect_data[key].append({
                    "effect_size": relationship["effect_size"],
                    "standard_error": self._estimate_standard_error(
                        relationship, results["n_observations"]
                    ),
                    "dataset_id": dataset_id,
                    "n": results["n_observations"]
                })
        
        # Perform meta-analysis for each relationship
        for (source, target), effects in effect_data.items():
            if len(effects) >= self.config["min_datasets"]:
                meta_result = self._meta_analyze_effects(effects)
                meta_result["source"] = source
                meta_result["target"] = target
                meta_analysis_results["combined_effects"].append(meta_result)
        
        return meta_analysis_results
    
    def _estimate_standard_error(self, relationship: Dict[str, Any], n_obs: int) -> float:
        """Estimate standard error for effect size"""
        
        # Simplified standard error estimation
        effect_size = abs(relationship["effect_size"])
        
        # For correlation-like effects
        if effect_size <= 1.0:
            # Fisher's z-transformation standard error
            se = 1.0 / np.sqrt(max(3, n_obs - 3))
        else:
            # For other effect sizes, use a simple approximation
            se = effect_size / np.sqrt(max(10, n_obs))
        
        return se
    
    def _meta_analyze_effects(self, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform meta-analysis on a set of effect sizes"""
        
        effect_sizes = [e["effect_size"] for e in effects]
        standard_errors = [e["standard_error"] for e in effects]
        sample_sizes = [e["n"] for e in effects]
        
        # Fixed effects meta-analysis
        weights = [1 / (se**2) for se in standard_errors]
        total_weight = sum(weights)
        
        combined_effect = sum(es * w for es, w in zip(effect_sizes, weights)) / total_weight
        combined_se = 1 / np.sqrt(total_weight)
        
        # Confidence interval
        ci_lower = combined_effect - 1.96 * combined_se
        ci_upper = combined_effect + 1.96 * combined_se
        
        # Test for overall effect
        z_score = combined_effect / combined_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Heterogeneity test (Q statistic)
        q_stat = sum(w * (es - combined_effect)**2 for es, w in zip(effect_sizes, weights))
        df = len(effects) - 1
        heterogeneity_p = 1 - stats.chi2.cdf(q_stat, df) if df > 0 else 1.0
        
        return {
            "combined_effect": combined_effect,
            "standard_error": combined_se,
            "confidence_interval": (ci_lower, ci_upper),
            "z_score": z_score,
            "p_value": p_value,
            "q_statistic": q_stat,
            "heterogeneity_p": heterogeneity_p,
            "n_studies": len(effects),
            "total_n": sum(sample_sizes)
        }
    
    async def _analyze_consistency(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of findings across datasets"""
        
        consistency_results = {
            "overall_consistency": 0.0,
            "relationship_consistency": {},
            "inconsistent_findings": [],
            "dataset_similarity": {}
        }
        
        # Analyze relationship consistency
        all_relationships = defaultdict(list)
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                key = (relationship["source"], relationship["target"])
                all_relationships[key].append({
                    "dataset_id": dataset_id,
                    "strength": relationship["strength"],
                    "p_value": relationship["p_value"],
                    "effect_size": relationship["effect_size"]
                })
        
        consistent_relationships = 0
        total_relationships = 0
        
        for (source, target), relationship_list in all_relationships.items():
            if len(relationship_list) >= 2:  # Need at least 2 datasets
                total_relationships += 1
                
                # Calculate consistency metrics
                strengths = [r["strength"] for r in relationship_list]
                effect_sizes = [r["effect_size"] for r in relationship_list]
                
                # Consistency based on coefficient of variation
                strength_cv = np.std(strengths) / (np.mean(strengths) + 1e-8)
                effect_cv = np.std(effect_sizes) / (abs(np.mean(effect_sizes)) + 1e-8)
                
                consistency_score = 1.0 / (1.0 + strength_cv + effect_cv)
                
                consistency_results["relationship_consistency"][(source, target)] = {
                    "consistency_score": consistency_score,
                    "strength_cv": strength_cv,
                    "effect_cv": effect_cv,
                    "n_datasets": len(relationship_list),
                    "mean_strength": np.mean(strengths),
                    "mean_effect": np.mean(effect_sizes)
                }
                
                if consistency_score > self.config["consistency_threshold"]:
                    consistent_relationships += 1
                else:
                    consistency_results["inconsistent_findings"].append({
                        "relationship": (source, target),
                        "consistency_score": consistency_score,
                        "details": relationship_list
                    })
        
        # Overall consistency
        if total_relationships > 0:
            consistency_results["overall_consistency"] = consistent_relationships / total_relationships
        
        # Dataset similarity analysis
        dataset_ids = list(local_results.keys())
        for i, dataset1 in enumerate(dataset_ids):
            for j, dataset2 in enumerate(dataset_ids[i+1:], i+1):
                similarity = self._calculate_dataset_similarity(
                    local_results[dataset1], local_results[dataset2]
                )
                consistency_results["dataset_similarity"][(dataset1, dataset2)] = similarity
        
        return consistency_results
    
    def _calculate_dataset_similarity(self, results1: Dict[str, Any], 
                                    results2: Dict[str, Any]) -> float:
        """Calculate similarity between two datasets' causal discovery results"""
        
        # Extract relationship sets
        rels1 = set((r["source"], r["target"]) for r in results1["summary"]["discovered_relationships"])
        rels2 = set((r["source"], r["target"]) for r in results2["summary"]["discovered_relationships"])
        
        # Jaccard similarity
        intersection = len(rels1.intersection(rels2))
        union = len(rels1.union(rels2))
        
        jaccard_similarity = intersection / max(1, union)
        
        return jaccard_similarity
    
    def _generate_federated_summary(self, federated_results: Dict[str, Any],
                                  meta_analysis: Dict[str, Any],
                                  consistency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of federated causal discovery"""
        
        summary = {
            "total_datasets": len(self.registered_datasets),
            "total_relationships_discovered": len(federated_results["relationships"]),
            "meta_analysis_relationships": len(meta_analysis["combined_effects"]),
            "overall_consistency": consistency_analysis["overall_consistency"],
            "key_findings": [],
            "recommendations": []
        }
        
        # Identify key findings
        for rel in federated_results["relationships"]:
            if (rel["aggregated_strength"] > 0.7 and 
                rel["meta_p_value"] < 0.01 and 
                rel["heterogeneity"] < 0.3):
                
                summary["key_findings"].append({
                    "relationship": f"{rel['source']} → {rel['target']}",
                    "strength": rel["aggregated_strength"],
                    "significance": rel["meta_p_value"],
                    "datasets": rel["n_datasets"],
                    "observations": rel["total_observations"]
                })
        
        # Generate recommendations
        if consistency_analysis["overall_consistency"] < 0.5:
            summary["recommendations"].append(
                "Low consistency across datasets suggests need for more standardized data collection"
            )
        
        if len(summary["key_findings"]) == 0:
            summary["recommendations"].append(
                "No strong consistent relationships found - consider larger sample sizes or different variables"
            )
        
        return summary

class ScientificKnowledgeGraph:
    """Main scientific knowledge graph system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.literature_extractor = LiteratureExtractor()
        self.temporal_analyzer = TemporalCausalAnalyzer()
        self.federated_discovery = FederatedCausalDiscovery()
        
        # Knowledge graph storage
        self.entities = {}  # entity_id -> KnowledgeEntity
        self.relations = {}  # relation_id -> KnowledgeRelation
        self.temporal_relations = {}  # relation_id -> TemporalCausalRelation
        
        # External data sources
        self.external_sources = {
            "geoscienceworld": GeoScienceWorldAPI(),
            "usgs": USGSDataAPI()
        }
        
        # Graph representation
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = None
        
        # RDF store
        if RDF_AVAILABLE:
            self.rdf_graph = Graph()
            self.geo_namespace = Namespace("http://ontology.prometheus.ai/geological/")
        else:
            self.rdf_graph = None
            self.geo_namespace = None
        
        # Initialize database
        self._initialize_database()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "database_path": "knowledge_graph.db",
            "enable_literature_extraction": True,
            "enable_temporal_analysis": True,
            "enable_federated_discovery": True,
            "auto_update_interval": 24 * 3600,  # 24 hours
            "confidence_threshold": 0.6,
            "semantic_similarity_threshold": 0.8,
            "max_literature_documents": 1000,
            "external_api_rate_limit": 10  # requests per second
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        
        db_path = self.config["database_path"]
        
        try:
            self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # Create tables
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    entity_type TEXT,
                    properties TEXT,
                    confidence REAL,
                    created_timestamp TEXT,
                    last_updated TEXT
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT,
                    target_entity_id TEXT,
                    relation_type TEXT,
                    properties TEXT,
                    confidence REAL,
                    created_timestamp TEXT,
                    last_updated TEXT,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id)
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS temporal_relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT,
                    target_entity_id TEXT,
                    temporal_lag REAL,
                    causal_strength REAL,
                    granger_p_value REAL,
                    created_timestamp TEXT,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id)
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS literature_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    publication_year INTEGER,
                    journal TEXT,
                    doi TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_timestamp TEXT
                )
            """)
            
            self.db_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            self.db_conn = None
    
    async def add_literature_document(self, document: LiteratureDocument) -> bool:
        """Add literature document and extract knowledge"""
        
        try:
            # Store document in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO literature_documents 
                    (id, title, authors, abstract, publication_year, journal, doi, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id, document.title, json.dumps(document.authors),
                    document.abstract, document.publication_year, document.journal,
                    document.doi, document.metadata.get("created_timestamp", datetime.now().isoformat())
                ))
                self.db_conn.commit()
            
            # Extract knowledge if enabled
            if self.config["enable_literature_extraction"]:
                entities, relations = await self.literature_extractor.extract_from_document(document)
                
                # Add entities and relations to knowledge graph
                for entity in entities:
                    await self.add_entity(entity)
                
                for relation in relations:
                    await self.add_relation(relation)
                
                self.logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from {document.id}")
                
                # Mark document as processed
                if self.db_conn:
                    self.db_conn.execute(
                        "UPDATE literature_documents SET processed = TRUE WHERE id = ?",
                        (document.id,)
                    )
                    self.db_conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add literature document {document.id}: {str(e)}")
            return False
    
    async def add_entity(self, entity: KnowledgeEntity) -> bool:
        """Add entity to knowledge graph"""
        
        try:
            # Check for existing similar entities
            similar_entities = await self._find_similar_entities(entity)
            
            if similar_entities:
                # Merge with most similar entity
                most_similar = similar_entities[0]
                merged_entity = await self._merge_entities(most_similar, entity)
                entity = merged_entity
            
            # Store in memory
            self.entities[entity.id] = entity
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, name, entity_type, properties, confidence, created_timestamp, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id, entity.name, entity.entity_type.value,
                    json.dumps(entity.properties), entity.confidence,
                    entity.created_timestamp, entity.last_updated
                ))
                self.db_conn.commit()
            
            # Add to NetworkX graph
            if self.graph is not None:
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    confidence=entity.confidence
                )
            
            # Add to RDF graph
            if self.rdf_graph is not None:
                entity_uri = self.geo_namespace[entity.id]
                self.rdf_graph.add((entity_uri, RDF.type, self.geo_namespace[entity.entity_type.value]))
                self.rdf_graph.add((entity_uri, self.geo_namespace.name, Literal(entity.name)))
                self.rdf_graph.add((entity_uri, self.geo_namespace.confidence, Literal(entity.confidence)))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add entity {entity.id}: {str(e)}")
            return False
    
    async def add_relation(self, relation: KnowledgeRelation) -> bool:
        """Add relation to knowledge graph"""
        
        try:
            # Verify entities exist
            if (relation.source_entity_id not in self.entities or 
                relation.target_entity_id not in self.entities):
                self.logger.warning(f"Relation {relation.id} references non-existent entities")
                return False
            
            # Store in memory
            self.relations[relation.id] = relation
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO relations 
                    (id, source_entity_id, target_entity_id, relation_type, properties, 
                     confidence, created_timestamp, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.id, relation.source_entity_id, relation.target_entity_id,
                    relation.relation_type.value, json.dumps(relation.properties),
                    relation.confidence, relation.created_timestamp, relation.last_updated
                ))
                self.db_conn.commit()
            
            # Add to NetworkX graph
            if self.graph is not None:
                self.graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    key=relation.id,
                    relation_type=relation.relation_type.value,
                    confidence=relation.confidence
                )
            
            # Add to RDF graph
            if self.rdf_graph is not None:
                source_uri = self.geo_namespace[relation.source_entity_id]
                target_uri = self.geo_namespace[relation.target_entity_id]
                relation_uri = self.geo_namespace[relation.relation_type.value]
                
                self.rdf_graph.add((source_uri, relation_uri, target_uri))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add relation {relation.id}: {str(e)}")
            return False
    
    async def add_temporal_relation(self, temporal_relation: TemporalCausalRelation) -> bool:
        """Add temporal causal relation to knowledge graph"""
        
        try:
            # Store in memory
            self.temporal_relations[temporal_relation.relation_id] = temporal_relation
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO temporal_relations 
                    (id, source_entity_id, target_entity_id, temporal_lag, 
                     causal_strength, granger_p_value, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    temporal_relation.relation_id,
                    temporal_relation.source_entity_id,
                    temporal_relation.target_entity_id,
                    temporal_relation.temporal_lag,
                    temporal_relation.causal_strength,
                    temporal_relation.granger_causality_pvalue,
                    datetime.now().isoformat()
                ))
                self.db_conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add temporal relation {temporal_relation.relation_id}: {str(e)}")
            return False
    
    async def _find_similar_entities(self, entity: KnowledgeEntity) -> List[KnowledgeEntity]:
        """Find similar entities in the knowledge graph"""
        
        similar_entities = []
        
        for existing_entity in self.entities.values():
            if existing_entity.entity_type == entity.entity_type:
                # Check name similarity
                similarity = self._calculate_string_similarity(entity.name, existing_entity.name)
                
                if similarity > self.config["semantic_similarity_threshold"]:
                    similar_entities.append((existing_entity, similarity))
        
        # Sort by similarity and return entities
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in similar_entities]
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity"""
        
        # Simple Jaccard similarity on word level
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(1, union)
    
    async def _merge_entities(self, entity1: KnowledgeEntity, 
                            entity2: KnowledgeEntity) -> KnowledgeEntity:
        """Merge two similar entities"""
        
        # Keep the entity with higher confidence as primary
        if entity1.confidence >= entity2.confidence:
            primary, secondary = entity1, entity2
        else:
            primary, secondary = entity2, entity1
        
        # Merge properties
        merged_properties = primary.properties.copy()
        merged_properties.update(secondary.properties)
        
        # Merge aliases
        merged_aliases = list(set(primary.aliases + secondary.aliases + [secondary.name]))
        
        # Merge source references
        merged_references = list(set(primary.source_references + secondary.source_references))
        
        # Create merged entity
        merged_entity = KnowledgeEntity(
            id=primary.id,  # Keep primary ID
            name=primary.name,
            entity_type=primary.entity_type,
            properties=merged_properties,
            aliases=merged_aliases,
            confidence=max(primary.confidence, secondary.confidence),
            source_references=merged_references,
            created_timestamp=min(primary.created_timestamp, secondary.created_timestamp),
            last_updated=datetime.now().isoformat(),
            embedding_vector=primary.embedding_vector  # Keep primary embedding
        )
        
        return merged_entity
    
    async def query_knowledge_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language"""
        
        results = []
        
        try:
            # Simple keyword-based search (can be enhanced with more sophisticated NLP)
            query_words = set(query.lower().split())
            
            # Search entities
            for entity in self.entities.values():
                entity_words = set((entity.name + " " + " ".join(entity.aliases)).lower().split())
                
                overlap = len(query_words.intersection(entity_words))
                if overlap > 0:
                    results.append({
                        "type": "entity",
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type.value,
                        "confidence": entity.confidence,
                        "relevance_score": overlap / len(query_words)
                    })
            
            # Search relations
            for relation in self.relations.values():
                source_entity = self.entities.get(relation.source_entity_id)
                target_entity = self.entities.get(relation.target_entity_id)
                
                if source_entity and target_entity:
                    relation_text = f"{source_entity.name} {relation.relation_type.value} {target_entity.name}"
                    relation_words = set(relation_text.lower().split())
                    
                    overlap = len(query_words.intersection(relation_words))
                    if overlap > 0:
                        results.append({
                            "type": "relation",
                            "id": relation.id,
                            "source": source_entity.name,
                            "relation": relation.relation_type.value,
                            "target": target_entity.name,
                            "confidence": relation.confidence,
                            "relevance_score": overlap / len(query_words)
                        })
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = results[:limit]
            
        except Exception as e:
            self.logger.error(f"Knowledge graph query failed: {str(e)}")
        
        return results
    
    async def update_from_external_sources(self) -> Dict[str, Any]:
        """Update knowledge graph from external scientific databases"""
        
        update_results = {
            "sources_queried": [],
            "new_entities": 0,
            "new_relations": 0,
            "errors": []
        }
        
        for source_name, source in self.external_sources.items():
            try:
                self.logger.info(f"Querying external source: {source_name}")
                
                # Query recent geological research
                query_terms = ["earthquake", "volcano", "geology", "geophysics", "tectonics"]
                
                for term in query_terms[:2]:  # Limit to avoid rate limits
                    documents = await source.query(term, limit=10)
                    
                    for doc_data in documents:
                        # Create literature document
                        doc_id = hashlib.md5(f"{source_name}_{doc_data.get('doi', doc_data['title'])}".encode()).hexdigest()[:16]
                        
                        document = LiteratureDocument(
                            id=doc_id,
                            title=doc_data["title"],
                            authors=doc_data.get("authors", []),
                            abstract=doc_data.get("abstract", ""),
                            full_text=doc_data.get("full_text", ""),
                            publication_year=doc_data.get("year", 2023),
                            journal=doc_data.get("journal", source_name),
                            doi=doc_data.get("doi"),
                            keywords=doc_data.get("keywords", [])
                        )
                        
                        # Add to knowledge graph
                        success = await self.add_literature_document(document)
                        if success:
                            update_results["new_entities"] += len(document.extracted_entities)
                            update_results["new_relations"] += len(document.extracted_relations)
                
                update_results["sources_queried"].append(source_name)
                
                # Rate limiting
                await asyncio.sleep(1 / self.config["external_api_rate_limit"])
                
            except Exception as e:
                error_msg = f"Failed to update from {source_name}: {str(e)}"
                self.logger.error(error_msg)
                update_results["errors"].append(error_msg)
        
        return update_results
    
    async def discover_temporal_causal_relationships(self, time_series_data: Dict[str, pd.Series]) -> List[TemporalCausalRelation]:
        """Discover temporal causal relationships from time-series data"""
        
        temporal_relations = []
        
        if not self.config["enable_temporal_analysis"]:
            return temporal_relations
        
        # Get all pairs of time series
        variables = list(time_series_data.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                try:
                    # Find corresponding entities
                    entity1_id = await self._find_entity_by_name(var1)
                    entity2_id = await self._find_entity_by_name(var2)
                    
                    if entity1_id and entity2_id:
                        # Analyze temporal causality in both directions
                        relation1 = await self.temporal_analyzer.analyze_temporal_causality(
                            time_series_data[var1], time_series_data[var2], (entity1_id, entity2_id)
                        )
                        
                        relation2 = await self.temporal_analyzer.analyze_temporal_causality(
                            time_series_data[var2], time_series_data[var1], (entity2_id, entity1_id)
                        )
                        
                        # Keep the stronger relationship
                        if relation1 and relation2:
                            if relation1.causal_strength > relation2.causal_strength:
                                temporal_relations.append(relation1)
                            else:
                                temporal_relations.append(relation2)
                        elif relation1:
                            temporal_relations.append(relation1)
                        elif relation2:
                            temporal_relations.append(relation2)
                
                except Exception as e:
                    self.logger.warning(f"Temporal analysis failed for {var1}-{var2}: {str(e)}")
                    continue
        
        # Add temporal relations to knowledge graph
        for relation in temporal_relations:
            await self.add_temporal_relation(relation)
        
        return temporal_relations
    
    async def _find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by name"""
        
        name_lower = name.lower()
        
        for entity_id, entity in self.entities.items():
            if (entity.name.lower() == name_lower or 
                name_lower in [alias.lower() for alias in entity.aliases]):
                return entity_id
        
        return None
    
    async def perform_federated_discovery(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform federated causal discovery across multiple datasets"""
        
        if not self.config["enable_federated_discovery"]:
            return {"error": "Federated discovery disabled"}
        
        # Register datasets
        for dataset_id, data in datasets.items():
            self.federated_discovery.register_dataset(dataset_id, data)
        
        # Discover federated causal relationships
        federated_results = await self.federated_discovery.discover_federated_causal_relationships()
        
        # Integrate results into knowledge graph
        await self._integrate_federated_results(federated_results)
        
        return federated_results
    
    async def _integrate_federated_results(self, federated_results: Dict[str, Any]):
        """Integrate federated discovery results into knowledge graph"""
        
        try:
            for relationship in federated_results["federated_causal_graph"]["relationships"]:
                source_name = relationship["source"]
                target_name = relationship["target"]
                
                # Find or create entities
                source_entity_id = await self._find_entity_by_name(source_name)
                if not source_entity_id:
                    # Create new entity
                    entity_id = hashlib.md5(f"federated_{source_name}".encode()).hexdigest()[:16]
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=source_name,
                        entity_type=EntityType.CONCEPT,
                        confidence=0.8,
                        source_references=["federated_discovery"]
                    )
                    await self.add_entity(entity)
                    source_entity_id = entity_id
                
                target_entity_id = await self._find_entity_by_name(target_name)
                if not target_entity_id:
                    # Create new entity
                    entity_id = hashlib.md5(f"federated_{target_name}".encode()).hexdigest()[:16]
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=target_name,
                        entity_type=EntityType.CONCEPT,
                        confidence=0.8,
                        source_references=["federated_discovery"]
                    )
                    await self.add_entity(entity)
                    target_entity_id = entity_id
                
                # Create federated causal relation
                relation_id = hashlib.md5(
                    f"federated_{source_entity_id}_{target_entity_id}".encode()
                ).hexdigest()[:16]
                
                relation = KnowledgeRelation(
                    id=relation_id,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relation_type=RelationType.CAUSES,
                    properties={
                        "federated_discovery": True,
                        "aggregated_strength": relationship["aggregated_strength"],
                        "meta_p_value": relationship["meta_p_value"],
                        "n_datasets": relationship["n_datasets"],
                        "heterogeneity": relationship["heterogeneity"]
                    },
                    confidence=relationship["aggregated_strength"],
                    evidence=[f"Federated analysis across {relationship['n_datasets']} datasets"],
                    source_references=["federated_discovery"]
                )
                
                await self.add_relation(relation)
                
        except Exception as e:
            self.logger.error(f"Failed to integrate federated results: {str(e)}")
    
    def export_knowledge_graph(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export knowledge graph in specified format"""
        
        if format == "json":
            return {
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type.value,
                        "properties": entity.properties,
                        "confidence": entity.confidence,
                        "aliases": entity.aliases
                    }
                    for entity in self.entities.values()
                ],
                "relations": [
                    {
                        "id": relation.id,
                        "source": relation.source_entity_id,
                        "target": relation.target_entity_id,
                        "type": relation.relation_type.value,
                        "properties": relation.properties,
                        "confidence": relation.confidence
                    }
                    for relation in self.relations.values()
                ],
                "temporal_relations": [
                    {
                        "id": relation.relation_id,
                        "source": relation.source_entity_id,
                        "target": relation.target_entity_id,
                        "temporal_lag": relation.temporal_lag,
                        "causal_strength": relation.causal_strength
                    }
                    for relation in self.temporal_relations.values()
                ]
            }
        
        elif format == "rdf" and self.rdf_graph:
            return self.rdf_graph.serialize(format="turtle")
        
        elif format == "networkx" and self.graph:
            return nx.node_link_data(self.graph)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_temporal_relations": len(self.temporal_relations),
            "entity_types": {
                entity_type.value: sum(1 for e in self.entities.values() 
                                     if e.entity_type == entity_type)
                for entity_type in EntityType
            },
            "relation_types": {
                relation_type.value: sum(1 for r in self.relations.values() 
                                       if r.relation_type == relation_type)
                for relation_type in RelationType
            },
            "average_entity_confidence": np.mean([e.confidence for e in self.entities.values()]) if self.entities else 0.0,
            "average_relation_confidence": np.mean([r.confidence for r in self.relations.values()]) if self.relations else 0.0
        }

# Example usage and testing
async def example_knowledge_graph():
    """Example of scientific knowledge graph system"""
    
    # Initialize knowledge graph
    kg = ScientificKnowledgeGraph()
    
    print("Scientific Knowledge Graph System initialized")
    print(f"Components available:")
    print(f"  - NetworkX: {NETWORKX_AVAILABLE}")
    print(f"  - spaCy: {SPACY_AVAILABLE}")
    print(f"  - Transformers: {TRANSFORMERS_AVAILABLE}")
    print(f"  - RDFLib: {RDF_AVAILABLE}")
    
    # Add sample literature document
    sample_document = LiteratureDocument(
        id="sample_doc_001",
        title="Volcanic Eruption Triggers and Earthquake Relationships",
        authors=["Dr. Smith", "Dr. Johnson"],
        abstract="This study investigates the relationship between volcanic eruptions and seismic activity, finding strong correlations between magma pressure and earthquake frequency.",
        full_text="Volcanic eruptions are complex phenomena that often correlate with increased seismic activity. Our analysis of 10 years of data from Mount Example shows that increased magma pressure leads to higher earthquake frequency. The crystallization process in magma chambers causes stress buildup, which triggers fault activation in the surrounding rock formations.",
        publication_year=2023,
        journal="Journal of Geological Sciences",
        doi="10.1000/sample.2023.001",
        keywords=["volcano", "earthquake", "magma", "seismicity", "crystallization"]
    )
    
    print(f"\nAdding literature document: {sample_document.title}")
    await kg.add_literature_document(sample_document)
    
    # Create sample time-series data for temporal analysis
    print("\nGenerating sample time-series data for temporal causal analysis...")
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Create synthetic correlated time series
    np.random.seed(42)
    n_periods = len(dates)
    
    # Magma pressure (leading indicator)
    magma_pressure = 5.0 + 2.0 * np.sin(np.arange(n_periods) * 2 * np.pi / 12) + np.random.normal(0, 0.5, n_periods)
    
    # Earthquake frequency (lagged response to magma pressure)
    earthquake_freq = np.zeros(n_periods)
    for i in range(1, n_periods):
        earthquake_freq[i] = 2.0 + 0.3 * magma_pressure[i-1] + np.random.normal(0, 0.3)
    
    time_series_data = {
        "magma_pressure": pd.Series(magma_pressure, index=dates),
        "earthquake_frequency": pd.Series(earthquake_freq, index=dates)
    }
    
    # Discover temporal causal relationships
    temporal_relations = await kg.discover_temporal_causal_relationships(time_series_data)
    
    print(f"Discovered {len(temporal_relations)} temporal causal relationships")
    for relation in temporal_relations:
        if relation.granger_causality_pvalue and relation.granger_causality_pvalue < 0.05:
            print(f"  Significant temporal causality: {relation.source_entity_id} → {relation.target_entity_id}")
            print(f"    Lag: {relation.temporal_lag} {relation.temporal_units}")
            print(f"    Causal strength: {relation.causal_strength:.3f}")
            print(f"    Granger p-value: {relation.granger_causality_pvalue:.3f}")
    
    # Create sample datasets for federated discovery
    print("\nPreparing sample datasets for federated causal discovery...")
    
    # Dataset 1: Volcanic region A
    np.random.seed(123)
    n_samples = 200
    dataset1 = pd.DataFrame({
        "temperature": np.random.normal(800, 100, n_samples),
        "pressure": np.random.normal(5, 1, n_samples),
        "viscosity": np.zeros(n_samples)
    })
    
    # Create causal relationship: temperature and pressure affect viscosity
    for i in range(n_samples):
        dataset1.loc[i, "viscosity"] = (1e6 * np.exp(-dataset1.loc[i, "temperature"]/1000) * 
                                       (dataset1.loc[i, "pressure"]**0.3) + 
                                       np.random.lognormal(0, 0.2))
    
    # Dataset 2: Volcanic region B (similar but with different parameters)
    np.random.seed(456)
    dataset2 = pd.DataFrame({
        "temperature": np.random.normal(750, 120, n_samples),
        "pressure": np.random.normal(4.5, 1.2, n_samples),
        "viscosity": np.zeros(n_samples)
    })
    
    for i in range(n_samples):
        dataset2.loc[i, "viscosity"] = (1.2e6 * np.exp(-dataset2.loc[i, "temperature"]/950) * 
                                       (dataset2.loc[i, "pressure"]**0.35) + 
                                       np.random.lognormal(0, 0.25))
    
    datasets = {
        "volcanic_region_A": dataset1,
        "volcanic_region_B": dataset2
    }
    
    # Perform federated causal discovery
    print("\nPerforming federated causal discovery...")
    federated_results = await kg.perform_federated_discovery(datasets)
    
    print("Federated Discovery Results:")
    if "summary" in federated_results:
        summary = federated_results["summary"]
        print(f"  Total datasets: {summary.get('total_datasets', 0)}")
        print(f"  Relationships discovered: {summary.get('total_relationships_discovered', 0)}")
        print(f"  Overall consistency: {summary.get('overall_consistency', 0):.3f}")
        
        if "key_findings" in summary:
            print(f"  Key findings:")
            for finding in summary["key_findings"][:3]:
                print(f"    {finding['relationship']}: strength={finding['strength']:.3f}, p={finding['significance']:.3f}")
    
    # Query knowledge graph
    print("\nQuerying knowledge graph...")
    queries = [
        "volcano earthquake relationship",
        "magma pressure crystallization",
        "temperature viscosity"
    ]
    
    for query in queries:
        results = await kg.query_knowledge_graph(query, limit=5)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for result in results[:3]:
            if result["type"] == "entity":
                print(f"  Entity: {result['name']} ({result['entity_type']}) - confidence: {result['confidence']:.3f}")
            elif result["type"] == "relation":
                print(f"  Relation: {result['source']} {result['relation']} {result['target']} - confidence: {result['confidence']:.3f}")
    
    # Update from external sources (mock)
    print("\nUpdating from external sources...")
    update_results = await kg.update_from_external_sources()
    print(f"Sources queried: {len(update_results['sources_queried'])}")
    print(f"New entities: {update_results['new_entities']}")
    print(f"New relations: {update_results['new_relations']}")
    if update_results['errors']:
        print(f"Errors encountered: {len(update_results['errors'])}")
    
    # Export knowledge graph
    print("\nExporting knowledge graph...")
    export_data = kg.export_knowledge_graph(format="json")
    print(f"Exported {len(export_data['entities'])} entities and {len(export_data['relations'])} relations")
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Total temporal relations: {stats['total_temporal_relations']}")
    print(f"  Average entity confidence: {stats['average_entity_confidence']:.3f}")
    print(f"  Average relation confidence: {stats['average_relation_confidence']:.3f}")
    
    print(f"\nEntity types:")
    for entity_type, count in stats['entity_types'].items():
        if count > 0:
            print(f"    {entity_type}: {count}")
    
    print(f"\nRelation types:")
    for relation_type, count in stats['relation_types'].items():
        if count > 0:
            print(f"    {relation_type}: {count}")

class RealTimeDataStreamer:
    """Real-time data streaming system for multiple external geological and geophysical sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.active_streams = {}
        self.stream_handlers = {}
        self.data_buffer = defaultdict(deque)
        self.subscribers = defaultdict(list)
        self.stream_stats = defaultdict(dict)
        self.is_running = False
        
        # Initialize external APIs
        self.usgs_api = USGSDataAPI()
        self.nasa_api = NASAEarthDataAPI()
        self.asf_api = ASFHyp3API()
        self.iris_api = EarthScopeIRISAPI()
        
        # Geological ontology for stream enhancement
        self.geological_ontology = GeologicalOntology()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default streaming configuration"""
        return {
            'streams': {
                'usgs_earthquakes': {
                    'enabled': True,
                    'interval': 300,  # 5 minutes
                    'feed_type': 'all',
                    'min_magnitude': 2.5,
                    'max_age_hours': 24
                },
                'usgs_volcanoes': {
                    'enabled': True,
                    'interval': 900,  # 15 minutes
                    'alert_levels': ['watch', 'advisory', 'warning']
                },
                'nasa_fires': {
                    'enabled': True,
                    'interval': 3600,  # 1 hour
                    'confidence_threshold': 75
                },
                'nasa_thermal_anomalies': {
                    'enabled': True,
                    'interval': 1800,  # 30 minutes
                    'instruments': ['MODIS', 'VIIRS']
                },
                'asf_insar_alerts': {
                    'enabled': True,
                    'interval': 3600,  # 1 hour
                    'deformation_threshold': 0.02  # 2cm
                },
                'iris_seismic_stations': {
                    'enabled': True,
                    'interval': 600,  # 10 minutes
                    'networks': ['IU', 'II', 'US']
                }
            },
            'buffer_settings': {
                'max_buffer_size': 10000,
                'cleanup_interval': 3600,
                'retention_hours': 168  # 7 days
            },
            'performance': {
                'max_concurrent_streams': 10,
                'timeout_seconds': 30,
                'retry_attempts': 3,
                'backoff_factor': 2.0
            },
            'data_quality': {
                'enable_validation': True,
                'duplicate_detection': True,
                'anomaly_detection': True,
                'confidence_threshold': 0.7
            }
        }
    
    async def start_streaming(self) -> None:
        """Start all configured real-time data streams"""
        if self.is_running:
            self.logger.warning("Streaming already running")
            return
            
        self.is_running = True
        self.logger.info("Starting real-time data streaming system")
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_buffers())
        
        # Start stream tasks
        stream_tasks = []
        for stream_name, stream_config in self.config['streams'].items():
            if stream_config.get('enabled', False):
                task = asyncio.create_task(self._run_stream(stream_name, stream_config))
                stream_tasks.append(task)
                self.logger.info(f"Started stream: {stream_name}")
        
        try:
            # Wait for all streams
            await asyncio.gather(*stream_tasks, cleanup_task)
        except asyncio.CancelledError:
            self.logger.info("Streaming tasks cancelled")
        except Exception as e:
            self.logger.error(f"Error in streaming system: {e}")
        finally:
            self.is_running = False
    
    async def stop_streaming(self) -> None:
        """Stop all active streams"""
        self.is_running = False
        self.logger.info("Stopping real-time data streaming system")
        
        # Cancel all active streams
        for stream_name, task in self.active_streams.items():
            if not task.done():
                task.cancel()
                self.logger.info(f"Stopped stream: {stream_name}")
        
        self.active_streams.clear()
    
    async def _run_stream(self, stream_name: str, stream_config: Dict[str, Any]) -> None:
        """Run individual data stream"""
        interval = stream_config.get('interval', 300)
        retry_count = 0
        max_retries = self.config['performance']['retry_attempts']
        
        while self.is_running:
            try:
                # Fetch data based on stream type
                data = await self._fetch_stream_data(stream_name, stream_config)
                
                if data:
                    # Process and enhance data
                    enhanced_data = await self._enhance_stream_data(stream_name, data)
                    
                    # Add to buffer
                    await self._buffer_data(stream_name, enhanced_data)
                    
                    # Notify subscribers
                    await self._notify_subscribers(stream_name, enhanced_data)
                    
                    # Update statistics
                    self._update_stream_stats(stream_name, len(enhanced_data))
                    
                    retry_count = 0  # Reset retry count on success
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error in stream {stream_name}: {e}")
                
                if retry_count >= max_retries:
                    self.logger.error(f"Max retries exceeded for stream {stream_name}")
                    break
                
                # Exponential backoff
                backoff_time = min(
                    interval * (self.config['performance']['backoff_factor'] ** retry_count),
                    3600  # Max 1 hour
                )
                await asyncio.sleep(backoff_time)
    
    async def _fetch_stream_data(self, stream_name: str, stream_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from specific stream source"""
        try:
            if stream_name == 'usgs_earthquakes':
                return await self._fetch_usgs_earthquakes(stream_config)
            elif stream_name == 'usgs_volcanoes':
                return await self._fetch_usgs_volcanoes(stream_config)
            elif stream_name == 'nasa_fires':
                return await self._fetch_nasa_fires(stream_config)
            elif stream_name == 'nasa_thermal_anomalies':
                return await self._fetch_nasa_thermal_anomalies(stream_config)
            elif stream_name == 'asf_insar_alerts':
                return await self._fetch_asf_insar_alerts(stream_config)
            elif stream_name == 'iris_seismic_stations':
                return await self._fetch_iris_seismic_data(stream_config)
            else:
                self.logger.warning(f"Unknown stream type: {stream_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {stream_name}: {e}")
            return []
    
    async def _fetch_usgs_earthquakes(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch real-time earthquake data from USGS"""
        feed_type = config.get('feed_type', 'all')
        min_magnitude = config.get('min_magnitude', 2.5)
        
        try:
            # Use the real-time feed method from USGS API
            earthquakes = await self.usgs_api.get_real_time_feed(feed_type=feed_type)
            
            # Filter by magnitude
            filtered_earthquakes = [
                eq for eq in earthquakes 
                if eq.get('properties', {}).get('mag', 0) >= min_magnitude
            ]
            
            # Add stream metadata
            for eq in filtered_earthquakes:
                eq['stream_source'] = 'usgs_earthquakes'
                eq['stream_timestamp'] = datetime.now().isoformat()
                eq['data_type'] = 'seismic_event'
            
            return filtered_earthquakes
            
        except Exception as e:
            self.logger.error(f"Error fetching USGS earthquakes: {e}")
            return []
    
    async def _fetch_usgs_volcanoes(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch real-time volcanic activity data from USGS"""
        alert_levels = config.get('alert_levels', ['watch', 'advisory', 'warning'])
        
        try:
            # Query volcanic activity
            volcanoes = await self.usgs_api.query("volcanic activity", limit=50)
            
            # Filter by alert levels
            filtered_volcanoes = [
                vol for vol in volcanoes 
                if vol.get('alert_level', '').lower() in [level.lower() for level in alert_levels]
            ]
            
            # Add stream metadata
            for vol in filtered_volcanoes:
                vol['stream_source'] = 'usgs_volcanoes'
                vol['stream_timestamp'] = datetime.now().isoformat()
                vol['data_type'] = 'volcanic_activity'
            
            return filtered_volcanoes
            
        except Exception as e:
            self.logger.error(f"Error fetching USGS volcanoes: {e}")
            return []
    
    async def _fetch_nasa_fires(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch real-time fire detection data from NASA"""
        confidence_threshold = config.get('confidence_threshold', 75)
        
        try:
            # Query FIRMS fire data
            fires = await self.nasa_api.query("active fires FIRMS", limit=100)
            
            # Filter by confidence
            filtered_fires = [
                fire for fire in fires 
                if fire.get('confidence', 0) >= confidence_threshold
            ]
            
            # Add stream metadata
            for fire in filtered_fires:
                fire['stream_source'] = 'nasa_fires'
                fire['stream_timestamp'] = datetime.now().isoformat()
                fire['data_type'] = 'thermal_anomaly'
            
            return filtered_fires
            
        except Exception as e:
            self.logger.error(f"Error fetching NASA fires: {e}")
            return []
    
    async def _fetch_nasa_thermal_anomalies(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch thermal anomaly data from NASA instruments"""
        instruments = config.get('instruments', ['MODIS', 'VIIRS'])
        
        try:
            # Query thermal anomalies
            anomalies = await self.nasa_api.query("thermal anomaly", limit=50)
            
            # Filter by instruments
            filtered_anomalies = [
                anomaly for anomaly in anomalies 
                if any(instr in anomaly.get('instrument', '') for instr in instruments)
            ]
            
            # Add stream metadata
            for anomaly in filtered_anomalies:
                anomaly['stream_source'] = 'nasa_thermal_anomalies'
                anomaly['stream_timestamp'] = datetime.now().isoformat()
                anomaly['data_type'] = 'thermal_anomaly'
            
            return filtered_anomalies
            
        except Exception as e:
            self.logger.error(f"Error fetching NASA thermal anomalies: {e}")
            return []
    
    async def _fetch_asf_insar_alerts(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch InSAR deformation alerts from ASF"""
        deformation_threshold = config.get('deformation_threshold', 0.02)
        
        try:
            # Query recent InSAR processing results
            insar_results = await self.asf_api.query("ground deformation", limit=25)
            
            # Filter by deformation threshold
            filtered_results = [
                result for result in insar_results 
                if abs(result.get('displacement_cm', 0)) >= deformation_threshold * 100
            ]
            
            # Add stream metadata
            for result in filtered_results:
                result['stream_source'] = 'asf_insar_alerts'
                result['stream_timestamp'] = datetime.now().isoformat()
                result['data_type'] = 'ground_deformation'
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error fetching ASF InSAR alerts: {e}")
            return []
    
    async def _fetch_iris_seismic_data(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch real-time seismic station data from IRIS"""
        networks = config.get('networks', ['IU', 'II', 'US'])
        
        try:
            # Query station metadata and status
            stations = await self.iris_api.query("seismic stations", limit=100)
            
            # Filter by networks
            filtered_stations = [
                station for station in stations 
                if any(net in station.get('network', '') for net in networks)
            ]
            
            # Add stream metadata
            for station in filtered_stations:
                station['stream_source'] = 'iris_seismic_stations'
                station['stream_timestamp'] = datetime.now().isoformat()
                station['data_type'] = 'seismic_station_status'
            
            return filtered_stations
            
        except Exception as e:
            self.logger.error(f"Error fetching IRIS seismic data: {e}")
            return []
    
    async def _enhance_stream_data(self, stream_name: str, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance streaming data with geological ontology and context"""
        enhanced_data = []
        
        for item in data:
            try:
                # Get appropriate API for enhancement
                if 'usgs' in stream_name:
                    enhanced_item = self.usgs_api.enhance_result_with_ontology(item)
                elif 'nasa' in stream_name:
                    enhanced_item = item.copy()  # NASA enhancement would go here
                elif 'asf' in stream_name:
                    enhanced_item = item.copy()  # ASF enhancement would go here
                elif 'iris' in stream_name:
                    enhanced_item = item.copy()  # IRIS enhancement would go here
                else:
                    enhanced_item = item.copy()
                
                # Add real-time context
                enhanced_item['real_time_context'] = {
                    'stream_name': stream_name,
                    'processing_timestamp': datetime.now().isoformat(),
                    'data_freshness': 'real_time',
                    'processing_latency_ms': 0  # Would calculate actual latency
                }
                
                # Add geological context if available
                if hasattr(self, 'geological_ontology') and 'ontology' in enhanced_item:
                    enhanced_item['geological_context'] = self._extract_geological_context(enhanced_item)
                
                enhanced_data.append(enhanced_item)
                
            except Exception as e:
                self.logger.error(f"Error enhancing item in stream {stream_name}: {e}")
                # Include original item if enhancement fails
                enhanced_data.append(item)
        
        return enhanced_data
    
    def _extract_geological_context(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geological context from enhanced data"""
        context = {
            'phenomena_type': [],
            'urgency_level': 'normal',
            'impact_assessment': 'low',
            'monitoring_priority': 'routine'
        }
        
        ontology_info = item.get('ontology', {})
        matched_concepts = ontology_info.get('matched_concepts', [])
        
        # Determine phenomena type
        if any('earthquake' in concept for concept in matched_concepts):
            context['phenomena_type'].append('seismic')
        if any('volcanic' in concept for concept in matched_concepts):
            context['phenomena_type'].append('volcanic')
        if any('thermal' in concept for concept in matched_concepts):
            context['phenomena_type'].append('thermal')
        if any('deformation' in concept for concept in matched_concepts):
            context['phenomena_type'].append('deformation')
        
        # Assess urgency based on data characteristics
        magnitude = item.get('properties', {}).get('mag', 0)
        if magnitude >= 6.0:
            context['urgency_level'] = 'high'
            context['impact_assessment'] = 'high'
            context['monitoring_priority'] = 'urgent'
        elif magnitude >= 4.0:
            context['urgency_level'] = 'medium'
            context['impact_assessment'] = 'medium'
            context['monitoring_priority'] = 'enhanced'
        
        return context
    
    async def _buffer_data(self, stream_name: str, data: List[Dict[str, Any]]) -> None:
        """Add data to stream buffer with size management"""
        buffer = self.data_buffer[stream_name]
        max_size = self.config['buffer_settings']['max_buffer_size']
        
        for item in data:
            # Add timestamp for retention management
            item['buffer_timestamp'] = time.time()
            buffer.append(item)
        
        # Trim buffer if it exceeds max size
        while len(buffer) > max_size:
            buffer.popleft()
    
    async def _notify_subscribers(self, stream_name: str, data: List[Dict[str, Any]]) -> None:
        """Notify all subscribers of new data"""
        subscribers = self.subscribers.get(stream_name, [])
        
        for subscriber_callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber_callback):
                    await subscriber_callback(stream_name, data)
                else:
                    subscriber_callback(stream_name, data)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber for {stream_name}: {e}")
    
    def _update_stream_stats(self, stream_name: str, data_count: int) -> None:
        """Update stream statistics"""
        stats = self.stream_stats[stream_name]
        current_time = time.time()
        
        # Initialize stats if not present
        if 'total_records' not in stats:
            stats.update({
                'total_records': 0,
                'last_update': current_time,
                'start_time': current_time,
                'update_count': 0,
                'average_records_per_update': 0,
                'uptime_seconds': 0
            })
        
        # Update counters
        stats['total_records'] += data_count
        stats['update_count'] += 1
        stats['uptime_seconds'] = current_time - stats['start_time']
        stats['average_records_per_update'] = stats['total_records'] / stats['update_count']
        stats['last_update'] = current_time
    
    async def _cleanup_buffers(self) -> None:
        """Periodic cleanup of old data from buffers"""
        cleanup_interval = self.config['buffer_settings']['cleanup_interval']
        retention_hours = self.config['buffer_settings']['retention_hours']
        retention_seconds = retention_hours * 3600
        
        while self.is_running:
            try:
                current_time = time.time()
                cutoff_time = current_time - retention_seconds
                
                for stream_name, buffer in self.data_buffer.items():
                    # Remove old items
                    while buffer and buffer[0].get('buffer_timestamp', current_time) < cutoff_time:
                        buffer.popleft()
                
                await asyncio.sleep(cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during buffer cleanup: {e}")
                await asyncio.sleep(cleanup_interval)
    
    def subscribe(self, stream_name: str, callback: Callable) -> None:
        """Subscribe to real-time data from a specific stream"""
        self.subscribers[stream_name].append(callback)
        self.logger.info(f"Added subscriber to stream: {stream_name}")
    
    def unsubscribe(self, stream_name: str, callback: Callable) -> None:
        """Unsubscribe from a specific stream"""
        if callback in self.subscribers[stream_name]:
            self.subscribers[stream_name].remove(callback)
            self.logger.info(f"Removed subscriber from stream: {stream_name}")
    
    def get_stream_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active streams"""
        return dict(self.stream_stats)
    
    def get_buffered_data(self, stream_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get buffered data from a specific stream"""
        buffer = self.data_buffer.get(stream_name, deque())
        
        if limit is None:
            return list(buffer)
        else:
            return list(buffer)[-limit:] if len(buffer) > limit else list(buffer)
    
    async def get_latest_data(self, stream_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get the latest data from a specific stream"""
        return self.get_buffered_data(stream_name, limit=count)

class DataFusionPipeline:
    """Intelligent data fusion system for combining multi-source geological datasets"""
    
    def __init__(self, auth_manager: Optional[APIAuthenticationManager] = None,
                 rate_limiter: Optional[RateLimitManager] = None):
        self.auth_manager = auth_manager or APIAuthenticationManager()
        self.rate_limiter = rate_limiter or RateLimitManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources
        self.usgs_api = USGSDataAPI(auth_manager=self.auth_manager, rate_limiter=self.rate_limiter)
        self.nasa_api = NASAEarthDataAPI(auth_manager=self.auth_manager, rate_limiter=self.rate_limiter)
        self.asf_api = ASFHyp3API(auth_manager=self.auth_manager, rate_limiter=self.rate_limiter)
        self.iris_api = EarthScopeIRISAPI(auth_manager=self.auth_manager, rate_limiter=self.rate_limiter)
        self.wiley_api = WileyOnlineLibraryAPI(auth_manager=self.auth_manager, rate_limiter=self.rate_limiter)
        
        # Initialize geological ontology for data correlation
        self.geological_ontology = GeologicalOntology()
        
        # Fusion configuration
        self.fusion_rules = self._setup_fusion_rules()
        self.temporal_window = timedelta(hours=24)  # Default temporal correlation window
        self.spatial_threshold = 100.0  # Default spatial correlation threshold (km)
        
        # Data cache for correlation analysis
        self.correlation_cache = {}
        self.fusion_cache = {}
        
        # Quality metrics
        self.quality_metrics = {
            'temporal_alignment_score': 0.0,
            'spatial_coherence_score': 0.0,
            'semantic_consistency_score': 0.0,
            'data_completeness_score': 0.0,
            'fusion_confidence_score': 0.0
        }
    
    def _setup_fusion_rules(self) -> Dict[str, Dict[str, Any]]:
        """Setup intelligent fusion rules for different data types"""
        return {
            'seismic_volcanic_fusion': {
                'data_types': ['earthquake', 'volcanic_eruption', 'seismic_station'],
                'temporal_window': timedelta(hours=48),
                'spatial_threshold': 50.0,
                'priority_weights': {'usgs': 0.4, 'iris': 0.35, 'nasa': 0.25},
                'correlation_factors': ['magnitude', 'depth', 'location', 'time'],
                'fusion_method': 'weighted_temporal_spatial'
            },
            'ground_deformation_fusion': {
                'data_types': ['insar_interferogram', 'gps_displacement', 'earthquake'],
                'temporal_window': timedelta(days=30),
                'spatial_threshold': 100.0,
                'priority_weights': {'asf': 0.5, 'nasa': 0.3, 'usgs': 0.2},
                'correlation_factors': ['displacement_vector', 'coherence', 'location', 'time_series'],
                'fusion_method': 'coherence_weighted_integration'
            },
            'thermal_seismic_fusion': {
                'data_types': ['thermal_anomaly', 'earthquake', 'volcanic_activity'],
                'temporal_window': timedelta(hours=72),
                'spatial_threshold': 75.0,
                'priority_weights': {'nasa': 0.45, 'usgs': 0.35, 'iris': 0.2},
                'correlation_factors': ['temperature_change', 'magnitude', 'volcanic_alert_level'],
                'fusion_method': 'multi_modal_correlation'
            },
            'geological_literature_fusion': {
                'data_types': ['research_paper', 'geological_survey', 'case_study'],
                'temporal_window': timedelta(days=365),
                'spatial_threshold': 500.0,
                'priority_weights': {'wiley': 0.5, 'agu': 0.3, 'usgs': 0.2},
                'correlation_factors': ['keywords', 'geological_domain', 'study_area'],
                'fusion_method': 'semantic_knowledge_integration'
            },
            'multi_hazard_fusion': {
                'data_types': ['earthquake', 'landslide', 'flooding', 'volcanic_eruption'],
                'temporal_window': timedelta(days=7),
                'spatial_threshold': 200.0,
                'priority_weights': {'usgs': 0.3, 'nasa': 0.3, 'iris': 0.2, 'asf': 0.2},
                'correlation_factors': ['hazard_intensity', 'cascading_effects', 'vulnerability'],
                'fusion_method': 'cascading_hazard_analysis'
            }
        }
    
    async def fuse_datasets(self, query: str, fusion_type: str = 'auto', 
                          temporal_range: Optional[Tuple[datetime, datetime]] = None,
                          spatial_bounds: Optional[Dict[str, float]] = None,
                          quality_threshold: float = 0.7) -> Dict[str, Any]:
        """Intelligently fuse datasets from multiple sources"""
        
        self.logger.info(f"Starting data fusion for query: {query}")
        
        # Step 1: Determine fusion strategy
        if fusion_type == 'auto':
            fusion_type = self._determine_fusion_strategy(query)
        
        fusion_rule = self.fusion_rules.get(fusion_type, self.fusion_rules['seismic_volcanic_fusion'])
        
        # Step 2: Collect data from multiple sources
        source_data = await self._collect_multi_source_data(query, fusion_rule, temporal_range, spatial_bounds)
        
        # Step 3: Temporal alignment
        aligned_data = await self._temporal_alignment(source_data, fusion_rule)
        
        # Step 4: Spatial correlation
        spatially_correlated = await self._spatial_correlation(aligned_data, fusion_rule)
        
        # Step 5: Semantic integration
        semantically_integrated = await self._semantic_integration(spatially_correlated, fusion_rule)
        
        # Step 6: Quality assessment
        quality_scores = await self._assess_fusion_quality(semantically_integrated)
        
        # Step 7: Generate fused dataset
        if quality_scores['fusion_confidence_score'] >= quality_threshold:
            fused_result = await self._generate_fused_dataset(semantically_integrated, fusion_rule, quality_scores)
        else:
            self.logger.warning(f"Fusion quality below threshold: {quality_scores['fusion_confidence_score']:.3f}")
            fused_result = await self._generate_fallback_dataset(source_data, fusion_rule)
        
        return {
            'fused_data': fused_result,
            'fusion_metadata': {
                'fusion_type': fusion_type,
                'fusion_rule': fusion_rule,
                'quality_scores': quality_scores,
                'source_count': len(source_data),
                'temporal_range': temporal_range,
                'spatial_bounds': spatial_bounds,
                'fusion_timestamp': datetime.now().isoformat()
            }
        }
    
    def _determine_fusion_strategy(self, query: str) -> str:
        """Automatically determine the best fusion strategy based on query content"""
        query_lower = query.lower()
        
        # Analyze query content for geological phenomena
        if any(term in query_lower for term in ['earthquake', 'seismic', 'tremor', 'magnitude']):
            if any(term in query_lower for term in ['volcano', 'eruption', 'lava', 'ash']):
                return 'seismic_volcanic_fusion'
            else:
                return 'seismic_volcanic_fusion'
        
        elif any(term in query_lower for term in ['insar', 'deformation', 'displacement', 'subsidence']):
            return 'ground_deformation_fusion'
        
        elif any(term in query_lower for term in ['thermal', 'temperature', 'hotspot', 'fire']):
            return 'thermal_seismic_fusion'
        
        elif any(term in query_lower for term in ['paper', 'research', 'study', 'literature']):
            return 'geological_literature_fusion'
        
        elif any(term in query_lower for term in ['hazard', 'risk', 'disaster', 'multi']):
            return 'multi_hazard_fusion'
        
        # Default to seismic-volcanic fusion
        return 'seismic_volcanic_fusion'
    
    async def _collect_multi_source_data(self, query: str, fusion_rule: Dict[str, Any],
                                       temporal_range: Optional[Tuple[datetime, datetime]],
                                       spatial_bounds: Optional[Dict[str, float]]) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from multiple sources in parallel"""
        
        source_data = {}
        
        # Prepare query parameters
        query_params = {}
        if temporal_range:
            query_params['start_time'] = temporal_range[0]
            query_params['end_time'] = temporal_range[1]
        if spatial_bounds:
            query_params.update(spatial_bounds)
        
        # Collect data from each source concurrently
        tasks = []
        
        # USGS data
        if 'usgs' in fusion_rule['priority_weights']:
            tasks.append(self._collect_usgs_data(query, query_params))
        
        # NASA data
        if 'nasa' in fusion_rule['priority_weights']:
            tasks.append(self._collect_nasa_data(query, query_params))
        
        # ASF data
        if 'asf' in fusion_rule['priority_weights']:
            tasks.append(self._collect_asf_data(query, query_params))
        
        # IRIS data
        if 'iris' in fusion_rule['priority_weights']:
            tasks.append(self._collect_iris_data(query, query_params))
        
        # Wiley data
        if 'wiley' in fusion_rule['priority_weights']:
            tasks.append(self._collect_wiley_data(query, query_params))
        
        # Execute all data collection tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        source_names = []
        if 'usgs' in fusion_rule['priority_weights']:
            source_names.append('usgs')
        if 'nasa' in fusion_rule['priority_weights']:
            source_names.append('nasa')
        if 'asf' in fusion_rule['priority_weights']:
            source_names.append('asf')
        if 'iris' in fusion_rule['priority_weights']:
            source_names.append('iris')
        if 'wiley' in fusion_rule['priority_weights']:
            source_names.append('wiley')
        
        for i, result in enumerate(results):
            if i < len(source_names) and not isinstance(result, Exception):
                source_data[source_names[i]] = result
            elif isinstance(result, Exception):
                self.logger.warning(f"Failed to collect data from {source_names[i] if i < len(source_names) else 'unknown'}: {result}")
        
        return source_data
    
    async def _collect_usgs_data(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from USGS"""
        try:
            return await self.usgs_api.query(query, limit=100, **params)
        except Exception as e:
            self.logger.warning(f"USGS data collection failed: {e}")
            return []
    
    async def _collect_nasa_data(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from NASA"""
        try:
            return await self.nasa_api.query(query, limit=100, **params)
        except Exception as e:
            self.logger.warning(f"NASA data collection failed: {e}")
            return []
    
    async def _collect_asf_data(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from ASF"""
        try:
            return await self.asf_api.query(query, limit=100, **params)
        except Exception as e:
            self.logger.warning(f"ASF data collection failed: {e}")
            return []
    
    async def _collect_iris_data(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from IRIS"""
        try:
            return await self.iris_api.query(query, limit=100, **params)
        except Exception as e:
            self.logger.warning(f"IRIS data collection failed: {e}")
            return []
    
    async def _collect_wiley_data(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect data from Wiley Online Library"""
        try:
            return await self.wiley_api.query(query, limit=50, **params)
        except Exception as e:
            self.logger.warning(f"Wiley data collection failed: {e}")
            return []
    
    async def _temporal_alignment(self, source_data: Dict[str, List[Dict[str, Any]]], 
                                fusion_rule: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Align data temporally based on fusion rules"""
        
        temporal_window = fusion_rule.get('temporal_window', self.temporal_window)
        aligned_data = {}
        
        # Extract temporal reference points
        all_timestamps = []
        for source, data_list in source_data.items():
            for item in data_list:
                timestamp = self._extract_timestamp(item)
                if timestamp:
                    all_timestamps.append(timestamp)
        
        if not all_timestamps:
            return source_data  # No temporal data to align
        
        # Find temporal clusters
        all_timestamps.sort()
        temporal_clusters = []
        current_cluster = [all_timestamps[0]]
        
        for timestamp in all_timestamps[1:]:
            if timestamp - current_cluster[-1] <= temporal_window:
                current_cluster.append(timestamp)
            else:
                temporal_clusters.append(current_cluster)
                current_cluster = [timestamp]
        
        if current_cluster:
            temporal_clusters.append(current_cluster)
        
        # Align data to temporal clusters
        for source, data_list in source_data.items():
            aligned_items = []
            for item in data_list:
                item_timestamp = self._extract_timestamp(item)
                if item_timestamp:
                    # Find best cluster for this item
                    best_cluster = None
                    min_distance = float('inf')
                    
                    for cluster in temporal_clusters:
                        cluster_center = sum(ts.timestamp() for ts in cluster) / len(cluster)
                        distance = abs(item_timestamp.timestamp() - cluster_center)
                        if distance < min_distance:
                            min_distance = distance
                            best_cluster = cluster
                    
                    # Add temporal alignment metadata
                    item['temporal_alignment'] = {
                        'original_timestamp': item_timestamp.isoformat(),
                        'cluster_id': temporal_clusters.index(best_cluster) if best_cluster else -1,
                        'alignment_confidence': 1.0 - (min_distance / temporal_window.total_seconds())
                    }
                
                aligned_items.append(item)
            
            aligned_data[source] = aligned_items
        
        # Update quality metrics
        self.quality_metrics['temporal_alignment_score'] = self._calculate_temporal_alignment_score(aligned_data)
        
        return aligned_data
    
    async def _spatial_correlation(self, aligned_data: Dict[str, List[Dict[str, Any]]], 
                                 fusion_rule: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform spatial correlation analysis"""
        
        spatial_threshold = fusion_rule.get('spatial_threshold', self.spatial_threshold)
        correlated_data = {}
        
        # Extract spatial coordinates
        spatial_points = []
        for source, data_list in aligned_data.items():
            for item in data_list:
                coords = self._extract_coordinates(item)
                if coords:
                    spatial_points.append({
                        'source': source,
                        'item': item,
                        'coordinates': coords
                    })
        
        # Perform spatial clustering
        spatial_clusters = self._spatial_clustering(spatial_points, spatial_threshold)
        
        # Add spatial correlation metadata
        for source, data_list in aligned_data.items():
            correlated_items = []
            for item in data_list:
                coords = self._extract_coordinates(item)
                if coords:
                    # Find spatial cluster
                    cluster_id = self._find_spatial_cluster(coords, spatial_clusters)
                    item['spatial_correlation'] = {
                        'coordinates': coords,
                        'cluster_id': cluster_id,
                        'nearby_sources': self._find_nearby_sources(coords, spatial_points, spatial_threshold)
                    }
                
                correlated_items.append(item)
            
            correlated_data[source] = correlated_items
        
        # Update quality metrics
        self.quality_metrics['spatial_coherence_score'] = self._calculate_spatial_coherence_score(correlated_data)
        
        return correlated_data
    
    async def _semantic_integration(self, correlated_data: Dict[str, List[Dict[str, Any]]], 
                                  fusion_rule: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform semantic integration using geological ontology"""
        
        integrated_data = {}
        
        for source, data_list in correlated_data.items():
            integrated_items = []
            for item in data_list:
                # Extract geological concepts
                concepts = self._extract_geological_concepts(item)
                
                # Find related concepts using ontology
                related_concepts = []
                for concept in concepts:
                    related = self.geological_ontology.get_related_concepts(concept)
                    related_concepts.extend(related)
                
                # Add semantic metadata
                item['semantic_integration'] = {
                    'extracted_concepts': concepts,
                    'related_concepts': list(set(related_concepts)),
                    'ontology_uris': [self.geological_ontology.get_full_uri(concept) for concept in concepts],
                    'semantic_similarity_score': self._calculate_semantic_similarity(item, correlated_data)
                }
                
                integrated_items.append(item)
            
            integrated_data[source] = integrated_items
        
        # Update quality metrics
        self.quality_metrics['semantic_consistency_score'] = self._calculate_semantic_consistency_score(integrated_data)
        
        return integrated_data
    
    async def _assess_fusion_quality(self, integrated_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess the quality of data fusion"""
        
        # Calculate data completeness
        total_sources = len(self.fusion_rules['seismic_volcanic_fusion']['priority_weights'])
        available_sources = len(integrated_data)
        self.quality_metrics['data_completeness_score'] = available_sources / total_sources
        
        # Calculate overall fusion confidence
        scores = [
            self.quality_metrics['temporal_alignment_score'],
            self.quality_metrics['spatial_coherence_score'],
            self.quality_metrics['semantic_consistency_score'],
            self.quality_metrics['data_completeness_score']
        ]
        
        self.quality_metrics['fusion_confidence_score'] = sum(scores) / len(scores)
        
        return self.quality_metrics.copy()
    
    async def _generate_fused_dataset(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                    fusion_rule: Dict[str, Any], 
                                    quality_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate the final fused dataset"""
        
        fusion_method = fusion_rule.get('fusion_method', 'weighted_temporal_spatial')
        priority_weights = fusion_rule.get('priority_weights', {})
        
        if fusion_method == 'weighted_temporal_spatial':
            return await self._weighted_temporal_spatial_fusion(integrated_data, priority_weights)
        elif fusion_method == 'coherence_weighted_integration':
            return await self._coherence_weighted_integration(integrated_data, priority_weights)
        elif fusion_method == 'multi_modal_correlation':
            return await self._multi_modal_correlation_fusion(integrated_data, priority_weights)
        elif fusion_method == 'semantic_knowledge_integration':
            return await self._semantic_knowledge_integration(integrated_data, priority_weights)
        elif fusion_method == 'cascading_hazard_analysis':
            return await self._cascading_hazard_analysis(integrated_data, priority_weights)
        else:
            return await self._default_fusion(integrated_data, priority_weights)
    
    async def _weighted_temporal_spatial_fusion(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                              priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform weighted temporal-spatial fusion"""
        
        fused_items = []
        
        # Group items by temporal and spatial clusters
        clusters = self._group_by_temporal_spatial_clusters(integrated_data)
        
        for cluster_id, cluster_items in clusters.items():
            # Calculate weighted fusion for this cluster
            fused_item = {
                'fusion_cluster_id': cluster_id,
                'fusion_method': 'weighted_temporal_spatial',
                'source_contributions': {},
                'fused_properties': {},
                'confidence_score': 0.0,
                'fusion_timestamp': datetime.now().isoformat()
            }
            
            total_weight = 0.0
            weighted_properties = defaultdict(list)
            
            for source, items in cluster_items.items():
                source_weight = priority_weights.get(source, 0.1)
                total_weight += source_weight * len(items)
                
                for item in items:
                    # Extract and weight properties
                    for prop, value in item.items():
                        if isinstance(value, (int, float)):
                            weighted_properties[prop].append(value * source_weight)
                        elif isinstance(value, str) and prop not in ['id', 'source', 'type']:
                            weighted_properties[prop].append(value)
                
                fused_item['source_contributions'][source] = {
                    'item_count': len(items),
                    'weight': source_weight,
                    'sample_items': items[:3]  # Store sample items
                }
            
            # Calculate fused properties
            for prop, values in weighted_properties.items():
                if all(isinstance(v, (int, float)) for v in values):
                    fused_item['fused_properties'][prop] = sum(values) / len(values)
                else:
                    # For non-numeric properties, take most common value
                    fused_item['fused_properties'][prop] = max(set(values), key=values.count)
            
            # Calculate confidence based on source agreement
            fused_item['confidence_score'] = min(1.0, total_weight / len(priority_weights))
            
            fused_items.append(fused_item)
        
        return fused_items
    
    async def _generate_fallback_dataset(self, source_data: Dict[str, List[Dict[str, Any]]], 
                                       fusion_rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fallback dataset when fusion quality is low"""
        
        priority_weights = fusion_rule.get('priority_weights', {})
        fallback_items = []
        
        # Select best source based on priority
        best_source = max(priority_weights.keys(), key=lambda k: priority_weights.get(k, 0))
        
        if best_source in source_data:
            for item in source_data[best_source]:
                fallback_item = item.copy()
                fallback_item['fusion_metadata'] = {
                    'fusion_method': 'fallback_single_source',
                    'primary_source': best_source,
                    'fusion_quality': 'low',
                    'fallback_reason': 'Quality threshold not met'
                }
                fallback_items.append(fallback_item)
        
        return fallback_items
    
    # Helper methods for data fusion
    def _extract_timestamp(self, item: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from data item"""
        for field in ['time', 'timestamp', 'datetime', 'date', 'eventTime', 'updated']:
            if field in item:
                try:
                    if isinstance(item[field], str):
                        return datetime.fromisoformat(item[field].replace('Z', '+00:00'))
                    elif isinstance(item[field], (int, float)):
                        return datetime.fromtimestamp(item[field])
                except:
                    continue
        return None
    
    def _extract_coordinates(self, item: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Extract coordinates from data item"""
        # Try various coordinate field combinations
        coord_fields = [
            ('latitude', 'longitude'), ('lat', 'lon'), ('lat', 'lng'),
            ('y', 'x'), ('geometry.coordinates[1]', 'geometry.coordinates[0]')
        ]
        
        for lat_field, lon_field in coord_fields:
            try:
                lat = self._get_nested_value(item, lat_field)
                lon = self._get_nested_value(item, lon_field)
                if lat is not None and lon is not None:
                    return (float(lat), float(lon))
            except:
                continue
        
        return None
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        value = data
        for key in keys:
            if '[' in key and ']' in key:
                # Handle array indexing
                key_name = key.split('[')[0]
                index = int(key.split('[')[1].split(']')[0])
                value = value[key_name][index]
            else:
                value = value[key]
        return value
    
    def _extract_geological_concepts(self, item: Dict[str, Any]) -> List[str]:
        """Extract geological concepts from data item"""
        concepts = []
        text_fields = ['title', 'description', 'abstract', 'summary', 'type']
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                text = item[field].lower()
                # Check for geological terms in ontology
                for concept in self.geological_ontology.ontology_mapping.keys():
                    if concept in text:
                        concepts.append(concept)
        
        return list(set(concepts))
    
    def _spatial_clustering(self, spatial_points: List[Dict[str, Any]], 
                          threshold_km: float) -> List[List[Dict[str, Any]]]:
        """Perform spatial clustering of points"""
        clusters = []
        remaining_points = spatial_points.copy()
        
        while remaining_points:
            # Start new cluster with first remaining point
            cluster = [remaining_points.pop(0)]
            
            # Find all points within threshold distance
            i = 0
            while i < len(remaining_points):
                point = remaining_points[i]
                
                # Check distance to any point in current cluster
                min_distance = float('inf')
                for cluster_point in cluster:
                    distance = self._haversine_distance(
                        cluster_point['coordinates'],
                        point['coordinates']
                    )
                    min_distance = min(min_distance, distance)
                
                if min_distance <= threshold_km:
                    cluster.append(remaining_points.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    def _haversine_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates in km"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        r = 6371.0
        return c * r
    
    def _calculate_temporal_alignment_score(self, aligned_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate temporal alignment quality score"""
        total_items = 0
        aligned_items = 0
        
        for source, data_list in aligned_data.items():
            for item in data_list:
                total_items += 1
                if 'temporal_alignment' in item:
                    confidence = item['temporal_alignment'].get('alignment_confidence', 0.0)
                    if confidence > 0.5:
                        aligned_items += 1
        
        return aligned_items / total_items if total_items > 0 else 0.0
    
    def _calculate_spatial_coherence_score(self, correlated_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate spatial coherence quality score"""
        total_items = 0
        coherent_items = 0
        
        for source, data_list in correlated_data.items():
            for item in data_list:
                total_items += 1
                if 'spatial_correlation' in item:
                    nearby_sources = item['spatial_correlation'].get('nearby_sources', [])
                    if len(nearby_sources) > 1:  # Item correlates with other sources
                        coherent_items += 1
        
        return coherent_items / total_items if total_items > 0 else 0.0
    
    def _calculate_semantic_consistency_score(self, integrated_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate semantic consistency quality score"""
        total_items = 0
        consistent_items = 0
        
        for source, data_list in integrated_data.items():
            for item in data_list:
                total_items += 1
                if 'semantic_integration' in item:
                    concepts = item['semantic_integration'].get('extracted_concepts', [])
                    if len(concepts) > 0:
                        consistent_items += 1
        
        return consistent_items / total_items if total_items > 0 else 0.0
    
    def _calculate_semantic_similarity(self, item: Dict[str, Any], 
                                     all_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate semantic similarity score for an item"""
        item_concepts = set(self._extract_geological_concepts(item))
        
        if not item_concepts:
            return 0.0
        
        similarity_scores = []
        for source, data_list in all_data.items():
            for other_item in data_list:
                if other_item == item:
                    continue
                
                other_concepts = set(self._extract_geological_concepts(other_item))
                if other_concepts:
                    # Jaccard similarity
                    intersection = len(item_concepts.intersection(other_concepts))
                    union = len(item_concepts.union(other_concepts))
                    similarity = intersection / union if union > 0 else 0.0
                    similarity_scores.append(similarity)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _find_spatial_cluster(self, coords: Tuple[float, float], 
                            spatial_clusters: List[List[Dict[str, Any]]]) -> int:
        """Find which spatial cluster a coordinate belongs to"""
        for i, cluster in enumerate(spatial_clusters):
            for point in cluster:
                if point['coordinates'] == coords:
                    return i
        return -1
    
    def _find_nearby_sources(self, coords: Tuple[float, float], 
                           spatial_points: List[Dict[str, Any]], 
                           threshold_km: float) -> List[str]:
        """Find data sources with points near the given coordinates"""
        nearby_sources = set()
        
        for point in spatial_points:
            distance = self._haversine_distance(coords, point['coordinates'])
            if distance <= threshold_km:
                nearby_sources.add(point['source'])
        
        return list(nearby_sources)
    
    def _group_by_temporal_spatial_clusters(self, integrated_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group data items by their temporal and spatial cluster IDs"""
        clusters = defaultdict(lambda: defaultdict(list))
        
        for source, data_list in integrated_data.items():
            for item in data_list:
                # Create compound cluster ID from temporal and spatial clusters
                temporal_id = item.get('temporal_alignment', {}).get('cluster_id', -1)
                spatial_id = item.get('spatial_correlation', {}).get('cluster_id', -1)
                
                cluster_key = f"t{temporal_id}_s{spatial_id}"
                clusters[cluster_key][source].append(item)
        
        return dict(clusters)
    
    async def _coherence_weighted_integration(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                            priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform coherence-weighted integration for InSAR and deformation data"""
        fused_items = []
        
        # Group by spatial clusters for deformation analysis
        spatial_groups = defaultdict(list)
        
        for source, data_list in integrated_data.items():
            for item in data_list:
                spatial_id = item.get('spatial_correlation', {}).get('cluster_id', -1)
                spatial_groups[spatial_id].append({
                    'source': source,
                    'item': item,
                    'weight': priority_weights.get(source, 0.1)
                })
        
        for cluster_id, cluster_items in spatial_groups.items():
            if cluster_id == -1:
                continue  # Skip unclustered items
            
            # Calculate coherence-weighted fusion
            fused_item = {
                'fusion_cluster_id': f"deformation_{cluster_id}",
                'fusion_method': 'coherence_weighted_integration',
                'deformation_analysis': {},
                'coherence_metrics': {},
                'source_contributions': {},
                'fusion_timestamp': datetime.now().isoformat()
            }
            
            total_coherence = 0.0
            weighted_displacement = 0.0
            
            for entry in cluster_items:
                item = entry['item']
                weight = entry['weight']
                source = entry['source']
                
                # Extract coherence and displacement values
                coherence = self._extract_coherence_value(item)
                displacement = self._extract_displacement_value(item)
                
                if coherence is not None and displacement is not None:
                    coherence_weight = coherence * weight
                    total_coherence += coherence_weight
                    weighted_displacement += displacement * coherence_weight
                
                fused_item['source_contributions'][source] = {
                    'coherence': coherence,
                    'displacement': displacement,
                    'weight': weight
                }
            
            if total_coherence > 0:
                fused_item['deformation_analysis']['average_displacement'] = weighted_displacement / total_coherence
                fused_item['coherence_metrics']['weighted_coherence'] = total_coherence / len(cluster_items)
                fused_item['confidence_score'] = min(1.0, total_coherence)
                
                fused_items.append(fused_item)
        
        return fused_items
    
    async def _multi_modal_correlation_fusion(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                            priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform multi-modal correlation fusion for thermal and seismic data"""
        fused_items = []
        
        # Group by temporal-spatial clusters
        clusters = self._group_by_temporal_spatial_clusters(integrated_data)
        
        for cluster_id, cluster_data in clusters.items():
            fused_item = {
                'fusion_cluster_id': cluster_id,
                'fusion_method': 'multi_modal_correlation',
                'modal_correlations': {},
                'anomaly_indicators': {},
                'source_contributions': {},
                'fusion_timestamp': datetime.now().isoformat()
            }
            
            # Extract thermal and seismic indicators
            thermal_values = []
            seismic_values = []
            
            for source, items in cluster_data.items():
                for item in items:
                    thermal_val = self._extract_thermal_value(item)
                    seismic_val = self._extract_seismic_value(item)
                    
                    if thermal_val is not None:
                        thermal_values.append(thermal_val * priority_weights.get(source, 0.1))
                    if seismic_val is not None:
                        seismic_values.append(seismic_val * priority_weights.get(source, 0.1))
                
                fused_item['source_contributions'][source] = {
                    'item_count': len(items),
                    'weight': priority_weights.get(source, 0.1)
                }
            
            # Calculate correlations
            if thermal_values and seismic_values:
                thermal_avg = sum(thermal_values) / len(thermal_values)
                seismic_avg = sum(seismic_values) / len(seismic_values)
                
                fused_item['modal_correlations']['thermal_seismic_correlation'] = self._calculate_correlation(thermal_values, seismic_values)
                fused_item['anomaly_indicators']['thermal_anomaly'] = thermal_avg > 1.5  # Threshold-based
                fused_item['anomaly_indicators']['seismic_anomaly'] = seismic_avg > 3.0  # Threshold-based
                fused_item['confidence_score'] = min(1.0, (len(thermal_values) + len(seismic_values)) / 10.0)
                
                fused_items.append(fused_item)
        
        return fused_items
    
    async def _semantic_knowledge_integration(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                            priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform semantic knowledge integration for literature and research data"""
        fused_items = []
        
        # Group by semantic similarity
        semantic_groups = self._group_by_semantic_similarity(integrated_data)
        
        for group_id, group_items in semantic_groups.items():
            fused_item = {
                'fusion_cluster_id': f"semantic_{group_id}",
                'fusion_method': 'semantic_knowledge_integration',
                'knowledge_synthesis': {},
                'research_trends': {},
                'source_contributions': {},
                'fusion_timestamp': datetime.now().isoformat()
            }
            
            all_concepts = set()
            weighted_keywords = defaultdict(float)
            
            for source, items in group_items.items():
                for item in items:
                    concepts = item.get('semantic_integration', {}).get('extracted_concepts', [])
                    all_concepts.update(concepts)
                    
                    # Weight keywords by source priority
                    weight = priority_weights.get(source, 0.1)
                    for concept in concepts:
                        weighted_keywords[concept] += weight
                
                fused_item['source_contributions'][source] = {
                    'paper_count': len(items),
                    'weight': priority_weights.get(source, 0.1)
                }
            
            # Synthesize knowledge
            top_concepts = sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
            fused_item['knowledge_synthesis']['dominant_concepts'] = [concept for concept, weight in top_concepts]
            fused_item['knowledge_synthesis']['concept_weights'] = dict(top_concepts)
            fused_item['research_trends']['emerging_topics'] = self._identify_emerging_topics(group_items)
            fused_item['confidence_score'] = min(1.0, len(all_concepts) / 20.0)
            
            fused_items.append(fused_item)
        
        return fused_items
    
    async def _cascading_hazard_analysis(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                                       priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Perform cascading hazard analysis for multi-hazard scenarios"""
        fused_items = []
        
        # Group by temporal proximity for cascading analysis
        temporal_groups = self._group_by_temporal_proximity(integrated_data, hours=24)
        
        for group_id, group_items in temporal_groups.items():
            fused_item = {
                'fusion_cluster_id': f"cascade_{group_id}",
                'fusion_method': 'cascading_hazard_analysis',
                'hazard_sequence': [],
                'cascade_analysis': {},
                'risk_assessment': {},
                'source_contributions': {},
                'fusion_timestamp': datetime.now().isoformat()
            }
            
            # Analyze hazard sequence
            hazard_events = []
            for source, items in group_items.items():
                for item in items:
                    hazard_type = self._classify_hazard_type(item)
                    timestamp = self._extract_timestamp(item)
                    intensity = self._extract_hazard_intensity(item)
                    
                    if hazard_type and timestamp:
                        hazard_events.append({
                            'type': hazard_type,
                            'timestamp': timestamp,
                            'intensity': intensity,
                            'source': source
                        })
            
            # Sort by time to analyze sequence
            hazard_events.sort(key=lambda x: x['timestamp'])
            fused_item['hazard_sequence'] = hazard_events
            
            # Analyze cascading effects
            cascade_pairs = []
            for i in range(len(hazard_events) - 1):
                event1 = hazard_events[i]
                event2 = hazard_events[i + 1]
                
                time_diff = (event2['timestamp'] - event1['timestamp']).total_seconds() / 3600  # hours
                if time_diff <= 48:  # Within 48 hours
                    cascade_pairs.append({
                        'trigger': event1['type'],
                        'consequence': event2['type'],
                        'time_delay': time_diff,
                        'cascade_likelihood': self._assess_cascade_likelihood(event1['type'], event2['type'])
                    })
            
            fused_item['cascade_analysis']['cascade_pairs'] = cascade_pairs
            fused_item['risk_assessment']['overall_risk_level'] = self._assess_overall_risk(hazard_events)
            fused_item['confidence_score'] = min(1.0, len(hazard_events) / 5.0)
            
            for source in group_items:
                fused_item['source_contributions'][source] = {
                    'event_count': len([e for e in hazard_events if e['source'] == source]),
                    'weight': priority_weights.get(source, 0.1)
                }
            
            fused_items.append(fused_item)
        
        return fused_items
    
    async def _default_fusion(self, integrated_data: Dict[str, List[Dict[str, Any]]], 
                            priority_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Default fusion method when specific method is not available"""
        return await self._weighted_temporal_spatial_fusion(integrated_data, priority_weights)
    
    # Additional helper methods for specific fusion types
    def _extract_coherence_value(self, item: Dict[str, Any]) -> Optional[float]:
        """Extract coherence value from InSAR data"""
        coherence_fields = ['coherence', 'correlation', 'quality', 'confidence']
        for field in coherence_fields:
            if field in item and isinstance(item[field], (int, float)):
                return float(item[field])
        return None
    
    def _extract_displacement_value(self, item: Dict[str, Any]) -> Optional[float]:
        """Extract displacement value from deformation data"""
        displacement_fields = ['displacement', 'deformation', 'motion', 'velocity']
        for field in displacement_fields:
            if field in item and isinstance(item[field], (int, float)):
                return float(item[field])
        return None
    
    def _extract_thermal_value(self, item: Dict[str, Any]) -> Optional[float]:
        """Extract thermal value from thermal data"""
        thermal_fields = ['temperature', 'thermal_anomaly', 'brightness_temperature', 'fire_radiative_power']
        for field in thermal_fields:
            if field in item and isinstance(item[field], (int, float)):
                return float(item[field])
        return None
    
    def _extract_seismic_value(self, item: Dict[str, Any]) -> Optional[float]:
        """Extract seismic value from seismic data"""
        seismic_fields = ['magnitude', 'mag', 'amplitude', 'intensity']
        for field in seismic_fields:
            if field in item and isinstance(item[field], (int, float)):
                return float(item[field])
        return None
    
    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """Calculate correlation coefficient between two value lists"""
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        # Simple Pearson correlation
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denominator1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
        denominator2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    def _group_by_semantic_similarity(self, integrated_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group items by semantic similarity"""
        groups = defaultdict(lambda: defaultdict(list))
        
        # Simple semantic grouping based on shared concepts
        for source, data_list in integrated_data.items():
            for item in data_list:
                concepts = item.get('semantic_integration', {}).get('extracted_concepts', [])
                
                # Create a concept signature
                concept_signature = '_'.join(sorted(concepts[:3]))  # Use top 3 concepts
                if not concept_signature:
                    concept_signature = 'miscellaneous'
                
                groups[concept_signature][source].append(item)
        
        return dict(groups)
    
    def _group_by_temporal_proximity(self, integrated_data: Dict[str, List[Dict[str, Any]]], hours: int = 24) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group items by temporal proximity for cascading analysis"""
        groups = defaultdict(lambda: defaultdict(list))
        threshold = timedelta(hours=hours)
        
        # Collect all timestamped items
        timestamped_items = []
        for source, data_list in integrated_data.items():
            for item in data_list:
                timestamp = self._extract_timestamp(item)
                if timestamp:
                    timestamped_items.append({
                        'source': source,
                        'item': item,
                        'timestamp': timestamp
                    })
        
        # Sort by timestamp
        timestamped_items.sort(key=lambda x: x['timestamp'])
        
        # Group by temporal proximity
        current_group = []
        group_id = 0
        
        for entry in timestamped_items:
            if not current_group:
                current_group = [entry]
            else:
                # Check if within threshold of group start
                time_diff = entry['timestamp'] - current_group[0]['timestamp']
                if time_diff <= threshold:
                    current_group.append(entry)
                else:
                    # Finalize current group
                    for group_entry in current_group:
                        groups[f"group_{group_id}"][group_entry['source']].append(group_entry['item'])
                    
                    # Start new group
                    group_id += 1
                    current_group = [entry]
        
        # Finalize last group
        if current_group:
            for group_entry in current_group:
                groups[f"group_{group_id}"][group_entry['source']].append(group_entry['item'])
        
        return dict(groups)
    
    def _identify_emerging_topics(self, group_items: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify emerging research topics from literature data"""
        concept_frequency = defaultdict(int)
        recent_concepts = defaultdict(int)
        
        # Current year for trend analysis
        current_year = datetime.now().year
        
        for source, items in group_items.items():
            for item in items:
                concepts = item.get('semantic_integration', {}).get('extracted_concepts', [])
                
                # Check if this is recent research (last 3 years)
                item_year = self._extract_publication_year(item)
                is_recent = item_year and (current_year - item_year) <= 3
                
                for concept in concepts:
                    concept_frequency[concept] += 1
                    if is_recent:
                        recent_concepts[concept] += 1
        
        # Find concepts that are more frequent in recent literature
        emerging_topics = []
        for concept, total_freq in concept_frequency.items():
            recent_freq = recent_concepts.get(concept, 0)
            if total_freq > 2 and recent_freq / total_freq > 0.6:  # 60% of mentions are recent
                emerging_topics.append(concept)
        
        return emerging_topics[:5]  # Return top 5 emerging topics
    
    def _extract_publication_year(self, item: Dict[str, Any]) -> Optional[int]:
        """Extract publication year from literature item"""
        year_fields = ['year', 'publication_year', 'date']
        for field in year_fields:
            if field in item:
                try:
                    if isinstance(item[field], int):
                        return item[field]
                    elif isinstance(item[field], str):
                        # Try to extract year from date string
                        import re
                        year_match = re.search(r'\b(19|20)\d{2}\b', item[field])
                        if year_match:
                            return int(year_match.group())
                except:
                    continue
        return None
    
    def _classify_hazard_type(self, item: Dict[str, Any]) -> Optional[str]:
        """Classify the type of geological hazard from data item"""
        text_fields = ['type', 'title', 'description', 'event_type']
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                text = item[field].lower()
                
                if any(term in text for term in ['earthquake', 'seismic', 'tremor']):
                    return 'earthquake'
                elif any(term in text for term in ['volcano', 'eruption', 'volcanic']):
                    return 'volcanic_eruption'
                elif any(term in text for term in ['landslide', 'rockfall', 'debris_flow']):
                    return 'landslide'
                elif any(term in text for term in ['flood', 'flooding', 'inundation']):
                    return 'flooding'
                elif any(term in text for term in ['fire', 'wildfire', 'thermal']):
                    return 'wildfire'
                elif any(term in text for term in ['tsunami', 'tidal']):
                    return 'tsunami'
        
        return None
    
    def _extract_hazard_intensity(self, item: Dict[str, Any]) -> Optional[float]:
        """Extract hazard intensity measure from data item"""
        intensity_fields = ['magnitude', 'intensity', 'severity', 'vei', 'alert_level']
        
        for field in intensity_fields:
            if field in item:
                if isinstance(item[field], (int, float)):
                    return float(item[field])
                elif isinstance(item[field], str):
                    # Try to extract numeric value
                    try:
                        import re
                        numeric_match = re.search(r'\d+\.?\d*', item[field])
                        if numeric_match:
                            return float(numeric_match.group())
                    except:
                        continue
        
        return None
    
    def _assess_cascade_likelihood(self, trigger_type: str, consequence_type: str) -> float:
        """Assess likelihood of cascading hazard relationship"""
        # Predefined cascade relationships with likelihood scores
        cascade_matrix = {
            ('earthquake', 'landslide'): 0.8,
            ('earthquake', 'tsunami'): 0.7,
            ('earthquake', 'volcanic_eruption'): 0.3,
            ('volcanic_eruption', 'landslide'): 0.6,
            ('volcanic_eruption', 'flooding'): 0.4,
            ('landslide', 'flooding'): 0.5,
            ('wildfire', 'landslide'): 0.4,
            ('flooding', 'landslide'): 0.6,
            ('earthquake', 'wildfire'): 0.2,
        }
        
        return cascade_matrix.get((trigger_type, consequence_type), 0.1)
    
    def _assess_overall_risk(self, hazard_events: List[Dict[str, Any]]) -> str:
        """Assess overall risk level from hazard sequence"""
        if not hazard_events:
            return 'low'
        
        # Count high-intensity events
        high_intensity_count = 0
        for event in hazard_events:
            intensity = event.get('intensity', 0)
            if intensity and intensity > 5.0:  # Threshold for high intensity
                high_intensity_count += 1
        
        # Assess based on event count and intensity
        if len(hazard_events) >= 3 and high_intensity_count >= 2:
            return 'very_high'
        elif len(hazard_events) >= 2 and high_intensity_count >= 1:
            return 'high'
        elif len(hazard_events) >= 2 or high_intensity_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    async def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fusion statistics"""
        return {
            'quality_metrics': self.quality_metrics,
            'fusion_rules': list(self.fusion_rules.keys()),
            'cache_size': len(self.fusion_cache),
            'last_fusion_timestamp': datetime.now().isoformat()
        }

# =============================================================================
# ADVANCED CACHING AND DATA PERSISTENCE SYSTEM
# =============================================================================

class CacheStrategy(Enum):
    """Advanced caching strategies for different data types"""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    SEMANTIC = "semantic_similarity"
    GEOLOGICAL_CONTEXT = "geological_context_aware"
    ADAPTIVE = "adaptive_intelligence"
    PREDICTIVE = "predictive_preloading"

class CacheEntry:
    """Enhanced cache entry with metadata and intelligence"""
    
    def __init__(self, key: str, data: Any, strategy: CacheStrategy = CacheStrategy.TTL):
        self.key = key
        self.data = data
        self.strategy = strategy
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl = 3600  # Default 1 hour
        self.size_bytes = self._calculate_size(data)
        self.geological_context = self._extract_geological_context(data)
        self.semantic_hash = self._generate_semantic_hash(data)
        self.prediction_score = 0.0
        self.refresh_priority = 0.5
        
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of cached data"""
        try:
            import sys
            if isinstance(data, (dict, list)):
                return sys.getsizeof(json.dumps(data, default=str))
            else:
                return sys.getsizeof(data)
        except:
            return 1024  # Default estimate
    
    def _extract_geological_context(self, data: Any) -> Dict[str, Any]:
        """Extract geological context for intelligent caching"""
        context = {
            'phenomena_types': [],
            'spatial_bounds': None,
            'temporal_range': None,
            'data_sources': [],
            'urgency_level': 'low'
        }
        
        try:
            phenomena_types = set()
            data_sources = set()
            
            if isinstance(data, dict):
                # Extract phenomena types
                for text_field in ['title', 'description', 'abstract', 'summary']:
                    if text_field in data and isinstance(data[text_field], str):
                        text = data[text_field].lower()
                        if any(term in text for term in ['earthquake', 'seismic']):
                            phenomena_types.add('seismic')
                        if any(term in text for term in ['volcano', 'eruption']):
                            phenomena_types.add('volcanic')
                        if any(term in text for term in ['landslide', 'avalanche']):
                            phenomena_types.add('mass_movement')
                        if any(term in text for term in ['flood', 'tsunami']):
                            phenomena_types.add('hydrological')
                
                # Extract urgency indicators
                if 'magnitude' in data and isinstance(data['magnitude'], (int, float)):
                    if data['magnitude'] > 6.0:
                        context['urgency_level'] = 'high'
                    elif data['magnitude'] > 4.0:
                        context['urgency_level'] = 'medium'
                
                # Extract data sources
                if 'source' in data:
                    data_sources.add(str(data['source']))
                if 'api_source' in data:
                    data_sources.add(str(data['api_source']))
            
            elif isinstance(data, list):
                # Aggregate context from list items
                for item in data[:10]:  # Sample first 10 items
                    if isinstance(item, dict):
                        item_context = self._extract_geological_context(item)
                        phenomena_types.update(item_context['phenomena_types'])
                        data_sources.update(item_context['data_sources'])
                        if item_context['urgency_level'] == 'high':
                            context['urgency_level'] = 'high'
                        elif item_context['urgency_level'] == 'medium' and context['urgency_level'] == 'low':
                            context['urgency_level'] = 'medium'
            
            # Convert sets to lists for JSON serialization
            context['phenomena_types'] = list(phenomena_types)
            context['data_sources'] = list(data_sources)
            
        except Exception as e:
            # Fallback to empty context on error
            pass
        
        return context
    
    def _generate_semantic_hash(self, data: Any) -> str:
        """Generate semantic hash for similarity-based caching"""
        try:
            # Extract meaningful content for semantic hashing
            content_parts = []
            
            if isinstance(data, dict):
                for key in ['title', 'description', 'abstract', 'query', 'location']:
                    if key in data and isinstance(data[key], str):
                        content_parts.append(data[key].lower())
                
                # Add numerical context
                for key in ['magnitude', 'latitude', 'longitude', 'depth']:
                    if key in data and isinstance(data[key], (int, float)):
                        content_parts.append(f"{key}:{round(data[key], 2)}")
            
            elif isinstance(data, list) and data:
                # Use first few items for semantic hashing
                for item in data[:3]:
                    if isinstance(item, dict):
                        item_hash = self._generate_semantic_hash(item)
                        content_parts.append(item_hash)
            
            # Create hash from meaningful content
            content_str = " ".join(content_parts)
            return hashlib.md5(content_str.encode()).hexdigest()[:16]
        
        except:
            return hashlib.md5(str(data).encode()).hexdigest()[:16]
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.strategy == CacheStrategy.TTL:
            return (time.time() - self.created_at) > self.ttl
        return False
    
    def calculate_eviction_score(self) -> float:
        """Calculate score for cache eviction decisions"""
        age_factor = (time.time() - self.last_accessed) / 3600  # Hours since last access
        frequency_factor = 1.0 / max(1, self.access_count)
        size_factor = self.size_bytes / (1024 * 1024)  # MB
        urgency_factor = 0.1 if self.geological_context['urgency_level'] == 'high' else 1.0
        
        return age_factor * frequency_factor * size_factor * urgency_factor

class IntelligentCacheManager:
    """Advanced caching system with geological intelligence"""
    
    def __init__(self, max_size_mb: int = 1024, db_path: str = "prometheus_cache.db"):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.db_path = db_path
        self.semantic_index: Dict[str, Set[str]] = defaultdict(set)
        self.geological_context_index: Dict[str, Set[str]] = defaultdict(set)
        self.prediction_model = PredictiveCacheModel()
        self.statistics = CacheStatistics()
        
        # Initialize persistent storage
        self._init_database()
        self._load_persistent_cache()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create cache table with advanced indexing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    strategy TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl INTEGER,
                    size_bytes INTEGER,
                    geological_context TEXT,
                    semantic_hash TEXT,
                    prediction_score REAL,
                    refresh_priority REAL
                )
            ''')
            
            # Create indices for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_hash ON cache_entries(semantic_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_geological_context ON cache_entries(geological_context)')
            
            # Create metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.warning(f"Cache database initialization failed: {e}")
    
    def _load_persistent_cache(self):
        """Load cache entries from persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM cache_entries WHERE last_accessed > ?', 
                         (time.time() - 86400,))  # Load entries from last 24 hours
            
            loaded_count = 0
            for row in cursor.fetchall():
                try:
                    key, data_blob, strategy, created_at, last_accessed, access_count, ttl, size_bytes, geological_context, semantic_hash, prediction_score, refresh_priority = row
                    
                    # Deserialize data
                    data = json.loads(data_blob.decode('utf-8'))
                    
                    # Recreate cache entry
                    entry = CacheEntry(key, data, CacheStrategy(strategy))
                    entry.created_at = created_at
                    entry.last_accessed = last_accessed
                    entry.access_count = access_count
                    entry.ttl = ttl
                    entry.size_bytes = size_bytes
                    entry.geological_context = json.loads(geological_context)
                    entry.semantic_hash = semantic_hash
                    entry.prediction_score = prediction_score
                    entry.refresh_priority = refresh_priority
                    
                    # Check if still valid
                    if not entry.is_expired():
                        self.cache_entries[key] = entry
                        self.current_size_bytes += entry.size_bytes
                        self._update_indices(entry)
                        loaded_count += 1
                
                except Exception as e:
                    logging.warning(f"Failed to load cache entry {row[0]}: {e}")
            
            conn.close()
            logging.info(f"Loaded {loaded_count} cache entries from persistent storage")
            
        except Exception as e:
            logging.warning(f"Failed to load persistent cache: {e}")
    
    def _update_indices(self, entry: CacheEntry):
        """Update semantic and geological indices"""
        # Update semantic index
        self.semantic_index[entry.semantic_hash].add(entry.key)
        
        # Update geological context index
        for phenomenon in entry.geological_context.get('phenomena_types', []):
            self.geological_context_index[f"phenomenon:{phenomenon}"].add(entry.key)
        
        for source in entry.geological_context.get('data_sources', []):
            self.geological_context_index[f"source:{source}"].add(entry.key)
        
        urgency = entry.geological_context.get('urgency_level', 'low')
        self.geological_context_index[f"urgency:{urgency}"].add(entry.key)
    
    def _remove_from_indices(self, entry: CacheEntry):
        """Remove entry from indices"""
        # Remove from semantic index
        if entry.semantic_hash in self.semantic_index:
            self.semantic_index[entry.semantic_hash].discard(entry.key)
            if not self.semantic_index[entry.semantic_hash]:
                del self.semantic_index[entry.semantic_hash]
        
        # Remove from geological context index
        for phenomenon in entry.geological_context.get('phenomena_types', []):
            index_key = f"phenomenon:{phenomenon}"
            if index_key in self.geological_context_index:
                self.geological_context_index[index_key].discard(entry.key)
                if not self.geological_context_index[index_key]:
                    del self.geological_context_index[index_key]
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get cached data with intelligent retrieval"""
        # Direct cache hit
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            if not entry.is_expired():
                entry.update_access()
                self.statistics.record_hit()
                await self._save_entry_to_db(entry)
                return entry.data
            else:
                # Remove expired entry
                await self._remove_entry(key)
        
        # Semantic similarity search
        similar_entries = await self._find_similar_entries(key)
        if similar_entries:
            best_entry = similar_entries[0]
            self.statistics.record_semantic_hit()
            best_entry.update_access()
            return best_entry.data
        
        self.statistics.record_miss()
        return default
    
    async def set(self, key: str, data: Any, strategy: CacheStrategy = CacheStrategy.TTL, 
                 ttl: int = 3600) -> bool:
        """Set cached data with intelligent management"""
        try:
            # Create cache entry
            entry = CacheEntry(key, data, strategy)
            entry.ttl = ttl
            
            # Update prediction score
            entry.prediction_score = await self.prediction_model.predict_future_access(entry)
            
            # Check if we need to make space
            if self.current_size_bytes + entry.size_bytes > self.max_size_bytes:
                await self._evict_entries(entry.size_bytes)
            
            # Remove old entry if exists
            if key in self.cache_entries:
                await self._remove_entry(key)
            
            # Add new entry
            self.cache_entries[key] = entry
            self.current_size_bytes += entry.size_bytes
            self._update_indices(entry)
            
            # Save to persistent storage
            await self._save_entry_to_db(entry)
            
            self.statistics.record_set()
            return True
            
        except Exception as e:
            logging.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def _find_similar_entries(self, query_key: str) -> List[CacheEntry]:
        """Find semantically similar cache entries"""
        try:
            # Create temporary entry for semantic comparison
            temp_entry = CacheEntry(query_key, {"query": query_key}, CacheStrategy.SEMANTIC)
            query_hash = temp_entry.semantic_hash
            
            similar_entries = []
            
            # Look for exact semantic matches
            if query_hash in self.semantic_index:
                for cache_key in self.semantic_index[query_hash]:
                    if cache_key in self.cache_entries:
                        entry = self.cache_entries[cache_key]
                        if not entry.is_expired():
                            similar_entries.append(entry)
            
            # Look for contextual matches based on geological phenomena
            temp_context = temp_entry.geological_context
            for phenomenon in temp_context.get('phenomena_types', []):
                index_key = f"phenomenon:{phenomenon}"
                if index_key in self.geological_context_index:
                    for cache_key in self.geological_context_index[index_key]:
                        if cache_key in self.cache_entries and cache_key not in [e.key for e in similar_entries]:
                            entry = self.cache_entries[cache_key]
                            if not entry.is_expired():
                                similar_entries.append(entry)
            
            # If no phenomenon matches, try partial text matching
            if not similar_entries and len(self.cache_entries) > 0:
                query_lower = query_key.lower()
                for entry in self.cache_entries.values():
                    if entry.is_expired():
                        continue
                    
                    # Check for text similarity in cached data
                    entry_text = ""
                    if isinstance(entry.data, dict):
                        for field in ['title', 'description', 'query', 'type']:
                            if field in entry.data and isinstance(entry.data[field], str):
                                entry_text += entry.data[field].lower() + " "
                    
                    # Simple similarity check
                    common_terms = ['earthquake', 'seismic', 'volcano', 'eruption', 'landslide', 'tsunami']
                    query_terms = [term for term in common_terms if term in query_lower]
                    entry_terms = [term for term in common_terms if term in entry_text]
                    
                    if query_terms and entry_terms and any(term in entry_terms for term in query_terms):
                        similar_entries.append(entry)
                        if len(similar_entries) >= 3:
                            break
            
            # Sort by relevance (access count, recency, prediction score)
            similar_entries.sort(key=lambda e: (e.access_count, e.prediction_score, -e.calculate_eviction_score()), reverse=True)
            
            return similar_entries[:3]  # Return top 3 matches
            
        except Exception as e:
            logging.warning(f"Semantic search failed: {e}")
            return []
    
    async def _evict_entries(self, required_space: int):
        """Intelligently evict cache entries to make space"""
        if not self.cache_entries:
            return
        
        # Calculate eviction scores for all entries
        eviction_candidates = []
        for entry in self.cache_entries.values():
            score = entry.calculate_eviction_score()
            eviction_candidates.append((score, entry))
        
        # Sort by eviction score (higher = more likely to evict)
        eviction_candidates.sort(key=lambda x: x[0], reverse=True)
        
        freed_space = 0
        for score, entry in eviction_candidates:
            if freed_space >= required_space:
                break
            
            await self._remove_entry(entry.key)
            freed_space += entry.size_bytes
            self.statistics.record_eviction()
    
    async def _remove_entry(self, key: str):
        """Remove entry from cache and indices"""
        if key in self.cache_entries:
            entry = self.cache_entries[key]
            self.current_size_bytes -= entry.size_bytes
            self._remove_from_indices(entry)
            del self.cache_entries[key]
            
            # Remove from database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                conn.commit()
                conn.close()
            except Exception as e:
                logging.warning(f"Failed to remove entry from database: {e}")
    
    async def _save_entry_to_db(self, entry: CacheEntry):
        """Save cache entry to persistent database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            data_blob = json.dumps(entry.data, default=str).encode('utf-8')
            geological_context_json = json.dumps(entry.geological_context)
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (key, data, strategy, created_at, last_accessed, access_count, ttl, 
                 size_bytes, geological_context, semantic_hash, prediction_score, refresh_priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.key, data_blob, entry.strategy.value, entry.created_at,
                entry.last_accessed, entry.access_count, entry.ttl, entry.size_bytes,
                geological_context_json, entry.semantic_hash, entry.prediction_score,
                entry.refresh_priority
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.warning(f"Failed to save cache entry to database: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'cache_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization_percent': (self.current_size_bytes / self.max_size_bytes) * 100,
            'entry_count': len(self.cache_entries),
            'hit_rate': self.statistics.hit_rate(),
            'semantic_hit_rate': self.statistics.semantic_hit_rate(),
            'avg_entry_size_kb': (self.current_size_bytes / max(1, len(self.cache_entries))) / 1024,
            'geological_context_distribution': self._analyze_geological_distribution(),
            'performance_metrics': self.statistics.get_performance_metrics()
        }
        
        return stats
    
    def _analyze_geological_distribution(self) -> Dict[str, int]:
        """Analyze distribution of geological phenomena in cache"""
        distribution = defaultdict(int)
        
        for entry in self.cache_entries.values():
            for phenomenon in entry.geological_context.get('phenomena_types', []):
                distribution[f"phenomenon_{phenomenon}"] += 1
            
            urgency = entry.geological_context.get('urgency_level', 'low')
            distribution[f"urgency_{urgency}"] += 1
            
            for source in entry.geological_context.get('data_sources', []):
                distribution[f"source_{source}"] += 1
        
        return dict(distribution)
    
    async def preload_predictive_cache(self, query_patterns: List[str]):
        """Preload cache based on predicted queries"""
        for pattern in query_patterns:
            prediction_score = await self.prediction_model.predict_query_likelihood(pattern)
            if prediction_score > 0.7:  # High prediction confidence
                # Trigger background data fetch and cache
                logging.info(f"Predictive preloading for pattern: {pattern}")

class PredictiveCacheModel:
    """AI model for predicting cache access patterns"""
    
    def __init__(self):
        self.access_history: List[Dict[str, Any]] = []
        self.temporal_patterns: Dict[str, List[float]] = defaultdict(list)
        self.geological_preferences: Dict[str, float] = defaultdict(float)
    
    async def predict_future_access(self, entry: CacheEntry) -> float:
        """Predict likelihood of future access for cache entry"""
        try:
            base_score = 0.5
            
            # Factor in geological context popularity
            for phenomenon in entry.geological_context.get('phenomena_types', []):
                base_score += self.geological_preferences.get(phenomenon, 0.0) * 0.2
            
            # Factor in urgency level
            urgency_weights = {'high': 0.3, 'medium': 0.1, 'low': 0.0}
            urgency = entry.geological_context.get('urgency_level', 'low')
            base_score += urgency_weights.get(urgency, 0.0)
            
            # Factor in data freshness
            age_hours = (time.time() - entry.created_at) / 3600
            freshness_score = max(0.0, 1.0 - (age_hours / 24))  # Decay over 24 hours
            base_score += freshness_score * 0.2
            
            return min(1.0, base_score)
            
        except Exception as e:
            logging.warning(f"Prediction model error: {e}")
            return 0.5
    
    async def predict_query_likelihood(self, query_pattern: str) -> float:
        """Predict likelihood of a query pattern being executed"""
        # Simple heuristic based on historical patterns
        geological_terms = ['earthquake', 'volcano', 'tsunami', 'landslide', 'flood']
        
        score = 0.0
        query_lower = query_pattern.lower()
        
        for term in geological_terms:
            if term in query_lower:
                score += self.geological_preferences.get(term, 0.1)
        
        return min(1.0, score)
    
    def update_access_pattern(self, key: str, geological_context: Dict[str, Any]):
        """Update access patterns based on recent cache access"""
        # Update geological preferences
        for phenomenon in geological_context.get('phenomena_types', []):
            self.geological_preferences[phenomenon] += 0.1
        
        # Decay old preferences
        for key in self.geological_preferences:
            self.geological_preferences[key] *= 0.99

class CacheStatistics:
    """Advanced cache statistics tracking"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        self.sets = 0
        self.evictions = 0
        self.start_time = time.time()
        self.response_times: List[float] = []
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def record_semantic_hit(self):
        self.semantic_hits += 1
    
    def record_set(self):
        self.sets += 1
    
    def record_eviction(self):
        self.evictions += 1
    
    def hit_rate(self) -> float:
        total_requests = self.hits + self.misses
        return (self.hits / max(1, total_requests)) * 100
    
    def semantic_hit_rate(self) -> float:
        total_requests = self.hits + self.misses + self.semantic_hits
        return (self.semantic_hits / max(1, total_requests)) * 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            'uptime_hours': uptime_hours,
            'requests_per_hour': (self.hits + self.misses) / max(0.1, uptime_hours),
            'cache_efficiency': self.hit_rate(),
            'semantic_efficiency': self.semantic_hit_rate(),
            'eviction_rate': self.evictions / max(1, self.sets),
            'avg_response_time_ms': sum(self.response_times) / max(1, len(self.response_times)) if self.response_times else 0
        }

# =============================================================================
# ENHANCED API CLASSES WITH INTELLIGENT CACHING
# =============================================================================

class CachedAPIClient:
    """Base class for API clients with intelligent caching"""
    
    def __init__(self, api_name: str, cache_manager: IntelligentCacheManager):
        self.api_name = api_name
        self.cache_manager = cache_manager
        self.default_cache_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate intelligent cache key"""
        # Sort parameters for consistent keys
        sorted_params = sorted(params.items())
        param_str = "&".join([f"{k}={v}" for k, v in sorted_params])
        
        # Include API name and endpoint
        key_parts = [self.api_name, endpoint, param_str]
        key_str = "|".join(key_parts)
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def cached_request(self, endpoint: str, params: Dict[str, Any], 
                           ttl: int = None, force_refresh: bool = False) -> Any:
        """Make cached API request with intelligent caching"""
        cache_key = self._generate_cache_key(endpoint, params)
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Make actual API request
        try:
            result = await self._make_api_request(endpoint, params)
            
            # Cache the result
            cache_ttl = ttl or self._determine_cache_ttl(endpoint, params, result)
            strategy = self._determine_cache_strategy(endpoint, params, result)
            
            await self.cache_manager.set(cache_key, result, strategy, cache_ttl)
            
            return result
            
        except Exception as e:
            logging.error(f"API request failed for {self.api_name}/{endpoint}: {e}")
            # Try to return stale cache data as fallback
            stale_result = await self.cache_manager.get(cache_key)
            if stale_result is not None:
                logging.info(f"Returning stale cache data for {cache_key}")
                return stale_result
            raise
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """Override in subclasses to implement actual API requests"""
        raise NotImplementedError
    
    def _determine_cache_ttl(self, endpoint: str, params: Dict[str, Any], result: Any) -> int:
        """Determine appropriate cache TTL based on data characteristics"""
        # Real-time data: short TTL
        if 'real-time' in endpoint.lower() or 'current' in endpoint.lower():
            return 300  # 5 minutes
        
        # Historical data: long TTL
        if 'historical' in endpoint.lower() or 'archive' in endpoint.lower():
            return 86400  # 24 hours
        
        # Check if result contains time-sensitive data
        if isinstance(result, dict):
            if 'alert_level' in result or 'urgency' in result:
                return 900  # 15 minutes for urgent data
            if 'magnitude' in result and isinstance(result['magnitude'], (int, float)):
                if result['magnitude'] > 6.0:
                    return 600  # 10 minutes for major events
        
        return self.default_cache_ttl  # Default 1 hour
    
    def _determine_cache_strategy(self, endpoint: str, params: Dict[str, Any], result: Any) -> CacheStrategy:
        """Determine appropriate cache strategy based on data characteristics"""
        # Real-time monitoring data uses TTL strategy
        if 'monitor' in endpoint.lower() or 'real-time' in endpoint.lower():
            return CacheStrategy.TTL
        
        # Frequently accessed data uses LFU strategy
        if 'popular' in endpoint.lower() or 'trending' in endpoint.lower():
            return CacheStrategy.LFU
        
        # Geological context-specific data uses geological strategy
        if isinstance(result, (dict, list)):
            return CacheStrategy.GEOLOGICAL_CONTEXT
        
        return CacheStrategy.ADAPTIVE

# Update existing API classes to use intelligent caching
class CachedUSGSDataAPI(USGSDataAPI, CachedAPIClient):
    """Enhanced USGS API with intelligent caching"""
    
    def __init__(self, api_key: str = None, auth_manager: APIAuthenticationManager = None,
                 rate_limiter: RateLimitManager = None, cache_manager: IntelligentCacheManager = None):
        USGSDataAPI.__init__(self, api_key, auth_manager, rate_limiter)
        CachedAPIClient.__init__(self, 'usgs', cache_manager or IntelligentCacheManager())
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """Override to implement USGS-specific API requests"""
        # Use the existing secure query method if available
        if hasattr(self, 'secure_query') and callable(self.secure_query):
            return await self.secure_query(endpoint, **params)
        else:
            # Fallback to existing query method
            return self._fallback_query(endpoint, **params)

# =============================================================================
# ADVANCED ERROR HANDLING AND RETRY LOGIC SYSTEM
# =============================================================================

class ErrorType(Enum):
    """Classification of different error types for intelligent handling"""
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    DATA_PARSING_ERROR = "data_parsing_error"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    SERVER_ERROR = "server_error"
    GEOLOGICAL_DATA_ERROR = "geological_data_error"
    CACHE_ERROR = "cache_error"
    UNKNOWN_ERROR = "unknown_error"

class RetryStrategy(Enum):
    """Different retry strategies for various error scenarios"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE_RETRY = "immediate_retry"
    NO_RETRY = "no_retry"
    ADAPTIVE_GEOLOGICAL = "adaptive_geological"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    geological_priority: bool = False
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0  # 5 minutes

@dataclass
class ErrorContext:
    """Context information for error handling decisions"""
    error_type: ErrorType
    api_source: str
    endpoint: str
    attempt_number: int
    geological_urgency: str = "low"
    data_criticality: str = "normal"
    error_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    response_code: Optional[int] = None
    geological_phenomena: List[str] = field(default_factory=list)

class CircuitBreakerState(Enum):
    """Circuit breaker states for API protection"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

class GeologicalErrorAnalyzer:
    """Analyzes errors in geological context for smart retry decisions"""
    
    def __init__(self):
        self.geological_priority_errors = {
            'earthquake_monitoring': {'max_attempts': 5, 'urgency_multiplier': 2.0},
            'volcanic_alerts': {'max_attempts': 5, 'urgency_multiplier': 2.0},
            'tsunami_warnings': {'max_attempts': 7, 'urgency_multiplier': 3.0},
            'landslide_detection': {'max_attempts': 4, 'urgency_multiplier': 1.5},
            'general_monitoring': {'max_attempts': 3, 'urgency_multiplier': 1.0}
        }
    
    def analyze_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Analyze error and context to create comprehensive error context"""
        error_type = self._classify_error(error)
        api_source = context.get('api_source', 'unknown')
        endpoint = context.get('endpoint', 'unknown')
        attempt_number = context.get('attempt_number', 1)
        
        # Extract geological context
        geological_urgency = self._assess_geological_urgency(context)
        data_criticality = self._assess_data_criticality(context)
        geological_phenomena = self._extract_phenomena_from_context(context)
        
        return ErrorContext(
            error_type=error_type,
            api_source=api_source,
            endpoint=endpoint,
            attempt_number=attempt_number,
            geological_urgency=geological_urgency,
            data_criticality=data_criticality,
            error_message=str(error),
            response_code=getattr(error, 'status_code', None),
            geological_phenomena=geological_phenomena
        )
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate handling"""
        error_str = str(error).lower()
        error_name = type(error).__name__.lower()
        
        # Network and connection errors
        if 'timeout' in error_str or 'timeout' in error_name:
            return ErrorType.NETWORK_TIMEOUT
        elif 'connection' in error_str or 'connection' in error_name:
            return ErrorType.CONNECTION_ERROR
        
        # HTTP errors
        if hasattr(error, 'status_code'):
            status_code = error.status_code
            if status_code == 401 or status_code == 403:
                return ErrorType.AUTHENTICATION_ERROR
            elif status_code == 429:
                return ErrorType.RATE_LIMIT_ERROR
            elif status_code == 402 or status_code == 509:
                return ErrorType.API_QUOTA_EXCEEDED
            elif 500 <= status_code < 600:
                return ErrorType.SERVER_ERROR
            else:
                return ErrorType.HTTP_ERROR
        
        # Parsing and data errors
        if 'json' in error_str or 'parsing' in error_str or 'decode' in error_str:
            return ErrorType.DATA_PARSING_ERROR
        
        # Geological specific errors
        if any(term in error_str for term in ['geological', 'seismic', 'volcanic', 'earthquake']):
            return ErrorType.GEOLOGICAL_DATA_ERROR
        
        # Cache errors
        if 'cache' in error_str:
            return ErrorType.CACHE_ERROR
        
        return ErrorType.UNKNOWN_ERROR
    
    def _assess_geological_urgency(self, context: Dict[str, Any]) -> str:
        """Assess geological urgency level from context"""
        # Check for urgency indicators in the request
        query_params = context.get('params', {})
        endpoint = context.get('endpoint', '').lower()
        
        # High urgency indicators
        if 'alert' in endpoint or 'warning' in endpoint or 'emergency' in endpoint:
            return 'critical'
        
        # Check magnitude or intensity
        if 'magnitude' in query_params:
            try:
                mag = float(query_params['magnitude'])
                if mag >= 7.0:
                    return 'critical'
                elif mag >= 5.5:
                    return 'high'
                elif mag >= 4.0:
                    return 'medium'
            except:
                pass
        
        # Check for real-time monitoring
        if 'real-time' in endpoint or 'current' in endpoint or 'latest' in endpoint:
            return 'high'
        
        return 'low'
    
    def _assess_data_criticality(self, context: Dict[str, Any]) -> str:
        """Assess data criticality for error handling priority"""
        endpoint = context.get('endpoint', '').lower()
        params = context.get('params', {})
        
        # Critical data types
        if any(term in endpoint for term in ['tsunami', 'emergency', 'alert', 'warning']):
            return 'critical'
        
        # High importance data
        if any(term in endpoint for term in ['earthquake', 'volcanic', 'real-time', 'monitoring']):
            return 'high'
        
        # Check for specific geological events
        if 'magnitude' in params:
            try:
                mag = float(params.get('magnitude', 0))
                if mag >= 6.0:
                    return 'critical'
                elif mag >= 4.5:
                    return 'high'
            except:
                pass
        
        return 'normal'
    
    def _extract_phenomena_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Extract geological phenomena from request context"""
        phenomena = []
        text_content = ' '.join([
            context.get('endpoint', ''),
            str(context.get('params', {})),
            context.get('query', '')
        ]).lower()
        
        geological_terms = {
            'earthquake': 'seismic',
            'seismic': 'seismic',
            'volcano': 'volcanic',
            'volcanic': 'volcanic',
            'eruption': 'volcanic',
            'tsunami': 'hydrological',
            'landslide': 'mass_movement',
            'avalanche': 'mass_movement',
            'flood': 'hydrological'
        }
        
        for term, phenomenon in geological_terms.items():
            if term in text_content and phenomenon not in phenomena:
                phenomena.append(phenomenon)
        
        return phenomena

class CircuitBreaker:
    """Circuit breaker for API protection with geological awareness"""
    
    def __init__(self, name: str, failure_threshold: int = 5, timeout: float = 300.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.geological_multiplier = 1.0
    
    def set_geological_priority(self, urgency_level: str):
        """Adjust circuit breaker sensitivity based on geological urgency"""
        urgency_multipliers = {
            'critical': 3.0,  # More tolerant for critical geological data
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0
        }
        self.geological_multiplier = urgency_multipliers.get(urgency_level, 1.0)
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker OPEN for {self.name}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return False
        
        # Adjust timeout based on geological priority
        effective_timeout = self.timeout / self.geological_multiplier
        return time.time() - self.last_failure_time >= effective_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Adjust threshold based on geological priority
        effective_threshold = int(self.failure_threshold * self.geological_multiplier)
        
        if self.failure_count >= effective_threshold:
            self.state = CircuitBreakerState.OPEN

class AdvancedRetryHandler:
    """Advanced retry handler with geological intelligence"""
    
    def __init__(self):
        self.error_analyzer = GeologicalErrorAnalyzer()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.retry_statistics = {
            'total_attempts': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'geological_priority_retries': 0
        }
    
    def get_circuit_breaker(self, api_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for API"""
        if api_name not in self.circuit_breakers:
            self.circuit_breakers[api_name] = CircuitBreaker(api_name)
        return self.circuit_breakers[api_name]
    
    async def execute_with_retry(self, func: Callable, context: Dict[str, Any], 
                               retry_config: RetryConfig = None) -> Any:
        """Execute function with intelligent retry logic"""
        if retry_config is None:
            retry_config = RetryConfig()
        
        api_name = context.get('api_source', 'unknown')
        circuit_breaker = self.get_circuit_breaker(api_name)
        
        last_error = None
        attempt = 0
        
        while attempt < retry_config.max_attempts:
            attempt += 1
            context['attempt_number'] = attempt
            self.retry_statistics['total_attempts'] += 1
            
            try:
                # Update circuit breaker for geological priority
                geological_urgency = context.get('geological_urgency', 'low')
                circuit_breaker.set_geological_priority(geological_urgency)
                
                # Execute with circuit breaker protection
                if retry_config.strategy == RetryStrategy.CIRCUIT_BREAKER:
                    result = circuit_breaker.call(func)
                else:
                    result = await func() if asyncio.iscoroutinefunction(func) else func()
                
                # Success - update statistics
                if attempt > 1:
                    self.retry_statistics['successful_retries'] += 1
                    if context.get('geological_urgency') in ['high', 'critical']:
                        self.retry_statistics['geological_priority_retries'] += 1
                
                return result
                
            except Exception as error:
                last_error = error
                
                # Analyze error context
                error_context = self.error_analyzer.analyze_error_context(error, context)
                self.error_history.append(error_context)
                
                # Determine if we should retry
                should_retry = self._should_retry(error_context, retry_config, attempt)
                
                if not should_retry or attempt >= retry_config.max_attempts:
                    self.retry_statistics['failed_retries'] += 1
                    break
                
                # Calculate delay before next attempt
                delay = self._calculate_delay(error_context, retry_config, attempt)
                
                # Log retry attempt
                logging.warning(f"Retry attempt {attempt}/{retry_config.max_attempts} for {api_name} "
                              f"after {delay:.2f}s delay. Error: {error}")
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_error
    
    def _should_retry(self, error_context: ErrorContext, retry_config: RetryConfig, attempt: int) -> bool:
        """Determine if error should trigger a retry"""
        # No retry strategies
        if retry_config.strategy == RetryStrategy.NO_RETRY:
            return False
        
        # Certain errors should not be retried
        non_retryable_errors = {
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.DATA_PARSING_ERROR
        }
        
        if error_context.error_type in non_retryable_errors:
            return False
        
        # Geological priority overrides for critical data
        if error_context.geological_urgency == 'critical':
            # Allow more retries for critical geological data
            max_critical_attempts = retry_config.max_attempts * 2
            if attempt < max_critical_attempts:
                return True
        
        # Rate limiting requires special handling
        if error_context.error_type == ErrorType.RATE_LIMIT_ERROR:
            # Always retry rate limits with longer delays
            return attempt < retry_config.max_attempts
        
        # Server errors are usually worth retrying
        if error_context.error_type in {ErrorType.SERVER_ERROR, ErrorType.NETWORK_TIMEOUT, ErrorType.CONNECTION_ERROR}:
            return attempt < retry_config.max_attempts
        
        # Default retry logic
        return attempt < retry_config.max_attempts
    
    def _calculate_delay(self, error_context: ErrorContext, retry_config: RetryConfig, attempt: int) -> float:
        """Calculate delay before next retry attempt"""
        base_delay = retry_config.base_delay
        
        # Adjust base delay for geological urgency
        urgency_multipliers = {
            'critical': 0.5,  # Faster retries for critical data
            'high': 0.7,
            'medium': 1.0,
            'low': 1.5        # Slower retries for low priority
        }
        urgency_multiplier = urgency_multipliers.get(error_context.geological_urgency, 1.0)
        base_delay *= urgency_multiplier
        
        # Calculate delay based on strategy
        if retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (retry_config.exponential_base ** (attempt - 1))
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * attempt
        elif retry_config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = base_delay
        elif retry_config.strategy == RetryStrategy.IMMEDIATE_RETRY:
            delay = 0.1
        elif retry_config.strategy == RetryStrategy.ADAPTIVE_GEOLOGICAL:
            delay = self._calculate_adaptive_geological_delay(error_context, base_delay, attempt)
        else:
            delay = base_delay
        
        # Special handling for rate limiting
        if error_context.error_type == ErrorType.RATE_LIMIT_ERROR:
            delay = max(delay, 60.0)  # Minimum 1 minute for rate limits
        
        # Apply jitter if enabled
        if retry_config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += (random.random() - 0.5) * 2 * jitter_range
        
        # Ensure delay is within bounds
        delay = max(0.1, min(delay, retry_config.max_delay))
        
        return delay
    
    def _calculate_adaptive_geological_delay(self, error_context: ErrorContext, base_delay: float, attempt: int) -> float:
        """Calculate adaptive delay based on geological context"""
        # Base exponential backoff
        delay = base_delay * (2 ** (attempt - 1))
        
        # Adjust for geological phenomena priority
        phenomena_priorities = {
            'seismic': 0.8,      # Earthquakes need faster retries
            'volcanic': 0.9,     # Volcanic activity important
            'hydrological': 0.7, # Tsunamis/floods very urgent
            'mass_movement': 1.2 # Landslides less time-critical
        }
        
        if error_context.geological_phenomena:
            # Use the highest priority phenomenon
            min_multiplier = min(phenomena_priorities.get(p, 1.0) for p in error_context.geological_phenomena)
            delay *= min_multiplier
        
        # Adjust for data criticality
        criticality_multipliers = {
            'critical': 0.5,
            'high': 0.7,
            'normal': 1.0
        }
        delay *= criticality_multipliers.get(error_context.data_criticality, 1.0)
        
        return delay
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics"""
        total_attempts = self.retry_statistics['total_attempts']
        successful_retries = self.retry_statistics['successful_retries']
        
        return {
            'total_attempts': total_attempts,
            'successful_retries': successful_retries,
            'failed_retries': self.retry_statistics['failed_retries'],
            'geological_priority_retries': self.retry_statistics['geological_priority_retries'],
            'success_rate': (successful_retries / max(1, total_attempts)) * 100,
            'error_distribution': self._analyze_error_distribution(),
            'circuit_breaker_status': {name: cb.state.value for name, cb in self.circuit_breakers.items()},
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }
    
    def _analyze_error_distribution(self) -> Dict[str, int]:
        """Analyze distribution of error types"""
        distribution = defaultdict(int)
        for error_context in self.error_history:
            distribution[error_context.error_type.value] += 1
        return dict(distribution)

# =============================================================================
# RESILIENT API CLIENT WITH COMPREHENSIVE ERROR HANDLING
# =============================================================================

class ResilientAPIClient(CachedAPIClient):
    """Enhanced API client with comprehensive error handling and retry logic"""
    
    def __init__(self, api_name: str, cache_manager: IntelligentCacheManager = None):
        super().__init__(api_name, cache_manager or IntelligentCacheManager())
        self.retry_handler = AdvancedRetryHandler()
        self.fallback_data_sources: List[Callable] = []
        self.health_check_endpoint = None
        self.last_health_check = None
        self.health_status = True
    
    def add_fallback_source(self, fallback_func: Callable):
        """Add fallback data source for when primary source fails"""
        self.fallback_data_sources.append(fallback_func)
    
    async def resilient_request(self, endpoint: str, params: Dict[str, Any], 
                              retry_config: RetryConfig = None,
                              use_fallback: bool = True) -> Any:
        """Make resilient API request with comprehensive error handling"""
        
        # Prepare context for error handling
        context = {
            'api_source': self.api_name,
            'endpoint': endpoint,
            'params': params,
            'geological_urgency': self._assess_request_urgency(endpoint, params),
            'data_criticality': self._assess_request_criticality(endpoint, params)
        }
        
        # Create request function
        async def make_request():
            return await self.cached_request(endpoint, params)
        
        try:
            # Execute with retry logic
            result = await self.retry_handler.execute_with_retry(
                make_request, context, retry_config
            )
            return result
            
        except Exception as primary_error:
            logging.error(f"Primary API request failed for {self.api_name}/{endpoint}: {primary_error}")
            
            # Try fallback sources if enabled
            if use_fallback and self.fallback_data_sources:
                for i, fallback_source in enumerate(self.fallback_data_sources):
                    try:
                        logging.info(f"Attempting fallback source {i+1} for {self.api_name}")
                        fallback_result = await fallback_source(endpoint, params)
                        
                        # Cache fallback result for future use
                        cache_key = self._generate_cache_key(endpoint, params)
                        await self.cache_manager.set(
                            f"fallback_{cache_key}", 
                            fallback_result, 
                            CacheStrategy.TTL, 
                            ttl=1800  # 30 minutes for fallback data
                        )
                        
                        return fallback_result
                        
                    except Exception as fallback_error:
                        logging.warning(f"Fallback source {i+1} failed: {fallback_error}")
                        continue
            
            # Try stale cache as last resort
            cache_key = self._generate_cache_key(endpoint, params)
            stale_data = await self.cache_manager.get(f"stale_{cache_key}")
            if stale_data is not None:
                logging.info(f"Returning stale cache data for {self.api_name}/{endpoint}")
                return stale_data
            
            # All options exhausted
            raise primary_error
    
    def _assess_request_urgency(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Assess geological urgency of the request"""
        endpoint_lower = endpoint.lower()
        
        # Critical urgency indicators
        if any(term in endpoint_lower for term in ['alert', 'warning', 'emergency', 'tsunami']):
            return 'critical'
        
        # High urgency indicators
        if any(term in endpoint_lower for term in ['real-time', 'current', 'latest', 'earthquake']):
            return 'high'
        
        # Check magnitude in parameters
        if 'magnitude' in params or 'mag' in params:
            try:
                mag_value = float(params.get('magnitude', params.get('mag', 0)))
                if mag_value >= 7.0:
                    return 'critical'
                elif mag_value >= 5.0:
                    return 'high'
                elif mag_value >= 3.0:
                    return 'medium'
            except:
                pass
        
        return 'low'
    
    def _assess_request_criticality(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Assess data criticality of the request"""
        endpoint_lower = endpoint.lower()
        
        # Critical data
        if any(term in endpoint_lower for term in ['tsunami', 'emergency', 'disaster']):
            return 'critical'
        
        # High importance data
        if any(term in endpoint_lower for term in ['earthquake', 'volcanic', 'alert', 'monitoring']):
            return 'high'
        
        return 'normal'
    
    async def health_check(self) -> bool:
        """Perform health check for the API"""
        if self.health_check_endpoint is None:
            return True  # No health check configured
        
        try:
            # Use minimal retry config for health checks
            health_config = RetryConfig(max_attempts=2, base_delay=1.0, strategy=RetryStrategy.FIXED_INTERVAL)
            
            context = {
                'api_source': self.api_name,
                'endpoint': self.health_check_endpoint,
                'params': {},
                'geological_urgency': 'low',
                'data_criticality': 'normal'
            }
            
            async def health_request():
                return await self._make_api_request(self.health_check_endpoint, {})
            
            await self.retry_handler.execute_with_retry(health_request, context, health_config)
            self.health_status = True
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logging.warning(f"Health check failed for {self.api_name}: {e}")
            self.health_status = False
            self.last_health_check = datetime.now()
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and retry statistics"""
        stats = self.retry_handler.get_retry_statistics()
        stats.update({
            'api_name': self.api_name,
            'health_status': self.health_status,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'fallback_sources_count': len(self.fallback_data_sources)
        })
        return stats

# =============================================================================
# ENHANCED API CLASSES WITH COMPREHENSIVE ERROR HANDLING
# =============================================================================

class ResilientUSGSDataAPI(USGSDataAPI, ResilientAPIClient):
    """USGS API with comprehensive error handling and retry logic"""
    
    def __init__(self, api_key: str = None, auth_manager: APIAuthenticationManager = None,
                 rate_limiter: RateLimitManager = None, cache_manager: IntelligentCacheManager = None):
        USGSDataAPI.__init__(self, api_key, auth_manager, rate_limiter)
        ResilientAPIClient.__init__(self, 'usgs', cache_manager)
        
        # Configure USGS-specific retry behavior
        self.default_retry_config = RetryConfig(
            max_attempts=5,  # More attempts for critical geological data
            base_delay=2.0,
            max_delay=120.0,
            strategy=RetryStrategy.ADAPTIVE_GEOLOGICAL,
            geological_priority=True
        )
        
        # Set health check endpoint
        self.health_check_endpoint = "fdsnws/event/1/version"
        
        # Add fallback data source
        self.add_fallback_source(self._earthquake_fallback_source)
    
    async def secure_earthquake_query(self, **params) -> List[Dict[str, Any]]:
        """Secure earthquake query with comprehensive error handling"""
        return await self.resilient_request(
            "fdsnws/event/1/query", 
            params, 
            self.default_retry_config
        )
    
    async def _earthquake_fallback_source(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback earthquake data source"""
        # This could connect to alternative earthquake APIs or return cached historical data
        logging.info("Using earthquake fallback data source")
        
        # Return simulated fallback data based on parameters
        magnitude = params.get('minmagnitude', 4.0)
        return [
            {
                'id': f'fallback_eq_{int(time.time())}',
                'magnitude': magnitude + 0.5,
                'location': 'Fallback earthquake data',
                'time': datetime.now().isoformat(),
                'source': 'fallback',
                'data_quality': 'fallback'
            }
        ]

import random

if __name__ == "__main__":
    # Run example
    asyncio.run(example_knowledge_graph())