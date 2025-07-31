"""
Domain Adaptation Utilities for Geological Settings

This module provides utilities for managing geological domains, data preprocessing
for domain adaptation, and specialized adaptation techniques for different
geological settings.

Requirements addressed: 5.1, 5.2
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class GeologicalSetting(Enum):
    """Enumeration of geological settings"""
    SUBDUCTION_ZONE = "subduction_zone"
    HOTSPOT = "hotspot"
    RIFT_VALLEY = "rift_valley"
    TRANSFORM_FAULT = "transform_fault"
    CONTINENTAL_ARC = "continental_arc"
    OCEANIC_ARC = "oceanic_arc"
    BACK_ARC = "back_arc"
    INTRAPLATE = "intraplate"
    CALDERA_SYSTEM = "caldera_system"
    STRATOVOLCANO = "stratovolcano"
    SHIELD_VOLCANO = "shield_volcano"
    MONOGENETIC_FIELD = "monogenetic_field"


class TectonicContext(Enum):
    """Enumeration of tectonic contexts"""
    CONVERGENT_MARGIN = "convergent_margin"
    DIVERGENT_MARGIN = "divergent_margin"
    TRANSFORM_MARGIN = "transform_margin"
    INTRAPLATE_SETTING = "intraplate_setting"
    EXTENSIONAL_SETTING = "extensional_setting"
    COMPRESSIONAL_SETTING = "compressional_setting"


@dataclass
class DataCharacteristics:
    """Characteristics of available data for a domain"""
    temporal_resolution: str  # "seconds", "minutes", "hours", "days", "years"
    spatial_resolution: float  # meters
    data_quality: float  # 0.0 to 1.0
    completeness: float  # 0.0 to 1.0
    noise_level: float  # 0.0 to 1.0
    sensor_types: List[str]
    measurement_frequency: float  # Hz or samples per day
    historical_coverage: int  # years of historical data


class GeologicalDomainRegistry:
    """Registry for managing geological domains and their characteristics"""
    
    def __init__(self):
        self.domains = {}
        self.domain_similarities = {}
        self._initialize_known_domains()
    
    def _initialize_known_domains(self):
        """Initialize registry with known geological domains"""
        # Yellowstone Caldera System
        self.register_domain(
            domain_id="yellowstone_usa",
            region_name="Yellowstone National Park",
            geological_setting=GeologicalSetting.CALDERA_SYSTEM,
            tectonic_context=TectonicContext.INTRAPLATE_SETTING,
            data_characteristics=DataCharacteristics(
                temporal_resolution="minutes",
                spatial_resolution=100.0,
                data_quality=0.95,
                completeness=0.90,
                noise_level=0.05,
                sensor_types=["seismometer", "gps", "thermal", "gas"],
                measurement_frequency=1.0,
                historical_coverage=50
            ),
            available_data_types=["seismic", "deformation", "thermal", "geochemical"],
            data_sparsity_level=0.9
        )
        
        # Mount Vesuvius
        self.register_domain(
            domain_id="vesuvius_italy",
            region_name="Mount Vesuvius",
            geological_setting=GeologicalSetting.STRATOVOLCANO,
            tectonic_context=TectonicContext.CONVERGENT_MARGIN,
            data_characteristics=DataCharacteristics(
                temporal_resolution="hours",
                spatial_resolution=50.0,
                data_quality=0.85,
                completeness=0.80,
                noise_level=0.10,
                sensor_types=["seismometer", "gps", "thermal"],
                measurement_frequency=0.25,
                historical_coverage=30
            ),
            available_data_types=["seismic", "deformation", "thermal"],
            data_sparsity_level=0.7
        )
        
        # Kilauea Hawaii
        self.register_domain(
            domain_id="kilauea_hawaii",
            region_name="Kilauea Volcano",
            geological_setting=GeologicalSetting.SHIELD_VOLCANO,
            tectonic_context=TectonicContext.INTRAPLATE_SETTING,
            data_characteristics=DataCharacteristics(
                temporal_resolution="seconds",
                spatial_resolution=10.0,
                data_quality=0.98,
                completeness=0.95,
                noise_level=0.02,
                sensor_types=["seismometer", "gps", "thermal", "gas", "tiltmeter"],
                measurement_frequency=10.0,
                historical_coverage=40
            ),
            available_data_types=["seismic", "deformation", "thermal", "geochemical", "tilt"],
            data_sparsity_level=0.95
        )
        
        # Andes Subduction Zone
        self.register_domain(
            domain_id="andes_chile",
            region_name="Chilean Andes",
            geological_setting=GeologicalSetting.CONTINENTAL_ARC,
            tectonic_context=TectonicContext.CONVERGENT_MARGIN,
            data_characteristics=DataCharacteristics(
                temporal_resolution="days",
                spatial_resolution=1000.0,
                data_quality=0.70,
                completeness=0.60,
                noise_level=0.20,
                sensor_types=["seismometer", "gps"],
                measurement_frequency=0.1,
                historical_coverage=20
            ),
            available_data_types=["seismic", "deformation"],
            data_sparsity_level=0.4
        )
        
        # Data-sparse volcanic region
        self.register_domain(
            domain_id="kamchatka_russia",
            region_name="Kamchatka Peninsula",
            geological_setting=GeologicalSetting.OCEANIC_ARC,
            tectonic_context=TectonicContext.CONVERGENT_MARGIN,
            data_characteristics=DataCharacteristics(
                temporal_resolution="days",
                spatial_resolution=5000.0,
                data_quality=0.60,
                completeness=0.40,
                noise_level=0.30,
                sensor_types=["seismometer"],
                measurement_frequency=0.05,
                historical_coverage=10
            ),
            available_data_types=["seismic"],
            data_sparsity_level=0.2
        )
    
    def register_domain(self, domain_id: str, region_name: str, 
                       geological_setting: GeologicalSetting,
                       tectonic_context: TectonicContext,
                       data_characteristics: DataCharacteristics,
                       available_data_types: List[str],
                       data_sparsity_level: float):
        """Register a new geological domain"""
        from .transfer_learning_engine import GeologicalDomain
        
        domain = GeologicalDomain(
            domain_id=domain_id,
            region_name=region_name,
            geological_setting=geological_setting.value,
            tectonic_context=tectonic_context.value,
            data_characteristics=asdict(data_characteristics),
            available_data_types=available_data_types,
            data_sparsity_level=data_sparsity_level
        )
        
        self.domains[domain_id] = domain
        logger.info(f"Registered geological domain: {region_name} ({domain_id})")
    
    def get_domain(self, domain_id: str):
        """Get domain by ID"""
        return self.domains.get(domain_id)
    
    def get_similar_domains(self, target_domain_id: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find domains similar to target domain"""
        target_domain = self.domains.get(target_domain_id)
        if not target_domain:
            return []
        
        similarities = []
        for domain_id, domain in self.domains.items():
            if domain_id != target_domain_id:
                similarity = self._calculate_domain_similarity(target_domain, domain)
                similarities.append((domain_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_domain_similarity(self, domain1, domain2) -> float:
        """Calculate similarity between two geological domains"""
        similarity_score = 0.0
        
        # Geological setting similarity
        if domain1.geological_setting == domain2.geological_setting:
            similarity_score += 0.3
        
        # Tectonic context similarity
        if domain1.tectonic_context == domain2.tectonic_context:
            similarity_score += 0.2
        
        # Data characteristics similarity
        data_char_sim = self._calculate_data_characteristics_similarity(
            domain1.data_characteristics, domain2.data_characteristics
        )
        similarity_score += 0.3 * data_char_sim
        
        # Available data types similarity
        data_types_sim = self._calculate_data_types_similarity(
            domain1.available_data_types, domain2.available_data_types
        )
        similarity_score += 0.2 * data_types_sim
        
        return similarity_score
    
    def _calculate_data_characteristics_similarity(self, char1: Dict, char2: Dict) -> float:
        """Calculate similarity between data characteristics"""
        
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


        quality_sim = 1.0 - abs(char1.get('data_quality', 0) - char2.get('data_quality', 0))
        completeness_sim = 1.0 - abs(char1.get('completeness', 0) - char2.get('completeness', 0))
        sparsity_sim = 1.0 - abs(char1.get('data_sparsity_level', 0) - char2.get('data_sparsity_level', 0))
        
        return (quality_sim + completeness_sim + sparsity_sim) / 3.0
    
    def _calculate_data_types_similarity(self, types1: List[str], types2: List[str]) -> float:
        """Calculate similarity between available data types"""
        set1 = set(types1)
        set2 = set(types2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def list_domains(self) -> List[str]:
        """List all registered domain IDs"""
        return list(self.domains.keys())
    
    def export_domains(self, filepath: str):
        """Export domain registry to JSON file"""
        export_data = {}
        for domain_id, domain in self.domains.items():
            export_data[domain_id] = {
                'region_name': domain.region_name,
                'geological_setting': domain.geological_setting,
                'tectonic_context': domain.tectonic_context,
                'data_characteristics': domain.data_characteristics,
                'available_data_types': domain.available_data_types,
                'data_sparsity_level': domain.data_sparsity_level
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Domain registry exported to {filepath}")


class DomainDataPreprocessor:
    """Preprocessor for domain adaptation data"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_extractors = {}
    
    def preprocess_for_adaptation(self, source_data: np.ndarray, target_data: np.ndarray,
                                 domain_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for domain adaptation"""
        logger.info(f"Preprocessing data for domain adaptation: {domain_id}")
        
        # Normalize data
        source_normalized = self._normalize_data(source_data, f"{domain_id}_source")
        target_normalized = self._normalize_data(target_data, f"{domain_id}_target")
        
        # Feature alignment
        source_aligned, target_aligned = self._align_features(source_normalized, target_normalized)
        
        # Domain-specific preprocessing
        source_processed = self._apply_domain_specific_preprocessing(source_aligned, domain_id)
        target_processed = self._apply_domain_specific_preprocessing(target_aligned, domain_id)
        
        return source_processed, target_processed
    
    def _normalize_data(self, data: np.ndarray, scaler_key: str) -> np.ndarray:
        """Normalize data using stored or new scaler"""
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
            normalized_data = self.scalers[scaler_key].fit_transform(data)
        else:
            normalized_data = self.scalers[scaler_key].transform(data)
        
        return normalized_data
    
    def _align_features(self, source_data: np.ndarray, target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align features between source and target domains"""
        # Ensure same number of features
        min_features = min(source_data.shape[1], target_data.shape[1])
        
        source_aligned = source_data[:, :min_features]
        target_aligned = target_data[:, :min_features]
        
        return source_aligned, target_aligned
    
    def _apply_domain_specific_preprocessing(self, data: np.ndarray, domain_id: str) -> np.ndarray:
        """Apply domain-specific preprocessing"""
        # This would include geological domain-specific transformations
        # For now, return data as-is
        return data
    
    def extract_domain_invariant_features(self, data: np.ndarray, domain_id: str) -> np.ndarray:
        """Extract features that are invariant across domains"""
        # Use PCA to extract domain-invariant features
        if f"{domain_id}_pca" not in self.feature_extractors:
            self.feature_extractors[f"{domain_id}_pca"] = PCA(n_components=0.95)  # Keep 95% variance
            invariant_features = self.feature_extractors[f"{domain_id}_pca"].fit_transform(data)
        else:
            invariant_features = self.feature_extractors[f"{domain_id}_pca"].transform(data)
        
        return invariant_features
    
    def visualize_domain_shift(self, source_data: np.ndarray, target_data: np.ndarray,
                              domain_names: Tuple[str, str], save_path: Optional[str] = None):
        """Visualize domain shift using t-SNE"""
        logger.info(f"Visualizing domain shift between {domain_names[0]} and {domain_names[1]}")
        
        # Combine data for t-SNE
        combined_data = np.vstack([source_data, target_data])
        labels = np.hstack([
            np.zeros(source_data.shape[0]),  # Source domain = 0
            np.ones(target_data.shape[0])    # Target domain = 1
        ])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(combined_data)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red']
        labels_text = [domain_names[0], domain_names[1]]
        
        for i in range(2):
            mask = labels == i
            plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1], 
                       c=colors[i], label=labels_text[i], alpha=0.6)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('Domain Shift Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Domain shift visualization saved to {save_path}")
        
        plt.show()


class AdaptationStrategySelector:
    """Selects optimal adaptation strategy based on domain characteristics"""
    
    def __init__(self):
        self.strategy_rules = self._initialize_strategy_rules()
    
    def _initialize_strategy_rules(self) -> Dict[str, Dict]:
        """Initialize rules for strategy selection"""
        return {
            'fine_tuning': {
                'min_data_sparsity': 0.5,
                'min_data_quality': 0.7,
                'suitable_settings': [
                    GeologicalSetting.STRATOVOLCANO,
                    GeologicalSetting.SHIELD_VOLCANO,
                    GeologicalSetting.CALDERA_SYSTEM
                ],
                'complexity': 'low'
            },
            'domain_adversarial': {
                'min_data_sparsity': 0.3,
                'min_data_quality': 0.6,
                'suitable_settings': [
                    GeologicalSetting.SUBDUCTION_ZONE,
                    GeologicalSetting.CONTINENTAL_ARC,
                    GeologicalSetting.OCEANIC_ARC
                ],
                'complexity': 'medium'
            },
            'few_shot': {
                'max_data_sparsity': 0.3,
                'min_data_quality': 0.5,
                'suitable_settings': [
                    GeologicalSetting.MONOGENETIC_FIELD,
                    GeologicalSetting.BACK_ARC,
                    GeologicalSetting.INTRAPLATE
                ],
                'complexity': 'high'
            }
        }
    
    def select_strategy(self, target_domain, available_strategies: List[str] = None) -> str:
        """Select optimal adaptation strategy for target domain"""
        if available_strategies is None:
            available_strategies = list(self.strategy_rules.keys())
        
        strategy_scores = {}
        
        for strategy in available_strategies:
            if strategy not in self.strategy_rules:
                continue
            
            score = self._calculate_strategy_score(strategy, target_domain)
            strategy_scores[strategy] = score
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        logger.info(f"Selected adaptation strategy '{best_strategy}' for {target_domain.region_name}")
        logger.info(f"Strategy scores: {strategy_scores}")
        
        return best_strategy
    
    def _calculate_strategy_score(self, strategy: str, target_domain) -> float:
        """Calculate suitability score for a strategy"""
        rules = self.strategy_rules[strategy]
        score = 0.0
        
        # Data sparsity check
        if 'min_data_sparsity' in rules:
            if target_domain.data_sparsity_level >= rules['min_data_sparsity']:
                score += 0.3
        
        if 'max_data_sparsity' in rules:
            if target_domain.data_sparsity_level <= rules['max_data_sparsity']:
                score += 0.3
        
        # Data quality check
        data_quality = target_domain.data_characteristics.get('data_quality', 0.5)
        if data_quality >= rules.get('min_data_quality', 0.0):
            score += 0.3
        
        # Geological setting suitability
        target_setting = GeologicalSetting(target_domain.geological_setting)
        if target_setting in rules.get('suitable_settings', []):
            score += 0.4
        
        return score
    
    def get_strategy_recommendations(self, target_domain) -> Dict[str, Any]:
        """Get detailed recommendations for adaptation strategies"""
        recommendations = {}
        
        for strategy, rules in self.strategy_rules.items():
            score = self._calculate_strategy_score(strategy, target_domain)
            
            recommendations[strategy] = {
                'suitability_score': score,
                'complexity': rules['complexity'],
                'estimated_performance': self._estimate_performance(strategy, target_domain),
                'resource_requirements': self._estimate_resources(strategy, target_domain)
            }
        
        return recommendations
    
    def _estimate_performance(self, strategy: str, target_domain) -> Dict[str, float]:
        """Estimate expected performance for strategy"""
        
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


        base_performance = {
            'fine_tuning': 0.85,
            'domain_adversarial': 0.80,
            'few_shot': 0.75
        }
        
        # Adjust based on domain characteristics
        performance = base_performance.get(strategy, 0.7)
        
        # Adjust for data quality
        data_quality = target_domain.data_characteristics.get('data_quality', 0.5)
        performance *= (0.5 + 0.5 * data_quality)
        
        return {
            'expected_accuracy': performance,
            'confidence_interval': (performance - 0.1, performance + 0.05)
        }
    
    def _estimate_resources(self, strategy: str, target_domain) -> Dict[str, Any]:
        """Estimate resource requirements for strategy"""
        base_resources = {
            'fine_tuning': {'compute_hours': 10, 'memory_gb': 8, 'samples_needed': 1000},
            'domain_adversarial': {'compute_hours': 20, 'memory_gb': 16, 'samples_needed': 500},
            'few_shot': {'compute_hours': 5, 'memory_gb': 4, 'samples_needed': 50}
        }
        
        return base_resources.get(strategy, {'compute_hours': 15, 'memory_gb': 12, 'samples_needed': 750})