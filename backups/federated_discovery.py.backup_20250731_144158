"""
Federated Causal Relationship Discovery across multiple datasets
Implements distributed causal discovery with privacy preservation
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import structlog

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
import aiohttp
import asyncpg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from .config import settings
from .temporal_causal_engine import TemporalCausalRelationship, CausalNetwork

logger = structlog.get_logger()


@dataclass
class DatasetMetadata:
    """Metadata for federated dataset"""
    dataset_id: str
    institution: str
    location: str
    temporal_range: Tuple[datetime, datetime]
    variables: List[str]
    sample_count: int
    data_quality_score: float
    privacy_level: str  # "public", "restricted", "private"
    contact_info: str


@dataclass
class FederatedCausalResult:
    """Result from federated causal discovery"""
    relationship: TemporalCausalRelationship
    participating_datasets: List[str]
    consensus_score: float
    privacy_preserved: bool
    aggregation_method: str
    discovery_timestamp: datetime


@dataclass
class PrivacyPreservingStatistics:
    """Privacy-preserving statistics for federated learning"""
    encrypted_correlations: Dict[str, str]
    differential_privacy_noise: float
    k_anonymity_level: int
    data_hash: str


class FederatedCausalDiscovery:
    """Federated causal discovery across multiple geological datasets"""
    
    def __init__(self):
        self.registered_datasets = {}
        self.federation_nodes = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Privacy parameters
        self.differential_privacy_epsilon = 1.0
        self.k_anonymity_threshold = 5
        self.min_consensus_datasets = 3
        
        # Federated learning parameters
        self.aggregation_methods = {
            "weighted_average": self._weighted_average_aggregation,
            "median_consensus": self._median_consensus_aggregation,
            "bayesian_fusion": self._bayesian_fusion_aggregation,
            "secure_multiparty": self._secure_multiparty_computation
        }
    
    async def register_dataset(self, metadata: DatasetMetadata, data_endpoint: str):
        """Register a dataset for federated causal discovery"""
        logger.info("Registering dataset for federation", 
                   dataset_id=metadata.dataset_id,
                   institution=metadata.institution)
        
        # Validate dataset metadata
        if not await self._validate_dataset_metadata(metadata):
            raise ValueError(f"Invalid dataset metadata for {metadata.dataset_id}")
        
        # Store dataset registration
        self.registered_datasets[metadata.dataset_id] = {
            "metadata": metadata,
            "endpoint": data_endpoint,
            "registration_time": datetime.utcnow(),
            "last_sync": None,
            "status": "active"
        }
        
        # Initialize federation node
        self.federation_nodes[metadata.dataset_id] = {
            "node_id": metadata.dataset_id,
            "capabilities": self._assess_node_capabilities(metadata),
            "trust_score": 1.0,  # Initial trust score
            "communication_key": Fernet.generate_key()
        }
        
        logger.info("Dataset registered successfully", dataset_id=metadata.dataset_id)
    
    async def discover_federated_causality(
        self, 
        target_variables: List[str],
        max_lag: int = 50,
        confidence_threshold: float = 0.7,
        privacy_level: str = "differential_privacy"
    ) -> List[FederatedCausalResult]:
        """Discover causal relationships across federated datasets"""
        
        logger.info("Starting federated causal discovery",
                   target_variables=target_variables,
                   participating_datasets=len(self.registered_datasets))
        
        # Find datasets containing target variables
        participating_datasets = await self._find_participating_datasets(target_variables)
        
        if len(participating_datasets) < self.min_consensus_datasets:
            logger.warning("Insufficient datasets for federated discovery",
                         available=len(participating_datasets),
                         required=self.min_consensus_datasets)
            return []
        
        # Coordinate federated causal discovery
        federated_results = []
        
        for aggregation_method in self.aggregation_methods.keys():
            try:
                logger.info("Applying federated aggregation method", method=aggregation_method)
                
                method_results = await self._coordinate_federated_discovery(
                    participating_datasets,
                    target_variables,
                    max_lag,
                    confidence_threshold,
                    privacy_level,
                    aggregation_method
                )
                
                federated_results.extend(method_results)
                
            except Exception as e:
                logger.warning("Federated method failed", 
                             method=aggregation_method, error=str(e))
        
        # Consensus filtering
        consensus_results = await self._apply_consensus_filtering(
            federated_results, confidence_threshold
        )
        
        logger.info("Federated causal discovery completed",
                   discovered_relationships=len(consensus_results))
        
        return consensus_results
    
    async def _find_participating_datasets(self, target_variables: List[str]) -> List[str]:
        """Find datasets that contain the target variables"""
        
        participating_datasets = []
        
        for dataset_id, dataset_info in self.registered_datasets.items():
            metadata = dataset_info["metadata"]
            
            # Check if dataset contains any target variables
            if any(var in metadata.variables for var in target_variables):
                # Check data quality and privacy constraints
                if (metadata.data_quality_score >= 0.6 and 
                    metadata.privacy_level in ["public", "restricted"]):
                    participating_datasets.append(dataset_id)
        
        return participating_datasets
    
    async def _coordinate_federated_discovery(
        self,
        participating_datasets: List[str],
        target_variables: List[str],
        max_lag: int,
        confidence_threshold: float,
        privacy_level: str,
        aggregation_method: str
    ) -> List[FederatedCausalResult]:
        """Coordinate causal discovery across federated datasets"""
        
        # Step 1: Request local causal discovery from each dataset
        local_results = {}
        
        for dataset_id in participating_datasets:
            try:
                local_result = await self._request_local_causal_discovery(
                    dataset_id, target_variables, max_lag, confidence_threshold, privacy_level
                )
                local_results[dataset_id] = local_result
                
            except Exception as e:
                logger.warning("Local causal discovery failed",
                             dataset_id=dataset_id, error=str(e))
        
        # Step 2: Aggregate results using specified method
        if len(local_results) < self.min_consensus_datasets:
            return []
        
        aggregation_func = self.aggregation_methods[aggregation_method]
        federated_results = await aggregation_func(local_results, target_variables)
        
        return federated_results
    
    async def _request_local_causal_discovery(
        self,
        dataset_id: str,
        target_variables: List[str],
        max_lag: int,
        confidence_threshold: float,
        privacy_level: str
    ) -> Dict[str, Any]:
        """Request local causal discovery from a federated dataset"""
        
        dataset_info = self.registered_datasets[dataset_id]
        endpoint = dataset_info["endpoint"]
        
        # Prepare request with privacy constraints
        request_data = {
            "target_variables": target_variables,
            "max_lag": max_lag,
            "confidence_threshold": confidence_threshold,
            "privacy_level": privacy_level,
            "differential_privacy_epsilon": self.differential_privacy_epsilon,
            "k_anonymity_threshold": self.k_anonymity_threshold
        }
        
        # Encrypt request if required
        if privacy_level == "encrypted":
            node_key = self.federation_nodes[dataset_id]["communication_key"]
            cipher = Fernet(node_key)
            request_data = {
                "encrypted_request": cipher.encrypt(
                    json.dumps(request_data).encode()
                ).decode()
            }
        
        # Make HTTP request to dataset endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/causal_discovery",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Decrypt response if needed
                    if privacy_level == "encrypted":
                        node_key = self.federation_nodes[dataset_id]["communication_key"]
                        cipher = Fernet(node_key)
                        decrypted_data = cipher.decrypt(
                            result["encrypted_response"].encode()
                        )
                        result = json.loads(decrypted_data.decode())
                    
                    return result
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    async def _weighted_average_aggregation(
        self, 
        local_results: Dict[str, Dict[str, Any]], 
        target_variables: List[str]
    ) -> List[FederatedCausalResult]:
        """Aggregate causal relationships using weighted averaging"""
        
        # Group relationships by cause-effect pairs
        relationship_groups = {}
        
        for dataset_id, result in local_results.items():
            dataset_weight = self._calculate_dataset_weight(dataset_id)
            
            for relationship_data in result.get("relationships", []):
                cause = relationship_data["cause_variable"]
                effect = relationship_data["effect_variable"]
                key = (cause, effect)
                
                if key not in relationship_groups:
                    relationship_groups[key] = []
                
                relationship_groups[key].append({
                    "dataset_id": dataset_id,
                    "weight": dataset_weight,
                    "relationship": relationship_data
                })
        
        # Aggregate relationships
        federated_results = []
        
        for (cause, effect), group in relationship_groups.items():
            if len(group) >= self.min_consensus_datasets:
                
                # Calculate weighted averages
                total_weight = sum(item["weight"] for item in group)
                
                weighted_strength = sum(
                    item["weight"] * item["relationship"]["causal_strength"] 
                    for item in group
                ) / total_weight
                
                weighted_confidence = sum(
                    item["weight"] * item["relationship"]["confidence"] 
                    for item in group
                ) / total_weight
                
                weighted_lag = int(sum(
                    item["weight"] * item["relationship"]["time_lag"] 
                    for item in group
                ) / total_weight)
                
                # Calculate consensus score
                consensus_score = self._calculate_consensus_score(group)
                
                # Create federated result
                federated_relationship = TemporalCausalRelationship(
                    cause_variable=cause,
                    effect_variable=effect,
                    time_lag=weighted_lag,
                    causal_strength=weighted_strength,
                    confidence=weighted_confidence,
                    p_value=1 - weighted_confidence,
                    method="federated_weighted_average",
                    temporal_window=(datetime.min, datetime.max),  # Placeholder
                    supporting_evidence={
                        "participating_datasets": len(group),
                        "consensus_score": consensus_score,
                        "dataset_weights": {item["dataset_id"]: item["weight"] for item in group}
                    }
                )
                
                federated_result = FederatedCausalResult(
                    relationship=federated_relationship,
                    participating_datasets=[item["dataset_id"] for item in group],
                    consensus_score=consensus_score,
                    privacy_preserved=True,
                    aggregation_method="weighted_average",
                    discovery_timestamp=datetime.utcnow()
                )
                
                federated_results.append(federated_result)
        
        return federated_results
    
    async def _median_consensus_aggregation(
        self, 
        local_results: Dict[str, Dict[str, Any]], 
        target_variables: List[str]
    ) -> List[FederatedCausalResult]:
        """Aggregate using median consensus (robust to outliers)"""
        
        # Group relationships by cause-effect pairs
        relationship_groups = {}
        
        for dataset_id, result in local_results.items():
            for relationship_data in result.get("relationships", []):
                cause = relationship_data["cause_variable"]
                effect = relationship_data["effect_variable"]
                key = (cause, effect)
                
                if key not in relationship_groups:
                    relationship_groups[key] = []
                
                relationship_groups[key].append({
                    "dataset_id": dataset_id,
                    "relationship": relationship_data
                })
        
        # Aggregate using median
        federated_results = []
        
        for (cause, effect), group in relationship_groups.items():
            if len(group) >= self.min_consensus_datasets:
                
                # Calculate medians
                strengths = [item["relationship"]["causal_strength"] for item in group]
                confidences = [item["relationship"]["confidence"] for item in group]
                lags = [item["relationship"]["time_lag"] for item in group]
                
                median_strength = np.median(strengths)
                median_confidence = np.median(confidences)
                median_lag = int(np.median(lags))
                
                # Calculate consensus score based on agreement
                consensus_score = self._calculate_median_consensus_score(
                    strengths, confidences, lags
                )
                
                # Create federated result
                federated_relationship = TemporalCausalRelationship(
                    cause_variable=cause,
                    effect_variable=effect,
                    time_lag=median_lag,
                    causal_strength=median_strength,
                    confidence=median_confidence,
                    p_value=1 - median_confidence,
                    method="federated_median_consensus",
                    temporal_window=(datetime.min, datetime.max),
                    supporting_evidence={
                        "participating_datasets": len(group),
                        "consensus_score": consensus_score,
                        "strength_variance": np.var(strengths),
                        "confidence_variance": np.var(confidences)
                    }
                )
                
                federated_result = FederatedCausalResult(
                    relationship=federated_relationship,
                    participating_datasets=[item["dataset_id"] for item in group],
                    consensus_score=consensus_score,
                    privacy_preserved=True,
                    aggregation_method="median_consensus",
                    discovery_timestamp=datetime.utcnow()
                )
                
                federated_results.append(federated_result)
        
        return federated_results
    
    async def _bayesian_fusion_aggregation(
        self, 
        local_results: Dict[str, Dict[str, Any]], 
        target_variables: List[str]
    ) -> List[FederatedCausalResult]:
        """Aggregate using Bayesian fusion of evidence"""
        
        # This would implement Bayesian evidence fusion
        # For now, return empty list as placeholder
        logger.info("Bayesian fusion aggregation not yet implemented")
        return []
    
    async def _secure_multiparty_computation(
        self, 
        local_results: Dict[str, Dict[str, Any]], 
        target_variables: List[str]
    ) -> List[FederatedCausalResult]:
        """Aggregate using secure multiparty computation"""
        
        # This would implement secure multiparty computation
        # For now, return empty list as placeholder
        logger.info("Secure multiparty computation not yet implemented")
        return []
    
    def _calculate_dataset_weight(self, dataset_id: str) -> float:
        """Calculate weight for a dataset based on quality and trust"""
        
        dataset_info = self.registered_datasets[dataset_id]
        metadata = dataset_info["metadata"]
        node_info = self.federation_nodes[dataset_id]
        
        # Factors for weight calculation
        quality_weight = metadata.data_quality_score
        trust_weight = node_info["trust_score"]
        size_weight = min(1.0, metadata.sample_count / 10000)  # Normalize by 10k samples
        
        # Combine weights
        total_weight = (quality_weight * 0.4 + 
                       trust_weight * 0.4 + 
                       size_weight * 0.2)
        
        return max(0.1, total_weight)  # Minimum weight of 0.1
    
    def _calculate_consensus_score(self, group: List[Dict[str, Any]]) -> float:
        """Calculate consensus score for a group of relationships"""
        
        if len(group) < 2:
            return 1.0
        
        # Extract values for consensus calculation
        strengths = [item["relationship"]["causal_strength"] for item in group]
        confidences = [item["relationship"]["confidence"] for item in group]
        
        # Calculate coefficient of variation (lower = better consensus)
        strength_cv = np.std(strengths) / np.mean(strengths) if np.mean(strengths) > 0 else 1.0
        confidence_cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else 1.0
        
        # Convert to consensus score (higher = better consensus)
        consensus_score = 1.0 / (1.0 + (strength_cv + confidence_cv) / 2.0)
        
        return min(1.0, max(0.0, consensus_score))
    
    def _calculate_median_consensus_score(
        self, strengths: List[float], confidences: List[float], lags: List[int]
    ) -> float:
        """Calculate consensus score based on median absolute deviation"""
        
        if len(strengths) < 2:
            return 1.0
        
        # Calculate median absolute deviations
        strength_mad = np.median(np.abs(strengths - np.median(strengths)))
        confidence_mad = np.median(np.abs(confidences - np.median(confidences)))
        
        # Normalize and combine
        strength_consensus = 1.0 / (1.0 + strength_mad)
        confidence_consensus = 1.0 / (1.0 + confidence_mad)
        
        return (strength_consensus + confidence_consensus) / 2.0
    
    async def _apply_consensus_filtering(
        self, 
        federated_results: List[FederatedCausalResult],
        confidence_threshold: float
    ) -> List[FederatedCausalResult]:
        """Filter results based on consensus and confidence thresholds"""
        
        filtered_results = []
        
        # Group by relationship
        relationship_groups = {}
        for result in federated_results:
            key = (result.relationship.cause_variable, result.relationship.effect_variable)
            if key not in relationship_groups:
                relationship_groups[key] = []
            relationship_groups[key].append(result)
        
        # Select best result for each relationship
        for key, group in relationship_groups.items():
            # Sort by consensus score and confidence
            group.sort(key=lambda x: (x.consensus_score, x.relationship.confidence), reverse=True)
            
            best_result = group[0]
            
            # Apply thresholds
            if (best_result.relationship.confidence >= confidence_threshold and
                best_result.consensus_score >= 0.6):  # Minimum consensus threshold
                filtered_results.append(best_result)
        
        return filtered_results
    
    async def _validate_dataset_metadata(self, metadata: DatasetMetadata) -> bool:
        """Validate dataset metadata for federation"""
        
        # Check required fields
        if not all([
            metadata.dataset_id,
            metadata.institution,
            metadata.variables,
            metadata.sample_count > 0,
            0.0 <= metadata.data_quality_score <= 1.0
        ]):
            return False
        
        # Check temporal range validity
        if metadata.temporal_range[0] >= metadata.temporal_range[1]:
            return False
        
        # Check privacy level
        if metadata.privacy_level not in ["public", "restricted", "private"]:
            return False
        
        return True
    
    def _assess_node_capabilities(self, metadata: DatasetMetadata) -> Dict[str, Any]:
        """Assess computational capabilities of a federation node"""
        
        capabilities = {
            "data_size": "large" if metadata.sample_count > 100000 else "medium" if metadata.sample_count > 10000 else "small",
            "temporal_coverage": (metadata.temporal_range[1] - metadata.temporal_range[0]).days,
            "variable_count": len(metadata.variables),
            "privacy_level": metadata.privacy_level,
            "estimated_compute_time": self._estimate_compute_time(metadata)
        }
        
        return capabilities
    
    def _estimate_compute_time(self, metadata: DatasetMetadata) -> float:
        """Estimate computation time for causal discovery"""
        
        # Simple heuristic based on data size and variable count
        base_time = metadata.sample_count * len(metadata.variables) * 0.001  # milliseconds
        
        # Adjust for data quality (lower quality = more preprocessing time)
        quality_factor = 2.0 - metadata.data_quality_score
        
        return base_time * quality_factor
    
    async def synchronize_federated_knowledge(self) -> Dict[str, Any]:
        """Synchronize knowledge across federated datasets"""
        logger.info("Starting federated knowledge synchronization")
        
        sync_results = {
            "synchronized_datasets": [],
            "failed_datasets": [],
            "new_relationships": 0,
            "updated_relationships": 0,
            "sync_timestamp": datetime.utcnow()
        }
        
        # Sync with each registered dataset
        for dataset_id, dataset_info in self.registered_datasets.items():
            try:
                # Request latest knowledge updates
                updates = await self._request_knowledge_updates(dataset_id)
                
                # Process updates
                if updates:
                    sync_results["new_relationships"] += len(updates.get("new_relationships", []))
                    sync_results["updated_relationships"] += len(updates.get("updated_relationships", []))
                
                sync_results["synchronized_datasets"].append(dataset_id)
                
                # Update last sync time
                self.registered_datasets[dataset_id]["last_sync"] = datetime.utcnow()
                
            except Exception as e:
                logger.warning("Failed to sync with dataset", 
                             dataset_id=dataset_id, error=str(e))
                sync_results["failed_datasets"].append(dataset_id)
        
        logger.info("Federated knowledge synchronization completed", 
                   results=sync_results)
        
        return sync_results
    
    async def _request_knowledge_updates(self, dataset_id: str) -> Dict[str, Any]:
        """Request knowledge updates from a federated dataset"""
        
        dataset_info = self.registered_datasets[dataset_id]
        endpoint = dataset_info["endpoint"]
        last_sync = dataset_info.get("last_sync")
        
        request_data = {
            "since": last_sync.isoformat() if last_sync else None,
            "include_metadata": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/knowledge_updates",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get status of the federated discovery system"""
        
        active_datasets = sum(1 for info in self.registered_datasets.values() 
                            if info["status"] == "active")
        
        total_variables = set()
        for dataset_info in self.registered_datasets.values():
            total_variables.update(dataset_info["metadata"].variables)
        
        return {
            "registered_datasets": len(self.registered_datasets),
            "active_datasets": active_datasets,
            "total_unique_variables": len(total_variables),
            "federation_nodes": len(self.federation_nodes),
            "last_sync_times": {
                dataset_id: info.get("last_sync")
                for dataset_id, info in self.registered_datasets.items()
            },
            "privacy_settings": {
                "differential_privacy_epsilon": self.differential_privacy_epsilon,
                "k_anonymity_threshold": self.k_anonymity_threshold,
                "min_consensus_datasets": self.min_consensus_datasets
            }
        }
    
    async def close(self):
        """Cleanup federated discovery resources"""
        logger.info("Closing Federated Causal Discovery")
        # Cleanup any resources if needed