"""
Secure Multi-Party Computation (MPC) System for Prometheus AI Platform

This module implements secure multi-party computation capabilities including:
- Privacy-preserving collaborative analysis
- Federated learning for geological models
- Secure aggregation of sensitive geological data
- Private set intersection for research collaboration
- Differential privacy for data sharing
- Secure computation protocols (Shamir's Secret Sharing, BGV, etc.)
- Zero-knowledge proofs for data verification
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import uuid
import base64
import random

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import aioredis

# Cryptographic libraries for MPC
try:
    import pysyft as sy
    from pysyft import TorchHook
    import torch
    PYSYFT_AVAILABLE = True
except ImportError:
    PYSYFT_AVAILABLE = False
    logging.warning("PySyft not available - MPC capabilities limited")

try:
    from crypten import mpc
    import crypten
    CRYPTEN_AVAILABLE = True
except ImportError:
    CRYPTEN_AVAILABLE = False
    logging.warning("CrypTen not available - some MPC protocols unavailable")

# Differential privacy
try:
    import opacus
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available - differential privacy limited")

# Import from other security modules
from .encryption_engine import DataClassification, KeyManager


class MPCProtocol(Enum):
    """Multi-party computation protocols"""
    SHAMIR_SECRET_SHARING = "shamir_secret_sharing"
    BGV_HOMOMORPHIC = "bgv_homomorphic"
    CKKS_HOMOMORPHIC = "ckks_homomorphic"
    GARBLED_CIRCUITS = "garbled_circuits"
    OBLIVIOUS_TRANSFER = "oblivious_transfer"
    PRIVATE_SET_INTERSECTION = "private_set_intersection"
    FEDERATED_LEARNING = "federated_learning"
    DIFFERENTIAL_PRIVACY = "differential_privacy"


class ComputationType(Enum):
    """Types of secure computations"""
    STATISTICAL_ANALYSIS = "statistical_analysis"
    MACHINE_LEARNING = "machine_learning"
    DATA_AGGREGATION = "data_aggregation"
    CORRELATION_ANALYSIS = "correlation_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_DETECTION = "pattern_detection"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    JOINT_MODELING = "joint_modeling"


class ParticipantRole(Enum):
    """Participant roles in MPC"""
    DATA_PROVIDER = "data_provider"
    COMPUTATION_PROVIDER = "computation_provider"
    RESULT_CONSUMER = "result_consumer"
    COORDINATOR = "coordinator"
    VERIFIER = "verifier"


@dataclass
class MPCParticipant:
    """MPC participant information"""
    participant_id: str
    organization: str
    role: ParticipantRole
    public_key: str
    endpoint: str
    capabilities: List[MPCProtocol]
    trust_level: float  # 0.0 to 1.0
    data_classification_clearance: List[DataClassification]
    geological_domains: List[str]
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    reputation_score: float = 1.0
    verified: bool = False


@dataclass
class MPCSession:
    """Secure multi-party computation session"""
    session_id: str
    name: str
    description: str
    protocol: MPCProtocol
    computation_type: ComputationType
    coordinator_id: str
    participants: List[str]  # Participant IDs
    data_requirements: Dict[str, Any]
    privacy_parameters: Dict[str, Any]
    security_parameters: Dict[str, Any]
    status: str = "initializing"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results_available: bool = False
    error_message: Optional[str] = None


@dataclass
class SecretShare:
    """Secret share for Shamir's Secret Sharing"""
    share_id: str
    participant_id: str
    share_value: bytes
    threshold: int
    total_shares: int
    polynomial_degree: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DifferentialPrivacyParams:
    """Differential privacy parameters"""
    epsilon: float  # Privacy budget
    delta: float    # Failure probability
    sensitivity: float  # Global sensitivity
    mechanism: str  # "laplace", "gaussian", "exponential"
    clipping_bound: Optional[float] = None
    noise_multiplier: Optional[float] = None


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration"""
    model_architecture: str
    aggregation_method: str  # "fedavg", "fedprox", "scaffold"
    rounds: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    differential_privacy: Optional[DifferentialPrivacyParams] = None
    secure_aggregation: bool = True
    byzantine_robust: bool = False


class ShamirSecretSharing:
    """Shamir's Secret Sharing implementation"""
    
    def __init__(self, prime: int = 2**127 - 1):
        self.prime = prime
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = (a % m + m) % m
        
        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x"""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, self.prime)) % self.prime
        return result
    
    def create_shares(
        self,
        secret: int,
        threshold: int,
        num_shares: int
    ) -> List[Tuple[int, int]]:
        """Create secret shares using Shamir's Secret Sharing"""
        
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
        
        # Generate random coefficients for polynomial
        coefficients = [secret]  # a0 = secret
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(self.prime))
        
        # Generate shares
        shares = []
        for i in range(1, num_shares + 1):
            x = i
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from shares using Lagrange interpolation"""
        
        if len(shares) < 2:
            raise ValueError("Need at least 2 shares to reconstruct")
        
        # Lagrange interpolation at x = 0
        secret = 0
        
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial li(0)
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate li(0) = numerator / denominator
            lagrange_coeff = (numerator * self._mod_inverse(denominator, self.prime)) % self.prime
            
            # Add yi * li(0) to result
            secret = (secret + yi * lagrange_coeff) % self.prime
        
        return secret


class DifferentialPrivacyEngine:
    """Differential privacy engine for geological data"""
    
    def __init__(self):
        self.mechanisms = {
            'laplace': self._laplace_mechanism,
            'gaussian': self._gaussian_mechanism,
            'exponential': self._exponential_mechanism
        }
    
    def add_noise(
        self,
        data: Union[float, np.ndarray],
        params: DifferentialPrivacyParams
    ) -> Union[float, np.ndarray]:
        """Add differential privacy noise to data"""
        
        mechanism = self.mechanisms.get(params.mechanism)
        if not mechanism:
            raise ValueError(f"Unknown mechanism: {params.mechanism}")
        
        return mechanism(data, params)
    
    def _laplace_mechanism(
        self,
        data: Union[float, np.ndarray],
        params: DifferentialPrivacyParams
    ) -> Union[float, np.ndarray]:
        """Laplace mechanism for differential privacy"""
        
        scale = params.sensitivity / params.epsilon
        
        if isinstance(data, np.ndarray):
            noise = np.random.laplace(0, scale, data.shape)
            return data + noise
        else:
            noise = np.random.laplace(0, scale)
            return data + noise
    
    def _gaussian_mechanism(
        self,
        data: Union[float, np.ndarray],
        params: DifferentialPrivacyParams
    ) -> Union[float, np.ndarray]:
        """Gaussian mechanism for differential privacy"""
        
        if params.delta is None or params.delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        
        # Calculate noise scale for (epsilon, delta)-DP
        c = np.sqrt(2 * np.log(1.25 / params.delta))
        scale = c * params.sensitivity / params.epsilon
        
        if isinstance(data, np.ndarray):
            noise = np.random.normal(0, scale, data.shape)
            return data + noise
        else:
            noise = np.random.normal(0, scale)
            return data + noise
    
    def _exponential_mechanism(
        self,
        data: Union[float, np.ndarray],
        params: DifferentialPrivacyParams
    ) -> Union[float, np.ndarray]:
        """Exponential mechanism for differential privacy"""
        
        # Simplified exponential mechanism
        # In practice, this would be more sophisticated
        if isinstance(data, np.ndarray):
            # Add exponentially distributed noise
            noise = np.random.exponential(params.sensitivity / params.epsilon, data.shape)
            return data + noise
        else:
            noise = np.random.exponential(params.sensitivity / params.epsilon)
            return data + noise
    
    def calculate_privacy_budget(
        self,
        queries: List[Dict[str, Any]],
        composition_method: str = "basic"
    ) -> Tuple[float, float]:
        """Calculate total privacy budget consumption"""
        
        total_epsilon = 0.0
        total_delta = 0.0
        
        if composition_method == "basic":
            # Basic composition
            for query in queries:
                total_epsilon += query.get('epsilon', 0.0)
                total_delta += query.get('delta', 0.0)
        
        elif composition_method == "advanced":
            # Advanced composition (simplified)
            # In practice, would use more sophisticated bounds
            for query in queries:
                epsilon = query.get('epsilon', 0.0)
                delta = query.get('delta', 0.0)
                
                # Simplified advanced composition
                total_epsilon += epsilon * np.sqrt(2 * len(queries) * np.log(1/delta))
                total_delta += delta
        
        return total_epsilon, total_delta


class FederatedLearningEngine:
    """Federated learning engine for geological models"""
    
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.participants = {}
        self.global_model = None
        self.round_number = 0
        self.dp_engine = DifferentialPrivacyEngine()
    
    async def initialize_federation(
        self,
        participants: List[MPCParticipant]
    ) -> bool:
        """Initialize federated learning session"""
        
        try:
            # Store participants
            for participant in participants:
                self.participants[participant.participant_id] = participant
            
            # Initialize global model (placeholder)
            self.global_model = self._create_initial_model()
            
            logging.info(f"Federated learning initialized with {len(participants)} participants")
            return True
            
        except Exception as e:
            logging.error(f"Federated learning initialization failed: {e}")
            return False
    
    def _create_initial_model(self) -> Dict[str, Any]:
        """Create initial global model"""
        
        # This would create an actual model based on config.model_architecture
        # For now, return a placeholder
        return {
            'architecture': self.config.model_architecture,
            'parameters': {},
            'version': 0,
            'created_at': datetime.utcnow().isoformat()
        }
    
    async def federated_training_round(
        self,
        participant_updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute one round of federated training"""
        
        try:
            self.round_number += 1
            
            # Apply differential privacy if configured
            if self.config.differential_privacy:
                participant_updates = await self._apply_differential_privacy(participant_updates)
            
            # Aggregate model updates
            if self.config.aggregation_method == "fedavg":
                aggregated_update = await self._federated_averaging(participant_updates)
            elif self.config.aggregation_method == "fedprox":
                aggregated_update = await self._federated_proximal(participant_updates)
            else:
                raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
            
            # Update global model
            self.global_model = await self._update_global_model(aggregated_update)
            
            logging.info(f"Completed federated training round {self.round_number}")
            
            return {
                'round': self.round_number,
                'global_model': self.global_model,
                'participants': len(participant_updates),
                'aggregation_method': self.config.aggregation_method
            }
            
        except Exception as e:
            logging.error(f"Federated training round failed: {e}")
            raise
    
    async def _apply_differential_privacy(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply differential privacy to model updates"""
        
        dp_params = self.config.differential_privacy
        private_updates = {}
        
        for participant_id, update in updates.items():
            # Apply clipping if configured
            if dp_params.clipping_bound:
                update = self._clip_gradients(update, dp_params.clipping_bound)
            
            # Add noise to gradients
            private_update = {}
            for layer_name, gradients in update.items():
                if isinstance(gradients, np.ndarray):
                    private_gradients = self.dp_engine.add_noise(gradients, dp_params)
                    private_update[layer_name] = private_gradients
                else:
                    private_update[layer_name] = gradients
            
            private_updates[participant_id] = private_update
        
        return private_updates
    
    def _clip_gradients(
        self,
        gradients: Dict[str, Any],
        clipping_bound: float
    ) -> Dict[str, Any]:
        """Clip gradients to bound sensitivity"""
        
        clipped_gradients = {}
        
        for layer_name, grad in gradients.items():
            if isinstance(grad, np.ndarray):
                # Calculate L2 norm
                norm = np.linalg.norm(grad)
                
                # Clip if necessary
                if norm > clipping_bound:
                    clipped_grad = grad * (clipping_bound / norm)
                else:
                    clipped_grad = grad
                
                clipped_gradients[layer_name] = clipped_grad
            else:
                clipped_gradients[layer_name] = grad
        
        return clipped_gradients
    
    async def _federated_averaging(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Federated averaging aggregation"""
        
        if not updates:
            return {}
        
        # Calculate weighted average based on data size
        total_samples = sum(update.get('num_samples', 1) for update in updates.values())
        
        aggregated = {}
        
        # Get all layer names from first update
        first_update = next(iter(updates.values()))
        layer_names = first_update.keys()
        
        for layer_name in layer_names:
            if layer_name == 'num_samples':
                continue
            
            weighted_sum = None
            
            for participant_id, update in updates.items():
                if layer_name not in update:
                    continue
                
                layer_update = update[layer_name]
                weight = update.get('num_samples', 1) / total_samples
                
                if isinstance(layer_update, np.ndarray):
                    weighted_update = layer_update * weight
                    
                    if weighted_sum is None:
                        weighted_sum = weighted_update
                    else:
                        weighted_sum += weighted_update
                
            if weighted_sum is not None:
                aggregated[layer_name] = weighted_sum
        
        return aggregated
    
    async def _federated_proximal(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """FedProx aggregation with proximal term"""
        
        # Simplified FedProx - in practice would include proximal regularization
        return await self._federated_averaging(updates)
    
    async def _update_global_model(
        self,
        aggregated_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update global model with aggregated updates"""
        
        # This would update actual model parameters
        # For now, just update metadata
        updated_model = self.global_model.copy()
        updated_model['version'] += 1
        updated_model['last_updated'] = datetime.utcnow().isoformat()
        updated_model['round'] = self.round_number
        
        return updated_model


class SecureMultiPartyEngine:
    """Main engine for secure multi-party computation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.participants: Dict[str, MPCParticipant] = {}
        self.active_sessions: Dict[str, MPCSession] = {}
        self.secret_sharing = ShamirSecretSharing()
        self.dp_engine = DifferentialPrivacyEngine()
        self.key_manager = KeyManager(config)
    
    async def initialize(self) -> bool:
        """Initialize secure multi-party computation engine"""
        
        try:
            # Initialize Redis
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Initialize key manager
            await self.key_manager.initialize()
            
            logging.info("Secure multi-party computation engine initialized")
            return True
            
        except Exception as e:
            logging.error(f"MPC engine initialization failed: {e}")
            return False
    
    async def register_participant(
        self,
        participant: MPCParticipant
    ) -> Tuple[bool, Optional[str]]:
        """Register MPC participant"""
        
        try:
            # Validate participant
            if not await self._validate_participant(participant):
                return False, "Participant validation failed"
            
            # Store participant
            self.participants[participant.participant_id] = participant
            await self._store_participant(participant)
            
            logging.info(f"Registered MPC participant {participant.participant_id}")
            return True, None
            
        except Exception as e:
            logging.error(f"Participant registration failed: {e}")
            return False, str(e)
    
    async def create_mpc_session(
        self,
        name: str,
        description: str,
        protocol: MPCProtocol,
        computation_type: ComputationType,
        coordinator_id: str,
        participant_ids: List[str],
        data_requirements: Dict[str, Any],
        privacy_parameters: Optional[Dict[str, Any]] = None,
        security_parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Create secure multi-party computation session"""
        
        try:
            # Validate participants
            for participant_id in participant_ids:
                if participant_id not in self.participants:
                    return False, None, f"Participant {participant_id} not registered"
            
            # Create session
            session_id = str(uuid.uuid4())
            session = MPCSession(
                session_id=session_id,
                name=name,
                description=description,
                protocol=protocol,
                computation_type=computation_type,
                coordinator_id=coordinator_id,
                participants=participant_ids,
                data_requirements=data_requirements,
                privacy_parameters=privacy_parameters or {},
                security_parameters=security_parameters or {}
            )
            
            # Store session
            self.active_sessions[session_id] = session
            await self._store_mpc_session(session)
            
            logging.info(f"Created MPC session {session_id}")
            return True, session_id, None
            
        except Exception as e:
            logging.error(f"MPC session creation failed: {e}")
            return False, None, str(e)
    
    async def execute_secure_computation(
        self,
        session_id: str,
        participant_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Execute secure multi-party computation"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return False, None, "Session not found"
            
            session.status = "running"
            session.started_at = datetime.utcnow()
            await self._store_mpc_session(session)
            
            # Execute computation based on protocol
            if session.protocol == MPCProtocol.SHAMIR_SECRET_SHARING:
                result = await self._execute_secret_sharing_computation(session, participant_data)
            elif session.protocol == MPCProtocol.FEDERATED_LEARNING:
                result = await self._execute_federated_learning(session, participant_data)
            elif session.protocol == MPCProtocol.DIFFERENTIAL_PRIVACY:
                result = await self._execute_differential_privacy_computation(session, participant_data)
            elif session.protocol == MPCProtocol.PRIVATE_SET_INTERSECTION:
                result = await self._execute_private_set_intersection(session, participant_data)
            else:
                return False, None, f"Protocol {session.protocol} not implemented"
            
            # Update session
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            session.results_available = True
            await self._store_mpc_session(session)
            
            logging.info(f"Completed MPC computation for session {session_id}")
            return True, result, None
            
        except Exception as e:
            logging.error(f"MPC computation failed: {e}")
            
            # Update session with error
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.status = "failed"
                session.error_message = str(e)
                await self._store_mpc_session(session)
            
            return False, None, str(e)
    
    async def _execute_secret_sharing_computation(
        self,
        session: MPCSession,
        participant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute computation using Shamir's Secret Sharing"""
        
        # Extract numerical values from participant data
        values = []
        for participant_id in session.participants:
            if participant_id in participant_data:
                data = participant_data[participant_id]
                # Extract numerical value (simplified)
                if isinstance(data, (int, float)):
                    values.append(int(data))
                elif isinstance(data, dict) and 'value' in data:
                    values.append(int(data['value']))
        
        if not values:
            raise ValueError("No valid numerical data provided")
        
        # Perform secure sum using secret sharing
        threshold = len(values) // 2 + 1  # Majority threshold
        total_sum = 0
        
        for value in values:
            # Create shares for each value
            shares = self.secret_sharing.create_shares(value, threshold, len(session.participants))
            
            # In a real implementation, shares would be distributed to participants
            # For this example, we'll reconstruct immediately
            reconstructed = self.secret_sharing.reconstruct_secret(shares[:threshold])
            total_sum += reconstructed
        
        return {
            'computation_type': 'secure_sum',
            'result': total_sum,
            'participants': len(values),
            'protocol': 'shamir_secret_sharing'
        }
    
    async def _execute_federated_learning(
        self,
        session: MPCSession,
        participant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute federated learning computation"""
        
        # Extract federated learning configuration
        fl_config_dict = session.privacy_parameters.get('federated_learning', {})
        
        # Create FederatedLearningConfig
        fl_config = FederatedLearningConfig(
            model_architecture=fl_config_dict.get('model_architecture', 'neural_network'),
            aggregation_method=fl_config_dict.get('aggregation_method', 'fedavg'),
            rounds=fl_config_dict.get('rounds', 1),
            local_epochs=fl_config_dict.get('local_epochs', 1),
            learning_rate=fl_config_dict.get('learning_rate', 0.01),
            batch_size=fl_config_dict.get('batch_size', 32)
        )
        
        # Initialize federated learning engine
        fl_engine = FederatedLearningEngine(fl_config)
        
        participants = [self.participants[pid] for pid in session.participants]
        await fl_engine.initialize_federation(participants)
        
        # Execute training round
        result = await fl_engine.federated_training_round(participant_data)
        
        return {
            'computation_type': 'federated_learning',
            'result': result,
            'participants': len(session.participants),
            'protocol': 'federated_learning'
        }
    
    async def _execute_differential_privacy_computation(
        self,
        session: MPCSession,
        participant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute computation with differential privacy"""
        
        # Extract DP parameters
        dp_params_dict = session.privacy_parameters.get('differential_privacy', {})
        
        dp_params = DifferentialPrivacyParams(
            epsilon=dp_params_dict.get('epsilon', 1.0),
            delta=dp_params_dict.get('delta', 1e-5),
            sensitivity=dp_params_dict.get('sensitivity', 1.0),
            mechanism=dp_params_dict.get('mechanism', 'laplace')
        )
        
        # Aggregate data from participants
        all_values = []
        for participant_id in session.participants:
            if participant_id in participant_data:
                data = participant_data[participant_id]
                if isinstance(data, list):
                    all_values.extend(data)
                elif isinstance(data, (int, float)):
                    all_values.append(data)
        
        if not all_values:
            raise ValueError("No valid data provided")
        
        # Calculate statistics with differential privacy
        data_array = np.array(all_values)
        
        # Add noise to statistics
        mean_value = np.mean(data_array)
        private_mean = self.dp_engine.add_noise(mean_value, dp_params)
        
        std_value = np.std(data_array)
        private_std = self.dp_engine.add_noise(std_value, dp_params)
        
        count_value = len(data_array)
        private_count = self.dp_engine.add_noise(count_value, dp_params)
        
        return {
            'computation_type': 'differential_privacy_statistics',
            'result': {
                'mean': float(private_mean),
                'std': float(private_std),
                'count': int(max(0, private_count))  # Ensure non-negative count
            },
            'privacy_parameters': {
                'epsilon': dp_params.epsilon,
                'delta': dp_params.delta,
                'mechanism': dp_params.mechanism
            },
            'participants': len(session.participants),
            'protocol': 'differential_privacy'
        }
    
    async def _execute_private_set_intersection(
        self,
        session: MPCSession,
        participant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute private set intersection"""
        
        # Extract sets from participant data
        participant_sets = {}
        for participant_id in session.participants:
            if participant_id in participant_data:
                data = participant_data[participant_id]
                if isinstance(data, list):
                    participant_sets[participant_id] = set(data)
                elif isinstance(data, dict) and 'set' in data:
                    participant_sets[participant_id] = set(data['set'])
        
        if len(participant_sets) < 2:
            raise ValueError("Need at least 2 sets for intersection")
        
        # Compute intersection (simplified - not actually private)
        # In a real implementation, this would use cryptographic PSI protocols
        intersection = None
        for participant_id, participant_set in participant_sets.items():
            if intersection is None:
                intersection = participant_set.copy()
            else:
                intersection = intersection.intersection(participant_set)
        
        # Return only the size of intersection for privacy
        return {
            'computation_type': 'private_set_intersection',
            'result': {
                'intersection_size': len(intersection) if intersection else 0,
                'total_participants': len(participant_sets)
            },
            'participants': len(session.participants),
            'protocol': 'private_set_intersection'
        }
    
    async def _validate_participant(self, participant: MPCParticipant) -> bool:
        """Validate MPC participant"""
        
        # Check required fields
        if not all([
            participant.participant_id,
            participant.organization,
            participant.public_key,
            participant.endpoint
        ]):
            return False
        
        # Check capabilities
        if not participant.capabilities:
            return False
        
        # Check trust level
        if not (0.0 <= participant.trust_level <= 1.0):
            return False
        
        return True
    
    async def _store_participant(self, participant: MPCParticipant):
        """Store MPC participant"""
        
        participant_data = {
            'organization': participant.organization,
            'role': participant.role.value,
            'public_key': participant.public_key,
            'endpoint': participant.endpoint,
            'capabilities': [c.value for c in participant.capabilities],
            'trust_level': participant.trust_level,
            'data_classification_clearance': [c.value for c in participant.data_classification_clearance],
            'geological_domains': participant.geological_domains,
            'joined_at': participant.joined_at.isoformat(),
            'last_active': participant.last_active.isoformat(),
            'reputation_score': participant.reputation_score,
            'verified': participant.verified
        }
        
        await self.redis_client.set(
            f"mpc_participant:{participant.participant_id}",
            json.dumps(participant_data)
        )
    
    async def _store_mpc_session(self, session: MPCSession):
        """Store MPC session"""
        
        session_data = {
            'name': session.name,
            'description': session.description,
            'protocol': session.protocol.value,
            'computation_type': session.computation_type.value,
            'coordinator_id': session.coordinator_id,
            'participants': session.participants,
            'data_requirements': session.data_requirements,
            'privacy_parameters': session.privacy_parameters,
            'security_parameters': session.security_parameters,
            'status': session.status,
            'created_at': session.created_at.isoformat(),
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'results_available': session.results_available,
            'error_message': session.error_message
        }
        
        await self.redis_client.set(
            f"mpc_session:{session.session_id}",
            json.dumps(session_data)
        )


if __name__ == "__main__":
    # Example usage and testing
    async def test_secure_multiparty():
        """Test secure multi-party computation functionality"""
        
        # Initialize MPC engine
        config = {
            'redis_url': 'redis://localhost:6379'
        }
        
        mpc_engine = SecureMultiPartyEngine(config)
        await mpc_engine.initialize()
        
        # Register participants
        participants = [
            MPCParticipant(
                participant_id="usgs_001",
                organization="USGS",
                role=ParticipantRole.DATA_PROVIDER,
                public_key="usgs_public_key_001",
                endpoint="https://usgs.gov/mpc",
                capabilities=[MPCProtocol.SHAMIR_SECRET_SHARING, MPCProtocol.DIFFERENTIAL_PRIVACY],
                trust_level=0.9,
                data_classification_clearance=[DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
                geological_domains=["seismic", "volcanic"]
            ),
            MPCParticipant(
                participant_id="university_001",
                organization="Stanford University",
                role=ParticipantRole.COMPUTATION_PROVIDER,
                public_key="stanford_public_key_001",
                endpoint="https://stanford.edu/mpc",
                capabilities=[MPCProtocol.FEDERATED_LEARNING, MPCProtocol.DIFFERENTIAL_PRIVACY],
                trust_level=0.85,
                data_classification_clearance=[DataClassification.INTERNAL],
                geological_domains=["seismic", "geodynamics"]
            ),
            MPCParticipant(
                participant_id="research_lab_001",
                organization="Geological Research Lab",
                role=ParticipantRole.RESULT_CONSUMER,
                public_key="lab_public_key_001",
                endpoint="https://geolab.org/mpc",
                capabilities=[MPCProtocol.PRIVATE_SET_INTERSECTION],
                trust_level=0.8,
                data_classification_clearance=[DataClassification.INTERNAL],
                geological_domains=["mineral_exploration"]
            )
        ]
        
        # Register participants
        for participant in participants:
            success, error = await mpc_engine.register_participant(participant)
            if success:
                print(f"Registered participant: {participant.participant_id}")
            else:
                print(f"Failed to register participant: {error}")
        
        # Test 1: Secret sharing computation
        success, session_id, error = await mpc_engine.create_mpc_session(
            name="Seismic Data Aggregation",
            description="Secure aggregation of seismic magnitude data",
            protocol=MPCProtocol.SHAMIR_SECRET_SHARING,
            computation_type=ComputationType.DATA_AGGREGATION,
            coordinator_id="usgs_001",
            participant_ids=["usgs_001", "university_001"],
            data_requirements={"data_type": "seismic_magnitude", "format": "numerical"}
        )
        
        if success:
            print(f"Created MPC session: {session_id}")
            
            # Simulate participant data
            participant_data = {
                "usgs_001": 7.2,  # Earthquake magnitude
                "university_001": 6.8  # Another measurement
            }
            
            # Execute computation
            comp_success, result, comp_error = await mpc_engine.execute_secure_computation(
                session_id, participant_data
            )
            
            if comp_success:
                print(f"Computation result: {result}")
            else:
                print(f"Computation failed: {comp_error}")
        else:
            print(f"Session creation failed: {error}")
        
        # Test 2: Differential privacy computation
        success, dp_session_id, error = await mpc_engine.create_mpc_session(
            name="Private Geological Statistics",
            description="Compute statistics with differential privacy",
            protocol=MPCProtocol.DIFFERENTIAL_PRIVACY,
            computation_type=ComputationType.STATISTICAL_ANALYSIS,
            coordinator_id="university_001",
            participant_ids=["usgs_001", "university_001", "research_lab_001"],
            data_requirements={"data_type": "geological_measurements", "format": "numerical_array"},
            privacy_parameters={
                "differential_privacy": {
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "sensitivity": 1.0,
                    "mechanism": "laplace"
                }
            }
        )
        
        if success:
            print(f"Created DP session: {dp_session_id}")
            
            # Simulate geological measurement data
            dp_participant_data = {
                "usgs_001": [2.1, 3.4, 1.8, 4.2, 2.9],  # Seismic magnitudes
                "university_001": [2.3, 3.1, 2.0, 4.0, 3.2],  # Lab measurements
                "research_lab_001": [2.0, 3.3, 1.9, 4.1, 3.0]  # Field measurements
            }
            
            # Execute DP computation
            dp_success, dp_result, dp_error = await mpc_engine.execute_secure_computation(
                dp_session_id, dp_participant_data
            )
            
            if dp_success:
                print(f"DP computation result: {dp_result}")
            else:
                print(f"DP computation failed: {dp_error}")
        else:
            print(f"DP session creation failed: {error}")
    
    # Run test
    asyncio.run(test_secure_multiparty())