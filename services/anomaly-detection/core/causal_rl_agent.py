"""
Causal reinforcement learning agent for geological anomaly detection

This module implements a reinforcement learning agent that learns causal
relationships in geological data and uses them for anomaly detection.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# Causal inference libraries
from causal_learn.search.ConstraintBased.PC import pc
from causal_learn.search.ScoreBased.GES import ges
from causal_learn.utils.cit import CIT
from dowhy import CausalModel
import networkx as nx

from core.config import Settings
from core.database import DatabaseManager
from models.schemas import (
    GeologicalDataStream, DetectedAnomaly, AnomalyType, AnomalySeverity,
    CausalRelationship, ReinforcementLearningState, ReinforcementLearningAction
)

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class CausalNetwork(nn.Module):
    """Neural network for learning causal relationships"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(CausalNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder for state representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Causal structure learning head
        self.causal_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim * input_dim),  # Adjacency matrix
            nn.Sigmoid()
        )
        
        # Intervention effect prediction head
        self.intervention_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim)  # Predicted effects
        )
    
    def forward(self, x, intervention=None):
        encoded = self.encoder(x)
        
        # Predict causal structure (adjacency matrix)
        causal_structure = self.causal_head(encoded)
        causal_structure = causal_structure.view(-1, self.input_dim, self.input_dim)
        
        # Predict intervention effects if intervention is provided
        if intervention is not None:
            combined = torch.cat([encoded, intervention], dim=1)
            intervention_effects = self.intervention_head(combined)
            return causal_structure, intervention_effects
        
        return causal_structure


class DQNAgent(nn.Module):
    """Deep Q-Network agent for causal anomaly detection"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 4, action_dim)
        )
    
    def forward(self, state):
        features = self.q_network[:-1](state)  # All layers except the last
        
        # Dueling DQN
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def select_action(self, state, epsilon=0.1):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax().item()


class CausalReinforcementLearningAgent:
    """Causal reinforcement learning agent for geological anomaly detection"""
    
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager
        
        # Model components
        self.causal_network: Optional[CausalNetwork] = None
        self.dqn_agent: Optional[DQNAgent] = None
        self.target_network: Optional[DQNAgent] = None
        
        # Causal discovery
        self.causal_graph: Optional[nx.DiGraph] = None
        self.causal_relationships: List[CausalRelationship] = []
        
        # RL components
        self.replay_buffer = deque(maxlen=settings.rl_memory_size)
        self.epsilon = settings.rl_exploration_rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = settings.rl_learning_rate
        
        # Action space definition
        self.action_space = {
            0: "investigate_correlation",
            1: "test_causality",
            2: "flag_anomaly",
            3: "request_more_data",
            4: "update_causal_model",
            5: "ignore_observation"
        }
        
        # State and reward tracking
        self.current_state: Optional[torch.Tensor] = None
        self.state_history: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        
        # Training parameters
        self.batch_size = 32
        self.target_update_frequency = 100
        self.training_step = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model persistence
        self.model_save_path = os.path.join(settings.model_cache_dir, "causal_rl_agent")
        os.makedirs(self.model_save_path, exist_ok=True)
    
    async def initialize(self):
        """Initialize the causal RL agent"""
        try:
            logger.info("Initializing causal reinforcement learning agent...")
            
            # Initialize networks
            await self._initialize_networks()
            
            # Load or discover initial causal relationships
            await self._initialize_causal_knowledge()
            
            # Load saved models if available
            await self._load_models()
            
            logger.info("Causal RL agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize causal RL agent: {e}")
            raise
    
    async def detect_anomalies(
        self,
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using causal reinforcement learning"""
        try:
            if not data_stream.data_points:
                return []
            
            # Prepare state representation
            state_data = await self._prepare_state_data(data_stream)
            
            detected_anomalies = []
            
            # Process each data point with RL agent
            for i, data_point in enumerate(data_stream.data_points):
                # Create state for current observation
                current_state = self._create_state_vector(state_data, i)
                
                # Agent selects action
                action = self._select_action(current_state)
                
                # Execute action and get reward
                anomaly, reward = await self._execute_action(
                    action, data_point, current_state, data_stream
                )
                
                if anomaly:
                    detected_anomalies.append(anomaly)
                
                # Store experience for learning
                if self.current_state is not None:
                    experience = Experience(
                        state=self.current_state,
                        action=action,
                        reward=reward,
                        next_state=current_state,
                        done=i == len(data_stream.data_points) - 1
                    )
                    self.replay_buffer.append(experience)
                
                self.current_state = current_state
                self.reward_history.append(reward)
            
            # Train the agent
            if len(self.replay_buffer) >= self.batch_size:
                await self._train_agent()
            
            # Update causal relationships based on new data
            await self._update_causal_relationships(data_stream, detected_anomalies)
            
            return detected_anomalies
            
        except Exception as e:
            logger.error(f"Causal RL anomaly detection failed: {e}")
            return []
    
    async def _prepare_state_data(self, data_stream: GeologicalDataStream) -> pd.DataFrame:
        """Prepare state data for RL agent"""
        try:
            # Convert data stream to DataFrame
            data_records = []
            for point in data_stream.data_points:
                record = {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'quality_score': point.quality_score,
                    'measurement_type': point.measurement_type,
                    'sensor_type': point.sensor_type,
                    'latitude': point.location.get('lat', 0.0),
                    'longitude': point.location.get('lon', 0.0),
                    'elevation': point.location.get('elevation', 0.0),
                    'depth': point.location.get('depth', 0.0)
                }
                data_records.append(record)
            
            df = pd.DataFrame(data_records)
            
            # Add temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Add rolling statistics
            window_size = min(10, len(df))
            if window_size > 1:
                df['rolling_mean'] = df['value'].rolling(window=window_size, min_periods=1).mean()
                df['rolling_std'] = df['value'].rolling(window=window_size, min_periods=1).std().fillna(0)
                df['rolling_min'] = df['value'].rolling(window=window_size, min_periods=1).min()
                df['rolling_max'] = df['value'].rolling(window=window_size, min_periods=1).max()
            else:
                df['rolling_mean'] = df['value']
                df['rolling_std'] = 0.0
                df['rolling_min'] = df['value']
                df['rolling_max'] = df['value']
            
            # Add causal features based on discovered relationships
            df = await self._add_causal_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"State data preparation failed: {e}")
            return pd.DataFrame()
    
    def _create_state_vector(self, state_data: pd.DataFrame, index: int) -> torch.Tensor:
        """Create state vector for RL agent"""
        try:
            if state_data.empty or index >= len(state_data):
                return torch.zeros(20).to(self.device)  # Default state size
            
            row = state_data.iloc[index]
            
            # Basic features
            state_features = [
                row.get('value', 0.0),
                row.get('quality_score', 0.0),
                row.get('latitude', 0.0),
                row.get('longitude', 0.0),
                row.get('elevation', 0.0),
                row.get('depth', 0.0),
                row.get('hour', 0.0) / 24.0,  # Normalize
                row.get('day_of_week', 0.0) / 7.0,  # Normalize
                row.get('month', 0.0) / 12.0,  # Normalize
                row.get('rolling_mean', 0.0),
                row.get('rolling_std', 0.0),
                row.get('rolling_min', 0.0),
                row.get('rolling_max', 0.0)
            ]
            
            # Add causal features
            causal_features = []
            for rel in self.causal_relationships[:7]:  # Limit to top 7 relationships
                cause_val = row.get(f'causal_{rel.cause_variable}', 0.0)
                effect_val = row.get(f'causal_{rel.effect_variable}', 0.0)
                causal_features.append(cause_val * rel.strength)
            
            # Pad causal features to fixed size
            while len(causal_features) < 7:
                causal_features.append(0.0)
            
            state_features.extend(causal_features)
            
            # Ensure fixed size
            state_vector = torch.FloatTensor(state_features[:20]).to(self.device)
            if len(state_vector) < 20:
                padding = torch.zeros(20 - len(state_vector)).to(self.device)
                state_vector = torch.cat([state_vector, padding])
            
            return state_vector
            
        except Exception as e:
            logger.error(f"State vector creation failed: {e}")
            return torch.zeros(20).to(self.device)
    
    def _select_action(self, state: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy"""
        if self.dqn_agent is None:
            return random.randint(0, len(self.action_space) - 1)
        
        return self.dqn_agent.select_action(state, self.epsilon)
    
    async def _execute_action(
        self,
        action: int,
        data_point,
        state: torch.Tensor,
        data_stream: GeologicalDataStream
    ) -> Tuple[Optional[DetectedAnomaly], float]:
        """Execute selected action and return anomaly (if any) and reward"""
        try:
            action_name = self.action_space.get(action, "unknown")
            anomaly = None
            reward = 0.0
            
            if action_name == "investigate_correlation":
                # Investigate correlations with other variables
                correlation_strength = await self._investigate_correlations(data_point, data_stream)
                reward = correlation_strength * 0.5
                
            elif action_name == "test_causality":
                # Test causal relationships
                causality_evidence = await self._test_causality(data_point, data_stream)
                reward = causality_evidence * 0.7
                
            elif action_name == "flag_anomaly":
                # Flag as anomaly
                anomaly_score = await self._calculate_anomaly_score(data_point, state)
                if anomaly_score > 0.7:  # High confidence threshold
                    anomaly = self._create_causal_anomaly(data_point, anomaly_score, action_name)
                    reward = 1.0  # High reward for correct anomaly detection
                else:
                    reward = -0.5  # Penalty for false positive
                
            elif action_name == "request_more_data":
                # Request additional data (simulated)
                data_quality = data_point.quality_score
                if data_quality < 0.8:
                    reward = 0.3  # Good decision to request more data
                else:
                    reward = -0.1  # Unnecessary request
                
            elif action_name == "update_causal_model":
                # Update causal model with new evidence
                model_improvement = await self._update_causal_model(data_point, data_stream)
                reward = model_improvement * 0.6
                
            elif action_name == "ignore_observation":
                # Ignore the observation
                anomaly_score = await self._calculate_anomaly_score(data_point, state)
                if anomaly_score < 0.3:  # Correctly ignored normal observation
                    reward = 0.2
                else:
                    reward = -0.8  # Penalty for missing anomaly
            
            return anomaly, reward
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return None, -0.1  # Small penalty for errors
    
    async def _investigate_correlations(self, data_point, data_stream: GeologicalDataStream) -> float:
        """Investigate correlations with other variables"""
        try:
            # Get recent data for correlation analysis
            recent_data = await self.db_manager.query_geological_data(
                measurement_type="all",
                start_time=data_point.timestamp - timedelta(hours=1),
                end_time=data_point.timestamp + timedelta(hours=1)
            )
            
            if len(recent_data) < 10:
                return 0.1  # Insufficient data
            
            df = pd.DataFrame(recent_data)
            
            # Calculate correlations
            correlations = []
            for measurement_type in df['measurement_type'].unique():
                if measurement_type != data_point.measurement_type:
                    type_data = df[df['measurement_type'] == measurement_type]['value']
                    current_data = df[df['measurement_type'] == data_point.measurement_type]['value']
                    
                    if len(type_data) > 5 and len(current_data) > 5:
                        # Align data lengths
                        min_len = min(len(type_data), len(current_data))
                        correlation = np.corrcoef(
                            type_data.iloc[:min_len], 
                            current_data.iloc[:min_len]
                        )[0, 1]
                        
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))
            
            return np.mean(correlations) if correlations else 0.1
            
        except Exception as e:
            logger.error(f"Correlation investigation failed: {e}")
            return 0.1
    
    async def _test_causality(self, data_point, data_stream: GeologicalDataStream) -> float:
        """Test causal relationships"""
        try:
            # Simple causality test based on temporal precedence and correlation
            causality_evidence = 0.0
            
            for rel in self.causal_relationships:
                if rel.effect_variable == data_point.measurement_type:
                    # Check if cause variable shows expected pattern
                    cause_data = await self.db_manager.query_geological_data(
                        measurement_type=rel.cause_variable,
                        start_time=data_point.timestamp - timedelta(seconds=rel.lag_time or 300),
                        end_time=data_point.timestamp
                    )
                    
                    if cause_data:
                        # Check if cause preceded effect
                        cause_values = [d['value'] for d in cause_data]
                        if len(cause_values) > 0:
                            cause_strength = np.mean(cause_values)
                            expected_effect = cause_strength * rel.strength
                            actual_effect = data_point.value
                            
                            # Calculate how well the causal relationship explains the observation
                            if expected_effect > 0:
                                explanation_quality = 1.0 - abs(actual_effect - expected_effect) / expected_effect
                                causality_evidence += max(0, explanation_quality) * rel.confidence
            
            return min(1.0, causality_evidence)
            
        except Exception as e:
            logger.error(f"Causality testing failed: {e}")
            return 0.1
    
    async def _calculate_anomaly_score(self, data_point, state: torch.Tensor) -> float:
        """Calculate anomaly score for a data point"""
        try:
            # Use causal network to predict expected value
            if self.causal_network is not None:
                with torch.no_grad():
                    causal_structure = self.causal_network(state.unsqueeze(0))
                    # Use causal structure to compute expected value
                    # This is simplified - in practice, you'd use the full causal model
                    expected_value = torch.mean(causal_structure).item()
            else:
                expected_value = 0.5  # Default expectation
            
            # Calculate deviation from expected value
            actual_value = data_point.value
            deviation = abs(actual_value - expected_value)
            
            # Normalize to [0, 1] range
            max_expected_deviation = 10.0  # Domain-specific threshold
            anomaly_score = min(1.0, deviation / max_expected_deviation)
            
            # Adjust based on data quality
            quality_factor = data_point.quality_score
            adjusted_score = anomaly_score * quality_factor
            
            return adjusted_score
            
        except Exception as e:
            logger.error(f"Anomaly score calculation failed: {e}")
            return 0.5
    
    def _create_causal_anomaly(
        self,
        data_point,
        anomaly_score: float,
        detection_method: str
    ) -> DetectedAnomaly:
        """Create a causal anomaly object"""
        
        # Determine severity
        if anomaly_score > 0.9:
            severity = AnomalySeverity.CRITICAL
        elif anomaly_score > 0.8:
            severity = AnomalySeverity.HIGH
        elif anomaly_score > 0.7:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
        
        # Create causal context description
        relevant_relationships = [
            rel for rel in self.causal_relationships
            if rel.effect_variable == data_point.measurement_type or 
               rel.cause_variable == data_point.measurement_type
        ]
        
        causal_context = f"Causal anomaly detected using {detection_method}. "
        if relevant_relationships:
            causal_context += f"Relevant causal relationships: "
            for rel in relevant_relationships[:3]:  # Limit to top 3
                causal_context += f"{rel.cause_variable} -> {rel.effect_variable} (strength: {rel.strength:.2f}), "
        
        return DetectedAnomaly(
            anomaly_id=str(uuid.uuid4()),
            detection_timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.CAUSAL,
            severity=severity,
            temporal_extent={
                "start": data_point.timestamp,
                "end": data_point.timestamp
            },
            spatial_extent=data_point.location,
            affected_variables=[data_point.measurement_type],
            anomaly_score=anomaly_score,
            deviation_magnitude=anomaly_score,  # Simplified
            baseline_comparison={
                "causal_relationships_count": len(relevant_relationships),
                "detection_method": detection_method,
                "rl_action": detection_method
            },
            geological_context=causal_context,
            detection_method=f"causal_rl_{detection_method}",
            model_version="1.0.0",
            processing_time=0.0
        )
    
    async def _update_causal_model(self, data_point, data_stream: GeologicalDataStream) -> float:
        """Update causal model with new evidence"""
        try:
            # This is a simplified update - in practice, you'd use more sophisticated methods
            improvement = 0.0
            
            # Add new data point to causal discovery
            if len(data_stream.data_points) > 10:
                # Prepare data for causal discovery
                causal_data = []
                for point in data_stream.data_points[-10:]:  # Use last 10 points
                    causal_data.append({
                        'value': point.value,
                        'measurement_type': point.measurement_type,
                        'timestamp': point.timestamp.timestamp(),
                        'quality': point.quality_score
                    })
                
                df = pd.DataFrame(causal_data)
                
                # Simple causal discovery (correlation-based)
                measurement_types = df['measurement_type'].unique()
                if len(measurement_types) > 1:
                    for i, cause_type in enumerate(measurement_types):
                        for j, effect_type in enumerate(measurement_types):
                            if i != j:
                                cause_data = df[df['measurement_type'] == cause_type]['value']
                                effect_data = df[df['measurement_type'] == effect_type]['value']
                                
                                if len(cause_data) > 3 and len(effect_data) > 3:
                                    # Calculate correlation (simplified causality)
                                    min_len = min(len(cause_data), len(effect_data))
                                    correlation = np.corrcoef(
                                        cause_data.iloc[:min_len],
                                        effect_data.iloc[:min_len]
                                    )[0, 1]
                                    
                                    if not np.isnan(correlation) and abs(correlation) > 0.5:
                                        # Update or add causal relationship
                                        existing_rel = None
                                        for rel in self.causal_relationships:
                                            if (rel.cause_variable == cause_type and 
                                                rel.effect_variable == effect_type):
                                                existing_rel = rel
                                                break
                                        
                                        if existing_rel:
                                            # Update existing relationship
                                            old_strength = existing_rel.strength
                                            existing_rel.strength = 0.9 * existing_rel.strength + 0.1 * abs(correlation)
                                            improvement += abs(existing_rel.strength - old_strength)
                                        else:
                                            # Add new relationship
                                            new_rel = CausalRelationship(
                                                cause_variable=cause_type,
                                                effect_variable=effect_type,
                                                strength=abs(correlation),
                                                confidence=0.7,  # Initial confidence
                                                lag_time=300.0  # 5 minutes default
                                            )
                                            self.causal_relationships.append(new_rel)
                                            improvement += 0.5
            
            return min(1.0, improvement)
            
        except Exception as e:
            logger.error(f"Causal model update failed: {e}")
            return 0.1
    
    async def _train_agent(self):
        """Train the DQN agent using experience replay"""
        try:
            if len(self.replay_buffer) < self.batch_size or self.dqn_agent is None:
                return
            
            # Sample batch from replay buffer
            batch = random.sample(self.replay_buffer, self.batch_size)
            
            states = torch.stack([exp.state for exp in batch])
            actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
            next_states = torch.stack([exp.next_state for exp in batch])
            dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
            
            # Current Q values
            current_q_values = self.dqn_agent(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            if self.target_network is not None:
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0]
                    target_q_values = rewards + (0.99 * next_q_values * ~dones)
            else:
                target_q_values = rewards
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            optimizer = optim.Adam(self.dqn_agent.parameters(), lr=self.learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target network periodically
            self.training_step += 1
            if self.training_step % self.target_update_frequency == 0:
                if self.target_network is not None:
                    self.target_network.load_state_dict(self.dqn_agent.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        except Exception as e:
            logger.error(f"Agent training failed: {e}")
    
    async def _initialize_networks(self):
        """Initialize neural networks"""
        try:
            state_dim = 20  # Fixed state dimension
            action_dim = len(self.action_space)
            
            # Initialize DQN agent
            self.dqn_agent = DQNAgent(state_dim, action_dim)
            self.dqn_agent.to(self.device)
            
            # Initialize target network
            self.target_network = DQNAgent(state_dim, action_dim)
            self.target_network.to(self.device)
            self.target_network.load_state_dict(self.dqn_agent.state_dict())
            
            # Initialize causal network
            self.causal_network = CausalNetwork(input_dim=state_dim)
            self.causal_network.to(self.device)
            
            logger.info("Neural networks initialized")
            
        except Exception as e:
            logger.error(f"Network initialization failed: {e}")
            raise
    
    async def _initialize_causal_knowledge(self):
        """Initialize causal knowledge base"""
        try:
            # Load basic geological causal relationships
            basic_relationships = [
                CausalRelationship(
                    cause_variable="seismic",
                    effect_variable="deformation",
                    strength=0.8,
                    confidence=0.9,
                    lag_time=60.0  # 1 minute
                ),
                CausalRelationship(
                    cause_variable="thermal",
                    effect_variable="gas_emissions",
                    strength=0.7,
                    confidence=0.8,
                    lag_time=300.0  # 5 minutes
                ),
                CausalRelationship(
                    cause_variable="pressure",
                    effect_variable="seismic",
                    strength=0.6,
                    confidence=0.7,
                    lag_time=120.0  # 2 minutes
                )
            ]
            
            self.causal_relationships = basic_relationships
            
            # Create initial causal graph
            self.causal_graph = nx.DiGraph()
            for rel in self.causal_relationships:
                self.causal_graph.add_edge(
                    rel.cause_variable,
                    rel.effect_variable,
                    weight=rel.strength,
                    confidence=rel.confidence
                )
            
            logger.info(f"Initialized with {len(self.causal_relationships)} causal relationships")
            
        except Exception as e:
            logger.error(f"Causal knowledge initialization failed: {e}")
    
    async def _add_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add causal features to dataframe"""
        try:
            # Add features based on causal relationships
            for rel in self.causal_relationships:
                cause_col = f"causal_{rel.cause_variable}"
                effect_col = f"causal_{rel.effect_variable}"
                
                # Simple feature: strength-weighted values
                if rel.cause_variable in df.columns:
                    df[cause_col] = df[rel.cause_variable] * rel.strength
                else:
                    df[cause_col] = 0.0
                
                if rel.effect_variable in df.columns:
                    df[effect_col] = df[rel.effect_variable] * rel.strength
                else:
                    df[effect_col] = 0.0
            
            return df
            
        except Exception as e:
            logger.error(f"Causal feature addition failed: {e}")
            return df
    
    async def _update_causal_relationships(self, data_stream, anomalies):
        """Update causal relationships based on detected anomalies"""
        try:
            # Store causal relationships in database
            for rel in self.causal_relationships:
                await self.db_manager.store_causal_relationship(
                    cause=rel.cause_variable,
                    effect=rel.effect_variable,
                    strength=rel.strength,
                    confidence=rel.confidence,
                    context={"lag_time": rel.lag_time}
                )
            
        except Exception as e:
            logger.error(f"Causal relationship update failed: {e}")
    
    async def _load_models(self):
        """Load saved models"""
        try:
            dqn_path = os.path.join(self.model_save_path, "dqn_agent.pth")
            causal_path = os.path.join(self.model_save_path, "causal_network.pth")
            
            if os.path.exists(dqn_path) and self.dqn_agent is not None:
                self.dqn_agent.load_state_dict(torch.load(dqn_path, map_location=self.device))
                logger.info("DQN agent loaded from saved model")
            
            if os.path.exists(causal_path) and self.causal_network is not None:
                self.causal_network.load_state_dict(torch.load(causal_path, map_location=self.device))
                logger.info("Causal network loaded from saved model")
            
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
    
    async def save_models(self):
        """Save trained models"""
        try:
            if self.dqn_agent is not None:
                dqn_path = os.path.join(self.model_save_path, "dqn_agent.pth")
                torch.save(self.dqn_agent.state_dict(), dqn_path)
            
            if self.causal_network is not None:
                causal_path = os.path.join(self.model_save_path, "causal_network.pth")
                torch.save(self.causal_network.state_dict(), causal_path)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the causal RL agent"""
        return {
            "models_initialized": self.dqn_agent is not None and self.causal_network is not None,
            "causal_relationships_count": len(self.causal_relationships),
            "replay_buffer_size": len(self.replay_buffer),
            "epsilon": self.epsilon,
            "training_steps": self.training_step,
            "action_space": self.action_space,
            "causal_graph_nodes": list(self.causal_graph.nodes()) if self.causal_graph else [],
            "causal_graph_edges": list(self.causal_graph.edges()) if self.causal_graph else [],
            "recent_rewards": self.reward_history[-10:] if self.reward_history else [],
            "average_reward": np.mean(self.reward_history) if self.reward_history else 0.0
        }