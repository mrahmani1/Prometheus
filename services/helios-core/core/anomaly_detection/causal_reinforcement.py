"""
Causal Reinforcement Learning for Geological Anomaly Detection

Implements causal reinforcement learning to discover causal relationships
in geological anomalies and learn optimal detection strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from sklearn.preprocessing import StandardScaler
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class CausalState:
    """State representation for causal reinforcement learning"""
    geological_variables: np.ndarray
    temporal_context: np.ndarray
    spatial_context: np.ndarray
    causal_history: List[Tuple[str, str, float]]  # (cause, effect, strength)
    anomaly_indicators: np.ndarray
    timestamp: float

@dataclass
class CausalAction:
    """Action in causal reinforcement learning"""
    action_type: str  # 'investigate', 'monitor', 'alert', 'ignore'
    target_variables: List[str]
    investigation_depth: float
    confidence_threshold: float
    time_horizon: float

@dataclass
class CausalReward:
    """Reward signal for causal learning"""
    detection_accuracy: float
    false_positive_penalty: float
    discovery_bonus: float
    temporal_efficiency: float
    causal_consistency: float
    total_reward: float

class CausalGraph:
    """Dynamic causal graph for geological processes"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.edge_strengths = {}
        self.temporal_lags = {}
        self.confidence_scores = {}
        
    def add_causal_relationship(self, cause: str, effect: str, strength: float,
                              lag: float = 0.0, confidence: float = 1.0):
        """Add or update causal relationship"""
        self.graph.add_edge(cause, effect)
        self.edge_strengths[(cause, effect)] = strength
        self.temporal_lags[(cause, effect)] = lag
        self.confidence_scores[(cause, effect)] = confidence
        
    def get_causal_parents(self, variable: str) -> List[Tuple[str, float, float]]:
        """Get causal parents of a variable with strengths and lags"""
        parents = []
        for parent in self.graph.predecessors(variable):
            strength = self.edge_strengths.get((parent, variable), 0.0)
            lag = self.temporal_lags.get((parent, variable), 0.0)
            parents.append((parent, strength, lag))
        return parents
    
    def get_causal_children(self, variable: str) -> List[Tuple[str, float, float]]:
        """Get causal children of a variable"""
        children = []
        for child in self.graph.successors(variable):
            strength = self.edge_strengths.get((variable, child), 0.0)
            lag = self.temporal_lags.get((variable, child), 0.0)
            children.append((child, strength, lag))
        return children
    
    def compute_causal_influence(self, source: str, target: str) -> float:
        """Compute total causal influence from source to target"""
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=3))
            total_influence = 0.0
            
            for path in paths:
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge_strength = self.edge_strengths.get((path[i], path[i+1]), 0.0)
                    path_strength *= edge_strength
                total_influence += path_strength
                
            return min(1.0, total_influence)
        except nx.NetworkXNoPath:
            return 0.0
    
    def update_from_data(self, data: Dict[str, np.ndarray], method: str = 'granger'):
        """Update causal graph from observational data"""
        if method == 'granger':
            self._update_granger_causality(data)
        elif method == 'pc_algorithm':
            self._update_pc_algorithm(data)
    
    def _update_granger_causality(self, data: Dict[str, np.ndarray]):
        """Update using Granger causality analysis"""
        variables = list(data.keys())
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Compute Granger causality from var1 to var2
                    causality_strength = self._compute_granger_causality(
                        data[var1], data[var2]
                    )
                    
                    if causality_strength > 0.1:  # Threshold for significance
                        self.add_causal_relationship(
                            var1, var2, causality_strength, confidence=0.8
                        )
    
    def _compute_granger_causality(self, x: np.ndarray, y: np.ndarray, 
                                 max_lag: int = 5) -> float:
        """Compute Granger causality from x to y"""
        if len(x) < max_lag + 10 or len(y) < max_lag + 10:
            return 0.0
        
        # Simple implementation of Granger causality test
        # In practice, would use more sophisticated methods
        
        # Fit autoregressive model for y
        y_lagged = self._create_lagged_matrix(y, max_lag)
        if y_lagged.shape[0] == 0:
            return 0.0
            
        y_target = y[max_lag:]
        
        # Model 1: y predicted by its own lags
        try:
            coeffs1 = np.linalg.lstsq(y_lagged, y_target, rcond=None)[0]
            pred1 = y_lagged @ coeffs1
            rss1 = np.sum((y_target - pred1) ** 2)
        except:
            return 0.0
        
        # Model 2: y predicted by its own lags + x lags
        x_lagged = self._create_lagged_matrix(x, max_lag)
        if x_lagged.shape[0] != y_lagged.shape[0]:
            return 0.0
            
        combined_lagged = np.hstack([y_lagged, x_lagged])
        
        try:
            coeffs2 = np.linalg.lstsq(combined_lagged, y_target, rcond=None)[0]
            pred2 = combined_lagged @ coeffs2
            rss2 = np.sum((y_target - pred2) ** 2)
        except:
            return 0.0
        
        # Granger causality strength
        if rss2 > 0:
            causality = max(0.0, (rss1 - rss2) / rss1)
            return causality
        
        return 0.0
    
    def _create_lagged_matrix(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Create matrix of lagged variables"""
        if len(data) <= max_lag:
            return np.array([]).reshape(0, max_lag)
            
        lagged_matrix = []
        for i in range(max_lag, len(data)):
            lagged_row = [data[i-lag-1] for lag in range(max_lag)]
            lagged_matrix.append(lagged_row)
        
        return np.array(lagged_matrix)

class CausalQNetwork(nn.Module):
    """Q-Network for causal reinforcement learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # State encoder
        encoder_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.state_encoder = nn.Sequential(*encoder_layers)
        
        # Causal attention mechanism
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=prev_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Q-value heads for different action types
        self.q_investigate = nn.Linear(prev_dim, action_dim)
        self.q_monitor = nn.Linear(prev_dim, action_dim)
        self.q_alert = nn.Linear(prev_dim, action_dim)
        self.q_ignore = nn.Linear(prev_dim, 1)
        
        # Value function
        self.value_head = nn.Linear(prev_dim, 1)
        
    def forward(self, state, causal_context=None):
        # Encode state
        encoded_state = self.state_encoder(state)
        
        # Apply causal attention if context provided
        if causal_context is not None:
            attended_state, attention_weights = self.causal_attention(
                encoded_state.unsqueeze(1), causal_context, causal_context
            )
            encoded_state = attended_state.squeeze(1)
        
        # Compute Q-values for different action types
        q_investigate = self.q_investigate(encoded_state)
        q_monitor = self.q_monitor(encoded_state)
        q_alert = self.q_alert(encoded_state)
        q_ignore = self.q_ignore(encoded_state)
        
        # State value
        state_value = self.value_head(encoded_state)
        
        return {
            'q_investigate': q_investigate,
            'q_monitor': q_monitor,
            'q_alert': q_alert,
            'q_ignore': q_ignore,
            'state_value': state_value
        }

class CausalReinforcementLearner:
    """
    Causal reinforcement learning for geological anomaly detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize causal graph
        self.causal_graph = CausalGraph()
        
        # Initialize neural networks
        state_dim = config.get('state_dim', 128)
        action_dim = config.get('action_dim', 32)
        hidden_dims = config.get('hidden_dims', [256, 128, 64])
        
        self.q_network = CausalQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = CausalQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # Experience replay
        self.replay_buffer = deque(maxlen=config.get('buffer_size', 10000))
        
        # Learning parameters
        self.epsilon = config.get('initial_epsilon', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 32)
        
        # Causal discovery parameters
        self.causal_discovery_interval = config.get('causal_discovery_interval', 100)
        self.causal_confidence_threshold = config.get('causal_confidence_threshold', 0.7)
        
        # Data storage for causal analysis
        self.observation_history = defaultdict(list)
        self.action_history = []
        self.reward_history = []
        
        # State preprocessing
        self.state_scaler = StandardScaler()
        
        self.step_count = 0
        
        logger.info("Initialized CausalReinforcementLearner")
    
    def encode_state(self, geological_data: Dict[str, np.ndarray], 
                    temporal_context: np.ndarray, spatial_context: np.ndarray,
                    anomaly_indicators: np.ndarray) -> CausalState:
        """Encode geological observations into causal state"""
        # Flatten geological variables
        geological_vars = np.concatenate([
            data.flatten() for data in geological_data.values()
        ])
        
        # Get causal history
        causal_history = []
        for var in geological_data.keys():
            parents = self.causal_graph.get_causal_parents(var)
            for parent, strength, lag in parents:
                causal_history.append((parent, var, strength))
        
        state = CausalState(
            geological_variables=geological_vars,
            temporal_context=temporal_context,
            spatial_context=spatial_context,
            causal_history=causal_history,
            anomaly_indicators=anomaly_indicators,
            timestamp=self.step_count
        )
        
        return state
    
    def state_to_tensor(self, state: CausalState) -> torch.Tensor:
        """Convert causal state to tensor representation"""
        # Combine all state components
        state_components = [
            state.geological_variables,
            state.temporal_context,
            state.spatial_context,
            state.anomaly_indicators
        ]
        
        # Add causal features
        causal_features = np.zeros(50)  # Fixed size for causal features
        for i, (cause, effect, strength) in enumerate(state.causal_history[:50]):
            if i < len(causal_features):
                causal_features[i] = strength
        
        state_components.append(causal_features)
        
        # Concatenate and normalize
        state_vector = np.concatenate(state_components)
        
        # Ensure fixed size
        target_size = self.config.get('state_dim', 128)
        if len(state_vector) > target_size:
            state_vector = state_vector[:target_size]
        elif len(state_vector) < target_size:
            padding = np.zeros(target_size - len(state_vector))
            state_vector = np.concatenate([state_vector, padding])
        
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state: CausalState, training: bool = True) -> CausalAction:
        """Select action using epsilon-greedy policy with causal reasoning"""
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            # Random action
            action_type = np.random.choice(['investigate', 'monitor', 'alert', 'ignore'])
            target_variables = np.random.choice(
                list(range(len(state.geological_variables))), 
                size=min(3, len(state.geological_variables)), 
                replace=False
            ).tolist()
            investigation_depth = np.random.uniform(0.1, 1.0)
            confidence_threshold = np.random.uniform(0.5, 0.9)
            time_horizon = np.random.uniform(1.0, 24.0)
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
                # Select action type with highest Q-value
                action_scores = {
                    'investigate': torch.mean(q_values['q_investigate']).item(),
                    'monitor': torch.mean(q_values['q_monitor']).item(),
                    'alert': torch.mean(q_values['q_alert']).item(),
                    'ignore': q_values['q_ignore'].item()
                }
                
                action_type = max(action_scores.items(), key=lambda x: x[1])[0]
                
                # Select target variables based on causal importance
                if action_type != 'ignore':
                    causal_importance = self._compute_causal_importance(state)
                    top_indices = np.argsort(causal_importance)[-3:]
                    target_variables = top_indices.tolist()
                else:
                    target_variables = []
                
                # Set parameters based on action type
                if action_type == 'investigate':
                    investigation_depth = 0.8
                    confidence_threshold = 0.7
                    time_horizon = 6.0
                elif action_type == 'monitor':
                    investigation_depth = 0.3
                    confidence_threshold = 0.8
                    time_horizon = 12.0
                elif action_type == 'alert':
                    investigation_depth = 0.5
                    confidence_threshold = 0.9
                    time_horizon = 1.0
                else:  # ignore
                    investigation_depth = 0.0
                    confidence_threshold = 1.0
                    time_horizon = 24.0
        
        action = CausalAction(
            action_type=action_type,
            target_variables=target_variables,
            investigation_depth=investigation_depth,
            confidence_threshold=confidence_threshold,
            time_horizon=time_horizon
        )
        
        return action
    
    def _compute_causal_importance(self, state: CausalState) -> np.ndarray:
        """Compute causal importance of each variable"""
        n_vars = len(state.geological_variables)
        importance = np.ones(n_vars) * 0.1  # Base importance
        
        # Add importance based on causal relationships
        for cause, effect, strength in state.causal_history:
            # This is simplified - in practice would map variable names to indices
            cause_idx = hash(cause) % n_vars
            effect_idx = hash(effect) % n_vars
            
            importance[cause_idx] += strength * 0.5
            importance[effect_idx] += strength * 0.3
        
        # Add importance based on anomaly indicators
        anomaly_contribution = state.anomaly_indicators * 0.4
        if len(anomaly_contribution) == len(importance):
            importance += anomaly_contribution
        
        return importance
    
    def compute_reward(self, state: CausalState, action: CausalAction, 
                      next_state: CausalState, ground_truth: Optional[Dict] = None) -> CausalReward:
        """Compute reward for causal reinforcement learning"""
        # Detection accuracy reward
        detection_accuracy = 0.0
        if ground_truth and 'true_anomalies' in ground_truth:
            predicted_anomalies = np.mean(state.anomaly_indicators) > 0.5
            true_anomalies = ground_truth['true_anomalies']
            detection_accuracy = 1.0 if predicted_anomalies == true_anomalies else 0.0
        
        # False positive penalty
        false_positive_penalty = 0.0
        if action.action_type == 'alert' and not np.any(state.anomaly_indicators > 0.7):
            false_positive_penalty = -0.5
        
        # Discovery bonus for finding new causal relationships
        discovery_bonus = 0.0
        current_edges = len(self.causal_graph.graph.edges())
        if hasattr(self, '_previous_edge_count'):
            if current_edges > self._previous_edge_count:
                discovery_bonus = 0.3 * (current_edges - self._previous_edge_count)
        self._previous_edge_count = current_edges
        
        # Temporal efficiency reward
        temporal_efficiency = 0.0
        if action.action_type in ['investigate', 'alert']:
            # Reward faster response to anomalies
            response_time = action.time_horizon
            temporal_efficiency = max(0.0, 1.0 - response_time / 24.0)
        
        # Causal consistency reward
        causal_consistency = 0.0
        if len(state.causal_history) > 0:
            # Reward actions consistent with causal relationships
            for cause, effect, strength in state.causal_history:
                if strength > 0.5 and cause in [str(i) for i in action.target_variables]:
                    causal_consistency += 0.1 * strength
        
        # Total reward
        total_reward = (
            detection_accuracy + 
            false_positive_penalty + 
            discovery_bonus + 
            temporal_efficiency * 0.3 + 
            causal_consistency * 0.2
        )
        
        reward = CausalReward(
            detection_accuracy=detection_accuracy,
            false_positive_penalty=false_positive_penalty,
            discovery_bonus=discovery_bonus,
            temporal_efficiency=temporal_efficiency,
            causal_consistency=causal_consistency,
            total_reward=total_reward
        )
        
        return reward
    
    def store_experience(self, state: CausalState, action: CausalAction, 
                        reward: CausalReward, next_state: CausalState, done: bool):
        """Store experience in replay buffer"""
        experience = {
            'state': self.state_to_tensor(state),
            'action': self._encode_action(action),
            'reward': reward.total_reward,
            'next_state': self.state_to_tensor(next_state),
            'done': done
        }
        
        self.replay_buffer.append(experience)
        
        # Store for causal analysis
        for i, var_value in enumerate(state.geological_variables):
            self.observation_history[f'var_{i}'].append(var_value)
        
        self.action_history.append(action)
        self.reward_history.append(reward.total_reward)
    
    def _encode_action(self, action: CausalAction) -> torch.Tensor:
        """Encode action as tensor"""
        # One-hot encode action type
        action_types = ['investigate', 'monitor', 'alert', 'ignore']
        action_type_encoding = [1.0 if action.action_type == at else 0.0 for at in action_types]
        
        # Encode other action parameters
        action_encoding = action_type_encoding + [
            action.investigation_depth,
            action.confidence_threshold,
            action.time_horizon / 24.0,  # Normalize
            len(action.target_variables) / 10.0  # Normalize
        ]
        
        return torch.FloatTensor(action_encoding).to(self.device)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
        
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.stack([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            target_values = rewards + self.gamma * next_q_values['state_value'].squeeze() * ~dones
        
        # Compute loss (simplified - would need more sophisticated action handling)
        current_values = current_q_values['state_value'].squeeze()
        loss = F.mse_loss(current_values, target_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.step_count += 1
        
        # Perform causal discovery periodically
        if self.step_count % self.causal_discovery_interval == 0:
            self.update_causal_graph()
    
    def update_causal_graph(self):
        """Update causal graph from accumulated observations"""
        if len(self.observation_history) < 2:
            return
        
        logger.info("Updating causal graph from observations")
        
        # Convert observation history to format suitable for causal discovery
        data_dict = {}
        min_length = min(len(values) for values in self.observation_history.values())
        
        for var_name, values in self.observation_history.items():
            if len(values) >= min_length:
                data_dict[var_name] = np.array(values[:min_length])
        
        # Update causal graph
        self.causal_graph.update_from_data(data_dict, method='granger')
        
        # Log discovered relationships
        n_edges = len(self.causal_graph.graph.edges())
        logger.info(f"Causal graph updated: {n_edges} causal relationships discovered")
    
    def get_causal_explanations(self, state: CausalState, action: CausalAction) -> Dict[str, Any]:
        """Generate causal explanations for actions"""
        explanations = {
            'action_rationale': f"Selected {action.action_type} based on causal analysis",
            'causal_factors': [],
            'predicted_outcomes': [],
            'confidence': 0.0
        }
        
        # Analyze causal factors
        for cause, effect, strength in state.causal_history:
            if strength > self.causal_confidence_threshold:
                explanations['causal_factors'].append({
                    'cause': cause,
                    'effect': effect,
                    'strength': strength,
                    'description': f"{cause} causally influences {effect} (strength: {strength:.2f})"
                })
        
        # Predict outcomes based on causal graph
        if action.target_variables:
            for var_idx in action.target_variables:
                var_name = f"var_{var_idx}"
                children = self.causal_graph.get_causal_children(var_name)
                
                for child, strength, lag in children:
                    explanations['predicted_outcomes'].append({
                        'variable': child,
                        'expected_change': strength * action.investigation_depth,
                        'time_lag': lag,
                        'description': f"Action on {var_name} may affect {child} after {lag:.1f} time units"
                    })
        
        # Compute overall confidence
        if explanations['causal_factors']:
            avg_strength = np.mean([cf['strength'] for cf in explanations['causal_factors']])
            explanations['confidence'] = avg_strength
        
        return explanations
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'causal_graph': self.causal_graph,
            'config': self.config,
            'step_count': self.step_count
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.causal_graph = checkpoint['causal_graph']
        self.step_count = checkpoint['step_count']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_learned_policies(self) -> Dict[str, Any]:
        """Get learned policies and causal insights"""
        policies = {
            'causal_relationships': [],
            'action_preferences': defaultdict(float),
            'temporal_patterns': {},
            'effectiveness_metrics': {}
        }
        
        # Extract causal relationships
        for edge in self.causal_graph.graph.edges():
            cause, effect = edge
            strength = self.causal_graph.edge_strengths.get(edge, 0.0)
            lag = self.causal_graph.temporal_lags.get(edge, 0.0)
            confidence = self.causal_graph.confidence_scores.get(edge, 0.0)
            
            policies['causal_relationships'].append({
                'cause': cause,
                'effect': effect,
                'strength': strength,
                'lag': lag,
                'confidence': confidence
            })
        
        # Analyze action preferences
        for action in self.action_history[-1000:]:  # Last 1000 actions
            policies['action_preferences'][action.action_type] += 1
        
        # Normalize action preferences
        total_actions = sum(policies['action_preferences'].values())
        if total_actions > 0:
            for action_type in policies['action_preferences']:
                policies['action_preferences'][action_type] /= total_actions
        
        # Compute effectiveness metrics
        if len(self.reward_history) > 0:
            policies['effectiveness_metrics'] = {
                'average_reward': np.mean(self.reward_history[-1000:]),
                'reward_trend': np.corrcoef(
                    range(len(self.reward_history[-100:])), 
                    self.reward_history[-100:]
                )[0, 1] if len(self.reward_history) >= 100 else 0.0,
                'exploration_rate': self.epsilon
            }
        
        return policies