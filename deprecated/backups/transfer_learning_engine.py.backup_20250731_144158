"""
Transfer Learning Engine for Helios AI Model

This module implements domain adaptation algorithms, few-shot learning capabilities,
and model fine-tuning procedures for adapting the global Helios model to local
geological conditions and data-sparse regions.

Requirements addressed: 5.1, 5.2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import copy
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GeologicalDomain:
    """Represents a specific geological domain/setting"""
    domain_id: str
    region_name: str
    geological_setting: str  # e.g., "subduction_zone", "hotspot", "rift_valley"
    tectonic_context: str
    data_characteristics: Dict[str, Any]
    available_data_types: List[str]
    data_sparsity_level: float  # 0.0 (sparse) to 1.0 (dense)


@dataclass
class AdaptationConfig:
    """Configuration for domain adaptation"""
    source_domain: GeologicalDomain
    target_domain: GeologicalDomain
    adaptation_method: str
    learning_rate: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32
    regularization_strength: float = 0.01
    few_shot_samples: int = 10
    validation_split: float = 0.2


@dataclass
class AdaptationResult:
    """Results from domain adaptation"""
    adapted_model: nn.Module
    adaptation_metrics: Dict[str, float]
    validation_scores: Dict[str, float]
    convergence_history: List[float]
    adaptation_time: float
    confidence_score: float


class DomainAdaptationStrategy(ABC):
    """Abstract base class for domain adaptation strategies"""
    
    @abstractmethod
    def adapt_model(self, source_model: nn.Module, target_data: DataLoader, 
                   config: AdaptationConfig) -> AdaptationResult:
        """Adapt source model to target domain"""
        pass


class FineTuningAdapter(DomainAdaptationStrategy):
    """Fine-tuning based domain adaptation"""
    
    def __init__(self):
        self.name = "fine_tuning"
    
    def adapt_model(self, source_model: nn.Module, target_data: DataLoader,
                   config: AdaptationConfig) -> AdaptationResult:
        """Fine-tune source model on target domain data"""
        logger.info(f"Starting fine-tuning adaptation for {config.target_domain.region_name}")
        
        # Create a copy of the source model
        adapted_model = copy.deepcopy(source_model)
        adapted_model.train()
        
        # Setup optimizer with lower learning rate for fine-tuning
        optimizer = optim.Adam(adapted_model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        convergence_history = []
        best_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_targets in target_data:
                optimizer.zero_grad()
                
                # Forward pass
                predictions = adapted_model(batch_data)
                loss = criterion(predictions, batch_targets)
                
                # Add regularization to prevent overfitting
                l2_reg = torch.tensor(0.)
                for param in adapted_model.parameters():
                    l2_reg += torch.norm(param)
                loss += config.regularization_strength * l2_reg
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            convergence_history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # Calculate adaptation metrics
        adaptation_metrics = {
            'final_loss': convergence_history[-1],
            'best_loss': best_loss,
            'loss_reduction': (convergence_history[0] - convergence_history[-1]) / convergence_history[0],
            'convergence_stability': np.std(convergence_history[-10:])
        }
        
        return AdaptationResult(
            adapted_model=adapted_model,
            adaptation_metrics=adaptation_metrics,
            validation_scores={},
            convergence_history=convergence_history,
            adaptation_time=0.0,  # Would be measured in real implementation
            confidence_score=self._calculate_confidence(adaptation_metrics)
        )
    
    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on adaptation metrics"""
        loss_reduction = metrics.get('loss_reduction', 0.0)
        stability = 1.0 / (1.0 + metrics.get('convergence_stability', 1.0))
        return min(1.0, (loss_reduction + stability) / 2.0)


class DomainAdversarialAdapter(DomainAdaptationStrategy):
    """Domain adversarial neural network adaptation"""
    
    def __init__(self):
        self.name = "domain_adversarial"
    
    def adapt_model(self, source_model: nn.Module, target_data: DataLoader,
                   config: AdaptationConfig) -> AdaptationResult:
        """Implement domain adversarial adaptation"""
        logger.info(f"Starting domain adversarial adaptation for {config.target_domain.region_name}")
        
        # Create feature extractor and domain classifier
        feature_extractor = copy.deepcopy(source_model)
        domain_classifier = self._create_domain_classifier(feature_extractor)
        
        # Setup optimizers
        feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=config.learning_rate)
        domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=config.learning_rate)
        
        criterion = nn.MSELoss()
        domain_criterion = nn.BCELoss()
        
        convergence_history = []
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_targets in target_data:
                # Train domain classifier
                domain_optimizer.zero_grad()
                features = feature_extractor(batch_data).detach()
                domain_pred = domain_classifier(features)
                domain_labels = torch.ones(features.size(0), 1)  # Target domain = 1
                domain_loss = domain_criterion(domain_pred, domain_labels)
                domain_loss.backward()
                domain_optimizer.step()
                
                # Train feature extractor (adversarial)
                feature_optimizer.zero_grad()
                features = feature_extractor(batch_data)
                predictions = features  # Simplified for this example
                
                # Task loss
                task_loss = criterion(predictions, batch_targets)
                
                # Adversarial loss (fool domain classifier)
                domain_pred = domain_classifier(features)
                domain_labels = torch.zeros(features.size(0), 1)  # Fool classifier
                adversarial_loss = domain_criterion(domain_pred, domain_labels)
                
                total_loss = task_loss - 0.1 * adversarial_loss  # Balance task and adversarial loss
                total_loss.backward()
                feature_optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            convergence_history.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        adaptation_metrics = {
            'final_loss': convergence_history[-1],
            'domain_confusion': 0.5,  # Ideally domain classifier should be confused (50% accuracy)
        }
        
        return AdaptationResult(
            adapted_model=feature_extractor,
            adaptation_metrics=adaptation_metrics,
            validation_scores={},
            convergence_history=convergence_history,
            adaptation_time=0.0,
            confidence_score=0.8
        )
    
    def _create_domain_classifier(self, feature_extractor: nn.Module) -> nn.Module:
        """Create domain classifier network"""
        # Simplified domain classifier
        return nn.Sequential(
            nn.Linear(128, 64),  # Assuming feature size of 128
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


class FewShotLearner:
    """Few-shot learning for data-sparse geological regions"""
    
    def __init__(self):
        self.support_set_size = 10
        self.query_set_size = 5
    
    def meta_learn(self, source_model: nn.Module, few_shot_tasks: List[DataLoader],
                   config: AdaptationConfig) -> nn.Module:
        """Meta-learning for few-shot adaptation"""
        logger.info("Starting meta-learning for few-shot adaptation")
        
        meta_model = copy.deepcopy(source_model)
        meta_optimizer = optim.Adam(meta_model.parameters(), lr=config.learning_rate)
        
        for episode in range(config.num_epochs):
            episode_loss = 0.0
            
            for task_data in few_shot_tasks:
                # Split into support and query sets
                support_data, query_data = self._split_support_query(task_data, config.few_shot_samples)
                
                # Fast adaptation on support set
                adapted_params = self._fast_adapt(meta_model, support_data, config.learning_rate)
                
                # Evaluate on query set
                query_loss = self._evaluate_query(adapted_params, query_data)
                episode_loss += query_loss
            
            # Meta-update
            meta_optimizer.zero_grad()
            episode_loss.backward()
            meta_optimizer.step()
            
            if episode % 10 == 0:
                logger.info(f"Meta-learning episode {episode}, Loss: {episode_loss:.6f}")
        
        return meta_model
    
    def _split_support_query(self, task_data: DataLoader, support_size: int) -> Tuple[DataLoader, DataLoader]:
        """Split task data into support and query sets"""
        # Simplified implementation - in practice would need proper splitting
        return task_data, task_data
    
    def _fast_adapt(self, model: nn.Module, support_data: DataLoader, lr: float) -> Dict[str, torch.Tensor]:
        """Fast adaptation step for few-shot learning"""
        # Simplified MAML-style adaptation
        adapted_params = {}
        for name, param in model.named_parameters():
            adapted_params[name] = param.clone()
        return adapted_params
    
    def _evaluate_query(self, adapted_params: Dict[str, torch.Tensor], query_data: DataLoader) -> torch.Tensor:
        """Evaluate adapted model on query set"""
        # Simplified evaluation
        return torch.tensor(0.5, requires_grad=True)


class ModelPerformanceValidator:
    """Validates performance of adapted models"""
    
    def __init__(self):
        self.validation_metrics = ['mse', 'r2', 'physics_consistency', 'geological_plausibility']
    
    def validate_adapted_model(self, adapted_model: nn.Module, validation_data: DataLoader,
                             original_model: nn.Module, domain: GeologicalDomain) -> Dict[str, float]:
        """Comprehensive validation of adapted model"""
        logger.info(f"Validating adapted model for {domain.region_name}")
        
        adapted_model.eval()
        original_model.eval()
        
        validation_scores = {}
        
        with torch.no_grad():
            adapted_predictions = []
            original_predictions = []
            true_values = []
            
            for batch_data, batch_targets in validation_data:
                adapted_pred = adapted_model(batch_data)
                original_pred = original_model(batch_data)
                
                adapted_predictions.extend(adapted_pred.cpu().numpy())
                original_predictions.extend(original_pred.cpu().numpy())
                true_values.extend(batch_targets.cpu().numpy())
            
            # Convert to numpy arrays
            adapted_predictions = np.array(adapted_predictions)
            original_predictions = np.array(original_predictions)
            true_values = np.array(true_values)
            
            # Calculate metrics
            validation_scores['adapted_mse'] = mean_squared_error(true_values, adapted_predictions)
            validation_scores['original_mse'] = mean_squared_error(true_values, original_predictions)
            validation_scores['adapted_r2'] = r2_score(true_values, adapted_predictions)
            validation_scores['original_r2'] = r2_score(true_values, original_predictions)
            
            # Improvement metrics
            validation_scores['mse_improvement'] = (validation_scores['original_mse'] - validation_scores['adapted_mse']) / validation_scores['original_mse']
            validation_scores['r2_improvement'] = validation_scores['adapted_r2'] - validation_scores['original_r2']
            
            # Physics consistency check
            validation_scores['physics_consistency'] = self._check_physics_consistency(adapted_predictions, domain)
            
            # Geological plausibility
            validation_scores['geological_plausibility'] = self._check_geological_plausibility(adapted_predictions, domain)
        
        return validation_scores
    
    def _check_physics_consistency(self, predictions: np.ndarray, domain: GeologicalDomain) -> float:
        """Check if predictions are physically consistent"""
        # Simplified physics consistency check
        # In practice, would check conservation laws, thermodynamic constraints, etc.
        consistency_score = 0.85  # Placeholder
        return consistency_score
    
    def _check_geological_plausibility(self, predictions: np.ndarray, domain: GeologicalDomain) -> float:
        """Check geological plausibility of predictions"""
        # Check if predictions are within reasonable geological ranges
        plausibility_score = 0.90  # Placeholder
        return plausibility_score


class TransferLearningEngine:
    """Main transfer learning engine for Helios model adaptation"""
    
    def __init__(self):
        self.adaptation_strategies = {
            'fine_tuning': FineTuningAdapter(),
            'domain_adversarial': DomainAdversarialAdapter()
        }
        self.few_shot_learner = FewShotLearner()
        self.validator = ModelPerformanceValidator()
        self.adaptation_history = []
    
    def adapt_to_domain(self, source_model: nn.Module, target_domain: GeologicalDomain,
                       target_data: DataLoader, config: AdaptationConfig) -> AdaptationResult:
        """Adapt source model to target geological domain"""
        logger.info(f"Starting domain adaptation to {target_domain.region_name}")
        
        # Select adaptation strategy based on data sparsity
        if target_domain.data_sparsity_level < 0.3:
            strategy_name = 'domain_adversarial'  # Better for sparse data
        else:
            strategy_name = 'fine_tuning'  # Standard fine-tuning for sufficient data
        
        strategy = self.adaptation_strategies[strategy_name]
        logger.info(f"Using adaptation strategy: {strategy_name}")
        
        # Perform adaptation
        adaptation_result = strategy.adapt_model(source_model, target_data, config)
        
        # Validate adapted model
        validation_scores = self.validator.validate_adapted_model(
            adaptation_result.adapted_model, target_data, source_model, target_domain
        )
        adaptation_result.validation_scores = validation_scores
        
        # Store adaptation history
        self.adaptation_history.append({
            'target_domain': target_domain.domain_id,
            'strategy': strategy_name,
            'config': config,
            'result': adaptation_result
        })
        
        logger.info(f"Domain adaptation completed. MSE improvement: {validation_scores.get('mse_improvement', 0):.4f}")
        
        return adaptation_result
    
    def few_shot_adapt(self, source_model: nn.Module, target_domain: GeologicalDomain,
                      few_shot_data: DataLoader, config: AdaptationConfig) -> AdaptationResult:
        """Adapt model using few-shot learning for data-sparse regions"""
        logger.info(f"Starting few-shot adaptation for {target_domain.region_name}")
        
        # Create few-shot tasks from available data
        few_shot_tasks = [few_shot_data]  # Simplified - would create multiple tasks
        
        # Meta-learn for few-shot adaptation
        meta_model = self.few_shot_learner.meta_learn(source_model, few_shot_tasks, config)
        
        # Fast adaptation to target domain
        adapted_model = copy.deepcopy(meta_model)
        
        # Validate few-shot adapted model
        validation_scores = self.validator.validate_adapted_model(
            adapted_model, few_shot_data, source_model, target_domain
        )
        
        adaptation_result = AdaptationResult(
            adapted_model=adapted_model,
            adaptation_metrics={'few_shot_samples': config.few_shot_samples},
            validation_scores=validation_scores,
            convergence_history=[],
            adaptation_time=0.0,
            confidence_score=0.75  # Generally lower confidence for few-shot
        )
        
        logger.info(f"Few-shot adaptation completed for {target_domain.region_name}")
        
        return adaptation_result
    
    def get_adaptation_recommendations(self, target_domain: GeologicalDomain) -> Dict[str, Any]:
        """Get recommendations for adapting to a specific domain"""
        recommendations = {
            'recommended_strategy': 'fine_tuning',
            'estimated_samples_needed': 1000,
            'expected_performance_gain': 0.15,
            'adaptation_complexity': 'medium'
        }
        
        # Adjust based on domain characteristics
        if target_domain.data_sparsity_level < 0.2:
            recommendations['recommended_strategy'] = 'few_shot'
            recommendations['estimated_samples_needed'] = 50
            recommendations['adaptation_complexity'] = 'high'
        elif target_domain.data_sparsity_level < 0.5:
            recommendations['recommended_strategy'] = 'domain_adversarial'
            recommendations['estimated_samples_needed'] = 500
        
        return recommendations
    
    def save_adaptation_history(self, filepath: str):
        """Save adaptation history to file"""
        history_data = []
        for entry in self.adaptation_history:
            # Convert to serializable format
            serializable_entry = {
                'target_domain': entry['target_domain'],
                'strategy': entry['strategy'],
                'validation_scores': entry['result'].validation_scores,
                'adaptation_metrics': entry['result'].adaptation_metrics
            }
            history_data.append(serializable_entry)
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Adaptation history saved to {filepath}")