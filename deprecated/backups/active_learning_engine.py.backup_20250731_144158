"""
Active Learning Engine for Optimal Data Selection

This module implements various active learning strategies for geological model improvement,
including uncertainty-based sampling, query-by-committee methods, and diversity-based sampling.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import logging
from scipy.stats import entropy
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning strategies"""
    strategy: str = "uncertainty"  # uncertainty, committee, diversity, hybrid
    batch_size: int = 10
    uncertainty_threshold: float = 0.1
    diversity_weight: float = 0.3
    committee_size: int = 5
    geological_features: List[str] = None
    evaluation_metrics: List[str] = None


@dataclass
class GeologicalSample:
    """Represents a geological data sample"""
    id: str
    features: np.ndarray
    location: Tuple[float, float, float]  # lat, lon, depth
    timestamp: Optional[str] = None
    sample_type: str = "unknown"
    metadata: Dict[str, Any] = None
    label: Optional[Union[float, np.ndarray]] = None
    uncertainty: Optional[float] = None


@dataclass
class ActiveLearningResult:
    """Result of active learning sample selection"""
    selected_samples: List[GeologicalSample]
    selection_scores: np.ndarray
    strategy_used: str
    evaluation_metrics: Dict[str, float]
    geological_coverage: Dict[str, float]


class ActiveLearningStrategy(ABC):
    """Abstract base class for active learning strategies"""
    
    @abstractmethod
    def select_samples(self, 
                      unlabeled_pool: List[GeologicalSample],
                      labeled_data: List[GeologicalSample],
                      model: nn.Module,
                      batch_size: int) -> List[int]:
        """Select samples from unlabeled pool"""
        pass
    
    @abstractmethod
    def compute_scores(self, 
                      samples: List[GeologicalSample],
                      model: nn.Module) -> np.ndarray:
        """Compute selection scores for samples"""
        pass


class UncertaintyBasedSampling(ActiveLearningStrategy):
    """Uncertainty-based sampling for geological model improvement"""
    
    def __init__(self, uncertainty_method: str = "entropy"):
        self.uncertainty_method = uncertainty_method
        
    def select_samples(self, 
                      unlabeled_pool: List[GeologicalSample],
                      labeled_data: List[GeologicalSample],
                      model: nn.Module,
                      batch_size: int) -> List[int]:
        """Select samples with highest uncertainty"""
        scores = self.compute_scores(unlabeled_pool, model)
        selected_indices = np.argsort(scores)[-batch_size:]
        return selected_indices.tolist()
    
    def compute_scores(self, 
                      samples: List[GeologicalSample],
                      model: nn.Module) -> np.ndarray:
        """Compute uncertainty scores for geological samples"""
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for sample in samples:
                features = torch.FloatTensor(sample.features).unsqueeze(0)
                
                if self.uncertainty_method == "entropy":
                    predictions = model(features)
                    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                        probs = torch.softmax(predictions, dim=1)
                        uncertainty = entropy(probs.numpy().flatten())
                    else:
                        uncertainty = torch.var(predictions).item()
                        
                elif self.uncertainty_method == "variance":
                    model.train()
                    predictions = []
                    for _ in range(10):
                        pred = model(features)
                        predictions.append(pred.numpy())
                    predictions = np.array(predictions)
                    uncertainty = np.var(predictions)
                    model.eval()
                    
                elif self.uncertainty_method == "geological_physics":
                    predictions = model(features)
                    physics_violation = self._check_physics_constraints(
                        sample, predictions.numpy()
                    )
                    uncertainty = physics_violation
                    
                uncertainties.append(uncertainty)
        
        return np.array(uncertainties)
    
    def _check_physics_constraints(self, 
                                 sample: GeologicalSample, 
                                 prediction: np.ndarray) -> float:
        """Check geological physics constraints and return violation score"""
        violation_score = 0.0
        
        if sample.sample_type == "seismic":
            if len(prediction) > 0:
                velocity = prediction[0]
                if velocity < 1.5 or velocity > 8.0:
                    violation_score += abs(velocity - np.clip(velocity, 1.5, 8.0))
                    
        elif sample.sample_type == "thermal":
            if len(prediction) > 0:
                temperature = prediction[0]
                depth = sample.location[2] if len(sample.location) > 2 else 0
                expected_temp = 15 + depth * 0.025
                violation_score += abs(temperature - expected_temp) / expected_temp
                
        elif sample.sample_type == "deformation":
            if len(prediction) >= 3:
                deformation_rate = np.linalg.norm(prediction[:3])
                if deformation_rate > 0.1:
                    violation_score += deformation_rate - 0.1
        
        return violation_score


class QueryByCommittee(ActiveLearningStrategy):
    """Query-by-committee method for ensemble model training"""
    
    def __init__(self, committee_models: List[nn.Module], disagreement_measure: str = "vote_entropy"):
        self.committee_models = committee_models
        self.disagreement_measure = disagreement_measure
        
    def select_samples(self, 
                      unlabeled_pool: List[GeologicalSample],
                      labeled_data: List[GeologicalSample],
                      model: nn.Module,
                      batch_size: int) -> List[int]:
        """Select samples with highest committee disagreement"""
        scores = self.compute_scores(unlabeled_pool, model)
        selected_indices = np.argsort(scores)[-batch_size:]
        return selected_indices.tolist()
    
    def compute_scores(self, 
                      samples: List[GeologicalSample],
                      model: nn.Module) -> np.ndarray:
        """Compute committee disagreement scores"""
        disagreements = []
        
        for sample in samples:
            features = torch.FloatTensor(sample.features).unsqueeze(0)
            predictions = []
            
            for committee_model in self.committee_models:
                committee_model.eval()
                with torch.no_grad():
                    pred = committee_model(features)
                    predictions.append(pred.numpy().flatten())
            
            predictions = np.array(predictions)
            
            if self.disagreement_measure == "vote_entropy":
                if predictions.shape[1] > 1:
                    votes = np.argmax(predictions, axis=1)
                    vote_counts = np.bincount(votes, minlength=predictions.shape[1])
                    vote_probs = vote_counts / len(votes)
                    disagreement = entropy(vote_probs)
                else:
                    disagreement = np.var(predictions)
                    
            elif self.disagreement_measure == "kl_divergence":
                mean_pred = np.mean(predictions, axis=0)
                kl_divs = []
                for pred in predictions:
                    pred_norm = pred / (np.sum(pred) + 1e-8)
                    mean_norm = mean_pred / (np.sum(mean_pred) + 1e-8)
                    kl_div = entropy(pred_norm, mean_norm)
                    kl_divs.append(kl_div)
                disagreement = np.mean(kl_divs)
                
            elif self.disagreement_measure == "geological_consensus":
                disagreement = self._geological_disagreement(sample, predictions)
            
            disagreements.append(disagreement)
        
        return np.array(disagreements)
    
    def _geological_disagreement(self, 
                               sample: GeologicalSample, 
                               predictions: np.ndarray) -> float:
        """Calculate geological domain-specific disagreement"""
        if sample.sample_type == "seismic":
            if predictions.shape[1] >= 3:
                location_preds = predictions[:, :3]
                location_std = np.std(location_preds, axis=0)
                return np.mean(location_std)
                
        elif sample.sample_type == "volcanic":
            if predictions.shape[1] >= 1:
                prob_preds = predictions[:, 0]
                return np.std(prob_preds)
                
        return np.var(predictions)


class DiversityBasedSampling(ActiveLearningStrategy):
    """Diversity-based sampling for comprehensive geological coverage"""
    
    def __init__(self, diversity_method: str = "kmeans", geological_features: List[str] = None):
        self.diversity_method = diversity_method
        self.geological_features = geological_features or []
        
    def select_samples(self, 
                      unlabeled_pool: List[GeologicalSample],
                      labeled_data: List[GeologicalSample],
                      model: nn.Module,
                      batch_size: int) -> List[int]:
        """Select diverse samples for comprehensive coverage"""
        if self.diversity_method == "kmeans":
            return self._kmeans_selection(unlabeled_pool, batch_size)
        elif self.diversity_method == "max_distance":
            return self._max_distance_selection(unlabeled_pool, labeled_data, batch_size)
        elif self.diversity_method == "geological_coverage":
            return self._geological_coverage_selection(unlabeled_pool, labeled_data, batch_size)
        else:
            scores = self.compute_scores(unlabeled_pool, model)
            selected_indices = np.argsort(scores)[-batch_size:]
            return selected_indices.tolist()
    
    def compute_scores(self, 
                      samples: List[GeologicalSample],
                      model: nn.Module) -> np.ndarray:
        """Compute diversity scores"""
        features_matrix = np.array([sample.features for sample in samples])
        distances = pairwise_distances(features_matrix)
        diversity_scores = np.mean(distances, axis=1)
        return diversity_scores
    
    def _kmeans_selection(self, 
                         unlabeled_pool: List[GeologicalSample], 
                         batch_size: int) -> List[int]:
        """Select samples using K-means clustering"""
        features_matrix = np.array([sample.features for sample in unlabeled_pool])
        
        kmeans = KMeans(n_clusters=batch_size, random_state=42)
        cluster_labels = kmeans.fit_predict(features_matrix)
        
        selected_indices = []
        for i in range(batch_size):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_features = features_matrix[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return selected_indices
    
    def _max_distance_selection(self, 
                              unlabeled_pool: List[GeologicalSample],
                              labeled_data: List[GeologicalSample],
                              batch_size: int) -> List[int]:
        """Select samples with maximum distance to labeled data"""
        unlabeled_features = np.array([sample.features for sample in unlabeled_pool])
        labeled_features = np.array([sample.features for sample in labeled_data])
        
        selected_indices = []
        
        for _ in range(batch_size):
            if len(labeled_features) == 0:
                remaining_indices = [i for i in range(len(unlabeled_pool)) 
                                   if i not in selected_indices]
                if remaining_indices:
                    selected_indices.append(np.random.choice(remaining_indices))
                continue
            
            distances_to_labeled = pairwise_distances(unlabeled_features, labeled_features)
            min_distances = np.min(distances_to_labeled, axis=1)
            
            for idx in selected_indices:
                min_distances[idx] = -1
            
            max_distance_idx = np.argmax(min_distances)
            selected_indices.append(max_distance_idx)
            
            labeled_features = np.vstack([labeled_features, 
                                        unlabeled_features[max_distance_idx:max_distance_idx+1]])
        
        return selected_indices
    
    def _geological_coverage_selection(self, 
                                     unlabeled_pool: List[GeologicalSample],
                                     labeled_data: List[GeologicalSample],
                                     batch_size: int) -> List[int]:
        """Select samples for optimal geological coverage"""
        geological_coverage = self._analyze_geological_coverage(labeled_data)
        selected_indices = []
        
        for _ in range(batch_size):
            best_idx = -1
            best_coverage_improvement = -1
            
            for i, sample in enumerate(unlabeled_pool):
                if i in selected_indices:
                    continue
                
                coverage_improvement = self._calculate_coverage_improvement(
                    sample, geological_coverage
                )
                
                if coverage_improvement > best_coverage_improvement:
                    best_coverage_improvement = coverage_improvement
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                self._update_coverage(unlabeled_pool[best_idx], geological_coverage)
        
        return selected_indices
    
    def _analyze_geological_coverage(self, 
                                   labeled_data: List[GeologicalSample]) -> Dict[str, Dict]:
        """Analyze current geological feature coverage"""
        coverage = {
            'sample_types': {},
            'depth_ranges': {},
            'geographic_regions': {}
        }
        
        for sample in labeled_data:
            sample_type = sample.sample_type
            coverage['sample_types'][sample_type] = coverage['sample_types'].get(sample_type, 0) + 1
            
            if len(sample.location) > 2:
                depth = sample.location[2]
                depth_bin = f"{int(depth//1000)*1000}-{int(depth//1000)*1000+1000}m"
                coverage['depth_ranges'][depth_bin] = coverage['depth_ranges'].get(depth_bin, 0) + 1
            
            lat, lon = sample.location[:2]
            geo_bin = f"{int(lat//5)*5}-{int(lon//5)*5}"
            coverage['geographic_regions'][geo_bin] = coverage['geographic_regions'].get(geo_bin, 0) + 1
        
        return coverage
    
    def _calculate_coverage_improvement(self, 
                                      sample: GeologicalSample, 
                                      current_coverage: Dict[str, Dict]) -> float:
        """Calculate how much this sample would improve geological coverage"""
        improvement = 0.0
        
        sample_type = sample.sample_type
        current_type_count = current_coverage['sample_types'].get(sample_type, 0)
        if current_type_count == 0:
            improvement += 1.0
        else:
            improvement += 1.0 / (current_type_count + 1)
        
        if len(sample.location) > 2:
            depth = sample.location[2]
            depth_bin = f"{int(depth//1000)*1000}-{int(depth//1000)*1000+1000}m"
            current_depth_count = current_coverage['depth_ranges'].get(depth_bin, 0)
            if current_depth_count == 0:
                improvement += 0.5
            else:
                improvement += 0.5 / (current_depth_count + 1)
        
        lat, lon = sample.location[:2]
        geo_bin = f"{int(lat//5)*5}-{int(lon//5)*5}"
        current_geo_count = current_coverage['geographic_regions'].get(geo_bin, 0)
        if current_geo_count == 0:
            improvement += 0.5
        else:
            improvement += 0.5 / (current_geo_count + 1)
        
        return improvement
    
    def _update_coverage(self, 
                        sample: GeologicalSample, 
                        coverage: Dict[str, Dict]) -> None:
        """Update coverage statistics with new sample"""
        sample_type = sample.sample_type
        coverage['sample_types'][sample_type] = coverage['sample_types'].get(sample_type, 0) + 1
        
        if len(sample.location) > 2:
            depth = sample.location[2]
            depth_bin = f"{int(depth//1000)*1000}-{int(depth//1000)*1000+1000}m"
            coverage['depth_ranges'][depth_bin] = coverage['depth_ranges'].get(depth_bin, 0) + 1
        
        lat, lon = sample.location[:2]
        geo_bin = f"{int(lat//5)*5}-{int(lon//5)*5}"
        coverage['geographic_regions'][geo_bin] = coverage['geographic_regions'].get(geo_bin, 0) + 1


class ActiveLearningEvaluator:
    """Evaluation metrics for geological active learning applications"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_selection(self, 
                          selected_samples: List[GeologicalSample],
                          unlabeled_pool: List[GeologicalSample],
                          labeled_data: List[GeologicalSample],
                          model: nn.Module,
                          strategy_name: str) -> Dict[str, float]:
        """Evaluate active learning sample selection"""
        metrics = {}
        
        metrics.update(self._evaluate_geological_coverage(selected_samples, labeled_data))
        metrics.update(self._evaluate_diversity(selected_samples, unlabeled_pool))
        metrics.update(self._evaluate_uncertainty(selected_samples, model))
        metrics.update(self._evaluate_information_gain(selected_samples, model))
        
        self.metrics_history.append({
            'timestamp': np.datetime64('now'),
            'strategy': strategy_name,
            'metrics': metrics,
            'sample_count': len(selected_samples)
        })
        
        return metrics
    
    def _evaluate_geological_coverage(self, 
                                    selected_samples: List[GeologicalSample],
                                    labeled_data: List[GeologicalSample]) -> Dict[str, float]:
        """Evaluate geological feature coverage"""
        metrics = {}
        
        all_samples = labeled_data + selected_samples
        sample_types = set(sample.sample_type for sample in all_samples)
        selected_types = set(sample.sample_type for sample in selected_samples)
        
        metrics['sample_type_diversity'] = len(selected_types) / max(len(sample_types), 1)
        
        if selected_samples:
            locations = np.array([sample.location[:2] for sample in selected_samples])
            if len(locations) > 1:
                distances = pdist(locations)
                metrics['geographic_spread'] = np.mean(distances)
                metrics['geographic_coverage'] = np.std(distances)
            else:
                metrics['geographic_spread'] = 0.0
                metrics['geographic_coverage'] = 0.0
        
        depths = [sample.location[2] for sample in selected_samples 
                 if len(sample.location) > 2]
        if depths:
            metrics['depth_range_coverage'] = max(depths) - min(depths)
            metrics['depth_diversity'] = np.std(depths)
        else:
            metrics['depth_range_coverage'] = 0.0
            metrics['depth_diversity'] = 0.0
        
        return metrics
    
    def _evaluate_diversity(self, 
                          selected_samples: List[GeologicalSample],
                          unlabeled_pool: List[GeologicalSample]) -> Dict[str, float]:
        """Evaluate sample diversity"""
        metrics = {}
        
        if len(selected_samples) < 2:
            return {'diversity_score': 0.0, 'intra_diversity': 0.0}
        
        selected_features = np.array([sample.features for sample in selected_samples])
        intra_distances = pdist(selected_features)
        metrics['intra_diversity'] = np.mean(intra_distances)
        
        if unlabeled_pool:
            pool_features = np.array([sample.features for sample in unlabeled_pool])
            pool_distances = pdist(pool_features)
            pool_mean_distance = np.mean(pool_distances)
            
            metrics['diversity_score'] = metrics['intra_diversity'] / max(pool_mean_distance, 1e-8)
        else:
            metrics['diversity_score'] = metrics['intra_diversity']
        
        return metrics
    
    def _evaluate_uncertainty(self, 
                            selected_samples: List[GeologicalSample],
                            model: nn.Module) -> Dict[str, float]:
        """Evaluate uncertainty-based selection quality"""
        metrics = {}
        
        if not selected_samples:
            return {'mean_uncertainty': 0.0, 'uncertainty_variance': 0.0}
        
        uncertainties = []
        model.eval()
        
        with torch.no_grad():
            for sample in selected_samples:
                features = torch.FloatTensor(sample.features).unsqueeze(0)
                
                model.train()
                predictions = []
                for _ in range(10):
                    pred = model(features)
                    predictions.append(pred.numpy())
                
                predictions = np.array(predictions)
                uncertainty = np.var(predictions)
                uncertainties.append(uncertainty)
                
                model.eval()
        
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['uncertainty_variance'] = np.var(uncertainties)
        metrics['max_uncertainty'] = np.max(uncertainties)
        
        return metrics
    
    def _evaluate_information_gain(self, 
                                 selected_samples: List[GeologicalSample],
                                 model: nn.Module) -> Dict[str, float]:
        """Evaluate expected information gain"""
        metrics = {}
        
        if selected_samples:
            features_matrix = np.array([sample.features for sample in selected_samples])
            
            feature_variances = np.var(features_matrix, axis=0)
            metrics['feature_information_gain'] = np.mean(feature_variances)
            
            sample_types = [sample.sample_type for sample in selected_samples]
            unique_types = len(set(sample_types))
            metrics['geological_information_gain'] = unique_types / len(selected_samples)
        else:
            metrics['feature_information_gain'] = 0.0
            metrics['geological_information_gain'] = 0.0
        
        return metrics
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        if not self.metrics_history:
            return {}
        
        summary = {
            'total_evaluations': len(self.metrics_history),
            'strategies_used': list(set(eval_data['strategy'] for eval_data in self.metrics_history)),
            'average_metrics': {},
            'best_performing_strategy': None
        }
        
        all_metrics = {}
        for eval_data in self.metrics_history:
            for metric_name, value in eval_data['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            summary['average_metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        strategy_performance = {}
        for eval_data in self.metrics_history:
            strategy = eval_data['strategy']
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            
            diversity_score = eval_data['metrics'].get('diversity_score', 0.0)
            coverage_score = eval_data['metrics'].get('sample_type_diversity', 0.0)
            overall_score = (diversity_score + coverage_score) / 2.0
            strategy_performance[strategy].append(overall_score)
        
        best_strategy = None
        best_score = -1
        for strategy, scores in strategy_performance.items():
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_strategy = strategy
        
        summary['best_performing_strategy'] = {
            'strategy': best_strategy,
            'average_score': best_score
        }
        
        return summary


class ActiveLearningEngine:
    """Main active learning engine for optimal geological data selection"""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.strategies = self._initialize_strategies()
        self.evaluator = ActiveLearningEvaluator()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_strategies(self) -> Dict[str, ActiveLearningStrategy]:
        """Initialize active learning strategies"""
        strategies = {
            'uncertainty': UncertaintyBasedSampling(),
            'diversity': DiversityBasedSampling(
                geological_features=self.config.geological_features
            )
        }
        
        return strategies
    
    def select_optimal_samples(self, 
                             unlabeled_pool: List[GeologicalSample],
                             labeled_data: List[GeologicalSample],
                             model: nn.Module,
                             committee_models: Optional[List[nn.Module]] = None) -> ActiveLearningResult:
        """Select optimal samples for geological model improvement"""
        
        strategy_name = self.config.strategy
        batch_size = self.config.batch_size
        
        if strategy_name == "committee" and committee_models:
            self.strategies['committee'] = QueryByCommittee(
                committee_models, 
                disagreement_measure="geological_consensus"
            )
        
        if strategy_name == "hybrid":
            return self._hybrid_selection(unlabeled_pool, labeled_data, model, committee_models)
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        selected_indices = strategy.select_samples(
            unlabeled_pool, labeled_data, model, batch_size
        )
        
        selected_samples = [unlabeled_pool[i] for i in selected_indices]
        selection_scores = strategy.compute_scores(unlabeled_pool, model)
        
        evaluation_metrics = self.evaluator.evaluate_selection(
            selected_samples, unlabeled_pool, labeled_data, model, strategy_name
        )
        
        geological_coverage = self._calculate_geological_coverage(
            selected_samples, labeled_data
        )
        
        result = ActiveLearningResult(
            selected_samples=selected_samples,
            selection_scores=selection_scores[selected_indices],
            strategy_used=strategy_name,
            evaluation_metrics=evaluation_metrics,
            geological_coverage=geological_coverage
        )
        
        self.logger.info(f"Selected {len(selected_samples)} samples using {strategy_name} strategy")
        
        return result
    
    def _hybrid_selection(self, 
                         unlabeled_pool: List[GeologicalSample],
                         labeled_data: List[GeologicalSample],
                         model: nn.Module,
                         committee_models: Optional[List[nn.Module]] = None) -> ActiveLearningResult:
        """Hybrid selection combining multiple strategies"""
        
        batch_size = self.config.batch_size
        uncertainty_weight = 1.0 - self.config.diversity_weight
        diversity_weight = self.config.diversity_weight
        
        uncertainty_strategy = self.strategies['uncertainty']
        diversity_strategy = self.strategies['diversity']
        
        uncertainty_scores = uncertainty_strategy.compute_scores(unlabeled_pool, model)
        diversity_scores = diversity_strategy.compute_scores(unlabeled_pool, model)
        
        uncertainty_scores = (uncertainty_scores - np.min(uncertainty_scores)) / \
                           (np.max(uncertainty_scores) - np.min(uncertainty_scores) + 1e-8)
        diversity_scores = (diversity_scores - np.min(diversity_scores)) / \
                         (np.max(diversity_scores) - np.min(diversity_scores) + 1e-8)
        
        combined_scores = uncertainty_weight * uncertainty_scores + \
                         diversity_weight * diversity_scores
        
        if committee_models:
            committee_strategy = QueryByCommittee(committee_models)
            committee_scores = committee_strategy.compute_scores(unlabeled_pool, model)
            committee_scores = (committee_scores - np.min(committee_scores)) / \
                             (np.max(committee_scores) - np.min(committee_scores) + 1e-8)
            combined_scores += 0.3 * committee_scores
        
        selected_indices = np.argsort(combined_scores)[-batch_size:]
        selected_samples = [unlabeled_pool[i] for i in selected_indices]
        
        evaluation_metrics = self.evaluator.evaluate_selection(
            selected_samples, unlabeled_pool, labeled_data, model, "hybrid"
        )
        
        geological_coverage = self._calculate_geological_coverage(
            selected_samples, labeled_data
        )
        
        return ActiveLearningResult(
            selected_samples=selected_samples,
            selection_scores=combined_scores[selected_indices],
            strategy_used="hybrid",
            evaluation_metrics=evaluation_metrics,
            geological_coverage=geological_coverage
        )
    
    def _calculate_geological_coverage(self, 
                                     selected_samples: List[GeologicalSample],
                                     labeled_data: List[GeologicalSample]) -> Dict[str, float]:
        """Calculate geological coverage metrics"""
        all_samples = labeled_data + selected_samples
        
        coverage = {
            'sample_type_coverage': 0.0,
            'geographic_coverage': 0.0,
            'depth_coverage': 0.0
        }
        
        if not all_samples:
            return coverage
        
        sample_types = set(sample.sample_type for sample in all_samples)
        expected_types = {'seismic', 'volcanic', 'thermal', 'deformation', 'geochemical'}
        coverage['sample_type_coverage'] = len(sample_types & expected_types) / len(expected_types)
        
        locations = np.array([sample.location[:2] for sample in all_samples])
        if len(locations) > 1:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(locations)
                coverage['geographic_coverage'] = hull.volume
            except:
                coverage['geographic_coverage'] = np.std(locations.flatten())
        
        depths = [sample.location[2] for sample in all_samples if len(sample.location) > 2]
        if depths:
            depth_range = max(depths) - min(depths)
            coverage['depth_coverage'] = min(depth_range / 10000.0, 1.0)
        
        return coverage
    
    def update_strategy_config(self, new_config: ActiveLearningConfig) -> None:
        """Update active learning configuration"""
        self.config = new_config
        self.strategies = self._initialize_strategies()
        self.logger.info(f"Updated active learning configuration")
    
    def get_strategy_recommendations(self, 
                                   unlabeled_pool: List[GeologicalSample],
                                   labeled_data: List[GeologicalSample]) -> Dict[str, str]:
        """Get recommendations for optimal strategy selection"""
        recommendations = {}
        
        pool_size = len(unlabeled_pool)
        labeled_size = len(labeled_data)
        
        if labeled_size < 100:
            recommendations['primary'] = "diversity"
            recommendations['reason'] = "Limited labeled data - focus on diverse coverage"
        elif pool_size > 10000:
            recommendations['primary'] = "uncertainty"
            recommendations['reason'] = "Large unlabeled pool - focus on uncertain samples"
        else:
            recommendations['primary'] = "hybrid"
            recommendations['reason'] = "Balanced dataset - use hybrid approach"
        
        if labeled_data:
            sample_types = set(sample.sample_type for sample in labeled_data)
            if len(sample_types) < 3:
                recommendations['secondary'] = "diversity"
                recommendations['coverage_note'] = "Limited geological diversity in labeled data"
        
        return recommendations
    
    def export_evaluation_report(self, filepath: str) -> None:
        """Export detailed evaluation report"""
        summary = self.evaluator.get_evaluation_summary()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Exported evaluation report to {filepath}")


def create_sample_geological_data(n_samples: int = 1000) -> List[GeologicalSample]:
    """Create sample geological data for testing"""
    samples = []
    
    sample_types = ['seismic', 'volcanic', 'thermal', 'deformation', 'geochemical']
    
    for i in range(n_samples):
        sample_type = np.random.choice(sample_types)
        
        if sample_type == 'seismic':
            features = np.random.normal(0, 1, 10)
        elif sample_type == 'volcanic':
            features = np.random.exponential(1, 8)
        elif sample_type == 'thermal':
            features = np.random.gamma(2, 2, 6)
        elif sample_type == 'deformation':
            features = np.random.uniform(-0.1, 0.1, 12)
        else:
            features = np.random.lognormal(0, 1, 15)
        
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        depth = np.random.exponential(1000)
        
        sample = GeologicalSample(
            id=f"sample_{i}",
            features=features,
            location=(lat, lon, depth),
            sample_type=sample_type,
            metadata={'synthetic': True}
        )
        
        samples.append(sample)
    
    return samples


def test_active_learning_engine():
    """Test the active learning engine"""
    # Create samples with consistent feature size
    all_samples = create_sample_geological_data_consistent(1000)
    labeled_data = all_samples[:100]
    unlabeled_pool = all_samples[100:]
    
    class SimpleGeologicalModel(nn.Module):
        def __init__(self, input_size=10, output_size=1):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_size)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleGeologicalModel(input_size=10)
    
    strategies = ['uncertainty', 'diversity', 'hybrid']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        
        config = ActiveLearningConfig(
            strategy=strategy,
            batch_size=20,
            diversity_weight=0.3
        )
        
        engine = ActiveLearningEngine(config)
        
        result = engine.select_optimal_samples(
            unlabeled_pool, labeled_data, model
        )
        
        print(f"Selected {len(result.selected_samples)} samples")
        print(f"Evaluation metrics: {result.evaluation_metrics}")
        print(f"Geological coverage: {result.geological_coverage}")


def create_sample_geological_data_consistent(n_samples: int = 1000) -> List[GeologicalSample]:
    """Create sample geological data with consistent feature size for testing"""
    samples = []
    
    sample_types = ['seismic', 'volcanic', 'thermal', 'deformation', 'geochemical']
    
    for i in range(n_samples):
        sample_type = np.random.choice(sample_types)
        
        # All samples have 10 features for consistency
        features = np.random.normal(0, 1, 10)
        
        # Add some sample-type specific variations
        if sample_type == 'seismic':
            features[:3] *= 2  # Amplify first 3 features for seismic
        elif sample_type == 'volcanic':
            features[3:6] = np.abs(features[3:6])  # Make some features positive
        elif sample_type == 'thermal':
            features[6:8] = features[6:8] + 5  # Shift thermal features
        elif sample_type == 'deformation':
            features[8:] *= 0.1  # Scale down deformation features
        
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        depth = np.random.exponential(1000)
        
        sample = GeologicalSample(
            id=f"sample_{i}",
            features=features,
            location=(lat, lon, depth),
            sample_type=sample_type,
            metadata={'synthetic': True}
        )
        
        samples.append(sample)
    
    return samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_active_learning_engine()