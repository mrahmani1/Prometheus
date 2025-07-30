"""
Anomaly Detection Module for Geological Phenomena Discovery

This module implements comprehensive anomaly detection capabilities for discovering
new geological phenomena through multi-variate, contextual, and collective analysis.
"""

from .multivariate_detector import MultivariateAnomalyDetector
from .contextual_detector import ContextualAnomalyDetector
from .collective_detector import CollectiveAnomalyDetector
from .causal_reinforcement import CausalReinforcementLearner
from .continuous_learner import ContinuousAnomalyLearner
from .explanation_engine import AnomalyExplanationEngine
from .significance_assessor import SignificanceAssessor
from .anomaly_manager import AnomalyDetectionManager

__all__ = [
    'MultivariateAnomalyDetector',
    'ContextualAnomalyDetector', 
    'CollectiveAnomalyDetector',
    'CausalReinforcementLearner',
    'ContinuousAnomalyLearner',
    'AnomalyExplanationEngine',
    'SignificanceAssessor',
    'AnomalyDetectionManager'
]