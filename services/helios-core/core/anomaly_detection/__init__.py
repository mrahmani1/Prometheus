"""
Anomaly Detection Module for Geological Phenomena Discovery

This module implements comprehensive anomaly detection capabilities for discovering
new geological phenomena through multi-variate, contextual, and collective analysis.
"""

from .anomaly_manager import AnomalyDetectionManager
from .causal_reinforcement import CausalReinforcementLearner
from .collective_detector import CollectiveAnomalyDetector
from .contextual_detector import ContextualAnomalyDetector
from .continuous_learner import ContinuousAnomalyLearner
from .explanation_engine import AnomalyExplanationEngine
from .multivariate_detector import MultivariateAnomalyDetector
from .significance_assessor import SignificanceAssessor

__all__ = [
    "MultivariateAnomalyDetector",
    "ContextualAnomalyDetector",
    "CollectiveAnomalyDetector",
    "CausalReinforcementLearner",
    "ContinuousAnomalyLearner",
    "AnomalyExplanationEngine",
    "SignificanceAssessor",
    "AnomalyDetectionManager",
]
