"""
Intent Classification for Geological Queries
Classifies user intents for geological and geophysical queries
"""

import re
import time
from typing import Dict, List, Tuple

import numpy as np
import structlog

# Optional imports for ML functionality
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import signal
from typing import Tuple

import numpy as np
import torch.nn.functional as F
from models.nlp_models import AlternativeIntent, IntentClassificationResult, QueryIntent

logger = structlog.get_logger()

class IntentClassifier:
    """
    Classifies geological query intents using hybrid approach:
    - Rule-based patterns for high-confidence geological terms
    - ML model for complex intent classification
    """

    def __init__(self):
        self.logger = logger.bind(component="intent_classifier")

        # Initialize geological intent patterns
        self.intent_patterns = self._initialize_intent_patterns()

        # Initialize ML components if available
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            self.ml_classifier = MultinomialNB()

            # Training data for geological intents
            self.training_data = self._get_training_data()
            self._train_classifier()
        else:
            self.logger.warning(
                "scikit-learn not available, using pattern-based classification only"
            )
            self.vectorizer = None
            self.ml_classifier = None

        self.logger.info("Intent classifier initialized successfully")

    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize rule-based patterns for geological intents"""
        return {
            QueryIntent.SIMULATION_REQUEST: [
                r"\b(simulate|model|run simulation|predict)\b.*\b(volcano|earthquake|tectonic|magma|eruption)\b",
                r"\b(what if|scenario|forecast)\b.*\b(geological|volcanic|seismic)\b",
                r"\b(long.?term|evolution|development)\b.*\b(volcanic system|magma chamber)\b",
                r"\b(physics.?informed|PINN|neural network)\b.*\b(geological|geodynamic)\b",
            ],
            QueryIntent.DATA_ANALYSIS: [
                r"\b(analyze|examine|study|investigate)\b.*\b(seismic data|GPS|InSAR|geochemical)\b",
                r"\b(correlation|pattern|trend)\b.*\b(geological|volcanic|seismic)\b",
                r"\b(time series|temporal|historical data)\b.*\b(analysis|processing)\b",
                r"\b(statistical|machine learning)\b.*\b(geological data|sensor data)\b",
            ],
            QueryIntent.MONITORING_QUERY: [
                r"\b(current|real.?time|live|now)\b.*\b(volcanic activity|seismic|monitoring)\b",
                r"\b(status|condition|state)\b.*\b(volcano|earthquake|deformation)\b",
                r"\b(alert level|warning|risk level)\b",
                r"\b(sensor|instrument|measurement)\b.*\b(reading|data|signal)\b",
            ],
            QueryIntent.HAZARD_ASSESSMENT: [
                r"\b(risk|hazard|danger|threat)\b.*\b(volcanic|seismic|geological)\b",
                r"\b(probability|likelihood|chance)\b.*\b(eruption|earthquake|disaster)\b",
                r"\b(evacuation|emergency|preparedness)\b",
                r"\b(impact|damage|consequence)\b.*\b(volcanic|seismic)\b",
            ],
            QueryIntent.RESOURCE_PREDICTION: [
                r"\b(mineral|ore|geothermal|resource)\b.*\b(deposit|exploration|potential)\b",
                r"\b(hydrothermal|magmatic)\b.*\b(ore formation|mineralization)\b",
                r"\b(prospectivity|exploration target)\b",
                r"\b(economic geology|mining|extraction)\b",
            ],
            QueryIntent.EXPLANATION_REQUEST: [
                r"\b(what is|explain|describe|tell me about)\b.*\b(geological|volcanic|seismic)\b",
                r"\b(how does|why does|what causes)\b.*\b(volcano|earthquake|plate tectonics)\b",
                r"\b(definition|meaning|concept)\b.*\b(geological|geophysical)\b",
                r"\b(mechanism|process|formation)\b.*\b(volcanic|tectonic|magmatic)\b",
            ],
            QueryIntent.COMPARISON_QUERY: [
                r"\b(compare|difference|similar|contrast)\b.*\b(volcano|earthquake|geological)\b",
                r"\b(versus|vs|compared to)\b.*\b(volcanic system|seismic event)\b",
                r"\b(which is|what are the differences)\b.*\b(geological|volcanic)\b",
            ],
            QueryIntent.HISTORICAL_QUERY: [
                r"\b(history|historical|past|previous)\b.*\b(eruption|earthquake|geological event)\b",
                r"\b(when did|what happened|timeline)\b.*\b(volcanic|seismic|geological)\b",
                r"\b(geological time|deep time|evolution)\b",
                r"\b(paleontology|stratigraphy|geochronology)\b",
            ],
        }

    def _get_training_data(self) -> Tuple[List[str], List[str]]:
        """Get training data for ML classifier"""
        texts = []
        labels = []

        # Simulation requests
        simulation_texts = [
            "Can you simulate the evolution of Yellowstone volcanic system?",
            "Run a PINN model for magma chamber dynamics",
            "What would happen if Mount Vesuvius erupted again?",
            "Model the long-term behavior of the San Andreas fault",
            "Predict volcanic hazards for the next 100 years",
            "Simulate crustal deformation in the Cascadia subduction zone",
        ]
        texts.extend(simulation_texts)
        labels.extend([QueryIntent.SIMULATION_REQUEST.value] * len(simulation_texts))

        # Data analysis
        analysis_texts = [
            "Analyze the seismic data from the recent earthquake swarm",
            "Examine GPS deformation patterns around Mount St. Helens",
            "Study the correlation between gas emissions and eruptions",
            "Investigate temporal trends in volcanic activity",
            "Process InSAR data for ground deformation analysis",
            "Analyze geochemical composition of volcanic gases",
        ]
        texts.extend(analysis_texts)
        labels.extend([QueryIntent.DATA_ANALYSIS.value] * len(analysis_texts))

        # Monitoring queries
        monitoring_texts = [
            "What is the current status of Kilauea volcano?",
            "Show me real-time seismic activity",
            "What are the latest sensor readings?",
            "Is there any volcanic unrest detected?",
            "Current alert level for Mount Rainier",
            "Live monitoring data from Yellowstone",
        ]
        texts.extend(monitoring_texts)
        labels.extend([QueryIntent.MONITORING_QUERY.value] * len(monitoring_texts))

        # Hazard assessment
        hazard_texts = [
            "What is the eruption probability for Mount Fuji?",
            "Assess volcanic hazards for nearby communities",
            "Earthquake risk assessment for San Francisco",
            "Potential impact of a major eruption",
            "Emergency preparedness for volcanic hazards",
            "Seismic risk evaluation for infrastructure",
        ]
        texts.extend(hazard_texts)
        labels.extend([QueryIntent.HAZARD_ASSESSMENT.value] * len(hazard_texts))

        # Resource prediction
        resource_texts = [
            "Predict mineral deposits in this region",
            "Geothermal energy potential assessment",
            "Hydrothermal ore formation modeling",
            "Exploration targets for copper deposits",
            "Magmatic sulfide deposit prediction",
            "Geothermal resource evaluation",
        ]
        texts.extend(resource_texts)
        labels.extend([QueryIntent.RESOURCE_PREDICTION.value] * len(resource_texts))

        # Explanation requests
        explanation_texts = [
            "Explain how volcanoes form",
            "What causes earthquakes?",
            "Describe the process of plate tectonics",
            "How do magma chambers work?",
            "What is subduction?",
            "Explain volcanic eruption mechanisms",
        ]
        texts.extend(explanation_texts)
        labels.extend([QueryIntent.EXPLANATION_REQUEST.value] * len(explanation_texts))

        # Comparison queries
        comparison_texts = [
            "Compare shield volcanoes to stratovolcanoes",
            "Difference between P-waves and S-waves",
            "How do explosive eruptions differ from effusive ones?",
            "Compare the 1980 Mount St. Helens eruption to Pinatubo",
            "What's the difference between magnitude and intensity?",
            "Compare continental and oceanic crust",
        ]
        texts.extend(comparison_texts)
        labels.extend([QueryIntent.COMPARISON_QUERY.value] * len(comparison_texts))

        # Historical queries
        historical_texts = [
            "History of eruptions at Mount Vesuvius",
            "When was the last major earthquake in California?",
            "Timeline of Yellowstone eruptions",
            "Past volcanic activity in the Cascades",
            "Historical seismic events in Japan",
            "Evolution of the Hawaiian island chain",
        ]
        texts.extend(historical_texts)
        labels.extend([QueryIntent.HISTORICAL_QUERY.value] * len(historical_texts))

        return texts, labels

    def _train_classifier(self):
        """Train the ML classifier with geological training data"""
        if not SKLEARN_AVAILABLE:
            return

        try:
            texts, labels = self.training_data

            # Vectorize training texts
            X_train = self.vectorizer.fit_transform(texts)

            # Train classifier
            self.ml_classifier.fit(X_train, labels)

            self.logger.info(
                "ML classifier trained successfully",
                training_samples=len(texts),
                unique_intents=len(set(labels)),
            )

        except Exception as e:
            self.logger.error("Failed to train ML classifier", error=str(e))
            raise

    def _classify_with_patterns(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify intent using rule-based patterns"""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    # Calculate confidence based on pattern specificity
                    confidence = min(0.95, 0.7 + len(pattern) / 200)
                    return intent, confidence

        return None, 0.0

    def _classify_with_ml(
        self, query: str
    ) -> Tuple[QueryIntent, float, List[Dict[str, float]]]:
        """Classify intent using ML model"""
        if not SKLEARN_AVAILABLE or not self.ml_classifier:
            return QueryIntent.EXPLANATION_REQUEST, 0.3, []

        try:
            # Vectorize query
            X_query = self.vectorizer.transform([query])

            # Get prediction probabilities
            probabilities = self.ml_classifier.predict_proba(X_query)[0]
            classes = self.ml_classifier.classes_

            # Get top prediction
            top_idx = np.argmax(probabilities)
            top_intent = QueryIntent(classes[top_idx])
            top_confidence = probabilities[top_idx]

            # Get alternative intents
            alternatives = []
            for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
                if (
                    i != top_idx and prob > 0.1
                ):  # Only include alternatives with >10% probability
                    alternatives.append(
                        AlternativeIntent(intent=class_name, confidence=float(prob))
                    )

            # Sort alternatives by confidence
            alternatives.sort(key=lambda x: x.confidence, reverse=True)

            return (
                top_intent,
                float(top_confidence),
                alternatives[:3],
            )  # Top 3 alternatives

        except Exception as e:
            self.logger.error("ML classification failed", error=str(e))
            return QueryIntent.EXPLANATION_REQUEST, 0.3, []  # Default fallback

    async def classify_intent(self, query: str) -> IntentClassificationResult:
        """
        Classify the intent of a geological query

        Args:
            query: Natural language query text

        Returns:
            IntentClassificationResult with classified intent and confidence
        """
        start_time = time.time()

        try:
            self.logger.info("Classifying intent", query=query[:100])

            # First try rule-based classification
            pattern_intent, pattern_confidence = self._classify_with_patterns(query)

            # Also get ML classification
            ml_intent, ml_confidence, alternatives = self._classify_with_ml(query)

            # Choose best classification
            if pattern_intent and pattern_confidence > ml_confidence:
                final_intent = pattern_intent
                final_confidence = pattern_confidence
                self.logger.info(
                    "Using pattern-based classification",
                    intent=final_intent.value,
                    confidence=final_confidence,
                )
            else:
                final_intent = ml_intent
                final_confidence = ml_confidence
                self.logger.info(
                    "Using ML-based classification",
                    intent=final_intent.value,
                    confidence=final_confidence,
                )

            processing_time = (time.time() - start_time) * 1000

            return IntentClassificationResult(
                intent=final_intent,
                confidence=final_confidence,
                alternative_intents=alternatives,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            self.logger.error("Intent classification failed", error=str(e))
            processing_time = (time.time() - start_time) * 1000

            # Return default intent with low confidence
            return IntentClassificationResult(
                intent=QueryIntent.EXPLANATION_REQUEST,
                confidence=0.3,
                alternative_intents=[],
                processing_time_ms=processing_time,
            )
