"""
Anomaly Explanation Engine

Provides comprehensive explanations for detected geological anomalies,
including causal reasoning, geological context, and significance assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class AnomalyExplanation:
    """Comprehensive explanation for a detected anomaly"""
    anomaly_id: str
    timestamp: float
    explanation_type: str
    primary_cause: str
    contributing_factors: List[str]
    geological_context: str
    causal_chain: List[Tuple[str, str, float]]  # (cause, effect, strength)
    confidence_level: float
    uncertainty_sources: List[str]
    recommended_actions: List[str]
    similar_historical_cases: List[str]
    scientific_references: List[str]
    explanation_text: str

@dataclass
class CausalExplanation:
    """Causal explanation for anomaly occurrence"""
    root_causes: List[str]
    intermediate_causes: List[str]
    proximate_causes: List[str]
    causal_pathway: List[Tuple[str, str, float, float]]  # (cause, effect, strength, lag)
    alternative_explanations: List[str]
    causal_confidence: float

class GeologicalKnowledgeBase:
    """Knowledge base for geological explanations"""
    
    def __init__(self):
        # Geological process relationships
        self.process_relationships = {
            'seismic_activity': {
                'causes': ['tectonic_stress', 'magma_intrusion', 'fluid_pressure'],
                'effects': ['ground_deformation', 'landslides', 'volcanic_unrest'],
                'indicators': ['p_wave_velocity', 's_wave_velocity', 'frequency_content'],
                'timescales': {'immediate': 'seconds', 'short_term': 'days', 'long_term': 'years'}
            },
            'volcanic_unrest': {
                'causes': ['magma_ascent', 'gas_exsolution', 'hydrothermal_activity'],
                'effects': ['ground_uplift', 'gas_emissions', 'thermal_anomalies'],
                'indicators': ['so2_flux', 'co2_ratio', 'temperature_increase'],
                'timescales': {'precursory': 'weeks', 'eruptive': 'hours', 'post_eruptive': 'months'}
            },
            'ground_deformation': {
                'causes': ['magma_intrusion', 'aquifer_changes', 'tectonic_loading'],
                'effects': ['surface_displacement', 'stress_changes', 'infrastructure_damage'],
                'indicators': ['insar_phase', 'gps_displacement', 'tilt_measurements'],
                'timescales': {'rapid': 'hours', 'gradual': 'months', 'secular': 'years'}
            },
            'geochemical_changes': {
                'causes': ['magma_degassing', 'hydrothermal_circulation', 'rock_alteration'],
                'effects': ['gas_composition_change', 'water_chemistry_change', 'mineral_precipitation'],
                'indicators': ['isotope_ratios', 'element_concentrations', 'ph_changes'],
                'timescales': {'rapid': 'days', 'moderate': 'weeks', 'slow': 'years'}
            }
        }
        
        # Anomaly patterns and their typical explanations
        self.anomaly_patterns = {
            'sudden_increase': {
                'description': 'Rapid increase in parameter values',
                'typical_causes': ['system_activation', 'threshold_crossing', 'external_trigger'],
                'geological_examples': ['earthquake_swarm_onset', 'eruption_precursor', 'landslide_acceleration']
            },
            'gradual_trend': {
                'description': 'Slow, persistent change over time',
                'typical_causes': ['long_term_loading', 'slow_process_evolution', 'climate_influence'],
                'geological_examples': ['volcanic_inflation', 'tectonic_strain_accumulation', 'groundwater_depletion']
            },
            'periodic_anomaly': {
                'description': 'Regular, repeating anomalous behavior',
                'typical_causes': ['cyclic_processes', 'tidal_effects', 'seasonal_variations'],
                'geological_examples': ['tidal_triggering', 'seasonal_seismicity', 'periodic_gas_emissions']
            },
            'clustered_events': {
                'description': 'Spatially or temporally clustered anomalies',
                'typical_causes': ['common_source', 'cascade_effects', 'stress_transfer'],
                'geological_examples': ['aftershock_sequence', 'volcanic_swarm', 'landslide_cluster']
            },
            'isolated_spike': {
                'description': 'Single, isolated anomalous measurement',
                'typical_causes': ['measurement_error', 'external_interference', 'rare_event'],
                'geological_examples': ['instrument_malfunction', 'lightning_strike', 'meteorite_impact']
            }
        }
        
        # Historical case database
        self.historical_cases = {
            'mount_st_helens_1980': {
                'precursors': ['seismic_swarm', 'ground_deformation', 'gas_emissions'],
                'timeline': 'weeks_to_months',
                'outcome': 'explosive_eruption',
                'lessons': 'Multiple precursors provide stronger warning'
            },
            'loma_prieta_1989': {
                'precursors': ['foreshock_activity', 'groundwater_changes'],
                'timeline': 'days_to_weeks',
                'outcome': 'major_earthquake',
                'lessons': 'Subtle precursors can precede major events'
            },
            'yellowstone_2014': {
                'precursors': ['earthquake_swarm', 'ground_uplift'],
                'timeline': 'months',
                'outcome': 'unrest_without_eruption',
                'lessons': 'Not all unrest leads to eruption'
            }
        }
        
        # Scientific references
        self.scientific_references = {
            'seismic_precursors': [
                'Scholz et al. (1973) - Earthquake prediction',
                'Kanamori & Brodsky (2004) - Physics of earthquakes',
                'Stein & Wysession (2003) - Seismology textbook'
            ],
            'volcanic_monitoring': [
                'Sparks et al. (2012) - Volcanic unrest',
                'Dzurisin (2007) - Volcano deformation',
                'Aiuppa et al. (2019) - Gas monitoring'
            ],
            'anomaly_detection': [
                'Chandola et al. (2009) - Anomaly detection survey',
                'Pimentel et al. (2014) - Machine learning review',
                'Aggarwal (2017) - Outlier analysis'
            ]
        }

class AnomalyExplanationEngine:
    """
    Comprehensive explanation engine for geological anomalies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_base = GeologicalKnowledgeBase()
        
        # Explanation templates
        self.explanation_templates = {
            'causal': "The anomaly in {parameter} is likely caused by {primary_cause}, "
                     "with contributing factors including {contributing_factors}. "
                     "This follows the causal pathway: {causal_chain}.",
            
            'contextual': "Given the geological setting of {setting} and current conditions "
                         "{conditions}, this anomaly is {significance_level} significant. "
                         "Similar patterns have been observed in {historical_cases}.",
            
            'temporal': "The anomaly occurred at {timestamp} and shows {temporal_pattern}. "
                       "The timescale suggests {process_type} geological processes are involved.",
            
            'spatial': "The spatial distribution shows {spatial_pattern}, indicating "
                      "{spatial_interpretation}. This is consistent with {geological_process}.",
            
            'uncertainty': "The explanation has {confidence_level} confidence. "
                          "Main uncertainty sources include {uncertainty_sources}. "
                          "Alternative explanations could be {alternatives}."
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        logger.info("Initialized AnomalyExplanationEngine")
    
    def explain_anomaly(self, anomaly_data: Dict[str, Any], 
                       context: Dict[str, Any] = None) -> AnomalyExplanation:
        """
        Generate comprehensive explanation for an anomaly
        
        Args:
            anomaly_data: Information about the detected anomaly
            context: Additional context information
            
        Returns:
            Comprehensive anomaly explanation
        """
        logger.info(f"Generating explanation for anomaly: {anomaly_data.get('anomaly_id', 'unknown')}")
        
        # Extract key information
        anomaly_id = anomaly_data.get('anomaly_id', f"anomaly_{np.random.randint(10000)}")
        timestamp = anomaly_data.get('timestamp', 0.0)
        affected_parameters = anomaly_data.get('affected_parameters', [])
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        
        # Generate causal explanation
        causal_explanation = self._generate_causal_explanation(anomaly_data, context)
        
        # Identify geological context
        geological_context = self._identify_geological_context(anomaly_data, context)
        
        # Find similar historical cases
        historical_cases = self._find_similar_cases(anomaly_data, context)
        
        # Assess confidence and uncertainty
        confidence_assessment = self._assess_confidence(anomaly_data, causal_explanation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomaly_data, causal_explanation, context)
        
        # Get relevant scientific references
        references = self._get_relevant_references(affected_parameters, anomaly_type)
        
        # Compose final explanation text
        explanation_text = self._compose_explanation_text(
            anomaly_data, causal_explanation, geological_context, 
            confidence_assessment, historical_cases
        )
        
        explanation = AnomalyExplanation(
            anomaly_id=anomaly_id,
            timestamp=timestamp,
            explanation_type=self._classify_explanation_type(anomaly_data),
            primary_cause=causal_explanation.root_causes[0] if causal_explanation.root_causes else "unknown",
            contributing_factors=causal_explanation.intermediate_causes,
            geological_context=geological_context,
            causal_chain=causal_explanation.causal_pathway,
            confidence_level=confidence_assessment['overall_confidence'],
            uncertainty_sources=confidence_assessment['uncertainty_sources'],
            recommended_actions=recommendations,
            similar_historical_cases=historical_cases,
            scientific_references=references,
            explanation_text=explanation_text
        )
        
        logger.info(f"Generated explanation with {explanation.confidence_level:.2f} confidence")
        return explanation
    
    def _generate_causal_explanation(self, anomaly_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> CausalExplanation:
        """Generate causal explanation for the anomaly"""
        affected_parameters = anomaly_data.get('affected_parameters', [])
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        
        # Identify potential root causes
        root_causes = []
        intermediate_causes = []
        proximate_causes = []
        
        # Analyze each affected parameter
        for parameter in affected_parameters:
            # Find geological processes that could affect this parameter
            for process, info in self.knowledge_base.process_relationships.items():
                if any(indicator in parameter.lower() for indicator in info['indicators']):
                    # This process could be involved
                    root_causes.extend(info['causes'])
                    intermediate_causes.append(process)
                    proximate_causes.extend(info['effects'])
        
        # Remove duplicates and rank by likelihood
        root_causes = list(set(root_causes))
        intermediate_causes = list(set(intermediate_causes))
        proximate_causes = list(set(proximate_causes))
        
        # Build causal pathway
        causal_pathway = []
        for i, cause in enumerate(root_causes[:3]):  # Top 3 causes
            for effect in intermediate_causes[:2]:  # Top 2 effects
                strength = 0.8 - i * 0.2  # Decreasing strength
                lag = float(i + 1)  # Increasing lag
                causal_pathway.append((cause, effect, strength, lag))
        
        # Generate alternative explanations
        alternative_explanations = []
        if anomaly_type in self.knowledge_base.anomaly_patterns:
            pattern_info = self.knowledge_base.anomaly_patterns[anomaly_type]
            alternative_explanations = pattern_info['typical_causes']
        
        # Assess causal confidence
        causal_confidence = self._assess_causal_confidence(
            root_causes, intermediate_causes, affected_parameters
        )
        
        return CausalExplanation(
            root_causes=root_causes,
            intermediate_causes=intermediate_causes,
            proximate_causes=proximate_causes,
            causal_pathway=causal_pathway,
            alternative_explanations=alternative_explanations,
            causal_confidence=causal_confidence
        )
    
    def _identify_geological_context(self, anomaly_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> str:
        """Identify geological context for the anomaly"""
        if context and 'geological_setting' in context:
            setting = context['geological_setting']
            
            context_descriptions = {
                'subduction_zone': 'active subduction zone with high seismic and volcanic activity',
                'mid_ocean_ridge': 'divergent plate boundary with active seafloor spreading',
                'continental_rift': 'extensional tectonic environment with normal faulting',
                'hotspot': 'intraplate volcanic region with mantle plume activity',
                'stable_craton': 'geologically stable continental interior',
                'transform_fault': 'strike-slip fault system with lateral motion'
            }
            
            return context_descriptions.get(setting, f"geological setting: {setting}")
        
        # Default context based on anomaly characteristics
        affected_params = anomaly_data.get('affected_parameters', [])
        
        if any('seismic' in param.lower() for param in affected_params):
            return 'seismically active region with ongoing tectonic processes'
        elif any('volcanic' in param.lower() or 'gas' in param.lower() for param in affected_params):
            return 'volcanic environment with active magmatic processes'
        elif any('deformation' in param.lower() for param in affected_params):
            return 'tectonically active area with crustal deformation'
        else:
            return 'geologically active region with multiple ongoing processes'
    
    def _find_similar_cases(self, anomaly_data: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> List[str]:
        """Find similar historical cases"""
        affected_parameters = anomaly_data.get('affected_parameters', [])
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        
        similar_cases = []
        
        # Search historical cases
        for case_name, case_info in self.knowledge_base.historical_cases.items():
            similarity_score = 0
            
            # Check parameter overlap
            for precursor in case_info['precursors']:
                if any(precursor in param.lower() for param in affected_parameters):
                    similarity_score += 1
            
            # Check pattern similarity
            if anomaly_type in case_info.get('pattern_type', ''):
                similarity_score += 2
            
            # Include if sufficiently similar
            if similarity_score >= 1:
                similar_cases.append(f"{case_name}: {case_info['lessons']}")
        
        return similar_cases[:3]  # Return top 3 similar cases
    
    def _assess_confidence(self, anomaly_data: Dict[str, Any], 
                          causal_explanation: CausalExplanation) -> Dict[str, Any]:
        """Assess confidence in the explanation"""
        confidence_factors = []
        uncertainty_sources = []
        
        # Data quality factors
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        if anomaly_score > 0.8:
            confidence_factors.append(0.9)
        elif anomaly_score > 0.6:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            uncertainty_sources.append('low anomaly score')
        
        # Causal chain strength
        if causal_explanation.causal_confidence > 0.7:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
            uncertainty_sources.append('weak causal evidence')
        
        # Number of affected parameters
        n_params = len(anomaly_data.get('affected_parameters', []))
        if n_params > 3:
            confidence_factors.append(0.8)
        elif n_params > 1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            uncertainty_sources.append('limited parameter involvement')
        
        # Historical precedent
        if len(causal_explanation.alternative_explanations) > 0:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            uncertainty_sources.append('no historical precedent')
        
        # Overall confidence
        overall_confidence = np.mean(confidence_factors)
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_factors': confidence_factors,
            'uncertainty_sources': uncertainty_sources,
            'confidence_level': self._categorize_confidence(overall_confidence)
        }
    
    def _categorize_confidence(self, confidence_score: float) -> str:
        """Categorize confidence level"""
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, anomaly_data: Dict[str, Any], 
                                causal_explanation: CausalExplanation,
                                context: Dict[str, Any] = None) -> List[str]:
        """Generate recommended actions based on the anomaly"""
        recommendations = []
        
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        affected_parameters = anomaly_data.get('affected_parameters', [])
        
        # High-priority recommendations for high-score anomalies
        if anomaly_score > 0.8:
            recommendations.append("Immediate investigation required - high anomaly score detected")
            recommendations.append("Increase monitoring frequency for affected parameters")
            recommendations.append("Alert relevant authorities and stakeholders")
        
        # Parameter-specific recommendations
        for parameter in affected_parameters:
            if 'seismic' in parameter.lower():
                recommendations.append("Deploy additional seismometers in the affected area")
                recommendations.append("Analyze earthquake catalog for precursory patterns")
            elif 'deformation' in parameter.lower():
                recommendations.append("Increase GPS and InSAR monitoring frequency")
                recommendations.append("Check for infrastructure impacts")
            elif 'gas' in parameter.lower() or 'chemical' in parameter.lower():
                recommendations.append("Collect additional gas samples for analysis")
                recommendations.append("Monitor for health and safety implications")
        
        # Causal-based recommendations
        for cause in causal_explanation.root_causes[:2]:  # Top 2 causes
            if 'magma' in cause:
                recommendations.append("Monitor volcanic gas emissions and thermal signatures")
            elif 'tectonic' in cause:
                recommendations.append("Analyze regional stress field and fault activity")
            elif 'fluid' in cause:
                recommendations.append("Monitor groundwater levels and hydrogeology")
        
        # Remove duplicates and limit to top recommendations
        recommendations = list(set(recommendations))
        return recommendations[:5]
    
    def _get_relevant_references(self, affected_parameters: List[str], 
                               anomaly_type: str) -> List[str]:
        """Get relevant scientific references"""
        references = []
        
        # Parameter-based references
        for parameter in affected_parameters:
            if 'seismic' in parameter.lower():
                references.extend(self.knowledge_base.scientific_references['seismic_precursors'])
            elif any(term in parameter.lower() for term in ['volcanic', 'gas', 'thermal']):
                references.extend(self.knowledge_base.scientific_references['volcanic_monitoring'])
        
        # Always include anomaly detection references
        references.extend(self.knowledge_base.scientific_references['anomaly_detection'])
        
        # Remove duplicates and limit
        references = list(set(references))
        return references[:5]
    
    def _classify_explanation_type(self, anomaly_data: Dict[str, Any]) -> str:
        """Classify the type of explanation"""
        anomaly_type = anomaly_data.get('anomaly_type', 'unknown')
        affected_parameters = anomaly_data.get('affected_parameters', [])
        
        if len(affected_parameters) > 3:
            return 'multi_parameter_anomaly'
        elif anomaly_type in ['point_anomaly', 'isolated_spike']:
            return 'isolated_anomaly'
        elif anomaly_type in ['collective_anomaly', 'clustered_events']:
            return 'collective_anomaly'
        elif anomaly_type in ['contextual_anomaly']:
            return 'contextual_anomaly'
        else:
            return 'general_anomaly'
    
    def _assess_causal_confidence(self, root_causes: List[str], 
                                intermediate_causes: List[str], 
                                affected_parameters: List[str]) -> float:
        """Assess confidence in causal explanation"""
        confidence_score = 0.0
        
        # More causes identified = higher confidence
        if len(root_causes) > 0:
            confidence_score += 0.3
        if len(intermediate_causes) > 0:
            confidence_score += 0.3
        
        # Parameter coverage
        if len(affected_parameters) > 0:
            coverage = min(1.0, len(intermediate_causes) / len(affected_parameters))
            confidence_score += 0.4 * coverage
        
        return min(1.0, confidence_score)
    
    def _compose_explanation_text(self, anomaly_data: Dict[str, Any],
                                causal_explanation: CausalExplanation,
                                geological_context: str,
                                confidence_assessment: Dict[str, Any],
                                historical_cases: List[str]) -> str:
        """Compose the final explanation text"""
        components = []
        
        # Basic anomaly description
        anomaly_id = anomaly_data.get('anomaly_id', 'unknown')
        affected_params = anomaly_data.get('affected_parameters', [])
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        
        components.append(f"Anomaly {anomaly_id} detected with score {anomaly_score:.2f}")
        
        if affected_params:
            param_str = ", ".join(affected_params[:3])
            components.append(f"affecting parameters: {param_str}")
        
        # Causal explanation
        if causal_explanation.root_causes:
            primary_cause = causal_explanation.root_causes[0]
            components.append(f"Primary cause appears to be {primary_cause}")
            
            if len(causal_explanation.root_causes) > 1:
                other_causes = ", ".join(causal_explanation.root_causes[1:3])
                components.append(f"with contributing factors: {other_causes}")
        
        # Geological context
        components.append(f"This occurs in the context of {geological_context}")
        
        # Confidence and uncertainty
        confidence_level = confidence_assessment['confidence_level']
        components.append(f"Explanation confidence is {confidence_level}")
        
        if confidence_assessment['uncertainty_sources']:
            uncertainty_str = ", ".join(confidence_assessment['uncertainty_sources'][:2])
            components.append(f"Main uncertainties: {uncertainty_str}")
        
        # Historical precedent
        if historical_cases:
            components.append(f"Similar patterns observed in: {historical_cases[0]}")
        
        return ". ".join(components) + "."
    
    def explain_multiple_anomalies(self, anomalies: List[Dict[str, Any]], 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explain multiple related anomalies"""
        logger.info(f"Generating explanations for {len(anomalies)} anomalies")
        
        individual_explanations = []
        for anomaly in anomalies:
            explanation = self.explain_anomaly(anomaly, context)
            individual_explanations.append(explanation)
        
        # Analyze collective patterns
        collective_analysis = self._analyze_collective_patterns(individual_explanations)
        
        # Generate meta-explanation
        meta_explanation = self._generate_meta_explanation(
            individual_explanations, collective_analysis
        )
        
        return {
            'individual_explanations': individual_explanations,
            'collective_analysis': collective_analysis,
            'meta_explanation': meta_explanation,
            'summary': self._generate_summary(individual_explanations, collective_analysis)
        }
    
    def _analyze_collective_patterns(self, explanations: List[AnomalyExplanation]) -> Dict[str, Any]:
        """Analyze patterns across multiple anomaly explanations"""
        # Count common causes
        cause_counts = defaultdict(int)
        for explanation in explanations:
            cause_counts[explanation.primary_cause] += 1
            for factor in explanation.contributing_factors:
                cause_counts[factor] += 1
        
        # Find temporal patterns
        timestamps = [exp.timestamp for exp in explanations]
        temporal_span = max(timestamps) - min(timestamps) if timestamps else 0
        
        # Confidence distribution
        confidences = [exp.confidence_level for exp in explanations]
        avg_confidence = np.mean(confidences)
        
        return {
            'common_causes': dict(cause_counts),
            'dominant_cause': max(cause_counts.items(), key=lambda x: x[1])[0] if cause_counts else None,
            'temporal_span': temporal_span,
            'average_confidence': avg_confidence,
            'explanation_types': [exp.explanation_type for exp in explanations]
        }
    
    def _generate_meta_explanation(self, explanations: List[AnomalyExplanation],
                                 collective_analysis: Dict[str, Any]) -> str:
        """Generate meta-explanation for multiple anomalies"""
        components = []
        
        n_anomalies = len(explanations)
        components.append(f"Analysis of {n_anomalies} related anomalies")
        
        # Dominant cause
        if collective_analysis['dominant_cause']:
            dominant_cause = collective_analysis['dominant_cause']
            components.append(f"reveals {dominant_cause} as the primary underlying cause")
        
        # Temporal pattern
        temporal_span = collective_analysis['temporal_span']
        if temporal_span > 0:
            if temporal_span < 3600:  # Less than 1 hour
                components.append("occurring within a short time window, suggesting rapid process")
            elif temporal_span < 86400:  # Less than 1 day
                components.append("spanning several hours, indicating evolving geological process")
            else:
                components.append("occurring over extended period, suggesting long-term geological evolution")
        
        # Confidence assessment
        avg_confidence = collective_analysis['average_confidence']
        if avg_confidence > 0.7:
            components.append("The collective evidence provides high confidence in this interpretation")
        elif avg_confidence > 0.5:
            components.append("The evidence provides moderate confidence, with some uncertainties remaining")
        else:
            components.append("Significant uncertainties remain in the interpretation")
        
        return ". ".join(components) + "."
    
    def _generate_summary(self, explanations: List[AnomalyExplanation],
                         collective_analysis: Dict[str, Any]) -> str:
        """Generate concise summary of multiple anomaly explanations"""
        n_anomalies = len(explanations)
        dominant_cause = collective_analysis.get('dominant_cause', 'unknown processes')
        avg_confidence = collective_analysis.get('average_confidence', 0.0)
        
        confidence_desc = "high" if avg_confidence > 0.7 else "moderate" if avg_confidence > 0.5 else "low"
        
        return (f"{n_anomalies} anomalies detected, primarily attributed to {dominant_cause}. "
                f"Explanation confidence: {confidence_desc}. "
                f"Collective analysis suggests coordinated geological processes.")
    
    def update_knowledge_base(self, new_cases: Dict[str, Any], 
                            new_relationships: Dict[str, Any]) -> None:
        """Update the geological knowledge base with new information"""
        logger.info("Updating geological knowledge base")
        
        # Update historical cases
        if 'historical_cases' in new_cases:
            self.knowledge_base.historical_cases.update(new_cases['historical_cases'])
        
        # Update process relationships
        if 'process_relationships' in new_relationships:
            for process, info in new_relationships['process_relationships'].items():
                if process in self.knowledge_base.process_relationships:
                    # Merge with existing information
                    existing = self.knowledge_base.process_relationships[process]
                    for key, value in info.items():
                        if isinstance(value, list):
                            existing[key] = list(set(existing.get(key, []) + value))
                        else:
                            existing[key] = value
                else:
                    # Add new process
                    self.knowledge_base.process_relationships[process] = info
        
        logger.info("Knowledge base updated successfully")