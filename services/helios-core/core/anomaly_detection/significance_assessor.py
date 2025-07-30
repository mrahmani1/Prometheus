"""
Significance Assessor for Geological Anomalies

Assesses the geological and scientific significance of detected anomalies
using domain expertise and statistical analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class SignificanceLevel(Enum):
    """Significance levels for geological anomalies"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NEGLIGIBLE = "negligible"

@dataclass
class SignificanceAssessment:
    """Assessment of anomaly significance"""
    anomaly_id: str
    overall_significance: SignificanceLevel
    statistical_significance: float
    geological_significance: float
    societal_significance: float
    scientific_significance: float
    temporal_significance: float
    spatial_significance: float
    significance_factors: Dict[str, float]
    risk_level: str
    urgency_level: str
    confidence_in_assessment: float
    assessment_reasoning: str
    recommended_response_level: str

class GeologicalSignificanceFramework:
    """Framework for assessing geological significance"""
    
    def __init__(self):
        # Significance criteria for different geological processes
        self.process_significance_criteria = {
            'seismic_activity': {
                'magnitude_thresholds': {
                    'critical': 7.0,
                    'high': 6.0,
                    'moderate': 5.0,
                    'low': 4.0
                },
                'frequency_thresholds': {
                    'critical': 10,  # events per day
                    'high': 5,
                    'moderate': 2,
                    'low': 1
                },
                'depth_factors': {
                    'shallow': 2.0,  # < 10 km
                    'intermediate': 1.5,  # 10-70 km
                    'deep': 1.0  # > 70 km
                }
            },
            'volcanic_activity': {
                'vei_thresholds': {
                    'critical': 4,
                    'high': 3,
                    'moderate': 2,
                    'low': 1
                },
                'gas_flux_thresholds': {
                    'critical': 10000,  # tons/day SO2
                    'high': 1000,
                    'moderate': 100,
                    'low': 10
                },
                'deformation_thresholds': {
                    'critical': 0.1,  # m/year
                    'high': 0.05,
                    'moderate': 0.01,
                    'low': 0.005
                }
            },
            'ground_deformation': {
                'rate_thresholds': {
                    'critical': 0.5,  # m/year
                    'high': 0.1,
                    'moderate': 0.05,
                    'low': 0.01
                },
                'area_thresholds': {
                    'critical': 1000,  # km²
                    'high': 100,
                    'moderate': 10,
                    'low': 1
                }
            },
            'geochemical_changes': {
                'concentration_change_thresholds': {
                    'critical': 10.0,  # factor change
                    'high': 5.0,
                    'moderate': 2.0,
                    'low': 1.5
                },
                'isotope_shift_thresholds': {
                    'critical': 5.0,  # sigma units
                    'high': 3.0,
                    'moderate': 2.0,
                    'low': 1.0
                }
            }
        }
        
        # Population and infrastructure impact factors
        self.impact_factors = {
            'population_density': {
                'urban': 3.0,
                'suburban': 2.0,
                'rural': 1.0,
                'remote': 0.5
            },
            'infrastructure_criticality': {
                'critical_infrastructure': 3.0,
                'major_infrastructure': 2.0,
                'minor_infrastructure': 1.0,
                'no_infrastructure': 0.5
            },
            'economic_importance': {
                'major_economic_center': 3.0,
                'regional_center': 2.0,
                'local_center': 1.0,
                'minimal_economic_activity': 0.5
            }
        }
        
        # Scientific significance factors
        self.scientific_factors = {
            'novelty': {
                'unprecedented': 3.0,
                'rare': 2.5,
                'uncommon': 2.0,
                'known': 1.0
            },
            'research_value': {
                'breakthrough_potential': 3.0,
                'high_research_value': 2.5,
                'moderate_research_value': 2.0,
                'limited_research_value': 1.0
            },
            'theoretical_implications': {
                'paradigm_changing': 3.0,
                'theory_extending': 2.5,
                'theory_confirming': 2.0,
                'routine_observation': 1.0
            }
        }

class SignificanceAssessor:
    """
    Comprehensive significance assessor for geological anomalies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.significance_framework = GeologicalSignificanceFramework()
        
        # Statistical significance parameters
        self.statistical_thresholds = {
            'p_value_critical': 0.001,
            'p_value_high': 0.01,
            'p_value_moderate': 0.05,
            'p_value_low': 0.1
        }
        
        # Temporal significance parameters
        self.temporal_factors = {
            'recurrence_intervals': {
                'daily': 0.5,
                'weekly': 1.0,
                'monthly': 1.5,
                'yearly': 2.0,
                'decadal': 2.5,
                'centennial': 3.0
            },
            'duration_factors': {
                'seconds': 0.5,
                'minutes': 1.0,
                'hours': 1.5,
                'days': 2.0,
                'weeks': 2.5,
                'months': 3.0
            }
        }
        
        # Historical significance database
        self.historical_significance = {}
        
        logger.info("Initialized SignificanceAssessor")
    
    def assess_significance(self, anomaly_data: Dict[str, Any], 
                          context: Dict[str, Any] = None) -> SignificanceAssessment:
        """
        Assess the overall significance of a geological anomaly
        
        Args:
            anomaly_data: Information about the detected anomaly
            context: Additional context information
            
        Returns:
            Comprehensive significance assessment
        """
        logger.info(f"Assessing significance for anomaly: {anomaly_data.get('anomaly_id', 'unknown')}")
        
        anomaly_id = anomaly_data.get('anomaly_id', f"anomaly_{np.random.randint(10000)}")
        
        # Assess different dimensions of significance
        statistical_sig = self._assess_statistical_significance(anomaly_data)
        geological_sig = self._assess_geological_significance(anomaly_data, context)
        societal_sig = self._assess_societal_significance(anomaly_data, context)
        scientific_sig = self._assess_scientific_significance(anomaly_data, context)
        temporal_sig = self._assess_temporal_significance(anomaly_data, context)
        spatial_sig = self._assess_spatial_significance(anomaly_data, context)
        
        # Compile significance factors
        significance_factors = {
            'statistical': statistical_sig,
            'geological': geological_sig,
            'societal': societal_sig,
            'scientific': scientific_sig,
            'temporal': temporal_sig,
            'spatial': spatial_sig
        }
        
        # Compute overall significance
        overall_sig = self._compute_overall_significance(significance_factors)
        
        # Assess risk and urgency
        risk_level = self._assess_risk_level(significance_factors, context)
        urgency_level = self._assess_urgency_level(significance_factors, context)
        
        # Determine recommended response
        response_level = self._determine_response_level(overall_sig, risk_level, urgency_level)
        
        # Assess confidence in the assessment
        assessment_confidence = self._assess_assessment_confidence(
            anomaly_data, significance_factors
        )
        
        # Generate reasoning
        reasoning = self._generate_assessment_reasoning(
            significance_factors, overall_sig, risk_level, urgency_level
        )
        
        assessment = SignificanceAssessment(
            anomaly_id=anomaly_id,
            overall_significance=overall_sig,
            statistical_significance=statistical_sig,
            geological_significance=geological_sig,
            societal_significance=societal_sig,
            scientific_significance=scientific_sig,
            temporal_significance=temporal_sig,
            spatial_significance=spatial_sig,
            significance_factors=significance_factors,
            risk_level=risk_level,
            urgency_level=urgency_level,
            confidence_in_assessment=assessment_confidence,
            assessment_reasoning=reasoning,
            recommended_response_level=response_level
        )
        
        logger.info(f"Assessed significance as {overall_sig.value} with {assessment_confidence:.2f} confidence")
        return assessment
    
    def _assess_statistical_significance(self, anomaly_data: Dict[str, Any]) -> float:
        """Assess statistical significance of the anomaly"""
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        
        # Convert anomaly score to approximate p-value
        # Higher anomaly scores correspond to lower p-values
        if anomaly_score > 0.95:
            p_value = 0.001
        elif anomaly_score > 0.9:
            p_value = 0.01
        elif anomaly_score > 0.8:
            p_value = 0.05
        elif anomaly_score > 0.7:
            p_value = 0.1
        else:
            p_value = 0.2
        
        # Convert p-value to significance score (0-1)
        if p_value <= self.statistical_thresholds['p_value_critical']:
            return 1.0
        elif p_value <= self.statistical_thresholds['p_value_high']:
            return 0.8
        elif p_value <= self.statistical_thresholds['p_value_moderate']:
            return 0.6
        elif p_value <= self.statistical_thresholds['p_value_low']:
            return 0.4
        else:
            return 0.2
    
    def _assess_geological_significance(self, anomaly_data: Dict[str, Any], 
                                      context: Dict[str, Any] = None) -> float:
        """Assess geological significance based on process characteristics"""
        affected_parameters = anomaly_data.get('affected_parameters', [])
        anomaly_magnitude = anomaly_data.get('magnitude', 0.0)
        
        geological_scores = []
        
        # Assess significance for each affected parameter
        for parameter in affected_parameters:
            param_score = self._assess_parameter_significance(parameter, anomaly_magnitude, context)
            geological_scores.append(param_score)
        
        # Consider multi-parameter interactions
        if len(geological_scores) > 1:
            # Multi-parameter anomalies are generally more significant
            interaction_bonus = min(0.3, 0.1 * (len(geological_scores) - 1))
            base_score = np.mean(geological_scores)
            return min(1.0, base_score + interaction_bonus)
        elif len(geological_scores) == 1:
            return geological_scores[0]
        else:
            return 0.1  # Default low significance
    
    def _assess_parameter_significance(self, parameter: str, magnitude: float, 
                                     context: Dict[str, Any] = None) -> float:
        """Assess significance of a specific parameter anomaly"""
        # Identify parameter type
        if 'seismic' in parameter.lower() or 'earthquake' in parameter.lower():
            return self._assess_seismic_significance(magnitude, context)
        elif 'volcanic' in parameter.lower() or 'eruption' in parameter.lower():
            return self._assess_volcanic_significance(magnitude, context)
        elif 'deformation' in parameter.lower():
            return self._assess_deformation_significance(magnitude, context)
        elif 'gas' in parameter.lower() or 'chemical' in parameter.lower():
            return self._assess_geochemical_significance(magnitude, context)
        else:
            # Generic assessment based on magnitude
            return min(1.0, magnitude / 10.0)
    
    def _assess_seismic_significance(self, magnitude: float, 
                                   context: Dict[str, Any] = None) -> float:
        """Assess significance of seismic anomaly"""
        criteria = self.significance_framework.process_significance_criteria['seismic_activity']
        
        # Magnitude-based significance
        mag_thresholds = criteria['magnitude_thresholds']
        if magnitude >= mag_thresholds['critical']:
            mag_score = 1.0
        elif magnitude >= mag_thresholds['high']:
            mag_score = 0.8
        elif magnitude >= mag_thresholds['moderate']:
            mag_score = 0.6
        elif magnitude >= mag_thresholds['low']:
            mag_score = 0.4
        else:
            mag_score = 0.2
        
        # Depth factor
        depth_factor = 1.0
        if context and 'depth' in context:
            depth = context['depth']
            if depth < 10:
                depth_factor = criteria['depth_factors']['shallow']
            elif depth < 70:
                depth_factor = criteria['depth_factors']['intermediate']
            else:
                depth_factor = criteria['depth_factors']['deep']
        
        return min(1.0, mag_score * depth_factor)
    
    def _assess_volcanic_significance(self, magnitude: float, 
                                    context: Dict[str, Any] = None) -> float:
        """Assess significance of volcanic anomaly"""
        criteria = self.significance_framework.process_significance_criteria['volcanic_activity']
        
        # Use VEI if available, otherwise use magnitude as proxy
        vei = context.get('vei', magnitude) if context else magnitude
        
        vei_thresholds = criteria['vei_thresholds']
        if vei >= vei_thresholds['critical']:
            return 1.0
        elif vei >= vei_thresholds['high']:
            return 0.8
        elif vei >= vei_thresholds['moderate']:
            return 0.6
        elif vei >= vei_thresholds['low']:
            return 0.4
        else:
            return 0.2
    
    def _assess_deformation_significance(self, magnitude: float, 
                                       context: Dict[str, Any] = None) -> float:
        """Assess significance of ground deformation anomaly"""
        criteria = self.significance_framework.process_significance_criteria['ground_deformation']
        
        # Rate-based significance
        rate_thresholds = criteria['rate_thresholds']
        if magnitude >= rate_thresholds['critical']:
            return 1.0
        elif magnitude >= rate_thresholds['high']:
            return 0.8
        elif magnitude >= rate_thresholds['moderate']:
            return 0.6
        elif magnitude >= rate_thresholds['low']:
            return 0.4
        else:
            return 0.2
    
    def _assess_geochemical_significance(self, magnitude: float, 
                                       context: Dict[str, Any] = None) -> float:
        """Assess significance of geochemical anomaly"""
        criteria = self.significance_framework.process_significance_criteria['geochemical_changes']
        
        # Concentration change significance
        conc_thresholds = criteria['concentration_change_thresholds']
        if magnitude >= conc_thresholds['critical']:
            return 1.0
        elif magnitude >= conc_thresholds['high']:
            return 0.8
        elif magnitude >= conc_thresholds['moderate']:
            return 0.6
        elif magnitude >= conc_thresholds['low']:
            return 0.4
        else:
            return 0.2
    
    def _assess_societal_significance(self, anomaly_data: Dict[str, Any], 
                                    context: Dict[str, Any] = None) -> float:
        """Assess societal significance based on potential impacts"""
        if not context:
            return 0.5  # Default moderate significance
        
        impact_scores = []
        
        # Population impact
        population_density = context.get('population_density', 'rural')
        pop_factor = self.significance_framework.impact_factors['population_density'].get(
            population_density, 1.0
        )
        impact_scores.append(min(1.0, pop_factor / 3.0))
        
        # Infrastructure impact
        infrastructure = context.get('infrastructure_criticality', 'minor_infrastructure')
        infra_factor = self.significance_framework.impact_factors['infrastructure_criticality'].get(
            infrastructure, 1.0
        )
        impact_scores.append(min(1.0, infra_factor / 3.0))
        
        # Economic impact
        economic_importance = context.get('economic_importance', 'local_center')
        econ_factor = self.significance_framework.impact_factors['economic_importance'].get(
            economic_importance, 1.0
        )
        impact_scores.append(min(1.0, econ_factor / 3.0))
        
        return np.mean(impact_scores)
    
    def _assess_scientific_significance(self, anomaly_data: Dict[str, Any], 
                                      context: Dict[str, Any] = None) -> float:
        """Assess scientific significance of the anomaly"""
        scientific_scores = []
        
        # Novelty assessment
        anomaly_type = anomaly_data.get('anomaly_type', 'known')
        if anomaly_type in ['novel_pattern', 'unprecedented']:
            novelty_score = 1.0
        elif anomaly_type in ['rare_pattern', 'unusual']:
            novelty_score = 0.8
        elif anomaly_type in ['uncommon_pattern']:
            novelty_score = 0.6
        else:
            novelty_score = 0.4
        
        scientific_scores.append(novelty_score)
        
        # Research value based on anomaly characteristics
        affected_parameters = anomaly_data.get('affected_parameters', [])
        if len(affected_parameters) > 3:
            research_score = 0.9  # Multi-parameter anomalies have high research value
        elif len(affected_parameters) > 1:
            research_score = 0.7
        else:
            research_score = 0.5
        
        scientific_scores.append(research_score)
        
        # Theoretical implications
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        if anomaly_score > 0.9:
            theory_score = 0.9  # Very high anomaly scores may challenge existing theory
        elif anomaly_score > 0.8:
            theory_score = 0.7
        else:
            theory_score = 0.5
        
        scientific_scores.append(theory_score)
        
        return np.mean(scientific_scores)
    
    def _assess_temporal_significance(self, anomaly_data: Dict[str, Any], 
                                    context: Dict[str, Any] = None) -> float:
        """Assess temporal significance of the anomaly"""
        temporal_scores = []
        
        # Duration significance
        duration = anomaly_data.get('duration', 0.0)
        if duration > 86400 * 30:  # > 1 month
            duration_score = 1.0
        elif duration > 86400 * 7:  # > 1 week
            duration_score = 0.8
        elif duration > 86400:  # > 1 day
            duration_score = 0.6
        elif duration > 3600:  # > 1 hour
            duration_score = 0.4
        else:
            duration_score = 0.2
        
        temporal_scores.append(duration_score)
        
        # Recurrence interval significance
        if context and 'recurrence_interval' in context:
            recurrence = context['recurrence_interval']
            if recurrence > 365 * 100:  # > 100 years
                recurrence_score = 1.0
            elif recurrence > 365 * 10:  # > 10 years
                recurrence_score = 0.8
            elif recurrence > 365:  # > 1 year
                recurrence_score = 0.6
            elif recurrence > 30:  # > 1 month
                recurrence_score = 0.4
            else:
                recurrence_score = 0.2
            
            temporal_scores.append(recurrence_score)
        
        return np.mean(temporal_scores) if temporal_scores else 0.5
    
    def _assess_spatial_significance(self, anomaly_data: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> float:
        """Assess spatial significance of the anomaly"""
        spatial_scores = []
        
        # Spatial extent
        spatial_extent = anomaly_data.get('spatial_extent', 0.0)
        if spatial_extent > 1000:  # > 1000 km²
            extent_score = 1.0
        elif spatial_extent > 100:  # > 100 km²
            extent_score = 0.8
        elif spatial_extent > 10:  # > 10 km²
            extent_score = 0.6
        elif spatial_extent > 1:  # > 1 km²
            extent_score = 0.4
        else:
            extent_score = 0.2
        
        spatial_scores.append(extent_score)
        
        # Location significance
        if context and 'location_significance' in context:
            location_sig = context['location_significance']
            if location_sig in ['major_fault', 'active_volcano', 'populated_area']:
                location_score = 1.0
            elif location_sig in ['minor_fault', 'dormant_volcano', 'rural_area']:
                location_score = 0.6
            else:
                location_score = 0.3
            
            spatial_scores.append(location_score)
        
        return np.mean(spatial_scores) if spatial_scores else 0.5
    
    def _compute_overall_significance(self, significance_factors: Dict[str, float]) -> SignificanceLevel:
        """Compute overall significance level"""
        # Weighted combination of significance factors
        weights = {
            'geological': 0.3,
            'statistical': 0.2,
            'societal': 0.2,
            'scientific': 0.1,
            'temporal': 0.1,
            'spatial': 0.1
        }
        
        weighted_score = sum(
            weights.get(factor, 0.1) * score 
            for factor, score in significance_factors.items()
        )
        
        # Convert to significance level
        if weighted_score >= 0.9:
            return SignificanceLevel.CRITICAL
        elif weighted_score >= 0.7:
            return SignificanceLevel.HIGH
        elif weighted_score >= 0.5:
            return SignificanceLevel.MODERATE
        elif weighted_score >= 0.3:
            return SignificanceLevel.LOW
        else:
            return SignificanceLevel.NEGLIGIBLE
    
    def _assess_risk_level(self, significance_factors: Dict[str, float], 
                          context: Dict[str, Any] = None) -> str:
        """Assess risk level based on significance and context"""
        geological_sig = significance_factors.get('geological', 0.0)
        societal_sig = significance_factors.get('societal', 0.0)
        
        # Risk is primarily driven by geological and societal significance
        risk_score = 0.6 * geological_sig + 0.4 * societal_sig
        
        if risk_score >= 0.8:
            return 'extreme'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'moderate'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_urgency_level(self, significance_factors: Dict[str, float], 
                            context: Dict[str, Any] = None) -> str:
        """Assess urgency level for response"""
        temporal_sig = significance_factors.get('temporal', 0.0)
        geological_sig = significance_factors.get('geological', 0.0)
        
        # Urgency depends on temporal characteristics and geological severity
        urgency_score = 0.5 * (1.0 - temporal_sig) + 0.5 * geological_sig
        
        if urgency_score >= 0.8:
            return 'immediate'
        elif urgency_score >= 0.6:
            return 'urgent'
        elif urgency_score >= 0.4:
            return 'moderate'
        elif urgency_score >= 0.2:
            return 'low'
        else:
            return 'routine'
    
    def _determine_response_level(self, overall_significance: SignificanceLevel, 
                                risk_level: str, urgency_level: str) -> str:
        """Determine recommended response level"""
        # Response matrix based on significance, risk, and urgency
        if overall_significance == SignificanceLevel.CRITICAL:
            return 'emergency_response'
        elif overall_significance == SignificanceLevel.HIGH:
            if risk_level in ['extreme', 'high'] or urgency_level in ['immediate', 'urgent']:
                return 'immediate_investigation'
            else:
                return 'priority_investigation'
        elif overall_significance == SignificanceLevel.MODERATE:
            if urgency_level in ['immediate', 'urgent']:
                return 'expedited_investigation'
            else:
                return 'routine_investigation'
        elif overall_significance == SignificanceLevel.LOW:
            return 'monitoring_enhancement'
        else:
            return 'routine_monitoring'
    
    def _assess_assessment_confidence(self, anomaly_data: Dict[str, Any], 
                                    significance_factors: Dict[str, float]) -> float:
        """Assess confidence in the significance assessment"""
        confidence_factors = []
        
        # Data quality
        anomaly_score = anomaly_data.get('anomaly_score', 0.0)
        if anomaly_score > 0.8:
            confidence_factors.append(0.9)
        elif anomaly_score > 0.6:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Number of parameters involved
        n_params = len(anomaly_data.get('affected_parameters', []))
        if n_params > 2:
            confidence_factors.append(0.8)
        elif n_params > 0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Consistency across significance dimensions
        sig_values = list(significance_factors.values())
        if len(sig_values) > 1:
            sig_variance = np.var(sig_values)
            consistency_score = max(0.3, 1.0 - sig_variance * 2)
            confidence_factors.append(consistency_score)
        
        return np.mean(confidence_factors)
    
    def _generate_assessment_reasoning(self, significance_factors: Dict[str, float],
                                     overall_significance: SignificanceLevel,
                                     risk_level: str, urgency_level: str) -> str:
        """Generate reasoning for the significance assessment"""
        reasoning_parts = []
        
        # Overall assessment
        reasoning_parts.append(f"Overall significance assessed as {overall_significance.value}")
        
        # Key contributing factors
        sorted_factors = sorted(significance_factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_factors[:2]
        
        if top_factors:
            factor_names = [f"{factor} ({score:.2f})" for factor, score in top_factors]
            reasoning_parts.append(f"driven primarily by {' and '.join(factor_names)}")
        
        # Risk and urgency
        reasoning_parts.append(f"Risk level: {risk_level}, Urgency: {urgency_level}")
        
        # Specific thresholds exceeded
        high_factors = [factor for factor, score in significance_factors.items() if score > 0.7]
        if high_factors:
            reasoning_parts.append(f"High significance in: {', '.join(high_factors)}")
        
        return ". ".join(reasoning_parts) + "."
    
    def assess_multiple_anomalies(self, anomalies: List[Dict[str, Any]], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess significance of multiple related anomalies"""
        logger.info(f"Assessing significance for {len(anomalies)} anomalies")
        
        individual_assessments = []
        for anomaly in anomalies:
            assessment = self.assess_significance(anomaly, context)
            individual_assessments.append(assessment)
        
        # Collective significance analysis
        collective_analysis = self._analyze_collective_significance(individual_assessments)
        
        # Meta-assessment
        meta_assessment = self._generate_meta_assessment(
            individual_assessments, collective_analysis
        )
        
        return {
            'individual_assessments': individual_assessments,
            'collective_analysis': collective_analysis,
            'meta_assessment': meta_assessment,
            'highest_significance': max(
                individual_assessments, 
                key=lambda x: self._significance_to_numeric(x.overall_significance)
            ),
            'summary': self._generate_significance_summary(individual_assessments)
        }
    
    def _analyze_collective_significance(self, assessments: List[SignificanceAssessment]) -> Dict[str, Any]:
        """Analyze collective significance patterns"""
        significance_levels = [a.overall_significance for a in assessments]
        risk_levels = [a.risk_level for a in assessments]
        urgency_levels = [a.urgency_level for a in assessments]
        
        # Count significance levels
        sig_counts = defaultdict(int)
        for sig in significance_levels:
            sig_counts[sig.value] += 1
        
        # Temporal clustering
        timestamps = [a.anomaly_id for a in assessments]  # Simplified
        
        return {
            'significance_distribution': dict(sig_counts),
            'dominant_significance': max(sig_counts.items(), key=lambda x: x[1])[0],
            'highest_significance': max(significance_levels, key=self._significance_to_numeric).value,
            'risk_distribution': {level: risk_levels.count(level) for level in set(risk_levels)},
            'urgency_distribution': {level: urgency_levels.count(level) for level in set(urgency_levels)},
            'collective_confidence': np.mean([a.confidence_in_assessment for a in assessments])
        }
    
    def _significance_to_numeric(self, significance: SignificanceLevel) -> float:
        """Convert significance level to numeric value for comparison"""
        mapping = {
            SignificanceLevel.CRITICAL: 5.0,
            SignificanceLevel.HIGH: 4.0,
            SignificanceLevel.MODERATE: 3.0,
            SignificanceLevel.LOW: 2.0,
            SignificanceLevel.NEGLIGIBLE: 1.0
        }
        return mapping.get(significance, 0.0)
    
    def _generate_meta_assessment(self, assessments: List[SignificanceAssessment],
                                collective_analysis: Dict[str, Any]) -> str:
        """Generate meta-assessment for multiple anomalies"""
        n_anomalies = len(assessments)
        highest_sig = collective_analysis['highest_significance']
        dominant_sig = collective_analysis['dominant_significance']
        
        meta_parts = []
        meta_parts.append(f"Collective assessment of {n_anomalies} anomalies")
        meta_parts.append(f"Highest individual significance: {highest_sig}")
        meta_parts.append(f"Most common significance level: {dominant_sig}")
        
        # Risk assessment
        risk_dist = collective_analysis['risk_distribution']
        high_risk_count = sum(count for level, count in risk_dist.items() 
                            if level in ['extreme', 'high'])
        
        if high_risk_count > 0:
            meta_parts.append(f"{high_risk_count} anomalies pose high or extreme risk")
        
        # Collective confidence
        collective_confidence = collective_analysis['collective_confidence']
        if collective_confidence > 0.7:
            meta_parts.append("High confidence in collective assessment")
        elif collective_confidence > 0.5:
            meta_parts.append("Moderate confidence in collective assessment")
        else:
            meta_parts.append("Low confidence in collective assessment")
        
        return ". ".join(meta_parts) + "."
    
    def _generate_significance_summary(self, assessments: List[SignificanceAssessment]) -> str:
        """Generate concise summary of significance assessments"""
        n_anomalies = len(assessments)
        
        # Count by significance level
        critical_count = sum(1 for a in assessments if a.overall_significance == SignificanceLevel.CRITICAL)
        high_count = sum(1 for a in assessments if a.overall_significance == SignificanceLevel.HIGH)
        
        summary_parts = [f"{n_anomalies} anomalies assessed"]
        
        if critical_count > 0:
            summary_parts.append(f"{critical_count} critical")
        if high_count > 0:
            summary_parts.append(f"{high_count} high significance")
        
        # Response recommendation
        if critical_count > 0:
            summary_parts.append("Emergency response recommended")
        elif high_count > 0:
            summary_parts.append("Immediate investigation recommended")
        else:
            summary_parts.append("Routine monitoring sufficient")
        
        return ". ".join(summary_parts) + "."
    
    def update_significance_criteria(self, new_criteria: Dict[str, Any]) -> None:
        """Update significance assessment criteria"""
        logger.info("Updating significance assessment criteria")
        
        if 'process_criteria' in new_criteria:
            for process, criteria in new_criteria['process_criteria'].items():
                if process in self.significance_framework.process_significance_criteria:
                    self.significance_framework.process_significance_criteria[process].update(criteria)
                else:
                    self.significance_framework.process_significance_criteria[process] = criteria
        
        if 'statistical_thresholds' in new_criteria:
            self.statistical_thresholds.update(new_criteria['statistical_thresholds'])
        
        logger.info("Significance criteria updated successfully")