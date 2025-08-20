"""
Knowledge Graph Integration for Enhanced Kalman Filter
Connects EnKF results with scientific knowledge representation and causal inference
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import asyncio

from .scientific_knowledge_graph import ScientificKnowledgeGraph, EntityType, RelationType
from .causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType
from .enhanced_kalman_filter import EnhancedEnsembleKalmanFilter, ReservoirParameters

class EnKFKnowledgeIntegrator:
    """
    Integrates Enhanced Kalman Filter results with scientific knowledge graph
    and causal inference systems for improved volcanic process understanding
    """
    
    def __init__(self, knowledge_graph: ScientificKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)
        
        # Initialize volcano deformation ontology
        self._initialize_deformation_ontology()
    
    def _initialize_deformation_ontology(self):
        """Initialize ontological framework for volcanic deformation"""
        
        # Core deformation concepts
        deformation_entities = [
            ('magma_reservoir', EntityType.CONCEPT, {
                'description': 'Underground magma storage system causing surface deformation',
                'physical_properties': ['depth', 'volume', 'pressure', 'shape']
            }),
            ('surface_deformation', EntityType.PHENOMENON, {
                'description': 'Observable ground displacement patterns',
                'measurement_types': ['gps', 'insar', 'tilt', 'strain']
            }),
            ('ensemble_kalman_filter', EntityType.METHOD, {
                'description': 'Data assimilation technique for parameter estimation',
                'reformulation': 'Albright_Gregg_2022',
                'parameters': ['dv_star', 'gamma', 'delta', 'psi']
            }),
            ('unitless_parameterization', EntityType.CONCEPT, {
                'description': 'Reformulated parameter space avoiding physical constraints',
                'advantages': ['no_impossible_conditions', 'improved_convergence', 'reduced_correlations']
            }),
            ('geodetic_observations', EntityType.DATASET, {
                'description': 'GPS and InSAR measurements of ground displacement',
                'spatial_resolution': 'variable',
                'temporal_resolution': 'variable'
            })
        ]
        
        # Add entities to knowledge graph
        for entity_id, entity_type, properties in deformation_entities:
            asyncio.create_task(
                self.knowledge_graph.add_entity(entity_id, entity_type, properties)
            )
        
        # Define causal relationships
        causal_relationships = [
            ('magma_reservoir', 'surface_deformation', RelationType.CAUSES, 0.95),
            ('geodetic_observations', 'ensemble_kalman_filter', RelationType.MEASURED_BY, 0.90),
            ('ensemble_kalman_filter', 'magma_reservoir', RelationType.INFLUENCES, 0.85),
            ('unitless_parameterization', 'ensemble_kalman_filter', RelationType.PART_OF, 1.0)
        ]
        
        # Add relationships
        for source, target, relation_type, confidence in causal_relationships:
            asyncio.create_task(
                self.knowledge_graph.add_relationship(source, target, relation_type, confidence)
            )
    
    async def integrate_enkf_results(self, volcano_id: str, enkf_results: Dict[str, Any],
                                   reservoir_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate EnKF results into knowledge graph and extract insights
        """
        
        self.logger.info(f"Integrating EnKF results for {volcano_id} into knowledge graph")
        
        # Create volcano-specific entities
        volcano_entity = f"volcano_{volcano_id}"
        await self.knowledge_graph.add_entity(
            volcano_entity, 
            EntityType.LOCATION,
            {
                'volcano_id': volcano_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'enkf_results': enkf_results
            }
        )
        
        # Add reservoir state entity
        reservoir_entity = f"reservoir_{volcano_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self.knowledge_graph.add_entity(
            reservoir_entity,
            EntityType.CONCEPT,
            {
                'depth_estimate': reservoir_state.get('depth_estimate', {}),
                'volume_estimate': reservoir_state.get('volume_estimate', {}),
                'pressure_change': reservoir_state.get('pressure_change', {}),
                'parameter_correlations': reservoir_state.get('parameter_correlations', {}),
                'source': 'enkf_assimilation'
            }
        )
        
        # Link volcano to reservoir
        await self.knowledge_graph.add_relationship(
            volcano_entity, 
            reservoir_entity, 
            RelationType.PART_OF,
            confidence=0.9
        )
        
        # Add assimilation process entity
        assimilation_entity = f"assimilation_{volcano_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self.knowledge_graph.add_entity(
            assimilation_entity,
            EntityType.PROCESS,
            {
                'update_metrics': enkf_results.get('update_metrics', {}),
                'ensemble_size': enkf_results.get('ensemble_size', 0),
                'assimilation_success': enkf_results.get('assimilation_success', False),
                'method': 'enhanced_ensemble_kalman_filter'
            }
        )
        
        # Connect assimilation to results
        await self.knowledge_graph.add_relationship(
            assimilation_entity,
            reservoir_entity,
            RelationType.CAUSES,
            confidence=enkf_results.get('assimilation_success', False) * 0.9
        )
        
        # Extract and store parameter insights
        parameter_insights = await self._extract_parameter_insights(reservoir_state, enkf_results)
        
        # Add parameter confidence entity
        confidence_entity = f"parameter_confidence_{volcano_id}"
        await self.knowledge_graph.add_entity(
            confidence_entity,
            EntityType.MEASUREMENT,
            parameter_insights
        )
        
        # Connect confidence to reservoir
        await self.knowledge_graph.add_relationship(
            confidence_entity,
            reservoir_entity,
            RelationType.MEASURED_BY,
            confidence=parameter_insights.get('overall_confidence', 0.5)
        )
        
        # Generate causal insights
        causal_insights = await self._generate_causal_insights(volcano_id, reservoir_state, enkf_results)
        
        return {
            'knowledge_integration': {
                'volcano_entity': volcano_entity,
                'reservoir_entity': reservoir_entity,
                'assimilation_entity': assimilation_entity,
                'parameter_insights': parameter_insights,
                'causal_insights': causal_insights
            },
            'integration_timestamp': datetime.now().isoformat()
        }
    
    async def _extract_parameter_insights(self, reservoir_state: Dict[str, Any], 
                                        enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights about parameter estimation quality"""
        
        insights = {}
        
        # Depth estimate quality
        depth_estimate = reservoir_state.get('depth_estimate', {})
        if depth_estimate:
            depth_uncertainty = depth_estimate.get('std', 0) / max(depth_estimate.get('mean', 1), 1)
            insights['depth_quality'] = {
                'relative_uncertainty': depth_uncertainty,
                'quality_level': 'high' if depth_uncertainty < 0.1 else 'medium' if depth_uncertainty < 0.3 else 'low',
                'mean_depth_km': depth_estimate.get('mean', 0) / 1000
            }
        
        # Volume estimate quality
        volume_estimate = reservoir_state.get('volume_estimate', {})
        if volume_estimate:
            volume_uncertainty = volume_estimate.get('std', 0) / max(volume_estimate.get('mean', 1), 1)
            insights['volume_quality'] = {
                'relative_uncertainty': volume_uncertainty,
                'quality_level': 'high' if volume_uncertainty < 0.2 else 'medium' if volume_uncertainty < 0.5 else 'low',
                'mean_volume_km3': volume_estimate.get('mean', 0) / 1e9
            }
        
        # Pressure change quality
        pressure_estimate = reservoir_state.get('pressure_change', {})
        if pressure_estimate:
            pressure_uncertainty = abs(pressure_estimate.get('std', 0)) / max(abs(pressure_estimate.get('mean', 1)), 1)
            insights['pressure_quality'] = {
                'relative_uncertainty': pressure_uncertainty,
                'quality_level': 'high' if pressure_uncertainty < 0.15 else 'medium' if pressure_uncertainty < 0.4 else 'low',
                'mean_pressure_mpa': pressure_estimate.get('mean', 0) / 1e6
            }
        
        # Overall parameter correlations
        correlations = reservoir_state.get('parameter_correlations', {})
        max_correlation = correlations.get('max_correlation', 0)
        
        insights['parameter_independence'] = {
            'max_correlation': max_correlation,
            'independence_level': 'high' if max_correlation < 0.3 else 'medium' if max_correlation < 0.6 else 'low',
            'correlation_broken': max_correlation < 0.5  # Indicates successful reformulation
        }
        
        # Assimilation effectiveness
        update_metrics = enkf_results.get('update_metrics', {})
        spread_reduction = update_metrics.get('spread_reduction', 0)
        
        insights['assimilation_effectiveness'] = {
            'spread_reduction': spread_reduction,
            'effectiveness_level': 'high' if spread_reduction > 0.3 else 'medium' if spread_reduction > 0.1 else 'low',
            'information_gain': spread_reduction > 0.05  # Indicates meaningful update
        }
        
        # Overall confidence assessment
        quality_scores = []
        for param_quality in ['depth_quality', 'volume_quality', 'pressure_quality']:
            if param_quality in insights:
                quality_level = insights[param_quality]['quality_level']
                score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}[quality_level]
                quality_scores.append(score)
        
        overall_confidence = np.mean(quality_scores) if quality_scores else 0.5
        
        # Boost confidence if parameter independence is high
        if insights.get('parameter_independence', {}).get('independence_level') == 'high':
            overall_confidence = min(1.0, overall_confidence + 0.1)
        
        # Boost confidence if assimilation was effective
        if insights.get('assimilation_effectiveness', {}).get('effectiveness_level') in ['high', 'medium']:
            overall_confidence = min(1.0, overall_confidence + 0.05)
        
        insights['overall_confidence'] = overall_confidence
        insights['confidence_level'] = 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.6 else 'low'
        
        return insights
    
    async def _generate_causal_insights(self, volcano_id: str, reservoir_state: Dict[str, Any],
                                      enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate causal insights about volcanic processes"""
        
        causal_insights = {}
        
        # Analyze depth-volume relationships
        depth_estimate = reservoir_state.get('depth_estimate', {})
        volume_estimate = reservoir_state.get('volume_estimate', {})
        
        if depth_estimate and volume_estimate:
            depth_mean = depth_estimate.get('mean', 0)
            volume_mean = volume_estimate.get('mean', 0)
            
            # Shallow vs deep reservoir implications
            if depth_mean < 3000:  # Shallow (<3km)
                causal_insights['reservoir_type'] = {
                    'classification': 'shallow',
                    'implications': ['high_eruption_potential', 'fast_response_time', 'high_hazard'],
                    'confidence': 0.8
                }
            elif depth_mean > 8000:  # Deep (>8km)
                causal_insights['reservoir_type'] = {
                    'classification': 'deep',
                    'implications': ['slow_response_time', 'potential_differentiation', 'long_term_evolution'],
                    'confidence': 0.7
                }
            else:  # Intermediate depth
                causal_insights['reservoir_type'] = {
                    'classification': 'intermediate',
                    'implications': ['moderate_response_time', 'mixed_processes'],
                    'confidence': 0.6
                }
        
        # Pressure change implications
        pressure_estimate = reservoir_state.get('pressure_change', {})
        if pressure_estimate:
            pressure_mean = pressure_estimate.get('mean', 0)
            
            if pressure_mean > 0:  # Inflation
                causal_insights['deformation_mode'] = {
                    'type': 'inflation',
                    'process': 'magma_injection',
                    'implications': ['increased_eruption_risk', 'ongoing_unrest'],
                    'confidence': 0.85
                }
            elif pressure_mean < 0:  # Deflation
                causal_insights['deformation_mode'] = {
                    'type': 'deflation',
                    'process': 'magma_withdrawal',
                    'implications': ['eruption_aftermath', 'system_depressurization'],
                    'confidence': 0.80
                }
        
        # Parameter correlation insights
        correlations = reservoir_state.get('parameter_correlations', {})
        max_correlation = correlations.get('max_correlation', 0)
        
        if max_correlation < 0.3:
            causal_insights['parameter_resolution'] = {
                'quality': 'excellent',
                'indication': 'well_constrained_parameters',
                'reliability': 'high',
                'confidence': 0.9
            }
        elif max_correlation < 0.6:
            causal_insights['parameter_resolution'] = {
                'quality': 'good',
                'indication': 'moderately_constrained_parameters',
                'reliability': 'medium',
                'confidence': 0.7
            }
        else:
            causal_insights['parameter_resolution'] = {
                'quality': 'poor',
                'indication': 'poorly_constrained_parameters',
                'reliability': 'low',
                'confidence': 0.4,
                'recommendation': 'additional_data_needed'
            }
        
        # Assimilation success patterns
        update_metrics = enkf_results.get('update_metrics', {})
        spread_reduction = update_metrics.get('spread_reduction', 0)
        
        if spread_reduction > 0.3:
            causal_insights['data_information_content'] = {
                'level': 'high',
                'indication': 'strong_observational_constraints',
                'implication': 'reliable_parameter_updates',
                'confidence': 0.85
            }
        elif spread_reduction > 0.1:
            causal_insights['data_information_content'] = {
                'level': 'medium',
                'indication': 'moderate_observational_constraints',
                'implication': 'some_parameter_refinement',
                'confidence': 0.6
            }
        else:
            causal_insights['data_information_content'] = {
                'level': 'low',
                'indication': 'weak_observational_constraints',
                'implication': 'limited_parameter_updates',
                'confidence': 0.3,
                'recommendation': 'improve_observation_network'
            }
        
        # Generate process chain insights
        causal_insights['process_chain'] = await self._infer_process_chain(
            reservoir_state, enkf_results
        )
        
        return causal_insights
    
    async def _infer_process_chain(self, reservoir_state: Dict[str, Any], 
                                 enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Infer causal process chain from EnKF results"""
        
        process_chain = {
            'primary_process': None,
            'secondary_processes': [],
            'confidence': 0.0,
            'causal_links': []
        }
        
        # Extract key parameters
        depth_estimate = reservoir_state.get('depth_estimate', {})
        volume_estimate = reservoir_state.get('volume_estimate', {})
        pressure_estimate = reservoir_state.get('pressure_change', {})
        
        depth_mean = depth_estimate.get('mean', 0)
        volume_mean = volume_estimate.get('mean', 0)
        pressure_mean = pressure_estimate.get('mean', 0)
        
        # Infer primary process
        if pressure_mean > 1e6 and volume_mean > 1e6:  # High pressure, significant volume
            process_chain['primary_process'] = {
                'name': 'magma_intrusion',
                'description': 'Fresh magma injection into reservoir',
                'indicators': ['pressure_increase', 'volume_expansion'],
                'confidence': 0.8
            }
            process_chain['causal_links'].append({
                'cause': 'deep_magma_ascent',
                'effect': 'reservoir_pressurization',
                'strength': 0.85
            })
            
        elif pressure_mean < -5e5:  # Significant deflation
            process_chain['primary_process'] = {
                'name': 'magma_withdrawal',
                'description': 'Magma extraction from reservoir',
                'indicators': ['pressure_decrease', 'deflation'],
                'confidence': 0.75
            }
            process_chain['causal_links'].append({
                'cause': 'magma_eruption_or_lateral_intrusion',
                'effect': 'reservoir_depressurization',
                'strength': 0.8
            })
        
        # Secondary processes based on depth
        if depth_mean < 5000:  # Shallow
            process_chain['secondary_processes'].append({
                'name': 'volatile_exsolution',
                'description': 'Gas separation due to shallow pressures',
                'confidence': 0.7
            })
            
        if depth_mean > 8000:  # Deep
            process_chain['secondary_processes'].append({
                'name': 'crystal_fractionation',
                'description': 'Magma differentiation at depth',
                'confidence': 0.6
            })
        
        # Assess overall confidence
        if process_chain['primary_process']:
            base_confidence = process_chain['primary_process']['confidence']
            # Boost confidence if parameters are well constrained
            correlations = reservoir_state.get('parameter_correlations', {})
            max_correlation = correlations.get('max_correlation', 1.0)
            if max_correlation < 0.5:
                base_confidence = min(1.0, base_confidence + 0.1)
            
            process_chain['confidence'] = base_confidence
        
        return process_chain
    
    async def query_related_knowledge(self, volcano_id: str, query_type: str = 'similar_reservoirs') -> Dict[str, Any]:
        """Query knowledge graph for related volcanic systems and insights"""
        
        if query_type == 'similar_reservoirs':
            # Find volcanoes with similar reservoir characteristics
            similar_systems = await self._find_similar_reservoir_systems(volcano_id)
            
        elif query_type == 'historical_patterns':
            # Find historical deformation patterns
            similar_systems = await self._find_historical_deformation_patterns(volcano_id)
            
        elif query_type == 'causal_relationships':
            # Find causal relationships involving this volcano
            similar_systems = await self._find_causal_relationships(volcano_id)
            
        else:
            similar_systems = {}
        
        return {
            'query_type': query_type,
            'volcano_id': volcano_id,
            'results': similar_systems,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _find_similar_reservoir_systems(self, volcano_id: str) -> Dict[str, Any]:
        """Find volcanoes with similar reservoir characteristics"""
        
        # This would query the knowledge graph for similar reservoir properties
        # Simplified implementation for demonstration
        
        current_volcano_entity = f"volcano_{volcano_id}"
        
        # Get current volcano's reservoir characteristics
        try:
            volcano_data = await self.knowledge_graph.get_entity(current_volcano_entity)
            
            if not volcano_data:
                return {'similar_volcanoes': [], 'similarity_criteria': []}
            
            # Extract reservoir parameters for comparison
            reservoir_criteria = {}
            
            # This would implement similarity matching based on reservoir parameters
            # For now, return placeholder structure
            
            return {
                'similar_volcanoes': [],  # Would contain similar volcano IDs
                'similarity_criteria': ['depth_range', 'volume_range', 'pressure_regime'],
                'matching_algorithm': 'parameter_space_distance'
            }
            
        except Exception as e:
            self.logger.error(f"Error finding similar reservoir systems: {e}")
            return {'error': str(e)}
    
    async def _find_historical_deformation_patterns(self, volcano_id: str) -> Dict[str, Any]:
        """Find historical deformation patterns for comparison"""
        
        # This would analyze historical deformation data patterns
        # Placeholder implementation
        
        return {
            'historical_events': [],  # Would contain historical deformation events
            'pattern_types': ['inflation_cycles', 'deflation_events', 'long_term_trends'],
            'comparison_metrics': ['rate_of_change', 'spatial_pattern', 'temporal_evolution']
        }
    
    async def _find_causal_relationships(self, volcano_id: str) -> Dict[str, Any]:
        """Find causal relationships involving this volcano"""
        
        current_volcano_entity = f"volcano_{volcano_id}"
        
        try:
            # Get relationships from knowledge graph
            relationships = await self.knowledge_graph.get_entity_relationships(current_volcano_entity)
            
            causal_links = []
            for rel in relationships:
                if rel.get('type') in [RelationType.CAUSES, RelationType.INFLUENCES]:
                    causal_links.append({
                        'source': rel.get('source'),
                        'target': rel.get('target'),
                        'relationship': rel.get('type'),
                        'confidence': rel.get('confidence', 0.5)
                    })
            
            return {
                'causal_links': causal_links,
                'total_relationships': len(relationships),
                'causal_relationships': len(causal_links)
            }
            
        except Exception as e:
            self.logger.error(f"Error finding causal relationships: {e}")
            return {'error': str(e)}

    async def generate_knowledge_summary(self, volcano_id: str) -> Dict[str, Any]:
        """Generate comprehensive knowledge summary for volcano"""
        
        summary = {
            'volcano_id': volcano_id,
            'knowledge_entities': {},
            'causal_insights': {},
            'parameter_quality': {},
            'process_understanding': {},
            'recommendations': []
        }
        
        try:
            # Get all volcano-related entities
            volcano_entity = f"volcano_{volcano_id}"
            volcano_data = await self.knowledge_graph.get_entity(volcano_entity)
            
            if volcano_data:
                summary['knowledge_entities']['volcano'] = volcano_data
                
                # Get related entities
                relationships = await self.knowledge_graph.get_entity_relationships(volcano_entity)
                summary['knowledge_entities']['relationships'] = relationships
            
            # Generate recommendations based on knowledge
            summary['recommendations'] = await self._generate_recommendations(volcano_id)
            
        except Exception as e:
            self.logger.error(f"Error generating knowledge summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    async def _generate_recommendations(self, volcano_id: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on accumulated knowledge"""
        
        recommendations = []
        
        # Get latest reservoir state and EnKF results from knowledge graph
        try:
            volcano_entity = f"volcano_{volcano_id}"
            relationships = await self.knowledge_graph.get_entity_relationships(volcano_entity)
            
            # Analyze parameter quality and suggest improvements
            for rel in relationships:
                if 'parameter_confidence' in rel.get('target', ''):
                    confidence_data = await self.knowledge_graph.get_entity(rel['target'])
                    
                    if confidence_data:
                        overall_confidence = confidence_data.get('overall_confidence', 0.5)
                        
                        if overall_confidence < 0.6:
                            recommendations.append({
                                'type': 'data_collection',
                                'priority': 'high',
                                'description': 'Improve observation network to enhance parameter constraints',
                                'specific_actions': [
                                    'Add GPS stations in near-field',
                                    'Increase InSAR temporal sampling',
                                    'Deploy tilt meters for higher sensitivity'
                                ]
                            })
                        
                        # Check parameter correlations
                        max_correlation = confidence_data.get('parameter_correlations', {}).get('max_correlation', 0)
                        if max_correlation > 0.6:
                            recommendations.append({
                                'type': 'methodology',
                                'priority': 'medium', 
                                'description': 'Address parameter correlation issues',
                                'specific_actions': [
                                    'Consider alternative observation types',
                                    'Implement advanced regularization',
                                    'Use longer observation time series'
                                ]
                            })
            
            # Add general monitoring recommendations
            recommendations.append({
                'type': 'monitoring',
                'priority': 'medium',
                'description': 'Maintain continuous EnKF assimilation',
                'specific_actions': [
                    'Run daily parameter updates',
                    'Monitor ensemble spread evolution', 
                    'Track parameter correlation changes'
                ]
            })
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                'type': 'system',
                'priority': 'high',
                'description': f'Address knowledge integration error: {e}'
            })
        
        return recommendations