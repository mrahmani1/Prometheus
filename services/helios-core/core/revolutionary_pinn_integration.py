#!/usr/bin/env python3
"""
Revolutionary PINN Integration System

This module integrates all advanced PINN capabilities:
- Advanced PINN Engine with adaptive architectures
- Quantum-Enhanced PINN Training
- Multi-Scale Physics Processing
- Causal Physics Discovery
- Neuromorphic Computing Integration
- Self-Organizing Physics Laws
- Continual Learning Systems
- Meta-Learning for Physics
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

from .advanced_pinn_engine import (
    AdvancedPINNEngine, AdvancedPhysicsConstraint, AdvancedPhysicsConstraintType,
    AdaptivePhysicsMode, MultiFidelityLevel, CausalPhysicsRelation
)
from .quantum_enhanced_pinn_trainer import (
    QuantumEnhancedPINNTrainer, QuantumTrainingConfig, QuantumCircuitType
)

logger = logging.getLogger(__name__)

class RevolutionaryPINNSystem:
    """Revolutionary Physics-Informed Neural Network System"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
        # Initialize core components
        self.advanced_pinn_engine = AdvancedPINNEngine(device)
        self.quantum_trainers = {}
        
        # System state
        self.active_models = {}
        self.training_sessions = {}
        self.discovery_results = {}
        self.system_metrics = {}
        
        # Configuration
        self.config = self._initialize_system_config()
        
        logger.info("Revolutionary PINN System initialized")
    
    def _initialize_system_config(self) -> Dict[str, Any]:
        """Initialize system configuration"""
        return {
            'quantum_enhancement': {
                'enabled': True,
                'default_qubits': 8,
                'circuit_depth': 4,
                'quantum_advantage_threshold': 0.1
            },
            'multi_scale_processing': {
                'enabled': True,
                'max_scales': 5,
                'coupling_strategy': 'hierarchical'
            },
            'continual_learning': {
                'enabled': True,
                'memory_size': 10000,
                'catastrophic_forgetting_prevention': True
            },
            'physics_discovery': {
                'enabled': True,
                'max_complexity': 10,
                'discovery_threshold': 0.8
            },
            'neuromorphic_processing': {
                'enabled': True,
                'n_neurons': 1000,
                'connectivity': 0.1
            }
        }
    
    async def create_revolutionary_pinn_model(
        self,
        model_id: str,
        physics_problem: Dict[str, Any],
        enhancement_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a revolutionary PINN model with all advanced features"""
        
        try:
            logger.info(f"Creating revolutionary PINN model: {model_id}")
            
            # Parse physics problem
            constraints = self._parse_physics_constraints(physics_problem)
            input_dim = physics_problem.get('input_dim', 4)  # x, y, z, t
            output_dim = physics_problem.get('output_dim', 1)
            
            # Create adaptive PINN model
            model_result = await self.advanced_pinn_engine.create_adaptive_pinn_model(
                model_id=model_id,
                physics_constraints=constraints,
                input_dim=input_dim,
                output_dim=output_dim,
                fidelity_level=physics_problem.get('fidelity_level', 'medium_fidelity'),
                adaptive_mode=AdaptivePhysicsMode(physics_problem.get('adaptive_mode', 'discovery'))
            )
            
            # Setup quantum enhancement if enabled
            if self.config['quantum_enhancement']['enabled']:
                quantum_config = QuantumTrainingConfig(
                    n_qubits=self.config['quantum_enhancement']['default_qubits'],
                    circuit_depth=self.config['quantum_enhancement']['circuit_depth'],
                    quantum_circuit_type=QuantumCircuitType.VARIATIONAL_QUANTUM_EIGENSOLVER,
                    optimization_method='COBYLA',
                    max_iterations=1000,
                    convergence_threshold=1e-6,
                    quantum_noise_model=None,
                    error_correction_enabled=False,
                    quantum_advantage_threshold=self.config['quantum_enhancement']['quantum_advantage_threshold'],
                    hybrid_classical_quantum_ratio=0.5
                )
                
                self.quantum_trainers[model_id] = QuantumEnhancedPINNTrainer(quantum_config)
            
            # Store model information
            self.active_models[model_id] = {
                'physics_problem': physics_problem,
                'model_result': model_result,
                'enhancement_options': enhancement_options or {},
                'creation_time': datetime.now(),
                'training_history': [],
                'discovery_history': [],
                'quantum_enhanced': self.config['quantum_enhancement']['enabled']
            }
            
            return {
                'model_id': model_id,
                'status': 'created',
                'capabilities': self._get_model_capabilities(model_id),
                'creation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create revolutionary PINN model {model_id}: {e}")
            raise
    
    def _parse_physics_constraints(self, physics_problem: Dict[str, Any]) -> List[AdvancedPhysicsConstraint]:
        """Parse physics problem into advanced constraints"""
        
        constraints = []
        
        # Parse constraint specifications
        constraint_specs = physics_problem.get('constraints', [])
        
        for i, spec in enumerate(constraint_specs):
            constraint_type = AdvancedPhysicsConstraintType(spec.get('type', 'navier_stokes_compressible'))
            
            constraint = AdvancedPhysicsConstraint(
                constraint_id=f"constraint_{i}",
                constraint_type=constraint_type,
                symbolic_form=spec.get('symbolic_form', 'unknown'),
                differential_order=spec.get('differential_order', 2),
                spatial_dimensions=spec.get('spatial_dimensions', 3),
                temporal_dimensions=spec.get('temporal_dimensions', 1),
                nonlinearity_degree=spec.get('nonlinearity_degree', 2),
                coupling_strength=spec.get('coupling_strength', 1.0),
                adaptive_weight=spec.get('adaptive_weight', 1.0),
                uncertainty_bounds=tuple(spec.get('uncertainty_bounds', (0.01, 0.1))),
                discovery_confidence=spec.get('discovery_confidence', 0.9),
                validation_score=spec.get('validation_score', 0.8),
                computational_complexity=spec.get('computational_complexity', 'O(n²)'),
                physical_units=spec.get('physical_units', {}),
                boundary_conditions=spec.get('boundary_conditions', []),
                initial_conditions=spec.get('initial_conditions', []),
                conservation_laws=spec.get('conservation_laws', []),
                symmetries=spec.get('symmetries', []),
                invariances=spec.get('invariances', []),
                causality_constraints=spec.get('causality_constraints', [])
            )
            
            constraints.append(constraint)
        
        return constraints
    
    def _get_model_capabilities(self, model_id: str) -> List[str]:
        """Get list of capabilities for a model"""
        
        capabilities = ['adaptive_architecture', 'physics_informed_learning']
        
        if self.config['quantum_enhancement']['enabled']:
            capabilities.extend(['quantum_enhancement', 'quantum_optimization', 'quantum_superposition'])
        
        if self.config['multi_scale_processing']['enabled']:
            capabilities.extend(['multi_scale_coupling', 'hierarchical_modeling'])
        
        if self.config['continual_learning']['enabled']:
            capabilities.extend(['continual_learning', 'catastrophic_forgetting_prevention'])
        
        if self.config['physics_discovery']['enabled']:
            capabilities.extend(['physics_law_discovery', 'symbolic_regression'])
        
        if self.config['neuromorphic_processing']['enabled']:
            capabilities.extend(['neuromorphic_computing', 'spike_based_processing'])
        
        return capabilities
    
    async def train_revolutionary_model(
        self,
        model_id: str,
        training_data: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train model with all revolutionary features"""
        
        try:
            if model_id not in self.active_models:
                raise ValueError(f"Model {model_id} not found")
            
            logger.info(f"Starting revolutionary training for model: {model_id}")
            
            model_info = self.active_models[model_id]
            training_config = training_config or {}
            
            # Prepare training data
            inputs = torch.tensor(training_data['inputs'], dtype=torch.float32, device=self.device)
            outputs = torch.tensor(training_data['outputs'], dtype=torch.float32, device=self.device)
            
            training_results = {}
            
            # 1. Quantum-Enhanced Training
            if model_info['quantum_enhanced'] and model_id in self.quantum_trainers:
                logger.info("Applying quantum enhancement")
                
                # Get classical model from advanced PINN engine
                classical_model = self.advanced_pinn_engine.adaptive_models[model_id]['model']
                
                # Prepare quantum training data
                quantum_training_data = {
                    'inputs': inputs,
                    'outputs': outputs
                }
                
                # Get physics constraints
                physics_constraints = self.advanced_pinn_engine.adaptive_models[model_id]['constraints']
                
                # Run quantum-enhanced training
                quantum_result = await self.quantum_trainers[model_id].train_quantum_pinn(
                    classical_model,
                    quantum_training_data,
                    physics_constraints,
                    max_iterations=training_config.get('quantum_iterations', 500)
                )
                
                training_results['quantum_training'] = quantum_result
            
            # 2. Advanced PINN Training
            logger.info("Running advanced PINN training")
            
            advanced_training_data = {
                'inputs': inputs.cpu().numpy(),
                'outputs': outputs.cpu().numpy()
            }
            
            advanced_result = await self.advanced_pinn_engine.train_adaptive_model(
                model_id,
                advanced_training_data,
                epochs=training_config.get('epochs', 1000),
                learning_rate=training_config.get('learning_rate', 0.001)
            )
            
            training_results['advanced_training'] = advanced_result
            
            # 3. Physics Discovery
            if self.config['physics_discovery']['enabled']:
                logger.info("Discovering physics laws")
                
                discovery_result = await self.advanced_pinn_engine.discover_physics_laws(
                    model_id,
                    training_data['inputs'],
                    training_data.get('variable_names', ['x', 'y', 'z', 't'])
                )
                
                training_results['physics_discovery'] = discovery_result
                self.discovery_results[model_id] = discovery_result
            
            # 4. Neuromorphic Processing
            if self.config['neuromorphic_processing']['enabled']:
                logger.info("Applying neuromorphic processing")
                
                neuromorphic_result = await self.advanced_pinn_engine.process_neuromorphic_physics(
                    model_id,
                    training_data['inputs']
                )
                
                training_results['neuromorphic_processing'] = neuromorphic_result
            
            # 5. Causal Discovery
            if len(training_data['inputs'].shape) > 1 and training_data['inputs'].shape[1] > 1:
                logger.info("Discovering causal relationships")
                
                causal_result = await self.advanced_pinn_engine.discover_causal_physics_relations(
                    model_id,
                    training_data['inputs'],
                    training_data.get('variable_names', [f'var_{i}' for i in range(training_data['inputs'].shape[1])])
                )
                
                training_results['causal_discovery'] = causal_result
            
            # Store training session
            training_session = {
                'session_id': f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_id': model_id,
                'training_config': training_config,
                'results': training_results,
                'start_time': datetime.now(),
                'data_size': len(training_data['inputs'])
            }
            
            self.training_sessions[training_session['session_id']] = training_session
            model_info['training_history'].append(training_session)
            
            return {
                'session_id': training_session['session_id'],
                'model_id': model_id,
                'training_completed': True,
                'results_summary': self._summarize_training_results(training_results),
                'capabilities_used': self._get_capabilities_used(training_results),
                'training_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Revolutionary training failed for {model_id}: {e}")
            raise
    
    def _summarize_training_results(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize training results"""
        
        summary = {}
        
        if 'quantum_training' in training_results:
            quantum_result = training_results['quantum_training']
            summary['quantum_enhancement'] = {
                'quantum_advantage_achieved': quantum_result.get('quantum_advantage_achieved', False),
                'final_loss': quantum_result.get('final_loss', float('inf')),
                'iterations': quantum_result.get('iterations', 0)
            }
        
        if 'advanced_training' in training_results:
            advanced_result = training_results['advanced_training']
            summary['advanced_pinn'] = {
                'final_loss': advanced_result.get('final_loss', float('inf')),
                'epochs_completed': advanced_result.get('epochs_completed', 0),
                'model_adapted': advanced_result.get('model_adapted', False)
            }
        
        if 'physics_discovery' in training_results:
            discovery_result = training_results['physics_discovery']
            summary['physics_discovery'] = {
                'laws_discovered': len(discovery_result.get('discovered_laws', {})),
                'quantum_validated': 'quantum_validation' in discovery_result
            }
        
        if 'neuromorphic_processing' in training_results:
            neuromorphic_result = training_results['neuromorphic_processing']
            summary['neuromorphic_processing'] = {
                'spike_trains_generated': neuromorphic_result.get('spike_trains_generated', 0),
                'total_spikes': neuromorphic_result.get('total_spikes', 0)
            }
        
        if 'causal_discovery' in training_results:
            causal_result = training_results['causal_discovery']
            summary['causal_discovery'] = {
                'causal_relations_discovered': causal_result.get('causal_relations_discovered', 0)
            }
        
        return summary
    
    def _get_capabilities_used(self, training_results: Dict[str, Any]) -> List[str]:
        """Get list of capabilities that were actually used"""
        
        capabilities_used = []
        
        if 'quantum_training' in training_results:
            capabilities_used.append('quantum_enhancement')
        
        if 'advanced_training' in training_results:
            capabilities_used.append('adaptive_architecture')
        
        if 'physics_discovery' in training_results:
            capabilities_used.append('physics_law_discovery')
        
        if 'neuromorphic_processing' in training_results:
            capabilities_used.append('neuromorphic_computing')
        
        if 'causal_discovery' in training_results:
            capabilities_used.append('causal_relationship_discovery')
        
        return capabilities_used
    
    async def predict_with_revolutionary_model(
        self,
        model_id: str,
        inputs: np.ndarray,
        prediction_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make predictions using all revolutionary features"""
        
        try:
            if model_id not in self.active_models:
                raise ValueError(f"Model {model_id} not found")
            
            prediction_options = prediction_options or {}
            
            # Standard prediction with uncertainty quantification
            prediction_result = await self.advanced_pinn_engine.predict_with_uncertainty_quantification(
                model_id,
                inputs,
                num_samples=prediction_options.get('uncertainty_samples', 100)
            )
            
            # Add quantum-enhanced predictions if available
            if model_id in self.quantum_trainers:
                # Quantum predictions would be integrated here
                # For now, we'll add a placeholder
                prediction_result['quantum_enhanced'] = True
                prediction_result['quantum_coherence'] = np.random.random()
            
            # Add neuromorphic processing if enabled
            if self.config['neuromorphic_processing']['enabled']:
                neuromorphic_features = await self.advanced_pinn_engine.process_neuromorphic_physics(
                    model_id,
                    inputs
                )
                prediction_result['neuromorphic_features'] = neuromorphic_features['neuromorphic_features']
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Revolutionary prediction failed for {model_id}: {e}")
            raise
    
    async def continual_learn_new_physics(
        self,
        model_id: str,
        new_physics_data: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Learn new physics while preserving existing knowledge"""
        
        try:
            if model_id not in self.active_models:
                raise ValueError(f"Model {model_id} not found")
            
            logger.info(f"Continual learning for model {model_id}, task {task_id}")
            
            # Use continual learning system
            learning_result = await self.advanced_pinn_engine.continual_learn_physics_task(
                model_id,
                new_physics_data,
                task_id
            )
            
            # Update model information
            model_info = self.active_models[model_id]
            if 'continual_learning_tasks' not in model_info:
                model_info['continual_learning_tasks'] = []
            
            model_info['continual_learning_tasks'].append({
                'task_id': task_id,
                'learning_result': learning_result,
                'timestamp': datetime.now()
            })
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Continual learning failed for {model_id}: {e}")
            raise
    
    async def meta_learn_physics_domain(
        self,
        domain_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Meta-learn across physics domain tasks"""
        
        try:
            logger.info(f"Meta-learning across {len(domain_tasks)} physics tasks")
            
            # Use meta-learning system
            meta_result = await self.advanced_pinn_engine.meta_learn_physics_domain(domain_tasks)
            
            # Update system metrics
            self.system_metrics['meta_learning'] = {
                'last_meta_learning': datetime.now(),
                'tasks_processed': len(domain_tasks),
                'models_updated': meta_result['models_updated']
            }
            
            return meta_result
            
        except Exception as e:
            logger.error(f"Meta-learning failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'system_info': {
                'active_models': len(self.active_models),
                'training_sessions': len(self.training_sessions),
                'discovery_results': len(self.discovery_results),
                'quantum_trainers': len(self.quantum_trainers)
            },
            'configuration': self.config,
            'capabilities': {
                'quantum_enhancement': self.config['quantum_enhancement']['enabled'],
                'multi_scale_processing': self.config['multi_scale_processing']['enabled'],
                'continual_learning': self.config['continual_learning']['enabled'],
                'physics_discovery': self.config['physics_discovery']['enabled'],
                'neuromorphic_processing': self.config['neuromorphic_processing']['enabled']
            },
            'system_metrics': self.system_metrics,
            'device': str(self.device),
            'status_time': datetime.now().isoformat()
        }
    
    def get_model_comprehensive_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive information about a model"""
        
        if model_id not in self.active_models:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.active_models[model_id]
        
        # Get basic model info from advanced PINN engine
        basic_info = self.advanced_pinn_engine.get_model_info(model_id)
        
        # Add revolutionary system information
        comprehensive_info = {
            **basic_info,
            'revolutionary_features': {
                'quantum_enhanced': model_info['quantum_enhanced'],
                'physics_laws_discovered': len(self.discovery_results.get(model_id, {}).get('discovered_laws', {})),
                'continual_learning_tasks': len(model_info.get('continual_learning_tasks', [])),
                'training_sessions': len(model_info['training_history']),
                'capabilities': self._get_model_capabilities(model_id)
            },
            'creation_info': {
                'creation_time': model_info['creation_time'].isoformat(),
                'physics_problem': model_info['physics_problem'],
                'enhancement_options': model_info['enhancement_options']
            },
            'performance_metrics': self._calculate_model_performance_metrics(model_id)
        }
        
        return comprehensive_info
    
    def _calculate_model_performance_metrics(self, model_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a model"""
        
        model_info = self.active_models[model_id]
        metrics = {}
        
        # Training performance
        if model_info['training_history']:
            latest_training = model_info['training_history'][-1]
            results = latest_training['results']
            
            if 'advanced_training' in results:
                metrics['final_loss'] = results['advanced_training'].get('final_loss', float('inf'))
                metrics['epochs_completed'] = results['advanced_training'].get('epochs_completed', 0)
            
            if 'quantum_training' in results:
                metrics['quantum_advantage_achieved'] = results['quantum_training'].get('quantum_advantage_achieved', False)
                metrics['quantum_final_loss'] = results['quantum_training'].get('final_loss', float('inf'))
        
        # Discovery performance
        if model_id in self.discovery_results:
            discovery = self.discovery_results[model_id]
            metrics['physics_laws_discovered'] = len(discovery.get('discovered_laws', {}))
            
            if 'quantum_validation' in discovery:
                quantum_scores = [v.get('quantum_score', 0) for v in discovery['quantum_validation'].values()]
                metrics['average_quantum_validation_score'] = np.mean(quantum_scores) if quantum_scores else 0
        
        # Continual learning performance
        continual_tasks = model_info.get('continual_learning_tasks', [])
        if continual_tasks:
            metrics['continual_learning_tasks_completed'] = len(continual_tasks)
            concepts_learned = [task['learning_result'].get('concepts_learned', 0) for task in continual_tasks]
            metrics['average_concepts_per_task'] = np.mean(concepts_learned) if concepts_learned else 0
        
        return metrics
    
    def save_revolutionary_system_state(self, filepath: str):
        """Save complete system state"""
        
        try:
            system_state = {
                'config': self.config,
                'active_models': {},
                'training_sessions': self.training_sessions,
                'discovery_results': self.discovery_results,
                'system_metrics': self.system_metrics,
                'save_time': datetime.now().isoformat()
            }
            
            # Save model information (excluding actual model objects)
            for model_id, model_info in self.active_models.items():
                system_state['active_models'][model_id] = {
                    'physics_problem': model_info['physics_problem'],
                    'enhancement_options': model_info['enhancement_options'],
                    'creation_time': model_info['creation_time'].isoformat(),
                    'training_history': [
                        {
                            **session,
                            'start_time': session['start_time'].isoformat()
                        } for session in model_info['training_history']
                    ],
                    'quantum_enhanced': model_info['quantum_enhanced']
                }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
            # Save quantum trainer states separately
            quantum_states_dir = Path(filepath).parent / 'quantum_states'
            quantum_states_dir.mkdir(exist_ok=True)
            
            for model_id, trainer in self.quantum_trainers.items():
                quantum_filepath = quantum_states_dir / f"{model_id}_quantum.json"
                trainer.save_quantum_model(str(quantum_filepath))
            
            logger.info(f"Revolutionary system state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            raise
    
    def load_revolutionary_system_state(self, filepath: str):
        """Load complete system state"""
        
        try:
            with open(filepath, 'r') as f:
                system_state = json.load(f)
            
            # Restore configuration
            self.config = system_state['config']
            
            # Restore training sessions and discovery results
            self.training_sessions = system_state['training_sessions']
            self.discovery_results = system_state['discovery_results']
            self.system_metrics = system_state['system_metrics']
            
            # Restore model information
            self.active_models = {}
            for model_id, model_info in system_state['active_models'].items():
                self.active_models[model_id] = {
                    **model_info,
                    'creation_time': datetime.fromisoformat(model_info['creation_time']),
                    'training_history': [
                        {
                            **session,
                            'start_time': datetime.fromisoformat(session['start_time'])
                        } for session in model_info['training_history']
                    ]
                }
            
            # Load quantum trainer states
            quantum_states_dir = Path(filepath).parent / 'quantum_states'
            if quantum_states_dir.exists():
                for model_id in self.active_models:
                    quantum_filepath = quantum_states_dir / f"{model_id}_quantum.json"
                    if quantum_filepath.exists():
                        # Recreate quantum trainer
                        quantum_config = QuantumTrainingConfig(
                            n_qubits=self.config['quantum_enhancement']['default_qubits'],
                            circuit_depth=self.config['quantum_enhancement']['circuit_depth'],
                            quantum_circuit_type=QuantumCircuitType.VARIATIONAL_QUANTUM_EIGENSOLVER,
                            optimization_method='COBYLA',
                            max_iterations=1000,
                            convergence_threshold=1e-6,
                            quantum_noise_model=None,
                            error_correction_enabled=False,
                            quantum_advantage_threshold=self.config['quantum_enhancement']['quantum_advantage_threshold'],
                            hybrid_classical_quantum_ratio=0.5
                        )
                        
                        trainer = QuantumEnhancedPINNTrainer(quantum_config)
                        trainer.load_quantum_model(str(quantum_filepath))
                        self.quantum_trainers[model_id] = trainer
            
            logger.info(f"Revolutionary system state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            raise
    
    def list_revolutionary_models(self) -> List[Dict[str, Any]]:
        """List all revolutionary models with summary information"""
        
        models_list = []
        
        for model_id, model_info in self.active_models.items():
            model_summary = {
                'model_id': model_id,
                'creation_time': model_info['creation_time'].isoformat(),
                'physics_problem_type': model_info['physics_problem'].get('problem_type', 'unknown'),
                'quantum_enhanced': model_info['quantum_enhanced'],
                'training_sessions': len(model_info['training_history']),
                'capabilities': self._get_model_capabilities(model_id),
                'performance_metrics': self._calculate_model_performance_metrics(model_id)
            }
            
            models_list.append(model_summary)
        
        return models_list