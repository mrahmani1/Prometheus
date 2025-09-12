"""
Continuous Validation Framework
==============================

Provides continuous validation against new geological discoveries and research.
Implements automated validation pipelines and research integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import asyncio
import aiohttp
import schedule
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Import all validation modules
from .benchmark_datasets import BenchmarkDatasetManager
from .geological_accuracy_validator import GeologicalAccuracyValidator
from .seismic_validation import SeismicValidationSuite
from .volcanic_validation import VolcanicValidationSuite
from .geochemical_validation import GeochemicalValidationSuite
from .visualization_validation import VisualizationValidationSuite
from .data_integrity_validator import DataIntegrityValidator

logger = logging.getLogger(__name__)

@dataclass
class ValidationSchedule:
    """Validation schedule configuration"""
    validation_type: str
    frequency: str  # 'daily', 'weekly', 'monthly'
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

@dataclass
class ResearchUpdate:
    """New research update for validation"""
    paper_id: str
    title: str
    authors: List[str]
    publication_date: datetime
    doi: str
    abstract: str
    geological_domain: str
    validation_impact: str
    new_datasets: List[str] = field(default_factory=list)

class ContinuousValidationFramework:
    """
    Continuous validation framework for geological algorithms.
    
    Features:
    - Automated validation scheduling
    - Integration with research databases
    - Real-time validation monitoring
    - Automated report generation
    - Performance trend analysis
    - Research impact assessment
    """
    
    def __init__(self, benchmark_manager: BenchmarkDatasetManager):
        self.benchmark_manager = benchmark_manager
        
        # Initialize validation suites
        self.accuracy_validator = GeologicalAccuracyValidator(benchmark_manager)
        self.seismic_validator = SeismicValidationSuite(benchmark_manager)
        self.volcanic_validator = VolcanicValidationSuite(benchmark_manager)
        self.geochemical_validator = GeochemicalValidationSuite(benchmark_manager)
        self.visualization_validator = VisualizationValidationSuite(benchmark_manager)
        self.data_integrity_validator = DataIntegrityValidator(benchmark_manager)
        
        # Validation configuration
        self.validation_schedules = self._initialize_validation_schedules()
        self.research_sources = self._initialize_research_sources()
        
        # Results storage
        self.validation_history: List[Dict[str, Any]] = []
        self.research_updates: List[ResearchUpdate] = []
        
        # Monitoring
        self.is_running = False
        self.monitoring_thread = None
        
        # Configuration
        self.config = {
            'validation_timeout_minutes': 30,
            'max_concurrent_validations': 3,
            'research_check_interval_hours': 24,
            'performance_threshold_degradation': 0.05,  # 5% degradation threshold
            'auto_report_generation': True,
            'notification_enabled': True
        }
    
    def _initialize_validation_schedules(self) -> List[ValidationSchedule]:
        """Initialize validation schedules"""
        return [
            ValidationSchedule('seismic_phase_picking', 'daily'),
            ValidationSchedule('volcanic_forecasting', 'daily'),
            ValidationSchedule('geochemical_analysis', 'weekly'),
            ValidationSchedule('3d_visualization', 'weekly'),
            ValidationSchedule('data_integrity', 'daily'),
            ValidationSchedule('production_performance', 'daily'),
            ValidationSchedule('cross_dataset_consistency', 'weekly'),
            ValidationSchedule('fault_tolerance', 'monthly')
        ]
    
    def _initialize_research_sources(self) -> Dict[str, str]:
        """Initialize research data sources"""
        return {
            'arxiv_geology': 'http://export.arxiv.org/api/query?search_query=cat:physics.geo-ph',
            'usgs_publications': 'https://pubs.usgs.gov/api/publications',
            'nature_geoscience': 'https://api.nature.com/content/journals/ngeo',
            'jgr_solid_earth': 'https://api.wiley.com/onlinelibrary/tdm/v1/articles',
            'grl_letters': 'https://api.wiley.com/onlinelibrary/tdm/v1/articles'
        }
    
    def start_continuous_validation(self):
        """Start continuous validation monitoring"""
        if self.is_running:
            logger.warning("Continuous validation is already running")
            return
        
        logger.info("Starting continuous validation framework")
        self.is_running = True
        
        # Schedule validation tasks
        self._schedule_validation_tasks()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Continuous validation framework started successfully")
    
    def stop_continuous_validation(self):
        """Stop continuous validation monitoring"""
        logger.info("Stopping continuous validation framework")
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Continuous validation framework stopped")
    
    def _schedule_validation_tasks(self):
        """Schedule validation tasks based on configuration"""
        for schedule_config in self.validation_schedules:
            if not schedule_config.enabled:
                continue
            
            if schedule_config.frequency == 'daily':
                schedule.every().day.at("02:00").do(
                    self._run_scheduled_validation, schedule_config.validation_type
                )
            elif schedule_config.frequency == 'weekly':
                schedule.every().sunday.at("03:00").do(
                    self._run_scheduled_validation, schedule_config.validation_type
                )
            elif schedule_config.frequency == 'monthly':
                schedule.every().month.do(
                    self._run_scheduled_validation, schedule_config.validation_type
                )
        
        # Schedule research updates check
        schedule.every(self.config['research_check_interval_hours']).hours.do(
            self._check_research_updates
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                # Check system health
                self._check_system_health()
                
                # Sleep for a minute before next check
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Continue after error
    
    def _run_scheduled_validation(self, validation_type: str):
        """Run a scheduled validation"""
        logger.info(f"Running scheduled validation: {validation_type}")
        
        try:
            # Update schedule
            for schedule_config in self.validation_schedules:
                if schedule_config.validation_type == validation_type:
                    schedule_config.last_run = datetime.now()
                    break
            
            # Run validation based on type
            if validation_type == 'seismic_phase_picking':
                result = self.seismic_validator.validate_phase_picking_accuracy()
            elif validation_type == 'volcanic_forecasting':
                result = self.volcanic_validator.validate_eruption_forecasting()
            elif validation_type == 'geochemical_analysis':
                result = self.geochemical_validator.validate_major_element_analysis()
            elif validation_type == '3d_visualization':
                result = self.visualization_validator.validate_spatial_coordinate_accuracy()
            elif validation_type == 'data_integrity':
                result = self.data_integrity_validator.validate_data_completeness()
            elif validation_type == 'production_performance':
                result = self.data_integrity_validator.validate_production_scale_performance()
            elif validation_type == 'cross_dataset_consistency':
                result = self.data_integrity_validator.validate_cross_dataset_consistency()
            elif validation_type == 'fault_tolerance':
                result = self.data_integrity_validator.validate_fault_tolerance()
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
                return
            
            # Store result
            validation_record = {
                'timestamp': datetime.now(),
                'validation_type': validation_type,
                'result': result,
                'scheduled': True
            }
            
            self.validation_history.append(validation_record)
            
            # Check for performance degradation
            self._check_performance_degradation(validation_type, result)
            
            # Generate report if needed
            if self.config['auto_report_generation']:
                self._generate_validation_report(validation_record)
            
            logger.info(f"Completed scheduled validation: {validation_type}, Passed: {result.passed}")
            
        except Exception as e:
            logger.error(f"Failed to run scheduled validation {validation_type}: {e}")
    
    async def _check_research_updates(self):
        """Check for new research updates"""
        logger.info("Checking for new research updates")
        
        try:
            new_updates = await self._fetch_research_updates()
            
            for update in new_updates:
                # Check if this research impacts our validation
                impact_assessment = self._assess_research_impact(update)
                
                if impact_assessment['requires_validation_update']:
                    logger.info(f"New research requires validation update: {update.title}")
                    
                    # Add to research updates
                    self.research_updates.append(update)
                    
                    # Trigger relevant validations
                    await self._trigger_research_based_validation(update, impact_assessment)
            
            logger.info(f"Research update check completed. Found {len(new_updates)} new updates")
            
        except Exception as e:
            logger.error(f"Failed to check research updates: {e}")
    
    async def _fetch_research_updates(self) -> List[ResearchUpdate]:
        """Fetch new research from configured sources"""
        updates = []
        
        # This would integrate with real research APIs
        # For now, simulate research updates
        simulated_updates = [
            ResearchUpdate(
                paper_id="arxiv_2024_001",
                title="Improved Seismic Phase Picking Using Deep Learning",
                authors=["Smith, J.", "Johnson, A."],
                publication_date=datetime.now() - timedelta(days=1),
                doi="10.1000/example.2024.001",
                abstract="New deep learning approach for seismic phase picking...",
                geological_domain="seismology",
                validation_impact="high"
            ),
            ResearchUpdate(
                paper_id="nature_2024_002",
                title="Volcanic Eruption Forecasting with Multi-Parameter Analysis",
                authors=["Brown, K.", "Davis, L."],
                publication_date=datetime.now() - timedelta(days=2),
                doi="10.1038/example.2024.002",
                abstract="Novel approach to volcanic eruption forecasting...",
                geological_domain="volcanology",
                validation_impact="medium"
            )
        ]
        
        # Filter for recent updates (last 7 days)
        recent_updates = [
            update for update in simulated_updates
            if (datetime.now() - update.publication_date).days <= 7
        ]
        
        return recent_updates
    
    def _assess_research_impact(self, research_update: ResearchUpdate) -> Dict[str, Any]:
        """Assess the impact of new research on validation requirements"""
        impact_assessment = {
            'requires_validation_update': False,
            'affected_algorithms': [],
            'priority': 'low',
            'estimated_effort': 'minimal'
        }
        
        # Analyze based on geological domain
        if research_update.geological_domain == 'seismology':
            if 'phase picking' in research_update.title.lower():
                impact_assessment['requires_validation_update'] = True
                impact_assessment['affected_algorithms'] = ['SeismicTransformer']
                impact_assessment['priority'] = 'high'
            elif 'magnitude' in research_update.title.lower():
                impact_assessment['requires_validation_update'] = True
                impact_assessment['affected_algorithms'] = ['SeismicTransformer']
                impact_assessment['priority'] = 'medium'
        
        elif research_update.geological_domain == 'volcanology':
            if 'forecasting' in research_update.title.lower() or 'prediction' in research_update.title.lower():
                impact_assessment['requires_validation_update'] = True
                impact_assessment['affected_algorithms'] = ['EruptionForecasterML']
                impact_assessment['priority'] = 'high'
        
        elif research_update.geological_domain == 'geochemistry':
            impact_assessment['requires_validation_update'] = True
            impact_assessment['affected_algorithms'] = ['GeochemicalML']
            impact_assessment['priority'] = 'medium'
        
        # Assess validation impact level
        if research_update.validation_impact == 'high':
            impact_assessment['priority'] = 'high'
            impact_assessment['estimated_effort'] = 'significant'
        elif research_update.validation_impact == 'medium':
            impact_assessment['priority'] = 'medium'
            impact_assessment['estimated_effort'] = 'moderate'
        
        return impact_assessment
    
    async def _trigger_research_based_validation(self, research_update: ResearchUpdate, 
                                               impact_assessment: Dict[str, Any]):
        """Trigger validation based on new research"""
        logger.info(f"Triggering research-based validation for: {research_update.title}")
        
        try:
            # Run relevant validations based on research domain
            if research_update.geological_domain == 'seismology':
                result = self.seismic_validator.validate_phase_picking_accuracy()
                validation_type = 'seismic_research_update'
            elif research_update.geological_domain == 'volcanology':
                result = self.volcanic_validator.validate_eruption_forecasting()
                validation_type = 'volcanic_research_update'
            elif research_update.geological_domain == 'geochemistry':
                result = self.geochemical_validator.validate_major_element_analysis()
                validation_type = 'geochemical_research_update'
            else:
                logger.warning(f"Unknown research domain: {research_update.geological_domain}")
                return
            
            # Store result with research context
            validation_record = {
                'timestamp': datetime.now(),
                'validation_type': validation_type,
                'result': result,
                'scheduled': False,
                'research_trigger': {
                    'paper_id': research_update.paper_id,
                    'title': research_update.title,
                    'impact_assessment': impact_assessment
                }
            }
            
            self.validation_history.append(validation_record)
            
            # Generate research impact report
            self._generate_research_impact_report(validation_record)
            
            logger.info(f"Research-based validation completed: {validation_type}")
            
        except Exception as e:
            logger.error(f"Failed to run research-based validation: {e}")
    
    def _check_performance_degradation(self, validation_type: str, current_result):
        """Check for performance degradation compared to historical results"""
        # Get recent results for this validation type
        recent_results = [
            record for record in self.validation_history[-10:]  # Last 10 results
            if record['validation_type'] == validation_type
        ]
        
        if len(recent_results) < 3:
            return  # Need at least 3 results for trend analysis
        
        # Calculate performance trend
        accuracy_scores = [record['result'].accuracy_score for record in recent_results]
        current_accuracy = current_result.accuracy_score
        
        # Calculate moving average
        historical_average = np.mean(accuracy_scores[:-1])  # Exclude current result
        
        # Check for significant degradation
        degradation = historical_average - current_accuracy
        degradation_percentage = degradation / historical_average if historical_average > 0 else 0
        
        if degradation_percentage > self.config['performance_threshold_degradation']:
            logger.warning(
                f"Performance degradation detected in {validation_type}: "
                f"{degradation_percentage:.2%} decrease in accuracy"
            )
            
            # Generate degradation alert
            self._generate_degradation_alert(validation_type, degradation_percentage, current_result)
    
    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check recent validation success rate
            recent_validations = self.validation_history[-20:]  # Last 20 validations
            if recent_validations:
                success_rate = sum(1 for v in recent_validations if v['result'].passed) / len(recent_validations)
                
                if success_rate < 0.80:  # Less than 80% success rate
                    logger.warning(f"Low validation success rate: {success_rate:.2%}")
            
            # Check for overdue validations
            current_time = datetime.now()
            for schedule_config in self.validation_schedules:
                if schedule_config.enabled and schedule_config.last_run:
                    time_since_last = current_time - schedule_config.last_run
                    
                    if schedule_config.frequency == 'daily' and time_since_last > timedelta(days=2):
                        logger.warning(f"Overdue daily validation: {schedule_config.validation_type}")
                    elif schedule_config.frequency == 'weekly' and time_since_last > timedelta(days=8):
                        logger.warning(f"Overdue weekly validation: {schedule_config.validation_type}")
                    elif schedule_config.frequency == 'monthly' and time_since_last > timedelta(days=32):
                        logger.warning(f"Overdue monthly validation: {schedule_config.validation_type}")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    def _generate_validation_report(self, validation_record: Dict[str, Any]):
        """Generate validation report"""
        report = {
            'report_id': f"VAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': validation_record['timestamp'].isoformat(),
            'validation_type': validation_record['validation_type'],
            'result_summary': {
                'passed': validation_record['result'].passed,
                'accuracy_score': validation_record['result'].accuracy_score,
                'test_name': validation_record['result'].test_name,
                'algorithm': validation_record['result'].algorithm
            },
            'error_metrics': validation_record['result'].error_metrics,
            'notes': validation_record['result'].notes
        }
        
        # Save report
        report_path = Path("tests/geological_validation/reports")
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"{report['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report generated: {report_file}")
    
    def _generate_research_impact_report(self, validation_record: Dict[str, Any]):
        """Generate research impact report"""
        research_info = validation_record.get('research_trigger', {})
        
        report = {
            'report_id': f"RES_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': validation_record['timestamp'].isoformat(),
            'research_paper': {
                'paper_id': research_info.get('paper_id'),
                'title': research_info.get('title'),
                'impact_assessment': research_info.get('impact_assessment')
            },
            'validation_results': {
                'validation_type': validation_record['validation_type'],
                'passed': validation_record['result'].passed,
                'accuracy_score': validation_record['result'].accuracy_score,
                'performance_change': self._calculate_performance_change(validation_record)
            },
            'recommendations': self._generate_recommendations(validation_record)
        }
        
        # Save research impact report
        report_path = Path("tests/geological_validation/research_reports")
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"{report['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Research impact report generated: {report_file}")
    
    def _generate_degradation_alert(self, validation_type: str, degradation_percentage: float, 
                                  current_result):
        """Generate performance degradation alert"""
        alert = {
            'alert_id': f"DEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'performance_degradation',
            'validation_type': validation_type,
            'degradation_percentage': degradation_percentage,
            'current_accuracy': current_result.accuracy_score,
            'severity': 'high' if degradation_percentage > 0.10 else 'medium',
            'recommended_actions': [
                'Review recent algorithm changes',
                'Check data quality',
                'Investigate system performance',
                'Consider algorithm retraining'
            ]
        }
        
        # Save alert
        alert_path = Path("tests/geological_validation/alerts")
        alert_path.mkdir(exist_ok=True)
        
        alert_file = alert_path / f"{alert['alert_id']}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2, default=str)
        
        logger.warning(f"Performance degradation alert generated: {alert_file}")
    
    def _calculate_performance_change(self, validation_record: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance change compared to baseline"""
        validation_type = validation_record['validation_type']
        current_accuracy = validation_record['result'].accuracy_score
        
        # Get historical baseline
        historical_results = [
            record for record in self.validation_history
            if record['validation_type'] == validation_type and 
               record['timestamp'] < validation_record['timestamp']
        ]
        
        if not historical_results:
            return {'change_percentage': 0.0, 'baseline_accuracy': current_accuracy}
        
        baseline_accuracy = np.mean([record['result'].accuracy_score for record in historical_results])
        change_percentage = (current_accuracy - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
        
        return {
            'change_percentage': change_percentage,
            'baseline_accuracy': baseline_accuracy,
            'current_accuracy': current_accuracy
        }
    
    def _generate_recommendations(self, validation_record: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        result = validation_record['result']
        research_info = validation_record.get('research_trigger', {})
        
        if not result.passed:
            recommendations.append("Investigate algorithm performance issues")
            recommendations.append("Review input data quality")
            recommendations.append("Consider algorithm parameter tuning")
        
        if research_info:
            impact_assessment = research_info.get('impact_assessment', {})
            if impact_assessment.get('priority') == 'high':
                recommendations.append("Consider incorporating new research findings")
                recommendations.append("Update algorithm based on latest research")
        
        if result.accuracy_score < 0.80:
            recommendations.append("Algorithm accuracy below acceptable threshold")
            recommendations.append("Consider retraining with additional data")
        
        return recommendations
    
    def run_manual_validation(self, validation_type: str, dataset_id: Optional[str] = None):
        """Run manual validation on demand"""
        logger.info(f"Running manual validation: {validation_type}")
        
        try:
            # Run validation based on type
            if validation_type == 'seismic_phase_picking':
                result = self.seismic_validator.validate_phase_picking_accuracy(dataset_id or "loma_prieta_1989")
            elif validation_type == 'volcanic_forecasting':
                result = self.volcanic_validator.validate_eruption_forecasting(dataset_id or "msh_1980_eruption")
            elif validation_type == 'geochemical_analysis':
                result = self.geochemical_validator.validate_major_element_analysis(dataset_id or "yellowstone_geochem_usgs")
            elif validation_type == 'all_validations':
                return self._run_comprehensive_validation()
            else:
                raise ValueError(f"Unknown validation type: {validation_type}")
            
            # Store result
            validation_record = {
                'timestamp': datetime.now(),
                'validation_type': validation_type,
                'result': result,
                'scheduled': False,
                'manual_trigger': True
            }
            
            self.validation_history.append(validation_record)
            
            logger.info(f"Manual validation completed: {validation_type}, Passed: {result.passed}")
            return result
            
        except Exception as e:
            logger.error(f"Manual validation failed: {e}")
            raise
    
    def _run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all systems"""
        logger.info("Running comprehensive validation suite")
        
        validation_results = {}
        
        # Run all validation types
        validation_types = [
            ('seismic_phase_picking', lambda: self.seismic_validator.validate_phase_picking_accuracy()),
            ('seismic_magnitude', lambda: self.seismic_validator.validate_magnitude_estimation()),
            ('volcanic_forecasting', lambda: self.volcanic_validator.validate_eruption_forecasting()),
            ('volcanic_vei_prediction', lambda: self.volcanic_validator.validate_vei_prediction()),
            ('geochemical_major_elements', lambda: self.geochemical_validator.validate_major_element_analysis()),
            ('geochemical_trace_elements', lambda: self.geochemical_validator.validate_trace_element_analysis()),
            ('visualization_spatial', lambda: self.visualization_validator.validate_spatial_coordinate_accuracy()),
            ('data_integrity', lambda: self.data_integrity_validator.validate_data_completeness())
        ]
        
        for validation_name, validation_func in validation_types:
            try:
                result = validation_func()
                validation_results[validation_name] = result
                
                # Store individual result
                validation_record = {
                    'timestamp': datetime.now(),
                    'validation_type': validation_name,
                    'result': result,
                    'scheduled': False,
                    'comprehensive_run': True
                }
                self.validation_history.append(validation_record)
                
            except Exception as e:
                logger.error(f"Failed validation {validation_name}: {e}")
                validation_results[validation_name] = None
        
        # Generate comprehensive report
        self._generate_comprehensive_report(validation_results)
        
        return validation_results
    
    def _generate_comprehensive_report(self, validation_results: Dict[str, Any]):
        """Generate comprehensive validation report"""
        passed_count = sum(1 for result in validation_results.values() 
                          if result and result.passed)
        total_count = len([r for r in validation_results.values() if r is not None])
        
        report = {
            'report_id': f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'report_type': 'comprehensive_validation',
            'summary': {
                'total_validations': total_count,
                'passed_validations': passed_count,
                'failed_validations': total_count - passed_count,
                'success_rate': passed_count / total_count if total_count > 0 else 0.0
            },
            'detailed_results': {}
        }
        
        for validation_name, result in validation_results.items():
            if result:
                report['detailed_results'][validation_name] = {
                    'passed': result.passed,
                    'accuracy_score': result.accuracy_score,
                    'test_name': result.test_name,
                    'algorithm': result.algorithm,
                    'notes': result.notes
                }
            else:
                report['detailed_results'][validation_name] = {
                    'passed': False,
                    'error': 'Validation failed to execute'
                }
        
        # Save comprehensive report
        report_path = Path("tests/geological_validation/comprehensive_reports")
        report_path.mkdir(exist_ok=True)
        
        report_file = report_path / f"{report['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive validation report generated: {report_file}")
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status and statistics"""
        current_time = datetime.now()
        
        # Recent validation statistics
        recent_validations = [
            record for record in self.validation_history
            if (current_time - record['timestamp']).days <= 7
        ]
        
        status = {
            'framework_status': 'running' if self.is_running else 'stopped',
            'last_update': current_time.isoformat(),
            'recent_statistics': {
                'total_validations_7_days': len(recent_validations),
                'passed_validations_7_days': sum(1 for v in recent_validations if v['result'].passed),
                'success_rate_7_days': sum(1 for v in recent_validations if v['result'].passed) / max(len(recent_validations), 1)
            },
            'schedule_status': [],
            'research_updates': {
                'total_updates': len(self.research_updates),
                'recent_updates': len([u for u in self.research_updates 
                                     if (current_time - u.publication_date).days <= 30])
            }
        }
        
        # Schedule status
        for schedule_config in self.validation_schedules:
            schedule_status = {
                'validation_type': schedule_config.validation_type,
                'frequency': schedule_config.frequency,
                'enabled': schedule_config.enabled,
                'last_run': schedule_config.last_run.isoformat() if schedule_config.last_run else None,
                'overdue': False
            }
            
            # Check if overdue
            if schedule_config.last_run:
                time_since_last = current_time - schedule_config.last_run
                if ((schedule_config.frequency == 'daily' and time_since_last > timedelta(days=1.5)) or
                    (schedule_config.frequency == 'weekly' and time_since_last > timedelta(days=8)) or
                    (schedule_config.frequency == 'monthly' and time_since_last > timedelta(days=32))):
                    schedule_status['overdue'] = True
            
            status['schedule_status'].append(schedule_status)
        
        return status
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent validations
        recent_validations = [
            record for record in self.validation_history
            if record['timestamp'] >= cutoff_date
        ]
        
        # Group by validation type
        trends = {}
        for validation_type in set(record['validation_type'] for record in recent_validations):
            type_validations = [
                record for record in recent_validations
                if record['validation_type'] == validation_type
            ]
            
            if type_validations:
                accuracy_scores = [record['result'].accuracy_score for record in type_validations]
                timestamps = [record['timestamp'] for record in type_validations]
                
                trends[validation_type] = {
                    'validation_count': len(type_validations),
                    'average_accuracy': np.mean(accuracy_scores),
                    'accuracy_trend': self._calculate_trend(timestamps, accuracy_scores),
                    'success_rate': sum(1 for record in type_validations if record['result'].passed) / len(type_validations),
                    'latest_accuracy': accuracy_scores[-1] if accuracy_scores else 0.0
                }
        
        return {
            'period_days': days,
            'trends_by_validation_type': trends,
            'overall_statistics': {
                'total_validations': len(recent_validations),
                'overall_success_rate': sum(1 for record in recent_validations if record['result'].passed) / max(len(recent_validations), 1),
                'average_accuracy': np.mean([record['result'].accuracy_score for record in recent_validations]) if recent_validations else 0.0
            }
        }
    
    def _calculate_trend(self, timestamps: List[datetime], values: List[float]) -> str:
        """Calculate trend direction (improving, declining, stable)"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.001:  # Improving
            return 'improving'
        elif slope < -0.001:  # Declining
            return 'declining'
        else:  # Stable
            return 'stable'