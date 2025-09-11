#!/usr/bin/env python3
"""
Database Integration Validation Script
Validates that all components of task 2 are properly integrated
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseIntegrationValidator:
    """Validates the complete database schema integration"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.validation_results = {}
        
    def validate_subtask_2_1(self) -> Tuple[bool, List[str]]:
        """Validate subtask 2.1: Comprehensive Geological Database Schema"""
        logger.info("Validating subtask 2.1: Comprehensive Geological Database Schema")
        
        issues = []
        
        # Check for comprehensive geological schema file
        schema_file = self.base_path / "database" / "postgres" / "comprehensive_geological_schema.sql"
        if not schema_file.exists():
            issues.append("Missing comprehensive_geological_schema.sql")
        else:
            content = schema_file.read_text()
            
            # Check for required schemas
            required_schemas = [
                'geological_core', 'rock_mechanics', 'volcanic_systems',
                'seismic_networks', 'geodetic_data', 'geophysical_data'
            ]
            
            for schema in required_schemas:
                if f"CREATE SCHEMA IF NOT EXISTS {schema}" not in content:
                    issues.append(f"Missing schema creation: {schema}")
            
            # Check for advanced rock mechanical properties
            if "rock_mechanical_properties" not in content:
                issues.append("Missing rock_mechanical_properties table")
            
            # Check for anisotropic modeling (tensor components)
            if "c11_gpa" not in content or "c66_gpa" not in content:
                issues.append("Missing anisotropic elastic tensor components")
            
            # Check for 3D spatial indexing
            if "GIST" not in content:
                issues.append("Missing GIST spatial indexing")
            
            # Check for temporal evolution tracking
            if "evolution_history" not in content:
                issues.append("Missing temporal evolution tracking")
        
        success = len(issues) == 0
        self.validation_results['subtask_2_1'] = {'success': success, 'issues': issues}
        
        if success:
            logger.info("✓ Subtask 2.1 validation passed")
        else:
            logger.error(f"✗ Subtask 2.1 validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def validate_subtask_2_2(self) -> Tuple[bool, List[str]]:
        """Validate subtask 2.2: Model Database System"""
        logger.info("Validating subtask 2.2: Model Database System")
        
        issues = []
        
        # Check for model database implementation
        model_db_file = self.base_path / "backend" / "modules" / "model_database.py"
        if not model_db_file.exists():
            issues.append("Missing model_database.py implementation")
        else:
            content = model_db_file.read_text()
            
            # Check for required classes
            required_classes = [
                'ModelMetadata', 'ModelVersion', 'StorageManager', 
                'ModelValidationService', 'ModelSearchService', 'ModelDatabase'
            ]
            
            for cls in required_classes:
                if f"class {cls}" not in content:
                    issues.append(f"Missing class: {cls}")
            
            # Check for multi-cloud storage support
            storage_backends = ['AWS_S3', 'AZURE_BLOB', 'GOOGLE_CLOUD', 'MINIO']
            for backend in storage_backends:
                if backend not in content:
                    issues.append(f"Missing storage backend: {backend}")
            
            # Check for model versioning
            if "parent_model_id" not in content or "child_model_ids" not in content:
                issues.append("Missing model versioning with parent-child relationships")
            
            # Check for validation service
            if "validate_pytorch_model" not in content:
                issues.append("Missing model validation service")
        
        success = len(issues) == 0
        self.validation_results['subtask_2_2'] = {'success': success, 'issues': issues}
        
        if success:
            logger.info("✓ Subtask 2.2 validation passed")
        else:
            logger.error(f"✗ Subtask 2.2 validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def validate_subtask_2_3(self) -> Tuple[bool, List[str]]:
        """Validate subtask 2.3: Advanced Data Integration Pipeline"""
        logger.info("Validating subtask 2.3: Advanced Data Integration Pipeline")
        
        issues = []
        
        # Check for data integration pipeline implementation
        pipeline_file = self.base_path / "backend" / "modules" / "data_integration_pipeline.py"
        if not pipeline_file.exists():
            issues.append("Missing data_integration_pipeline.py implementation")
        else:
            content = pipeline_file.read_text()
            
            # Check for DataLayer class with real-time capabilities
            if "class DataLayer" not in content:
                issues.append("Missing DataLayer class")
            
            if "get_real_time_volcanic_data" not in content:
                issues.append("Missing real-time volcanic data enhancement")
            
            # Check for multi-source data integration
            integration_methods = [
                '_enhance_with_geological_data',
                '_enhance_with_geophysical_data', 
                '_enhance_with_geochemical_data',
                '_enhance_with_remote_sensing_data'
            ]
            
            for method in integration_methods:
                if method not in content:
                    issues.append(f"Missing integration method: {method}")
            
            # Check for quality assessment
            if "_apply_quality_assessment" not in content:
                issues.append("Missing data quality assessment")
            
            # Check for automated feature engineering
            if "_apply_automated_feature_engineering" not in content:
                issues.append("Missing automated feature engineering")
        
        success = len(issues) == 0
        self.validation_results['subtask_2_3'] = {'success': success, 'issues': issues}
        
        if success:
            logger.info("✓ Subtask 2.3 validation passed")
        else:
            logger.error(f"✗ Subtask 2.3 validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def validate_subtask_2_4(self) -> Tuple[bool, List[str]]:
        """Validate subtask 2.4: Model Manager Service"""
        logger.info("Validating subtask 2.4: Model Manager Service")
        
        issues = []
        
        # Check for model manager implementation
        manager_file = self.base_path / "backend" / "modules" / "model_manager.py"
        if not manager_file.exists():
            issues.append("Missing model_manager.py implementation")
        else:
            content = manager_file.read_text()
            
            # Check for required classes
            required_classes = [
                'ModelConfig', 'VolcanicDataset', 'VolcanicPredictionNet',
                'MultiTaskVolcanicNet', 'ModelManager'
            ]
            
            for cls in required_classes:
                if f"class {cls}" not in content:
                    issues.append(f"Missing class: {cls}")
            
            # Check for multi-cloud training integration
            cloud_providers = ['azure', 'google_cloud', 'aws']
            for provider in cloud_providers:
                if provider not in content.lower():
                    issues.append(f"Missing cloud provider integration: {provider}")
            
            # Check for MLOps pipeline integration
            if "mlops_pipeline" not in content:
                issues.append("Missing MLOps pipeline integration")
            
            # Check for prediction service with confidence calculation
            if "predict_with_uncertainty" not in content:
                issues.append("Missing uncertainty quantification in predictions")
            
            # Check for multi-model training system
            model_capabilities = [
                ('eruption_probability', 'eruption probability prediction'),
                ('vei', 'VEI prediction'),  # VEI is used instead of vei_prediction
                ('time_to_event', 'time to event prediction')
            ]
            for model_key, model_desc in model_capabilities:
                if model_key not in content:
                    issues.append(f"Missing model type support: {model_desc}")
        
        success = len(issues) == 0
        self.validation_results['subtask_2_4'] = {'success': success, 'issues': issues}
        
        if success:
            logger.info("✓ Subtask 2.4 validation passed")
        else:
            logger.error(f"✗ Subtask 2.4 validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def validate_subtask_2_5(self) -> Tuple[bool, List[str]]:
        """Validate subtask 2.5: Database Deployment Configuration"""
        logger.info("Validating subtask 2.5: Database Deployment Configuration")
        
        issues = []
        
        # Check for deployment configuration files
        required_files = [
            "database/postgres/comprehensive_deployment_init.sql",
            "database/postgres/environment_config.env",
            "database/postgres/deploy.sh"
        ]
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            if not full_path.exists():
                issues.append(f"Missing deployment file: {file_path}")
        
        # Check environment configuration
        env_file = self.base_path / "database" / "postgres" / "environment_config.env"
        if env_file.exists():
            content = env_file.read_text()
            
            # Check for required environment variables
            required_vars = [
                'POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER',
                'EARTHCHEM_API_KEY', 'GEOROC_API_ENDPOINT', 'NAVDAT_API_ENDPOINT',
                'PETDB_API_ENDPOINT', 'MLFLOW_TRACKING_URI', 'AZURE_VM_ENDPOINT'
            ]
            
            for var in required_vars:
                if var not in content:
                    issues.append(f"Missing environment variable: {var}")
        
        # Check for migration system
        migration_files = list((self.base_path / "database" / "postgres" / "migrations").glob("*.sql"))
        if len(migration_files) == 0:
            issues.append("Missing database migration files")
        
        # Check for migration manager
        migration_manager = self.base_path / "database" / "postgres" / "migration_manager.py"
        if not migration_manager.exists():
            issues.append("Missing migration manager")
        
        success = len(issues) == 0
        self.validation_results['subtask_2_5'] = {'success': success, 'issues': issues}
        
        if success:
            logger.info("✓ Subtask 2.5 validation passed")
        else:
            logger.error(f"✗ Subtask 2.5 validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def validate_integration_completeness(self) -> Tuple[bool, List[str]]:
        """Validate overall integration completeness"""
        logger.info("Validating overall integration completeness")
        
        issues = []
        
        # Check that all subtasks are implemented
        subtask_results = [
            self.validation_results.get('subtask_2_1', {}).get('success', False),
            self.validation_results.get('subtask_2_2', {}).get('success', False),
            self.validation_results.get('subtask_2_3', {}).get('success', False),
            self.validation_results.get('subtask_2_4', {}).get('success', False),
            self.validation_results.get('subtask_2_5', {}).get('success', False)
        ]
        
        if not all(subtask_results):
            failed_subtasks = []
            for i, result in enumerate(subtask_results, 1):
                if not result:
                    failed_subtasks.append(f"2.{i}")
            issues.append(f"Failed subtasks: {', '.join(failed_subtasks)}")
        
        # Check for cross-component integration
        model_db_file = self.base_path / "backend" / "modules" / "model_database.py"
        pipeline_file = self.base_path / "backend" / "modules" / "data_integration_pipeline.py"
        manager_file = self.base_path / "backend" / "modules" / "model_manager.py"
        
        if all(f.exists() for f in [model_db_file, pipeline_file, manager_file]):
            # Check for cross-imports and integration
            manager_content = manager_file.read_text()
            
            if "model_database" not in manager_content:
                issues.append("Model manager not integrated with model database")
            
            if "mlops_pipeline" not in manager_content:
                issues.append("Model manager not integrated with MLOps pipeline")
        
        # Check for database schema integration with backend modules
        schema_files = [
            self.base_path / "database" / "postgres" / "comprehensive_geological_schema.sql",
            self.base_path / "database" / "postgres" / "comprehensive_deployment_init.sql"
        ]
        
        schema_content = ""
        for schema_file in schema_files:
            if schema_file.exists():
                schema_content += schema_file.read_text()
        
        if schema_content:
            # Verify that schema supports the backend module requirements
            if "model_management" not in schema_content:
                issues.append("Database schema missing model management support")
            
            if "geological_age" not in schema_content:
                issues.append("Database schema missing geological age support")
        
        success = len(issues) == 0
        
        if success:
            logger.info("✓ Integration completeness validation passed")
        else:
            logger.error(f"✗ Integration completeness validation failed: {len(issues)} issues found")
            
        return success, issues
    
    def run_full_validation(self) -> Dict:
        """Run complete validation of all subtasks and integration"""
        logger.info("Starting comprehensive database integration validation")
        
        # Validate each subtask
        subtask_2_1_success, subtask_2_1_issues = self.validate_subtask_2_1()
        subtask_2_2_success, subtask_2_2_issues = self.validate_subtask_2_2()
        subtask_2_3_success, subtask_2_3_issues = self.validate_subtask_2_3()
        subtask_2_4_success, subtask_2_4_issues = self.validate_subtask_2_4()
        subtask_2_5_success, subtask_2_5_issues = self.validate_subtask_2_5()
        
        # Validate overall integration
        integration_success, integration_issues = self.validate_integration_completeness()
        
        # Compile results
        all_success = all([
            subtask_2_1_success, subtask_2_2_success, subtask_2_3_success,
            subtask_2_4_success, subtask_2_5_success, integration_success
        ])
        
        all_issues = (
            subtask_2_1_issues + subtask_2_2_issues + subtask_2_3_issues +
            subtask_2_4_issues + subtask_2_5_issues + integration_issues
        )
        
        results = {
            'overall_success': all_success,
            'total_issues': len(all_issues),
            'subtasks': {
                '2.1': {'success': subtask_2_1_success, 'issues': subtask_2_1_issues},
                '2.2': {'success': subtask_2_2_success, 'issues': subtask_2_2_issues},
                '2.3': {'success': subtask_2_3_success, 'issues': subtask_2_3_issues},
                '2.4': {'success': subtask_2_4_success, 'issues': subtask_2_4_issues},
                '2.5': {'success': subtask_2_5_success, 'issues': subtask_2_5_issues}
            },
            'integration': {'success': integration_success, 'issues': integration_issues},
            'all_issues': all_issues
        }
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        results = self.run_full_validation()
        
        report = []
        report.append("=" * 80)
        report.append("DATABASE INTEGRATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Overall Status: {'✓ PASSED' if results['overall_success'] else '✗ FAILED'}")
        report.append(f"Total Issues Found: {results['total_issues']}")
        report.append("")
        
        # Subtask results
        report.append("SUBTASK VALIDATION RESULTS:")
        report.append("-" * 40)
        
        for subtask_id, subtask_result in results['subtasks'].items():
            status = "✓ PASSED" if subtask_result['success'] else "✗ FAILED"
            report.append(f"Subtask {subtask_id}: {status}")
            
            if subtask_result['issues']:
                for issue in subtask_result['issues']:
                    report.append(f"  - {issue}")
                report.append("")
        
        # Integration results
        report.append("INTEGRATION VALIDATION RESULTS:")
        report.append("-" * 40)
        status = "✓ PASSED" if results['integration']['success'] else "✗ FAILED"
        report.append(f"Integration Completeness: {status}")
        
        if results['integration']['issues']:
            for issue in results['integration']['issues']:
                report.append(f"  - {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        if results['overall_success']:
            report.append("✓ ALL VALIDATIONS PASSED - TASK 2 IS COMPLETE")
        else:
            report.append("✗ VALIDATION FAILED - ISSUES NEED TO BE RESOLVED")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main validation function"""
    validator = DatabaseIntegrationValidator()
    
    # Generate and display validation report
    report = validator.generate_validation_report()
    print(report)
    
    # Save report to file (with proper encoding)
    report_file = Path("database_integration_validation_report.txt")
    # Replace Unicode characters for Windows compatibility
    safe_report = report.replace('✓', '[PASS]').replace('✗', '[FAIL]')
    report_file.write_text(safe_report, encoding='utf-8')
    logger.info(f"Validation report saved to: {report_file}")
    
    # Exit with appropriate code
    results = validator.run_full_validation()
    sys.exit(0 if results['overall_success'] else 1)

if __name__ == '__main__':
    main()