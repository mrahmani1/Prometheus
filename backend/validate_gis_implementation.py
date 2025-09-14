"""
GIS PostgreSQL Implementation Validation Script
Validates the GIS-enabled PostgreSQL operations implementation
"""

import sys
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_imports():
    """Validate that all required modules can be imported"""
    logger.info("Validating module imports...")
    
    required_modules = [
        'backend.core.gis_postgresql_operations',
        'backend.core.gis_database_integration',
        'backend.config.gis_postgresql_config'
    ]
    
    import_results = {}
    
    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            import_results[module_name] = {'success': True, 'module': module}
            logger.info(f"✓ Successfully imported {module_name}")
        except Exception as e:
            import_results[module_name] = {'success': False, 'error': str(e)}
            logger.error(f"✗ Failed to import {module_name}: {e}")
    
    return import_results


def validate_class_definitions(import_results):
    """Validate that required classes are properly defined"""
    logger.info("Validating class definitions...")
    
    validation_results = {}
    
    # Check GIS PostgreSQL Operations classes
    gis_ops_module = import_results.get('backend.core.gis_postgresql_operations', {}).get('module')
    if gis_ops_module:
        required_classes = [
            'GISPostgreSQLOperations',
            'VolcanicSystem',
            'SeismicEvent',
            'DeformationMeasurement',
            'MonitoringStation',
            'GeologicalUnit',
            'TectonicFeature',
            'SpatialUtils'
        ]
        
        for class_name in required_classes:
            if hasattr(gis_ops_module, class_name):
                validation_results[f'gis_ops.{class_name}'] = True
                logger.info(f"✓ Found class {class_name}")
            else:
                validation_results[f'gis_ops.{class_name}'] = False
                logger.error(f"✗ Missing class {class_name}")
    
    # Check GIS Database Integration classes
    gis_db_module = import_results.get('backend.core.gis_database_integration', {}).get('module')
    if gis_db_module:
        required_classes = [
            'GISDatabaseIntegrator',
            'GISQueryRequest',
            'GISQueryResponse'
        ]
        
        for class_name in required_classes:
            if hasattr(gis_db_module, class_name):
                validation_results[f'gis_db.{class_name}'] = True
                logger.info(f"✓ Found class {class_name}")
            else:
                validation_results[f'gis_db.{class_name}'] = False
                logger.error(f"✗ Missing class {class_name}")
    
    # Check Configuration classes
    config_module = import_results.get('backend.config.gis_postgresql_config', {}).get('module')
    if config_module:
        required_classes = [
            'GISPostgreSQLConfig',
            'SpatialIndexConfig',
            'CoordinateSystemConfig',
            'QueryOptimizationConfig'
        ]
        
        for class_name in required_classes:
            if hasattr(config_module, class_name):
                validation_results[f'config.{class_name}'] = True
                logger.info(f"✓ Found class {class_name}")
            else:
                validation_results[f'config.{class_name}'] = False
                logger.error(f"✗ Missing class {class_name}")
    
    return validation_results


def validate_enum_definitions(import_results):
    """Validate that required enums are properly defined"""
    logger.info("Validating enum definitions...")
    
    validation_results = {}
    
    gis_ops_module = import_results.get('backend.core.gis_postgresql_operations', {}).get('module')
    if gis_ops_module:
        required_enums = [
            'CoordinateSystem',
            'SpatialIndexType',
            'ExportFormat'
        ]
        
        for enum_name in required_enums:
            if hasattr(gis_ops_module, enum_name):
                enum_class = getattr(gis_ops_module, enum_name)
                validation_results[f'enum.{enum_name}'] = True
                logger.info(f"✓ Found enum {enum_name} with values: {list(enum_class)}")
            else:
                validation_results[f'enum.{enum_name}'] = False
                logger.error(f"✗ Missing enum {enum_name}")
    
    return validation_results


def validate_method_signatures(import_results):
    """Validate that key methods have correct signatures"""
    logger.info("Validating method signatures...")
    
    validation_results = {}
    
    gis_ops_module = import_results.get('backend.core.gis_postgresql_operations', {}).get('module')
    if gis_ops_module and hasattr(gis_ops_module, 'GISPostgreSQLOperations'):
        gis_ops_class = getattr(gis_ops_module, 'GISPostgreSQLOperations')
        
        required_methods = [
            'initialize_spatial_extensions',
            'create_spatial_indexes',
            'query_volcanic_systems_by_region',
            'query_seismic_events_spatial_temporal',
            'query_monitoring_network_coverage',
            'transform_coordinates',
            'validate_geometry_quality',
            'perform_spatial_clustering',
            'export_geospatial_data'
        ]
        
        for method_name in required_methods:
            if hasattr(gis_ops_class, method_name):
                validation_results[f'method.{method_name}'] = True
                logger.info(f"✓ Found method {method_name}")
            else:
                validation_results[f'method.{method_name}'] = False
                logger.error(f"✗ Missing method {method_name}")
    
    return validation_results


def validate_file_structure():
    """Validate that all required files exist"""
    logger.info("Validating file structure...")
    
    required_files = [
        'backend/core/gis_postgresql_operations.py',
        'backend/core/gis_database_integration.py',
        'backend/config/gis_postgresql_config.py',
        'backend/tests/test_gis_postgresql_operations.py',
        'backend/examples/gis_postgresql_demo.py'
    ]
    
    validation_results = {}
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            validation_results[file_path] = True
            logger.info(f"✓ Found file {file_path}")
        else:
            validation_results[file_path] = False
            logger.error(f"✗ Missing file {file_path}")
    
    return validation_results


def validate_sqlalchemy_models(import_results):
    """Validate SQLAlchemy model definitions"""
    logger.info("Validating SQLAlchemy models...")
    
    validation_results = {}
    
    gis_ops_module = import_results.get('backend.core.gis_postgresql_operations', {}).get('module')
    if gis_ops_module:
        model_classes = [
            'VolcanicSystem',
            'SeismicEvent',
            'DeformationMeasurement',
            'MonitoringStation',
            'GeologicalUnit',
            'TectonicFeature'
        ]
        
        for model_name in model_classes:
            if hasattr(gis_ops_module, model_name):
                model_class = getattr(gis_ops_module, model_name)
                
                # Check if it has SQLAlchemy attributes
                has_tablename = hasattr(model_class, '__tablename__')
                has_table_args = hasattr(model_class, '__table_args__')
                
                validation_results[f'model.{model_name}'] = has_tablename
                
                if has_tablename:
                    logger.info(f"✓ Model {model_name} has __tablename__: {model_class.__tablename__}")
                else:
                    logger.error(f"✗ Model {model_name} missing __tablename__")
                
                if has_table_args:
                    logger.info(f"✓ Model {model_name} has spatial indexes defined")
            else:
                validation_results[f'model.{model_name}'] = False
                logger.error(f"✗ Missing model {model_name}")
    
    return validation_results


def validate_configuration_completeness(import_results):
    """Validate configuration completeness"""
    logger.info("Validating configuration completeness...")
    
    validation_results = {}
    
    config_module = import_results.get('backend.config.gis_postgresql_config', {}).get('module')
    if config_module and hasattr(config_module, 'GISPostgreSQLConfig'):
        try:
            # Test configuration instantiation
            config = config_module.GISPostgreSQLConfig()
            
            # Check required attributes
            required_attrs = [
                'spatial_indexing',
                'coordinate_systems',
                'query_optimization',
                'caching',
                'export',
                'validation',
                'performance',
                'database',
                'extensions',
                'table_configs'
            ]
            
            for attr_name in required_attrs:
                if hasattr(config, attr_name):
                    validation_results[f'config.{attr_name}'] = True
                    logger.info(f"✓ Configuration has {attr_name}")
                else:
                    validation_results[f'config.{attr_name}'] = False
                    logger.error(f"✗ Configuration missing {attr_name}")
            
            # Test configuration validation
            issues = config.validate_config()
            if not issues:
                validation_results['config.validation'] = True
                logger.info("✓ Configuration validation passed")
            else:
                validation_results['config.validation'] = False
                logger.warning(f"Configuration validation issues: {issues}")
            
        except Exception as e:
            validation_results['config.instantiation'] = False
            logger.error(f"✗ Failed to instantiate configuration: {e}")
    
    return validation_results


def generate_validation_report(all_results):
    """Generate comprehensive validation report"""
    logger.info("Generating validation report...")
    
    total_checks = sum(len(results) for results in all_results.values())
    passed_checks = sum(
        sum(1 for result in results.values() if result) 
        for results in all_results.values()
    )
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print("\n" + "="*80)
    print("GIS POSTGRESQL IMPLEMENTATION VALIDATION REPORT")
    print("="*80)
    print(f"Total checks: {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Failed checks: {total_checks - passed_checks}")
    print(f"Success rate: {success_rate:.1f}%")
    print("="*80)
    
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        for check, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status} {check}")
    
    print("\n" + "="*80)
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Implementation is highly complete and functional!")
    elif success_rate >= 75:
        print("✅ GOOD: Implementation is mostly complete with minor issues.")
    elif success_rate >= 50:
        print("⚠️  FAIR: Implementation has significant gaps that need attention.")
    else:
        print("❌ POOR: Implementation requires major work before use.")
    
    print("="*80)
    
    return success_rate


def main():
    """Main validation function"""
    logger.info("Starting GIS PostgreSQL implementation validation...")
    
    try:
        # Run all validation checks
        validation_results = {}
        
        # 1. Import validation
        import_results = validate_imports()
        validation_results['imports'] = {k: v['success'] for k, v in import_results.items()}
        
        # 2. Class definitions validation
        class_results = validate_class_definitions(import_results)
        validation_results['classes'] = class_results
        
        # 3. Enum definitions validation
        enum_results = validate_enum_definitions(import_results)
        validation_results['enums'] = enum_results
        
        # 4. Method signatures validation
        method_results = validate_method_signatures(import_results)
        validation_results['methods'] = method_results
        
        # 5. File structure validation
        file_results = validate_file_structure()
        validation_results['files'] = file_results
        
        # 6. SQLAlchemy models validation
        model_results = validate_sqlalchemy_models(import_results)
        validation_results['models'] = model_results
        
        # 7. Configuration validation
        config_results = validate_configuration_completeness(import_results)
        validation_results['configuration'] = config_results
        
        # Generate final report
        success_rate = generate_validation_report(validation_results)
        
        # Return appropriate exit code
        if success_rate >= 75:
            logger.info("Validation completed successfully!")
            return 0
        else:
            logger.error("Validation failed - implementation needs improvement")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)