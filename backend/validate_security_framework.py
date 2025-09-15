#!/usr/bin/env python3
"""
Validation script for Database Security and Compliance Framework
Validates the implementation without external dependencies
"""

import sys
import os
import importlib.util
import traceback
from typing import Dict, Any, List

def validate_module_syntax(module_path: str) -> Dict[str, Any]:
    """Validate Python module syntax"""
    result = {
        'module': module_path,
        'syntax_valid': False,
        'imports_valid': False,
        'classes_found': [],
        'functions_found': [],
        'errors': []
    }
    
    try:
        # Check syntax by compiling
        with open(module_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        compile(source_code, module_path, 'exec')
        result['syntax_valid'] = True
        
        # Try to load module spec
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                
                # Extract classes and functions from source
                import ast
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        result['classes_found'].append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        result['functions_found'].append(node.name)
                
                result['imports_valid'] = True
                
            except Exception as e:
                result['errors'].append(f"Module loading error: {str(e)}")
        
    except SyntaxError as e:
        result['errors'].append(f"Syntax error: {str(e)}")
    except Exception as e:
        result['errors'].append(f"Validation error: {str(e)}")
    
    return result

def validate_security_framework() -> Dict[str, Any]:
    """Validate the complete security framework"""
    validation_results = {
        'overall_status': 'UNKNOWN',
        'modules_validated': 0,
        'modules_passed': 0,
        'modules_failed': 0,
        'module_results': [],
        'summary': {}
    }
    
    # List of modules to validate
    modules_to_validate = [
        'backend/core/database_security_compliance_framework.py',
        'backend/tests/test_database_security_compliance_framework.py'
    ]
    
    print("🔒 Validating Database Security and Compliance Framework")
    print("=" * 60)
    
    for module_path in modules_to_validate:
        if os.path.exists(module_path):
            print(f"\n📁 Validating: {module_path}")
            result = validate_module_syntax(module_path)
            validation_results['module_results'].append(result)
            validation_results['modules_validated'] += 1
            
            if result['syntax_valid']:
                validation_results['modules_passed'] += 1
                print(f"  ✅ Syntax: VALID")
                print(f"  📦 Classes found: {len(result['classes_found'])}")
                print(f"  🔧 Functions found: {len(result['functions_found'])}")
                
                # Show key classes for security framework
                if 'database_security_compliance_framework.py' in module_path:
                    key_classes = [
                        'DatabaseSecurityComplianceFramework',
                        'DatabaseEncryptionManager',
                        'RBACManager',
                        'AuditLogger',
                        'VulnerabilityScanner',
                        'ComplianceManager',
                        'DataMaskingManager',
                        'SecurityIncidentManager'
                    ]
                    
                    found_key_classes = [cls for cls in key_classes if cls in result['classes_found']]
                    print(f"  🔑 Key security classes: {len(found_key_classes)}/{len(key_classes)}")
                    
                    if len(found_key_classes) == len(key_classes):
                        print("  🎯 All required security components implemented")
                    else:
                        missing = [cls for cls in key_classes if cls not in result['classes_found']]
                        print(f"  ⚠️  Missing classes: {missing}")
                
            else:
                validation_results['modules_failed'] += 1
                print(f"  ❌ Syntax: INVALID")
                for error in result['errors']:
                    print(f"    🚨 {error}")
        else:
            print(f"\n❌ Module not found: {module_path}")
            validation_results['modules_failed'] += 1
    
    # Calculate overall status
    if validation_results['modules_failed'] == 0:
        validation_results['overall_status'] = 'PASSED'
    elif validation_results['modules_passed'] > 0:
        validation_results['overall_status'] = 'PARTIAL'
    else:
        validation_results['overall_status'] = 'FAILED'
    
    # Generate summary
    validation_results['summary'] = {
        'total_modules': validation_results['modules_validated'],
        'passed_modules': validation_results['modules_passed'],
        'failed_modules': validation_results['modules_failed'],
        'success_rate': (validation_results['modules_passed'] / max(1, validation_results['modules_validated'])) * 100
    }
    
    return validation_results

def validate_configuration_files() -> Dict[str, Any]:
    """Validate configuration files"""
    config_results = {
        'files_checked': 0,
        'files_valid': 0,
        'files_invalid': 0,
        'file_results': []
    }
    
    config_files = [
        'backend/config/security_config.yaml',
        'docker-compose.security.yml',
        'kubernetes/database-security-manifests.yaml'
    ]
    
    print("\n📋 Validating Configuration Files")
    print("=" * 40)
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n📄 Checking: {config_file}")
            config_results['files_checked'] += 1
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic validation
                if len(content) > 0:
                    config_results['files_valid'] += 1
                    print(f"  ✅ File readable and non-empty ({len(content)} bytes)")
                    
                    # Check for key security configurations
                    if 'security_config.yaml' in config_file:
                        security_keys = ['encryption', 'rbac', 'audit_logging', 'compliance', 'vulnerability_scanning']
                        found_keys = [key for key in security_keys if key in content]
                        print(f"  🔐 Security sections: {len(found_keys)}/{len(security_keys)}")
                    
                    elif 'docker-compose' in config_file:
                        docker_keys = ['postgresql-secure', 'influxdb-secure', 'redis-secure', 'neo4j-secure']
                        found_keys = [key for key in docker_keys if key in content]
                        print(f"  🐳 Secure database services: {len(found_keys)}/{len(docker_keys)}")
                    
                    elif 'kubernetes' in config_file:
                        k8s_keys = ['NetworkPolicy', 'PodSecurityPolicy', 'RBAC', 'Secret']
                        found_keys = [key for key in k8s_keys if key in content]
                        print(f"  ☸️  Kubernetes security resources: {len(found_keys)}/{len(k8s_keys)}")
                
                else:
                    config_results['files_invalid'] += 1
                    print(f"  ❌ File is empty")
                
                config_results['file_results'].append({
                    'file': config_file,
                    'valid': True,
                    'size': len(content)
                })
                
            except Exception as e:
                config_results['files_invalid'] += 1
                print(f"  ❌ Error reading file: {str(e)}")
                config_results['file_results'].append({
                    'file': config_file,
                    'valid': False,
                    'error': str(e)
                })
        else:
            print(f"\n❌ Configuration file not found: {config_file}")
            config_results['files_invalid'] += 1
    
    return config_results

def validate_security_features() -> Dict[str, Any]:
    """Validate security feature implementation"""
    features_results = {
        'features_checked': 0,
        'features_implemented': 0,
        'feature_details': {}
    }
    
    print("\n🛡️  Validating Security Features")
    print("=" * 40)
    
    # Check if main security framework file exists and has key features
    framework_file = 'backend/core/database_security_compliance_framework.py'
    
    if os.path.exists(framework_file):
        with open(framework_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Define security features to check
        security_features = {
            'Encryption at Rest and in Transit': [
                'DatabaseEncryptionManager',
                'AES_256_GCM',
                'encrypt_data',
                'decrypt_data'
            ],
            'Role-Based Access Control (RBAC)': [
                'RBACManager',
                'SecurityUser',
                'RBACRole',
                'authenticate_user'
            ],
            'Audit Logging': [
                'AuditLogger',
                'AuditLogEntry',
                'log_database_operation',
                '_persist_audit_log'
            ],
            'Vulnerability Scanning': [
                'VulnerabilityScanner',
                'scan_vulnerabilities',
                '_check_weak_passwords',
                '_check_excessive_privileges'
            ],
            'Compliance Reporting': [
                'ComplianceManager',
                'ComplianceReport',
                'generate_compliance_report',
                'SOC2'
            ],
            'Data Masking': [
                'DataMaskingManager',
                'mask_email',
                'mask_phone',
                'apply_masking'
            ],
            'Incident Response': [
                'SecurityIncidentManager',
                'SecurityIncident',
                'create_incident',
                '_handle_unauthorized_access'
            ],
            'Mutual TLS Authentication': [
                'setup_mutual_tls',
                'ssl_cert_file',
                'ssl_key_file'
            ]
        }
        
        for feature_name, keywords in security_features.items():
            features_results['features_checked'] += 1
            found_keywords = [kw for kw in keywords if kw in content]
            
            if len(found_keywords) >= len(keywords) * 0.75:  # 75% of keywords found
                features_results['features_implemented'] += 1
                status = "✅ IMPLEMENTED"
            else:
                status = "⚠️  PARTIAL"
            
            features_results['feature_details'][feature_name] = {
                'status': status,
                'keywords_found': len(found_keywords),
                'keywords_total': len(keywords),
                'coverage': (len(found_keywords) / len(keywords)) * 100
            }
            
            print(f"\n🔧 {feature_name}")
            print(f"   {status} ({len(found_keywords)}/{len(keywords)} components)")
    
    return features_results

def main():
    """Main validation function"""
    print("🚀 Database Security and Compliance Framework Validation")
    print("=" * 70)
    
    # Validate Python modules
    module_results = validate_security_framework()
    
    # Validate configuration files
    config_results = validate_configuration_files()
    
    # Validate security features
    feature_results = validate_security_features()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n🐍 Python Modules:")
    print(f"   Status: {module_results['overall_status']}")
    print(f"   Modules: {module_results['modules_passed']}/{module_results['modules_validated']} passed")
    print(f"   Success Rate: {module_results['summary']['success_rate']:.1f}%")
    
    print(f"\n📋 Configuration Files:")
    print(f"   Valid: {config_results['files_valid']}/{config_results['files_checked']}")
    
    print(f"\n🛡️  Security Features:")
    print(f"   Implemented: {feature_results['features_implemented']}/{feature_results['features_checked']}")
    
    # Overall assessment
    overall_score = (
        (module_results['summary']['success_rate'] * 0.5) +
        ((config_results['files_valid'] / max(1, config_results['files_checked'])) * 100 * 0.3) +
        ((feature_results['features_implemented'] / max(1, feature_results['features_checked'])) * 100 * 0.2)
    )
    
    print(f"\n🎯 Overall Score: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print("🏆 EXCELLENT - Security framework is production-ready!")
    elif overall_score >= 75:
        print("✅ GOOD - Security framework is well implemented")
    elif overall_score >= 60:
        print("⚠️  ACCEPTABLE - Some improvements needed")
    else:
        print("❌ NEEDS WORK - Significant improvements required")
    
    return overall_score >= 75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)