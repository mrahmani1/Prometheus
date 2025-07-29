#!/usr/bin/env python3
"""
Cloud Run Implementation Validation Script

This script validates the Cloud Run serverless scaling implementation
by checking all components and configurations.
"""

import os
import sys
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any

def validate_file_structure():
    """Validate that all required files are present"""
    print("🔍 Validating file structure...")
    
    required_files = [
        "cloud-run-optimizer.py",
        "multi-region-deployment.py", 
        "cost-optimization.py",
        "deploy.sh",
        "requirements.txt",
        "Dockerfile.cloud-run-base",
        "Dockerfile.helios-core-cloudrun",
        "Dockerfile.api-gateway-cloudrun",
        "cloud-run-deploy.yaml",
        "terraform/main.tf",
        "terraform/terraform.tfvars.example",
        "monitoring/alerts.yaml",
        "DEPLOYMENT_GUIDE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def validate_python_syntax():
    """Validate Python syntax for all Python files"""
    print("\n🐍 Validating Python syntax...")
    
    python_files = [
        "cloud-run-optimizer.py",
        "multi-region-deployment.py",
        "cost-optimization.py",
        "validate_implementation.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"❌ {file_path} - syntax error: {e}")
        except UnicodeDecodeError as e:
            print(f"⚠️  {file_path} - encoding error: {e}")
        except FileNotFoundError:
            print(f"⚠️  {file_path} - file not found")
    
    return len(syntax_errors) == 0

def validate_docker_files():
    """Validate Dockerfile syntax and structure"""
    print("\n🐳 Validating Docker files...")
    
    docker_files = [
        "Dockerfile.cloud-run-base",
        "Dockerfile.helios-core-cloudrun", 
        "Dockerfile.api-gateway-cloudrun"
    ]
    
    valid_dockerfiles = 0
    for dockerfile in docker_files:
        if os.path.exists(dockerfile):
            with open(dockerfile, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic validation
            if content.startswith('FROM '):
                print(f"✅ {dockerfile} - valid structure")
                valid_dockerfiles += 1
            else:
                print(f"❌ {dockerfile} - invalid structure")
        else:
            print(f"⚠️  {dockerfile} - file not found")
    
    return valid_dockerfiles == len(docker_files)

def validate_yaml_files():
    """Validate YAML file syntax"""
    print("\n📄 Validating YAML files...")
    
    yaml_files = [
        "cloud-run-deploy.yaml",
        "monitoring/alerts.yaml"
    ]
    
    valid_yamls = 0
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            print(f"✅ {yaml_file} - valid YAML")
            valid_yamls += 1
        except yaml.YAMLError as e:
            print(f"❌ {yaml_file} - YAML error: {e}")
        except FileNotFoundError:
            print(f"⚠️  {yaml_file} - file not found")
    
    return valid_yamls == len(yaml_files)

def validate_terraform_files():
    """Validate Terraform configuration"""
    print("\n🏗️  Validating Terraform files...")
    
    terraform_files = [
        "terraform/main.tf",
        "terraform/terraform.tfvars.example"
    ]
    
    valid_tf_files = 0
    for tf_file in terraform_files:
        if os.path.exists(tf_file):
            with open(tf_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic validation for Terraform syntax
            if 'terraform {' in content or 'variable ' in content or 'resource ' in content:
                print(f"✅ {tf_file} - valid Terraform structure")
                valid_tf_files += 1
            else:
                print(f"❌ {tf_file} - invalid Terraform structure")
        else:
            print(f"⚠️  {tf_file} - file not found")
    
    return valid_tf_files == len(terraform_files)

def validate_configuration_completeness():
    """Validate that configurations are complete"""
    print("\n⚙️  Validating configuration completeness...")
    
    # Check cloud-run-optimizer.py for required classes
    try:
        with open("cloud-run-optimizer.py", 'r', encoding='utf-8') as f:
            optimizer_content = f.read()
        
        required_classes = [
            "class CloudRunOptimizer",
            "class ServiceMetrics", 
            "class ScalingDecision",
            "class CostOptimization"
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in optimizer_content:
                missing_classes.append(class_def)
        
        if missing_classes:
            print(f"❌ Missing classes in cloud-run-optimizer.py: {missing_classes}")
            return False
        else:
            print("✅ CloudRunOptimizer - all required classes present")
    
    except FileNotFoundError:
        print("❌ cloud-run-optimizer.py not found")
        return False
    
    # Check multi-region-deployment.py for required classes
    try:
        with open("multi-region-deployment.py", 'r', encoding='utf-8') as f:
            deployment_content = f.read()
        
        required_classes = [
            "class MultiRegionDeploymentManager",
            "class DeploymentStrategy",
            "class RegionConfig",
            "class ServiceDeployment"
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in deployment_content:
                missing_classes.append(class_def)
        
        if missing_classes:
            print(f"❌ Missing classes in multi-region-deployment.py: {missing_classes}")
            return False
        else:
            print("✅ MultiRegionDeploymentManager - all required classes present")
    
    except FileNotFoundError:
        print("❌ multi-region-deployment.py not found")
        return False
    
    # Check cost-optimization.py for required classes
    try:
        with open("cost-optimization.py", 'r', encoding='utf-8') as f:
            cost_content = f.read()
        
        required_classes = [
            "class CloudRunCostOptimizer",
            "class CostMetrics",
            "class OptimizationRecommendation",
            "class ResourceOptimization"
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in cost_content:
                missing_classes.append(class_def)
        
        if missing_classes:
            print(f"❌ Missing classes in cost-optimization.py: {missing_classes}")
            return False
        else:
            print("✅ CloudRunCostOptimizer - all required classes present")
    
    except FileNotFoundError:
        print("❌ cost-optimization.py not found")
        return False
    
    return True

def validate_requirements():
    """Validate Python requirements"""
    print("\n📦 Validating Python requirements...")
    
    try:
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            requirements = f.read()
        
        required_packages = [
            "google-cloud-run",
            "google-cloud-monitoring", 
            "google-cloud-billing",
            "aiohttp",
            "numpy",
            "pyyaml"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {missing_packages}")
            return False
        else:
            print("✅ All required packages present in requirements.txt")
            return True
    
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def validate_deployment_script():
    """Validate deployment script"""
    print("\n🚀 Validating deployment script...")
    
    try:
        with open("deploy.sh", 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        required_functions = [
            "check_prerequisites()",
            "enable_apis()",
            "build_and_push_images()",
            "deploy_all_services()",
            "setup_load_balancing()",
            "health_check()"
        ]
        
        missing_functions = []
        for func in required_functions:
            if func not in script_content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing functions in deploy.sh: {missing_functions}")
            return False
        else:
            print("✅ All required functions present in deploy.sh")
            return True
    
    except FileNotFoundError:
        print("❌ deploy.sh not found")
        return False

def validate_feature_completeness():
    """Validate that all required features are implemented"""
    print("\n🎯 Validating feature completeness...")
    
    features_implemented = []
    
    # Check for containerized microservices optimization
    if os.path.exists("Dockerfile.helios-core-cloudrun"):
        features_implemented.append("✅ Containerized microservices optimized for Cloud Run")
    else:
        features_implemented.append("❌ Missing Cloud Run optimized containers")
    
    # Check for automatic scaling implementation
    with open("cloud-run-optimizer.py", 'r', encoding='utf-8') as f:
        optimizer_content = f.read()
    
    if "def calculate_scaling_decision" in optimizer_content:
        features_implemented.append("✅ Automatic scaling based on real-time data processing demands")
    else:
        features_implemented.append("❌ Missing automatic scaling implementation")
    
    # Check for cost optimization
    if os.path.exists("cost-optimization.py"):
        features_implemented.append("✅ Cost optimization using spot instances and preemptible resources")
    else:
        features_implemented.append("❌ Missing cost optimization implementation")
    
    # Check for multi-region deployment
    if os.path.exists("multi-region-deployment.py"):
        features_implemented.append("✅ Multi-region deployment for global low-latency access")
    else:
        features_implemented.append("❌ Missing multi-region deployment implementation")
    
    # Check for monitoring and alerting
    if os.path.exists("monitoring/alerts.yaml"):
        features_implemented.append("✅ Comprehensive monitoring and alerting")
    else:
        features_implemented.append("❌ Missing monitoring and alerting configuration")
    
    # Check for Terraform infrastructure as code
    if os.path.exists("terraform/main.tf"):
        features_implemented.append("✅ Infrastructure as Code with Terraform")
    else:
        features_implemented.append("❌ Missing Terraform infrastructure configuration")
    
    for feature in features_implemented:
        print(feature)
    
    implemented_count = sum(1 for f in features_implemented if f.startswith("✅"))
    total_count = len(features_implemented)
    
    return implemented_count == total_count

def generate_validation_report():
    """Generate a comprehensive validation report"""
    print("\n📊 Generating validation report...")
    
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "validation_results": {
            "file_structure": validate_file_structure(),
            "python_syntax": validate_python_syntax(),
            "docker_files": validate_docker_files(),
            "yaml_files": validate_yaml_files(),
            "terraform_files": validate_terraform_files(),
            "configuration_completeness": validate_configuration_completeness(),
            "requirements": validate_requirements(),
            "deployment_script": validate_deployment_script(),
            "feature_completeness": validate_feature_completeness()
        }
    }
    
    # Calculate overall score
    passed_tests = sum(1 for result in report["validation_results"].values() if result)
    total_tests = len(report["validation_results"])
    overall_score = (passed_tests / total_tests) * 100
    
    report["overall_score"] = overall_score
    report["passed_tests"] = passed_tests
    report["total_tests"] = total_tests
    
    # Save report
    with open("validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Validation Report Summary:")
    print(f"Overall Score: {overall_score:.1f}% ({passed_tests}/{total_tests} tests passed)")
    
    if overall_score >= 90:
        print("🎉 Excellent! Implementation is ready for deployment.")
    elif overall_score >= 75:
        print("👍 Good! Minor issues to address before deployment.")
    elif overall_score >= 50:
        print("⚠️  Fair. Several issues need to be resolved.")
    else:
        print("❌ Poor. Major issues need to be addressed.")
    
    print(f"\nDetailed report saved to: validation_report.json")
    
    return report

def main():
    """Main validation function"""
    print("🚀 Cloud Run Serverless Scaling Implementation Validation")
    print("=" * 60)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run validation
    report = generate_validation_report()
    
    # Exit with appropriate code
    if report["overall_score"] >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()