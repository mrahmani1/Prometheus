#!/usr/bin/env python3
"""
Database Stack Validation Script
This script performs comprehensive syntax and configuration validation
for the Prometheus AI Platform database stack deployment.
"""

import os
import sys
import yaml
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseStackValidator:
    """Comprehensive validator for database stack configuration."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        
    def validate_yaml_files(self) -> bool:
        """Validate all YAML files for syntax errors."""
        logger.info("Validating YAML files...")
        
        yaml_files = [
            "kubernetes/database-statefulsets.yaml",
            "kubernetes/database-services.yaml",
            "kubernetes/database-storage.yaml",
            "kubernetes/database-backup-cronjobs.yaml",
            "kubernetes/database-monitoring.yaml",
            "kubernetes/database-service-mesh.yaml",
            "kubernetes/database-scaling-policies.yaml",
            "helm/database-stack/Chart.yaml",
            "helm/database-stack/values.yaml",
            "docker-compose.databases.yml"
        ]
        
        valid = True
        for yaml_file in yaml_files:
            file_path = self.base_path / yaml_file
            if not file_path.exists():
                self.errors.append(f"YAML file not found: {yaml_file}")
                valid = False
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Handle multi-document YAML files
                    documents = list(yaml.safe_load_all(f))
                    if not documents or all(doc is None for doc in documents):
                        self.warnings.append(f"Empty YAML file: {yaml_file}")
                    else:
                        logger.info(f"✓ Valid YAML: {yaml_file}")
            except yaml.YAMLError as e:
                self.errors.append(f"YAML syntax error in {yaml_file}: {e}")
                valid = False
            except Exception as e:
                self.errors.append(f"Error reading {yaml_file}: {e}")
                valid = False
                
        return valid
    
    def validate_kubernetes_manifests(self) -> bool:
        """Validate Kubernetes manifests using kubectl dry-run."""
        logger.info("Validating Kubernetes manifests...")
        
        k8s_files = [
            "kubernetes/database-statefulsets.yaml",
            "kubernetes/database-services.yaml",
            "kubernetes/database-storage.yaml",
            "kubernetes/database-backup-cronjobs.yaml",
            "kubernetes/database-monitoring.yaml",
            "kubernetes/database-scaling-policies.yaml"
        ]
        
        valid = True
        for k8s_file in k8s_files:
            file_path = self.base_path / k8s_file
            if not file_path.exists():
                continue
                
            try:
                # Use kubectl to validate the manifest
                result = subprocess.run([
                    'kubectl', 'apply', '--dry-run=client', '--validate=true',
                    '-f', str(file_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info(f"✓ Valid Kubernetes manifest: {k8s_file}")
                else:
                    self.errors.append(f"Kubernetes validation failed for {k8s_file}: {result.stderr}")
                    valid = False
                    
            except subprocess.TimeoutExpired:
                self.warnings.append(f"Kubernetes validation timeout for {k8s_file}")
            except FileNotFoundError:
                self.warnings.append("kubectl not found, skipping Kubernetes validation")
                break
            except Exception as e:
                self.errors.append(f"Error validating {k8s_file}: {e}")
                valid = False
                
        return valid
    
    def validate_helm_chart(self) -> bool:
        """Validate Helm chart structure and templates."""
        logger.info("Validating Helm chart...")
        
        chart_path = self.base_path / "helm/database-stack"
        if not chart_path.exists():
            self.errors.append("Helm chart directory not found")
            return False
            
        # Check required files
        required_files = ["Chart.yaml", "values.yaml"]
        for req_file in required_files:
            if not (chart_path / req_file).exists():
                self.errors.append(f"Required Helm file missing: {req_file}")
                return False
                
        try:
            # Validate Chart.yaml structure
            with open(chart_path / "Chart.yaml", 'r') as f:
                chart_data = yaml.safe_load(f)
                required_fields = ["apiVersion", "name", "version"]
                for field in required_fields:
                    if field not in chart_data:
                        self.errors.append(f"Missing required field in Chart.yaml: {field}")
                        return False
                        
            # Validate values.yaml structure
            with open(chart_path / "values.yaml", 'r') as f:
                values_data = yaml.safe_load(f)
                if not isinstance(values_data, dict):
                    self.errors.append("values.yaml must contain a dictionary")
                    return False
                    
            # Use helm to validate the chart if available
            try:
                result = subprocess.run([
                    'helm', 'lint', str(chart_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info("✓ Valid Helm chart")
                else:
                    self.warnings.append(f"Helm lint warnings: {result.stdout}")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.warnings.append("helm not found, skipping Helm validation")
                
        except Exception as e:
            self.errors.append(f"Error validating Helm chart: {e}")
            return False
            
        return True
    
    def validate_docker_compose(self) -> bool:
        """Validate Docker Compose configuration."""
        logger.info("Validating Docker Compose configuration...")
        
        compose_file = self.base_path / "docker-compose.databases.yml"
        if not compose_file.exists():
            self.errors.append("Docker Compose file not found")
            return False
            
        try:
            # Use docker-compose to validate if available
            result = subprocess.run([
                'docker-compose', '-f', str(compose_file), 'config'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✓ Valid Docker Compose configuration")
                return True
            else:
                self.errors.append(f"Docker Compose validation failed: {result.stderr}")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("docker-compose not found, skipping Docker Compose validation")
            return True
        except Exception as e:
            self.errors.append(f"Error validating Docker Compose: {e}")
            return False
    
    def validate_sql_scripts(self) -> bool:
        """Validate SQL initialization scripts."""
        logger.info("Validating SQL scripts...")
        
        sql_files = [
            "database/postgres/init/01-extensions.sql",
            "database/postgres/init/02-geological-schema.sql",
            "database/timescaledb/init/01-timescale-setup.sql"
        ]
        
        valid = True
        for sql_file in sql_files:
            file_path = self.base_path / sql_file
            if not file_path.exists():
                self.errors.append(f"SQL file not found: {sql_file}")
                valid = False
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Basic SQL syntax validation
                if not content.strip():
                    self.warnings.append(f"Empty SQL file: {sql_file}")
                    continue
                    
                # Check for common SQL syntax issues
                if content.count('(') != content.count(')'):
                    self.errors.append(f"Unmatched parentheses in {sql_file}")
                    valid = False
                    
                if content.count("'") % 2 != 0:
                    self.errors.append(f"Unmatched quotes in {sql_file}")
                    valid = False
                    
                # Check for required PostgreSQL/TimescaleDB patterns
                if "postgres" in sql_file.lower():
                    if "CREATE EXTENSION" not in content:
                        self.warnings.append(f"No extensions created in {sql_file}")
                        
                if "timescale" in sql_file.lower():
                    if "create_hypertable" not in content:
                        self.warnings.append(f"No hypertables created in {sql_file}")
                        
                logger.info(f"✓ Valid SQL script: {sql_file}")
                
            except Exception as e:
                self.errors.append(f"Error validating SQL file {sql_file}: {e}")
                valid = False
                
        return valid
    
    def validate_shell_scripts(self) -> bool:
        """Validate shell scripts for syntax errors."""
        logger.info("Validating shell scripts...")
        
        shell_scripts = [
            "scripts/deploy-database-stack.sh"
        ]
        
        valid = True
        for script in shell_scripts:
            script_path = self.base_path / script
            if not script_path.exists():
                self.errors.append(f"Shell script not found: {script}")
                valid = False
                continue
                
            try:
                # Use bash to check syntax
                result = subprocess.run([
                    'bash', '-n', str(script_path)
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    logger.info(f"✓ Valid shell script: {script}")
                else:
                    self.errors.append(f"Shell script syntax error in {script}: {result.stderr}")
                    valid = False
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.warnings.append("bash not found, skipping shell script validation")
                break
            except Exception as e:
                self.errors.append(f"Error validating shell script {script}: {e}")
                valid = False
                
        return valid
    
    def validate_powershell_scripts(self) -> bool:
        """Validate PowerShell scripts for syntax errors."""
        logger.info("Validating PowerShell scripts...")
        
        ps_scripts = [
            "scripts/deploy-database-stack.ps1"
        ]
        
        valid = True
        for script in ps_scripts:
            script_path = self.base_path / script
            if not script_path.exists():
                self.errors.append(f"PowerShell script not found: {script}")
                valid = False
                continue
                
            try:
                # Use PowerShell to check syntax (if available)
                result = subprocess.run([
                    'pwsh', '-NoProfile', '-Command', f'Get-Content "{script_path}" | Out-Null'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    logger.info(f"✓ Valid PowerShell script: {script}")
                else:
                    self.errors.append(f"PowerShell script error in {script}: {result.stderr}")
                    valid = False
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Try with powershell.exe on Windows
                try:
                    result = subprocess.run([
                        'powershell.exe', '-NoProfile', '-Command', f'Get-Content "{script_path}" | Out-Null'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        logger.info(f"✓ Valid PowerShell script: {script}")
                    else:
                        self.errors.append(f"PowerShell script error in {script}: {result.stderr}")
                        valid = False
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    self.warnings.append("PowerShell not found, skipping PowerShell script validation")
                    break
            except Exception as e:
                self.errors.append(f"Error validating PowerShell script {script}: {e}")
                valid = False
                
        return valid
    
    def validate_configuration_consistency(self) -> bool:
        """Validate consistency across configuration files."""
        logger.info("Validating configuration consistency...")
        
        valid = True
        
        try:
            # Check namespace consistency
            namespaces = set()
            
            # Check Kubernetes manifests
            k8s_files = [
                "kubernetes/database-statefulsets.yaml",
                "kubernetes/database-services.yaml",
                "kubernetes/database-storage.yaml"
            ]
            
            for k8s_file in k8s_files:
                file_path = self.base_path / k8s_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        documents = list(yaml.safe_load_all(f))
                        for doc in documents:
                            if doc and 'metadata' in doc and 'namespace' in doc['metadata']:
                                namespaces.add(doc['metadata']['namespace'])
                                
            if len(namespaces) > 1:
                self.errors.append(f"Inconsistent namespaces found: {namespaces}")
                valid = False
            elif namespaces:
                logger.info(f"✓ Consistent namespace: {list(namespaces)[0]}")
                
            # Check service names consistency
            services = set()
            statefulsets = set()
            
            # Extract service names
            services_file = self.base_path / "kubernetes/database-services.yaml"
            if services_file.exists():
                with open(services_file, 'r') as f:
                    documents = list(yaml.safe_load_all(f))
                    for doc in documents:
                        if doc and doc.get('kind') == 'Service':
                            services.add(doc['metadata']['name'])
                            
            # Extract StatefulSet names
            statefulsets_file = self.base_path / "kubernetes/database-statefulsets.yaml"
            if statefulsets_file.exists():
                with open(statefulsets_file, 'r') as f:
                    documents = list(yaml.safe_load_all(f))
                    for doc in documents:
                        if doc and doc.get('kind') == 'StatefulSet':
                            statefulsets.add(doc['metadata']['name'])
                            
            # Check if services match StatefulSets
            missing_services = statefulsets - services
            if missing_services:
                self.warnings.append(f"StatefulSets without matching services: {missing_services}")
                
            logger.info("✓ Configuration consistency validated")
            
        except Exception as e:
            self.errors.append(f"Error validating configuration consistency: {e}")
            valid = False
            
        return valid
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting comprehensive database stack validation...")
        
        validations = [
            self.validate_yaml_files,
            self.validate_kubernetes_manifests,
            self.validate_helm_chart,
            self.validate_docker_compose,
            self.validate_sql_scripts,
            self.validate_shell_scripts,
            self.validate_powershell_scripts,
            self.validate_configuration_consistency
        ]
        
        all_valid = True
        for validation in validations:
            try:
                if not validation():
                    all_valid = False
            except Exception as e:
                self.errors.append(f"Validation error: {e}")
                all_valid = False
                
        return all_valid
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "="*60)
        print("DATABASE STACK VALIDATION RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        if not self.errors and not self.warnings:
            print("\n✅ ALL VALIDATIONS PASSED!")
        elif not self.errors:
            print(f"\n✅ VALIDATION PASSED WITH {len(self.warnings)} WARNINGS")
        else:
            print(f"\n❌ VALIDATION FAILED WITH {len(self.errors)} ERRORS AND {len(self.warnings)} WARNINGS")
            
        print("="*60)

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Prometheus AI Platform database stack")
    parser.add_argument("--path", default=".", help="Base path for validation (default: current directory)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    validator = DatabaseStackValidator(args.path)
    success = validator.run_all_validations()
    validator.print_results()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()