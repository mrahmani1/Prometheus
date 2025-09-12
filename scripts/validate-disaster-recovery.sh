#!/bin/bash

# Prometheus AI Platform - Disaster Recovery Validation Script
# Comprehensive validation script for disaster recovery implementation

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation results
VALIDATIONS_PASSED=0
VALIDATIONS_FAILED=0
VALIDATION_RESULTS=()

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validation result functions
validation_passed() {
    local validation_name="$1"
    VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1))
    VALIDATION_RESULTS+=("✅ $validation_name")
    log_success "$validation_name"
}

validation_failed() {
    local validation_name="$1"
    local error_msg="$2"
    VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1))
    VALIDATION_RESULTS+=("❌ $validation_name: $error_msg")
    log_error "$validation_name: $error_msg"
}

validation_warning() {
    local validation_name="$1"
    local warning_msg="$2"
    VALIDATION_RESULTS+=("⚠️  $validation_name: $warning_msg")
    log_warning "$validation_name: $warning_msg"
}

# Validate file structure
validate_file_structure() {
    log_info "Validating file structure..."
    
    local required_files=(
        "services/disaster-recovery/main.py"
        "services/disaster-recovery/models/disaster_recovery_models.py"
        "services/disaster-recovery/core/backup_manager.py"
        "services/disaster-recovery/core/replication_engine.py"
        "services/disaster-recovery/core/recovery_orchestrator.py"
        "services/disaster-recovery/core/continuity_planner.py"
        "services/disaster-recovery/core/monitoring_system.py"
        "services/disaster-recovery/Dockerfile"
        "services/disaster-recovery/requirements.txt"
        "services/disaster-recovery/README.md"
        "kubernetes/disaster-recovery-deployment.yaml"
        "kubernetes/disaster-recovery-configmap.yaml"
        "scripts/disaster-recovery-deploy.sh"
        "scripts/disaster-recovery-test.sh"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        validation_passed "File structure validation"
    else
        validation_failed "File structure validation" "Missing files: ${missing_files[*]}"
    fi
}

# Validate Python syntax
validate_python_syntax() {
    log_info "Validating Python syntax..."
    
    local python_files=(
        "services/disaster-recovery/main.py"
        "services/disaster-recovery/models/disaster_recovery_models.py"
        "services/disaster-recovery/core/backup_manager.py"
        "services/disaster-recovery/core/replication_engine.py"
        "services/disaster-recovery/core/recovery_orchestrator.py"
        "services/disaster-recovery/core/continuity_planner.py"
        "services/disaster-recovery/core/monitoring_system.py"
    )
    
    local syntax_errors=()
    
    for file in "${python_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            if ! python3 -m py_compile "$PROJECT_ROOT/$file" 2>/dev/null; then
                syntax_errors+=("$file")
            fi
        fi
    done
    
    if [[ ${#syntax_errors[@]} -eq 0 ]]; then
        validation_passed "Python syntax validation"
    else
        validation_failed "Python syntax validation" "Syntax errors in: ${syntax_errors[*]}"
    fi
}

# Validate Kubernetes manifests
validate_kubernetes_manifests() {
    log_info "Validating Kubernetes manifests..."
    
    local k8s_files=(
        "kubernetes/disaster-recovery-deployment.yaml"
        "kubernetes/disaster-recovery-configmap.yaml"
    )
    
    local invalid_manifests=()
    
    for file in "${k8s_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            if ! kubectl apply --dry-run=client -f "$PROJECT_ROOT/$file" &>/dev/null; then
                invalid_manifests+=("$file")
            fi
        fi
    done
    
    if [[ ${#invalid_manifests[@]} -eq 0 ]]; then
        validation_passed "Kubernetes manifests validation"
    else
        validation_failed "Kubernetes manifests validation" "Invalid manifests: ${invalid_manifests[*]}"
    fi
}

# Validate Docker configuration
validate_docker_configuration() {
    log_info "Validating Docker configuration..."
    
    local dockerfile_path="$PROJECT_ROOT/services/disaster-recovery/Dockerfile"
    local requirements_path="$PROJECT_ROOT/services/disaster-recovery/requirements.txt"
    
    # Check Dockerfile exists and has required components
    if [[ -f "$dockerfile_path" ]]; then
        if grep -q "FROM python:" "$dockerfile_path" && \
           grep -q "COPY requirements.txt" "$dockerfile_path" && \
           grep -q "RUN pip install" "$dockerfile_path" && \
           grep -q "EXPOSE 8080" "$dockerfile_path"; then
            validation_passed "Dockerfile structure validation"
        else
            validation_failed "Dockerfile structure validation" "Missing required Dockerfile components"
        fi
    else
        validation_failed "Dockerfile validation" "Dockerfile not found"
    fi
    
    # Check requirements.txt exists and has key dependencies
    if [[ -f "$requirements_path" ]]; then
        if grep -q "fastapi" "$requirements_path" && \
           grep -q "uvicorn" "$requirements_path" && \
           grep -q "psycopg2" "$requirements_path" && \
           grep -q "structlog" "$requirements_path"; then
            validation_passed "Requirements.txt validation"
        else
            validation_failed "Requirements.txt validation" "Missing key dependencies"
        fi
    else
        validation_failed "Requirements.txt validation" "requirements.txt not found"
    fi
}

# Validate configuration completeness
validate_configuration_completeness() {
    log_info "Validating configuration completeness..."
    
    local configmap_file="$PROJECT_ROOT/kubernetes/disaster-recovery-configmap.yaml"
    
    if [[ -f "$configmap_file" ]]; then
        local required_configs=(
            "backup-configs.json"
            "replication-configs.json"
            "recovery-plans.json"
            "continuity-plans.json"
            "archival-policies.json"
            "dr-monitoring-config.json"
        )
        
        local missing_configs=()
        
        for config in "${required_configs[@]}"; do
            if ! grep -q "$config" "$configmap_file"; then
                missing_configs+=("$config")
            fi
        done
        
        if [[ ${#missing_configs[@]} -eq 0 ]]; then
            validation_passed "Configuration completeness validation"
        else
            validation_failed "Configuration completeness validation" "Missing configs: ${missing_configs[*]}"
        fi
    else
        validation_failed "Configuration completeness validation" "ConfigMap file not found"
    fi
}

# Validate API endpoint coverage
validate_api_endpoint_coverage() {
    log_info "Validating API endpoint coverage..."
    
    local main_py="$PROJECT_ROOT/services/disaster-recovery/main.py"
    
    if [[ -f "$main_py" ]]; then
        local required_endpoints=(
            "/health"
            "/ready"
            "/api/v1/backup/create"
            "/api/v1/backup/status"
            "/api/v1/backup/list"
            "/api/v1/replication/configure"
            "/api/v1/replication/status"
            "/api/v1/replication/failover"
            "/api/v1/recovery/plan"
            "/api/v1/recovery/execute"
            "/api/v1/recovery/status"
            "/api/v1/continuity/plan"
            "/api/v1/continuity/assessment"
            "/api/v1/metrics/rto-rpo"
            "/api/v1/metrics/backup-integrity"
            "/api/v1/test/disaster-recovery"
        )
        
        local missing_endpoints=()
        
        for endpoint in "${required_endpoints[@]}"; do
            if ! grep -q "$endpoint" "$main_py"; then
                missing_endpoints+=("$endpoint")
            fi
        done
        
        if [[ ${#missing_endpoints[@]} -eq 0 ]]; then
            validation_passed "API endpoint coverage validation"
        else
            validation_failed "API endpoint coverage validation" "Missing endpoints: ${missing_endpoints[*]}"
        fi
    else
        validation_failed "API endpoint coverage validation" "main.py not found"
    fi
}

# Validate data model completeness
validate_data_model_completeness() {
    log_info "Validating data model completeness..."
    
    local models_file="$PROJECT_ROOT/services/disaster-recovery/models/disaster_recovery_models.py"
    
    if [[ -f "$models_file" ]]; then
        local required_models=(
            "BackupConfiguration"
            "BackupMetadata"
            "ReplicationConfiguration"
            "ReplicationStatus"
            "RecoveryPlan"
            "RecoveryExecution"
            "BusinessContinuityPlan"
            "DisasterRecoveryMetrics"
            "DataArchivalPolicy"
            "IntegrityCheckResult"
            "DisasterRecoveryTest"
        )
        
        local missing_models=()
        
        for model in "${required_models[@]}"; do
            if ! grep -q "class $model" "$models_file"; then
                missing_models+=("$model")
            fi
        done
        
        if [[ ${#missing_models[@]} -eq 0 ]]; then
            validation_passed "Data model completeness validation"
        else
            validation_failed "Data model completeness validation" "Missing models: ${missing_models[*]}"
        fi
    else
        validation_failed "Data model completeness validation" "Models file not found"
    fi
}

# Validate core component implementation
validate_core_component_implementation() {
    log_info "Validating core component implementation..."
    
    local components=(
        "backup_manager.py:BackupManager"
        "replication_engine.py:ReplicationEngine"
        "recovery_orchestrator.py:RecoveryOrchestrator"
        "continuity_planner.py:ContinuityPlanner"
        "monitoring_system.py:DisasterRecoveryMonitoring"
    )
    
    local missing_components=()
    
    for component in "${components[@]}"; do
        local file_name="${component%:*}"
        local class_name="${component#*:}"
        local file_path="$PROJECT_ROOT/services/disaster-recovery/core/$file_name"
        
        if [[ -f "$file_path" ]]; then
            if ! grep -q "class $class_name" "$file_path"; then
                missing_components+=("$class_name in $file_name")
            fi
        else
            missing_components+=("$file_name")
        fi
    done
    
    if [[ ${#missing_components[@]} -eq 0 ]]; then
        validation_passed "Core component implementation validation"
    else
        validation_failed "Core component implementation validation" "Missing components: ${missing_components[*]}"
    fi
}

# Validate security configuration
validate_security_configuration() {
    log_info "Validating security configuration..."
    
    local deployment_file="$PROJECT_ROOT/kubernetes/disaster-recovery-deployment.yaml"
    
    if [[ -f "$deployment_file" ]]; then
        local security_checks=()
        
        # Check for security context
        if ! grep -q "securityContext" "$deployment_file"; then
            security_checks+=("Missing securityContext")
        fi
        
        # Check for non-root user
        if ! grep -q "runAsNonRoot: true" "$deployment_file"; then
            security_checks+=("Missing runAsNonRoot")
        fi
        
        # Check for service account
        if ! grep -q "serviceAccountName" "$deployment_file"; then
            security_checks+=("Missing serviceAccountName")
        fi
        
        # Check for RBAC
        if ! grep -q "ClusterRole" "$deployment_file"; then
            security_checks+=("Missing RBAC configuration")
        fi
        
        if [[ ${#security_checks[@]} -eq 0 ]]; then
            validation_passed "Security configuration validation"
        else
            validation_warning "Security configuration validation" "Issues: ${security_checks[*]}"
        fi
    else
        validation_failed "Security configuration validation" "Deployment file not found"
    fi
}

# Validate monitoring and observability
validate_monitoring_observability() {
    log_info "Validating monitoring and observability..."
    
    local main_py="$PROJECT_ROOT/services/disaster-recovery/main.py"
    local monitoring_py="$PROJECT_ROOT/services/disaster-recovery/core/monitoring_system.py"
    
    local monitoring_features=()
    
    # Check for health endpoints
    if [[ -f "$main_py" ]] && grep -q "/health" "$main_py" && grep -q "/ready" "$main_py"; then
        monitoring_features+=("health_endpoints")
    fi
    
    # Check for metrics endpoints
    if [[ -f "$main_py" ]] && grep -q "/metrics" "$main_py"; then
        monitoring_features+=("metrics_endpoints")
    fi
    
    # Check for structured logging
    if [[ -f "$main_py" ]] && grep -q "structlog" "$main_py"; then
        monitoring_features+=("structured_logging")
    fi
    
    # Check for monitoring system
    if [[ -f "$monitoring_py" ]] && grep -q "DisasterRecoveryMonitoring" "$monitoring_py"; then
        monitoring_features+=("monitoring_system")
    fi
    
    if [[ ${#monitoring_features[@]} -ge 3 ]]; then
        validation_passed "Monitoring and observability validation"
    else
        validation_warning "Monitoring and observability validation" "Limited monitoring features: ${monitoring_features[*]}"
    fi
}

# Validate deployment scripts
validate_deployment_scripts() {
    log_info "Validating deployment scripts..."
    
    local deploy_script="$PROJECT_ROOT/scripts/disaster-recovery-deploy.sh"
    local test_script="$PROJECT_ROOT/scripts/disaster-recovery-test.sh"
    
    local script_issues=()
    
    # Check deploy script
    if [[ -f "$deploy_script" ]]; then
        if ! grep -q "kubectl apply" "$deploy_script"; then
            script_issues+=("Deploy script missing kubectl apply")
        fi
        if ! grep -q "docker build" "$deploy_script"; then
            script_issues+=("Deploy script missing docker build")
        fi
    else
        script_issues+=("Deploy script not found")
    fi
    
    # Check test script
    if [[ -f "$test_script" ]]; then
        if ! grep -q "curl" "$test_script"; then
            script_issues+=("Test script missing API tests")
        fi
    else
        script_issues+=("Test script not found")
    fi
    
    if [[ ${#script_issues[@]} -eq 0 ]]; then
        validation_passed "Deployment scripts validation"
    else
        validation_failed "Deployment scripts validation" "Issues: ${script_issues[*]}"
    fi
}

# Validate documentation
validate_documentation() {
    log_info "Validating documentation..."
    
    local readme_file="$PROJECT_ROOT/services/disaster-recovery/README.md"
    
    if [[ -f "$readme_file" ]]; then
        local doc_sections=(
            "## Overview"
            "## Features"
            "## Architecture"
            "## Quick Start"
            "## API Reference"
            "## Configuration"
            "## Monitoring"
            "## Testing"
            "## Security"
            "## Troubleshooting"
        )
        
        local missing_sections=()
        
        for section in "${doc_sections[@]}"; do
            if ! grep -q "$section" "$readme_file"; then
                missing_sections+=("$section")
            fi
        done
        
        if [[ ${#missing_sections[@]} -eq 0 ]]; then
            validation_passed "Documentation validation"
        else
            validation_warning "Documentation validation" "Missing sections: ${missing_sections[*]}"
        fi
    else
        validation_failed "Documentation validation" "README.md not found"
    fi
}

# Validate task completion
validate_task_completion() {
    log_info "Validating task completion against requirements..."
    
    # Task 7.4 requirements from the specification
    local requirements=(
        "automated_backup_strategies"
        "cross_region_replication"
        "point_in_time_recovery"
        "disaster_recovery_testing"
        "business_continuity_planning"
        "automated_failover"
        "data_archival_policies"
        "backup_verification"
        "rto_rpo_monitoring"
        "geo_distributed_storage"
    )
    
    local implemented_features=()
    local missing_features=()
    
    # Check for automated backup strategies
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py" ]] && \
       grep -q "execute_scheduled_backups" "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py"; then
        implemented_features+=("automated_backup_strategies")
    else
        missing_features+=("automated_backup_strategies")
    fi
    
    # Check for cross-region replication
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/replication_engine.py" ]] && \
       grep -q "target_regions" "$PROJECT_ROOT/services/disaster-recovery/core/replication_engine.py"; then
        implemented_features+=("cross_region_replication")
    else
        missing_features+=("cross_region_replication")
    fi
    
    # Check for point-in-time recovery
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/recovery_orchestrator.py" ]] && \
       grep -q "point_in_time_recovery" "$PROJECT_ROOT/services/disaster-recovery/core/recovery_orchestrator.py"; then
        implemented_features+=("point_in_time_recovery")
    else
        missing_features+=("point_in_time_recovery")
    fi
    
    # Check for disaster recovery testing
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/monitoring_system.py" ]] && \
       grep -q "DisasterRecoveryTest" "$PROJECT_ROOT/services/disaster-recovery/core/monitoring_system.py"; then
        implemented_features+=("disaster_recovery_testing")
    else
        missing_features+=("disaster_recovery_testing")
    fi
    
    # Check for business continuity planning
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/continuity_planner.py" ]] && \
       grep -q "BusinessContinuityPlan" "$PROJECT_ROOT/services/disaster-recovery/core/continuity_planner.py"; then
        implemented_features+=("business_continuity_planning")
    else
        missing_features+=("business_continuity_planning")
    fi
    
    # Check for automated failover
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/replication_engine.py" ]] && \
       grep -q "auto_failover" "$PROJECT_ROOT/services/disaster-recovery/core/replication_engine.py"; then
        implemented_features+=("automated_failover")
    else
        missing_features+=("automated_failover")
    fi
    
    # Check for data archival policies
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/continuity_planner.py" ]] && \
       grep -q "DataArchivalPolicy" "$PROJECT_ROOT/services/disaster-recovery/core/continuity_planner.py"; then
        implemented_features+=("data_archival_policies")
    else
        missing_features+=("data_archival_policies")
    fi
    
    # Check for backup verification
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py" ]] && \
       grep -q "verify_backup_integrity" "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py"; then
        implemented_features+=("backup_verification")
    else
        missing_features+=("backup_verification")
    fi
    
    # Check for RTO/RPO monitoring
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/monitoring_system.py" ]] && \
       grep -q "rto_rpo_metrics" "$PROJECT_ROOT/services/disaster-recovery/core/monitoring_system.py"; then
        implemented_features+=("rto_rpo_monitoring")
    else
        missing_features+=("rto_rpo_monitoring")
    fi
    
    # Check for geo-distributed storage
    if [[ -f "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py" ]] && \
       grep -q "storage_locations" "$PROJECT_ROOT/services/disaster-recovery/core/backup_manager.py"; then
        implemented_features+=("geo_distributed_storage")
    else
        missing_features+=("geo_distributed_storage")
    fi
    
    local completion_rate=$(( ${#implemented_features[@]} * 100 / ${#requirements[@]} ))
    
    if [[ $completion_rate -eq 100 ]]; then
        validation_passed "Task completion validation (100%)"
    elif [[ $completion_rate -ge 80 ]]; then
        validation_warning "Task completion validation" "$completion_rate% complete, missing: ${missing_features[*]}"
    else
        validation_failed "Task completion validation" "$completion_rate% complete, missing: ${missing_features[*]}"
    fi
}

# Generate validation report
generate_validation_report() {
    local total_validations=$((VALIDATIONS_PASSED + VALIDATIONS_FAILED))
    local success_rate=0
    
    if [[ $total_validations -gt 0 ]]; then
        success_rate=$(( (VALIDATIONS_PASSED * 100) / total_validations ))
    fi
    
    echo ""
    echo "=========================================="
    echo "    DISASTER RECOVERY VALIDATION REPORT"
    echo "=========================================="
    echo "Total Validations: $total_validations"
    echo "Passed: $VALIDATIONS_PASSED"
    echo "Failed: $VALIDATIONS_FAILED"
    echo "Success Rate: $success_rate%"
    echo ""
    
    if [[ ${#VALIDATION_RESULTS[@]} -gt 0 ]]; then
        echo "Validation Results:"
        printf '%s\n' "${VALIDATION_RESULTS[@]}"
        echo ""
    fi
    
    if [[ $VALIDATIONS_FAILED -eq 0 ]]; then
        log_success "All validations passed! 🎉"
        log_info "The disaster recovery implementation is complete and ready for deployment."
        return 0
    else
        log_error "$VALIDATIONS_FAILED validation(s) failed"
        log_info "Please address the failed validations before deployment."
        return 1
    fi
}

# Main validation function
main() {
    log_info "Starting disaster recovery implementation validation..."
    
    # Run all validations
    validate_file_structure
    validate_python_syntax
    validate_kubernetes_manifests
    validate_docker_configuration
    validate_configuration_completeness
    validate_api_endpoint_coverage
    validate_data_model_completeness
    validate_core_component_implementation
    validate_security_configuration
    validate_monitoring_observability
    validate_deployment_scripts
    validate_documentation
    validate_task_completion
    
    # Generate report
    generate_validation_report
}

# Run main function
main "$@"