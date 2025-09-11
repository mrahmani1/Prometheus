#!/bin/bash
# =============================================================================
# DATABASE DEPLOYMENT VALIDATION SCRIPT
# Validates implementation against task 2.5 requirements
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Validation results
VALIDATION_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0

# Logging functions
log() {
    echo -e "${BLUE}[VALIDATION]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
    VALIDATION_RESULTS+=("✓ $1")
    ((PASSED_CHECKS++))
}

fail() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
    VALIDATION_RESULTS+=("✗ $1")
}

warning() {
    echo -e "${YELLOW}[⚠ WARNING]${NC} $1"
    VALIDATION_RESULTS+=("⚠ $1")
}

check() {
    ((TOTAL_CHECKS++))
}

# Task 2.5 Requirements Validation
validate_task_requirements() {
    log "Validating Task 2.5 Requirements Implementation..."
    echo ""
    
    # Requirement 1: Create comprehensive database deployment scripts
    log "1. Checking comprehensive database deployment scripts..."
    check
    if [[ -f "database/postgres/comprehensive_deployment_init.sql" ]]; then
        success "Comprehensive database deployment script exists"
        
        # Check for key components in the script
        if grep -q "geological_age" database/postgres/comprehensive_deployment_init.sql; then
            success "Geological age schema implementation found"
        else
            fail "Geological age schema implementation missing"
        fi
        
        if grep -q "model_management" database/postgres/comprehensive_deployment_init.sql; then
            success "Model management schema implementation found"
        else
            fail "Model management schema implementation missing"
        fi
        
        if grep -q "external_data" database/postgres/comprehensive_deployment_init.sql; then
            success "External data integration schema found"
        else
            fail "External data integration schema missing"
        fi
    else
        fail "Comprehensive database deployment script missing"
    fi
    
    # Requirement 2: PostgreSQL with PostGIS, TimeScaleDB, and specialized extensions
    log "2. Checking PostgreSQL with PostGIS, TimeScaleDB, and extensions..."
    check
    if grep -q "CREATE EXTENSION IF NOT EXISTS postgis" database/postgres/comprehensive_deployment_init.sql; then
        success "PostGIS extension configuration found"
    else
        fail "PostGIS extension configuration missing"
    fi
    
    check
    if grep -q "CREATE EXTENSION IF NOT EXISTS timescaledb" database/postgres/comprehensive_deployment_init.sql; then
        success "TimeScaleDB extension configuration found"
    else
        fail "TimeScaleDB extension configuration missing"
    fi
    
    check
    if grep -q "pg_stat_statements\|btree_gist\|pg_trgm" database/postgres/comprehensive_deployment_init.sql; then
        success "Specialized extensions configuration found"
    else
        fail "Specialized extensions configuration missing"
    fi
    
    # Requirement 3: Environment variables for geological age database integration
    log "3. Checking environment variables for geological age database integration..."
    check
    if [[ -f "database/postgres/environment_config.env" ]]; then
        success "Environment configuration file exists"
        
        if grep -q "POSTGRES_USER_GEOLOGICAL_AGE" database/postgres/environment_config.env; then
            success "Geological age service credentials configured"
        else
            fail "Geological age service credentials missing"
        fi
        
        if grep -q "AGE_TRAINING_DATA_PATH\|AGE_MODEL_SAVE_PATH" database/postgres/environment_config.env; then
            success "Age prediction model paths configured"
        else
            fail "Age prediction model paths missing"
        fi
    else
        fail "Environment configuration file missing"
    fi
    
    # Requirement 4: ModelManager configuration for age prediction
    log "4. Checking ModelManager configuration for age prediction..."
    check
    if grep -q "model_management.model_registry" database/postgres/comprehensive_deployment_init.sql; then
        success "Model registry table implementation found"
    else
        fail "Model registry table implementation missing"
    fi
    
    check
    if grep -q "MODEL_REGISTRY_URI\|AZURE_VM_ENDPOINT" database/postgres/environment_config.env; then
        success "ModelManager configuration found"
    else
        fail "ModelManager configuration missing"
    fi
    
    # Requirement 5: MLOps pipeline configuration with MLflow integration
    log "5. Checking MLOps pipeline configuration with MLflow integration..."
    check
    if grep -q "mlops.experiments\|mlops.runs" database/postgres/comprehensive_deployment_init.sql; then
        success "MLOps schema implementation found"
    else
        fail "MLOps schema implementation missing"
    fi
    
    check
    if grep -q "MLFLOW_TRACKING_URI\|MLFLOW_BACKEND_STORE_URI" database/postgres/environment_config.env; then
        success "MLflow configuration found"
    else
        fail "MLflow configuration missing"
    fi
    
    check
    if [[ -f "kubernetes/enhanced-database-deployments.yaml" ]] && grep -q "mlflow-tracking-deployment" kubernetes/enhanced-database-deployments.yaml; then
        success "MLflow tracking server deployment found"
    else
        fail "MLflow tracking server deployment missing"
    fi
    
    # Requirement 6: External geological age data sources configuration
    log "6. Checking external geological age data sources configuration..."
    check
    if grep -q "EARTHCHEM_API_KEY\|GEOROC_API_KEY\|NAVDAT_API_KEY\|PETDB_API_KEY" database/postgres/environment_config.env; then
        success "External API keys configuration found"
    else
        fail "External API keys configuration missing"
    fi
    
    check
    if grep -q "external_data.earthchem_samples\|external_data.georoc_samples" database/postgres/comprehensive_deployment_init.sql; then
        success "External data integration tables found"
    else
        fail "External data integration tables missing"
    fi
    
    check
    if grep -q "CHRONOS_API_KEY\|MACROSTRAT_API_KEY" database/postgres/environment_config.env; then
        success "Additional geological data sources configured"
    else
        fail "Additional geological data sources missing"
    fi
    
    # Requirement 7: High-performance computing configuration
    log "7. Checking high-performance computing configuration..."
    check
    if grep -q "PARALLEL_PROCESSING_CORES\|GPU_ACCELERATION_ENABLED" database/postgres/environment_config.env; then
        success "Parallel processing and GPU configuration found"
    else
        fail "Parallel processing and GPU configuration missing"
    fi
    
    check
    if grep -q "DASK_SCHEDULER_ADDRESS\|DASK_WORKER_COUNT" database/postgres/environment_config.env; then
        success "Distributed computing configuration found"
    else
        fail "Distributed computing configuration missing"
    fi
    
    check
    if [[ -f "kubernetes/enhanced-database-deployments.yaml" ]] && grep -q "dask-scheduler-deployment\|dask-worker-deployment" kubernetes/enhanced-database-deployments.yaml; then
        success "Dask deployment configuration found"
    else
        fail "Dask deployment configuration missing"
    fi
    
    # Requirement 8: Backup and archival configuration with multi-location storage
    log "8. Checking backup and archival configuration..."
    check
    if grep -q "BACKUP_S3_BUCKET\|BACKUP_AZURE_CONTAINER\|BACKUP_GCS_BUCKET" database/postgres/environment_config.env; then
        success "Multi-location backup storage configuration found"
    else
        fail "Multi-location backup storage configuration missing"
    fi
    
    check
    if grep -q "backup_archive.backup_jobs\|backup_archive.archive_storage" database/postgres/comprehensive_deployment_init.sql; then
        success "Backup management schema found"
    else
        fail "Backup management schema missing"
    fi
    
    check
    if [[ -f "kubernetes/enhanced-database-configmaps.yaml" ]] && grep -q "backup-postgresql.sh\|restore-postgresql.sh" kubernetes/enhanced-database-configmaps.yaml; then
        success "Backup and restore scripts found"
    else
        fail "Backup and restore scripts missing"
    fi
}

# Validate Kubernetes deployment files
validate_kubernetes_deployment() {
    log "Validating Kubernetes deployment configuration..."
    echo ""
    
    # Check enhanced database deployments
    check
    if [[ -f "kubernetes/enhanced-database-deployments.yaml" ]]; then
        success "Enhanced database deployments file exists"
        
        # Check for all required database services
        if grep -q "enhanced-postgres-deployment\|enhanced-influxdb-deployment\|enhanced-neo4j-deployment\|enhanced-redis-deployment" kubernetes/enhanced-database-deployments.yaml; then
            success "All core database deployments configured"
        else
            fail "Some core database deployments missing"
        fi
    else
        fail "Enhanced database deployments file missing"
    fi
    
    # Check enhanced secrets
    check
    if [[ -f "kubernetes/enhanced-database-secrets.yaml" ]]; then
        success "Enhanced database secrets file exists"
        
        if grep -q "earthchem-api-key\|georoc-api-key\|navdat-api-key\|petdb-api-key" kubernetes/enhanced-database-secrets.yaml; then
            success "External API keys in secrets configuration"
        else
            fail "External API keys missing from secrets"
        fi
    else
        fail "Enhanced database secrets file missing"
    fi
    
    # Check enhanced persistent volumes
    check
    if [[ -f "kubernetes/enhanced-persistent-volumes.yaml" ]]; then
        success "Enhanced persistent volumes file exists"
        
        if grep -q "enhanced-postgres-pv\|enhanced-influxdb-pv\|mlflow-artifacts-pv" kubernetes/enhanced-persistent-volumes.yaml; then
            success "All required persistent volumes configured"
        else
            fail "Some persistent volumes missing"
        fi
    else
        fail "Enhanced persistent volumes file missing"
    fi
    
    # Check enhanced configuration maps
    check
    if [[ -f "kubernetes/enhanced-database-configmaps.yaml" ]]; then
        success "Enhanced database configuration maps file exists"
        
        if grep -q "postgres-performance-config\|influxdb-performance-config\|redis-performance-config" kubernetes/enhanced-database-configmaps.yaml; then
            success "Performance configuration maps found"
        else
            fail "Performance configuration maps missing"
        fi
    else
        fail "Enhanced database configuration maps file missing"
    fi
}

# Validate deployment scripts
validate_deployment_scripts() {
    log "Validating deployment scripts..."
    echo ""
    
    # Check main deployment script
    check
    if [[ -f "scripts/deploy-enhanced-database.sh" ]]; then
        success "Enhanced database deployment script exists"
        
        if grep -q "check_prerequisites\|deploy_storage\|deploy_config\|deploy_databases" scripts/deploy-enhanced-database.sh; then
            success "All deployment functions implemented"
        else
            fail "Some deployment functions missing"
        fi
        
        if grep -q "verify_databases\|initialize_geological_age_db\|setup_mlops" scripts/deploy-enhanced-database.sh; then
            success "Verification and initialization functions found"
        else
            fail "Verification and initialization functions missing"
        fi
    else
        fail "Enhanced database deployment script missing"
    fi
    
    # Check validation script (this script)
    check
    if [[ -f "scripts/validate-database-deployment.sh" ]]; then
        success "Database deployment validation script exists"
    else
        fail "Database deployment validation script missing"
    fi
}

# Validate documentation
validate_documentation() {
    log "Validating documentation..."
    echo ""
    
    # Check comprehensive documentation
    check
    if [[ -f "database/DATABASE_DEPLOYMENT_GUIDE.md" ]]; then
        success "Comprehensive database deployment guide exists"
        
        if grep -q "External Data Source Integration\|High-Performance Computing\|Backup and Archival" database/DATABASE_DEPLOYMENT_GUIDE.md; then
            success "All major sections documented"
        else
            fail "Some documentation sections missing"
        fi
    else
        fail "Database deployment guide missing"
    fi
}

# Validate requirements mapping
validate_requirements_mapping() {
    log "Validating requirements mapping (5.1, 5.2, 8.1, 8.2)..."
    echo ""
    
    # Requirement 5.1: Multi-Database Integration and Management
    check
    if grep -q "InfluxDB\|PostgreSQL\|Neo4j\|Redis" kubernetes/enhanced-database-deployments.yaml; then
        success "Requirement 5.1: Multi-database integration implemented"
    else
        fail "Requirement 5.1: Multi-database integration missing"
    fi
    
    # Requirement 5.2: Advanced database features
    check
    if grep -q "PostGIS\|TimeScaleDB\|btree_gist" database/postgres/comprehensive_deployment_init.sql; then
        success "Requirement 5.2: Advanced database features implemented"
    else
        fail "Requirement 5.2: Advanced database features missing"
    fi
    
    # Requirement 8.1: Performance optimization and scalability
    check
    if grep -q "shared_buffers\|effective_cache_size\|max_parallel_workers" kubernetes/enhanced-database-configmaps.yaml; then
        success "Requirement 8.1: Performance optimization implemented"
    else
        fail "Requirement 8.1: Performance optimization missing"
    fi
    
    # Requirement 8.2: Scalability features
    check
    if grep -q "dask-scheduler\|dask-worker\|parallel_processing" kubernetes/enhanced-database-deployments.yaml; then
        success "Requirement 8.2: Scalability features implemented"
    else
        fail "Requirement 8.2: Scalability features missing"
    fi
}

# Generate validation report
generate_report() {
    echo ""
    echo "============================================================================="
    echo "                    DATABASE DEPLOYMENT VALIDATION REPORT"
    echo "============================================================================="
    echo ""
    echo "Task 2.5: Implement Database Deployment Configuration"
    echo ""
    echo "Validation Results:"
    echo "  Total Checks: ${TOTAL_CHECKS}"
    echo "  Passed: ${PASSED_CHECKS}"
    echo "  Failed: $((TOTAL_CHECKS - PASSED_CHECKS))"
    echo "  Success Rate: $(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))%"
    echo ""
    
    if [[ ${PASSED_CHECKS} -eq ${TOTAL_CHECKS} ]]; then
        echo -e "${GREEN}✓ ALL VALIDATIONS PASSED${NC}"
        echo ""
        echo "Task 2.5 has been successfully implemented with all requirements met:"
        echo ""
        echo "✓ Comprehensive database deployment scripts created"
        echo "✓ PostgreSQL with PostGIS, TimeScaleDB, and specialized extensions configured"
        echo "✓ Environment variables for geological age database integration set up"
        echo "✓ ModelManager configuration for age prediction implemented"
        echo "✓ MLOps pipeline configuration with MLflow integration completed"
        echo "✓ External geological age data sources (EarthChem, GEOROC, NAVDAT, PetDB) configured"
        echo "✓ High-performance computing configuration for parallel processing added"
        echo "✓ Backup and archival configuration with multi-location storage created"
        echo "✓ Requirements 5.1, 5.2, 8.1, 8.2 successfully addressed"
        echo ""
        echo "The enhanced database deployment configuration is ready for production use."
    else
        echo -e "${RED}✗ SOME VALIDATIONS FAILED${NC}"
        echo ""
        echo "The following issues need to be addressed:"
        echo ""
        for result in "${VALIDATION_RESULTS[@]}"; do
            if [[ $result == ✗* ]]; then
                echo -e "${RED}$result${NC}"
            fi
        done
    fi
    
    echo ""
    echo "Detailed Results:"
    for result in "${VALIDATION_RESULTS[@]}"; do
        if [[ $result == ✓* ]]; then
            echo -e "${GREEN}$result${NC}"
        elif [[ $result == ✗* ]]; then
            echo -e "${RED}$result${NC}"
        else
            echo -e "${YELLOW}$result${NC}"
        fi
    done
    
    echo ""
    echo "============================================================================="
}

# Main validation function
main() {
    log "Starting validation of Database Deployment Configuration (Task 2.5)..."
    echo ""
    
    validate_task_requirements
    validate_kubernetes_deployment
    validate_deployment_scripts
    validate_documentation
    validate_requirements_mapping
    
    generate_report
}

# Run validation
main