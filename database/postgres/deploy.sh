#!/bin/bash
# =============================================================================
# DATABASE DEPLOYMENT SCRIPT
# Prometheus AI Platform - Comprehensive Database Setup
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment_config.env"
MIGRATION_MANAGER="${SCRIPT_DIR}/migration_manager.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Load environment configuration
load_environment() {
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Loading environment configuration from $ENV_FILE"
        set -a  # Automatically export all variables
        source "$ENV_FILE"
        set +a
        log_success "Environment configuration loaded"
    else
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
}

# Validate required environment variables
validate_environment() {
    log_info "Validating environment configuration..."
    
    required_vars=(
        "POSTGRES_HOST"
        "POSTGRES_PORT"
        "POSTGRES_DB"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Test database connection
test_connection() {
    log_info "Testing database connection..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" > /dev/null 2>&1; then
        log_success "Database connection successful"
    else
        log_error "Failed to connect to database"
        log_error "Host: $POSTGRES_HOST:$POSTGRES_PORT"
        log_error "Database: $POSTGRES_DB"
        log_error "User: $POSTGRES_USER"
        exit 1
    fi
}

# Create database if it doesn't exist
create_database() {
    log_info "Ensuring database exists..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Connect to postgres database to create target database
    if ! psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "postgres" -c "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB';" | grep -q 1; then
        log_info "Creating database: $POSTGRES_DB"
        psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "postgres" -c "CREATE DATABASE $POSTGRES_DB;"
        log_success "Database created successfully"
    else
        log_info "Database already exists: $POSTGRES_DB"
    fi
}

# Install required PostgreSQL extensions
install_extensions() {
    log_info "Installing required PostgreSQL extensions..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    extensions=(
        "postgis"
        "postgis_topology"
        "postgis_raster"
        "timescaledb"
        "uuid-ossp"
        "btree_gist"
        "pg_stat_statements"
        "pg_trgm"
        "hstore"
        "ltree"
    )
    
    for ext in "${extensions[@]}"; do
        log_info "Installing extension: $ext"
        psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
             -c "CREATE EXTENSION IF NOT EXISTS \"$ext\" CASCADE;" || {
            log_warning "Failed to install extension: $ext (may not be available)"
        }
    done
    
    log_success "Extensions installation completed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Construct database URL
    DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    
    # Check if Python migration manager exists
    if [[ -f "$MIGRATION_MANAGER" ]]; then
        log_info "Using Python migration manager"
        python3 "$MIGRATION_MANAGER" \
            --database-url "$DATABASE_URL" \
            --migrations-path "${SCRIPT_DIR}/migrations" \
            --action migrate \
            --version "2.0.0"
    else
        log_warning "Python migration manager not found, running SQL files directly"
        
        # Run SQL files in order
        for sql_file in "${SCRIPT_DIR}"/migrations/*.sql; do
            if [[ -f "$sql_file" ]]; then
                log_info "Executing: $(basename "$sql_file")"
                export PGPASSWORD="$POSTGRES_PASSWORD"
                psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$sql_file"
            fi
        done
        
        # Run comprehensive deployment init
        if [[ -f "${SCRIPT_DIR}/comprehensive_deployment_init.sql" ]]; then
            log_info "Running comprehensive deployment initialization"
            export PGPASSWORD="$POSTGRES_PASSWORD"
            psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "${SCRIPT_DIR}/comprehensive_deployment_init.sql"
        fi
        
        # Run comprehensive geological schema
        if [[ -f "${SCRIPT_DIR}/comprehensive_geological_schema.sql" ]]; then
            log_info "Running comprehensive geological schema"
            export PGPASSWORD="$POSTGRES_PASSWORD"
            psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "${SCRIPT_DIR}/comprehensive_geological_schema.sql"
        fi
    fi
    
    log_success "Database migrations completed"
}

# Validate schema integrity
validate_schema() {
    log_info "Validating database schema integrity..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Check required schemas exist
    required_schemas=(
        "geological_core"
        "rock_mechanics"
        "volcanic_systems"
        "seismic_networks"
        "geodetic_data"
        "geophysical_data"
        "geological_age"
        "model_management"
        "mlops"
        "external_data"
        "backup_archive"
        "performance_monitoring"
    )
    
    for schema in "${required_schemas[@]}"; do
        if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
               -c "SELECT 1 FROM information_schema.schemata WHERE schema_name='$schema';" | grep -q 1; then
            log_success "Schema exists: $schema"
        else
            log_error "Missing schema: $schema"
            exit 1
        fi
    done
    
    log_success "Schema validation completed"
}

# Setup monitoring and performance optimization
setup_monitoring() {
    log_info "Setting up database monitoring and performance optimization..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Enable query performance monitoring
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
        ALTER SYSTEM SET pg_stat_statements.track = 'all';
        ALTER SYSTEM SET pg_stat_statements.max = 10000;
        ALTER SYSTEM SET log_statement = 'mod';
        ALTER SYSTEM SET log_min_duration_statement = 1000;
        SELECT pg_reload_conf();
    " || log_warning "Failed to configure monitoring settings"
    
    log_success "Monitoring setup completed"
}

# Create initial data and configurations
setup_initial_data() {
    log_info "Setting up initial data and configurations..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Insert default configurations if they don't exist
    psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
        -- Insert default MLOps experiments
        INSERT INTO mlops.experiments (experiment_name, experiment_id, description) VALUES
            ('geological_age_prediction', 'exp_001', 'Geological age prediction using multi-source data'),
            ('eruption_forecasting', 'exp_002', 'Volcanic eruption forecasting models'),
            ('resource_prospectivity', 'exp_003', 'Mineral resource prospectivity mapping'),
            ('seismic_analysis', 'exp_004', 'Seismic event detection and classification')
        ON CONFLICT (experiment_name) DO NOTHING;
        
        -- Insert external data sync configuration
        INSERT INTO geological_age.external_sync_status (data_source, api_endpoint, sync_configuration) VALUES
            ('earthchem', 'https://www.earthchem.org/portal/api/', '{\"sync_interval_hours\": 24, \"batch_size\": 1000}'),
            ('georoc', 'https://georoc.mpch-mainz.gwdg.de/georoc/api/', '{\"sync_interval_hours\": 24, \"batch_size\": 500}'),
            ('navdat', 'https://navdat.org/api/v1/', '{\"sync_interval_hours\": 12, \"batch_size\": 200}'),
            ('petdb', 'https://petdb.ldeo.columbia.edu/api/', '{\"sync_interval_hours\": 24, \"batch_size\": 300}')
        ON CONFLICT (data_source) DO NOTHING;
    " || log_warning "Failed to insert some initial data (may already exist)"
    
    log_success "Initial data setup completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Get database statistics
    stats=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "
        SELECT 
            (SELECT count(*) FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')) as custom_schemas,
            (SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')) as total_tables,
            (SELECT count(*) FROM pg_extension) as extensions_count,
            (SELECT pg_size_pretty(pg_database_size('$POSTGRES_DB'))) as database_size;
    ")
    
    echo ""
    echo "=============================================="
    echo "DATABASE DEPLOYMENT REPORT"
    echo "=============================================="
    echo "Deployment Time: $(date)"
    echo "Database: $POSTGRES_DB"
    echo "Host: $POSTGRES_HOST:$POSTGRES_PORT"
    echo "Statistics: $stats"
    echo "=============================================="
    echo ""
    
    log_success "Deployment report generated"
}

# Main deployment function
deploy() {
    log_info "Starting Prometheus AI Platform database deployment..."
    
    load_environment
    validate_environment
    test_connection
    create_database
    install_extensions
    run_migrations
    validate_schema
    setup_monitoring
    setup_initial_data
    generate_report
    
    log_success "Database deployment completed successfully!"
}

# Rollback function
rollback() {
    log_warning "Rolling back database changes..."
    
    load_environment
    validate_environment
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Drop all custom schemas (be very careful with this!)
    read -p "Are you sure you want to rollback? This will drop all data! (yes/no): " confirm
    if [[ "$confirm" == "yes" ]]; then
        psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
            DROP SCHEMA IF EXISTS geological_core CASCADE;
            DROP SCHEMA IF EXISTS rock_mechanics CASCADE;
            DROP SCHEMA IF EXISTS volcanic_systems CASCADE;
            DROP SCHEMA IF EXISTS seismic_networks CASCADE;
            DROP SCHEMA IF EXISTS geodetic_data CASCADE;
            DROP SCHEMA IF EXISTS geophysical_data CASCADE;
            DROP SCHEMA IF EXISTS geological_age CASCADE;
            DROP SCHEMA IF EXISTS model_management CASCADE;
            DROP SCHEMA IF EXISTS mlops CASCADE;
            DROP SCHEMA IF EXISTS external_data CASCADE;
            DROP SCHEMA IF EXISTS backup_archive CASCADE;
            DROP SCHEMA IF EXISTS performance_monitoring CASCADE;
            DROP TABLE IF EXISTS schema_migrations CASCADE;
            DROP TABLE IF EXISTS migration_log CASCADE;
            DROP TABLE IF EXISTS database_version CASCADE;
        "
        log_success "Rollback completed"
    else
        log_info "Rollback cancelled"
    fi
}

# Help function
show_help() {
    echo "Prometheus AI Platform Database Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy the complete database schema (default)"
    echo "  rollback   Rollback all database changes (DESTRUCTIVE!)"
    echo "  test       Test database connection"
    echo "  validate   Validate schema integrity"
    echo "  help       Show this help message"
    echo ""
    echo "Environment:"
    echo "  Configuration is loaded from: $ENV_FILE"
    echo ""
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "rollback")
        rollback
        ;;
    "test")
        load_environment
        validate_environment
        test_connection
        ;;
    "validate")
        load_environment
        validate_environment
        validate_schema
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac