#!/bin/bash

# Prometheus AI Platform - Disaster Recovery Testing Script
# Comprehensive testing script for disaster recovery capabilities

set -euo pipefail

# Configuration
NAMESPACE="prometheus-ai"
SERVICE_URL="http://disaster-recovery:8080"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TEST_RESULTS=()

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

# Test result functions
test_passed() {
    local test_name="$1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TEST_RESULTS+=("✅ $test_name")
    log_success "$test_name"
}

test_failed() {
    local test_name="$1"
    local error_msg="$2"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TEST_RESULTS+=("❌ $test_name: $error_msg")
    log_error "$test_name: $error_msg"
}

# Setup port forwarding
setup_port_forward() {
    log_info "Setting up port forwarding..."
    
    # Kill any existing port forwards
    pkill -f "kubectl.*port-forward.*disaster-recovery" 2>/dev/null || true
    sleep 2
    
    # Start port forward
    kubectl port-forward -n "$NAMESPACE" service/disaster-recovery 8080:8080 &
    PORT_FORWARD_PID=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Test connectivity
    if curl -s http://localhost:8080/health > /dev/null; then
        log_success "Port forwarding established"
        return 0
    else
        log_error "Failed to establish port forwarding"
        return 1
    fi
}

# Test health endpoints
test_health_endpoints() {
    log_info "Testing health endpoints..."
    
    # Test health endpoint
    if response=$(curl -s -w "%{http_code}" http://localhost:8080/health); then
        http_code="${response: -3}"
        if [[ "$http_code" == "200" ]]; then
            test_passed "Health endpoint test"
        else
            test_failed "Health endpoint test" "HTTP $http_code"
        fi
    else
        test_failed "Health endpoint test" "Connection failed"
    fi
    
    # Test readiness endpoint
    if response=$(curl -s -w "%{http_code}" http://localhost:8080/ready); then
        http_code="${response: -3}"
        if [[ "$http_code" == "200" ]]; then
            test_passed "Readiness endpoint test"
        else
            test_failed "Readiness endpoint test" "HTTP $http_code"
        fi
    else
        test_failed "Readiness endpoint test" "Connection failed"
    fi
}

# Test backup operations
test_backup_operations() {
    log_info "Testing backup operations..."
    
    # Test backup creation
    local backup_response
    if backup_response=$(curl -s -X POST http://localhost:8080/api/v1/backup/create \
        -H "Content-Type: application/json" \
        -d '{
            "database_type": "postgresql",
            "database_name": "test_db",
            "backup_type": "full"
        }'); then
        
        local backup_id
        backup_id=$(echo "$backup_response" | jq -r '.backup_id // empty')
        
        if [[ -n "$backup_id" && "$backup_id" != "null" ]]; then
            test_passed "Backup creation test"
            
            # Test backup status
            sleep 2
            if status_response=$(curl -s "http://localhost:8080/api/v1/backup/status/$backup_id"); then
                local status
                status=$(echo "$status_response" | jq -r '.status // empty')
                if [[ -n "$status" ]]; then
                    test_passed "Backup status test"
                else
                    test_failed "Backup status test" "No status returned"
                fi
            else
                test_failed "Backup status test" "Request failed"
            fi
        else
            test_failed "Backup creation test" "No backup ID returned"
        fi
    else
        test_failed "Backup creation test" "Request failed"
    fi
    
    # Test backup listing
    if list_response=$(curl -s http://localhost:8080/api/v1/backup/list); then
        local backup_count
        backup_count=$(echo "$list_response" | jq '.backups | length // 0')
        if [[ "$backup_count" -ge 0 ]]; then
            test_passed "Backup listing test"
        else
            test_failed "Backup listing test" "Invalid response format"
        fi
    else
        test_failed "Backup listing test" "Request failed"
    fi
}

# Test replication operations
test_replication_operations() {
    log_info "Testing replication operations..."
    
    # Test replication status
    if repl_response=$(curl -s http://localhost:8080/api/v1/replication/status); then
        local total_replications
        total_replications=$(echo "$repl_response" | jq '.total_replications // 0')
        if [[ "$total_replications" -ge 0 ]]; then
            test_passed "Replication status test"
        else
            test_failed "Replication status test" "Invalid response format"
        fi
    else
        test_failed "Replication status test" "Request failed"
    fi
    
    # Test replication configuration
    if config_response=$(curl -s -X POST http://localhost:8080/api/v1/replication/configure \
        -H "Content-Type: application/json" \
        -d '{
            "source_region": "us-west-2",
            "target_regions": ["us-east-1"],
            "database_type": "postgresql",
            "replication_mode": "async"
        }'); then
        
        local config_id
        config_id=$(echo "$config_response" | jq -r '.config_id // empty')
        if [[ -n "$config_id" && "$config_id" != "null" ]]; then
            test_passed "Replication configuration test"
        else
            test_failed "Replication configuration test" "No config ID returned"
        fi
    else
        test_failed "Replication configuration test" "Request failed"
    fi
}

# Test recovery operations
test_recovery_operations() {
    log_info "Testing recovery operations..."
    
    # Test recovery plan creation
    if plan_response=$(curl -s -X POST http://localhost:8080/api/v1/recovery/plan \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Test Recovery Plan",
            "description": "Test plan for DR testing",
            "disaster_types": ["hardware_failure"],
            "affected_services": ["postgresql"],
            "recovery_steps": [
                {
                    "step": 1,
                    "name": "Test Step",
                    "action": "test_action",
                    "timeout_minutes": 10,
                    "required": true
                }
            ],
            "recovery_objectives": {
                "rto_minutes": 60,
                "rpo_minutes": 15,
                "service_tier": "critical"
            }
        }'); then
        
        local plan_id
        plan_id=$(echo "$plan_response" | jq -r '.plan_id // empty')
        if [[ -n "$plan_id" && "$plan_id" != "null" ]]; then
            test_passed "Recovery plan creation test"
            
            # Test recovery plan execution (dry run)
            if exec_response=$(curl -s -X POST "http://localhost:8080/api/v1/recovery/execute/$plan_id"); then
                local execution_id
                execution_id=$(echo "$exec_response" | jq -r '.execution_id // empty')
                if [[ -n "$execution_id" && "$execution_id" != "null" ]]; then
                    test_passed "Recovery plan execution test"
                    
                    # Test recovery status
                    sleep 2
                    if status_response=$(curl -s "http://localhost:8080/api/v1/recovery/status/$execution_id"); then
                        local status
                        status=$(echo "$status_response" | jq -r '.status // empty')
                        if [[ -n "$status" ]]; then
                            test_passed "Recovery status test"
                        else
                            test_failed "Recovery status test" "No status returned"
                        fi
                    else
                        test_failed "Recovery status test" "Request failed"
                    fi
                else
                    test_failed "Recovery plan execution test" "No execution ID returned"
                fi
            else
                test_failed "Recovery plan execution test" "Request failed"
            fi
        else
            test_failed "Recovery plan creation test" "No plan ID returned"
        fi
    else
        test_failed "Recovery plan creation test" "Request failed"
    fi
}

# Test continuity operations
test_continuity_operations() {
    log_info "Testing continuity operations..."
    
    # Test continuity plan creation
    if plan_response=$(curl -s -X POST http://localhost:8080/api/v1/continuity/plan \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Test Continuity Plan",
            "description": "Test business continuity plan",
            "business_functions": ["test_function"],
            "critical_dependencies": ["test_dependency"],
            "alternative_procedures": [],
            "communication_plan": {},
            "resource_requirements": {},
            "recovery_priorities": ["test_priority"]
        }'); then
        
        local plan_id
        plan_id=$(echo "$plan_response" | jq -r '.plan_id // empty')
        if [[ -n "$plan_id" && "$plan_id" != "null" ]]; then
            test_passed "Continuity plan creation test"
        else
            test_failed "Continuity plan creation test" "No plan ID returned"
        fi
    else
        test_failed "Continuity plan creation test" "Request failed"
    fi
    
    # Test continuity assessment
    if assessment_response=$(curl -s http://localhost:8080/api/v1/continuity/assessment); then
        local overall_score
        overall_score=$(echo "$assessment_response" | jq '.overall_readiness_score // 0')
        if [[ $(echo "$overall_score >= 0" | bc -l) -eq 1 ]]; then
            test_passed "Continuity assessment test"
        else
            test_failed "Continuity assessment test" "Invalid score format"
        fi
    else
        test_failed "Continuity assessment test" "Request failed"
    fi
}

# Test monitoring and metrics
test_monitoring_metrics() {
    log_info "Testing monitoring and metrics..."
    
    # Test RTO/RPO metrics
    if rto_rpo_response=$(curl -s http://localhost:8080/api/v1/metrics/rto-rpo); then
        local rto_stats
        rto_stats=$(echo "$rto_rpo_response" | jq '.rto_statistics // {}')
        if [[ -n "$rto_stats" && "$rto_stats" != "null" ]]; then
            test_passed "RTO/RPO metrics test"
        else
            test_failed "RTO/RPO metrics test" "Invalid response format"
        fi
    else
        test_failed "RTO/RPO metrics test" "Request failed"
    fi
    
    # Test backup integrity metrics
    if integrity_response=$(curl -s http://localhost:8080/api/v1/metrics/backup-integrity); then
        local total_checks
        total_checks=$(echo "$integrity_response" | jq '.total_checks_24h // 0')
        if [[ "$total_checks" -ge 0 ]]; then
            test_passed "Backup integrity metrics test"
        else
            test_failed "Backup integrity metrics test" "Invalid response format"
        fi
    else
        test_failed "Backup integrity metrics test" "Request failed"
    fi
}

# Test disaster recovery testing
test_dr_testing() {
    log_info "Testing DR testing functionality..."
    
    # Test DR test initiation
    if test_response=$(curl -s -X POST http://localhost:8080/api/v1/test/disaster-recovery \
        -H "Content-Type: application/json" \
        -d '{
            "test_name": "Automated Test",
            "test_type": "backup_restore",
            "scope": ["postgresql"]
        }'); then
        
        local test_id
        test_id=$(echo "$test_response" | jq -r '.test_id // empty')
        if [[ -n "$test_id" && "$test_id" != "null" ]]; then
            test_passed "DR test initiation test"
        else
            test_failed "DR test initiation test" "No test ID returned"
        fi
    else
        test_failed "DR test initiation test" "Request failed"
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test invalid backup request
    if response=$(curl -s -w "%{http_code}" -X POST http://localhost:8080/api/v1/backup/create \
        -H "Content-Type: application/json" \
        -d '{"invalid": "data"}'); then
        
        http_code="${response: -3}"
        if [[ "$http_code" == "422" || "$http_code" == "400" ]]; then
            test_passed "Invalid backup request error handling"
        else
            test_failed "Invalid backup request error handling" "Expected 4xx, got $http_code"
        fi
    else
        test_failed "Invalid backup request error handling" "Request failed"
    fi
    
    # Test non-existent backup status
    if response=$(curl -s -w "%{http_code}" http://localhost:8080/api/v1/backup/status/non-existent-id); then
        http_code="${response: -3}"
        if [[ "$http_code" == "404" ]]; then
            test_passed "Non-existent backup error handling"
        else
            test_failed "Non-existent backup error handling" "Expected 404, got $http_code"
        fi
    else
        test_failed "Non-existent backup error handling" "Request failed"
    fi
}

# Test performance
test_performance() {
    log_info "Testing performance..."
    
    # Test response times
    local start_time end_time duration
    start_time=$(date +%s%N)
    
    if curl -s http://localhost:8080/health > /dev/null; then
        end_time=$(date +%s%N)
        duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        if [[ $duration -lt 1000 ]]; then # Less than 1 second
            test_passed "Health endpoint performance test (${duration}ms)"
        else
            test_failed "Health endpoint performance test" "Response time too slow (${duration}ms)"
        fi
    else
        test_failed "Health endpoint performance test" "Request failed"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill port forward
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
    fi
    
    # Kill any remaining port forwards
    pkill -f "kubectl.*port-forward.*disaster-recovery" 2>/dev/null || true
}

# Generate test report
generate_report() {
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    local success_rate=0
    
    if [[ $total_tests -gt 0 ]]; then
        success_rate=$(( (TESTS_PASSED * 100) / total_tests ))
    fi
    
    echo ""
    echo "=========================================="
    echo "         DISASTER RECOVERY TEST REPORT"
    echo "=========================================="
    echo "Total Tests: $total_tests"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Success Rate: $success_rate%"
    echo ""
    
    if [[ ${#TEST_RESULTS[@]} -gt 0 ]]; then
        echo "Test Results:"
        printf '%s\n' "${TEST_RESULTS[@]}"
        echo ""
    fi
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All tests passed! 🎉"
        return 0
    else
        log_error "$TESTS_FAILED test(s) failed"
        return 1
    fi
}

# Main test function
main() {
    log_info "Starting disaster recovery testing..."
    
    # Check prerequisites
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed"
        exit 1
    fi
    
    # Setup
    if ! setup_port_forward; then
        log_error "Failed to setup port forwarding"
        exit 1
    fi
    
    # Run tests
    test_health_endpoints
    test_backup_operations
    test_replication_operations
    test_recovery_operations
    test_continuity_operations
    test_monitoring_metrics
    test_dr_testing
    test_error_handling
    test_performance
    
    # Generate report
    generate_report
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"