# CI/CD System Documentation

## Overview

The Prometheus AI Platform includes a comprehensive CI/CD system that provides automated build, test, security scanning, and deployment capabilities with zero-downtime rollback functionality.

## Architecture

### Components

1. **Intelligent Agent Hook System** - Event-driven automation with smart decision making
2. **CI/CD Pipeline** - Comprehensive build, test, and deployment pipeline
3. **Security Scanner** - Multi-tool security vulnerability scanning
4. **Deployment Manager** - Zero-downtime deployments with automatic rollback
5. **Version Manager** - Semantic versioning for releases

### Key Features

- **Smart Test Selection** - Automatically selects relevant tests based on code changes
- **Automated Model Retraining** - Triggers model retraining based on performance degradation
- **Intelligent Resource Scaling** - Automatically scales resources based on workload patterns
- **Zero-Downtime Deployments** - Blue-green deployment strategy with health monitoring
- **Automatic Rollback** - Rolls back deployments on failure detection
- **Comprehensive Security Scanning** - Multi-tool security analysis
- **Quality Gates** - Enforces code quality, test coverage, and security standards

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- Kubernetes cluster
- Git repository

### Installation

1. Install dependencies:
```bash
pip install -r services/system-monitor/requirements.txt
```

2. Configure the system:
```bash
# Copy and customize the configuration
cp .kiro/cicd-config.yaml .kiro/cicd-config-local.yaml
```

3. Set up environment variables:
```bash
export SLACK_WEBHOOK_URL="your-slack-webhook"
export DOCKER_REGISTRY_URL="your-registry"
# ... other environment variables
```

### Starting the Agent Hook System

```bash
# Start the intelligent agent hook system
python scripts/start_agent_hooks.py

# Or run as daemon
python scripts/start_agent_hooks.py --daemon
```

### Running the CI/CD Pipeline

```bash
# Run pipeline for current commit and branch
python scripts/run_cicd_pipeline.py

# Run pipeline for specific commit
python scripts/run_cicd_pipeline.py --commit abc123def

# Run only specific stages
python scripts/run_cicd_pipeline.py --stages build,test

# Dry run mode
python scripts/run_cicd_pipeline.py --dry-run
```

## Configuration

### Agent Hooks Configuration

The intelligent agent hooks are configured in `.kiro/hooks/intelligent-agent-hooks.json`:

```json
{
  "hooks": {
    "smart_test_execution": {
      "enabled": true,
      "triggers": ["file_change", "code_commit"],
      "conditions": {
        "min_files_changed": 1,
        "max_files_changed": 50
      },
      "actions": ["smart_test_selection", "execute_selected_tests"]
    }
  }
}
```

### CI/CD Pipeline Configuration

The CI/CD pipeline is configured in `.kiro/cicd-config.yaml`:

```yaml
stages:
  build:
    enabled: true
    timeout: 900
    parallel: true
    
  test:
    enabled: true
    timeout: 2400
    coverage_required: true
    
quality_gates:
  test_coverage:
    min_coverage: 0.80
  security:
    max_high_vulnerabilities: 0
```

## Pipeline Stages

### 1. Build Stage
- Builds Docker images for all services
- Validates Dockerfiles and build configurations
- Pushes images to registry with version tags

### 2. Test Stage
- Runs comprehensive test suite
- Generates code coverage reports
- Enforces minimum coverage thresholds

### 3. Security Scan Stage
- **Bandit** - Python security vulnerability scanner
- **npm audit** - Node.js dependency vulnerability scanner
- **Trivy** - Container image vulnerability scanner
- **Secrets Scanner** - Detects exposed credentials

### 4. Quality Gate Stage
- Code quality analysis
- Technical debt assessment
- Complexity metrics validation

### 5. Deploy Staging Stage
- Deploys to staging environment
- Performs health checks
- Runs smoke tests

### 6. Integration Test Stage
- Runs integration tests against staging
- Validates end-to-end functionality
- Performance testing

### 7. Deploy Production Stage
- Blue-green deployment to production
- Zero-downtime deployment strategy
- Automatic rollback on failure

### 8. Health Check Stage
- Extended health monitoring
- Error rate monitoring
- Performance validation

## Intelligent Agent Hooks

### Smart Test Selection

Automatically selects relevant tests based on:
- Changed files and their dependencies
- Component mapping configuration
- Historical test execution data
- Risk assessment of changes

### Model Retraining Triggers

Automatically triggers model retraining when:
- Model accuracy drops below threshold
- Physics compliance degrades
- Prediction drift detected
- Training data becomes stale
- Model age exceeds limits

### Resource Scaling

Intelligently scales resources based on:
- CPU and memory utilization
- Queue length and response times
- Error rates and throughput
- Historical usage patterns

## Deployment Strategies

### Blue-Green Deployment

1. Deploy new version to green environment
2. Run health checks on green environment
3. Switch traffic from blue to green
4. Monitor for issues
5. Rollback to blue if problems detected

### Automatic Rollback

Rollback is triggered when:
- Health checks fail
- Error rate exceeds threshold
- Response time degrades significantly
- Manual rollback requested

## Security Features

### Vulnerability Scanning

- **Python Code**: Bandit static analysis
- **Node.js Dependencies**: npm audit
- **Container Images**: Trivy scanning
- **Secrets Detection**: Pattern-based scanning

### Security Gates

- Zero high-severity vulnerabilities allowed
- Limited medium-severity vulnerabilities
- Secrets detection prevents commits
- Container image scanning before deployment

## Monitoring and Observability

### Metrics

- Pipeline execution duration
- Stage success/failure rates
- Deployment frequency
- Rollback frequency
- Test coverage trends

### Logging

- Structured JSON logging
- Correlation IDs for tracing
- Centralized log aggregation
- Retention policies

### Notifications

- Slack integration for real-time updates
- Email notifications for failures
- Teams integration for deployment status

## Quality Gates

### Test Coverage
- Minimum 80% line coverage
- Minimum 75% branch coverage
- Minimum 85% function coverage

### Security
- Zero high-severity vulnerabilities
- Maximum 5 medium-severity vulnerabilities
- No exposed secrets or credentials

### Code Quality
- Minimum quality score of 7.0
- Maximum 10% code duplication
- Maximum cyclomatic complexity of 15

### Model Quality
- Minimum 80% model accuracy
- Minimum 70% physics compliance
- Maximum 15% prediction drift

## Troubleshooting

### Common Issues

1. **Pipeline Fails at Build Stage**
   - Check Dockerfile syntax
   - Verify base image availability
   - Check build context and dependencies

2. **Tests Fail Due to Coverage**
   - Add tests for uncovered code
   - Review coverage exclusions
   - Check test configuration

3. **Security Scan Failures**
   - Update vulnerable dependencies
   - Fix code security issues
   - Review and approve exceptions

4. **Deployment Failures**
   - Check Kubernetes cluster health
   - Verify resource availability
   - Review deployment manifests

### Debugging

```bash
# Check agent hook system status
python scripts/start_agent_hooks.py --status

# Run pipeline with verbose logging
python scripts/run_cicd_pipeline.py --verbose

# Check execution history
cat cicd-execution-history.json | jq '.[-1]'
```

## Best Practices

### Development Workflow

1. Create feature branch from develop
2. Make changes and commit frequently
3. Push to trigger automated testing
4. Create pull request when ready
5. Merge to develop after approval
6. Deploy to production from main branch

### Configuration Management

1. Use environment variables for secrets
2. Version control all configuration files
3. Test configuration changes in staging
4. Document configuration changes

### Security

1. Regularly update dependencies
2. Monitor security advisories
3. Use least privilege principles
4. Rotate secrets regularly

### Performance

1. Monitor pipeline execution times
2. Optimize slow stages
3. Use caching where appropriate
4. Parallelize independent operations

## API Reference

### Agent Hook System API

```python
from services.system_monitor.intelligent_agent_hooks import IntelligentAgentHookSystem

# Initialize system
hook_system = IntelligentAgentHookSystem()

# Start event processing
await hook_system.start_event_processing()

# Process manual event
event = HookEvent(
    event_id="manual_test",
    event_type=EventType.MANUAL_TRIGGER,
    timestamp=datetime.now(),
    source="api",
    metadata={"test": True}
)
await hook_system.process_event(event)
```

### CI/CD Pipeline API

```python
from services.system_monitor.cicd_pipeline import CICDPipeline

# Initialize pipeline
pipeline = CICDPipeline(project_root)

# Execute pipeline
execution = await pipeline.execute_pipeline(
    commit_hash="abc123",
    branch="main",
    trigger="push"
)

print(f"Pipeline status: {execution.status}")
```

## Contributing

1. Follow the development workflow
2. Add tests for new features
3. Update documentation
4. Ensure all quality gates pass
5. Request code review

## Support

For issues and questions:
- Check the troubleshooting section
- Review execution logs
- Contact the DevOps team
- Create GitHub issues for bugs