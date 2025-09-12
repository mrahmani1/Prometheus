# Compliance Framework for Prometheus AI Platform

This comprehensive compliance framework provides enterprise-grade compliance management for the Prometheus AI geological research platform, supporting GDPR, SOC2, ISO27001, and other regulatory requirements.

## Features

### 🔒 GDPR Compliance
- **Data Subject Rights**: Automated processing of access, rectification, erasure, and portability requests
- **Consent Management**: Cryptographic consent recording with audit trails
- **Privacy by Design**: Built-in privacy impact assessments and data protection controls
- **Automated Compliance Monitoring**: Real-time consent validity checking and violation detection

### 🛡️ SOC2 Type II Compliance
- **Trust Service Criteria**: Complete coverage of Security, Availability, Processing Integrity, Confidentiality, and Privacy
- **Automated Control Testing**: Continuous monitoring and testing of security controls
- **Evidence Collection**: Automated evidence gathering and control effectiveness measurement
- **Compliance Reporting**: Comprehensive SOC2 audit reports with control testing results

### 📋 ISO27001 ISMS
- **Risk Management**: Comprehensive risk assessment and treatment planning
- **Security Controls**: Implementation and monitoring of Annex A security controls
- **Incident Management**: Security incident tracking and response workflows
- **Continuous Improvement**: PDCA cycle implementation with automated metrics

### 📊 Tamper-Proof Audit Logging
- **Blockchain-Style Chaining**: Cryptographically linked audit blocks for tamper detection
- **Comprehensive Event Tracking**: All system activities logged with geological context
- **Integrity Verification**: Automated audit log integrity checking and violation detection
- **Compliance Reporting**: Automated compliance reports for multiple frameworks

### ⏰ Data Retention Management
- **Automated Policy Enforcement**: Time-based and event-based retention triggers
- **Legal Hold Management**: Comprehensive legal hold tracking and release workflows
- **Secure Deletion**: Multiple deletion methods including cryptographic erasure
- **Compliance Tracking**: Retention compliance monitoring and reporting

### 📈 Real-Time Compliance Dashboard
- **Unified View**: Single pane of glass for all compliance frameworks
- **Interactive Visualizations**: Real-time charts, gauges, and heatmaps
- **Automated Alerting**: Intelligent compliance violation detection and notification
- **Executive Reporting**: Automated compliance reports and executive dashboards

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Compliance Dashboard                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │    GDPR     │ │    SOC2     │ │      ISO27001 ISMS      ││
│  │ Compliance  │ │ Compliance  │ │                         ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Compliance Monitoring Engine                    │
│  ┌─────────────────┐ ┌─────────────────────────────────────┐│
│  │  Policy Engine  │ │     Violation Detection             ││
│  │                 │ │                                     ││
│  └─────────────────┘ └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Audit Logging System                        │
│  ┌─────────────────┐ ┌─────────────────────────────────────┐│
│  │ Tamper-Proof    │ │    Data Retention Engine            ││
│  │ Audit Logger    │ │                                     ││
│  └─────────────────┘ └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ PostgreSQL  │ │    Redis    │ │    Encrypted Storage    ││
│  │             │ │             │ │                         ││
│  └─────────────┘ └─────────────┘ └─────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd services/security-framework/compliance
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set environment variables**
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/prometheus"
export REDIS_URL="redis://localhost:6379"
export LOG_LEVEL="INFO"
```

4. **Initialize database schemas**
```bash
python -c "
import asyncio
from main import ComplianceFrameworkService
async def init():
    service = ComplianceFrameworkService()
    await service.initialize()
asyncio.run(init())
"
```

5. **Start the service**
```bash
python main.py
```

### Docker Deployment

1. **Start with Docker Compose**
```bash
docker-compose up -d
```

2. **View logs**
```bash
docker-compose logs -f compliance-service
```

3. **Access management tools** (optional)
```bash
# Start management tools
docker-compose --profile tools up -d

# Access pgAdmin: http://localhost:8080
# Access Redis Commander: http://localhost:8081
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://prometheus:prometheus123@localhost:5432/prometheus` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Deployment environment | `development` |

### Compliance Policies

Compliance policies are automatically initialized with sensible defaults for geological research platforms. You can customize policies by modifying the initialization methods in each compliance engine.

## API Usage

### GDPR Compliance

```python
from compliance import GDPRComplianceEngine

# Record user consent
consent_record = ConsentRecord(
    consent_id="consent_123",
    data_subject_id="user_456",
    purpose="geological_research",
    lawful_basis=ProcessingLawfulBasis.CONSENT,
    consent_text="I consent to processing my location data for geological research",
    timestamp=datetime.utcnow(),
    status=ConsentStatus.GIVEN,
    # ... other fields
)

success = await gdpr_engine.record_consent(consent_record)

# Process data subject request
request = DataSubjectRequest(
    request_id="req_789",
    data_subject_id="user_456",
    request_type=DataSubjectRights.ACCESS,
    request_date=datetime.utcnow(),
    # ... other fields
)

result = await gdpr_engine.process_data_subject_request(request)
```

### SOC2 Compliance

```python
from compliance import SOC2ComplianceEngine

# Execute control test
test_result = await soc2_engine.execute_control_test("CC6.1", "automated_system")

# Generate SOC2 report
report = await soc2_engine.generate_soc2_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

### Audit Logging

```python
from compliance import TamperProofAuditLogger, AuditEvent, AuditEventType

# Log audit event
event = AuditEvent(
    event_id="evt_123",
    timestamp=datetime.utcnow(),
    event_type=AuditEventType.DATA_ACCESS,
    severity=AuditSeverity.INFO,
    user_id="user_456",
    resource="geological_dataset_789",
    action="read",
    outcome="success",
    # ... other fields
)

await audit_logger.log_audit_event(event)

# Verify audit integrity
verification = await audit_logger.verify_audit_integrity(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow()
)
```

## Monitoring and Alerting

### Compliance Dashboard

Access the compliance dashboard to monitor:
- Real-time compliance scores
- Active violations and incidents
- Policy effectiveness metrics
- Risk assessment matrices
- Data retention compliance

### Automated Alerts

The system automatically generates alerts for:
- Compliance violations
- Policy failures
- Audit integrity issues
- Data retention violations
- Security incidents

### Metrics and Reporting

Key metrics tracked:
- Overall compliance score
- Framework-specific scores (GDPR, SOC2, ISO27001)
- Violation counts by severity
- Incident response times
- Control effectiveness ratings
- Data retention compliance rates

## Security Features

### Cryptographic Protection
- **Digital Signatures**: RSA-2048 signatures for audit log integrity
- **Encryption**: AES-256 encryption for sensitive data at rest
- **Key Management**: Secure key generation and rotation
- **Tamper Detection**: Blockchain-style chaining for audit logs

### Access Control
- **Role-Based Permissions**: Granular access control for compliance functions
- **Audit Trails**: Complete audit trails for all compliance activities
- **Multi-Factor Authentication**: MFA support for privileged operations
- **Zero-Trust Architecture**: Continuous verification and least privilege access

### Data Protection
- **Privacy by Design**: Built-in privacy controls and impact assessments
- **Data Minimization**: Automated data retention and deletion policies
- **Consent Management**: Cryptographic consent recording and validation
- **Anonymization**: Secure data anonymization and pseudonymization

## Compliance Frameworks

### GDPR (General Data Protection Regulation)
- ✅ Data Subject Rights (Articles 15-22)
- ✅ Consent Management (Article 7)
- ✅ Privacy by Design (Article 25)
- ✅ Data Protection Impact Assessments (Article 35)
- ✅ Breach Notification (Articles 33-34)
- ✅ Data Retention and Deletion (Article 5)

### SOC2 Type II
- ✅ Security (CC6.0 series)
- ✅ Availability (A1.0 series)
- ✅ Processing Integrity (PI1.0 series)
- ✅ Confidentiality (C1.0 series)
- ✅ Privacy (P1.0 series)

### ISO27001:2013
- ✅ Information Security Management System (ISMS)
- ✅ Risk Assessment and Treatment (Clause 6)
- ✅ Security Controls (Annex A)
- ✅ Incident Management (A.16)
- ✅ Business Continuity (A.17)
- ✅ Compliance Monitoring (A.18)

## Development

### Running Tests
```bash
pytest tests/ -v --cov=compliance
```

### Code Quality
```bash
# Format code
black compliance/
isort compliance/

# Lint code
flake8 compliance/
mypy compliance/
```

### Documentation
```bash
# Generate documentation
sphinx-build -b html docs/ docs/_build/
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify PostgreSQL is running and accessible
   - Check DATABASE_URL environment variable
   - Ensure database user has necessary permissions

2. **Redis Connection Errors**
   - Verify Redis is running and accessible
   - Check REDIS_URL environment variable
   - Ensure Redis is configured for persistence

3. **Compliance Policy Errors**
   - Check policy configuration in initialization methods
   - Verify database schemas are properly initialized
   - Review compliance engine logs for specific errors

### Logging

Logs are structured and include:
- Timestamp and log level
- Component name and function
- Compliance framework context
- Error details and stack traces

### Performance Tuning

For high-volume environments:
- Increase database connection pool size
- Enable Redis clustering
- Implement audit log partitioning
- Use read replicas for reporting queries

## Support

For support and questions:
- Review the troubleshooting section
- Check the logs for error details
- Consult the API documentation
- Contact the security team for compliance questions

## License

This compliance framework is part of the Prometheus AI Platform and is subject to the platform's licensing terms.

## Contributing

1. Follow the coding standards (Black, isort, flake8)
2. Write comprehensive tests for new features
3. Update documentation for API changes
4. Ensure compliance with security requirements
5. Submit pull requests with detailed descriptions