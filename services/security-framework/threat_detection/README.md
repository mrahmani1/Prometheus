# Advanced Threat Detection and Response Service

A comprehensive security monitoring, incident response, and vulnerability management system with specialized capabilities for geological data and research environments.

## Features

### 🛡️ SIEM (Security Information and Event Management)
- **Real-time Event Processing**: Advanced correlation engine for security events
- **Behavioral Analytics**: Machine learning-based anomaly detection for geological workflows
- **Threat Intelligence Integration**: Automated threat indicator matching and analysis
- **Geological Context Awareness**: Specialized threat detection for geological data and research

### 🤖 SOAR (Security Orchestration and Automated Response)
- **Automated Incident Response**: Configurable playbooks for different threat scenarios
- **Multi-Action Coordination**: Parallel and sequential task execution with dependency management
- **Geological-Specific Responses**: Specialized response actions for geological data breaches
- **Approval Workflows**: Manual approval gates for critical response actions

### 🔍 Vulnerability Management
- **Multi-Scanner Integration**: Network (Nmap), Web App (OWASP ZAP), Container (Trivy) scanning
- **Geological Service Discovery**: Specialized scanning for geological APIs and services
- **Automated Scheduling**: Configurable scan frequencies and target management
- **Risk Prioritization**: CVSS-based severity scoring with geological context

### 🎯 Penetration Testing
- **Automated Reconnaissance**: Passive and active information gathering
- **Geological-Specific Exploits**: Specialized testing for geological data systems
- **Comprehensive Reporting**: Detailed findings with remediation guidance
- **Authorization Tracking**: Proper documentation and approval workflows

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SIEM Engine   │    │   SOAR Engine   │    │ Vulnerability   │
│                 │    │                 │    │   Manager       │
│ • Event Proc.   │    │ • Playbooks     │    │ • Scanners      │
│ • Correlation   │    │ • Automation    │    │ • Scheduling    │
│ • Analytics     │    │ • Workflows     │    │ • Reporting     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   Service       │
                    │                 │
                    │ • REST APIs     │
                    │ • WebSocket     │
                    │ • Auth/RBAC     │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis      │    │ Elasticsearch   │
│                 │    │                 │    │                 │
│ • Events        │    │ • Cache         │    │ • Log Storage   │
│ • Incidents     │    │ • Sessions      │    │ • Full-text     │
│ • Vulns         │    │ • State         │    │ • Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd services/security-framework/threat_detection
   ```

2. **Start Services**
   ```bash
   docker-compose up -d
   ```

3. **Verify Installation**
   ```bash
   curl http://localhost:8000/health
   ```

### Manual Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Database**
   ```bash
   createdb threat_detection
   psql threat_detection < init.sql
   ```

3. **Install Security Tools**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nmap
   
   # Install Trivy
   curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
   ```

4. **Start Service**
   ```bash
   python main.py
   ```

## API Documentation

### Security Events

**Submit Security Event**
```bash
POST /api/v1/security/events
Content-Type: application/json

{
  "event_type": "authentication",
  "source_ip": "192.168.1.100",
  "user_id": "user123",
  "resource": "/api/geological-data",
  "action": "login",
  "outcome": "success",
  "severity": 3,
  "geological_context": {
    "workflow_type": "seismic_analysis",
    "data_type": "earthquake_catalog"
  }
}
```

**Get Security Dashboard**
```bash
GET /api/v1/security/dashboard
Authorization: Bearer <token>
```

### Incident Response

**Approve Response**
```bash
POST /api/v1/security/response/approve
Content-Type: application/json

{
  "response_id": "resp_123",
  "approved_by": "security_analyst"
}
```

**Get Response Status**
```bash
GET /api/v1/security/response/{response_id}
Authorization: Bearer <token>
```

### Vulnerability Management

**Add Scan Target**
```bash
POST /api/v1/security/scan-targets
Content-Type: application/json

{
  "name": "Geological API Server",
  "target_type": "domain",
  "target_value": "api.geological.com",
  "scan_types": ["web_application", "network_scan"],
  "geological_service": true,
  "scan_frequency": "weekly"
}
```

**Start Vulnerability Scan**
```bash
POST /api/v1/security/scans/{target_id}/web_application
Authorization: Bearer <token>
```

### Penetration Testing

**Add Pentest Target**
```bash
POST /api/v1/security/pentest/targets
Content-Type: application/json

{
  "name": "Geological Platform",
  "target_type": "network",
  "target_scope": ["192.168.1.0/24", "geological.com"],
  "geological_context": true,
  "authorized_by": "security@company.com",
  "restrictions": ["no_dos_attacks", "business_hours_only"]
}
```

**Start Pentest Session**
```bash
POST /api/v1/security/pentest/sessions
Content-Type: application/json

{
  "name": "Q4 Security Assessment",
  "description": "Quarterly security assessment of geological platform",
  "target_ids": ["target_123"],
  "phases": ["reconnaissance", "vulnerability_assessment", "exploitation"],
  "tester_id": "pentester_001",
  "authorization_ref": "AUTH-2024-001",
  "geological_focus": true
}
```

## Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/threat_detection
REDIS_URL=redis://localhost:6379
ELASTICSEARCH_URL=http://localhost:9200

# External Tool APIs
ZAP_API_URL=http://localhost:8080
ZAP_API_KEY=your_zap_api_key
FIREWALL_API_URL=http://firewall.company.com
FIREWALL_API_KEY=your_firewall_key

# Security Tools Paths
NMAP_PATH=/usr/bin/nmap
TRIVY_PATH=/usr/local/bin/trivy

# Notification Configuration
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EMAIL_SMTP_SERVER=smtp.company.com
EMAIL_USERNAME=alerts@company.com
EMAIL_PASSWORD=your_email_password
```

### Geological Context Configuration

The system includes specialized handling for geological data and workflows:

```python
GEOLOGICAL_WORKFLOWS = [
    "seismic_analysis",
    "volcanic_monitoring", 
    "geological_modeling",
    "resource_assessment",
    "hazard_analysis"
]

GEOLOGICAL_DATA_TYPES = [
    "seismic_data",
    "volcanic_data",
    "geological_models",
    "monitoring_data",
    "survey_data"
]
```

## Security Playbooks

### Brute Force Attack Response
1. **Detection**: Multiple failed login attempts
2. **Response**: 
   - Block source IP
   - Notify security team
   - Collect forensic evidence
3. **Geological Context**: Check for geological data access attempts

### Geological Data Breach Response
1. **Detection**: Suspicious geological data access patterns
2. **Response**:
   - Disable compromised user account
   - Block source IP for 72 hours
   - Collect comprehensive forensics
   - Notify legal and CISO teams
3. **Impact Assessment**: Evaluate research data exposure

### API Abuse Response
1. **Detection**: Rapid API requests or systematic enumeration
2. **Response**:
   - Implement rate limiting
   - Block abusive IP addresses
   - Monitor for data exfiltration
3. **Geological Focus**: Protect proprietary geological models

## Monitoring and Alerting

### Metrics Collected
- Security events per minute/hour/day
- Incident response times
- Vulnerability scan coverage
- Threat detection accuracy
- Geological data access patterns

### Alert Channels
- **Email**: Critical incidents and daily summaries
- **Slack**: Real-time alerts and status updates
- **SMS**: Critical geological data breaches
- **Dashboard**: Real-time metrics and trends

### Grafana Dashboards
- Security Operations Center (SOC) Overview
- Incident Response Metrics
- Vulnerability Management Status
- Geological Security Monitoring
- Threat Intelligence Feed Status

## Compliance and Reporting

### Supported Frameworks
- **GDPR**: Data subject rights and privacy protection
- **SOC2**: Security controls and audit trails
- **ISO 27001**: Information security management
- **NIST Cybersecurity Framework**: Risk management

### Automated Reports
- Daily security summary
- Weekly vulnerability report
- Monthly incident analysis
- Quarterly penetration testing summary
- Annual compliance assessment

## Development and Testing

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Security tests
pytest tests/security/

# Load tests
pytest tests/load/
```

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run with debugging
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
```

## Troubleshooting

### Common Issues

**Service Won't Start**
- Check database connectivity
- Verify Redis is running
- Ensure all required environment variables are set

**Scans Failing**
- Verify security tools are installed (nmap, trivy)
- Check network connectivity to targets
- Validate scan target configurations

**High Memory Usage**
- Adjust Elasticsearch heap size
- Tune behavioral analytics model parameters
- Implement event retention policies

### Logs and Debugging

```bash
# View service logs
docker-compose logs -f threat-detection

# Check database connections
docker-compose exec postgres psql -U threat_user -d threat_detection

# Monitor Redis
docker-compose exec redis redis-cli monitor

# Elasticsearch health
curl http://localhost:9200/_cluster/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the security team at security@company.com
- Check the documentation wiki

---

**⚠️ Security Notice**: This system handles sensitive security data. Ensure proper access controls, encryption, and monitoring are in place before deploying to production environments.