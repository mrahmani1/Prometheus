# Prometheus Security Framework Service

Advanced data protection and encryption service for geological data in the Prometheus AI Platform.

## Overview

The Security Framework Service provides comprehensive data protection capabilities including:

- **Advanced Encryption Engine**: End-to-end encryption with HSM integration
- **Data Loss Prevention (DLP)**: Policy-based access control and monitoring
- **Secure Data Sharing**: Encrypted collaboration with role-based permissions
- **Encrypted Backup & Archival**: Multi-cloud backup with lifecycle management
- **Secure Multi-Party Computation**: Privacy-preserving collaborative analysis
- **Homomorphic Encryption**: Computation on encrypted data

## Features

### 🔐 Advanced Encryption
- AES-256-GCM, ChaCha20-Poly1305, RSA-4096, ECC-P384
- Hardware Security Module (HSM) integration
- Automatic key rotation and lifecycle management
- Field-level encryption for sensitive geological data

### 🛡️ Data Loss Prevention
- Real-time policy enforcement
- Geological data sensitivity analysis
- Automated data classification and labeling
- PII detection and masking
- Compliance monitoring (GDPR, SOX, NIST)

### 🤝 Secure Collaboration
- End-to-end encrypted data sharing
- Time-limited access tokens
- Role-based permissions
- Audit trails and access monitoring
- Multi-party collaboration workspaces

### 💾 Encrypted Backup
- Multi-tier storage (Hot, Warm, Cold, Archive)
- Cross-cloud replication (AWS S3, Azure Blob, GCP Storage)
- Point-in-time recovery
- Automated backup scheduling
- Integrity verification

### 🔬 Privacy-Preserving Analytics
- Secure multi-party computation protocols
- Differential privacy mechanisms
- Homomorphic encryption for private computation
- Federated learning for geological models
- Private set intersection

## Quick Start

### Using Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd services/security-framework

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the service
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_URL=redis://localhost:6379
export LOCAL_BACKUP_DIR=/var/backups/prometheus

# Run the service
python main.py
```

## API Documentation

### Authentication

All API endpoints require JWT authentication:

```bash
curl -H "Authorization: Bearer <your-jwt-token>" \
     http://localhost:8000/api/v1/encrypt
```

### Encryption Endpoints

#### Encrypt Data

```bash
POST /api/v1/encrypt
Content-Type: application/json

{
  "data": {
    "volcano_name": "Mount St. Helens",
    "eruption_probability": 0.75,
    "seismic_events": [...]
  },
  "algorithm": "aes_256_gcm",
  "classification": "confidential"
}
```

#### Decrypt Data

```bash
POST /api/v1/decrypt
Content-Type: application/json

{
  "encrypted_data": "48656c6c6f20576f726c64",
  "metadata": {
    "algorithm": "aes_256_gcm",
    "key_id": "key_123",
    "iv": "...",
    "tag": "...",
    "timestamp": "2024-01-01T12:00:00Z",
    "classification": "confidential"
  }
}
```

### Data Classification

#### Classify Geological Data

```bash
POST /api/v1/classify
Content-Type: application/json

{
  "data": {
    "seismic_magnitude": 7.2,
    "location": "Mount St. Helens",
    "researcher_email": "scientist@university.edu"
  },
  "data_type": "volcanic_monitoring"
}
```

Response:
```json
{
  "success": true,
  "classified_fields": {
    "seismic_magnitude": {
      "classification": "confidential",
      "encryption_required": true,
      "pii_detected": false,
      "geological_sensitivity": "high"
    },
    "researcher_email": {
      "classification": "confidential",
      "encryption_required": true,
      "pii_detected": true,
      "geological_sensitivity": "low"
    }
  }
}
```

### Data Loss Prevention

#### Evaluate Data Access

```bash
POST /api/v1/dlp/evaluate
Content-Type: application/json

{
  "access_request": {
    "user_id": "researcher_001",
    "user_roles": ["volcanologist"],
    "resource_id": "mount_st_helens_data",
    "data_type": "volcanic_monitoring",
    "operation": "read"
  },
  "context": {
    "data_source": "usgs_monitoring",
    "geological_domain": "volcanic_monitoring",
    "temporal_scope": "real-time",
    "spatial_scope": "regional",
    "research_purpose": "eruption_forecasting",
    "public_safety_impact": "critical"
  },
  "data": { ... }
}
```

### Secure Data Sharing

#### Create Data Share

```bash
POST /api/v1/sharing/create
Content-Type: application/json

{
  "owner_id": "researcher_001",
  "data_resource_id": "volcanic_data_2024",
  "data": { ... },
  "shared_with": ["researcher_002", "researcher_003"],
  "permissions": ["read", "analyze"],
  "collaboration_type": "research_project",
  "data_classification": "confidential",
  "geological_sensitivity": "disaster_preparedness",
  "duration_hours": 720
}
```

#### Access Shared Data

```bash
POST /api/v1/sharing/access
Content-Type: application/json

{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "permission": "read"
}
```

### Encrypted Backup

#### Create Backup

```bash
POST /api/v1/backup/create
Content-Type: application/json

{
  "source_path": "/data/volcanic_monitoring/2024",
  "data_classification": "confidential",
  "geological_data_type": "volcanic_monitoring",
  "backup_type": "full"
}
```

#### Restore Backup

```bash
POST /api/v1/backup/restore
Content-Type: application/json

{
  "backup_id": "backup_123",
  "restore_path": "/restore/volcanic_data",
  "user_id": "researcher_001",
  "point_in_time": "2024-01-01T12:00:00Z"
}
```

### Secure Multi-Party Computation

#### Register MPC Participant

```bash
POST /api/v1/mpc/register
Content-Type: application/json

{
  "participant_id": "usgs_001",
  "organization": "USGS",
  "role": "data_provider",
  "public_key": "-----BEGIN PUBLIC KEY-----...",
  "endpoint": "https://usgs.gov/mpc",
  "capabilities": ["shamir_secret_sharing", "differential_privacy"],
  "trust_level": 0.9,
  "data_classification_clearance": ["internal", "confidential"],
  "geological_domains": ["seismic", "volcanic"]
}
```

#### Create MPC Session

```bash
POST /api/v1/mpc/session/create
Content-Type: application/json

{
  "name": "Seismic Data Aggregation",
  "description": "Secure aggregation of seismic magnitude data",
  "protocol": "shamir_secret_sharing",
  "computation_type": "data_aggregation",
  "coordinator_id": "usgs_001",
  "participant_ids": ["usgs_001", "university_001"],
  "data_requirements": {
    "data_type": "seismic_magnitude",
    "format": "numerical"
  }
}
```

#### Execute MPC Computation

```bash
POST /api/v1/mpc/compute
Content-Type: application/json

{
  "session_id": "session_123",
  "participant_data": {
    "usgs_001": 7.2,
    "university_001": 6.8
  }
}
```

### Homomorphic Encryption

#### Encrypt for Homomorphic Operations

```bash
POST /api/v1/homomorphic/encrypt
Content-Type: application/json

{
  "data": [1.5, 2.7, 3.9, 4.1, 5.3]
}
```

#### Homomorphic Addition

```bash
POST /api/v1/homomorphic/add
Content-Type: application/json

{
  "encrypted_a": "48656c6c6f...",
  "encrypted_b": "576f726c64..."
}
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379

# HSM Configuration (optional)
HSM_LIBRARY_PATH=/usr/lib/softhsm/libsofthsm2.so
HSM_TOKEN_LABEL=prometheus
HSM_USER_PIN=1234

# Backup Storage
LOCAL_BACKUP_DIR=/var/backups/prometheus

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2
AWS_BACKUP_BUCKET=prometheus-backups

# Azure Blob Storage (optional)
AZURE_STORAGE_ACCOUNT_URL=https://account.blob.core.windows.net
AZURE_STORAGE_ACCOUNT_KEY=your_account_key
AZURE_BACKUP_CONTAINER=prometheus-backups

# Google Cloud Storage (optional)
GCP_PROJECT_ID=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCP_BACKUP_BUCKET=prometheus-backups
```

### Security Configuration

The service automatically loads security policies for:

- **Data Classification**: Automatic classification based on geological data types
- **DLP Policies**: Access control for different sensitivity levels
- **Backup Policies**: Retention and storage tier policies
- **Sharing Policies**: Collaboration and access permissions

## Security Considerations

### Data Classification Levels

- **PUBLIC**: Publicly available geological data
- **INTERNAL**: Internal research data
- **CONFIDENTIAL**: Sensitive geological information
- **RESTRICTED**: Critical infrastructure data
- **TOP_SECRET**: National security related data

### Geological Sensitivity Levels

- **PUBLIC_RESEARCH**: Open research data
- **ACADEMIC_USE**: Academic research only
- **COMMERCIAL_SENSITIVE**: Proprietary exploration data
- **DISASTER_PREPAREDNESS**: Emergency response data
- **CRITICAL_INFRASTRUCTURE**: Infrastructure protection data
- **NATIONAL_SECURITY**: National security implications

### Compliance Features

- **GDPR**: Data subject rights, consent management, data portability
- **SOX**: Financial data protection and audit trails
- **NIST**: Cybersecurity framework compliance
- **NARA**: National Archives retention requirements

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_security_framework.py::TestEncryptionEngine -v
pytest tests/test_security_framework.py::TestDLPEngine -v
pytest tests/test_security_framework.py::TestSecureSharing -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## Monitoring and Logging

The service provides comprehensive monitoring:

- **Health Checks**: `/health` endpoint for service status
- **Metrics**: Prometheus metrics for monitoring
- **Audit Logs**: Comprehensive audit trails for all operations
- **Security Events**: Real-time security event monitoring

## Performance Considerations

### Encryption Performance

- **AES-256-GCM**: ~500 MB/s encryption throughput
- **ChaCha20-Poly1305**: ~400 MB/s encryption throughput
- **RSA-4096**: ~1000 operations/second
- **ECC-P384**: ~2000 operations/second

### Backup Performance

- **Local Storage**: Limited by disk I/O
- **Cloud Storage**: Limited by network bandwidth
- **Compression**: 60-80% size reduction typical
- **Deduplication**: Available for incremental backups

### MPC Performance

- **Secret Sharing**: Linear in number of participants
- **Homomorphic Encryption**: 1000x slower than plaintext
- **Differential Privacy**: Minimal overhead
- **Federated Learning**: Depends on model size

## Troubleshooting

### Common Issues

1. **HSM Not Available**: Service falls back to software encryption
2. **Redis Connection**: Check Redis URL and connectivity
3. **Cloud Storage**: Verify credentials and permissions
4. **Key Rotation**: Automatic rotation may fail if HSM unavailable

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

### Health Checks

```bash
# Check service health
curl http://localhost:8000/health

# Check component status
curl http://localhost:8000/health | jq '.components'
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
- Contact the development team
- Check the documentation wiki

## Changelog

### v1.0.0
- Initial release with full security framework
- Advanced encryption with HSM support
- Comprehensive DLP policies
- Secure data sharing and collaboration
- Multi-cloud encrypted backup
- Secure multi-party computation
- Homomorphic encryption support