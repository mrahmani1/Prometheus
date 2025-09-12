# Authentication Guide - Prometheus AI Platform

## Overview

The Prometheus AI Platform uses JWT (JSON Web Token) based authentication with role-based access control (RBAC) to secure API endpoints. This guide covers authentication methods, token management, and security best practices.

## Authentication Methods

### 1. JWT Token Authentication

All API requests require a valid JWT token in the Authorization header:

```http
Authorization: Bearer <your_jwt_token>
```

### 2. API Key Authentication (Service-to-Service)

For service-to-service communication, use API keys:

```http
X-API-Key: <your_api_key>
```

## Getting Started

### Step 1: Obtain Authentication Token

Make a POST request to the authentication endpoint:

```bash
curl -X POST "https://api.prometheus-ai.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "your_username",
    "role": "researcher",
    "permissions": ["read:geological_data", "write:analysis"]
  },
  "expires_in": 86400
}
```

### Step 2: Use Token in API Requests

Include the token in all subsequent API requests:

```bash
curl -X GET "https://api.prometheus-ai.com/api/v1/helios/health" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## User Roles and Permissions

### Role Hierarchy

1. **Admin** - Full system access
   - All permissions
   - User management
   - System configuration

2. **Researcher** - Scientific analysis access
   - `read:geological_data`
   - `write:analysis`
   - `read:simulations`
   - `write:simulations`

3. **Observer** - Read-only access
   - `read:geological_data`
   - `read:monitoring_data`
   - `read:public_reports`

4. **Operator** - Monitoring and alerts
   - `read:monitoring_data`
   - `write:alerts`
   - `read:system_status`

### Permission Matrix

| Endpoint | Admin | Researcher | Observer | Operator |
|----------|-------|------------|----------|----------|
| `/health` | ✅ | ✅ | ✅ | ✅ |
| `/analyze/geological` | ✅ | ✅ | ❌ | ❌ |
| `/simulate/physics` | ✅ | ✅ | ❌ | ❌ |
| `/monitor/volcano/*` | ✅ | ✅ | ✅ | ✅ |
| `/admin/*` | ✅ | ❌ | ❌ | ❌ |

## Token Management

### Token Expiration

- **Default Expiration:** 24 hours
- **Refresh Window:** 1 hour before expiration
- **Maximum Lifetime:** 7 days (with refresh)

### Token Refresh

Refresh your token before it expires:

```bash
curl -X POST "https://api.prometheus-ai.com/auth/refresh" \
  -H "Authorization: Bearer <current_token>"
```

**Response:**
```json
{
  "token": "new_jwt_token_here",
  "expires_in": 86400
}
```

### Token Revocation

Revoke a token (logout):

```bash
curl -X POST "https://api.prometheus-ai.com/auth/logout" \
  -H "Authorization: Bearer <token_to_revoke>"
```

## Security Best Practices

### 1. Token Storage

**✅ Recommended:**
- Store tokens in secure, httpOnly cookies
- Use environment variables for server-side applications
- Implement secure token storage in mobile apps

**❌ Avoid:**
- Storing tokens in localStorage (XSS vulnerability)
- Hardcoding tokens in source code
- Transmitting tokens in URL parameters

### 2. Token Transmission

**✅ Always:**
- Use HTTPS for all API communications
- Include tokens in Authorization header
- Validate SSL certificates

**❌ Never:**
- Send tokens over HTTP
- Include tokens in logs or error messages
- Share tokens between different applications

### 3. Error Handling

Handle authentication errors gracefully:

```javascript
async function makeAuthenticatedRequest(url, token) {
  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });
    
    if (response.status === 401) {
      // Token expired or invalid
      await refreshToken();
      // Retry request with new token
    } else if (response.status === 403) {
      // Insufficient permissions
      throw new Error('Access denied');
    }
    
    return response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}
```

## Rate Limiting

### Default Limits

- **Authenticated Users:** 1000 requests/hour
- **Premium Users:** 5000 requests/hour
- **Service Accounts:** 10000 requests/hour

### Rate Limit Headers

Monitor your usage with response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641024000
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_backoff(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:  # Rate limited
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue
        
        return response
    
    raise Exception("Max retries exceeded")
```

## Multi-Factor Authentication (MFA)

### Enabling MFA

1. **Setup TOTP:** Use authenticator app (Google Authenticator, Authy)
2. **Backup Codes:** Save recovery codes securely
3. **Hardware Keys:** Support for FIDO2/WebAuthn

### MFA Login Flow

```bash
# Step 1: Initial authentication
curl -X POST "https://api.prometheus-ai.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'

# Response includes MFA challenge
{
  "mfa_required": true,
  "challenge_token": "temp_token_for_mfa",
  "methods": ["totp", "sms"]
}

# Step 2: Complete MFA
curl -X POST "https://api.prometheus-ai.com/auth/mfa/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_token": "temp_token_for_mfa",
    "method": "totp",
    "code": "123456"
  }'

# Final response with JWT token
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {...},
  "expires_in": 86400
}
```

## API Key Management

### Creating API Keys

For service-to-service authentication:

```bash
curl -X POST "https://api.prometheus-ai.com/auth/api-keys" \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Monitoring Service",
    "permissions": ["read:monitoring_data", "write:alerts"],
    "expires_in": 2592000
  }'
```

**Response:**
```json
{
  "api_key": "pk_live_1234567890abcdef",
  "name": "Monitoring Service",
  "permissions": ["read:monitoring_data", "write:alerts"],
  "created_at": "2025-01-09T10:00:00Z",
  "expires_at": "2025-02-08T10:00:00Z"
}
```

### Using API Keys

```bash
curl -X GET "https://api.prometheus-ai.com/api/v1/monitor/status" \
  -H "X-API-Key: pk_live_1234567890abcdef"
```

## Troubleshooting

### Common Authentication Errors

#### 401 Unauthorized
```json
{
  "error": "Invalid token",
  "code": "INVALID_TOKEN",
  "message": "The provided token is invalid or expired"
}
```

**Solutions:**
- Check token format and encoding
- Verify token hasn't expired
- Ensure correct Authorization header format

#### 403 Forbidden
```json
{
  "error": "Insufficient permissions",
  "code": "INSUFFICIENT_PERMISSIONS",
  "message": "User role 'observer' cannot access this resource"
}
```

**Solutions:**
- Verify user role and permissions
- Request access from administrator
- Use appropriate service account

#### 429 Too Many Requests
```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 60
}
```

**Solutions:**
- Implement exponential backoff
- Reduce request frequency
- Upgrade to higher rate limit tier

### Debug Authentication Issues

Enable debug logging:

```python
import logging
import requests

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Make request with detailed logging
response = requests.get(
    'https://api.prometheus-ai.com/api/v1/helios/health',
    headers={'Authorization': 'Bearer your_token'},
    timeout=30
)

print(f"Status: {response.status_code}")
print(f"Headers: {response.headers}")
print(f"Response: {response.text}")
```

## SDK Examples

### Python SDK

```python
from prometheus_ai import PrometheusClient

# Initialize client with token
client = PrometheusClient(
    base_url="https://api.prometheus-ai.com",
    auth_token="your_jwt_token"
)

# Automatic token refresh
client.enable_auto_refresh()

# Make authenticated requests
result = client.helios.analyze_geological_data({
    "data_type": "seismic",
    "spatial_extent": {...}
})
```

### JavaScript SDK

```javascript
import { PrometheusClient } from '@prometheus-ai/sdk';

const client = new PrometheusClient({
  baseUrl: 'https://api.prometheus-ai.com',
  authToken: 'your_jwt_token',
  autoRefresh: true
});

// Make authenticated requests
const result = await client.helios.analyzeGeologicalData({
  dataType: 'seismic',
  spatialExtent: {...}
});
```

## Support

For authentication issues:

- **Documentation:** https://docs.prometheus-ai.com/auth
- **Support Email:** auth-support@prometheus-ai.com
- **Status Page:** https://status.prometheus-ai.com
- **Community Forum:** https://community.prometheus-ai.com

## Security Reporting

Report security vulnerabilities to: security@prometheus-ai.com

**Please include:**
- Detailed description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested mitigation (if any)

We follow responsible disclosure practices and will acknowledge reports within 24 hours.