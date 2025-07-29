# Prometheus AI API Gateway

The external API gateway for the Prometheus AI Geodynamic Platform, providing secure access to geological data, simulation services, and real-time monitoring capabilities.

## Features

### 🔐 Security
- **OAuth2 Authentication**: Full OAuth2 authorization server with support for authorization code, refresh token, and client credentials flows
- **JWT Token Management**: Secure token generation, validation, and revocation
- **API Key Authentication**: Support for API key-based authentication for external systems
- **Data Encryption**: End-to-end encryption for sensitive data transmission
- **Rate Limiting**: Intelligent rate limiting with tier-based quotas
- **CORS Protection**: Configurable CORS policies for secure cross-origin requests

### 🚀 API Endpoints
- **RESTful API**: Comprehensive REST API for geological data and simulations
- **GraphQL API**: Flexible GraphQL endpoint for complex data queries
- **Real-time Subscriptions**: WebSocket-based subscriptions for live data streams
- **Batch Operations**: Support for bulk data uploads and batch processing

### 📊 Monitoring & Analytics
- **Usage Tracking**: Detailed API usage analytics and reporting
- **Performance Monitoring**: Request/response time tracking and optimization
- **Error Tracking**: Comprehensive error logging and alerting
- **Health Checks**: Multiple health check endpoints for system monitoring

### 🔄 Data Integration
- **Multi-format Support**: JSON, XML, CSV, and binary data formats
- **Data Validation**: Comprehensive input validation and sanitization
- **Compression**: Automatic data compression for efficient transmission
- **Caching**: Intelligent caching strategies for improved performance

## Quick Start

### Prerequisites
- Node.js 18.0.0 or higher
- Redis server for rate limiting and caching
- PostgreSQL database (for production)

### Installation

```bash
# Clone the repository
git clone https://github.com/prometheus-ai/geodynamic-platform.git
cd geodynamic-platform/services/api-gateway

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Generate encryption keys
npm run keys:generate

# Start the development server
npm run dev
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t prometheus-ai/api-gateway .

# Run the container
docker run -d \
  --name prometheus-api-gateway \
  -p 8080:8080 \
  -e NODE_ENV=production \
  -e REDIS_HOST=redis \
  -e JWT_SECRET=your-jwt-secret \
  prometheus-ai/api-gateway
```

## API Documentation

### REST API Endpoints

#### Authentication
- `POST /oauth2/token` - Obtain access token
- `GET /oauth2/authorize` - OAuth2 authorization endpoint
- `POST /oauth2/revoke` - Revoke access token
- `POST /oauth2/introspect` - Token introspection

#### Geological Data
- `GET /api/external/v1/geological-data` - Query geological data
- `POST /api/external/v1/geological-data` - Submit geological data
- `GET /api/external/v1/monitoring/alerts` - Get volcanic alerts

#### Simulations
- `POST /api/external/v1/simulations` - Create simulation
- `GET /api/external/v1/simulations/{id}` - Get simulation status

### GraphQL API

Access the GraphQL playground at `/api/external/graphql` in development mode.

#### Example Queries

```graphql
# Query geological data
query GetSeismicEvents {
  seismicEvents(
    startTime: "2023-01-01T00:00:00Z"
    endTime: "2023-12-31T23:59:59Z"
    minMagnitude: 4.0
    limit: 100
  ) {
    id
    timestamp
    location {
      latitude
      longitude
      depth
    }
    magnitude
    eventType
  }
}

# Create a simulation
mutation CreateSimulation {
  createSimulation(
    input: {
      type: VOLCANIC_MONITORING
      parameters: {
        volcanoId: "kilauea"
        timeSpan: "30d"
        resolution: "1h"
      }
    }
  ) {
    id
    status
    estimatedCompletion
  }
}

# Subscribe to real-time alerts
subscription AlertStream {
  alertStream(minAlertLevel: 3) {
    id
    volcanoId
    alertLevel
    description
    timestamp
  }
}
```

## Authentication

### OAuth2 Flow

1. **Authorization Request**
```bash
GET /oauth2/authorize?response_type=code&client_id=your-client-id&redirect_uri=https://your-app.com/callback&scope=read:geological-data
```

2. **Token Exchange**
```bash
POST /oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=AUTH_CODE&redirect_uri=https://your-app.com/callback&client_id=your-client-id&client_secret=your-client-secret
```

3. **API Request**
```bash
GET /api/external/v1/geological-data
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### API Key Authentication

```bash
GET /api/external/v1/geological-data
X-API-Key: your-api-key
```

## Rate Limiting

The API implements tier-based rate limiting:

| Tier | Requests/15min | Data Upload/min | Simulations/hour |
|------|----------------|-----------------|------------------|
| Free | 100 | 5 | 1 |
| Basic | 1,000 | 50 | 10 |
| Premium | 10,000 | 500 | 100 |
| Enterprise | 100,000 | 5,000 | 1,000 |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Request limit for the current window
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Time when the current window resets

## Data Encryption

### Request Encryption

For sensitive data, you can encrypt requests:

```javascript
// Encrypt request body
const encryptedData = await encrypt(requestData, password);

fetch('/api/external/v1/geological-data', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Encoding': 'encrypted',
    'Authorization': 'Bearer ' + token
  },
  body: JSON.stringify(encryptedData)
});
```

### Response Encryption

Request encrypted responses:

```javascript
fetch('/api/external/v1/geological-data', {
  headers: {
    'Accept-Encoding': 'encrypted',
    'Authorization': 'Bearer ' + token
  }
});
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

```json
{
  "error": "validation_failed",
  "message": "Request validation failed",
  "details": [
    {
      "field": "latitude",
      "message": "Must be between -90 and 90"
    }
  ],
  "requestId": "req_123456789",
  "timestamp": "2023-06-15T10:30:00Z"
}
```

### Common Error Codes

- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

## Monitoring

### Health Checks

- `GET /health` - Basic health check
- `GET /ready` - Readiness check for Kubernetes
- `GET /health/rate-limiting` - Rate limiting system health

### Usage Analytics

```bash
GET /analytics/usage?period=day&endpoint=geological-data
Authorization: Bearer YOUR_TOKEN
```

Response:
```json
{
  "period": "day",
  "endpoint": "geological-data",
  "tier": "premium",
  "usage": {
    "requests": 1250,
    "averageResponseTime": 145,
    "totalRequestSize": 2048576,
    "totalResponseSize": 8192000,
    "errors": 12,
    "errorRate": "0.96%"
  },
  "limits": {
    "requests": 10000,
    "window": 900000
  }
}
```

## Configuration

### Environment Variables

```bash
# Server Configuration
NODE_ENV=production
PORT=8080
LOG_LEVEL=info

# Security
JWT_SECRET=your-super-secret-jwt-key
ENCRYPTION_PASSWORD=your-encryption-password

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prometheus
POSTGRES_USER=prometheus
POSTGRES_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=password

# External Services
USGS_API_KEY_HASH=hash-of-usgs-api-key
NASA_API_KEY_HASH=hash-of-nasa-api-key
ESA_API_KEY_HASH=hash-of-esa-api-key

# CORS
FRONTEND_URL=https://app.prometheus-ai.com
```

## Development

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Code Quality

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Security audit
npm run security:audit
```

### Key Management

```bash
# Generate new encryption keys
npm run keys:generate

# Rotate existing keys
npm run keys:rotate
```

## Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: prometheus-ai/api-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: NODE_ENV
          value: "production"
        - name: REDIS_HOST
          value: "redis-service"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Docker Compose

```yaml
version: '3.8'
services:
  api-gateway:
    build: .
    ports:
      - "8080:8080"
    environment:
      - NODE_ENV=production
      - REDIS_HOST=redis
    depends_on:
      - redis
      - postgres
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: prometheus
      POSTGRES_USER: prometheus
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://prometheus-ai.com/docs/api
- Issues: https://github.com/prometheus-ai/geodynamic-platform/issues
- Email: api-support@prometheus-ai.com
- Slack: #api-support in Prometheus AI workspace