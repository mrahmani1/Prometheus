# Prometheus AI Platform - Performance Testing Suite

Comprehensive performance testing infrastructure for the Prometheus AI Platform geological monitoring system, designed to validate system performance under various load conditions and detect performance regressions.

## 🎯 Overview

This performance testing suite provides comprehensive testing capabilities for:

- **Real-time seismic monitoring systems**
- **Volcanic alert and monitoring dashboards**
- **3D/4D geological visualizations**
- **Large-scale geological database operations**
- **Multi-user concurrent access scenarios**
- **Network-constrained field operations**
- **Memory-intensive long-running visualizations**

## 🏗️ Architecture

### Test Components

1. **K6 Load Testing** (`k6-load-tests.js`)
   - API endpoint performance testing
   - Seismic data processing load simulation
   - Volcanic monitoring system stress testing
   - Real-time data streaming performance

2. **Lighthouse CI** (`lighthouse-ci.js`)
   - Frontend performance monitoring
   - 3D geological visualization performance
   - WebGL rendering optimization validation
   - Accessibility and best practices compliance

3. **Database Performance Testing** (`database-performance-tests.py`)
   - PostgreSQL/PostGIS spatial query performance
   - InfluxDB time-series data ingestion
   - Neo4j graph relationship queries
   - Redis caching performance

4. **Memory Leak Detection** (`memory-leak-detection.js`)
   - Long-running visualization memory monitoring
   - WebGL resource cleanup validation
   - Real-time data processing memory patterns
   - Garbage collection analysis

5. **Network Throttling Tests** (`network-throttling-tests.js`)
   - Remote field station connectivity simulation
   - Emergency response network conditions
   - Satellite connection performance
   - Offline functionality validation

6. **Concurrent User Testing** (`concurrent-user-tests.js`)
   - Multi-user volcanic emergency scenarios
   - Seismic analyst workflow simulation
   - Field researcher data upload patterns
   - Public dashboard access patterns

7. **Resource Monitoring** (`resource-monitoring.js`)
   - System resource usage during peak loads
   - Geological service performance monitoring
   - Database connection and query monitoring
   - Alert generation for resource thresholds

8. **Regression Detection** (`regression-detection.js`)
   - Performance baseline management
   - Automated regression identification
   - Geological-specific performance metrics
   - Historical performance trend analysis

## 🚀 Quick Start

### Prerequisites

- Node.js 18.0.0+
- Python 3.11+
- K6 load testing tool
- Chrome/Chromium browser
- Access to Prometheus AI Platform services

### Installation

```bash
# Clone and navigate to performance tests
cd tests/performance

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Install K6 (if not already installed)
# macOS
brew install k6

# Ubuntu/Debian
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Windows (using Chocolatey)
choco install k6

# Create output directories
npm run setup
```

### Configuration

Create a configuration file for database connections:

```json
// tests/performance/db-config.json
{
  "postgres": {
    "host": "localhost",
    "port": 5432,
    "user": "prometheus",
    "password": "prometheus123",
    "database": "prometheus"
  },
  "influxdb": {
    "url": "http://localhost:8086",
    "token": "prometheus-token",
    "org": "prometheus",
    "bucket": "geological-data"
  },
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "prometheus123"
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "password": null
  }
}
```

## 🧪 Running Tests

### Test Suites

#### Smoke Tests (1 minute)
Quick validation of basic functionality:
```bash
npm run test:smoke
```

#### Quick Tests (5 minutes)
Essential performance validation:
```bash
npm run test:quick
```

#### Full Test Suite (30 minutes)
Comprehensive performance testing:
```bash
npm run test:full
```

### Individual Test Components

```bash
# K6 Load Testing
npm run test:k6

# Lighthouse Performance Audit
npm run test:lighthouse

# Database Performance Testing
npm run test:database

# Memory Leak Detection
npm run test:memory

# Network Throttling Tests
npm run test:network

# Concurrent User Testing
npm run test:concurrent

# Resource Monitoring
npm run test:resource

# Regression Detection
npm run test:regression
```

### Custom Configuration

```bash
# Custom base URL
BASE_URL=https://staging.prometheus-ai.com npm run test:quick

# Custom test suite with specific components
node performance-test-runner.js --suite=custom --no-database --no-network

# Specific output directory
OUTPUT_DIR=./custom-reports npm run test:full
```

## 📊 Test Results and Reports

### Output Structure

```
tests/performance/
├── reports/                          # Main test reports
│   ├── comprehensive-performance-report-*.json
│   ├── performance-dashboard-*.html
│   └── k6-results.json
├── baselines/                        # Performance baselines
│   ├── comprehensive-baseline.json
│   ├── quick-baseline.json
│   └── smoke-baseline.json
├── lighthouse-reports/               # Lighthouse audit results
│   ├── dashboard-*.json
│   ├── volcano-monitoring-*.json
│   └── summary-*.json
├── memory-reports/                   # Memory leak detection
│   ├── memory-leak-3d-geological-model-*.json
│   ├── memory-leak-report-*.html
│   └── summary-*.json
├── network-reports/                  # Network throttling results
│   ├── network-performance-report-*.html
│   └── network-throttling-report-*.json
├── concurrent-reports/               # Concurrent user testing
│   ├── concurrent-user-report-*.html
│   └── concurrent-user-report-*.json
├── resource-reports/                 # Resource monitoring
│   ├── resource-report-*.json
│   └── resource-visualization-*.html
└── regression-reports/               # Regression detection
    ├── regression-report-*.json
    └── regression-visualization-*.html
```

### Key Metrics

#### API Performance (K6)
- Response time percentiles (P50, P95, P99)
- Throughput (requests/second)
- Error rates
- Seismic processing latency
- Volcanic alert response time

#### Frontend Performance (Lighthouse)
- Performance scores for geological dashboards
- 3D visualization rendering performance
- WebGL memory usage
- Accessibility compliance
- Progressive web app capabilities

#### Database Performance
- Query execution times for geological data
- Spatial query performance (PostGIS)
- Time-series ingestion rates (InfluxDB)
- Graph traversal performance (Neo4j)
- Cache hit ratios (Redis)

#### Memory Analysis
- Memory growth patterns
- Garbage collection frequency
- WebGL resource cleanup
- Memory leak detection
- Peak memory usage

#### Network Performance
- Performance under various connection types
- Offline functionality validation
- Data compression effectiveness
- Real-time streaming resilience

#### Concurrent User Capacity
- Maximum concurrent users supported
- Performance degradation under load
- Emergency scenario handling
- Resource contention analysis

## 🎯 Geological-Specific Testing

### Seismic Monitoring Performance
- Real-time waveform processing
- Event detection algorithms
- Multi-station data correlation
- Alert generation latency

### Volcanic Monitoring Systems
- Thermal camera feed processing
- Deformation data analysis
- Gas emission monitoring
- Multi-parameter correlation

### 3D Geological Visualization
- Large geological model rendering
- Level-of-detail (LOD) performance
- Texture streaming optimization
- Interactive navigation responsiveness

### Field Data Collection
- Mobile device performance
- Offline data synchronization
- GPS tracking accuracy
- Photo/video upload optimization

## 🔧 Customization

### Adding New Test Scenarios

1. **Create test scenario file:**
```javascript
// tests/performance/custom-geological-test.js
const { check } = require('k6');
const http = require('k6/http');

export default function() {
  // Your custom geological test logic
  const response = http.get('http://localhost:3000/api/custom-endpoint');
  check(response, {
    'custom geological test passes': (r) => r.status === 200,
  });
}
```

2. **Add to test runner:**
```javascript
// In performance-test-runner.js
async runCustomGeologicalTest() {
  // Implementation
}
```

### Custom Performance Thresholds

```javascript
// In regression-detection.js config
const customThresholds = {
  seismicProcessingTime: 0.15, // 15% regression threshold
  volcanicAlertLatency: 0.10,  // 10% regression threshold
  geologicalModelRenderTime: 0.25, // 25% regression threshold
};
```

### Environment-Specific Configuration

```bash
# Development environment
NODE_ENV=development npm run test:quick

# Staging environment
NODE_ENV=staging BASE_URL=https://staging.prometheus-ai.com npm run test:full

# Production monitoring
NODE_ENV=production npm run test:smoke
```

## 📈 Performance Baselines

### Baseline Management

Performance baselines are automatically created and updated:

- **Initial Run**: Creates baseline from first test execution
- **Subsequent Runs**: Compares against baseline and updates using exponential moving average
- **Manual Baseline**: Can be manually set for specific performance targets

### Regression Detection

Automatic detection of:
- Response time regressions (>20% increase)
- Throughput degradation (>15% decrease)
- Memory usage increases (>25% growth)
- Error rate increases (>5% growth)

### Geological-Specific Regressions

Special monitoring for:
- Seismic processing performance
- Volcanic alert system latency
- 3D geological model rendering
- Real-time data streaming performance

## 🚨 Alerting and Monitoring

### Performance Alerts

Automatic alerts for:
- Critical performance regressions
- Memory leaks in visualizations
- High error rates in geological APIs
- Resource usage exceeding thresholds

### Integration with Monitoring Systems

Results can be integrated with:
- Prometheus metrics collection
- Grafana dashboards
- PagerDuty alerting
- Slack notifications

## 🛠️ Troubleshooting

### Common Issues

1. **K6 Tests Failing**
   ```bash
   # Check K6 installation
   k6 version
   
   # Verify service availability
   curl http://localhost:3000/health
   ```

2. **Lighthouse Tests Timing Out**
   ```bash
   # Increase timeout in lighthouse-ci.js
   maxWaitForLoad: 60000 // 60 seconds
   ```

3. **Database Connection Issues**
   ```bash
   # Verify database connectivity
   psql -h localhost -U prometheus -d prometheus
   ```

4. **Memory Tests Not Detecting Leaks**
   ```bash
   # Increase test duration
   TEST_DURATION=600000 npm run test:memory # 10 minutes
   ```

### Performance Optimization Tips

1. **API Performance**
   - Implement database query optimization
   - Add appropriate indexes for geological data
   - Use connection pooling
   - Implement caching strategies

2. **Frontend Performance**
   - Optimize WebGL usage for 3D models
   - Implement lazy loading for large datasets
   - Use web workers for geological calculations
   - Compress geological imagery and textures

3. **Database Performance**
   - Partition large geological datasets
   - Optimize spatial indexes (PostGIS)
   - Configure InfluxDB retention policies
   - Implement read replicas for queries

## 📚 Additional Resources

- [K6 Documentation](https://k6.io/docs/)
- [Lighthouse Performance Auditing](https://developers.google.com/web/tools/lighthouse)
- [PostgreSQL Performance Tuning](https://www.postgresql.org/docs/current/performance-tips.html)
- [InfluxDB Performance Guide](https://docs.influxdata.com/influxdb/v2.0/write-data/best-practices/)
- [WebGL Performance Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your performance tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Prometheus AI Platform Performance Testing Suite** - Ensuring reliable performance for critical geological monitoring systems.