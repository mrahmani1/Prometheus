# Prometheus AI Platform - Comprehensive Testing Suite

This directory contains the comprehensive testing framework for the Prometheus AI Geodynamic Platform, implementing all requirements from task 15.1.

## 🧪 Test Suite Overview

The testing suite is organized into four main categories:

### 1. Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components and algorithms in isolation
- **Coverage**: All core AI/ML components, data models, and business logic
- **Key Features**:
  - Physics-Informed Neural Network (PINN) engine testing
  - Generative Adversarial Network (GAN) validation
  - CNN processor verification
  - Transfer learning engine testing
  - Geological data model validation

### 2. Integration Tests (`tests/integration/`)
- **Purpose**: Test end-to-end workflows and service interactions
- **Coverage**: Complete data flow from ingestion to visualization
- **Key Features**:
  - Multi-source data fusion workflows
  - Real-time monitoring pipelines
  - API to visualization workflows
  - Edge-to-cloud synchronization
  - Natural language processing integration

### 3. Performance Tests (`tests/performance/`)
- **Purpose**: Validate system scalability and performance under load
- **Coverage**: Data ingestion, AI model processing, database operations
- **Key Features**:
  - Scalability validation with increasing data volumes
  - Concurrent user load testing
  - Memory leak detection
  - GPU performance optimization
  - Database performance benchmarking

### 4. Benchmark Tests (`tests/benchmark/`)
- **Purpose**: Validate geological models against established scientific benchmarks
- **Coverage**: Real-world geological datasets and peer-reviewed studies
- **Key Features**:
  - Yellowstone seismic pattern recognition
  - Kilauea deformation cycle modeling
  - Mount Etna thermal anomaly detection
  - Multi-physics coupling accuracy
  - Uncertainty quantification validation

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r tests/requirements.txt

# Install service dependencies
pip install -r services/helios-core/requirements.txt
pip install -r services/data-ingestion/requirements.txt

# Install Node.js dependencies (for API Gateway tests)
cd services/api-gateway && npm install
```

### Running Tests

#### Using the Test Runner (Recommended)

```bash
# Run all test suites
python tests/run_tests.py all --verbose

# Run specific test categories
python tests/run_tests.py unit --verbose
python tests/run_tests.py integration --verbose
python tests/run_tests.py performance --verbose
python tests/run_tests.py benchmark --verbose

# Generate coverage report
python tests/run_tests.py coverage

# Run smoke tests (quick validation)
python tests/run_tests.py smoke

# Re-run only failed tests
python tests/run_tests.py failed
```

#### Using pytest directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -m unit -v
pytest tests/integration/ -m integration -v
pytest tests/performance/ -m performance -v
pytest tests/benchmark/ -m benchmark -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
```

## 📊 Test Configuration

### Pytest Configuration (`tests/pytest.ini`)

The test suite uses comprehensive pytest configuration including:
- Custom markers for test categorization
- Coverage reporting with 80% minimum threshold
- Timeout settings for long-running tests
- Parallel execution support
- Detailed logging configuration

### Environment Variables

Set these environment variables for testing:

```bash
export TESTING=true
export LOG_LEVEL=DEBUG
export DATABASE_URL=postgresql://user:pass@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/1
export INFLUXDB_URL=http://localhost:8086
export NEO4J_URL=bolt://localhost:7687
```

## 🏗️ Test Architecture

### Test Data Generation

The test suite includes sophisticated test data generators:

- **Seismic Data**: Synthetic earthquake catalogs with realistic magnitude distributions
- **Deformation Data**: GPS and InSAR time series with volcanic cycles
- **Thermal Data**: Satellite thermal anomaly patterns
- **Geochemical Data**: Rock and gas composition samples

### Mock Services

Comprehensive mocking framework for:
- External API services (USGS, NASA, ESA)
- Database connections
- AI model inference
- Real-time data streams

### Performance Monitoring

Built-in performance monitoring includes:
- Execution time tracking
- Memory usage profiling
- CPU utilization monitoring
- GPU memory efficiency (when available)

## 📈 Benchmark Validation

### Scientific Benchmarks

The benchmark tests validate against real-world geological datasets:

#### Yellowstone Seismic Benchmark
- **Source**: USGS Yellowstone Volcano Observatory
- **Validation Metrics**:
  - Swarm detection accuracy: ≥85%
  - Magnitude prediction RMSE: ≤0.3
  - Depth prediction RMSE: ≤2.0 km

#### Kilauea Deformation Benchmark
- **Source**: USGS Hawaiian Volcano Observatory
- **Validation Metrics**:
  - Deformation prediction RMSE: ≤1 cm
  - Inflation detection accuracy: ≥90%
  - Eruption timing accuracy: ≥80%

#### Mount Etna Thermal Benchmark
- **Source**: INGV (Italian National Institute of Geophysics and Volcanology)
- **Validation Metrics**:
  - Temperature prediction RMSE: ≤2°C
  - Anomaly detection accuracy: ≥88%
  - False positive rate: ≤10%

### Model Validation Criteria

All AI models must meet these validation criteria:
- **Accuracy**: Meet or exceed benchmark accuracy thresholds
- **Uncertainty Quantification**: Properly calibrated confidence intervals
- **Physics Compliance**: Respect fundamental physical laws
- **Robustness**: Stable performance across different geological settings

## 🔧 Continuous Integration

### GitHub Actions Workflow

The comprehensive testing suite runs automatically on:
- Every push to main/develop branches
- Pull requests
- Daily scheduled runs for benchmark tests

### Test Stages

1. **Unit Tests**: Run on multiple Python versions (3.9, 3.10, 3.11)
2. **Integration Tests**: Full service stack with databases
3. **Performance Tests**: Scalability and load testing
4. **Benchmark Tests**: Scientific validation (scheduled daily)
5. **Coverage Report**: Code coverage analysis
6. **Security Tests**: Vulnerability scanning
7. **Code Quality**: Linting and formatting checks

## 📋 Test Reports

### Coverage Reports

- **HTML Report**: `tests/coverage_html/index.html`
- **Terminal Report**: Displayed after test execution
- **JSON Report**: `tests/coverage.json` for programmatic access

### Performance Reports

Performance tests generate detailed metrics:
- Execution time per test
- Memory usage patterns
- CPU utilization
- Throughput measurements
- Scalability curves

### Benchmark Reports

Benchmark tests produce scientific validation reports:
- Model accuracy against known datasets
- Statistical significance tests
- Uncertainty quantification validation
- Physics constraint compliance

## 🛠️ Development Guidelines

### Adding New Tests

1. **Unit Tests**: Add to appropriate service directory in `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/` with proper service mocking
3. **Performance Tests**: Add to `tests/performance/` with performance monitoring
4. **Benchmark Tests**: Add to `tests/benchmark/` with scientific validation

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Fixtures and Utilities

Common fixtures are available in `tests/conftest.py`:
- Geological data samples
- AI model configurations
- Database connections
- Performance monitors

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Ensure test databases are running
   - Check connection strings in environment variables

2. **GPU Tests Failing**
   - GPU tests are automatically skipped if CUDA is not available
   - Use `@pytest.mark.skipif(not torch.cuda.is_available())` for GPU-specific tests

3. **Performance Test Timeouts**
   - Increase timeout values in `pytest.ini` if needed
   - Check system resources during test execution

4. **Benchmark Test Failures**
   - Benchmark tests may fail if model accuracy degrades
   - Review model changes and retrain if necessary

### Debug Mode

Run tests in debug mode for detailed output:

```bash
python tests/run_tests.py unit --verbose
pytest tests/ -v -s --tb=long
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Geological Testing Best Practices](docs/geological_testing_guide.md)
- [Performance Testing Guidelines](docs/performance_testing_guide.md)

## 🤝 Contributing

When contributing to the test suite:

1. Ensure all new code has corresponding tests
2. Maintain minimum 80% code coverage
3. Add benchmark tests for new geological models
4. Update documentation for new test categories
5. Follow the established testing patterns and conventions

## 📞 Support

For questions about the testing framework:
- Create an issue in the project repository
- Contact the development team
- Review the testing documentation in `docs/`