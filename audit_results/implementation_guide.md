# Simulated Data Replacement Implementation Guide

## Overview

This guide provides step-by-step instructions for replacing simulated data patterns with real geological data sources across the Prometheus AI Platform backend modules.

**Audit Date:** 2025-09-14 18:34:33
**Total Patterns to Replace:** 12092
**Modules Affected:** 176

## Quick Start

1. **Review Priority Matrix**: Start with immediate priority modules
2. **Set up Real Data Connections**: Configure API credentials for geological data sources
3. **Replace Patterns Systematically**: Follow the replacement roadmap
4. **Run Validation Tests**: Ensure replacements maintain scientific accuracy
5. **Update Progress**: Track completion using monitoring tools

## Real Data Source Setup

### IRIS/EarthScope (Seismic Data)
```python
# Replace this:
seismic_data = np.random.randn(1000, 3)

# With this:
from obspy.clients.fdsn import Client
client = Client("IRIS")
seismic_data = client.get_waveforms("IU", "ANMO", "*", "BHZ", starttime, endtime)
```

### USGS Volcano Hazards Program
```python
# Replace this:
volcanic_alerts = generate_mock_alerts()

# With this:
import requests
response = requests.get("https://volcanoes.usgs.gov/vsc-api/volcanoAlerts")
volcanic_alerts = response.json()
```

### NASA EarthData
```python
# Replace this:
satellite_data = create_synthetic_satellite_data()

# With this:
from earthdata import Auth, DataGranules
auth = Auth().login(strategy="environment")
granules = DataGranules(auth).get(short_name="MOD14A1")
```

## Replacement Roadmap

### Phase 1: Immediate Priority (117 modules)

#### 1. api.py
- **Patterns:** 48
- **Effort:** 25.3 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 2. debug_bilinear_methods.py
- **Patterns:** 5
- **Effort:** 0.2 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 3. run_comprehensive_audit.py
- **Patterns:** 18
- **Effort:** 7.9 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 4. test_comprehensive_auditor.py
- **Patterns:** 46
- **Effort:** 9.2 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 5. validate_enhanced_coupling_performance.py
- **Patterns:** 11
- **Effort:** 3.8 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 6. validate_monitor_implementation.py
- **Patterns:** 6
- **Effort:** 1.6 hours
- **Critical Issues:** []

#### 7. validate_seismic_enhancement.py
- **Patterns:** 12
- **Effort:** 2.1 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 8. database_config_manager.py
- **Patterns:** 10
- **Effort:** 4.3 hours
- **Critical Issues:** []

#### 9. distributed_query_executor.py
- **Patterns:** 4
- **Effort:** 3.0 hours
- **Critical Issues:** ['random_generation', 'random_generation', 'random_generation', 'random_generation']

#### 10. placeholder_detection_engine.py
- **Patterns:** 59
- **Effort:** 33.3 hours
- **Critical Issues:** []


### Phase 2: High Priority (37 modules)

#### 1. comprehensive_simulated_data_auditor.py
- **Patterns:** 62
- **Effort:** 54.1 hours

#### 2. validate_database_implementation.py
- **Patterns:** 3
- **Effort:** 0.6 hours

#### 3. validate_gis_syntax.py
- **Patterns:** 3
- **Effort:** 0.7 hours

#### 4. validate_monitoring_network_registry.py
- **Patterns:** 2
- **Effort:** 0.2 hours

#### 5. hypothesis_testing_api.py
- **Patterns:** 7
- **Effort:** 3.1 hours

#### 6. database_backup_monitor.py
- **Patterns:** 32
- **Effort:** 22.5 hours

#### 7. database_integration.py
- **Patterns:** 4
- **Effort:** 2.2 hours

#### 8. database_optimization_engine.py
- **Patterns:** 1
- **Effort:** 0.7 hours

#### 9. placeholder_audit_orchestrator.py
- **Patterns:** 2
- **Effort:** 0.6 hours

#### 10. real_data_connectors.py
- **Patterns:** 3
- **Effort:** 1.4 hours


## Pattern Replacement Examples

### Random Data Generation
```python
# BEFORE (Simulated)
earthquake_magnitudes = np.random.uniform(1.0, 8.0, 1000)

# AFTER (Real Data)
from obspy.clients.fdsn import Client
client = Client("USGS")
catalog = client.get_events(minmagnitude=1.0, maxmagnitude=8.0)
earthquake_magnitudes = [event.magnitudes[0].mag for event in catalog]
```

### Mock Database Connections
```python
# BEFORE (Mock)
def connect_to_database():
    return MockDatabase()

# AFTER (Real)
def connect_to_database():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
```

### Synthetic Geological Data
```python
# BEFORE (Synthetic)
def generate_rock_composition():
    return {
        'SiO2': random.uniform(45, 75),
        'Al2O3': random.uniform(10, 20),
        'FeO': random.uniform(5, 15)
    }

# AFTER (Real)
def get_rock_composition(sample_id):
    response = requests.get(f"https://www.earthchem.org/portal/sample/{sample_id}")
    return response.json()['composition']
```

## Validation Checklist

For each replaced module, verify:

- [ ] No random number generation for data creation
- [ ] Real API connections established
- [ ] Error handling for network failures
- [ ] Data quality validation implemented
- [ ] Unit tests updated
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Documentation updated

## Common Pitfalls

1. **API Rate Limits**: Implement proper rate limiting and caching
2. **Data Format Changes**: Validate data structure and handle variations
3. **Network Failures**: Implement retry logic and fallback mechanisms
4. **Authentication**: Secure credential management
5. **Data Quality**: Validate real data meets scientific standards

## Monitoring Progress

Use the provided monitoring tools:

```bash
# Check overall progress
python monitoring/progress_dashboard.py

# Update module progress
python monitoring/update_progress.py module_path 0 completed

# Run validation tests
python validation_tests/run_validation_tests.py
```

## Support and Resources

- **Real Data Sources Documentation**: See `real_data_sources.md`
- **API Credentials Setup**: See `api_setup_guide.md`
- **Validation Framework**: See `validation_tests/README.md`
- **Progress Tracking**: See `monitoring/README.md`

## Next Steps

1. Review this implementation guide
2. Set up development environment with real data access
3. Begin with immediate priority modules
4. Follow systematic replacement process
5. Validate each replacement thoroughly
6. Update progress tracking regularly

---
*Generated by Comprehensive Simulated Data Auditor v1.0.0*
