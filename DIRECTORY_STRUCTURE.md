# Prometheus Platform - Directory Structure

## Current Project Organization

```
prometheus/
├── 📁 frontend/                    # Web UI Components
│   ├── 🌐 html/                   # HTML Pages
│   │   ├── advanced_insar_analysis.html
│   │   ├── volcanic_system_simulator.html
│   │   ├── mantle_dynamics_simulator.html
│   │   ├── ml_dashboard.html
│   │   └── [other visualization pages]
│   │
│   ├── 📊 js/                     # JavaScript Modules
│   │   ├── core/                  # Core functionality
│   │   │   ├── prometheus_enhanced_loader.js
│   │   │   ├── modern_dashboard_framework.js
│   │   │   ├── prometheus_components.js
│   │   │   └── theme_manager.js
│   │   │
│   │   ├── visualization/         # Visualization modules
│   │   │   ├── tomography_3d.js
│   │   │   ├── deformation_3d.js
│   │   │   ├── fringe_visualizer.js
│   │   │   ├── stress_visualizer.js
│   │   │   └── ml_visualizer.js
│   │   │
│   │   ├── algorithms/            # Scientific algorithms
│   │   │   ├── insar_algorithms.js
│   │   │   ├── seismic_ml_models.js
│   │   │   ├── volcano_ml_predictor.js
│   │   │   └── risk_calculator.js
│   │   │
│   │   ├── data/                  # Data management
│   │   │   ├── cache_system.js
│   │   │   ├── sar_data_manager.js
│   │   │   ├── velocity_model.js
│   │   │   └── comprehensive_volcano_database.js
│   │   │
│   │   ├── performance/           # Performance optimization
│   │   │   ├── performance_manager.js
│   │   │   ├── processing_queue.js
│   │   │   └── auto_optimizer.js
│   │   │
│   │   └── fixes/                 # Bug fixes and patches
│   │       ├── critical_ui_fixes.js
│   │       ├── ml_dashboard_fix.js
│   │       └── error_management_system.js
│   │
│   └── 🎨 css/                    # Stylesheets
│       └── component_styles.css
│
├── 📁 services/                    # Microservices (Python Backend)
│   ├── helios-core/               # Core AI engine
│   ├── data-ingestion/            # Data ingestion service
│   ├── volcano-watchtower/        # Volcanic monitoring
│   ├── anomaly-detection/         # Anomaly detection service
│   ├── system-monitor/            # System monitoring
│   ├── api-gateway/               # API gateway (Node.js)
│   └── vulcan-ui/                 # React UI (if used)
│
├── 📁 kubernetes/                  # Kubernetes configurations
│   ├── deployments/
│   ├── services/
│   └── configmaps/
│
├── 📁 database/                    # Database schemas
│   ├── postgres/
│   └── timescale/
│
├── 📁 scripts/                     # Utility scripts
│   ├── deployment/
│   ├── quality/
│   └── testing/
│
├── 📁 tests/                       # Test suites
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── 📁 docs/                        # Documentation
│   ├── api/
│   ├── user-guides/
│   └── development/
│
├── 📁 deprecated/                  # Old/unused files
│   ├── old_html/
│   └── backups/
│
└── 📁 config/                      # Configuration files
    ├── docker-compose.yml
    ├── prometheus-training_key.pem
    └── requirements.txt
```

## Recommended Reorganization

### Phase 1: Separate Frontend and Backend
1. Move all `.js` files to `frontend/js/` subdirectories
2. Move all `.html` files to `frontend/html/`
3. Keep Python services in `services/`

### Phase 2: Module Organization
1. Group JavaScript modules by functionality
2. Create clear separation between core, algorithms, and UI
3. Consolidate duplicate functionality

### Phase 3: Clean Up Root Directory
1. Move scripts to appropriate subdirectories
2. Archive deprecated files
3. Organize configuration files

## File Categories

### Frontend JavaScript Modules (Completed Tasks 1-6.1)

#### Core Infrastructure
- prometheus_enhanced_loader.js
- modern_dashboard_framework.js
- prometheus_components.js
- theme_manager.js
- performance_manager.js
- cache_system.js

#### Scientific Algorithms
- insar_algorithms.js
- seismic_ml_models.js
- volcano_ml_predictor.js
- risk_calculator.js
- velocity_model.js

#### Visualization
- tomography_3d.js
- deformation_3d.js
- fringe_visualizer.js
- stress_visualizer.js
- ml_visualizer.js
- mantle_convection_model.js

#### Data Management
- comprehensive_volcano_database.js
- sar_data_manager.js
- coherence_analyzer.js
- processing_queue.js

#### ML/AI Integration
- eqtransformer_integration.js
- eruption_forecaster.js
- auto_optimizer.js
- model_ensemble.js
- model_metrics_dashboard.js

### Backend Python Services

#### Core Services
- helios-core: Physics-informed neural networks
- data-ingestion: Real-time data processing
- volcano-watchtower: Volcanic monitoring
- anomaly-detection: Pattern detection

#### Support Services
- system-monitor: Health monitoring
- edge-device-manager: IoT integration
- knowledge-graph: Graph database
- model-training-orchestrator: ML training

### HTML Pages (UI Entry Points)
- advanced_insar_analysis.html
- volcanic_system_simulator.html
- mantle_dynamics_simulator.html
- ml_dashboard.html
- professional_tectonic_simulation.html

## Integration Status

### ✅ Completed (Tasks 1-6.1)
- Critical UI fixes
- Modern dashboard framework
- InSAR enhancement
- ML integration
- Advanced geophysics features
- Performance optimization

### 🔄 In Progress (Task 6.2)
- Comprehensive testing suite
- Documentation system
- Azure deployment optimization

### 📋 Planned
- Full microservices integration
- Kubernetes deployment
- Production monitoring
- API documentation

## Key Technologies

### Frontend
- Three.js for 3D visualization
- Chart.js for data plotting
- Leaflet/Mapbox for mapping
- WebWorkers for parallel processing
- IndexedDB for caching

### Backend
- Python 3.11+ with FastAPI
- PyTorch for deep learning
- PostgreSQL with PostGIS
- InfluxDB for time series
- Redis for caching
- Docker/Kubernetes for deployment

## Notes
- The project has evolved from a Python-centric platform to a hybrid system
- Recent work (Tasks 1-6) focused on JavaScript UI/UX improvements
- Backend services remain in Python with microservices architecture
- Integration between frontend and backend via REST APIs