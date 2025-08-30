# Claude Code Configuration

## Search Configuration

### Semantic Search with claude-context MCP
- **Primary Search Method**: Use semantic search via claude-context MCP for all codebase searches
- **Command**: Use `mcp__claude-context__search_code` tool for code searches instead of manual file searching
- **Benefits**: More efficient, context-aware code discovery and understanding
- **Indexing**: Ensure codebase is indexed using `mcp__claude-context__index_codebase` before searching

### Search Best Practices
1. Always use semantic search first for code understanding tasks
2. Use natural language queries to describe what you're looking for
3. Leverage semantic search for class relationships and method locations
4. Fall back to grep/glob only for specific string matches when needed

### Development Workflow
1. Index codebase if not already indexed
2. Use semantic search to understand existing code structure
3. Implement changes with full context awareness
4. Test implementations thoroughly

## Code Quality Standards

### No Placeholder/Mock Code Policy
- **NEVER implement simplified or placeholder features**
- **NEVER use mock data or dummy implementations**
- **ALL code must be production-ready with real functionality**
- **ALL integrations must connect to actual data sources and APIs**
- **ALL algorithms must be fully implemented, not simplified versions**
- **ALL database queries must be complete and functional**
- **ALL ML models must be real implementations, not placeholders**

### Implementation Requirements
- Use real external APIs and data sources
- Implement complete error handling and validation
- Include comprehensive logging and monitoring
- Add proper configuration management
- Ensure scalability and performance optimization

## Testing Commands
- Run tests and ensure code quality before completing tasks
- Use appropriate test frameworks based on project structure

## Claude Code Settings
Note: Claude Code settings.local.json already includes permissions for:
- `mcp__claude-context__index_codebase` (line 71)
- `mcp__claude-context__search_code` (line 83)

These permissions enable semantic search capabilities. The claude-context MCP is properly configured and ready for use.

## Project Structure

### Backend Modules (`backend/modules/`)
**Core Geological Modeling:**
- `3d_geological_modeler.py` - 3D geological model construction
- `3d_magmatic_system_modeler.py` - Magmatic system modeling
- `elastic_inversion_models.py` - Elastic deformation analysis
- `deformation_analyzer.py` - Deformation field analysis
- `coulomb_stress_modeler.py` - Stress field modeling
- `geological_constraints.py` - Geological constraint integration

**Seismic & Geophysical Analysis:**
- `seismic_transformer.py` - AI-powered seismic analysis
- `ambient_noise_engine.py` - Ambient noise processing
- `moment_tensor_inversion.py` - Earthquake source analysis
- `fwi_engine.py` - Full waveform inversion
- `insar_processor_ml.py` - ML-based InSAR processing
- `array_processor_ml.py` - Seismic array processing

**Volcanic Systems:**
- `eruption_forecaster_ml.py` - ML eruption forecasting
- `caldera_system_classifier.py` - Caldera classification
- `resurgent_caldera_monitoring.py` - Caldera monitoring
- `integrated_caldera_monitoring.py` - Integrated monitoring
- `volcanic_multi_chamber.py` - Multi-chamber modeling
- `supereruption_analysis.py` - Large eruption analysis
- `magma_reservoir_monitoring.py` - Reservoir monitoring

**Advanced ML & AI:**
- `deep_earth_ml.py` - Deep learning geological models
- `geochemical_ml.py` - ML geochemical analysis
- `source_characterization_ml.py` - Source characterization
- `geological_gnn.py` - Graph neural networks
- `large_scale_geological_graphsage.py` - Large-scale GNN
- `geological_anomaly_detection.py` - Anomaly detection
- `causal_inference_engine.py` - Causal analysis
- `geological_causal_discovery.py` - Causal discovery

**Data Integration & Processing:**
- `data_integration_pipeline.py` - Multi-source data integration
- `database_integration.py` - Database connectivity
- `model_database.py` - Model storage and retrieval
- `external_data_connectors.py` - External API connections
- `multi_sensor_fusion.py` - Sensor data fusion
- `stream_processing.py` - Real-time data streams

**Digital Twin System:**
- `digital_twin_earth.py` - Main digital twin engine
- `digital_twin_api.py` - API interfaces
- `digital_twin_integration.py` - System integration
- `digital_twin_monitoring.py` - Monitoring systems
- `digital_twin_multiuser.py` - Multi-user support
- `digital_twin_visualization.py` - Visualization engine
- `digital_twin_data_management.py` - Data management
- `digital_twin_performance_optimization.py` - Performance optimization

**Specialized Analysis:**
- `thermodynamic_engine.py` - Thermodynamic modeling
- `thermodynamic_integration.py` - Thermal integration
- `hydrothermal_flow_engine.py` - Hydrothermal systems
- `mantle_convection_3d.py` - Mantle dynamics
- `geodynamo_simulator.py` - Geodynamo modeling
- `plate_evolution_engine.py` - Plate tectonics evolution
- `isotope_modeler.py` - Isotopic analysis

**Advanced Algorithms:**
- `advanced_methods.py` - Advanced computational methods
- `adaptive_model_selection.py` - Model selection algorithms
- `enhanced_kalman_filter.py` - Advanced filtering
- `joint_inversion_framework.py` - Multi-parameter inversion
- `optimization_framework.py` - Optimization algorithms
- `genetic_model_integrator.py` - Genetic algorithms

**Machine Learning Infrastructure:**
- `mlops_pipeline.py` - ML operations pipeline
- `federated_learning.py` - Distributed learning
- `representation_learning_integration.py` - Representation learning
- `geological_network_representation_learning.py` - Network learning
- `model_manager.py` - Model lifecycle management

**Computational Infrastructure:**
- `parallel_computing.py` - Parallel processing
- `edge_computing.py` - Edge computing support
- `high_resolution_visualization.py` - Visualization engine
- `multi_physics_coupling_engine.py` - Multi-physics coupling

**Economic & Risk Analysis:**
- `economic_optimizer.py` - Economic optimization
- `resource_evaluator.py` - Resource assessment
- `prospectivity_engine.py` - Mineral prospectivity
- `hazard_interactions.py` - Multi-hazard analysis
- `cap_rock_stability_analyzer.py` - Cap rock analysis

**Knowledge Systems:**
- `scientific_knowledge_graph.py` - Knowledge graph
- `hypothesis_generation_system.py` - AI hypothesis generation
- `hidden_relationship_discovery.py` - Pattern discovery
- `autonomous_geoscience.py` - Autonomous research
- `scenario_engine.py` - Scenario modeling

**Supporting Utilities:**
- `geological_data_structures.py` - Data structures
- `geological_statistics.py` - Statistical analysis
- `geological_complexity_assessment.py` - Complexity metrics
- `stress_field_mapper.py` - Stress mapping
- `target_generator.py` - Target generation
- `satellite_scheduler.py` - Satellite operations