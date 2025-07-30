# Neo4j Graph Database Service

A specialized microservice for managing geological process relationships using Neo4j graph database technology. This service provides comprehensive graph modeling, analysis, and visualization capabilities for complex geological systems.

## Features

### Core Capabilities
- **Graph Models**: Complex geological process dependencies and relationships
- **Graph Algorithms**: Network analysis, centrality measures, community detection
- **Causal Discovery**: Automated causal relationship discovery and validation
- **Advanced Analytics**: Temporal evolution, anomaly detection, cascade prediction, resilience analysis
- **Process Clustering**: Spatial-temporal, process-type, and strength-based clustering
- **Visualization**: Interactive graph visualization with multiple export formats
- **Search & Query**: Full-text search and complex graph traversal queries

### Geological Process Types
- Volcanic processes (eruptions, magma movement)
- Seismic processes (earthquakes, tremors)
- Tectonic processes (plate movement, deformation)
- Hydrothermal processes (fluid circulation, mineralization)
- Magmatic processes (intrusion, differentiation)
- Metamorphic, sedimentary, and erosional processes

### Relationship Types
- Causal relationships (cause-effect chains)
- Correlational relationships (statistical associations)
- Temporal relationships (time-based sequences)
- Spatial relationships (geographic dependencies)
- Influence relationships (indirect effects)

## Architecture

### Technology Stack
- **Database**: Neo4j 5.12 with APOC and Graph Data Science plugins
- **Backend**: Python 3.11+ with FastAPI
- **Graph Processing**: Neo4j GDS algorithms and custom implementations
- **Visualization**: Cytoscape.js, GraphML, GEXF export formats
- **Deployment**: Docker containers with Kubernetes orchestration

### Service Structure
```
services/graph-database/
├── api/                    # REST API endpoints
├── core/                   # Core business logic
│   ├── database.py        # Neo4j database manager
│   ├── graph_algorithms.py # Graph analysis algorithms
│   ├── visualization.py   # Visualization engine
│   └── config.py          # Configuration settings
├── models/                 # Data models and schemas
├── tests/                  # Unit and integration tests
├── examples/               # Usage examples
└── main.py                # Service entry point
```

## API Endpoints

### Process Management
- `POST /api/v1/processes` - Create geological process
- `GET /api/v1/processes/{id}` - Get process by ID
- `GET /api/v1/processes` - Search processes

### Relationship Management
- `POST /api/v1/relationships` - Create process relationship
- `GET /api/v1/causal-chains/{id}` - Find causal chains

### Network Analysis
- `POST /api/v1/network-analysis` - Analyze process networks
- `GET /api/v1/shortest-paths` - Find shortest paths
- `GET /api/v1/critical-processes` - Identify critical processes
- `GET /api/v1/cascade-analysis/{id}` - Analyze cascade effects

### Visualization
- `POST /api/v1/visualization/config` - Create visualization config
- `GET /api/v1/visualization/{id}/data` - Get visualization data

### Query Interface
- `POST /api/v1/query` - Execute complex graph queries
- `GET /api/v1/statistics` - Get graph statistics

### Advanced Analytics
- `GET /api/v1/advanced-analytics/temporal-evolution` - Analyze network evolution over time
- `GET /api/v1/advanced-analytics/anomaly-detection` - Detect anomalous patterns
- `GET /api/v1/advanced-analytics/cascade-prediction/{id}` - Predict cascade probabilities
- `GET /api/v1/advanced-analytics/resilience-analysis` - Analyze network resilience
- `GET /api/v1/advanced-analytics/process-clustering` - Identify process clusters
- `POST /api/v1/advanced-analytics/comprehensive-report` - Generate comprehensive reports

### Causal Discovery
- `GET /api/v1/causal-discovery/hidden-relationships` - Discover hidden causal relationships
- `POST /api/v1/causal-discovery/validate-relationships` - Validate proposed relationships

## Data Models

### Geological Process
```python
{
    "process_id": "yellowstone_magma_chamber",
    "name": "Yellowstone Magma Chamber",
    "process_type": "magmatic",
    "description": "Large silicic magma chamber",
    "location_id": "yellowstone_001",
    "depth_range": [5.0, 15.0],
    "temperature_range": [750.0, 950.0],
    "pressure_range": [150.0, 400.0],
    "timescale": "thousands_of_years",
    "magnitude": 8.0,
    "confidence_level": 0.85
}
```

### Geological Relationship
```python
{
    "relationship_id": "rel_magma_hydrothermal",
    "source_process_id": "yellowstone_magma_chamber",
    "target_process_id": "yellowstone_hydrothermal",
    "relationship_type": "causal",
    "strength": 0.85,
    "confidence": 0.90,
    "mechanism": "heat_transfer_convection",
    "time_delay": 0.0,
    "spatial_scale": 15.0
}
```

## Graph Algorithms

### Centrality Measures
- **PageRank**: Importance based on relationship network
- **Betweenness**: Bridge processes in causal chains
- **Closeness**: Proximity to all other processes
- **Degree**: Direct connection count

### Community Detection
- **Louvain Algorithm**: Modularity-based community detection
- **Label Propagation**: Fast community identification
- **Simple Clustering**: Process type and connectivity-based grouping

### Path Analysis
- **Shortest Paths**: Minimum steps between processes
- **Causal Chains**: Cause-effect relationship sequences
- **Cascade Analysis**: Propagation effect modeling

### Network Metrics
- **Density**: Connection ratio in the network
- **Clustering Coefficient**: Local connectivity measure
- **Diameter**: Maximum shortest path length
- **Modularity**: Community structure quality

## Visualization Capabilities

### Layout Algorithms
- **Force-Directed**: Physics-based node positioning
- **Hierarchical**: Level-based process organization
- **Circular**: Circular arrangement of nodes

### Export Formats
- **Cytoscape.js**: Web-based interactive visualization
- **GraphML**: Standard graph exchange format
- **GEXF**: Gephi-compatible format
- **JSON**: Custom visualization data format

### Visual Properties
- Node size based on process magnitude
- Node color based on process type
- Edge width based on relationship strength
- Interactive labels and metadata

## Usage Examples

### Basic Process Creation
```python
from models.geological_models import GeologicalProcess, ProcessType

process = GeologicalProcess(
    process_id="mount_vesuvius_magma",
    name="Mount Vesuvius Magma Chamber",
    process_type=ProcessType.VOLCANIC,
    description="Primary magma chamber beneath Mount Vesuvius",
    location_id="vesuvius_001",
    depth_range=[5.0, 15.0],
    temperature_range=[800.0, 1200.0],
    confidence_level=0.85
)

# Create in database
process_id = await db_manager.create_geological_process(process)
```

### Causal Chain Analysis
```python
# Find causal chains from a trigger process
causal_chains = await db_manager.find_causal_chains(
    "yellowstone_magma_chamber", 
    max_depth=3
)

for chain in causal_chains:
    print(f"Chain strength: {chain.total_strength}")
    for process in chain.process_chain:
        print(f"  - {process['name']} ({process['process_type']})")
```

### Network Analysis
```python
from core.graph_algorithms import GeologicalGraphAlgorithms

algorithms = GeologicalGraphAlgorithms(db_manager)

# Find critical processes
critical_processes = await algorithms.find_critical_processes(
    process_ids, 
    criteria="centrality"
)

# Detect communities
communities = await algorithms.detect_communities(process_ids)

# Analyze cascade effects
cascade_analysis = await algorithms.analyze_cascade_effects(
    "trigger_process_id", 
    max_depth=5
)
```

### Visualization Creation
```python
from core.visualization import GraphVisualizationEngine
from models.geological_models import GraphVisualizationConfig

config = GraphVisualizationConfig(
    visualization_id="network_viz",
    layout_algorithm="force_directed",
    node_color_property="type",
    edge_width_property="strength",
    export_format="cytoscape"
)

viz_data = await viz_engine.create_interactive_visualization(
    nodes, edges, config
)
```

## Installation & Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=prometheus123

# Run the service
python main.py
```

### Docker Deployment
```bash
# Build container
docker build -t prometheus-graph-db .

# Run container
docker run -p 8000:8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_PASSWORD=prometheus123 \
  prometheus-graph-db
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/graph-database-deployment.yaml

# Check status
kubectl get pods -l app=graph-database
```

## Configuration

### Environment Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USER`: Database username (default: neo4j)
- `NEO4J_PASSWORD`: Database password
- `NEO4J_DATABASE`: Database name (default: prometheus)
- `API_PORT`: Service port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

### Neo4j Configuration
- Enable APOC plugin for advanced procedures
- Enable Graph Data Science plugin for algorithms
- Configure memory settings for large graphs
- Set up authentication and security

## Testing

### Run Tests
```bash
# Unit tests
pytest tests/test_database.py -v

# Integration tests
pytest tests/test_graph_algorithms.py -v

# All tests with coverage
pytest tests/ --cov=. --cov-report=html
```

### Health Checks
```bash
# Run health check
python healthcheck.py

# Check specific endpoints
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Performance Considerations

### Database Optimization
- Use appropriate indexes for frequent queries
- Optimize Cypher queries for large datasets
- Configure connection pooling
- Monitor memory usage for complex algorithms

### Scaling Strategies
- Horizontal scaling with read replicas
- Caching frequently accessed data
- Batch processing for large operations
- Asynchronous processing for long-running tasks

## Monitoring & Observability

### Metrics
- Query execution times
- Database connection pool status
- Graph algorithm performance
- API endpoint response times

### Logging
- Structured JSON logging
- Query performance logging
- Error tracking and alerting
- Audit trail for data modifications

## Security

### Authentication & Authorization
- JWT token-based authentication
- Role-based access control
- API key management
- Secure database connections

### Data Protection
- Encrypted connections (TLS)
- Sensitive data masking
- Audit logging
- Backup encryption

## Contributing

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Write comprehensive tests for new features
- Document API changes
- Use type hints for better code clarity

### Code Quality
- Run linting with flake8 and black
- Maintain test coverage above 80%
- Use meaningful commit messages
- Review code before merging

## License

This service is part of the Prometheus AI Geodynamic Platform and follows the project's licensing terms.

## Support

For technical support and questions:
- Check the examples directory for usage patterns
- Review the test files for implementation details
- Consult the API documentation at `/api/docs`
- Contact the development team for complex issues