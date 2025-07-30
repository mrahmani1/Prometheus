# Causal Inference Engine Documentation

## Overview

The Causal Inference Engine is a sophisticated component of the Prometheus AI platform that discovers causal relationships in geological data using state-of-the-art causal discovery algorithms. It implements PC (Peter-Clark), FCI (Fast Causal Inference), and GES (Greedy Equivalence Search) algorithms with geological domain knowledge validation.

## Features

### Core Capabilities

1. **Causal Discovery Algorithms**
   - PC Algorithm: Constraint-based causal discovery using conditional independence tests
   - FCI Algorithm: Handles latent confounders and selection bias
   - GES Algorithm: Score-based approach using Bayesian Information Criterion

2. **Geological Domain Knowledge Integration**
   - Known causal relationships validation
   - Impossible relationships filtering
   - Temporal constraints enforcement
   - Physical constraints validation

3. **Intervention Analysis**
   - Do-calculus approximation for causal effect estimation
   - Backdoor adjustment for confounding control
   - Confidence interval calculation using bootstrap methods

4. **Hypothesis Generation**
   - Automated generation of testable scientific hypotheses
   - Experimental design suggestions
   - Geological interpretation of discovered relationships

5. **Validation and Quality Assessment**
   - Independent data validation
   - Domain knowledge consistency checking
   - Statistical significance testing

## API Endpoints

### Causal Discovery

```http
POST /causal-inference/discover
```

Discover causal relationships in geological data.

**Request Body:**
```json
{
  "algorithm": "pc",
  "confidence_threshold": 0.05,
  "data_source": "volcanic_monitoring",
  "variables": ["magma_pressure", "ground_deformation", "gas_emissions"]
}
```

**Response:**
```json
{
  "graph_id": "graph_20240101_120000_pc",
  "nodes": ["magma_pressure", "ground_deformation", "gas_emissions"],
  "edges": [
    {
      "source": "magma_pressure",
      "target": "ground_deformation",
      "edge_type": "->",
      "strength": 0.75,
      "confidence": 0.95,
      "p_value": 0.001,
      "geological_interpretation": "Magma pressure increase causes ground deformation..."
    }
  ],
  "algorithm_used": "pc",
  "confidence_threshold": 0.05,
  "creation_timestamp": "2024-01-01T12:00:00Z",
  "validation_metrics": {
    "edge_count": 3,
    "node_count": 5,
    "average_edge_strength": 0.68,
    "domain_knowledge_consistency": 0.85
  }
}
```

### Intervention Analysis

```http
POST /causal-inference/intervene
```

Perform causal intervention analysis to understand effects of changing variables.

**Request Body:**
```json
{
  "graph_id": "graph_20240101_120000_pc",
  "intervention_variable": "magma_pressure",
  "intervention_value": 150.0,
  "target_variables": ["ground_deformation", "gas_emissions"]
}
```

**Response:**
```json
{
  "intervention_variable": "magma_pressure",
  "intervention_value": 150.0,
  "target_variables": ["ground_deformation", "gas_emissions"],
  "causal_effects": {
    "ground_deformation": 0.75,
    "gas_emissions": 0.45
  },
  "confidence_intervals": {
    "ground_deformation": [0.65, 0.85],
    "gas_emissions": [0.35, 0.55]
  },
  "statistical_significance": {
    "ground_deformation": 0.001,
    "gas_emissions": 0.01
  },
  "geological_interpretation": "Intervention analysis: Setting magma_pressure to 150.0..."
}
```

### Hypothesis Generation

```http
POST /causal-inference/generate-hypotheses
```

Generate testable geological hypotheses from discovered causal relationships.

**Request Body:**
```json
{
  "graph_id": "graph_20240101_120000_pc",
  "confidence_threshold": 0.8
}
```

**Response:**
```json
{
  "hypotheses": [
    "Hypothesis: Increases in magma pressure directly cause ground deformation. Test: Monitor ground deformation during periods of varying magma pressure. Expected: Deformation should increase proportionally with pressure (strength: 0.750)."
  ],
  "graph_summary": {
    "total_edges": 5,
    "high_confidence_edges": 3,
    "confidence_threshold": 0.8,
    "algorithm_used": "pc",
    "strongest_relationships": [
      {
        "source": "magma_pressure",
        "target": "ground_deformation",
        "strength": 0.75,
        "confidence": 0.95
      }
    ]
  },
  "generation_timestamp": "2024-01-01T12:30:00Z"
}
```

### Validation

```http
POST /causal-inference/validate
```

Validate discovered causal relationships with independent data.

**Request Body:**
```json
{
  "graph_id": "graph_20240101_120000_pc",
  "validation_data_source": "independent_monitoring"
}
```

**Response:**
```json
{
  "total_edges": 5,
  "validated_edges": 4,
  "failed_edges": 1,
  "overall_accuracy": 0.8,
  "domain_knowledge_consistency": 0.9,
  "edge_validations": {
    "magma_pressure->ground_deformation": {
      "valid": true,
      "original_strength": 0.75,
      "original_confidence": 0.95,
      "geological_interpretation": "Magma pressure causes ground deformation"
    }
  }
}
```

## Usage Examples

### Python Client Example

```python
import requests
import pandas as pd
import json

# Load geological data
data = pd.read_csv('volcanic_monitoring_data.csv')

# Discover causal relationships
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/causal-inference/discover',
        data={
            'algorithm': 'pc',
            'confidence_threshold': 0.05,
            'data_source': 'volcanic_monitoring'
        },
        files={'data_file': f}
    )

graph_result = response.json()
graph_id = graph_result['graph_id']

# Perform intervention analysis
intervention_response = requests.post(
    'http://localhost:8000/causal-inference/intervene',
    json={
        'graph_id': graph_id,
        'intervention_variable': 'magma_pressure',
        'intervention_value': 150.0
    }
)

intervention_result = intervention_response.json()
print(f"Causal effects: {intervention_result['causal_effects']}")

# Generate hypotheses
hypothesis_response = requests.post(
    'http://localhost:8000/causal-inference/generate-hypotheses',
    json={
        'graph_id': graph_id,
        'confidence_threshold': 0.8
    }
)

hypotheses = hypothesis_response.json()['hypotheses']
for i, hypothesis in enumerate(hypotheses):
    print(f"Hypothesis {i+1}: {hypothesis}")
```

### Command Line Example

```bash
# Discover causal relationships
curl -X POST "http://localhost:8000/causal-inference/discover" \
  -F "data_file=@volcanic_data.csv" \
  -F "algorithm=pc" \
  -F "confidence_threshold=0.05" \
  -F "data_source=volcanic_monitoring"

# Perform intervention analysis
curl -X POST "http://localhost:8000/causal-inference/intervene" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_id": "graph_20240101_120000_pc",
    "intervention_variable": "magma_pressure",
    "intervention_value": 150.0
  }'
```

## Data Format Requirements

### Input Data Format

The causal inference engine accepts CSV or JSON files with the following requirements:

1. **Tabular Structure**: Each row represents an observation, each column a variable
2. **Numerical Variables**: Most variables should be numerical for statistical analysis
3. **Minimum Samples**: At least 10 samples required (100+ recommended)
4. **Missing Values**: Less than 50% missing values per variable
5. **Variable Names**: Descriptive names that match geological domain knowledge

**Example CSV Format:**
```csv
timestamp,magma_pressure,temperature,ground_deformation,gas_emissions,seismic_activity
2024-01-01T00:00:00Z,100.5,1200.0,0.05,25.3,0.1
2024-01-01T01:00:00Z,102.1,1205.0,0.07,26.1,0.2
2024-01-01T02:00:00Z,98.7,1198.0,0.03,24.8,0.05
```

**Example JSON Format:**
```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "magma_pressure": 100.5,
    "temperature": 1200.0,
    "ground_deformation": 0.05,
    "gas_emissions": 25.3,
    "seismic_activity": 0.1
  }
]
```

## Algorithm Details

### PC Algorithm

The PC (Peter-Clark) algorithm is a constraint-based causal discovery method that:

1. **Skeleton Discovery**: Identifies potential causal relationships using conditional independence tests
2. **Edge Orientation**: Orients edges using v-structures and domain knowledge
3. **Constraint Propagation**: Applies additional orientation rules

**Advantages:**
- Well-established theoretical foundation
- Handles high-dimensional data
- Incorporates domain knowledge naturally

**Limitations:**
- Assumes causal sufficiency (no latent confounders)
- Sensitive to statistical test choices
- May produce undirected edges when direction cannot be determined

### FCI Algorithm

The FCI (Fast Causal Inference) algorithm extends PC to handle:

1. **Latent Confounders**: Variables not observed but affecting multiple variables
2. **Selection Bias**: Systematic differences in data collection
3. **Mixed Edge Types**: Directed, bidirected, and partially directed edges

**Advantages:**
- Handles latent confounders
- More robust to model violations
- Provides richer causal representations

**Limitations:**
- More complex interpretation
- Requires larger sample sizes
- Computationally more expensive

### GES Algorithm

The GES (Greedy Equivalence Search) algorithm uses a score-based approach:

1. **Forward Phase**: Adds edges that improve the BIC score
2. **Backward Phase**: Removes edges that improve the BIC score
3. **Score Optimization**: Balances model fit and complexity

**Advantages:**
- Directly optimizes model quality
- Handles continuous variables well
- Produces directed acyclic graphs

**Limitations:**
- Assumes causal sufficiency
- May get stuck in local optima
- Sensitive to score function choice

## Domain Knowledge Integration

### Known Causal Relationships

The engine incorporates geological domain knowledge about established causal relationships:

```python
known_relationships = {
    "magma_pressure": ["ground_deformation", "gas_emissions"],
    "temperature": ["thermal_expansion", "gas_solubility"],
    "stress_changes": ["seismic_activity", "fault_activation"]
}
```

### Impossible Relationships

Prevents discovery of physically impossible causal directions:

```python
impossible_relationships = {
    "ground_deformation": ["magma_pressure"],  # Effect cannot cause cause
    "seismic_activity": ["stress_changes"]
}
```

### Temporal Constraints

Enforces temporal ordering of geological processes:

```python
temporal_constraints = {
    "magma_intrusion": {"before": "ground_deformation"},
    "stress_buildup": {"before": "earthquake_occurrence"}
}
```

### Physical Constraints

Validates results against physical limits:

```python
physical_constraints = {
    "pressure": {"min": 0, "max": 1e9, "unit": "Pa"},
    "temperature": {"min": 273, "max": 2000, "unit": "K"}
}
```

## Performance Considerations

### Computational Complexity

- **PC Algorithm**: O(n^k) where n is number of variables, k is maximum conditioning set size
- **FCI Algorithm**: O(n^k+1) due to additional orientation rules
- **GES Algorithm**: O(n^3) for each forward/backward step

### Memory Requirements

- **Small datasets** (<1000 samples, <20 variables): ~100MB RAM
- **Medium datasets** (<10000 samples, <50 variables): ~1GB RAM
- **Large datasets** (>10000 samples, >50 variables): >2GB RAM

### Optimization Tips

1. **Variable Selection**: Remove irrelevant variables before analysis
2. **Sample Size**: Use at least 10 samples per variable
3. **Preprocessing**: Standardize variables and handle missing values
4. **Parallel Processing**: Use multiple cores for independence tests
5. **Caching**: Cache intermediate results for repeated analyses

## Error Handling

### Common Errors

1. **Insufficient Data**: Less than 10 samples or too many missing values
2. **Numerical Issues**: Singular matrices or numerical instability
3. **Domain Violations**: Results violating geological domain knowledge
4. **Convergence Failures**: Algorithms failing to converge

### Error Recovery

1. **Data Quality**: Automatic data cleaning and interpolation
2. **Algorithm Fallback**: Switch to simpler algorithms if complex ones fail
3. **Parameter Adjustment**: Automatically adjust thresholds and parameters
4. **Graceful Degradation**: Return partial results when full analysis fails

## Validation and Quality Assurance

### Statistical Validation

1. **Cross-Validation**: Split data for training and validation
2. **Bootstrap Confidence**: Use bootstrap methods for uncertainty quantification
3. **Permutation Tests**: Test significance using data permutation
4. **Multiple Testing**: Correct for multiple hypothesis testing

### Geological Validation

1. **Domain Expert Review**: Flag results for expert validation
2. **Literature Comparison**: Compare with published geological studies
3. **Physical Plausibility**: Check against known physical laws
4. **Temporal Consistency**: Verify temporal ordering of processes

### Quality Metrics

1. **Precision**: Fraction of discovered edges that are correct
2. **Recall**: Fraction of true edges that are discovered
3. **F1-Score**: Harmonic mean of precision and recall
4. **Domain Consistency**: Agreement with geological domain knowledge

## Integration with Other Components

### Helios AI Engine

The causal inference engine integrates with the main Helios AI engine to:

1. **Inform Model Architecture**: Use discovered relationships to design neural networks
2. **Guide Feature Selection**: Select relevant variables for modeling
3. **Validate Predictions**: Check if predictions respect causal relationships
4. **Improve Interpretability**: Provide causal explanations for model outputs

### Volcano Watchtower

Integration with real-time monitoring provides:

1. **Real-Time Causal Analysis**: Discover relationships in streaming data
2. **Alert Generation**: Trigger alerts based on causal relationship changes
3. **Predictive Monitoring**: Use causal models for eruption forecasting
4. **Adaptive Thresholds**: Adjust monitoring thresholds based on causal strength

### Vulcan UI

The user interface provides:

1. **Interactive Visualization**: Visualize causal graphs with geological context
2. **Hypothesis Management**: Track and validate generated hypotheses
3. **Intervention Planning**: Plan and simulate interventions
4. **Results Export**: Export results for further analysis

## Future Enhancements

### Planned Features

1. **Temporal Causal Discovery**: Discover causal relationships in time series data
2. **Nonlinear Relationships**: Handle nonlinear causal relationships
3. **Continuous Learning**: Update causal models with new data
4. **Multi-Scale Analysis**: Discover relationships across different time scales

### Research Directions

1. **Deep Causal Learning**: Combine deep learning with causal discovery
2. **Causal Reinforcement Learning**: Use causal knowledge for decision making
3. **Federated Causal Discovery**: Discover relationships across distributed datasets
4. **Quantum Causal Discovery**: Explore quantum algorithms for causal inference

## Troubleshooting

### Common Issues

1. **No Relationships Found**: Increase sample size or decrease confidence threshold
2. **Too Many Relationships**: Increase confidence threshold or add domain constraints
3. **Inconsistent Results**: Check data quality and remove outliers
4. **Slow Performance**: Reduce variable count or use simpler algorithms

### Debug Information

The engine provides detailed logging for troubleshooting:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check processing statistics
stats = engine.get_processing_statistics()
print(f"Processing time: {stats['processing_time_ms']}ms")
print(f"Variables processed: {stats['num_variables']}")
print(f"Independence tests performed: {stats['num_tests']}")
```

### Support

For technical support and questions:

1. **Documentation**: Check this documentation and API reference
2. **Examples**: Review example code and use cases
3. **Logs**: Check application logs for error details
4. **Community**: Join the Prometheus AI community forum
5. **Issues**: Report bugs on the project repository

## References

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, prediction, and search.
2. Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias.
3. Chickering, D. M. (2002). Optimal structure identification with greedy search.
4. Pearl, J. (2009). Causality: Models, reasoning, and inference.
5. Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of causal inference.