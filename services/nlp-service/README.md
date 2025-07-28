# NLP Service - Natural Language Processing for Geological Queries

The NLP Service provides natural language processing capabilities for the Prometheus AI platform, enabling users to interact with geological models and data using conversational interfaces.

## Features

### Core Components

1. **Intent Classification** - Classifies geological query intents
   - Simulation requests
   - Data analysis queries  
   - Monitoring queries
   - Hazard assessments
   - Resource predictions
   - Explanation requests
   - Comparison queries
   - Historical queries

2. **Entity Extraction** - Extracts geological entities from text
   - Locations (volcanoes, faults, regions)
   - Processes (eruptions, earthquakes, subduction)
   - Materials (magma, basalt, minerals)
   - Measurements (magnitude, depth, temperature)
   - Time periods (geological eras, recent events)
   - Instruments (seismometers, GPS, InSAR)

3. **Query Translation** - Converts natural language to system parameters
   - Extracts numerical values and units
   - Maps entities to system parameters
   - Determines target services for routing
   - Provides confidence scores

4. **Response Generation** - Generates conversational responses
   - Geological explanations
   - Process descriptions
   - Monitoring status updates
   - Simulation guidance
   - Follow-up suggestions

## API Endpoints

### Core Processing
- `POST /api/v1/nlp/classify-intent` - Classify query intent
- `POST /api/v1/nlp/extract-entities` - Extract geological entities
- `POST /api/v1/nlp/translate-query` - Translate to system parameters
- `POST /api/v1/nlp/generate-response` - Generate conversational response
- `POST /api/v1/nlp/process-geological-query` - Complete end-to-end processing

### Information
- `GET /api/v1/nlp/geological-terms` - Get supported geological terms
- `GET /api/v1/nlp/supported-intents` - Get supported query intents

### Health
- `GET /health` - Service health check
- `GET /ready` - Service readiness check

## Usage Examples

### Intent Classification
```python
import httpx

query = {
    "text": "What causes volcanic eruptions at Mount St. Helens?",
    "user_id": "user123",
    "session_id": "session456"
}

response = httpx.post("http://localhost:8005/api/v1/nlp/classify-intent", json=query)
result = response.json()
print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
```

### Complete Processing
```python
request = {
    "query": "Simulate magma chamber evolution at Yellowstone",
    "context": {"user_expertise": "expert"},
    "correlation_id": "req_789"
}

response = httpx.post("http://localhost:8005/api/v1/nlp/process-geological-query", json=request)
result = response.json()

print(f"Intent: {result['intent']['intent']}")
print(f"Entities: {len(result['entities']['entities'])}")
print(f"Target Service: {result['translation']['target_service']}")
print(f"Response: {result['response']['response']}")
```

## Development

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py

# Run tests
python test_nlp_components.py
python test_integration.py
```

### Docker
```bash
# Build image
docker build -t prometheus-nlp-service .

# Run container
docker run -p 8005:8005 prometheus-nlp-service
```

### Testing
The service includes comprehensive tests:
- `test_nlp_components.py` - Individual component testing
- `test_integration.py` - End-to-end integration testing
- `test_api_endpoints.py` - API endpoint testing

## Configuration

The service can be configured through environment variables:
- `LOG_LEVEL` - Logging level (default: INFO)
- `PORT` - Service port (default: 8005)
- `HOST` - Service host (default: 0.0.0.0)

## Dependencies

### Core Dependencies
- FastAPI - Web framework
- Pydantic - Data validation
- structlog - Structured logging
- scikit-learn - Machine learning
- numpy - Numerical computing

### Optional Dependencies
- spaCy - Advanced NLP (for enhanced entity extraction)
- transformers - Transformer models (for advanced intent classification)

## Architecture

The NLP service follows a modular architecture:

```
nlp-service/
├── api/           # API route handlers
├── core/          # Core NLP components
├── models/        # Pydantic data models
├── main.py        # FastAPI application
└── tests/         # Test files
```

Each core component is independently testable and can be enhanced with more sophisticated models as needed.

## Performance

- Intent classification: ~10ms per query
- Entity extraction: ~5ms per query  
- Query translation: ~2ms per query
- Response generation: ~3ms per query
- Complete processing: ~20ms per query

Performance can be improved by:
- Adding GPU acceleration for ML models
- Implementing caching for common queries
- Using more efficient NLP models
- Batch processing for multiple queries

## Future Enhancements

- Integration with transformer-based models
- Multi-language support
- Context-aware conversation management
- Integration with geological knowledge graphs
- Real-time learning from user interactions