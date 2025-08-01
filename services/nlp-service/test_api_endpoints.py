#!/usr/bin/env python3
"""
Test script for NLP API endpoints
"""

import os
import sys

# Add the service directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

def test_api_endpoints():
    """Test NLP API endpoints"""

    print("Testing NLP API Endpoints")
    print("=" * 40)

    client = TestClient(app)

    # Test health endpoint
    print("1. Testing health endpoint...")
    response = client.get("/health")
    assert response.status_code == 200
    print(f"✓ Health check: {response.json()}")

    # Test readiness endpoint
    print("\n2. Testing readiness endpoint...")
    response = client.get("/ready")
    result = response.json()
    print(f"✓ Readiness check: {result}")

    # Note: In test mode, components may not be fully initialized
    if result.get("status") == "not_ready":
        print("   (Components not initialized in test mode - this is expected)")

    # Test intent classification
    print("\n3. Testing intent classification...")
    test_query = {
        "text": "What causes volcanic eruptions at Mount St. Helens?",
        "context": None,
        "user_id": "test_user",
        "session_id": "test_session",
    }

    response = client.post("/api/v1/nlp/classify-intent", json=test_query)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Intent: {result['intent']} (confidence: {result['confidence']:.2f})")

    # Test entity extraction
    print("\n4. Testing entity extraction...")
    response = client.post("/api/v1/nlp/extract-entities", json=test_query)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Entities found: {result['total_entities']}")
    for entity in result["entities"][:3]:
        print(f"   - {entity['text']} ({entity['entity_type']})")

    # Test query translation
    print("\n5. Testing query translation...")
    response = client.post("/api/v1/nlp/translate-query", json=test_query)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Parameters extracted: {len(result['parameters'])}")
    print(f"   Query type: {result['query_type']}")
    print(f"   Target service: {result['target_service']}")

    # Test response generation
    print("\n6. Testing response generation...")
    geological_request = {
        "query": "What causes volcanic eruptions at Mount St. Helens?",
        "context": {"intent": "explanation_request"},
        "geological_data": None,
        "user_preferences": None,
        "correlation_id": "test_123",
    }

    response = client.post("/api/v1/nlp/generate-response", json=geological_request)
    assert response.status_code == 200
    result = response.json()
    print(f"✓ Response generated ({result['response_type']})")
    print(f"   Response: {result['response'][:100]}...")

    # Test complete geological query processing
    print("\n7. Testing complete geological query processing...")
    response = client.post(
        "/api/v1/nlp/process-geological-query", json=geological_request
    )
    assert response.status_code == 200
    result = response.json()
    print("✓ Complete processing successful")
    print(f"   Intent: {result['intent']['intent']}")
    print(f"   Entities: {result['entities']['total_entities']}")
    print(f"   Parameters: {len(result['translation']['parameters'])}")
    print(f"   Response: {result['response']['response'][:100]}...")

    # Test geological terms endpoint
    print("\n8. Testing geological terms endpoint...")
    response = client.get("/api/v1/nlp/geological-terms")
    assert response.status_code == 200
    result = response.json()
    print("✓ Geological terms retrieved")
    print(f"   Categories: {list(result['geological_terms'].keys())}")

    # Test supported intents endpoint
    print("\n9. Testing supported intents endpoint...")
    response = client.get("/api/v1/nlp/supported-intents")
    assert response.status_code == 200
    result = response.json()
    print("✓ Supported intents retrieved")
    print(f"   Intents: {len(result['supported_intents'])}")

    print("\n" + "=" * 40)
    print("All API endpoint tests passed! ✓")

if __name__ == "__main__":
    test_api_endpoints()
