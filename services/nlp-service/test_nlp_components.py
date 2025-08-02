#!/usr/bin/env python3
"""
Test script for NLP components
"""

import asyncio
import os
import sys

# Add the service directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
from core.entity_extractor import GeologicalEntityExtractor
from core.intent_classifier import IntentClassifier
from core.query_translator import QueryTranslator
from core.response_generator import ConversationalResponseGenerator


async def test_nlp_components():
    """Test all NLP components with sample geological queries"""

    print("Testing NLP Components for Geological Queries")
    print("=" * 50)

    # Initialize components
    print("Initializing components...")
    intent_classifier = IntentClassifier()
    entity_extractor = GeologicalEntityExtractor()
    query_translator = QueryTranslator()
    response_generator = ConversationalResponseGenerator()
    print("✓ All components initialized successfully\n")

    # Test queries
    test_queries = [
        "What causes volcanic eruptions at Mount St. Helens?",
        "Simulate the evolution of Yellowstone caldera over the next 1000 years",
        "Analyze seismic data from the recent magnitude 6.2 earthquake at 15 km depth",
        "What is the current volcanic activity status at Kilauea?",
        "Compare shield volcanoes to stratovolcanoes",
        "Predict geothermal resources in the Cascadia region",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        print("-" * 60)

        try:
            # Step 1: Intent Classification
            intent_result = await intent_classifier.classify_intent(query)
            print(
                f"Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})"
            )

            # Step 2: Entity Extraction
            entity_result = await entity_extractor.extract_entities(query)
            print(f"Entities found: {len(entity_result.entities)}")
            for entity in entity_result.entities[:3]:  # Show first 3 entities
                print(
                    f"  - {entity.text} ({entity.entity_type.value}, confidence: {entity.confidence:.2f})"
                )

            # Step 3: Query Translation
            translation_result = await query_translator.translate_query(
                query, intent_result.intent, entity_result.entities
            )
            print(f"Parameters extracted: {len(translation_result.parameters)}")
            print(f"Query type: {translation_result.query_type}")
            print(f"Target service: {translation_result.target_service}")

            # Step 4: Response Generation
            context = {
                "intent": intent_result.intent,
                "entities": entity_result.entities,
            }
            response_result = await response_generator.generate_response(query, context)
            print(f"Response: {response_result.response[:200]}...")
            if response_result.follow_up_suggestions:
                print(f"Follow-up: {response_result.follow_up_suggestions[0]}")

            print(
                f"✓ Processing completed in {intent_result.processing_time_ms + entity_result.processing_time_ms + translation_result.processing_time_ms + response_result.processing_time_ms:.1f}ms\n"
            )

        except Exception as e:
            print(f"✗ Error processing query: {str(e)}\n")

    print("NLP Component Testing Complete!")


if __name__ == "__main__":
    asyncio.run(test_nlp_components())
