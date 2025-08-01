#!/usr/bin/env python3
"""
Example usage of the Scientific Knowledge Graph Service
"""

import asyncio
from datetime import datetime, timedelta

import aiohttp
import numpy as np
import pandas as pd
import torch.nn.functional as F


async def example_knowledge_extraction():
    """Example of extracting knowledge from a scientific paper"""

    # Example paper data
    paper_data = {
        "title": "Volcanic Eruption Prediction Using Seismic Precursors",
        "authors": ["Dr. Jane Smith", "Prof. John Doe"],
        "abstract": "This study investigates the relationship between seismic activity and volcanic eruptions. We analyzed seismic data from Mount Vesuvius and found that increased earthquake frequency precedes eruptions by 2-3 weeks. The magnitude of seismic events correlates with eruption intensity.",
        "full_text": "Introduction: Volcanic eruptions pose significant hazards to populations living near active volcanoes. Seismic monitoring has emerged as a crucial tool for eruption prediction. Methods: We analyzed 10 years of seismic data from Mount Vesuvius, focusing on earthquake frequency, magnitude, and depth. Results: Our analysis reveals that earthquake frequency increases significantly 2-3 weeks before major eruptions. The b-value decreases from 1.2 to 0.8 during pre-eruptive periods. Conclusion: Seismic precursors provide reliable indicators for volcanic eruption prediction.",
        "doi": "10.1000/example.2023.001",
        "publication_date": "2023-06-15T00:00:00Z",
        "journal": "Journal of Volcanology",
        "keywords": [
            "volcanic eruption",
            "seismic precursors",
            "earthquake prediction",
            "Mount Vesuvius",
        ],
        "citations": ["10.1000/ref.2022.001", "10.1000/ref.2021.005"],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8007/api/v1/extract/paper", json=paper_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                print("Knowledge Extraction Results:")
                print(f"- Entities: {result['entities_count']}")
                print(f"- Relationships: {result['relationships_count']}")
                print(f"- Concepts: {result['concepts_count']}")
                print(f"- Temporal Facts: {result['temporal_facts_count']}")
                print(
                    f"- Overall Confidence: {result['confidence_scores']['overall']:.3f}"
                )
                print(f"- Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"Error: {response.status} - {await response.text()}")


async def example_temporal_causal_discovery():
    """Example of temporal causal discovery"""

    # Generate synthetic time series data
    np.random.seed(42)
    n_points = 1000

    # Create timestamps
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_points, 0, -1)]

    # Generate synthetic seismic and deformation data with causal relationship
    seismic_data = np.random.normal(0, 1, n_points)
    # Deformation follows seismic with a lag of 5 time steps
    deformation_data = np.zeros(n_points)
    for i in range(5, n_points):
        deformation_data[i] = 0.7 * seismic_data[i - 5] + np.random.normal(0, 0.3)

    # Prepare time series data
    time_series_data = {
        "seismic_activity": [
            {"timestamp": ts.isoformat(), "value": val}
            for ts, val in zip(timestamps, seismic_data)
        ],
        "ground_deformation": [
            {"timestamp": ts.isoformat(), "value": val}
            for ts, val in zip(timestamps, deformation_data)
        ],
    }

    request_data = {
        "time_series_data": time_series_data,
        "max_lag": 20,
        "confidence_threshold": 0.6,
        "methods": ["granger_causality", "transfer_entropy"],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8007/api/v1/causal/discover", json=request_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                print("\nTemporal Causal Discovery Results:")
                print(f"- Discovery ID: {result['discovery_id']}")
                print(f"- Relationships Found: {len(result['relationships'])}")

                for rel in result["relationships"]:
                    print(f"  * {rel['cause_variable']} → {rel['effect_variable']}")
                    print(f"    - Time Lag: {rel['time_lag']} steps")
                    print(f"    - Strength: {rel['causal_strength']:.3f}")
                    print(f"    - Confidence: {rel['confidence']:.3f}")
                    print(f"    - Method: {rel['method']}")

                print(
                    f"- Network Confidence: {result['causal_network']['network_confidence']:.3f}"
                )
                print(f"- Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"Error: {response.status} - {await response.text()}")


async def example_semantic_reasoning():
    """Example of semantic reasoning"""

    request_data = {
        "concepts": ["volcano", "earthquake", "magma", "eruption", "seismic_activity"],
        "reasoning_depth": 2,
        "confidence_threshold": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8007/api/v1/semantic/infer", json=request_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                print("\nSemantic Reasoning Results:")
                print(f"- Inference ID: {result['inference_id']}")
                print(f"- Inferred Relationships: {result['total_inferences']}")

                for rel in result["inferred_relationships"]:
                    print(
                        f"  * {rel['source_concept']} --[{rel['relationship_type']}]--> {rel['target_concept']}"
                    )
                    print(f"    - Confidence: {rel['confidence']:.3f}")
                    print(f"    - Reasoning: {', '.join(rel['reasoning_chain'])}")

                print(f"- Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"Error: {response.status} - {await response.text()}")


async def example_federated_dataset_registration():
    """Example of registering a dataset for federated discovery"""

    dataset_data = {
        "dataset_id": "usgs_yellowstone_2023",
        "institution": "United States Geological Survey",
        "location": "Yellowstone National Park, USA",
        "temporal_range_start": "2020-01-01T00:00:00Z",
        "temporal_range_end": "2023-12-31T23:59:59Z",
        "variables": [
            "seismic_magnitude",
            "ground_deformation",
            "gas_emission",
            "thermal_anomaly",
        ],
        "sample_count": 50000,
        "data_quality_score": 0.95,
        "privacy_level": "public",
        "contact_info": "yellowstone-data@usgs.gov",
        "data_endpoint": "https://api.usgs.gov/yellowstone/data",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8007/api/v1/federated/register", json=dataset_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                print("\nFederated Dataset Registration:")
                print(f"- Status: {result['status']}")
                print(f"- Dataset ID: {result['dataset_id']}")
                print(f"- Registration Time: {result['registration_time']}")
                print(f"- Message: {result['message']}")
            else:
                print(f"Error: {response.status} - {await response.text()}")


async def example_knowledge_search():
    """Example of searching the knowledge graph"""

    search_queries = [
        {"query": "volcanic eruption prediction", "search_type": "concepts"},
        {"query": "seismic precursors", "search_type": "concepts"},
        {"query": "magma chamber", "search_type": "concepts"},
    ]

    async with aiohttp.ClientSession() as session:
        for search_data in search_queries:
            async with session.post(
                "http://localhost:8007/api/v1/search", json=search_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"\nSearch Results for '{search_data['query']}':")
                    print(f"- Total Results: {result['total_results']}")

                    for i, res in enumerate(result["results"][:3], 1):  # Show top 3
                        print(f"  {i}. {res['name']}")
                        if res.get("definition"):
                            print(f"     Definition: {res['definition'][:100]}...")
                        print(f"     Confidence: {res['confidence']:.3f}")
                        print(f"     Relevance: {res['relevance_score']:.3f}")
                else:
                    print(f"Search Error: {response.status} - {await response.text()}")


async def example_get_statistics():
    """Example of getting knowledge graph statistics"""

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8007/api/v1/statistics") as response:
            if response.status == 200:
                stats = await response.json()
                print("\nKnowledge Graph Statistics:")
                print(f"- Total Concepts: {stats['concepts']}")
                print(f"- Total Relationships: {stats['relationships']}")
                print(f"- Inferred Relationships: {stats['inferred_relationships']}")
                print(f"- Average Confidence: {stats['average_confidence']:.3f}")
                print(f"- Total Updates: {stats['total_updates']}")
                print(f"- Total Discoveries: {stats['total_discoveries']}")
                print("- External Integrations:")
                for integration, status in stats["external_integrations"].items():
                    print(f"  * {integration}: {'✓' if status else '✗'}")
            else:
                print(f"Error: {response.status} - {await response.text()}")


async def main():
    """Run all examples"""
    print("Scientific Knowledge Graph Service Examples")
    print("=" * 50)

    try:
        # Wait a moment for service to be ready
        await asyncio.sleep(2)

        # Run examples
        await example_knowledge_extraction()
        await example_temporal_causal_discovery()
        await example_semantic_reasoning()
        await example_federated_dataset_registration()
        await example_knowledge_search()
        await example_get_statistics()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
