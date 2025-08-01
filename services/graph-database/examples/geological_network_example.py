#!/usr/bin/env python3
"""
Example usage of the Neo4j Graph Database Service for geological relationships
"""

import asyncio
import json

from core.database import GraphDatabaseManager
from core.graph_algorithms import GeologicalGraphAlgorithms
from core.visualization import GraphVisualizationEngine
from models.geological_models import (
    GeologicalProcess,
    GeologicalRelationship,
    GraphVisualizationConfig,
    ProcessType,
    RelationshipType,
)

async def create_sample_geological_network():
    """Create a sample geological network for demonstration"""

    # Initialize database manager
    db_manager = GraphDatabaseManager()
    await db_manager.connect()

    print("Creating sample geological processes...")

    # Create sample geological processes
    processes = [
        GeologicalProcess(
            process_id="yellowstone_magma_chamber",
            name="Yellowstone Magma Chamber",
            process_type=ProcessType.MAGMATIC,
            description="Large silicic magma chamber beneath Yellowstone caldera",
            location_id="yellowstone_001",
            depth_range=[5.0, 15.0],
            temperature_range=[750.0, 950.0],
            pressure_range=[150.0, 400.0],
            timescale="thousands_of_years",
            magnitude=8.0,
            confidence_level=0.85,
        ),
        GeologicalProcess(
            process_id="yellowstone_hydrothermal",
            name="Yellowstone Hydrothermal System",
            process_type=ProcessType.HYDROTHERMAL,
            description="Extensive hydrothermal system with geysers and hot springs",
            location_id="yellowstone_002",
            depth_range=[0.1, 3.0],
            temperature_range=[50.0, 300.0],
            pressure_range=[1.0, 30.0],
            timescale="years",
            magnitude=5.5,
            confidence_level=0.95,
        ),
        GeologicalProcess(
            process_id="yellowstone_seismic_swarm",
            name="Yellowstone Seismic Swarms",
            process_type=ProcessType.SEISMIC,
            description="Earthquake swarms associated with magma movement",
            location_id="yellowstone_003",
            depth_range=[1.0, 10.0],
            temperature_range=[200.0, 600.0],
            pressure_range=[10.0, 200.0],
            timescale="days_to_months",
            magnitude=4.2,
            confidence_level=0.90,
        ),
        GeologicalProcess(
            process_id="yellowstone_ground_deformation",
            name="Yellowstone Ground Deformation",
            process_type=ProcessType.DEFORMATIONAL,
            description="Caldera uplift and subsidence cycles",
            location_id="yellowstone_004",
            depth_range=[0.0, 20.0],
            temperature_range=[10.0, 800.0],
            pressure_range=[1.0, 500.0],
            timescale="years_to_decades",
            magnitude=3.8,
            confidence_level=0.88,
        ),
        GeologicalProcess(
            process_id="snake_river_plain_volcanism",
            name="Snake River Plain Volcanism",
            process_type=ProcessType.VOLCANIC,
            description="Basaltic volcanism along the Snake River Plain",
            location_id="snake_river_001",
            depth_range=[20.0, 50.0],
            temperature_range=[1000.0, 1200.0],
            pressure_range=[500.0, 1500.0],
            timescale="millions_of_years",
            magnitude=6.5,
            confidence_level=0.80,
        ),
    ]

    # Create processes in database
    process_ids = []
    for process in processes:
        process_id = await db_manager.create_geological_process(process)
        process_ids.append(process_id)
        print(f"Created process: {process.name} (ID: {process_id})")

    print("\nCreating geological relationships...")

    # Create relationships between processes
    relationships = [
        GeologicalRelationship(
            relationship_id="rel_magma_hydrothermal",
            source_process_id="yellowstone_magma_chamber",
            target_process_id="yellowstone_hydrothermal",
            relationship_type=RelationshipType.CAUSAL,
            strength=0.85,
            confidence=0.90,
            mechanism="heat_transfer_convection",
            time_delay=0.0,  # Continuous process
            spatial_scale=15.0,
        ),
        GeologicalRelationship(
            relationship_id="rel_magma_seismic",
            source_process_id="yellowstone_magma_chamber",
            target_process_id="yellowstone_seismic_swarm",
            relationship_type=RelationshipType.TRIGGERS,
            strength=0.75,
            confidence=0.85,
            mechanism="magma_intrusion_stress",
            time_delay=30.0,  # Days
            spatial_scale=25.0,
        ),
        GeologicalRelationship(
            relationship_id="rel_magma_deformation",
            source_process_id="yellowstone_magma_chamber",
            target_process_id="yellowstone_ground_deformation",
            relationship_type=RelationshipType.CAUSAL,
            strength=0.90,
            confidence=0.95,
            mechanism="magma_volume_pressure_changes",
            time_delay=365.0,  # Years
            spatial_scale=50.0,
        ),
        GeologicalRelationship(
            relationship_id="rel_seismic_hydrothermal",
            source_process_id="yellowstone_seismic_swarm",
            target_process_id="yellowstone_hydrothermal",
            relationship_type=RelationshipType.INFLUENCES,
            strength=0.60,
            confidence=0.75,
            mechanism="fracture_permeability_changes",
            time_delay=7.0,  # Days
            spatial_scale=10.0,
        ),
        GeologicalRelationship(
            relationship_id="rel_hotspot_track",
            source_process_id="snake_river_plain_volcanism",
            target_process_id="yellowstone_magma_chamber",
            relationship_type=RelationshipType.PRECEDES,
            strength=0.95,
            confidence=0.90,
            mechanism="hotspot_track_progression",
            time_delay=2000000.0,  # 2 million years
            spatial_scale=500.0,
        ),
    ]

    # Create relationships in database
    for relationship in relationships:
        relationship_id = await db_manager.create_relationship(relationship)
        print(f"Created relationship: {relationship.mechanism} (ID: {relationship_id})")

    await db_manager.close()
    return process_ids

async def analyze_geological_network():
    """Analyze the geological network using graph algorithms"""

    db_manager = GraphDatabaseManager()
    await db_manager.connect()

    algorithms = GeologicalGraphAlgorithms(db_manager)

    print("\n" + "=" * 60)
    print("GEOLOGICAL NETWORK ANALYSIS")
    print("=" * 60)

    # Get all process IDs
    all_processes = await db_manager.execute_query(
        "MATCH (p:GeologicalProcess) RETURN p.process_id as process_id"
    )
    process_ids = [record["process_id"] for record in all_processes]

    print(f"\nAnalyzing network with {len(process_ids)} processes...")

    # 1. Find causal chains from magma chamber
    print("\n1. CAUSAL CHAINS FROM YELLOWSTONE MAGMA CHAMBER:")
    print("-" * 50)

    causal_chains = await db_manager.find_causal_chains(
        "yellowstone_magma_chamber", max_depth=3
    )

    for i, chain in enumerate(causal_chains, 1):
        print(
            f"\nChain {i} (Strength: {chain.total_strength:.2f}, Confidence: {chain.confidence_score:.2f}):"
        )
        for j, process in enumerate(chain.process_chain):
            print(f"  {j+1}. {process['name']} ({process['process_type']})")
            if j < len(chain.relationships):
                rel = chain.relationships[j]
                print(f"     → {rel['mechanism']} (strength: {rel['strength']:.2f})")

    # 2. Network analysis
    print("\n2. NETWORK ANALYSIS:")
    print("-" * 30)

    network_result = await db_manager.analyze_process_network(process_ids)

    print("Network Statistics:")
    print(f"  - Nodes: {network_result.node_count}")
    print(f"  - Edges: {network_result.edge_count}")
    print(f"  - Density: {network_result.density:.3f}")

    # 3. Find critical processes
    print("\n3. CRITICAL PROCESSES:")
    print("-" * 25)

    critical_processes = await algorithms.find_critical_processes(
        process_ids, criteria="centrality"
    )

    for i, process in enumerate(critical_processes[:3], 1):
        print(
            f"{i}. {process['process_id']} (Criticality: {process['criticality_score']:.3f})"
        )
        centrality = process.get("centrality_measures", {})
        if centrality:
            print(f"   - PageRank: {centrality.get('pagerank', 0):.3f}")
            print(f"   - Betweenness: {centrality.get('betweenness', 0):.3f}")
            print(f"   - Closeness: {centrality.get('closeness', 0):.3f}")

    # 4. Community detection
    print("\n4. COMMUNITY DETECTION:")
    print("-" * 25)

    try:
        communities = await algorithms.detect_communities(process_ids)
        print(f"Found {len(communities)} communities:")

        for community_id, members in communities.items():
            print(f"\n{community_id}:")
            for member in members:
                process = await db_manager.get_process_by_id(member)
                if process:
                    print(f"  - {process['name']} ({process['process_type']})")
    except Exception as e:
        print(f"Community detection failed: {e}")

    # 5. Cascade analysis
    print("\n5. CASCADE EFFECT ANALYSIS:")
    print("-" * 30)

    cascade_analysis = await algorithms.analyze_cascade_effects(
        "yellowstone_magma_chamber", max_depth=3
    )

    print(f"Trigger: {cascade_analysis['trigger_process_id']}")
    print(f"Maximum cascade depth: {cascade_analysis['max_cascade_depth']}")
    print(f"Cascade breadth: {cascade_analysis['cascade_breadth']}")
    print(f"Total cascade strength: {cascade_analysis['total_cascade_strength']:.3f}")

    print("\nAffected process types:")
    for process_type, count in cascade_analysis["affected_process_types"].items():
        print(f"  - {process_type}: {count} processes")

    await db_manager.close()

async def create_visualization():
    """Create visualization of the geological network"""

    db_manager = GraphDatabaseManager()
    await db_manager.connect()

    visualization_engine = GraphVisualizationEngine(db_manager)

    print("\n" + "=" * 60)
    print("CREATING NETWORK VISUALIZATION")
    print("=" * 60)

    # Get all processes for visualization
    all_processes = await db_manager.execute_query(
        """
        MATCH (p:GeologicalProcess)
        RETURN p.process_id as id,
               p.name as label,
               p.process_type as type,
               p.magnitude as size,
               p.description as description
    """
    )

    # Get all relationships
    all_relationships = await db_manager.execute_query(
        """
        MATCH (source:GeologicalProcess)-[r:INFLUENCES]->(target:GeologicalProcess)
        RETURN source.process_id as source,
               target.process_id as target,
               r.relationship_type as type,
               r.strength as weight,
               r.mechanism as label,
               r.confidence as confidence
    """
    )

    # Prepare nodes and edges for visualization
    nodes = []
    for process in all_processes:
        nodes.append(
            {
                "id": process["id"],
                "label": process["label"],
                "type": process["type"],
                "size": (process["size"] or 5) * 2,  # Scale size for visibility
                "metadata": {"description": process["description"]},
            }
        )

    edges = []
    for rel in all_relationships:
        edges.append(
            {
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["type"],
                "weight": rel["weight"],
                "width": rel["weight"] * 3,  # Scale width for visibility
                "label": rel["label"],
                "metadata": {"confidence": rel["confidence"]},
            }
        )

    print(f"Visualizing {len(nodes)} nodes and {len(edges)} edges...")

    # Create visualization config
    config = GraphVisualizationConfig(
        visualization_id="yellowstone_network_viz",
        layout_algorithm="force_directed",
        node_size_property="size",
        node_color_property="type",
        edge_width_property="weight",
        show_node_labels=True,
        show_edge_labels=False,
        min_edge_strength=0.5,
        export_format="cytoscape",
        include_metadata=True,
    )

    # Generate visualization
    visualization_data = await visualization_engine.create_interactive_visualization(
        nodes, edges, config
    )

    # Save visualization data
    with open("yellowstone_network_visualization.json", "w") as f:
        json.dump(visualization_data, f, indent=2, default=str)

    print("Visualization saved to: yellowstone_network_visualization.json")

    # Export in different formats
    print("\nExporting in different formats...")

    # GraphML export
    graphml_data = await visualization_engine.export_to_graphml(nodes, edges)
    with open("yellowstone_network.graphml", "w") as f:
        f.write(graphml_data)
    print("GraphML export saved to: yellowstone_network.graphml")

    # GEXF export
    gexf_data = await visualization_engine.export_to_gexf(nodes, edges)
    with open("yellowstone_network.gex", "w") as f:
        f.write(gexf_data)
    print("GEXF export saved to: yellowstone_network.gex")

    await db_manager.close()

async def demonstrate_search_capabilities():
    """Demonstrate search and query capabilities"""

    db_manager = GraphDatabaseManager()
    await db_manager.connect()

    print("\n" + "=" * 60)
    print("SEARCH AND QUERY CAPABILITIES")
    print("=" * 60)

    # 1. Full-text search
    print("\n1. FULL-TEXT SEARCH:")
    print("-" * 20)

    search_results = await db_manager.search_processes("magma", limit=5)
    print("Search results for 'magma':")
    for result in search_results:
        process = result["process"]
        score = result["relevance_score"]
        print(f"  - {process['name']} (relevance: {score:.2f})")
        print(f"    Type: {process['process_type']}")
        print(f"    Description: {process.get('description', 'N/A')[:100]}...")

    # 2. Process type filtering
    print("\n2. PROCESS TYPE FILTERING:")
    print("-" * 25)

    volcanic_processes = await db_manager.search_processes(
        "", process_type="volcanic", limit=10
    )
    print(f"Volcanic processes ({len(volcanic_processes)} found):")
    for result in volcanic_processes:
        process = result["process"]
        print(f"  - {process['name']}")

    # 3. Shortest path queries
    print("\n3. SHORTEST PATH QUERIES:")
    print("-" * 25)

    algorithms = GeologicalGraphAlgorithms(db_manager)

    try:
        paths = await algorithms.find_shortest_paths(
            "yellowstone_magma_chamber",
            "yellowstone_hydrothermal",
            relationship_types=["causal", "influences"],
        )

        print("Paths from magma chamber to hydrothermal system:")
        for path in paths:
            print(
                f"  Path length: {path['length']}, Strength: {path['total_strength']:.2f}"
            )
            for i, node in enumerate(path["nodes"]):
                print(f"    {i+1}. {node['name']}")
                if i < len(path["relationships"]):
                    rel = path["relationships"][i]
                    print(
                        f"       → {rel['relationship_type']} (strength: {rel['strength']:.2f})"
                    )

    except Exception as e:
        print(f"Path finding failed: {e}")

    await db_manager.close()

async def main():
    """Main demonstration function"""

    print("=" * 60)
    print("NEO4J GRAPH DATABASE SERVICE DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the geological relationship")
    print("graph database capabilities using Yellowstone as a case study.")
    print("=" * 60)

    try:
        # Create sample network
        process_ids = await create_sample_geological_network()
        print(f"\nCreated geological network with {len(process_ids)} processes")

        # Analyze the network
        await analyze_geological_network()

        # Create visualizations
        await create_visualization()

        # Demonstrate search capabilities
        await demonstrate_search_capabilities()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Files created:")
        print("  - yellowstone_network_visualization.json")
        print("  - yellowstone_network.graphml")
        print("  - yellowstone_network.gex")
        print("\nYou can now:")
        print("  1. Import GraphML/GEXF files into Gephi or Cytoscape")
        print("  2. Use the JSON visualization data in web applications")
        print("  3. Query the Neo4j database directly for more analysis")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
