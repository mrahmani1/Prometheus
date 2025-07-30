"""
Graph visualization interfaces for geological process networks
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math

from .database import GraphDatabaseManager
from models.geological_models import GraphVisualizationConfig, NetworkAnalysisResult

logger = logging.getLogger(__name__)


class GraphVisualizationEngine:
    """Engine for generating graph visualizations of geological processes"""
    
    def __init__(self, db_manager: GraphDatabaseManager):
        self.db_manager = db_manager
    
    async def generate_force_directed_layout(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        iterations: int = 100,
        k: float = 1.0,
        repulsion_strength: float = 1000.0
    ) -> Dict[str, Dict[str, float]]:
        """Generate force-directed layout positions for nodes"""
        
        # Initialize random positions
        positions = {}
        for node in nodes:
            positions[node["id"]] = {
                "x": (hash(node["id"]) % 1000) / 10.0,
                "y": (hash(node["id"] + "y") % 1000) / 10.0
            }
        
        # Force-directed algorithm (simplified Fruchterman-Reingold)
        for iteration in range(iterations):
            # Calculate repulsive forces
            forces = {node_id: {"x": 0.0, "y": 0.0} for node_id in positions.keys()}
            
            # Repulsion between all nodes
            node_ids = list(positions.keys())
            for i, node1 in enumerate(node_ids):
                for j, node2 in enumerate(node_ids[i+1:], i+1):
                    pos1 = positions[node1]
                    pos2 = positions[node2]
                    
                    dx = pos1["x"] - pos2["x"]
                    dy = pos1["y"] - pos2["y"]
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01  # Avoid division by zero
                    
                    # Repulsive force
                    force_magnitude = repulsion_strength / (distance * distance)
                    force_x = (dx / distance) * force_magnitude
                    force_y = (dy / distance) * force_magnitude
                    
                    forces[node1]["x"] += force_x
                    forces[node1]["y"] += force_y
                    forces[node2]["x"] -= force_x
                    forces[node2]["y"] -= force_y
            
            # Attractive forces from edges
            for edge in edges:
                source = edge["source"]
                target = edge["target"]
                
                if source in positions and target in positions:
                    pos1 = positions[source]
                    pos2 = positions[target]
                    
                    dx = pos2["x"] - pos1["x"]
                    dy = pos2["y"] - pos1["y"]
                    distance = math.sqrt(dx*dx + dy*dy) + 0.01
                    
                    # Attractive force (spring)
                    force_magnitude = k * distance * edge.get("weight", 1.0)
                    force_x = (dx / distance) * force_magnitude
                    force_y = (dy / distance) * force_magnitude
                    
                    forces[source]["x"] += force_x
                    forces[source]["y"] += force_y
                    forces[target]["x"] -= force_x
                    forces[target]["y"] -= force_y
            
            # Apply forces with cooling
            cooling = 1.0 - (iteration / iterations)
            max_displacement = 10.0 * cooling
            
            for node_id in positions.keys():
                force = forces[node_id]
                displacement = math.sqrt(force["x"]**2 + force["y"]**2)
                
                if displacement > 0:
                    # Limit displacement
                    if displacement > max_displacement:
                        force["x"] = (force["x"] / displacement) * max_displacement
                        force["y"] = (force["y"] / displacement) * max_displacement
                    
                    positions[node_id]["x"] += force["x"]
                    positions[node_id]["y"] += force["y"]
        
        return positions
    
    async def generate_hierarchical_layout(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        root_node_id: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Generate hierarchical layout based on process dependencies"""
        
        # Build adjacency list
        adjacency = {node["id"]: [] for node in nodes}
        in_degree = {node["id"]: 0 for node in nodes}
        
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if source in adjacency and target in adjacency:
                adjacency[source].append(target)
                in_degree[target] += 1
        
        # Find root nodes (nodes with no incoming edges) or use specified root
        if root_node_id and root_node_id in in_degree:
            roots = [root_node_id]
        else:
            roots = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        if not roots:
            # If no clear roots, pick nodes with minimum in-degree
            min_degree = min(in_degree.values())
            roots = [node_id for node_id, degree in in_degree.items() if degree == min_degree]
        
        # Assign levels using BFS
        levels = {}
        queue = [(root, 0) for root in roots]
        visited = set()
        
        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            levels[node_id] = level
            
            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))
        
        # Assign positions within levels
        level_groups = {}
        for node_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)
        
        positions = {}
        y_spacing = 100.0
        x_spacing = 150.0
        
        for level, node_ids in level_groups.items():
            y = level * y_spacing
            total_width = (len(node_ids) - 1) * x_spacing
            start_x = -total_width / 2
            
            for i, node_id in enumerate(node_ids):
                positions[node_id] = {
                    "x": start_x + i * x_spacing,
                    "y": y
                }
        
        # Handle nodes not reached by BFS
        for node in nodes:
            if node["id"] not in positions:
                positions[node["id"]] = {
                    "x": (hash(node["id"]) % 1000) / 10.0,
                    "y": (hash(node["id"] + "y") % 1000) / 10.0
                }
        
        return positions
    
    async def generate_circular_layout(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        radius: float = 200.0
    ) -> Dict[str, Dict[str, float]]:
        """Generate circular layout for nodes"""
        
        positions = {}
        node_count = len(nodes)
        
        if node_count == 0:
            return positions
        
        angle_step = 2 * math.pi / node_count
        
        for i, node in enumerate(nodes):
            angle = i * angle_step
            positions[node["id"]] = {
                "x": radius * math.cos(angle),
                "y": radius * math.sin(angle)
            }
        
        return positions
    
    async def export_to_cytoscape(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        positions: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Export graph data in Cytoscape.js format"""
        
        cytoscape_elements = []
        
        # Add nodes
        for node in nodes:
            element = {
                "data": {
                    "id": node["id"],
                    "label": node.get("label", node["id"]),
                    "type": node.get("type", "unknown"),
                    **node.get("metadata", {})
                }
            }
            
            # Add position if available
            if positions and node["id"] in positions:
                element["position"] = positions[node["id"]]
            
            cytoscape_elements.append(element)
        
        # Add edges
        for edge in edges:
            element = {
                "data": {
                    "id": f"{edge['source']}-{edge['target']}",
                    "source": edge["source"],
                    "target": edge["target"],
                    "label": edge.get("label", ""),
                    "weight": edge.get("weight", 1.0),
                    "type": edge.get("type", "unknown"),
                    **edge.get("metadata", {})
                }
            }
            cytoscape_elements.append(element)
        
        return {
            "elements": cytoscape_elements,
            "style": self._get_cytoscape_style(),
            "layout": {"name": "preset" if positions else "cose"}
        }
    
    def _get_cytoscape_style(self) -> List[Dict[str, Any]]:
        """Get default Cytoscape.js style"""
        return [
            {
                "selector": "node",
                "style": {
                    "background-color": "data(color)",
                    "label": "data(label)",
                    "width": "data(size)",
                    "height": "data(size)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "font-size": "12px"
                }
            },
            {
                "selector": "node[type='volcanic']",
                "style": {
                    "background-color": "#ff4444",
                    "shape": "triangle"
                }
            },
            {
                "selector": "node[type='seismic']",
                "style": {
                    "background-color": "#4444ff",
                    "shape": "diamond"
                }
            },
            {
                "selector": "node[type='tectonic']",
                "style": {
                    "background-color": "#44ff44",
                    "shape": "rectangle"
                }
            },
            {
                "selector": "edge",
                "style": {
                    "width": "data(width)",
                    "line-color": "#666",
                    "target-arrow-color": "#666",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "label": "data(label)",
                    "font-size": "10px",
                    "text-rotation": "autorotate"
                }
            },
            {
                "selector": "edge[type='causal']",
                "style": {
                    "line-color": "#ff6666",
                    "target-arrow-color": "#ff6666"
                }
            },
            {
                "selector": "edge[type='correlational']",
                "style": {
                    "line-color": "#6666ff",
                    "line-style": "dashed"
                }
            }
        ]
    
    async def export_to_graphml(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> str:
        """Export graph data in GraphML format"""
        
        graphml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
            '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
            '',
            '  <!-- Node attributes -->',
            '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="size" for="node" attr.name="size" attr.type="double"/>',
            '',
            '  <!-- Edge attributes -->',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>',
            '  <key id="relationship_type" for="edge" attr.name="relationship_type" attr.type="string"/>',
            '',
            '  <graph id="geological_network" edgedefault="directed">'
        ]
        
        # Add nodes
        for node in nodes:
            node_xml = f'    <node id="{node["id"]}">'
            node_xml += f'      <data key="name">{node.get("label", node["id"])}</data>'
            node_xml += f'      <data key="type">{node.get("type", "unknown")}</data>'
            node_xml += f'      <data key="size">{node.get("size", 10)}</data>'
            node_xml += '    </node>'
            graphml_parts.append(node_xml)
        
        # Add edges
        for i, edge in enumerate(edges):
            edge_xml = f'    <edge id="e{i}" source="{edge["source"]}" target="{edge["target"]}">'
            edge_xml += f'      <data key="weight">{edge.get("weight", 1.0)}</data>'
            edge_xml += f'      <data key="relationship_type">{edge.get("type", "unknown")}</data>'
            edge_xml += '    </edge>'
            graphml_parts.append(edge_xml)
        
        graphml_parts.extend([
            '  </graph>',
            '</graphml>'
        ])
        
        return '\n'.join(graphml_parts)
    
    async def export_to_gexf(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> str:
        """Export graph data in GEXF format"""
        
        gexf_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">',
            '  <meta lastmodifieddate="' + datetime.now().strftime("%Y-%m-%d") + '">',
            '    <creator>Prometheus Graph Database Service</creator>',
            '    <description>Geological Process Network</description>',
            '  </meta>',
            '  <graph mode="static" defaultedgetype="directed">',
            '',
            '    <attributes class="node">',
            '      <attribute id="0" title="type" type="string"/>',
            '      <attribute id="1" title="size" type="double"/>',
            '    </attributes>',
            '',
            '    <attributes class="edge">',
            '      <attribute id="0" title="weight" type="double"/>',
            '      <attribute id="1" title="relationship_type" type="string"/>',
            '    </attributes>',
            '',
            '    <nodes>'
        ]
        
        # Add nodes
        for node in nodes:
            node_xml = f'      <node id="{node["id"]}" label="{node.get("label", node["id"])}">'
            node_xml += '        <attvalues>'
            node_xml += f'          <attvalue for="0" value="{node.get("type", "unknown")}"/>'
            node_xml += f'          <attvalue for="1" value="{node.get("size", 10)}"/>'
            node_xml += '        </attvalues>'
            node_xml += '      </node>'
            gexf_parts.append(node_xml)
        
        gexf_parts.append('    </nodes>')
        gexf_parts.append('')
        gexf_parts.append('    <edges>')
        
        # Add edges
        for i, edge in enumerate(edges):
            edge_xml = f'      <edge id="{i}" source="{edge["source"]}" target="{edge["target"]}">'
            edge_xml += '        <attvalues>'
            edge_xml += f'          <attvalue for="0" value="{edge.get("weight", 1.0)}"/>'
            edge_xml += f'          <attvalue for="1" value="{edge.get("type", "unknown")}"/>'
            edge_xml += '        </attvalues>'
            edge_xml += '      </edge>'
            gexf_parts.append(edge_xml)
        
        gexf_parts.extend([
            '    </edges>',
            '  </graph>',
            '</gexf>'
        ])
        
        return '\n'.join(gexf_parts)
    
    async def create_interactive_visualization(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        config: GraphVisualizationConfig
    ) -> Dict[str, Any]:
        """Create interactive visualization data"""
        
        # Generate layout based on algorithm
        if config.layout_algorithm == "force_directed":
            positions = await self.generate_force_directed_layout(nodes, edges)
        elif config.layout_algorithm == "hierarchical":
            positions = await self.generate_hierarchical_layout(nodes, edges)
        elif config.layout_algorithm == "circular":
            positions = await self.generate_circular_layout(nodes, edges)
        else:
            positions = {}
        
        # Apply positions to nodes
        positioned_nodes = []
        for node in nodes:
            positioned_node = node.copy()
            if node["id"] in positions:
                positioned_node["position"] = positions[node["id"]]
            positioned_nodes.append(positioned_node)
        
        # Create visualization data
        visualization_data = {
            "nodes": positioned_nodes,
            "edges": edges,
            "layout": {
                "algorithm": config.layout_algorithm,
                "positions": positions
            },
            "config": config.dict(),
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        # Export in requested format
        if config.export_format == "cytoscape":
            visualization_data["cytoscape"] = await self.export_to_cytoscape(
                positioned_nodes, edges, positions
            )
        elif config.export_format == "graphml":
            visualization_data["graphml"] = await self.export_to_graphml(positioned_nodes, edges)
        elif config.export_format == "gexf":
            visualization_data["gexf"] = await self.export_to_gexf(positioned_nodes, edges)
        
        return visualization_data
    
    async def generate_network_summary(
        self,
        network_analysis: NetworkAnalysisResult
    ) -> Dict[str, Any]:
        """Generate a summary visualization of network analysis results"""
        
        summary = {
            "network_id": network_analysis.network_id,
            "basic_metrics": {
                "node_count": network_analysis.node_count,
                "edge_count": network_analysis.edge_count,
                "density": network_analysis.density
            },
            "centrality_summary": {},
            "community_summary": {},
            "visualization_recommendations": []
        }
        
        # Summarize centrality measures
        if network_analysis.centrality_measures:
            centrality_stats = {}
            for measure in ["pagerank", "betweenness", "closeness", "degree"]:
                values = [
                    measures.get(measure, 0.0) 
                    for measures in network_analysis.centrality_measures.values()
                ]
                if values:
                    centrality_stats[measure] = {
                        "mean": sum(values) / len(values),
                        "max": max(values),
                        "min": min(values)
                    }
            summary["centrality_summary"] = centrality_stats
        
        # Summarize communities
        if network_analysis.communities:
            summary["community_summary"] = {
                "community_count": len(network_analysis.communities),
                "largest_community_size": max(len(comm) for comm in network_analysis.communities),
                "smallest_community_size": min(len(comm) for comm in network_analysis.communities),
                "modularity": network_analysis.modularity
            }
        
        # Generate visualization recommendations
        recommendations = []
        
        if network_analysis.node_count > 100:
            recommendations.append("Use hierarchical layout for large networks")
            recommendations.append("Consider filtering by centrality measures")
        
        if network_analysis.density > 0.5:
            recommendations.append("Network is dense - consider edge filtering")
        
        if network_analysis.communities and len(network_analysis.communities) > 1:
            recommendations.append("Use community-based coloring")
            recommendations.append("Consider separate visualizations for each community")
        
        summary["visualization_recommendations"] = recommendations
        
        return summary