import * as d3 from 'd3';
import { CausalNode, CausalEdge, CausalGraph, CausalLayoutConfig } from './types';

export class CausalLayoutEngine {
  private simulation: d3.Simulation<CausalNode, CausalEdge> | null = null;
  private width: number;
  private height: number;
  private config: CausalLayoutConfig;

  constructor(width: number, height: number, config: CausalLayoutConfig) {
    this.width = width;
    this.height = height;
    this.config = config;
  }

  /**
   * Generate deterministic hash from string for consistent positioning
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  public updateDimensions(width: number, height: number): void {
    this.width = width;
    this.height = height;
    if (this.simulation) {
      this.simulation.force('center', d3.forceCenter(width / 2, height / 2));
    }
  }

  public updateConfig(config: CausalLayoutConfig): void {
    this.config = config;
    if (this.simulation) {
      this.applyForces();
    }
  }

  public layoutGraph(
    graph: CausalGraph,
    onTick?: () => void,
    onEnd?: () => void
  ): d3.Simulation<CausalNode, CausalEdge> {
    // Stop existing simulation
    if (this.simulation) {
      this.simulation.stop();
    }

    // Initialize node positions if not set
    graph.nodes.forEach(node => {
      if (!node.position) {
        // Use deterministic positioning based on node properties instead of random
        const hash = this.hashString(node.id);
        const angle = (hash % 360) * (Math.PI / 180);
        const radius = Math.min(this.width, this.height) * 0.3;
        
        node.position = {
          x: this.width / 2 + Math.cos(angle) * radius,
          y: this.height / 2 + Math.sin(angle) * radius
        };
      }
    });

    // Create simulation based on algorithm
    switch (this.config.algorithm) {
      case 'force-directed':
        this.simulation = this.createForceDirectedLayout(graph);
        break;
      case 'hierarchical':
        this.simulation = this.createHierarchicalLayout(graph);
        break;
      case 'circular':
        this.simulation = this.createCircularLayout(graph);
        break;
      case 'dagre':
        this.simulation = this.createDagreLayout(graph);
        break;
      case 'cola':
        this.simulation = this.createColaLayout(graph);
        break;
      case 'cose-bilkent':
        this.simulation = this.createCoseBilkentLayout(graph);
        break;
      default:
        this.simulation = this.createForceDirectedLayout(graph);
    }

    // Set up event handlers
    if (onTick) {
      this.simulation.on('tick', onTick);
    }

    if (onEnd) {
      this.simulation.on('end', onEnd);
    }

    return this.simulation;
  }

  private createForceDirectedLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    const simulation = d3.forceSimulation<CausalNode>(graph.nodes);

    this.applyForces(simulation, graph);

    return simulation;
  }

  private createHierarchicalLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    // Create hierarchical structure using d3-hierarchy
    const hierarchy = this.buildHierarchy(graph);
    const treeLayout = d3.tree<CausalNode>()
      .size([this.width, this.height])
      .separation((a, b) => (a.parent === b.parent ? 1 : 2) / a.depth);

    const root = treeLayout(hierarchy);

    // Apply hierarchical positions
    root.descendants().forEach(d => {
      if (d.data.position) {
        d.data.position.x = d.x;
        d.data.position.y = d.y;
      }
    });

    // Create simulation with fixed positions for hierarchical layout
    const simulation = d3.forceSimulation<CausalNode>(graph.nodes)
      .force('x', d3.forceX<CausalNode>(d => d.position?.x || 0).strength(0.8))
      .force('y', d3.forceY<CausalNode>(d => d.position?.y || 0).strength(0.8))
      .force('collision', d3.forceCollide<CausalNode>(d => d.size + 5));

    return simulation;
  }

  private createCircularLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    const radius = Math.min(this.width, this.height) / 2 - 50;
    const angleStep = (2 * Math.PI) / graph.nodes.length;

    // Position nodes in a circle
    graph.nodes.forEach((node, i) => {
      const angle = i * angleStep;
      if (node.position) {
        node.position.x = this.width / 2 + radius * Math.cos(angle);
        node.position.y = this.height / 2 + radius * Math.sin(angle);
      }
    });

    const simulation = d3.forceSimulation<CausalNode>(graph.nodes)
      .force('x', d3.forceX<CausalNode>(d => d.position?.x || 0).strength(0.5))
      .force('y', d3.forceY<CausalNode>(d => d.position?.y || 0).strength(0.5))
      .force('collision', d3.forceCollide<CausalNode>(d => d.size + 5));

    return simulation;
  }

  private createDagreLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    // Implement Dagre layout algorithm for directed acyclic graphs
    const levels = this.computeNodeLevels(graph);
    const maxLevel = Math.max(...Object.values(levels));

    graph.nodes.forEach(node => {
      const level = levels[node.id] || 0;
      const nodesAtLevel = Object.values(levels).filter(l => l === level).length;
      const indexAtLevel = Object.keys(levels)
        .filter(id => levels[id] === level)
        .indexOf(node.id);

      if (node.position) {
        node.position.x = (indexAtLevel + 1) * (this.width / (nodesAtLevel + 1));
        node.position.y = (level + 1) * (this.height / (maxLevel + 2));
      }
    });

    const simulation = d3.forceSimulation<CausalNode>(graph.nodes)
      .force('x', d3.forceX<CausalNode>(d => d.position?.x || 0).strength(0.8))
      .force('y', d3.forceY<CausalNode>(d => d.position?.y || 0).strength(0.8))
      .force('collision', d3.forceCollide<CausalNode>(d => d.size + 10));

    return simulation;
  }

  private createColaLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    // Implement CoLa (Constraint-based Layout) algorithm
    const simulation = d3.forceSimulation<CausalNode>(graph.nodes);

    // Add constraint-based forces
    simulation
      .force('link', d3.forceLink<CausalNode, CausalEdge>(graph.edges)
        .id(d => d.id)
        .distance(d => this.config.linkDistance * (1 / Math.max(d.strength, 0.1)))
        .strength(d => d.strength * 0.5))
      .force('charge', d3.forceManyBody<CausalNode>()
        .strength(-this.config.nodeRepulsion))
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collision', d3.forceCollide<CausalNode>(d => d.size + 5));

    // Add clustering constraints if enabled
    if (this.config.clustering.enabled) {
      this.addClusteringConstraints(simulation, graph);
    }

    return simulation;
  }

  private createCoseBilkentLayout(graph: CausalGraph): d3.Simulation<CausalNode, CausalEdge> {
    // Implement COSE-Bilkent layout algorithm (Compound Spring Embedder)
    const simulation = d3.forceSimulation<CausalNode>(graph.nodes);

    // Multi-level force application
    const strongForce = d3.forceLink<CausalNode, CausalEdge>(graph.edges)
      .id(d => d.id)
      .distance(d => this.config.linkDistance * (2 - d.strength))
      .strength(d => d.strength);

    const repulsionForce = d3.forceManyBody<CausalNode>()
      .strength(d => -this.config.nodeRepulsion * Math.sqrt(d.size));

    simulation
      .force('link', strongForce)
      .force('charge', repulsionForce)
      .force('center', d3.forceCenter(this.width / 2, this.height / 2))
      .force('collision', d3.forceCollide<CausalNode>(d => d.size + 8));

    return simulation;
  }

  private applyForces(
    simulation?: d3.Simulation<CausalNode, CausalEdge>,
    graph?: CausalGraph
  ): void {
    const sim = simulation || this.simulation;
    if (!sim || !graph) return;

    // Link force with causal strength weighting
    const linkForce = d3.forceLink<CausalNode, CausalEdge>(graph.edges);
    if (linkForce.id) {
      linkForce.id(d => d.id);
    }
    if (linkForce.distance) {
      linkForce.distance(d => this.config.linkDistance * (1 / Math.max(d.strength, 0.1)));
    }
    if (linkForce.strength) {
      linkForce.strength(d => d.strength * 0.3);
    }
    sim.force('link', linkForce);

    // Repulsion force based on node importance
    const chargeForce = d3.forceManyBody<CausalNode>();
    if (chargeForce.strength) {
      chargeForce.strength(d => -this.config.nodeRepulsion * Math.sqrt(d.size));
    }
    sim.force('charge', chargeForce);

    // Centering force
    const centerForce = d3.forceCenter(this.width / 2, this.height / 2);
    if (centerForce.strength) {
      centerForce.strength(this.config.centerForce);
    }
    sim.force('center', centerForce);

    // Collision detection
    const collisionForce = d3.forceCollide<CausalNode>();
    if (collisionForce.radius) {
      collisionForce.radius(d => d.size + this.config.collisionRadius);
    }
    if (collisionForce.strength) {
      collisionForce.strength(0.7);
    }
    sim.force('collision', collisionForce);

    // Clustering forces if enabled
    if (this.config.clustering.enabled && graph) {
      this.addClusteringForces(sim, graph);
    }

    // Update simulation parameters
    if (sim.alphaDecay) {
      sim.alphaDecay(this.config.alphaDecay);
    }
    if (sim.velocityDecay) {
      sim.velocityDecay(this.config.velocityDecay);
    }
  }

  private addClusteringForces(
    simulation: d3.Simulation<CausalNode, CausalEdge>,
    graph: CausalGraph
  ): void {
    const clusters = this.createClusters(graph);

    // Add cluster centering forces
    Object.entries(clusters).forEach(([clusterKey, nodes]) => {
      const clusterCenter = this.calculateClusterCenter(nodes);
      
      simulation.force(`cluster-${clusterKey}`, d3.forceX<CausalNode>(clusterCenter.x)
        .strength(this.config.clustering.clusterStrength)
        .x(d => clusters[this.getClusterKey(d)] === nodes ? clusterCenter.x : d.position?.x || 0));
    });
  }

  private addClusteringConstraints(
    simulation: d3.Simulation<CausalNode, CausalEdge>,
    graph: CausalGraph
  ): void {
    // Add constraint-based clustering using CoLa-style constraints
    const clusters = this.createClusters(graph);

    Object.values(clusters).forEach(clusterNodes => {
      if (clusterNodes.length > 1) {
        // Add intra-cluster attraction
        for (let i = 0; i < clusterNodes.length; i++) {
          for (let j = i + 1; j < clusterNodes.length; j++) {
            const node1 = clusterNodes[i];
            const node2 = clusterNodes[j];
            
            // Create virtual edges for clustering
            simulation.force(`cluster-${node1.id}-${node2.id}`, 
              d3.forceLink<CausalNode, CausalEdge>([{
                id: `cluster-${node1.id}-${node2.id}`,
                source: node1.id,
                target: node2.id,
                strength: this.config.clustering.clusterStrength,
                confidence: [0.8, 1.0],
                edgeType: 'direct',
                metadata: {
                  discoveryMethod: 'clustering',
                  statisticalSignificance: 1.0,
                  validationStatus: 'validated',
                  geologicalContext: 'cluster-constraint'
                },
                properties: {}
              }])
              .id(d => d.id)
              .distance(50)
              .strength(0.1)
            );
          }
        }
      }
    });
  }

  private createClusters(graph: CausalGraph): Record<string, CausalNode[]> {
    const clusters: Record<string, CausalNode[]> = {};

    graph.nodes.forEach(node => {
      const clusterKey = this.getClusterKey(node);
      if (!clusters[clusterKey]) {
        clusters[clusterKey] = [];
      }
      clusters[clusterKey].push(node);
    });

    return clusters;
  }

  private getClusterKey(node: CausalNode): string {
    switch (this.config.clustering.clusterBy) {
      case 'processType':
        return node.processType;
      case 'spatialScale':
        return node.metadata.spatialScale;
      case 'temporalScale':
        return node.metadata.temporalScale;
      case 'dataSource':
        return node.metadata.dataSource;
      case 'geologicalContext':
        return node.properties.geologicalContext || 'unknown';
      default:
        return 'default';
    }
  }

  private calculateClusterCenter(nodes: CausalNode[]): { x: number; y: number } {
    const totalX = nodes.reduce((sum, node) => sum + (node.position?.x || 0), 0);
    const totalY = nodes.reduce((sum, node) => sum + (node.position?.y || 0), 0);
    
    return {
      x: totalX / nodes.length,
      y: totalY / nodes.length
    };
  }

  private buildHierarchy(graph: CausalGraph): d3.HierarchyNode<CausalNode> {
    // Find root nodes (nodes with no incoming edges)
    const incomingEdges = new Set(graph.edges.map(e => e.target));
    const rootNodes = graph.nodes.filter(node => !incomingEdges.has(node.id));

    // If no clear root, use the node with highest centrality
    if (rootNodes.length === 0) {
      const centrality = this.calculateCentrality(graph);
      const rootNode = graph.nodes.reduce((max, node) => 
        centrality[node.id] > centrality[max.id] ? node : max
      );
      rootNodes.push(rootNode);
    }

    // Build hierarchy from root nodes
    const hierarchyData = this.buildHierarchyRecursive(rootNodes[0], graph, new Set());
    return d3.hierarchy(hierarchyData);
  }

  private buildHierarchyRecursive(
    node: CausalNode,
    graph: CausalGraph,
    visited: Set<string>
  ): any {
    if (visited.has(node.id)) {
      return { ...node, children: [] };
    }

    visited.add(node.id);

    const children = graph.edges
      .filter(edge => edge.source === node.id)
      .map(edge => graph.nodes.find(n => n.id === edge.target))
      .filter(child => child && !visited.has(child.id))
      .map(child => this.buildHierarchyRecursive(child!, graph, visited));

    return { ...node, children };
  }

  private calculateCentrality(graph: CausalGraph): Record<string, number> {
    const centrality: Record<string, number> = {};
    
    // Initialize centrality scores
    graph.nodes.forEach(node => {
      centrality[node.id] = 0;
    });

    // Calculate degree centrality weighted by causal strength
    graph.edges.forEach(edge => {
      centrality[edge.source] += edge.strength;
      centrality[edge.target] += edge.strength;
    });

    return centrality;
  }

  private computeNodeLevels(graph: CausalGraph): Record<string, number> {
    const levels: Record<string, number> = {};
    const visited = new Set<string>();
    const visiting = new Set<string>();

    // Topological sort to determine levels
    const visit = (nodeId: string): number => {
      if (visiting.has(nodeId)) {
        // Cycle detected, assign arbitrary level
        return 0;
      }
      if (visited.has(nodeId)) {
        return levels[nodeId];
      }

      visiting.add(nodeId);

      const incomingEdges = graph.edges.filter(e => e.target === nodeId);
      let maxLevel = -1;

      incomingEdges.forEach(edge => {
        const sourceLevel = visit(edge.source);
        maxLevel = Math.max(maxLevel, sourceLevel);
      });

      visiting.delete(nodeId);
      visited.add(nodeId);
      levels[nodeId] = maxLevel + 1;

      return levels[nodeId];
    };

    graph.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        visit(node.id);
      }
    });

    return levels;
  }

  public stop(): void {
    if (this.simulation) {
      this.simulation.stop();
    }
  }

  public restart(): void {
    if (this.simulation) {
      this.simulation.restart();
    }
  }

  public getSimulation(): d3.Simulation<CausalNode, CausalEdge> | null {
    return this.simulation;
  }
}