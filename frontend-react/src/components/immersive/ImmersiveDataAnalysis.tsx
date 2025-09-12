import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { VRButton, XR, Controllers, Hands } from '@react-three/xr';
import { Text, Html, Line, Sphere, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector } from 'react-redux';
import { 
  DataVisualization, 
  ImmersiveAnalysis, 
  SpatialQuery,
  DataManipulation,
  AnalysisResult
} from '../../types/immersive';
import { 
  updateImmersiveAnalysis,
  addSpatialQuery,
  updateDataVisualization
} from '../../store/slices/immersiveSlice';

interface ImmersiveDataAnalysisProps {
  geologicalData: any[];
  analysisTools?: string[];
  enableSpatialQueries?: boolean;
  enableDataManipulation?: boolean;
  onAnalysisComplete?: (result: AnalysisResult) => void;
}

interface DataPoint {
  id: string;
  position: THREE.Vector3;
  value: number;
  properties: Record<string, any>;
  type: 'seismic' | 'geochemical' | 'structural' | 'thermal';
}

interface AnalysisTool {
  name: string;
  type: 'statistical' | 'spatial' | 'temporal' | 'comparative';
  description: string;
  icon: string;
  action: (data: DataPoint[]) => AnalysisResult;
}

const ImmersiveDataAnalysis: React.FC<ImmersiveDataAnalysisProps> = ({
  geologicalData,
  analysisTools = ['statistics', 'clustering', 'correlation', 'trend'],
  enableSpatialQueries = true,
  enableDataManipulation = true,
  onAnalysisComplete
}) => {
  const dispatch = useDispatch();
  const analysisState = useSelector((state: any) => state.immersive.immersiveAnalysis);
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [selectedPoints, setSelectedPoints] = useState<Set<string>>(new Set());
  const [activeAnalysis, setActiveAnalysis] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [spatialSelection, setSpatialSelection] = useState<THREE.Box3 | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Available analysis tools
  const availableTools: AnalysisTool[] = [
    {
      name: 'Statistical Analysis',
      type: 'statistical',
      description: 'Calculate mean, median, std dev, and distribution',
      icon: '📊',
      action: performStatisticalAnalysis
    },
    {
      name: 'Spatial Clustering',
      type: 'spatial',
      description: 'Identify spatial clusters and patterns',
      icon: '🎯',
      action: performSpatialClustering
    },
    {
      name: 'Correlation Analysis',
      type: 'comparative',
      description: 'Find correlations between variables',
      icon: '🔗',
      action: performCorrelationAnalysis
    },
    {
      name: 'Trend Analysis',
      type: 'temporal',
      description: 'Identify temporal trends and patterns',
      icon: '📈',
      action: performTrendAnalysis
    },
    {
      name: 'Anomaly Detection',
      type: 'statistical',
      description: 'Detect outliers and anomalies',
      icon: '⚠️',
      action: performAnomalyDetection
    },
    {
      name: 'Spatial Interpolation',
      type: 'spatial',
      description: 'Interpolate values across space',
      icon: '🗺️',
      action: performSpatialInterpolation
    }
  ];

  useEffect(() => {
    // Convert geological data to data points
    const points = geologicalData.map((data, index) => ({
      id: `point_${index}`,
      position: new THREE.Vector3(
        data.x || Math.random() * 10 - 5,
        data.y || Math.random() * 10 - 5,
        data.z || Math.random() * 10 - 5
      ),
      value: data.value || Math.random(),
      properties: data.properties || {},
      type: data.type || 'seismic'
    }));
    
    setDataPoints(points);
  }, [geologicalData]);

  function performStatisticalAnalysis(data: DataPoint[]): AnalysisResult {
    const values = data.map(d => d.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return {
      id: `stats_${Date.now()}`,
      type: 'statistical',
      title: 'Statistical Analysis',
      data: {
        count: values.length,
        mean: mean.toFixed(3),
        stdDev: stdDev.toFixed(3),
        min: min.toFixed(3),
        max: max.toFixed(3),
        variance: variance.toFixed(3)
      },
      visualization: 'histogram',
      timestamp: new Date()
    };
  }

  function performSpatialClustering(data: DataPoint[]): AnalysisResult {
    // Simple k-means clustering implementation
    const k = Math.min(3, Math.floor(data.length / 5));
    const clusters = kMeansClustering(data, k);
    
    return {
      id: `cluster_${Date.now()}`,
      type: 'spatial',
      title: 'Spatial Clustering',
      data: {
        clusters: clusters.length,
        clusterData: clusters.map((cluster, i) => ({
          id: i,
          center: cluster.center,
          points: cluster.points.length,
          avgValue: cluster.points.reduce((a, b) => a + b.value, 0) / cluster.points.length
        }))
      },
      visualization: 'scatter3d',
      timestamp: new Date()
    };
  }

  function performCorrelationAnalysis(data: DataPoint[]): AnalysisResult {
    // Calculate correlations between position and value
    const correlations = {
      x_value: calculateCorrelation(data.map(d => d.position.x), data.map(d => d.value)),
      y_value: calculateCorrelation(data.map(d => d.position.y), data.map(d => d.value)),
      z_value: calculateCorrelation(data.map(d => d.position.z), data.map(d => d.value))
    };
    
    return {
      id: `corr_${Date.now()}`,
      type: 'comparative',
      title: 'Correlation Analysis',
      data: {
        correlations,
        strongestCorrelation: Object.entries(correlations)
          .reduce((a, b) => Math.abs(a[1]) > Math.abs(b[1]) ? a : b)
      },
      visualization: 'correlation_matrix',
      timestamp: new Date()
    };
  }

  function performTrendAnalysis(data: DataPoint[]): AnalysisResult {
    // Simple linear trend analysis
    const sortedData = [...data].sort((a, b) => a.position.x - b.position.x);
    const trend = calculateLinearTrend(
      sortedData.map(d => d.position.x),
      sortedData.map(d => d.value)
    );
    
    return {
      id: `trend_${Date.now()}`,
      type: 'temporal',
      title: 'Trend Analysis',
      data: {
        slope: trend.slope.toFixed(4),
        intercept: trend.intercept.toFixed(4),
        rSquared: trend.rSquared.toFixed(4),
        direction: trend.slope > 0 ? 'increasing' : 'decreasing'
      },
      visualization: 'line_chart',
      timestamp: new Date()
    };
  }

  function performAnomalyDetection(data: DataPoint[]): AnalysisResult {
    const values = data.map(d => d.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
    
    const anomalies = data.filter(d => Math.abs(d.value - mean) > 2 * stdDev);
    
    return {
      id: `anomaly_${Date.now()}`,
      type: 'statistical',
      title: 'Anomaly Detection',
      data: {
        totalPoints: data.length,
        anomalies: anomalies.length,
        anomalyRate: (anomalies.length / data.length * 100).toFixed(2),
        threshold: (2 * stdDev).toFixed(3),
        anomalyPoints: anomalies.map(a => ({
          id: a.id,
          value: a.value,
          deviation: Math.abs(a.value - mean).toFixed(3)
        }))
      },
      visualization: 'scatter_highlight',
      timestamp: new Date()
    };
  }

  function performSpatialInterpolation(data: DataPoint[]): AnalysisResult {
    // Simple inverse distance weighting interpolation
    const gridSize = 10;
    const interpolatedGrid = [];
    
    for (let x = -5; x <= 5; x += 10/gridSize) {
      for (let y = -5; y <= 5; y += 10/gridSize) {
        const point = new THREE.Vector3(x, y, 0);
        const interpolatedValue = inverseDistanceWeighting(point, data);
        interpolatedGrid.push({ x, y, value: interpolatedValue });
      }
    }
    
    return {
      id: `interp_${Date.now()}`,
      type: 'spatial',
      title: 'Spatial Interpolation',
      data: {
        gridSize,
        interpolatedPoints: interpolatedGrid.length,
        minValue: Math.min(...interpolatedGrid.map(p => p.value)),
        maxValue: Math.max(...interpolatedGrid.map(p => p.value)),
        grid: interpolatedGrid
      },
      visualization: 'heatmap',
      timestamp: new Date()
    };
  }

  const kMeansClustering = (data: DataPoint[], k: number) => {
    // Initialize centroids randomly
    let centroids = Array.from({ length: k }, () => ({
      x: Math.random() * 10 - 5,
      y: Math.random() * 10 - 5,
      z: Math.random() * 10 - 5
    }));
    
    let clusters: { center: any; points: DataPoint[] }[] = [];
    let iterations = 0;
    const maxIterations = 100;
    
    while (iterations < maxIterations) {
      // Assign points to nearest centroid
      clusters = centroids.map(c => ({ center: c, points: [] }));
      
      data.forEach(point => {
        let minDistance = Infinity;
        let nearestCluster = 0;
        
        centroids.forEach((centroid, i) => {
          const distance = Math.sqrt(
            Math.pow(point.position.x - centroid.x, 2) +
            Math.pow(point.position.y - centroid.y, 2) +
            Math.pow(point.position.z - centroid.z, 2)
          );
          
          if (distance < minDistance) {
            minDistance = distance;
            nearestCluster = i;
          }
        });
        
        clusters[nearestCluster].points.push(point);
      });
      
      // Update centroids
      const newCentroids = clusters.map(cluster => {
        if (cluster.points.length === 0) return cluster.center;
        
        return {
          x: cluster.points.reduce((sum, p) => sum + p.position.x, 0) / cluster.points.length,
          y: cluster.points.reduce((sum, p) => sum + p.position.y, 0) / cluster.points.length,
          z: cluster.points.reduce((sum, p) => sum + p.position.z, 0) / cluster.points.length
        };
      });
      
      // Check for convergence
      const converged = centroids.every((centroid, i) => {
        const newCentroid = newCentroids[i];
        return Math.abs(centroid.x - newCentroid.x) < 0.01 &&
               Math.abs(centroid.y - newCentroid.y) < 0.01 &&
               Math.abs(centroid.z - newCentroid.z) < 0.01;
      });
      
      if (converged) break;
      
      centroids = newCentroids;
      iterations++;
    }
    
    return clusters;
  };

  const calculateCorrelation = (x: number[], y: number[]): number => {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  };

  const calculateLinearTrend = (x: number[], y: number[]) => {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R-squared
    const yMean = sumY / n;
    const totalSumSquares = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const residualSumSquares = y.reduce((sum, yi, i) => {
      const predicted = slope * x[i] + intercept;
      return sum + Math.pow(yi - predicted, 2);
    }, 0);
    const rSquared = 1 - (residualSumSquares / totalSumSquares);
    
    return { slope, intercept, rSquared };
  };

  const inverseDistanceWeighting = (point: THREE.Vector3, data: DataPoint[], power: number = 2): number => {
    let weightedSum = 0;
    let weightSum = 0;
    
    data.forEach(dataPoint => {
      const distance = point.distanceTo(dataPoint.position);
      if (distance === 0) return dataPoint.value; // Exact match
      
      const weight = 1 / Math.pow(distance, power);
      weightedSum += weight * dataPoint.value;
      weightSum += weight;
    });
    
    return weightSum === 0 ? 0 : weightedSum / weightSum;
  };

  const runAnalysis = useCallback(async (toolName: string) => {
    const tool = availableTools.find(t => t.name === toolName);
    if (!tool) return;
    
    setIsAnalyzing(true);
    setActiveAnalysis(toolName);
    
    try {
      // Use selected points if any, otherwise use all points
      const analysisData = selectedPoints.size > 0 
        ? dataPoints.filter(p => selectedPoints.has(p.id))
        : dataPoints;
      
      const result = tool.action(analysisData);
      
      setAnalysisResults(prev => [...prev, result]);
      dispatch(updateImmersiveAnalysis({
        activeAnalysis: toolName,
        results: result,
        timestamp: Date.now()
      }));
      
      onAnalysisComplete?.(result);
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
      setActiveAnalysis(null);
    }
  }, [dataPoints, selectedPoints, dispatch, onAnalysisComplete]);

  const selectPointsInRegion = useCallback((region: THREE.Box3) => {
    const pointsInRegion = dataPoints.filter(point => 
      region.containsPoint(point.position)
    );
    
    const newSelection = new Set(pointsInRegion.map(p => p.id));
    setSelectedPoints(newSelection);
    setSpatialSelection(region);
    
    dispatch(addSpatialQuery({
      id: `query_${Date.now()}`,
      region: {
        min: region.min,
        max: region.max
      },
      pointCount: pointsInRegion.length,
      timestamp: new Date()
    }));
  }, [dataPoints, dispatch]);

  return (
    <div className="immersive-data-analysis">
      <div className="analysis-controls">
        <h3>Immersive Data Analysis</h3>
        <div className="analysis-info">
          <p>Data Points: {dataPoints.length}</p>
          <p>Selected: {selectedPoints.size}</p>
          <p>Results: {analysisResults.length}</p>
          {isAnalyzing && <p>Analyzing: {activeAnalysis}</p>}
        </div>
      </div>

      <VRButton />

      <Canvas style={{ height: '100vh' }}>
        <XR>
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} />
          
          <Controllers />
          <Hands />
          
          <ImmersiveAnalysisScene
            dataPoints={dataPoints}
            selectedPoints={selectedPoints}
            analysisResults={analysisResults}
            spatialSelection={spatialSelection}
            onPointSelection={setSelectedPoints}
            onRegionSelection={selectPointsInRegion}
            enableSpatialQueries={enableSpatialQueries}
            enableDataManipulation={enableDataManipulation}
          />
          
          <AnalysisToolsInterface
            tools={availableTools}
            onRunAnalysis={runAnalysis}
            isAnalyzing={isAnalyzing}
            activeAnalysis={activeAnalysis}
          />
          
          <AnalysisResultsDisplay
            results={analysisResults}
            selectedPoints={selectedPoints}
          />
        </XR>
      </Canvas>
    </div>
  );
};

interface ImmersiveAnalysisSceneProps {
  dataPoints: DataPoint[];
  selectedPoints: Set<string>;
  analysisResults: AnalysisResult[];
  spatialSelection: THREE.Box3 | null;
  onPointSelection: (selection: Set<string>) => void;
  onRegionSelection: (region: THREE.Box3) => void;
  enableSpatialQueries: boolean;
  enableDataManipulation: boolean;
}

const ImmersiveAnalysisScene: React.FC<ImmersiveAnalysisSceneProps> = ({
  dataPoints,
  selectedPoints,
  analysisResults,
  spatialSelection,
  onPointSelection,
  onRegionSelection,
  enableSpatialQueries,
  enableDataManipulation
}) => {
  const selectionBoxRef = useRef<THREE.Mesh>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionStart, setSelectionStart] = useState<THREE.Vector3 | null>(null);

  const getPointColor = (point: DataPoint): number => {
    if (selectedPoints.has(point.id)) {
      return 0xFFFF00; // Yellow for selected
    }
    
    // Color by type
    switch (point.type) {
      case 'seismic': return 0xFF0000;
      case 'geochemical': return 0x00FF00;
      case 'structural': return 0x0000FF;
      case 'thermal': return 0xFF8800;
      default: return 0xFFFFFF;
    }
  };

  const getPointSize = (point: DataPoint): number => {
    return selectedPoints.has(point.id) ? 0.15 : 0.1;
  };

  return (
    <>
      {/* Render data points */}
      {dataPoints.map(point => (
        <DataPointVisualization
          key={point.id}
          point={point}
          color={getPointColor(point)}
          size={getPointSize(point)}
          onSelect={(pointId) => {
            const newSelection = new Set(selectedPoints);
            if (newSelection.has(pointId)) {
              newSelection.delete(pointId);
            } else {
              newSelection.add(pointId);
            }
            onPointSelection(newSelection);
          }}
        />
      ))}
      
      {/* Spatial selection box */}
      {spatialSelection && (
        <SelectionBox region={spatialSelection} />
      )}
      
      {/* Analysis result visualizations */}
      {analysisResults.map(result => (
        <AnalysisResultVisualization
          key={result.id}
          result={result}
          dataPoints={dataPoints}
        />
      ))}
      
      {/* Spatial query tools */}
      {enableSpatialQueries && (
        <SpatialQueryTools
          onRegionSelection={onRegionSelection}
        />
      )}
    </>
  );
};

const DataPointVisualization: React.FC<{
  point: DataPoint;
  color: number;
  size: number;
  onSelect: (pointId: string) => void;
}> = ({ point, color, size, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(hovered ? size * 1.2 : size);
    }
  });

  return (
    <group position={point.position}>
      <mesh
        ref={meshRef}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
        onClick={() => onSelect(point.id)}
      >
        <sphereGeometry args={[1, 8, 8]} />
        <meshStandardMaterial
          color={color}
          emissive={hovered ? 0x222222 : 0x000000}
        />
      </mesh>
      
      {hovered && (
        <Text
          position={[0, size + 0.1, 0]}
          fontSize={0.05}
          color="white"
          anchorX="center"
          anchorY="bottom"
        >
          {`${point.type}: ${point.value.toFixed(3)}`}
        </Text>
      )}
    </group>
  );
};

const SelectionBox: React.FC<{ region: THREE.Box3 }> = ({ region }) => {
  const size = region.getSize(new THREE.Vector3());
  const center = region.getCenter(new THREE.Vector3());

  return (
    <mesh position={center}>
      <boxGeometry args={[size.x, size.y, size.z]} />
      <meshBasicMaterial
        color={0x00FFFF}
        transparent
        opacity={0.2}
        wireframe
      />
    </mesh>
  );
};

const AnalysisResultVisualization: React.FC<{
  result: AnalysisResult;
  dataPoints: DataPoint[];
}> = ({ result, dataPoints }) => {
  // Render different visualizations based on result type
  switch (result.visualization) {
    case 'scatter3d':
      return <ClusterVisualization result={result} />;
    case 'heatmap':
      return <HeatmapVisualization result={result} />;
    default:
      return null;
  }
};

const ClusterVisualization: React.FC<{ result: AnalysisResult }> = ({ result }) => {
  if (!result.data.clusterData) return null;

  return (
    <>
      {result.data.clusterData.map((cluster: any, index: number) => (
        <group key={index}>
          <Sphere
            args={[0.2, 8, 8]}
            position={[cluster.center.x, cluster.center.y, cluster.center.z]}
          >
            <meshStandardMaterial
              color={`hsl(${index * 60}, 70%, 50%)`}
              transparent
              opacity={0.6}
            />
          </Sphere>
          
          <Text
            position={[cluster.center.x, cluster.center.y + 0.3, cluster.center.z]}
            fontSize={0.05}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {`Cluster ${index + 1}`}
          </Text>
        </group>
      ))}
    </>
  );
};

const HeatmapVisualization: React.FC<{ result: AnalysisResult }> = ({ result }) => {
  if (!result.data.grid) return null;

  return (
    <>
      {result.data.grid.map((point: any, index: number) => {
        const intensity = (point.value - result.data.minValue) / 
                         (result.data.maxValue - result.data.minValue);
        
        return (
          <Plane
            key={index}
            args={[0.5, 0.5]}
            position={[point.x, point.y, -0.1]}
          >
            <meshBasicMaterial
              color={`hsl(${(1 - intensity) * 240}, 100%, 50%)`}
              transparent
              opacity={0.7}
            />
          </Plane>
        );
      })}
    </>
  );
};

const SpatialQueryTools: React.FC<{
  onRegionSelection: (region: THREE.Box3) => void;
}> = ({ onRegionSelection }) => {
  // Placeholder for spatial query tools
  return null;
};

const AnalysisToolsInterface: React.FC<{
  tools: AnalysisTool[];
  onRunAnalysis: (toolName: string) => void;
  isAnalyzing: boolean;
  activeAnalysis: string | null;
}> = ({ tools, onRunAnalysis, isAnalyzing, activeAnalysis }) => {
  return (
    <group position={[-3, 2, -2]}>
      <Text
        position={[0, 1, 0]}
        fontSize={0.15}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Analysis Tools
      </Text>
      
      {tools.slice(0, 6).map((tool, index) => (
        <group key={tool.name} position={[0, 0.5 - index * 0.2, 0]}>
          <Box
            args={[2, 0.15, 0.05]}
            onClick={() => !isAnalyzing && onRunAnalysis(tool.name)}
          >
            <meshStandardMaterial
              color={activeAnalysis === tool.name ? 0xFFFF00 : 0x444444}
            />
          </Box>
          
          <Text
            position={[0, 0, 0.03]}
            fontSize={0.06}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {tool.icon} {tool.name}
          </Text>
        </group>
      ))}
      
      {isAnalyzing && (
        <Text
          position={[0, -1.5, 0]}
          fontSize={0.08}
          color="yellow"
          anchorX="center"
          anchorY="middle"
        >
          Analyzing...
        </Text>
      )}
    </group>
  );
};

const AnalysisResultsDisplay: React.FC<{
  results: AnalysisResult[];
  selectedPoints: Set<string>;
}> = ({ results, selectedPoints }) => {
  const latestResult = results[results.length - 1];
  
  if (!latestResult) return null;

  return (
    <group position={[3, 2, -2]}>
      <Text
        position={[0, 1, 0]}
        fontSize={0.15}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Latest Result
      </Text>
      
      <Text
        position={[0, 0.7, 0]}
        fontSize={0.1}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        {latestResult.title}
      </Text>
      
      {Object.entries(latestResult.data).slice(0, 5).map(([key, value], index) => (
        <Text
          key={key}
          position={[0, 0.4 - index * 0.15, 0]}
          fontSize={0.06}
          color="lightblue"
          anchorX="center"
          anchorY="middle"
        >
          {`${key}: ${typeof value === 'number' ? value.toFixed(3) : value}`}
        </Text>
      ))}
    </group>
  );
};

export default ImmersiveDataAnalysis;