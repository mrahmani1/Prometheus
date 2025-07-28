import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { SeismicEvent } from '../../types/geological';

// Level of Detail (LOD) system for geological data
export class GeologicalLODManager {
  private camera: THREE.Camera;
  private lodLevels: Map<string, number> = new Map();
  private distanceThresholds = [10, 50, 200, 1000]; // Distance thresholds for LOD levels
  
  constructor(camera: THREE.Camera) {
    this.camera = camera;
  }
  
  updateLOD(objects: Array<{ id: string; position: THREE.Vector3 }>): Map<string, number> {
    const cameraPosition = this.camera.position;
    
    objects.forEach(obj => {
      const distance = cameraPosition.distanceTo(obj.position);
      let lodLevel = 0;
      
      for (let i = 0; i < this.distanceThresholds.length; i++) {
        if (distance > this.distanceThresholds[i]) {
          lodLevel = i + 1;
        } else {
          break;
        }
      }
      
      this.lodLevels.set(obj.id, lodLevel);
    });
    
    return this.lodLevels;
  }
  
  getLODLevel(id: string): number {
    return this.lodLevels.get(id) || 0;
  }
}

// Spatial indexing for efficient data queries
export class SpatialIndex {
  private octree: Map<string, any[]> = new Map();
  private cellSize: number;
  
  constructor(cellSize: number = 1.0) {
    this.cellSize = cellSize;
  }
  
  private getCellKey(position: THREE.Vector3): string {
    const x = Math.floor(position.x / this.cellSize);
    const y = Math.floor(position.y / this.cellSize);
    const z = Math.floor(position.z / this.cellSize);
    return `${x},${y},${z}`;
  }
  
  insert(id: string, position: THREE.Vector3, data: any): void {
    const key = this.getCellKey(position);
    if (!this.octree.has(key)) {
      this.octree.set(key, []);
    }
    this.octree.get(key)!.push({ id, position, data });
  }
  
  query(center: THREE.Vector3, radius: number): any[] {
    const results: any[] = [];
    const cellRadius = Math.ceil(radius / this.cellSize);
    
    const centerCell = {
      x: Math.floor(center.x / this.cellSize),
      y: Math.floor(center.y / this.cellSize),
      z: Math.floor(center.z / this.cellSize)
    };
    
    for (let x = centerCell.x - cellRadius; x <= centerCell.x + cellRadius; x++) {
      for (let y = centerCell.y - cellRadius; y <= centerCell.y + cellRadius; y++) {
        for (let z = centerCell.z - cellRadius; z <= centerCell.z + cellRadius; z++) {
          const key = `${x},${y},${z}`;
          const cell = this.octree.get(key);
          if (cell) {
            cell.forEach(item => {
              if (item.position.distanceTo(center) <= radius) {
                results.push(item);
              }
            });
          }
        }
      }
    }
    
    return results;
  }
  
  clear(): void {
    this.octree.clear();
  }
}

// Instanced rendering for large numbers of similar objects
interface InstancedSeismicEventsProps {
  events: SeismicEvent[];
  currentTime: string;
  timeWindow: number;
  maxInstances?: number;
}

export const InstancedSeismicEvents: React.FC<InstancedSeismicEventsProps> = ({
  events,
  currentTime,
  timeWindow,
  maxInstances = 10000
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  // const { camera } = useThree();
  // const lodManager = useRef(new GeologicalLODManager(camera));
  
  const { visibleEvents, instanceData } = useMemo(() => {
    const currentTimestamp = new Date(currentTime).getTime();
    
    // Filter events by time window
    const filtered = events.filter(event => {
      const eventTime = new Date(event.timestamp).getTime();
      return Math.abs(currentTimestamp - eventTime) <= timeWindow;
    }).slice(0, maxInstances); // Limit for performance
    
    // Prepare instance data
    const positions: number[] = [];
    const scales: number[] = [];
    const colors: number[] = [];
    
    filtered.forEach(event => {
      // Convert geological coordinates to scene coordinates
      const x = (event.location.longitude + 180) * 0.1 - 18;
      const z = (event.location.latitude + 90) * 0.1 - 9;
      const y = (event.location.elevation || 0) * 0.001;
      
      positions.push(x, y, z);
      
      // Scale based on magnitude
      const scale = Math.max(0.1, event.magnitude * 0.05);
      scales.push(scale, scale, scale);
      
      // Color based on event type
      const color = new THREE.Color(
        event.eventType === 'volcanic' ? '#ff4444' : 
        event.eventType === 'earthquake' ? '#4444ff' : '#44ff44'
      );
      colors.push(color.r, color.g, color.b);
    });
    
    return {
      visibleEvents: filtered,
      instanceData: { positions, scales, colors }
    };
  }, [events, currentTime, timeWindow, maxInstances]);
  
  // Update instance matrices
  useEffect(() => {
    if (!meshRef.current || visibleEvents.length === 0) return;
    
    const mesh = meshRef.current;
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    
    for (let i = 0; i < visibleEvents.length; i++) {
      const posIndex = i * 3;
      const scaleIndex = i * 3;
      
      position.set(
        instanceData.positions[posIndex],
        instanceData.positions[posIndex + 1],
        instanceData.positions[posIndex + 2]
      );
      
      scale.set(
        instanceData.scales[scaleIndex],
        instanceData.scales[scaleIndex + 1],
        instanceData.scales[scaleIndex + 2]
      );
      
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(i, matrix);
    }
    
    mesh.instanceMatrix.needsUpdate = true;
    mesh.count = visibleEvents.length;
  }, [visibleEvents, instanceData]);
  
  if (visibleEvents.length === 0) return null;
  
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, maxInstances]}
      frustumCulled={true}
    >
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial />
    </instancedMesh>
  );
};

// Temporal data cache for 4D visualization
export class TemporalDataCache {
  private cache: Map<string, any> = new Map();
  private maxCacheSize: number;
  private accessTimes: Map<string, number> = new Map();
  
  constructor(maxCacheSize: number = 1000) {
    this.maxCacheSize = maxCacheSize;
  }
  
  get(key: string): any {
    this.accessTimes.set(key, Date.now());
    return this.cache.get(key);
  }
  
  set(key: string, value: any): void {
    // Implement LRU eviction if cache is full
    if (this.cache.size >= this.maxCacheSize && !this.cache.has(key)) {
      const oldestKey = this.findOldestKey();
      if (oldestKey) {
        this.cache.delete(oldestKey);
        this.accessTimes.delete(oldestKey);
      }
    }
    
    this.cache.set(key, value);
    this.accessTimes.set(key, Date.now());
  }
  
  private findOldestKey(): string | null {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;
    
    for (const [key, time] of this.accessTimes.entries()) {
      if (time < oldestTime) {
        oldestTime = time;
        oldestKey = key;
      }
    }
    
    return oldestKey;
  }
  
  clear(): void {
    this.cache.clear();
    this.accessTimes.clear();
  }
  
  size(): number {
    return this.cache.size;
  }
}

// Frustum culling for geological objects
export const useFrustumCulling = (objects: Array<{ position: THREE.Vector3; id: string }>) => {
  const { camera } = useThree();
  const frustum = useRef(new THREE.Frustum());
  const matrix = useRef(new THREE.Matrix4());
  
  return useMemo(() => {
    matrix.current.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.current.setFromProjectionMatrix(matrix.current);
    
    return objects.filter(obj => {
      const sphere = new THREE.Sphere(obj.position, 1); // Assume radius of 1
      return frustum.current.intersectsSphere(sphere);
    });
  }, [objects, camera.projectionMatrix, camera.matrixWorldInverse]);
};

// Performance monitoring hook
export const usePerformanceMonitoring = () => {
  const { gl } = useThree();
  const [metrics, setMetrics] = React.useState({
    fps: 0,
    frameTime: 0,
    triangles: 0,
    drawCalls: 0,
    memoryUsage: 0
  });
  
  const lastTime = useRef(performance.now());
  const frameCount = useRef(0);
  
  useFrame(() => {
    const now = performance.now();
    const deltaTime = now - lastTime.current;
    frameCount.current++;
    
    // Update metrics every second
    if (deltaTime >= 1000) {
      const fps = Math.round((frameCount.current * 1000) / deltaTime);
      const frameTime = deltaTime / frameCount.current;
      
      setMetrics({
        fps,
        frameTime: Math.round(frameTime * 100) / 100,
        triangles: gl.info.render.triangles,
        drawCalls: gl.info.render.calls,
        memoryUsage: (gl.info.memory?.geometries || 0) + (gl.info.memory?.textures || 0)
      });
      
      lastTime.current = now;
      frameCount.current = 0;
    }
  });
  
  return metrics;
};

// Adaptive quality system
export class AdaptiveQualityManager {
  private targetFPS: number = 60;
  private currentQuality: number = 1.0; // 0.1 to 1.0
  private fpsHistory: number[] = [];
  private historySize: number = 30; // 30 frame history
  
  updateQuality(currentFPS: number): number {
    this.fpsHistory.push(currentFPS);
    if (this.fpsHistory.length > this.historySize) {
      this.fpsHistory.shift();
    }
    
    const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
    
    if (avgFPS < this.targetFPS * 0.8) {
      // Reduce quality if FPS is too low
      this.currentQuality = Math.max(0.1, this.currentQuality - 0.1);
    } else if (avgFPS > this.targetFPS * 0.95) {
      // Increase quality if FPS is stable
      this.currentQuality = Math.min(1.0, this.currentQuality + 0.05);
    }
    
    return this.currentQuality;
  }
  
  getQualitySettings() {
    return {
      shadowMapSize: Math.floor(1024 * this.currentQuality),
      particleCount: Math.floor(10000 * this.currentQuality),
      geometryDetail: Math.max(4, Math.floor(16 * this.currentQuality)),
      textureResolution: Math.floor(512 * this.currentQuality)
    };
  }
  
  getCurrentQuality(): number {
    return this.currentQuality;
  }
}

// Memory management utilities
export const useMemoryManagement = () => {
  const { gl } = useThree();
  
  const cleanupGeometry = React.useCallback((geometry: THREE.BufferGeometry) => {
    geometry.dispose();
  }, []);
  
  const cleanupMaterial = React.useCallback((material: THREE.Material | THREE.Material[]) => {
    if (Array.isArray(material)) {
      material.forEach(mat => mat.dispose());
    } else {
      material.dispose();
    }
  }, []);
  
  const cleanupTexture = React.useCallback((texture: THREE.Texture) => {
    texture.dispose();
  }, []);
  
  const getMemoryInfo = React.useCallback(() => {
    return {
      geometries: gl.info.memory?.geometries || 0,
      textures: gl.info.memory?.textures || 0,
      programs: gl.info.programs?.length || 0
    };
  }, [gl]);
  
  return {
    cleanupGeometry,
    cleanupMaterial,
    cleanupTexture,
    getMemoryInfo
  };
};