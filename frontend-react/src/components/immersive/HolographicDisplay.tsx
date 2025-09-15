import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Text, Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector } from 'react-redux';
import { 
  HolographicProjection
} from '../../types/immersive';
import { 
  updateHolographicDisplay,
  addHologram,
  updateSpatialComputing
} from '../../store/slices/immersiveSlice';

interface HolographicDisplayProps {
  geologicalData: any[];
  projectionType?: 'volumetric' | 'planar' | 'cylindrical' | 'spherical';
  enableSpatialComputing?: boolean;
  enableMixedReality?: boolean;
  hologramScale?: number;
  interactionMode?: 'gesture' | 'voice' | 'controller' | 'eye_tracking';
}

interface Hologram {
  id: string;
  type: 'geological_model' | 'data_visualization' | 'interface_element' | 'annotation';
  position: THREE.Vector3;
  rotation: THREE.Euler;
  scale: THREE.Vector3;
  data: any;
  opacity: number;
  interactive: boolean;
  animation?: HologramAnimation;
}

interface HologramAnimation {
  type: 'rotate' | 'pulse' | 'float' | 'morph';
  duration: number;
  loop: boolean;
  parameters: Record<string, any>;
}

interface SpatialAnchor {
  id: string;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  persistent: boolean;
  holograms: string[];
}

const HolographicDisplay: React.FC<HolographicDisplayProps> = ({
  geologicalData,
  projectionType = 'volumetric',
  enableSpatialComputing = true,
  enableMixedReality = true,
  hologramScale = 1.0,
  interactionMode = 'gesture'
}) => {
  const dispatch = useDispatch();
  const holographicState = useSelector((state: any) => state.immersive.holographicDisplay);
  const [holograms, setHolograms] = useState<Map<string, Hologram>>(new Map());
  const [spatialAnchors, setSpatialAnchors] = useState<Map<string, SpatialAnchor>>(new Map());
  const [activeProjection, setActiveProjection] = useState<HolographicProjection | null>(null);
  const [eyeTrackingData, setEyeTrackingData] = useState<any>(null);
  const [spatialMesh, setSpatialMesh] = useState<THREE.Mesh[]>([]);
  const projectionVolumeRef = useRef<THREE.Group>(null);
  
  // Use holographic state for display configuration
  const displayEnabled = holographicState?.enabled ?? (enableSpatialComputing || enableMixedReality);
  
  // Use eye tracking data for gaze-based interactions
  const handleEyeTracking = useCallback((data: any) => {
    setEyeTrackingData(data);
    // Process eye tracking for hologram focus
  }, []);
  
  // Use projection volume for spatial calculations
  const updateProjectionVolume = useCallback(() => {
    if (projectionVolumeRef.current && activeProjection) {
      // Update projection volume based on active projection
      // Update projection volume based on active projection bounds
      const center = {
        x: (activeProjection.bounds.min.x + activeProjection.bounds.max.x) / 2,
        y: (activeProjection.bounds.min.y + activeProjection.bounds.max.y) / 2,
        z: (activeProjection.bounds.min.z + activeProjection.bounds.max.z) / 2
      };
      projectionVolumeRef.current.position.set(center.x, center.y, center.z);
    }
  }, [activeProjection]);
  const hologramShaderRef = useRef<THREE.ShaderMaterial | null>(null);

  // Holographic shader for realistic hologram effects
  const hologramVertexShader = `
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;
    
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const hologramFragmentShader = `
    uniform float time;
    uniform float opacity;
    uniform vec3 color;
    uniform float scanlineFrequency;
    uniform float interferenceStrength;
    
    varying vec3 vNormal;
    varying vec3 vPosition;
    varying vec2 vUv;
    
    void main() {
      // Fresnel effect for holographic rim lighting
      float fresnel = pow(1.0 - abs(dot(vNormal, normalize(vPosition))), 2.0);
      
      // Scanlines effect
      float scanlines = sin(vUv.y * scanlineFrequency + time * 2.0) * 0.1 + 0.9;
      
      // Interference pattern
      float interference = sin(vUv.x * 50.0 + time) * sin(vUv.y * 50.0 + time * 1.5) * interferenceStrength;
      
      // Holographic flicker
      float flicker = sin(time * 10.0) * 0.05 + 0.95;
      
      vec3 finalColor = color * (fresnel + 0.3) * scanlines * flicker + interference;
      float finalOpacity = opacity * fresnel * scanlines * flicker;
      
      gl_FragColor = vec4(finalColor, finalOpacity);
    }
  `;

  useEffect(() => {
    initializeHolographicSystem();
    
    if (enableSpatialComputing) {
      initializeSpatialComputing();
    }
    
    if (enableMixedReality) {
      initializeMixedReality();
    }
    
    // Initialize display based on configuration
    if (displayEnabled) {
      console.log('Holographic display enabled');
    }
    
    // Set up eye tracking if available
    if (eyeTrackingData) {
      handleEyeTracking(eyeTrackingData);
    }
    
    // Update projection volume
    updateProjectionVolume();
    
    // Create holograms from geological data
    createGeologicalHolograms();
  }, [geologicalData, enableSpatialComputing, enableMixedReality]);

  const initializeHolographicSystem = useCallback(() => {
    // Initialize holographic shader material
    const shaderMaterial = new THREE.ShaderMaterial({
      vertexShader: hologramVertexShader,
      fragmentShader: hologramFragmentShader,
      uniforms: {
        time: { value: 0 },
        opacity: { value: 0.8 },
        color: { value: new THREE.Color(0x00FFFF) },
        scanlineFrequency: { value: 100.0 },
        interferenceStrength: { value: 0.1 }
      },
      transparent: true,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending
    });
    
    hologramShaderRef.current = shaderMaterial;
    
    // Create projection volume based on type
    const projection = createProjectionVolume(projectionType);
    setActiveProjection(projection);
    
    dispatch(updateHolographicDisplay({
      projectionType,
      active: true,
      hologramCount: 0
    }));
  }, [projectionType, dispatch]);

  const initializeSpatialComputing = useCallback(async () => {
    // Initialize spatial computing capabilities
    try {
      // Check for WebXR spatial tracking
      if ('xr' in navigator && navigator.xr) {
        const session = await navigator.xr.requestSession('immersive-ar', {
          requiredFeatures: ['local-floor', 'hit-test'],
          optionalFeatures: ['plane-detection', 'mesh-detection']
        });
        
        // Set up spatial tracking
        session.addEventListener('select', handleSpatialInteraction);
        
        dispatch(updateSpatialComputing({
          enabled: true,
          trackingActive: true,
          anchors: [],
          meshDetection: true,
          planeDetection: true
        }));
      }
    } catch (error) {
      console.warn('Spatial computing not available:', error);
    }
  }, [dispatch]);

  const initializeMixedReality = useCallback(() => {
    // Initialize mixed reality features
    if (interactionMode === 'eye_tracking') {
      initializeEyeTracking();
    }
    
    // Set up spatial mesh detection
    detectSpatialMesh();
  }, [interactionMode]);

  const initializeEyeTracking = useCallback(() => {
    // Placeholder for eye tracking initialization
    // This would integrate with WebXR eye tracking or external libraries
    console.log('Eye tracking initialized');
  }, []);

  const detectSpatialMesh = useCallback(() => {
    // Simulate spatial mesh detection
    // In real implementation, this would use WebXR mesh detection
    const meshes: THREE.Mesh[] = [];
    
    // Create some example spatial surfaces
    for (let i = 0; i < 5; i++) {
      const geometry = new THREE.PlaneGeometry(2, 2);
      const material = new THREE.MeshBasicMaterial({
        color: 0x888888,
        transparent: true,
        opacity: 0.3,
        wireframe: true
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(
        (Math.random() - 0.5) * 10,
        Math.random() * 2,
        (Math.random() - 0.5) * 10
      );
      mesh.rotation.x = -Math.PI / 2;
      
      meshes.push(mesh);
    }
    
    setSpatialMesh(meshes);
  }, []);

  const createProjectionVolume = (type: string): HolographicProjection => {
    const volume = {
      id: `projection_${Date.now()}`,
      type: type as any,
      bounds: new THREE.Box3(
        new THREE.Vector3(-2, -2, -2),
        new THREE.Vector3(2, 2, 2)
      ),
      resolution: { width: 1024, height: 1024, depth: 1024 },
      active: true
    };
    
    return volume;
  };

  const createGeologicalHolograms = useCallback(() => {
    const newHolograms = new Map<string, Hologram>();
    
    geologicalData.forEach((data, index) => {
      const hologram: Hologram = {
        id: `geo_hologram_${index}`,
        type: 'geological_model',
        position: new THREE.Vector3(
          data.x || (Math.random() - 0.5) * 4,
          data.y || (Math.random() - 0.5) * 4,
          data.z || (Math.random() - 0.5) * 4
        ),
        rotation: new THREE.Euler(0, 0, 0),
        scale: new THREE.Vector3(hologramScale, hologramScale, hologramScale),
        data: data,
        opacity: 0.8,
        interactive: true,
        animation: {
          type: 'rotate',
          duration: 10000,
          loop: true,
          parameters: { axis: 'y', speed: 0.5 }
        }
      };
      
      newHolograms.set(hologram.id, hologram);
    });
    
    setHolograms(newHolograms);
    
    dispatch(updateHolographicDisplay({
      hologramCount: newHolograms.size
    }));
  }, [geologicalData, hologramScale, dispatch]);

  const handleSpatialInteraction = useCallback((event: any) => {
    // Handle spatial interaction events
    const hitTestResults = event.frame.getHitTestResults(event.inputSource);
    
    if (hitTestResults.length > 0) {
      const hit = hitTestResults[0];
      const pose = hit.getPose(event.frame.session.referenceSpace);
      
      if (pose) {
        const position = new THREE.Vector3(
          pose.transform.position.x,
          pose.transform.position.y,
          pose.transform.position.z
        );
        
        // Create spatial anchor at hit location
        createSpatialAnchor(position);
      }
    }
  }, []);

  const createSpatialAnchor = useCallback((position: THREE.Vector3) => {
    const anchor: SpatialAnchor = {
      id: `anchor_${Date.now()}`,
      position: position.clone(),
      rotation: new THREE.Euler(0, 0, 0),
      persistent: true,
      holograms: []
    };
    
    setSpatialAnchors(prev => new Map(prev.set(anchor.id, anchor)));
    
    // Create a hologram at the anchor
    const hologram: Hologram = {
      id: `anchor_hologram_${Date.now()}`,
      type: 'interface_element',
      position: position.clone(),
      rotation: new THREE.Euler(0, 0, 0),
      scale: new THREE.Vector3(0.5, 0.5, 0.5),
      data: { type: 'spatial_marker' },
      opacity: 0.9,
      interactive: true
    };
    
    setHolograms(prev => new Map(prev.set(hologram.id, hologram)));
    anchor.holograms.push(hologram.id);
    
    dispatch(addHologram(hologram));
  }, [dispatch]);

  const updateHologramAnimation = useCallback((hologram: Hologram, deltaTime: number) => {
    if (!hologram.animation) return hologram;
    
    const animation = hologram.animation;
    const progress = (Date.now() % animation.duration) / animation.duration;
    
    switch (animation.type) {
      case 'rotate':
        const axis = animation.parameters.axis || 'y';
        const speed = animation.parameters.speed || 1;
        if (axis === 'x') hologram.rotation.x = progress * Math.PI * 2 * speed;
        else if (axis === 'y') hologram.rotation.y = progress * Math.PI * 2 * speed;
        else if (axis === 'z') hologram.rotation.z = progress * Math.PI * 2 * speed;
        break;
        
      case 'pulse':
        const pulseScale = 1 + Math.sin(progress * Math.PI * 2) * 0.2;
        hologram.scale.setScalar(pulseScale * hologramScale);
        break;
        
      case 'float':
        const floatHeight = Math.sin(progress * Math.PI * 2) * 0.5;
        hologram.position.y += floatHeight * deltaTime;
        break;
    }
    
    return hologram;
  }, [hologramScale]);

  return (
    <div className="holographic-display">
      <div className="holographic-controls">
        <h3>Holographic Display System</h3>
        <div className="display-info">
          <p>Projection: {projectionType}</p>
          <p>Holograms: {holograms.size}</p>
          <p>Anchors: {spatialAnchors.size}</p>
          <p>Spatial Computing: {enableSpatialComputing ? 'Active' : 'Inactive'}</p>
          <p>Mixed Reality: {enableMixedReality ? 'Active' : 'Inactive'}</p>
        </div>
        
        <div className="projection-controls">
          <button onClick={() => setActiveProjection(createProjectionVolume('volumetric'))}>
            Volumetric
          </button>
          <button onClick={() => setActiveProjection(createProjectionVolume('planar'))}>
            Planar
          </button>
          <button onClick={() => setActiveProjection(createProjectionVolume('cylindrical'))}>
            Cylindrical
          </button>
          <button onClick={() => setActiveProjection(createProjectionVolume('spherical'))}>
            Spherical
          </button>
        </div>
      </div>

      <Canvas style={{ height: '100vh' }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        
        <HolographicScene
          holograms={holograms}
          spatialAnchors={spatialAnchors}
          spatialMesh={spatialMesh}
          activeProjection={activeProjection}
          hologramShader={hologramShaderRef.current}
          onHologramUpdate={updateHologramAnimation}
          projectionType={projectionType}
        />
        
        <HolographicInterface
          holograms={holograms}
          spatialAnchors={spatialAnchors}
          interactionMode={interactionMode}
        />
      </Canvas>
    </div>
  );
};

interface HolographicSceneProps {
  holograms: Map<string, Hologram>;
  spatialAnchors: Map<string, SpatialAnchor>;
  spatialMesh: THREE.Mesh[];
  activeProjection: HolographicProjection | null;
  hologramShader: THREE.ShaderMaterial | null;
  onHologramUpdate: (hologram: Hologram, deltaTime: number) => Hologram;
  projectionType: string;
}

const HolographicScene: React.FC<HolographicSceneProps> = ({
  holograms,
  spatialAnchors,
  spatialMesh,
  activeProjection,
  hologramShader,
  onHologramUpdate,
  projectionType
}) => {
  const { clock } = useThree();
  const lastTimeRef = useRef(0);

  useFrame(() => {
    const currentTime = clock.getElapsedTime();
    const deltaTime = currentTime - lastTimeRef.current;
    lastTimeRef.current = currentTime;
    
    // Update hologram shader time uniform
    if (hologramShader) {
      hologramShader.uniforms.time.value = currentTime;
    }
    
    // Update hologram animations
    holograms.forEach((hologram) => {
      onHologramUpdate(hologram, deltaTime);
    });
  });

  return (
    <>
      {/* Render projection volume */}
      {activeProjection && (
        <ProjectionVolume
          projection={activeProjection}
          projectionType={projectionType}
        />
      )}
      
      {/* Render spatial mesh */}
      {spatialMesh.map((mesh, index) => (
        <primitive key={`mesh_${index}`} object={mesh} />
      ))}
      
      {/* Render spatial anchors */}
      {Array.from(spatialAnchors.values()).map(anchor => (
        <SpatialAnchorVisualization
          key={anchor.id}
          anchor={anchor}
        />
      ))}
      
      {/* Render holograms */}
      {Array.from(holograms.values()).map(hologram => (
        <HologramVisualization
          key={hologram.id}
          hologram={hologram}
          shader={hologramShader}
        />
      ))}
    </>
  );
};

const ProjectionVolume: React.FC<{
  projection: HolographicProjection;
  projectionType: string;
}> = ({ projection, projectionType }) => {
  const size = new THREE.Vector3(
    projection.bounds.max.x - projection.bounds.min.x,
    projection.bounds.max.y - projection.bounds.min.y,
    projection.bounds.max.z - projection.bounds.min.z
  );
  const center = new THREE.Vector3(
    (projection.bounds.min.x + projection.bounds.max.x) / 2,
    (projection.bounds.min.y + projection.bounds.max.y) / 2,
    (projection.bounds.min.z + projection.bounds.max.z) / 2
  );

  const getProjectionGeometry = () => {
    switch (projectionType) {
      case 'volumetric':
        return <boxGeometry args={[size.x, size.y, size.z]} />;
      case 'planar':
        return <planeGeometry args={[size.x, size.y]} />;
      case 'cylindrical':
        return <cylinderGeometry args={[size.x / 2, size.x / 2, size.y, 32]} />;
      case 'spherical':
        return <sphereGeometry args={[size.x / 2, 32, 32]} />;
      default:
        return <boxGeometry args={[size.x, size.y, size.z]} />;
    }
  };

  return (
    <mesh position={center}>
      {getProjectionGeometry()}
      <meshBasicMaterial
        color={0x00FFFF}
        transparent
        opacity={0.1}
        wireframe
      />
    </mesh>
  );
};

const SpatialAnchorVisualization: React.FC<{
  anchor: SpatialAnchor;
}> = ({ anchor }) => {
  return (
    <group position={anchor.position} rotation={anchor.rotation}>
      <Sphere args={[0.05, 8, 8]}>
        <meshStandardMaterial
          color={0xFF00FF}
          emissive={0x440044}
        />
      </Sphere>
      
      <Text
        position={[0, 0.15, 0]}
        fontSize={0.03}
        color="magenta"
        anchorX="center"
        anchorY="middle"
      >
        ANCHOR
      </Text>
      
      {/* Anchor coordinate system */}
      <Line
        points={[[0, 0, 0], [0.2, 0, 0]]}
        color="red"
        lineWidth={2}
      />
      <Line
        points={[[0, 0, 0], [0, 0.2, 0]]}
        color="green"
        lineWidth={2}
      />
      <Line
        points={[[0, 0, 0], [0, 0, 0.2]]}
        color="blue"
        lineWidth={2}
      />
    </group>
  );
};

const HologramVisualization: React.FC<{
  hologram: Hologram;
  shader: THREE.ShaderMaterial | null;
}> = ({ hologram, shader }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current && hologram.animation) {
      // Apply animation transformations
      meshRef.current.position.copy(hologram.position);
      meshRef.current.rotation.copy(hologram.rotation);
      meshRef.current.scale.copy(hologram.scale);
    }
  });

  const getHologramGeometry = () => {
    switch (hologram.type) {
      case 'geological_model':
        return getGeologicalModelGeometry(hologram.data);
      case 'data_visualization':
        return <sphereGeometry args={[0.5, 16, 16]} />;
      case 'interface_element':
        return <boxGeometry args={[0.3, 0.3, 0.05]} />;
      case 'annotation':
        return <cylinderGeometry args={[0.1, 0.1, 0.5, 8]} />;
      default:
        return <sphereGeometry args={[0.3, 8, 8]} />;
    }
  };

  const getGeologicalModelGeometry = (data: any) => {
    switch (data.type) {
      case 'rock_sample':
        return <boxGeometry args={[0.4, 0.4, 0.4]} />;
      case 'fault_plane':
        return <planeGeometry args={[1, 2]} />;
      case 'mineral_vein':
        return <cylinderGeometry args={[0.05, 0.05, 1, 8]} />;
      case 'volcanic_vent':
        return <coneGeometry args={[0.3, 0.8, 8]} />;
      default:
        return <sphereGeometry args={[0.3, 8, 8]} />;
    }
  };

  const getHologramMaterial = () => {
    if (shader) {
      const material = shader.clone();
      material.uniforms.opacity.value = hologram.opacity;
      material.uniforms.color.value = getHologramColor(hologram);
      return material;
    }
    
    return (
      <meshStandardMaterial
        color={getHologramColor(hologram)}
        transparent
        opacity={hologram.opacity}
        emissive={getHologramColor(hologram)}
        emissiveIntensity={0.2}
      />
    );
  };

  const getHologramColor = (hologram: Hologram): THREE.Color => {
    switch (hologram.type) {
      case 'geological_model':
        return new THREE.Color(0x00FFFF);
      case 'data_visualization':
        return new THREE.Color(0x00FF00);
      case 'interface_element':
        return new THREE.Color(0xFFFF00);
      case 'annotation':
        return new THREE.Color(0xFF00FF);
      default:
        return new THREE.Color(0xFFFFFF);
    }
  };

  return (
    <group>
      <mesh
        ref={meshRef}
        position={hologram.position}
        rotation={hologram.rotation}
        scale={hologram.scale}
      >
        {getHologramGeometry()}
        <meshStandardMaterial {...(getHologramMaterial() as any)} />
      </mesh>
      
      {hologram.type === 'geological_model' && (
        <Text
          position={[
            hologram.position.x,
            hologram.position.y + 0.5,
            hologram.position.z
          ]}
          fontSize={0.05}
          color="cyan"
          anchorX="center"
          anchorY="middle"
        >
          {hologram.data.name || 'Geological Feature'}
        </Text>
      )}
    </group>
  );
};

const HolographicInterface: React.FC<{
  holograms: Map<string, Hologram>;
  spatialAnchors: Map<string, SpatialAnchor>;
  interactionMode: string;
}> = ({ holograms, spatialAnchors, interactionMode }) => {
  return (
    <group position={[-3, 2, -1]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.1}
        color="cyan"
        anchorX="center"
        anchorY="middle"
      >
        Holographic Display
      </Text>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.06}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Holograms: {holograms.size}
      </Text>
      
      <Text
        position={[0, 0, 0]}
        fontSize={0.06}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Anchors: {spatialAnchors.size}
      </Text>
      
      <Text
        position={[0, -0.2, 0]}
        fontSize={0.06}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        Mode: {interactionMode}
      </Text>
    </group>
  );
};

export default HolographicDisplay;