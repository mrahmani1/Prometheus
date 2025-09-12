/**
 * Immersive Scene Component for Geological VR/AR Visualization
 * 
 * This component renders the 3D geological scene with interactive elements,
 * haptic feedback, collaborative features, and accessibility support.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { 
  useXR, 
  useController, 
  useHitTest, 
  Interactive,
  useXREvent
} from '@react-three/xr';
import {
  Text,
  Html,
  Sphere,
  Box,
  Plane,
  Line,
  Points,
  useGLTF,
  useTexture,
  Instances,
  Instance,
  Trail,
  Billboard,
  Float
} from '@react-three/drei';
import * as THREE from 'three';

// Import types from the main component
interface ImmersiveSceneProps {
  config: any;
  scene: any;
  onGesture: (gesture: any) => void;
  onHapticFeedback: (feedback: any) => void;
  collaborativeSession: any;
}

interface GeologicalLayer {
  id: string;
  name: string;
  type: 'seismic' | 'thermal' | 'deformation' | 'chemical' | 'magnetic';
  data: Float32Array;
  colorMap: THREE.DataTexture;
  opacity: number;
  visible: boolean;
  animated: boolean;
  timeStep: number;
}

interface InteractiveElement {
  id: string;
  position: THREE.Vector3;
  type: 'sample' | 'measurement' | 'annotation' | 'portal';
  content: any;
  hapticProperties?: any;
}

interface CollaborativeAvatar {
  id: string;
  name: string;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  headPosition: THREE.Vector3;
  handPositions: [THREE.Vector3, THREE.Vector3];
  isActive: boolean;
  color: string;
}

export const ImmersiveScene: React.FC<ImmersiveSceneProps> = ({
  config,
  scene,
  onGesture,
  onHapticFeedback,
  collaborativeSession
}) => {
  const { gl, camera, scene: threeScene } = useThree();
  const { isPresenting, player } = useXR();
  
  const [geologicalLayers, setGeologicalLayers] = useState<GeologicalLayer[]>([]);
  const [interactiveElements, setInteractiveElements] = useState<InteractiveElement[]>([]);
  const [collaborativeAvatars, setCollaborativeAvatars] = useState<CollaborativeAvatar[]>([]);
  const [selectedObject, setSelectedObject] = useState<THREE.Object3D | null>(null);
  const [measurementMode, setMeasurementMode] = useState(false);
  const [measurementPoints, setMeasurementPoints] = useState<THREE.Vector3[]>([]);

  const sceneRef = useRef<THREE.Group>(null);
  const geologicalModelRef = useRef<THREE.Group>(null);
  const layersRef = useRef<THREE.Group>(null);
  const interactionsRef = useRef<THREE.Group>(null);
  const collaborativeRef = useRef<THREE.Group>(null);

  // Hand tracking and gesture recognition
  const leftController = useController('left');
  const rightController = useController('right');
  const hitTest = useHitTest();

  // Initialize geological scene
  useEffect(() => {
    if (scene) {
      initializeGeologicalScene();
    }
  }, [scene]);

  // Initialize collaborative features
  useEffect(() => {
    if (config.collaborativeMode && collaborativeSession) {
      initializeCollaborativeFeatures();
    }
  }, [config.collaborativeMode, collaborativeSession]);

  const initializeGeologicalScene = useCallback(() => {
    if (!scene) return;

    // Load geological models
    scene.models?.forEach((model: any) => {
      loadGeologicalModel(model);
    });

    // Initialize data layers
    scene.dataLayers?.forEach((layer: any) => {
      createDataLayer(layer);
    });

    // Create interaction points
    scene.interactionPoints?.forEach((point: any) => {
      createInteractionPoint(point);
    });
  }, [scene]);

  const initializeCollaborativeFeatures = useCallback(() => {
    if (!collaborativeSession) return;

    // Initialize avatars for other participants
    collaborativeSession.participants?.forEach((participant: any) => {
      createCollaborativeAvatar(participant);
    });

    // Set up real-time synchronization
    setupCollaborativeSync();
  }, [collaborativeSession]);

  const loadGeologicalModel = useCallback((modelData: any) => {
    // Create geological model from data
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(modelData.vertices, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(modelData.normals, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(modelData.uvs, 2));

    // Apply geological material properties
    const material = createGeologicalMaterial(modelData.metadata);
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(modelData.position);
    mesh.rotation.copy(modelData.rotation);
    mesh.scale.copy(modelData.scale);
    
    // Add haptic properties
    mesh.userData = {
      ...modelData.metadata,
      hapticProperties: modelData.hapticProperties,
      interactive: true
    };

    if (geologicalModelRef.current) {
      geologicalModelRef.current.add(mesh);
    }
  }, []);

  const createGeologicalMaterial = useCallback((metadata: any) => {
    // Create material based on geological properties
    const material = new THREE.MeshPhysicalMaterial({
      color: getGeologicalColor(metadata.rockType),
      roughness: getGeologicalRoughness(metadata.rockType),
      metalness: getGeologicalMetalness(metadata.composition),
      clearcoat: metadata.weathered ? 0.1 : 0.0,
      transmission: metadata.porosity || 0.0
    });

    // Add texture based on rock type
    if (metadata.rockType) {
      const texture = generateGeologicalTexture(metadata.rockType);
      material.map = texture;
      material.normalMap = generateNormalMap(metadata.rockType);
    }

    return material;
  }, []);

  const createDataLayer = useCallback((layerData: any) => {
    const layer: GeologicalLayer = {
      id: layerData.id,
      name: layerData.name,
      type: layerData.type,
      data: layerData.data,
      colorMap: createColorMap(layerData.colorMap),
      opacity: layerData.opacity,
      visible: layerData.visible,
      animated: layerData.animated,
      timeStep: 0
    };

    // Create 3D visualization of the data layer
    const layerMesh = createLayerVisualization(layer);
    
    if (layersRef.current) {
      layersRef.current.add(layerMesh);
    }

    setGeologicalLayers(prev => [...prev, layer]);
  }, []);

  const createLayerVisualization = useCallback((layer: GeologicalLayer) => {
    switch (layer.type) {
      case 'seismic':
        return createSeismicVisualization(layer);
      case 'thermal':
        return createThermalVisualization(layer);
      case 'deformation':
        return createDeformationVisualization(layer);
      case 'chemical':
        return createChemicalVisualization(layer);
      case 'magnetic':
        return createMagneticVisualization(layer);
      default:
        return new THREE.Group();
    }
  }, []);

  const createSeismicVisualization = useCallback((layer: GeologicalLayer) => {
    const group = new THREE.Group();
    
    // Create seismic wave visualization
    const waveGeometry = new THREE.SphereGeometry(0.1, 8, 8);
    const waveMaterial = new THREE.MeshBasicMaterial({
      color: 0xff4444,
      transparent: true,
      opacity: 0.6
    });

    // Add seismic events as spheres
    for (let i = 0; i < layer.data.length; i += 4) {
      const x = layer.data[i];
      const y = layer.data[i + 1];
      const z = layer.data[i + 2];
      const magnitude = layer.data[i + 3];

      const sphere = new THREE.Mesh(waveGeometry, waveMaterial.clone());
      sphere.position.set(x, y, z);
      sphere.scale.setScalar(magnitude);
      
      group.add(sphere);
    }

    return group;
  }, []);

  const createThermalVisualization = useCallback((layer: GeologicalLayer) => {
    const group = new THREE.Group();
    
    // Create thermal gradient visualization
    const geometry = new THREE.PlaneGeometry(10, 10, 32, 32);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        thermalData: { value: layer.colorMap },
        opacity: { value: layer.opacity }
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform sampler2D thermalData;
        uniform float opacity;
        varying vec2 vUv;
        
        void main() {
          vec4 thermal = texture2D(thermalData, vUv);
          vec3 color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), thermal.r);
          gl_FragColor = vec4(color, opacity);
        }
      `,
      transparent: true
    });

    const mesh = new THREE.Mesh(geometry, material);
    group.add(mesh);

    return group;
  }, []);

  const createDeformationVisualization = useCallback((layer: GeologicalLayer) => {
    const group = new THREE.Group();
    
    // Create deformation vectors
    const arrowGeometry = new THREE.ConeGeometry(0.05, 0.2, 8);
    const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });

    for (let i = 0; i < layer.data.length; i += 6) {
      const x = layer.data[i];
      const y = layer.data[i + 1];
      const z = layer.data[i + 2];
      const dx = layer.data[i + 3];
      const dy = layer.data[i + 4];
      const dz = layer.data[i + 5];

      const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
      arrow.position.set(x, y, z);
      
      const direction = new THREE.Vector3(dx, dy, dz).normalize();
      arrow.lookAt(direction);
      
      group.add(arrow);
    }

    return group;
  }, []);

  const createChemicalVisualization = useCallback((layer: GeologicalLayer) => {
    const group = new THREE.Group();
    
    // Create chemical composition visualization as colored particles
    const particleGeometry = new THREE.SphereGeometry(0.02, 6, 6);
    
    for (let i = 0; i < layer.data.length; i += 4) {
      const x = layer.data[i];
      const y = layer.data[i + 1];
      const z = layer.data[i + 2];
      const composition = layer.data[i + 3];

      const color = getChemicalColor(composition);
      const material = new THREE.MeshBasicMaterial({ color });
      
      const particle = new THREE.Mesh(particleGeometry, material);
      particle.position.set(x, y, z);
      
      group.add(particle);
    }

    return group;
  }, []);

  const createMagneticVisualization = useCallback((layer: GeologicalLayer) => {
    const group = new THREE.Group();
    
    // Create magnetic field lines
    const lineGeometry = new THREE.BufferGeometry();
    const positions = [];
    
    for (let i = 0; i < layer.data.length; i += 6) {
      positions.push(layer.data[i], layer.data[i + 1], layer.data[i + 2]);
      positions.push(layer.data[i + 3], layer.data[i + 4], layer.data[i + 5]);
    }
    
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    
    const lineMaterial = new THREE.LineBasicMaterial({ 
      color: 0xff00ff,
      transparent: true,
      opacity: 0.7
    });
    
    const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    group.add(lines);

    return group;
  }, []);

  const createInteractionPoint = useCallback((pointData: any) => {
    const element: InteractiveElement = {
      id: pointData.id,
      position: pointData.position,
      type: pointData.type,
      content: pointData.content,
      hapticProperties: pointData.hapticFeedback
    };

    setInteractiveElements(prev => [...prev, element]);
  }, []);

  const createCollaborativeAvatar = useCallback((participant: any) => {
    const avatar: CollaborativeAvatar = {
      id: participant.id,
      name: participant.name,
      position: new THREE.Vector3(),
      rotation: new THREE.Euler(),
      headPosition: new THREE.Vector3(),
      handPositions: [new THREE.Vector3(), new THREE.Vector3()],
      isActive: participant.isActive,
      color: getParticipantColor(participant.id)
    };

    setCollaborativeAvatars(prev => [...prev, avatar]);
  }, []);

  const setupCollaborativeSync = useCallback(() => {
    // Set up WebSocket or WebRTC for real-time collaboration
    // This would integrate with the collaborative backend service
  }, []);

  // Handle controller interactions
  useXREvent('selectstart', (event) => {
    if (event.target === leftController || event.target === rightController) {
      handleControllerSelect(event);
    }
  });

  useXREvent('selectend', (event) => {
    if (event.target === leftController || event.target === rightController) {
      handleControllerRelease(event);
    }
  });

  const handleControllerSelect = useCallback((event: any) => {
    const controller = event.target;
    const raycaster = new THREE.Raycaster();
    
    // Cast ray from controller
    raycaster.setFromXRController(controller);
    
    // Check intersections with geological objects
    const intersects = raycaster.intersectObjects(
      geologicalModelRef.current?.children || [], 
      true
    );

    if (intersects.length > 0) {
      const object = intersects[0].object;
      handleObjectInteraction(object, intersects[0].point);
    }
  }, []);

  const handleControllerRelease = useCallback((event: any) => {
    // Handle controller release
    setSelectedObject(null);
  }, []);

  const handleObjectInteraction = useCallback((object: THREE.Object3D, point: THREE.Vector3) => {
    setSelectedObject(object);
    
    // Trigger haptic feedback
    if (config.hapticEnabled && object.userData.hapticProperties) {
      const hapticFeedback = {
        intensity: object.userData.hapticProperties.hardness || 0.5,
        duration: 200,
        pattern: 'pulse' as const
      };
      onHapticFeedback(hapticFeedback);
    }

    // Handle measurement mode
    if (measurementMode) {
      setMeasurementPoints(prev => [...prev, point]);
    }

    // Show information panel
    showObjectInformation(object);
  }, [config.hapticEnabled, measurementMode, onHapticFeedback]);

  const showObjectInformation = useCallback((object: THREE.Object3D) => {
    // Display geological information about the selected object
    const metadata = object.userData;
    console.log('Geological Object Info:', metadata);
  }, []);

  // Animation loop
  useFrame((state, delta) => {
    // Animate geological layers
    geologicalLayers.forEach((layer, index) => {
      if (layer.animated) {
        layer.timeStep += delta;
        // Update layer visualization based on time
        updateLayerAnimation(layer, layer.timeStep);
      }
    });

    // Update collaborative avatars
    updateCollaborativeAvatars(delta);

    // Update gesture recognition
    if (config.gestureEnabled) {
      updateGestureRecognition();
    }
  });

  const updateLayerAnimation = useCallback((layer: GeologicalLayer, time: number) => {
    // Update animated geological layers
    switch (layer.type) {
      case 'seismic':
        updateSeismicAnimation(layer, time);
        break;
      case 'thermal':
        updateThermalAnimation(layer, time);
        break;
      // Add other layer types...
    }
  }, []);

  const updateSeismicAnimation = useCallback((layer: GeologicalLayer, time: number) => {
    // Animate seismic waves propagation
  }, []);

  const updateThermalAnimation = useCallback((layer: GeologicalLayer, time: number) => {
    // Animate thermal changes over time
  }, []);

  const updateCollaborativeAvatars = useCallback((delta: number) => {
    // Update positions and animations of collaborative avatars
    collaborativeAvatars.forEach(avatar => {
      if (avatar.isActive) {
        // Smooth avatar movement
        // This would be updated from collaborative session data
      }
    });
  }, [collaborativeAvatars]);

  const updateGestureRecognition = useCallback(() => {
    // Implement gesture recognition logic
    if (leftController && rightController) {
      const leftPos = leftController.position;
      const rightPos = rightController.position;
      
      // Detect pinch gesture
      const distance = leftPos.distanceTo(rightPos);
      if (distance < 0.1) {
        onGesture({
          type: 'pinch',
          confidence: 0.9,
          position: leftPos.clone().lerp(rightPos, 0.5)
        });
      }
    }
  }, [leftController, rightController, onGesture]);

  // Utility functions
  const getGeologicalColor = (rockType: string): number => {
    const colors: Record<string, number> = {
      'granite': 0xcccccc,
      'basalt': 0x333333,
      'limestone': 0xf0f0f0,
      'sandstone': 0xdeb887,
      'shale': 0x696969,
      'quartzite': 0xffffff,
      'marble': 0xf8f8ff,
      'gneiss': 0x808080
    };
    return colors[rockType] || 0x888888;
  };

  const getGeologicalRoughness = (rockType: string): number => {
    const roughness: Record<string, number> = {
      'granite': 0.8,
      'basalt': 0.9,
      'limestone': 0.6,
      'sandstone': 0.7,
      'shale': 0.5,
      'quartzite': 0.3,
      'marble': 0.2,
      'gneiss': 0.7
    };
    return roughness[rockType] || 0.5;
  };

  const getGeologicalMetalness = (composition: string[]): number => {
    const metallic = ['iron', 'copper', 'gold', 'silver', 'platinum'];
    const hasMetals = composition.some(comp => metallic.includes(comp.toLowerCase()));
    return hasMetals ? 0.8 : 0.1;
  };

  const generateGeologicalTexture = (rockType: string): THREE.Texture => {
    // Generate procedural texture based on rock type
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    
    // Generate texture pattern based on rock type
    // This is a simplified example - real implementation would be more complex
    ctx.fillStyle = '#888888';
    ctx.fillRect(0, 0, 512, 512);
    
    return new THREE.CanvasTexture(canvas);
  };

  const generateNormalMap = (rockType: string): THREE.Texture => {
    // Generate normal map for surface detail
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, 512, 512);
    
    return new THREE.CanvasTexture(canvas);
  };

  const createColorMap = (colorMapName: string): THREE.DataTexture => {
    // Create color map for data visualization
    const width = 256;
    const data = new Uint8Array(width * 4);
    
    for (let i = 0; i < width; i++) {
      const t = i / (width - 1);
      const r = Math.floor(255 * t);
      const g = Math.floor(255 * (1 - t));
      const b = 0;
      const a = 255;
      
      data[i * 4] = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = a;
    }
    
    return new THREE.DataTexture(data, width, 1, THREE.RGBAFormat);
  };

  const getChemicalColor = (composition: number): number => {
    // Map chemical composition to color
    const hue = composition * 360;
    return new THREE.Color().setHSL(hue / 360, 0.8, 0.5).getHex();
  };

  const getParticipantColor = (participantId: string): string => {
    // Generate consistent color for participant
    const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44', '#ff44ff', '#44ffff'];
    const hash = participantId.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  };

  return (
    <group ref={sceneRef}>
      {/* Environment */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <pointLight position={[-10, -10, -5]} intensity={0.5} />

      {/* Geological Models */}
      <group ref={geologicalModelRef} />

      {/* Data Layers */}
      <group ref={layersRef} />

      {/* Interactive Elements */}
      <group ref={interactionsRef}>
        {interactiveElements.map((element) => (
          <InteractivePoint
            key={element.id}
            element={element}
            config={config}
            onInteraction={handleObjectInteraction}
          />
        ))}
      </group>

      {/* Collaborative Avatars */}
      <group ref={collaborativeRef}>
        {collaborativeAvatars.map((avatar) => (
          <CollaborativeAvatar
            key={avatar.id}
            avatar={avatar}
            config={config}
          />
        ))}
      </group>

      {/* Measurement Tools */}
      {measurementPoints.length > 0 && (
        <MeasurementVisualization points={measurementPoints} />
      )}

      {/* Accessibility Features */}
      {config.accessibilityMode && (
        <AccessibilityOverlay />
      )}

      {/* Educational Features */}
      {config.educationMode && (
        <EducationalOverlay />
      )}
    </group>
  );
};

// Interactive Point Component
const InteractivePoint: React.FC<{
  element: InteractiveElement;
  config: any;
  onInteraction: (object: THREE.Object3D, point: THREE.Vector3) => void;
}> = ({ element, config, onInteraction }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  const getPointColor = () => {
    switch (element.type) {
      case 'sample': return '#ff4444';
      case 'measurement': return '#44ff44';
      case 'annotation': return '#4444ff';
      case 'portal': return '#ffff44';
      default: return '#ffffff';
    }
  };

  return (
    <Interactive onSelect={() => onInteraction(meshRef.current!, element.position)}>
      <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
        <Sphere
          ref={meshRef}
          position={element.position}
          args={[0.1, 16, 16]}
        >
          <meshBasicMaterial color={getPointColor()} transparent opacity={0.8} />
        </Sphere>
        
        {config.accessibilityMode && (
          <Billboard position={[element.position.x, element.position.y + 0.2, element.position.z]}>
            <Text
              fontSize={0.1}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {element.type}
            </Text>
          </Billboard>
        )}
      </Float>
    </Interactive>
  );
};

// Collaborative Avatar Component
const CollaborativeAvatar: React.FC<{
  avatar: CollaborativeAvatar;
  config: any;
}> = ({ avatar, config }) => {
  return (
    <group position={avatar.position} rotation={avatar.rotation}>
      {/* Head */}
      <Sphere position={avatar.headPosition} args={[0.1, 16, 16]}>
        <meshBasicMaterial color={avatar.color} />
      </Sphere>
      
      {/* Hands */}
      <Sphere position={avatar.handPositions[0]} args={[0.05, 8, 8]}>
        <meshBasicMaterial color={avatar.color} />
      </Sphere>
      <Sphere position={avatar.handPositions[1]} args={[0.05, 8, 8]}>
        <meshBasicMaterial color={avatar.color} />
      </Sphere>
      
      {/* Name Label */}
      <Billboard position={[0, 0.3, 0]}>
        <Text
          fontSize={0.08}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          {avatar.name}
        </Text>
      </Billboard>
    </group>
  );
};

// Measurement Visualization Component
const MeasurementVisualization: React.FC<{
  points: THREE.Vector3[];
}> = ({ points }) => {
  if (points.length < 2) return null;

  const linePoints = points.map(p => [p.x, p.y, p.z]);
  const distance = points[0].distanceTo(points[points.length - 1]);

  return (
    <group>
      <Line points={linePoints} color="yellow" lineWidth={3} />
      
      {points.map((point, index) => (
        <Sphere key={index} position={point} args={[0.02, 8, 8]}>
          <meshBasicMaterial color="yellow" />
        </Sphere>
      ))}
      
      <Billboard position={points[points.length - 1]}>
        <Text
          fontSize={0.05}
          color="yellow"
          anchorX="center"
          anchorY="middle"
        >
          {distance.toFixed(2)}m
        </Text>
      </Billboard>
    </group>
  );
};

// Accessibility Overlay Component
const AccessibilityOverlay: React.FC = () => {
  return (
    <group>
      {/* High contrast grid */}
      <gridHelper args={[20, 20, '#ffffff', '#888888']} />
      
      {/* Audio cues and descriptions would be implemented here */}
    </group>
  );
};

// Educational Overlay Component
const EducationalOverlay: React.FC = () => {
  return (
    <group>
      {/* Educational information panels and guided tours */}
      <Billboard position={[0, 2, 0]}>
        <Text
          fontSize={0.1}
          color="white"
          anchorX="center"
          anchorY="middle"
        >
          Educational Mode Active
        </Text>
      </Billboard>
    </group>
  );
};

export default ImmersiveScene;