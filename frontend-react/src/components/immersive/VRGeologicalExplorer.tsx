import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { VRButton, XR, createXRStore } from '@react-three/xr';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

import { GeologicalLayer } from '../../types/immersive';

interface VRGeologicalExplorerProps {
  geologicalData: GeologicalLayer[];
  onInteraction?: (interaction: VRInteraction) => void;
  enableHaptics?: boolean;
  collaborativeMode?: boolean;
}

interface VRInteraction {
  type: 'select' | 'grab' | 'point' | 'gesture';
  target: string;
  position: THREE.Vector3;
  data?: any;
}

interface GeologicalModel extends THREE.Object3D {
  userData: {
    layerId: string;
    properties: Record<string, any>;
    interactive: boolean;
  };
}

const VRGeologicalExplorer: React.FC<VRGeologicalExplorerProps> = ({
  geologicalData,
  onInteraction,
  enableHaptics = true,
  collaborativeMode = false
}) => {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [vrSupported, setVRSupported] = useState(false);
  const hapticDeviceRef = useRef<any>(null);
  
  // Create XR store
  const xrStore = createXRStore();

  useEffect(() => {
    // Check VR support
    if ('xr' in navigator) {
      navigator.xr?.isSessionSupported('immersive-vr').then(setVRSupported);
    }

    // Initialize haptic feedback if supported
    if (enableHaptics && 'getGamepads' in navigator) {
      initializeHapticFeedback();
    }
  }, [enableHaptics]);

  const initializeHapticFeedback = useCallback(() => {
    const checkGamepads = () => {
      const gamepads = navigator.getGamepads();
      for (let i = 0; i < gamepads.length; i++) {
        const gamepad = gamepads[i];
        if (gamepad && gamepad.hapticActuators && gamepad.hapticActuators.length > 0) {
          hapticDeviceRef.current = gamepad;
          break;
        }
      }
    };

    window.addEventListener('gamepadconnected', checkGamepads);
    checkGamepads();

    return () => {
      window.removeEventListener('gamepadconnected', checkGamepads);
    };
  }, []);

  const triggerHapticFeedback = useCallback((intensity: number, duration: number) => {
    if (hapticDeviceRef.current && hapticDeviceRef.current.hapticActuators.length > 0) {
      hapticDeviceRef.current.hapticActuators[0].pulse(intensity, duration);
      
      console.log('Haptic feedback triggered:', { intensity, duration, timestamp: Date.now() });
    }
  }, []);

  const handleLayerInteraction = useCallback((layerId: string, position: THREE.Vector3, type: string) => {
    setSelectedLayer(layerId);
    
    // Trigger haptic feedback based on geological properties
    const layer = geologicalData.find(l => l.id === layerId);
    if (layer && enableHaptics) {
      const intensity = layer.properties.hardness ? layer.properties.hardness * 0.1 : 0.5;
      triggerHapticFeedback(intensity, 100);
    }

    const interaction: VRInteraction = {
      type: type as any,
      target: layerId,
      position,
      data: layer
    };

    onInteraction?.(interaction);
  }, [geologicalData, enableHaptics, triggerHapticFeedback, onInteraction]);

  return (
    <div className="vr-geological-explorer">
      <div className="vr-controls">
        {vrSupported && <VRButton store={xrStore} />}
        <div className="vr-info">
          <p>VR Support: {vrSupported ? 'Available' : 'Not Available'}</p>
          <p>Selected Layer: {selectedLayer || 'None'}</p>
          <p>Haptic Feedback: {enableHaptics ? 'Enabled' : 'Disabled'}</p>
        </div>
      </div>

      <Canvas style={{ height: '100vh' }}>
        <XR store={xrStore}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          
          <GeologicalScene
            geologicalData={geologicalData}
            onInteraction={handleLayerInteraction}
            selectedLayer={selectedLayer}
            collaborativeMode={collaborativeMode}
          />
          
          <VRInterface
            selectedLayer={selectedLayer}
            geologicalData={geologicalData}
          />
        </XR>
      </Canvas>
    </div>
  );
};

interface GeologicalSceneProps {
  geologicalData: GeologicalLayer[];
  onInteraction: (layerId: string, position: THREE.Vector3, type: string) => void;
  selectedLayer: string | null;
  collaborativeMode: boolean;
}

const GeologicalScene: React.FC<GeologicalSceneProps> = ({
  geologicalData,
  onInteraction,
  selectedLayer,
  collaborativeMode
}) => {
  const { scene } = useThree();
  const modelsRef = useRef<Map<string, GeologicalModel>>(new Map());

  useEffect(() => {
    // Create 3D geological models
    geologicalData.forEach(layer => {
      if (!modelsRef.current.has(layer.id)) {
        const model = createGeologicalModel(layer);
        modelsRef.current.set(layer.id, model);
        scene.add(model);
      }
    });

    // Cleanup removed layers
    modelsRef.current.forEach((model, layerId) => {
      if (!geologicalData.find(l => l.id === layerId)) {
        scene.remove(model);
        modelsRef.current.delete(layerId);
      }
    });
  }, [geologicalData, scene]);

  const createGeologicalModel = (layer: GeologicalLayer): GeologicalModel => {
    const group = new THREE.Group();
    
    // Set up userData to match GeologicalModel interface
    group.userData = {
      layerId: layer.id,
      properties: layer.properties,
      interactive: true
    };
    
    // Create geometry based on layer type
    let geometry: THREE.BufferGeometry;
    let material: THREE.Material;

    switch (layer.type) {
      case 'sedimentary':
        geometry = new THREE.BoxGeometry(
          layer.dimensions.width,
          layer.dimensions.height,
          layer.dimensions.depth
        );
        material = new THREE.MeshLambertMaterial({
          color: layer.properties.color || 0x8B4513,
          transparent: true,
          opacity: 0.8
        });
        break;
      
      case 'igneous':
        geometry = new THREE.SphereGeometry(layer.dimensions.radius || 1, 32, 32);
        material = new THREE.MeshPhongMaterial({
          color: layer.properties.color || 0xFF4500,
          transparent: true,
          opacity: 0.9
        });
        break;
      
      case 'metamorphic':
        geometry = new THREE.CylinderGeometry(
          layer.dimensions.topRadius || 1,
          layer.dimensions.bottomRadius || 1,
          layer.dimensions.height || 2,
          32
        );
        material = new THREE.MeshStandardMaterial({
          color: layer.properties.color || 0x696969,
          metalness: 0.3,
          roughness: 0.7
        });
        break;
      
      default:
        geometry = new THREE.BoxGeometry(1, 1, 1);
        material = new THREE.MeshBasicMaterial({ color: 0x888888 });
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(
      layer.position.x,
      layer.position.y,
      layer.position.z
    );

    // Add interaction capabilities
    mesh.userData = {
      layerId: layer.id,
      properties: layer.properties,
      interactive: true
    };

    group.add(mesh);

    // Add geological features
    if (layer.features) {
      layer.features.forEach(feature => {
        const featureMesh = createGeologicalFeature(feature);
        group.add(featureMesh);
      });
    }

    return group as unknown as GeologicalModel;
  };

  const createGeologicalFeature = (feature: any): THREE.Mesh => {
    let geometry: THREE.BufferGeometry;
    let material: THREE.Material;

    switch (feature.type) {
      case 'fault':
        geometry = new THREE.PlaneGeometry(feature.length, feature.height);
        material = new THREE.MeshBasicMaterial({
          color: 0xFF0000,
          transparent: true,
          opacity: 0.5,
          side: THREE.DoubleSide
        });
        break;
      
      case 'fold':
        const curve = new THREE.CatmullRomCurve3(feature.points.map(
          (p: any) => new THREE.Vector3(p.x, p.y, p.z)
        ));
        geometry = new THREE.TubeGeometry(curve, 64, 0.1, 8, false);
        material = new THREE.MeshBasicMaterial({ color: 0x00FF00 });
        break;
      
      case 'mineral_vein':
        geometry = new THREE.CylinderGeometry(0.05, 0.05, feature.length, 8);
        material = new THREE.MeshPhongMaterial({
          color: feature.mineralColor || 0xFFD700,
          emissive: 0x111111
        });
        break;
      
      default:
        geometry = new THREE.SphereGeometry(0.1, 8, 8);
        material = new THREE.MeshBasicMaterial({ color: 0xFFFFFF });
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(feature.position.x, feature.position.y, feature.position.z);
    
    return mesh;
  };

  return (
    <>
      {/* Geological models are added to scene in useEffect */}
      <VRInteractionHandler
        models={Array.from(modelsRef.current.values())}
        onInteraction={onInteraction}
        selectedLayer={selectedLayer}
      />
      
      {collaborativeMode && (
        <CollaborativeIndicators />
      )}
    </>
  );
};

interface VRInteractionHandlerProps {
  models: GeologicalModel[];
  onInteraction: (layerId: string, position: THREE.Vector3, type: string) => void;
  selectedLayer: string | null;
}

const VRInteractionHandler: React.FC<VRInteractionHandlerProps> = ({
  models,
  onInteraction,
  selectedLayer
}) => {
  const raycaster = useRef(new THREE.Raycaster());
  const { gl } = useThree();

  useFrame(() => {
    // Handle controller interactions
    const session = gl.xr.getSession();
    if (session) {
      const inputSources = session.inputSources;
      
      for (let i = 0; i < inputSources.length; i++) {
        const inputSource = inputSources[i];
        if (inputSource.gamepad) {
          handleControllerInput(inputSource, i);
        }
      }
    }
  });

  const handleControllerInput = (inputSource: any, controllerIndex: number) => {
    const gamepad = inputSource.gamepad;
    
    // Check for trigger press (selection)
    if (gamepad.buttons[0] && gamepad.buttons[0].pressed) {
      // Filter interactions based on selected layer
      const layerFilter = selectedLayer || null;
      console.log('Layer filter applied:', layerFilter);
      performRaycast(controllerIndex, 'select');
    }
    
    // Check for grip press (grab)
    if (gamepad.buttons[1] && gamepad.buttons[1].pressed) {
      performRaycast(controllerIndex, 'grab');
    }
  };

  const performRaycast = (controllerIndex: number, interactionType: string) => {
    // Get controller position and direction
    const controller = gl.xr.getController(controllerIndex);
    if (!controller) return;

    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);

    raycaster.current.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    raycaster.current.ray.direction.set(0, 0, -1).applyMatrix4(tempMatrix);

    const intersects = raycaster.current.intersectObjects(models, true);
    
    if (intersects.length > 0) {
      const intersect = intersects[0];
      const object = intersect.object;
      
      if (object.userData.interactive) {
        onInteraction(
          object.userData.layerId,
          intersect.point,
          interactionType
        );
      }
    }
  };

  return null;
};

const VRInterface: React.FC<{
  selectedLayer: string | null;
  geologicalData: GeologicalLayer[];
}> = ({ selectedLayer, geologicalData }) => {
  const selectedLayerData = geologicalData.find(l => l.id === selectedLayer);

  return (
    <group position={[-2, 1.5, -1]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Geological Explorer
      </Text>
      
      {selectedLayerData && (
        <group position={[0, 0, 0]}>
          <Text
            position={[0, 0.2, 0]}
            fontSize={0.15}
            color="yellow"
            anchorX="center"
            anchorY="middle"
          >
            {selectedLayerData.name}
          </Text>
          
          <Text
            position={[0, 0, 0]}
            fontSize={0.1}
            color="lightblue"
            anchorX="center"
            anchorY="middle"
          >
            Type: {selectedLayerData.type}
          </Text>
          
          <Text
            position={[0, -0.2, 0]}
            fontSize={0.1}
            color="lightgreen"
            anchorX="center"
            anchorY="middle"
          >
            Age: {selectedLayerData.properties.age || 'Unknown'}
          </Text>
        </group>
      )}
    </group>
  );
};

const CollaborativeIndicators: React.FC = () => {
  // Placeholder for collaborative user indicators
  return (
    <group>
      {/* Add collaborative user avatars and indicators */}
    </group>
  );
};

export default VRGeologicalExplorer;