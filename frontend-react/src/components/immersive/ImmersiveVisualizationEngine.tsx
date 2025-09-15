/**
 * Immersive Visualization Engine for VR/AR Geological Exploration
 * 
 * This component provides comprehensive VR/AR capabilities for geological data visualization
 * including virtual reality exploration, augmented reality field overlays, haptic feedback,
 * collaborative environments, and spatial computing features.
 * 
 * Features:
 * - WebXR-based VR/AR support
 * - Haptic feedback integration
 * - Collaborative multi-user environments
 * - Gesture and voice control
 * - Spatial computing and mixed reality
 * - Accessibility features
 * - Real-time geological data rendering
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { 
  VRButton, 
  ARButton, 
  XR
  // useXR, useController, useHitTest, Interactive - unused or not exported
} from '@react-three/xr';
// import {
//   OrbitControls, PerspectiveCamera, Environment - unused
// } from '@react-three/drei';
// import * as THREE from 'three'; // unused
// import { useDispatch, useSelector } from 'react-redux'; // unused
import { 
  Box as MuiBox,
  Typography,
  // Button, Slider - unused
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  Grid as MuiGrid,
  Alert,
  Chip,
  // IconButton, Tooltip - unused
} from '@mui/material';
import {
  // Headset as VrHeadset, CameraAlt, ThreeDRotation, School, Settings, Fullscreen - unused
  Mic,
  MicOff,
  Vibration,
  Accessibility,
  Group,
  // VolumeUp, VolumeOff - unused
} from '@mui/icons-material';

// Types for VR/AR system
interface ImmersiveConfig {
  vrEnabled: boolean;
  arEnabled: boolean;
  hapticEnabled: boolean;
  voiceEnabled: boolean;
  gestureEnabled: boolean;
  collaborativeMode: boolean;
  accessibilityMode: boolean;
  educationMode: boolean;
  spatialAudio: boolean;
  handTracking: boolean;
}

interface GeologicalVRScene {
  id: string;
  name: string;
  description: string;
  type: 'volcano' | 'seismic' | 'mineral' | 'tectonic' | 'educational';
  models: GeologicalModel3D[];
  dataLayers: DataLayer[];
  interactionPoints: InteractionPoint[];
  collaborativeSpaces: CollaborativeSpace[];
}

interface GeologicalModel3D {
  id: string;
  name: string;
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  scale: THREE.Vector3;
  metadata: GeologicalMetadata;
  hapticProperties: HapticProperties;
}

interface DataLayer {
  id: string;
  name: string;
  type: 'seismic' | 'thermal' | 'deformation' | 'chemical' | 'magnetic';
  data: Float32Array;
  colorMap: string;
  opacity: number;
  visible: boolean;
  animated: boolean;
  timeRange?: [number, number];
}

interface InteractionPoint {
  id: string;
  position: THREE.Vector3;
  type: 'info' | 'measurement' | 'sample' | 'annotation';
  content: string;
  metadata: any;
  hapticFeedback?: HapticFeedback;
}

interface CollaborativeSpace {
  id: string;
  name: string;
  participants: Participant[];
  sharedObjects: SharedObject[];
  voiceChat: boolean;
  spatialAudio: boolean;
}

interface Participant {
  id: string;
  name: string;
  avatar: THREE.Object3D;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  isActive: boolean;
  role: 'researcher' | 'student' | 'instructor' | 'observer';
}

interface SharedObject {
  id: string;
  object: THREE.Object3D;
  owner: string;
  permissions: string[];
  synchronized: boolean;
}

interface HapticProperties {
  texture: 'smooth' | 'rough' | 'crystalline' | 'porous' | 'layered';
  hardness: number; // 0-1
  temperature: number; // Celsius
  vibrationPattern?: number[];
}

interface HapticFeedback {
  intensity: number;
  duration: number;
  pattern: 'pulse' | 'continuous' | 'wave' | 'impact';
  frequency?: number;
}

interface GeologicalMetadata {
  rockType: string;
  age: number;
  composition: string[];
  formation: string;
  properties: Record<string, any>;
}

// Voice Recognition Interface
interface VoiceCommand {
  command: string;
  action: string;
  parameters?: Record<string, any>;
}

// Gesture Recognition Interface
interface GestureRecognition {
  type: 'pinch' | 'grab' | 'point' | 'swipe' | 'rotate' | 'scale';
  confidence: number;
  position: THREE.Vector3;
  direction?: THREE.Vector3;
}

// Main Immersive Visualization Engine Component
export const ImmersiveVisualizationEngine: React.FC = () => {
  const [config, setConfig] = useState<ImmersiveConfig>({
    vrEnabled: false,
    arEnabled: false,
    hapticEnabled: true,
    voiceEnabled: false,
    gestureEnabled: true,
    collaborativeMode: false,
    accessibilityMode: false,
    educationMode: false,
    spatialAudio: true,
    handTracking: true
  });

  const [currentScene, setCurrentScene] = useState<GeologicalVRScene | null>(null);
  const [isXRSupported, setIsXRSupported] = useState(false);
  const [hapticDevice, setHapticDevice] = useState<any>(null);
  const [voiceRecognition, setVoiceRecognition] = useState<any>(null);
  const [collaborativeSession, setCollaborativeSession] = useState<CollaborativeSpace | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const _sceneRef = useRef<any>(null);
  const _hapticRef = useRef<any>(null);

  // Initialize XR capabilities
  useEffect(() => {
    const checkXRSupport = async () => {
      if ('xr' in navigator) {
        try {
          const vrSupported = await (navigator as any).xr.isSessionSupported('immersive-vr');
          const arSupported = await (navigator as any).xr.isSessionSupported('immersive-ar');
          setIsXRSupported(vrSupported || arSupported);
        } catch (error) {
          console.warn('XR support check failed:', error);
          setIsXRSupported(false);
        }
      }
    };

    checkXRSupport();
  }, []);

  // Initialize haptic feedback
  useEffect(() => {
    const initializeHaptics = async () => {
      if ('navigator' in window && 'vibrate' in navigator) {
        setHapticDevice({
          vibrate: (pattern: number[]) => navigator.vibrate(pattern),
          supported: true
        });
      }

      // Check for advanced haptic devices (e.g., Ultraleap, Tanvas)
      try {
        // This would integrate with actual haptic SDKs
        const advancedHaptics = await initializeAdvancedHaptics();
        if (advancedHaptics) {
          setHapticDevice(advancedHaptics);
        }
      } catch (error) {
        console.warn('Advanced haptics not available:', error);
      }
    };

    if (config.hapticEnabled) {
      initializeHaptics();
    }
  }, [config.hapticEnabled]);

  // Initialize voice recognition
  useEffect(() => {
    const initializeVoiceRecognition = () => {
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event: any) => {
          const command = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
          handleVoiceCommand(command);
        };

        recognition.onerror = (event: any) => {
          console.warn('Voice recognition error:', event.error);
        };

        setVoiceRecognition(recognition);
      }
    };

    if (config.voiceEnabled) {
      initializeVoiceRecognition();
    }
  }, [config.voiceEnabled]);

  // Voice command handler
  const handleVoiceCommand = useCallback((command: string) => {
    const commands: VoiceCommand[] = [
      { command: 'show volcano', action: 'loadScene', parameters: { type: 'volcano' } },
      { command: 'hide layers', action: 'toggleLayers', parameters: { visible: false } },
      { command: 'zoom in', action: 'zoom', parameters: { factor: 2 } },
      { command: 'zoom out', action: 'zoom', parameters: { factor: 0.5 } },
      { command: 'rotate model', action: 'rotate', parameters: { axis: 'y', angle: 90 } },
      { command: 'measure distance', action: 'startMeasurement' },
      { command: 'take sample', action: 'takeSample' },
      { command: 'start collaboration', action: 'startCollaboration' },
      { command: 'enable haptics', action: 'toggleHaptics', parameters: { enabled: true } },
      { command: 'accessibility mode', action: 'toggleAccessibility' }
    ];

    const matchedCommand = commands.find(cmd => command.includes(cmd.command));
    if (matchedCommand) {
      executeVoiceCommand(matchedCommand);
    }
  }, []);

  const executeVoiceCommand = useCallback((command: VoiceCommand) => {
    switch (command.action) {
      case 'loadScene':
        loadGeologicalScene(command.parameters?.type);
        break;
      case 'toggleLayers':
        toggleDataLayers(command.parameters?.visible);
        break;
      case 'zoom':
        handleZoom(command.parameters?.factor);
        break;
      case 'rotate':
        handleRotation(command.parameters?.axis, command.parameters?.angle);
        break;
      case 'startMeasurement':
        startMeasurementMode();
        break;
      case 'takeSample':
        takeSample();
        break;
      case 'startCollaboration':
        startCollaborativeSession();
        break;
      case 'toggleHaptics':
        setConfig(prev => ({ ...prev, hapticEnabled: command.parameters?.enabled }));
        break;
      case 'toggleAccessibility':
        setConfig(prev => ({ ...prev, accessibilityMode: !prev.accessibilityMode }));
        break;
    }
  }, []);

  // Haptic feedback functions
  const triggerHapticFeedback = useCallback((feedback: HapticFeedback) => {
    if (!config.hapticEnabled || !hapticDevice) return;

    switch (feedback.pattern) {
      case 'pulse':
        const pattern = Array(Math.floor(feedback.duration / 150)).fill([feedback.intensity * 100, 50]).flat();
        hapticDevice.vibrate(pattern);
        break;
      case 'continuous':
        hapticDevice.vibrate(feedback.duration);
        break;
      case 'wave':
        const wavePattern = Array.from({ length: 10 }, (_, i) => 
          Math.sin(i * 0.6) * feedback.intensity * 50 + 50
        );
        hapticDevice.vibrate(wavePattern);
        break;
      case 'impact':
        hapticDevice.vibrate([feedback.intensity * 200, 100, feedback.intensity * 100]);
        break;
    }
  }, [config.hapticEnabled, hapticDevice]);

  // Geological scene loading
  const loadGeologicalScene = useCallback(async (sceneType: string) => {
    try {
      // This would load from the backend geological data services
      const sceneData = await fetchGeologicalScene(sceneType);
      setCurrentScene(sceneData);
    } catch (error) {
      console.error('Failed to load geological scene:', error);
    }
  }, []);

  // Collaborative session management
  const startCollaborativeSession = useCallback(async () => {
    try {
      const session = await initializeCollaborativeSession();
      setCollaborativeSession(session);
      setConfig(prev => ({ ...prev, collaborativeMode: true }));
    } catch (error) {
      console.error('Failed to start collaborative session:', error);
    }
  }, []);

  // Gesture recognition handler
  const handleGesture = useCallback((gesture: GestureRecognition) => {
    if (!config.gestureEnabled) return;

    switch (gesture.type) {
      case 'pinch':
        handleZoom(0.8);
        break;
      case 'grab':
        startObjectManipulation(gesture.position);
        break;
      case 'point':
        selectObject(gesture.position, gesture.direction);
        break;
      case 'swipe':
        navigateScene(gesture.direction);
        break;
      case 'rotate':
        rotateView(gesture.direction);
        break;
      case 'scale':
        scaleObject(gesture.position, 1.2);
        break;
    }
  }, [config.gestureEnabled]);

  // Utility functions
  const handleZoom = useCallback((_factor: number) => {
    // Implementation for zoom functionality
  }, []);

  const handleRotation = useCallback((_axis: string, _angle: number) => {
    // Implementation for rotation functionality
  }, []);

  const toggleDataLayers = useCallback((visible: boolean) => {
    // Implementation for layer visibility toggle
  }, []);

  const startMeasurementMode = useCallback(() => {
    // Implementation for measurement mode
  }, []);

  const takeSample = useCallback(() => {
    // Implementation for sample taking
  }, []);

  const startObjectManipulation = useCallback((position: THREE.Vector3) => {
    // Implementation for object manipulation
  }, []);

  const selectObject = useCallback((position: THREE.Vector3, direction?: THREE.Vector3) => {
    // Implementation for object selection
  }, []);

  const navigateScene = useCallback((direction?: THREE.Vector3) => {
    // Implementation for scene navigation
  }, []);

  const rotateView = useCallback((direction?: THREE.Vector3) => {
    // Implementation for view rotation
  }, []);

  const scaleObject = useCallback((_position: any, _factor: number) => {
    // Implementation for object scaling
  }, []);

  // Async helper functions (would integrate with actual services)
  const initializeAdvancedHaptics = async () => {
    // Integration with haptic device SDKs
    return null;
  };

  const fetchGeologicalScene = async (sceneType: string): Promise<GeologicalVRScene> => {
    // Fetch from geological data services
    return {
      id: `scene-${sceneType}`,
      name: `${sceneType} Scene`,
      description: `Geological ${sceneType} visualization`,
      type: sceneType as any,
      models: [],
      dataLayers: [],
      interactionPoints: [],
      collaborativeSpaces: []
    };
  };

  const initializeCollaborativeSession = async (): Promise<CollaborativeSpace> => {
    // Initialize collaborative session
    return {
      id: 'collab-session',
      name: 'Geological Collaboration',
      participants: [],
      sharedObjects: [],
      voiceChat: true,
      spatialAudio: true
    };
  };

  return (
    <MuiBox sx={{ width: '100%', height: '100vh', position: 'relative' }}>
      {/* Control Panel */}
      <Card sx={{ 
        position: 'absolute', 
        top: 16, 
        left: 16, 
        zIndex: 1000,
        minWidth: 300,
        maxHeight: '80vh',
        overflow: 'auto'
      }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Immersive Geological Visualization
          </Typography>
          
          <MuiGrid container spacing={2}>
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.vrEnabled}
                    onChange={(e) => setConfig(prev => ({ ...prev, vrEnabled: e.target.checked }))}
                    disabled={!isXRSupported}
                  />
                }
                label="Virtual Reality"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.arEnabled}
                    onChange={(e) => setConfig(prev => ({ ...prev, arEnabled: e.target.checked }))}
                    disabled={!isXRSupported}
                  />
                }
                label="Augmented Reality"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.hapticEnabled}
                    onChange={(e) => setConfig(prev => ({ ...prev, hapticEnabled: e.target.checked }))}
                  />
                }
                label="Haptic Feedback"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.voiceEnabled}
                    onChange={(e) => setConfig(prev => ({ ...prev, voiceEnabled: e.target.checked }))}
                  />
                }
                label="Voice Control"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.gestureEnabled}
                    onChange={(e) => setConfig(prev => ({ ...prev, gestureEnabled: e.target.checked }))}
                  />
                }
                label="Gesture Recognition"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.collaborativeMode}
                    onChange={(e) => setConfig(prev => ({ ...prev, collaborativeMode: e.target.checked }))}
                  />
                }
                label="Collaborative Mode"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.accessibilityMode}
                    onChange={(e) => setConfig(prev => ({ ...prev, accessibilityMode: e.target.checked }))}
                  />
                }
                label="Accessibility Mode"
              />
            </MuiGrid>
            
            <MuiGrid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={config.educationMode}
                    onChange={(e) => setConfig(prev => ({ ...prev, educationMode: e.target.checked }))}
                  />
                }
                label="Education Mode"
              />
            </MuiGrid>
          </MuiGrid>

          {!isXRSupported && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              WebXR not supported in this browser. VR/AR features will be limited.
            </Alert>
          )}

          {config.collaborativeMode && collaborativeSession && (
            <MuiBox sx={{ mt: 2 }}>
              <Typography variant="subtitle2">
                Collaborative Session: {collaborativeSession.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Participants: {collaborativeSession.participants.length}
              </Typography>
            </MuiBox>
          )}
        </CardContent>
      </Card>

      {/* XR Buttons */}
      {isXRSupported && (
        <MuiBox sx={{ position: 'absolute', top: 16, right: 16, zIndex: 1000 }}>
          {config.vrEnabled && <VRButton />}
          {config.arEnabled && <ARButton />}
        </MuiBox>
      )}

      {/* Main 3D Canvas */}
      <Canvas
        ref={canvasRef}
        camera={{ position: [0, 5, 10], fov: 75 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ 
          antialias: true, 
          alpha: true,
          powerPreference: 'high-performance'
        }}
      >
        <XR>
          <ImmersiveScene 
            config={config}
            scene={currentScene}
            onGesture={handleGesture}
            onHapticFeedback={triggerHapticFeedback}
            collaborativeSession={collaborativeSession}
          />
          <Controllers />
          <Hands />
        </XR>
      </Canvas>

      {/* Status Indicators */}
      <MuiBox sx={{ 
        position: 'absolute', 
        bottom: 16, 
        left: 16, 
        zIndex: 1000,
        display: 'flex',
        gap: 1
      }}>
        {config.hapticEnabled && (
          <Chip 
            icon={<Vibration />} 
            label="Haptics" 
            color="primary" 
            size="small" 
          />
        )}
        {config.voiceEnabled && (
          <Chip 
            icon={voiceRecognition ? <Mic /> : <MicOff />} 
            label="Voice" 
            color={voiceRecognition ? "primary" : "default"} 
            size="small" 
          />
        )}
        {config.collaborativeMode && (
          <Chip 
            icon={<Group />} 
            label="Collaborative" 
            color="secondary" 
            size="small" 
          />
        )}
        {config.accessibilityMode && (
          <Chip 
            icon={<Accessibility />} 
            label="Accessible" 
            color="success" 
            size="small" 
          />
        )}
      </MuiBox>
    </MuiBox>
  );
};

export default ImmersiveVisualizationEngine;