import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree as _useThree } from '@react-three/fiber';
// import { ARButton, XR } from '@react-three/xr';
import { Text, Html as _Html, Line as _Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector as _useSelector } from 'react-redux';
import { FieldData, ARSession as _ARSession, GeolocationData as _GeolocationData } from '../../types/immersive';
import { updateARSession as _updateARSession, addFieldObservation } from '../../store/slices/immersiveSlice';

interface ARFieldDataOverlayProps {
  fieldData: FieldData[];
  onDataCapture?: (data: FieldData) => void;
  enableGPS?: boolean;
  enableCamera?: boolean;
  showGeologicalLayers?: boolean;
}

interface ARMarker {
  id: string;
  position: THREE.Vector3;
  type: 'sample' | 'measurement' | 'observation' | 'hazard';
  data: any;
  timestamp: Date;
}

interface GeolocationCoords {
  latitude: number;
  longitude: number;
  altitude?: number;
  accuracy: number;
}

const ARFieldDataOverlay: React.FC<ARFieldDataOverlayProps> = ({
  fieldData,
  onDataCapture,
  enableGPS = true,
  enableCamera = true,
  showGeologicalLayers = true
}) => {
  const dispatch = useDispatch();
  // const _arSession = useSelector((state: any) => state.immersive.arSession);
  const [arSupported, setARSupported] = useState(false);
  const [currentLocation, setCurrentLocation] = useState<GeolocationCoords | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [markers, setMarkers] = useState<ARMarker[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    // Check AR support
    if ('xr' in navigator) {
      navigator.xr?.isSessionSupported('immersive-ar').then(setARSupported);
    }

    // Initialize GPS tracking
    if (enableGPS && 'geolocation' in navigator) {
      initializeGPSTracking();
    }

    // Initialize camera access
    if (enableCamera) {
      initializeCameraAccess();
    }
  }, [enableGPS, enableCamera]);

  const initializeGPSTracking = useCallback(() => {
    const options = {
      enableHighAccuracy: true,
      timeout: 10000,
      maximumAge: 60000
    };

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const coords: GeolocationCoords = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          altitude: position.coords.altitude || undefined,
          accuracy: position.coords.accuracy
        };
        setCurrentLocation(coords);
      },
      (error) => {
        console.error('GPS tracking error:', error);
      },
      options
    );

    return () => {
      navigator.geolocation.clearWatch(watchId);
    };
  }, []);

  const initializeCameraAccess = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      
      setCameraStream(stream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error('Camera access error:', error);
    }
  }, []);

  const captureFieldData = useCallback(async (type: string, position: THREE.Vector3) => {
    if (!currentLocation) {
      console.warn('GPS location not available');
      return;
    }

    setIsCapturing(true);

    try {
      // Capture photo if camera is available
      let photoData: string | null = null;
      if (cameraStream && videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          photoData = canvas.toDataURL('image/jpeg', 0.8);
        }
      }

      const fieldObservation: FieldData = {
        id: `field_${Date.now()}`,
        type: type as any,
        position: {
          x: position.x,
          y: position.y,
          z: position.z
        },
        gpsCoordinates: currentLocation,
        timestamp: new Date(),
        photo: photoData || undefined,
        properties: {
          accuracy: currentLocation?.accuracy,
          deviceOrientation: await getDeviceOrientation(),
          environmentalConditions: await getEnvironmentalConditions()
        }
      };

      // Add AR marker
      const marker: ARMarker = {
        id: fieldObservation.id,
        position,
        type: type as any,
        data: fieldObservation,
        timestamp: new Date()
      };

      setMarkers(prev => [...prev, marker]);
      dispatch(addFieldObservation(fieldObservation));
      onDataCapture?.(fieldObservation);

    } catch (error) {
      console.error('Field data capture error:', error);
    } finally {
      setIsCapturing(false);
    }
  }, [currentLocation, cameraStream, dispatch, onDataCapture]);

  const getDeviceOrientation = async (): Promise<any> => {
    if ('DeviceOrientationEvent' in window) {
      return new Promise((resolve) => {
        const handler = (event: DeviceOrientationEvent) => {
          window.removeEventListener('deviceorientation', handler);
          resolve({
            alpha: event.alpha,
            beta: event.beta,
            gamma: event.gamma
          });
        };
        window.addEventListener('deviceorientation', handler);
        
        // Timeout after 1 second
        setTimeout(() => {
          window.removeEventListener('deviceorientation', handler);
          resolve(null);
        }, 1000);
      });
    }
    return null;
  };

  const getEnvironmentalConditions = async (): Promise<any> => {
    // Placeholder for environmental sensor data
    return {
      timestamp: new Date().toISOString(),
      weather: 'unknown',
      lighting: 'unknown'
    };
  };

  return (
    <div className="ar-field-overlay">
      <div className="ar-controls">
        {/* {arSupported && <ARButton />} */}
        <div className="ar-info">
          <p>AR Support: {arSupported ? 'Available' : 'Not Available'}</p>
          <p>GPS: {currentLocation ? 
            `${currentLocation.latitude.toFixed(6)}, ${currentLocation.longitude.toFixed(6)}` : 
            'Not Available'
          }</p>
          <p>Accuracy: {currentLocation?.accuracy.toFixed(1)}m</p>
          <p>Markers: {markers.length}</p>
        </div>
        
        <div className="capture-controls">
          <button 
            onClick={() => captureFieldData('sample', new THREE.Vector3(0, 0, -1))}
            disabled={isCapturing}
          >
            Capture Sample
          </button>
          <button 
            onClick={() => captureFieldData('measurement', new THREE.Vector3(0, 0, -1))}
            disabled={isCapturing}
          >
            Take Measurement
          </button>
          <button 
            onClick={() => captureFieldData('observation', new THREE.Vector3(0, 0, -1))}
            disabled={isCapturing}
          >
            Record Observation
          </button>
        </div>
      </div>

      {/* Camera video element (hidden, used for capture) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />

      <Canvas style={{ height: '100vh' }}>
        {/* <XR> */}
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={0.8} />
          
          <ARScene
            markers={markers}
            fieldData={fieldData}
            showGeologicalLayers={showGeologicalLayers}
            onMarkerInteraction={captureFieldData}
          />
          
          <ARInterface
            currentLocation={currentLocation}
            markerCount={markers.length}
            isCapturing={isCapturing}
          />
        {/* </XR> */}
      </Canvas>
    </div>
  );
};

interface ARSceneProps {
  markers: ARMarker[];
  fieldData: FieldData[];
  showGeologicalLayers: boolean;
  onMarkerInteraction: (type: string, position: THREE.Vector3) => void;
}

const ARScene: React.FC<ARSceneProps> = ({
  markers,
  fieldData,
  showGeologicalLayers,
  onMarkerInteraction
}) => {
  return (
    <>
      {/* Render AR markers */}
      {markers.map(marker => (
        <ARMarkerComponent
          key={marker.id}
          marker={marker}
          onInteraction={onMarkerInteraction}
        />
      ))}
      
      {/* Render existing field data */}
      {fieldData.map(data => (
        <FieldDataVisualization
          key={data.id}
          fieldData={data}
        />
      ))}
      
      {/* Geological layer overlays */}
      {showGeologicalLayers && (
        <GeologicalLayerOverlay />
      )}
      
      {/* GPS tracking indicator */}
      <GPSTrackingIndicator />
    </>
  );
};

interface ARMarkerComponentProps {
  marker: ARMarker;
  onInteraction: (type: string, position: THREE.Vector3) => void;
}

const ARMarkerComponent: React.FC<ARMarkerComponentProps> = ({
  marker,
  onInteraction
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      // Animate marker
      meshRef.current.rotation.y = state.clock.elapsedTime;
      meshRef.current.scale.setScalar(hovered ? 1.2 : 1.0);
    }
  });

  const getMarkerColor = (type: string): number => {
    switch (type) {
      case 'sample': return 0x00FF00;
      case 'measurement': return 0x0000FF;
      case 'observation': return 0xFFFF00;
      case 'hazard': return 0xFF0000;
      default: return 0xFFFFFF;
    }
  };

  const getMarkerGeometry = (type: string): JSX.Element => {
    switch (type) {
      case 'sample':
        return <Box args={[0.1, 0.1, 0.1]} />;
      case 'measurement':
        return <Sphere args={[0.05, 8, 8]} />;
      case 'observation':
        return <Box args={[0.08, 0.15, 0.08]} />;
      case 'hazard':
        return <Box args={[0.12, 0.05, 0.12]} />;
      default:
        return <Sphere args={[0.05, 8, 8]} />;
    }
  };

  return (
    <group position={marker.position}>
      <mesh
        ref={meshRef}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
        onClick={() => onInteraction(marker.type, marker.position)}
      >
        {getMarkerGeometry(marker.type)}
        <meshStandardMaterial
          color={getMarkerColor(marker.type)}
          emissive={hovered ? 0x222222 : 0x000000}
        />
      </mesh>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.05}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {marker.type.toUpperCase()}
      </Text>
      
      <Text
        position={[0, 0.15, 0]}
        fontSize={0.03}
        color="lightgray"
        anchorX="center"
        anchorY="middle"
      >
        {marker.timestamp.toLocaleTimeString()}
      </Text>
    </group>
  );
};

const FieldDataVisualization: React.FC<{ fieldData: FieldData }> = ({
  fieldData
}) => {
  return (
    <group position={[
      fieldData.position.x,
      fieldData.position.y,
      fieldData.position.z
    ]}>
      <Sphere args={[0.03, 8, 8]}>
        <meshBasicMaterial color={0x888888} transparent opacity={0.7} />
      </Sphere>
      
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.02}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {fieldData.type}
      </Text>
    </group>
  );
};

const GeologicalLayerOverlay: React.FC = () => {
  // Simplified geological layer visualization for AR
  const layers = [
    { name: 'Surface', y: 0, color: 0x8B4513 },
    { name: 'Sedimentary', y: -0.5, color: 0xDEB887 },
    { name: 'Igneous', y: -1.0, color: 0xFF4500 },
    { name: 'Metamorphic', y: -1.5, color: 0x696969 }
  ];

  return (
    <group>
      {layers.map((layer, _index) => (
        <group key={layer.name} position={[2, layer.y, -2]}>
          <Box args={[1, 0.1, 0.5]}>
            <meshStandardMaterial
              color={layer.color}
              transparent
              opacity={0.6}
            />
          </Box>
          
          <Text
            position={[0.6, 0, 0]}
            fontSize={0.05}
            color="white"
            anchorX="left"
            anchorY="middle"
          >
            {layer.name}
          </Text>
        </group>
      ))}
    </group>
  );
};

const GPSTrackingIndicator: React.FC = () => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      const material = meshRef.current.material;
      if (Array.isArray(material)) {
        material.forEach(mat => mat.opacity = 0.5 + 0.3 * Math.sin(state.clock.elapsedTime * 2));
      } else {
        material.opacity = 0.5 + 0.3 * Math.sin(state.clock.elapsedTime * 2);
      }
    }
  });

  return (
    <group position={[0, 2, -1]}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshBasicMaterial color={0x00FF00} transparent />
      </mesh>
      
      <Text
        position={[0, -0.15, 0]}
        fontSize={0.03}
        color="green"
        anchorX="center"
        anchorY="middle"
      >
        GPS ACTIVE
      </Text>
    </group>
  );
};

const ARInterface: React.FC<{
  currentLocation: GeolocationCoords | null;
  markerCount: number;
  isCapturing: boolean;
}> = ({ currentLocation, markerCount, isCapturing }) => {
  return (
    <group position={[-2, 1.5, -1]}>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Field Data Collection
      </Text>
      
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.06}
        color="lightblue"
        anchorX="center"
        anchorY="middle"
      >
        Markers: {markerCount}
      </Text>
      
      {currentLocation && (
        <Text
          position={[0, -0.1, 0]}
          fontSize={0.05}
          color="lightgreen"
          anchorX="center"
          anchorY="middle"
        >
          GPS: {currentLocation.accuracy.toFixed(1)}m
        </Text>
      )}
      
      {isCapturing && (
        <Text
          position={[0, -0.3, 0]}
          fontSize={0.06}
          color="yellow"
          anchorX="center"
          anchorY="middle"
        >
          CAPTURING...
        </Text>
      )}
    </group>
  );
};

export default ARFieldDataOverlay;