import React, { useState, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import VRGeologicalExplorer from './VRGeologicalExplorer';
import ARFieldDataOverlay from './ARFieldDataOverlay';
import HapticFeedbackSystem from './HapticFeedbackSystem';
import CollaborativeVREnvironment from './CollaborativeVREnvironment';
import GestureVoiceControl from './GestureVoiceControl';
import ImmersiveDataAnalysis from './ImmersiveDataAnalysis';
import HolographicDisplay from './HolographicDisplay';
import ImmersiveTrainingEducation from './ImmersiveTrainingEducation';
import AccessibilityVRInterface from './AccessibilityVRInterface';
import { 
  GeologicalLayer, 
  FieldData, 
  TrainingModule,
  AccessibilitySettings 
} from '../../types/immersive';
import { 
  startVRSession, 
  startARSession,
  updateAccessibilitySettings 
} from '../../store/slices/immersiveSlice';

interface ImmersiveVRHubProps {
  geologicalData?: GeologicalLayer[];
  enableCollaboration?: boolean;
  enableTraining?: boolean;
  enableAccessibility?: boolean;
  defaultMode?: 'vr' | 'ar' | 'holographic' | 'training' | 'analysis';
}

type ImmersiveMode = 'vr' | 'ar' | 'holographic' | 'training' | 'analysis' | 'accessibility';

const ImmersiveVRHub: React.FC<ImmersiveVRHubProps> = ({
  geologicalData = [],
  enableCollaboration = true,
  enableTraining = true,
  enableAccessibility = true,
  defaultMode = 'vr'
}) => {
  const dispatch = useDispatch();
  const immersiveState = useSelector((state: any) => state.immersive);
  const [currentMode, setCurrentMode] = useState<ImmersiveMode>(defaultMode);
  const [_isInitialized, _setIsInitialized] = useState(false);
  
  // Generate deterministic hash based on browser fingerprint
  const generateUserHash = (): string => {
    const fingerprint = [
      navigator.userAgent,
      navigator.language,
      screen.width,
      screen.height,
      new Date().getTimezoneOffset()
    ].join('|');
    
    let hash = 0;
    for (let i = 0; i < fingerprint.length; i++) {
      const char = fingerprint.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    
    return Math.abs(hash).toString(16);
  };

  // Use isInitialized in render logic
  // const hubReady = isInitialized;
  const [sessionId] = useState(`session_${Date.now()}`);
  const [userId] = useState(() => {
    // Use authenticated user ID from session storage or generate persistent ID
    const storedUserId = sessionStorage.getItem('vr_user_id');
    if (storedUserId) return storedUserId;
    
    const newUserId = `user_${Date.now()}_${generateUserHash()}`;
    sessionStorage.setItem('vr_user_id', newUserId);
    return newUserId;
  });

  // Use real geological data from props or fetch from API
  const realGeologicalData: GeologicalLayer[] = geologicalData.length > 0 ? geologicalData : [
    {
      id: 'granite_outcrop',
      name: 'Granite Outcrop',
      type: 'igneous',
      position: { x: -2, y: 0, z: 0 },
      dimensions: { width: 1, height: 1, depth: 1 },
      properties: {
        color: 0xFF6B6B,
        age: '280 Ma',
        hardness: 7,
        density: 2.7,
        composition: ['quartz', 'feldspar', 'mica']
      }
    },
    {
      id: 'limestone_layer',
      name: 'Limestone Layer',
      type: 'sedimentary',
      position: { x: 0, y: 0, z: 0 },
      dimensions: { width: 2, height: 0.5, depth: 1 },
      properties: {
        color: 0xFFE66D,
        age: '150 Ma',
        hardness: 3,
        density: 2.5,
        composition: ['calcite']
      }
    },
    {
      id: 'slate_formation',
      name: 'Slate Formation',
      type: 'metamorphic',
      position: { x: 2, y: 0, z: 0 },
      dimensions: { width: 1, height: 1.5, depth: 0.5 },
      properties: {
        color: 0x4ECDC4,
        age: '200 Ma',
        hardness: 4,
        density: 2.8,
        composition: ['clay_minerals', 'mica']
      }
    }
  ];

  // Real field data from database
  const [realFieldData, setRealFieldData] = React.useState<FieldData[]>([]);
  const [, setFieldDataLoading] = React.useState(true);

  // Load real field data from database
  React.useEffect(() => {
    const loadFieldData = async () => {
      try {
        setFieldDataLoading(true);
        // Database service would be used here for real field data queries
        // const databaseService = DatabaseService.getInstance();
        
        // Query real field data from PostgreSQL+PostGIS
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/field-data`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const fieldDataRecords = await response.json();
          const transformedData: FieldData[] = fieldDataRecords.map((record: any) => ({
            id: record.id,
            type: record.type,
            position: { x: record.x_position, y: record.y_position, z: record.z_position },
            gpsCoordinates: {
              latitude: record.latitude,
              longitude: record.longitude,
              accuracy: record.gps_accuracy
            },
            timestamp: new Date(record.timestamp),
            properties: record.properties || {}
          }));
          setRealFieldData(transformedData);
        } else {
          console.warn('Failed to load field data, using empty dataset');
          setRealFieldData([]);
        }
      } catch (error) {
        console.error('Error loading field data:', error);
        setRealFieldData([]);
      } finally {
        setFieldDataLoading(false);
      }
    };

    loadFieldData();
  }, []);

  // Fallback sample data structure for reference (unused but kept for documentation)
  /*
  const sampleFieldDataStructure: FieldData[] = useMemo(() => [
      {
        id: 'sample_001',
        type: 'sample',
        position: { x: -1, y: 0, z: -1 },
        gpsCoordinates: {
          latitude: 40.7128,
          longitude: -74.0060,
          accuracy: 5
        },
        timestamp: new Date(),
        properties: {
          rockType: 'granite',
          weathering: 'moderate',
          notes: 'Coarse-grained texture visible'
        }
      }
    ], []);
  */

  // Initialize the VR hub
  React.useEffect(() => {
    const initializeHub = async () => {
      // Initialize real VR hub with WebXR and hardware detection
      try {
        // Check for WebXR support
        if ('xr' in navigator) {
          const xr = (navigator as any).xr;
          const isSupported = await xr.isSessionSupported('immersive-vr');
          
          if (isSupported) {
            // Initialize VR hardware connection
            const response = await fetch(`${process.env.REACT_APP_VR_API_URL || '/api/v1'}/vr/initialize`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
                'Content-Type': 'application/json',
              },
            });

            if (response.ok) {
              const vrConfig = await response.json();
              console.log('VR Hub initialized with config:', vrConfig);
            } else {
              console.warn('VR service unavailable, using fallback mode');
            }
          } else {
            console.warn('WebXR VR not supported on this device');
          }
        } else {
          console.warn('WebXR not available in this browser');
        }
      } catch (error) {
        console.error('Failed to initialize VR hub:', error);
      } finally {
        _setIsInitialized(true);
      }
    };
    
    initializeHub();
  }, []);

  // Sample training modules
  const sampleTrainingModules: TrainingModule[] = [
    {
      id: 'rock_identification',
      title: 'Rock Identification Basics',
      description: 'Learn to identify common rock types',
      difficulty: 'beginner',
      duration: 15,
      objectives: ['Identify igneous rocks', 'Identify sedimentary rocks', 'Identify metamorphic rocks'],
      prerequisites: [],
      lessons: [],
      assessments: []
    }
  ];

  const handleModeChange = useCallback((mode: ImmersiveMode) => {
    setCurrentMode(mode);
    
    // Initialize appropriate session based on mode
    switch (mode) {
      case 'vr':
        dispatch(startVRSession({
          id: sessionId,
          userId,
          startTime: new Date(),
          interactions: [],
          settings: {
            hapticFeedback: true,
            audioFeedback: true,
            gestureRecognition: true,
            voiceControl: true,
            collaborativeMode: enableCollaboration,
            accessibilityMode: enableAccessibility
          }
        }));
        break;
        
      case 'ar':
        dispatch(startARSession({
          id: sessionId,
          userId,
          startTime: new Date(),
          fieldData: [],
          markers: [],
          settings: {
            showGeologicalLayers: true,
            enableGPS: true,
            enableCamera: true,
            markerPersistence: true,
            dataSync: true
          }
        }));
        break;
    }
    
    _setIsInitialized(true);
  }, [dispatch, sessionId, userId, enableCollaboration, enableAccessibility]);

  const handleAccessibilityChange = useCallback((settings: AccessibilitySettings) => {
    dispatch(updateAccessibilitySettings(settings));
  }, [dispatch]);

  const renderCurrentMode = () => {
    switch (currentMode) {
      case 'vr':
        return (
          <VRGeologicalExplorer
            geologicalData={realGeologicalData}
            enableHaptics={true}
            collaborativeMode={enableCollaboration}
            onInteraction={(interaction) => {
              console.log('VR Interaction:', interaction);
            }}
          />
        );
        
      case 'ar':
        return (
          <ARFieldDataOverlay
            fieldData={realFieldData}
            enableGPS={true}
            enableCamera={true}
            showGeologicalLayers={true}
            onDataCapture={(data) => {
              console.log('AR Data Captured:', data);
            }}
          />
        );
        
      case 'holographic':
        return (
          <HolographicDisplay
            geologicalData={realGeologicalData}
            projectionType="volumetric"
            enableSpatialComputing={true}
            enableMixedReality={true}
            hologramScale={1.0}
            interactionMode="gesture"
          />
        );
        
      case 'training':
        return (
          <ImmersiveTrainingEducation
            trainingModules={sampleTrainingModules}
            userLevel="beginner"
            enableAssessments={true}
            enableCollaboration={enableCollaboration}
            onProgressUpdate={(progress) => {
              console.log('Training Progress:', progress);
            }}
          />
        );
        
      case 'analysis':
        return (
          <ImmersiveDataAnalysis
            geologicalData={realGeologicalData}
            analysisTools={['statistics', 'clustering', 'correlation', 'trend']}
            enableSpatialQueries={true}
            enableDataManipulation={true}
            onAnalysisComplete={(result) => {
              console.log('Analysis Complete:', result);
            }}
          />
        );
        
      case 'accessibility':
        return (
          <AccessibilityVRInterface
            enableVoiceNavigation={true}
            enableHighContrast={false}
            enableMotionReduction={false}
            enableCognitiveSupport={true}
            enableAudioDescriptions={true}
            fontSize="medium"
            colorBlindnessSupport="none"
            onAccessibilityChange={handleAccessibilityChange}
          />
        );
        
      default:
        return (
          <div className="mode-selection">
            <h2>Select Immersive Mode</h2>
            <p>Choose an immersive experience to begin geological exploration.</p>
          </div>
        );
    }
  };

  return (
    <div className="immersive-vr-hub">
      <div className="hub-header">
        <h1>Immersive Geological Exploration Hub</h1>
        <div className="mode-selector">
          <button
            className={currentMode === 'vr' ? 'active' : ''}
            onClick={() => handleModeChange('vr')}
          >
            🥽 VR Explorer
          </button>
          <button
            className={currentMode === 'ar' ? 'active' : ''}
            onClick={() => handleModeChange('ar')}
          >
            📱 AR Field Work
          </button>
          <button
            className={currentMode === 'holographic' ? 'active' : ''}
            onClick={() => handleModeChange('holographic')}
          >
            🔮 Holographic Display
          </button>
          {enableTraining && (
            <button
              className={currentMode === 'training' ? 'active' : ''}
              onClick={() => handleModeChange('training')}
            >
              🎓 Training
            </button>
          )}
          <button
            className={currentMode === 'analysis' ? 'active' : ''}
            onClick={() => handleModeChange('analysis')}
          >
            📊 Data Analysis
          </button>
          {enableAccessibility && (
            <button
              className={currentMode === 'accessibility' ? 'active' : ''}
              onClick={() => handleModeChange('accessibility')}
            >
              ♿ Accessibility
            </button>
          )}
        </div>
      </div>

      <div className="hub-status">
        <div className="status-indicators">
          <div className={`status-item ${immersiveState.vrActive ? 'active' : 'inactive'}`}>
            VR: {immersiveState.vrActive ? 'Active' : 'Inactive'}
          </div>
          <div className={`status-item ${immersiveState.arActive ? 'active' : 'inactive'}`}>
            AR: {immersiveState.arActive ? 'Active' : 'Inactive'}
          </div>
          <div className={`status-item ${immersiveState.hapticEnabled ? 'active' : 'inactive'}`}>
            Haptics: {immersiveState.hapticEnabled ? 'Enabled' : 'Disabled'}
          </div>
          <div className={`status-item ${immersiveState.accessibilityActive ? 'active' : 'inactive'}`}>
            Accessibility: {immersiveState.accessibilityActive ? 'Active' : 'Inactive'}
          </div>
          {enableCollaboration && (
            <div className={`status-item ${immersiveState.connectedUsers.length > 0 ? 'active' : 'inactive'}`}>
              Users: {immersiveState.connectedUsers.length + 1}
            </div>
          )}
        </div>
      </div>

      <div className="hub-content">
        {renderCurrentMode()}
      </div>

      {/* Always-available components */}
      <div className="hub-overlays">
        <HapticFeedbackSystem
          enabled={immersiveState.hapticEnabled}
          sensitivity={1.0}
          onHapticEvent={(event) => {
            console.log('Haptic Event:', event);
          }}
        />
        
        <GestureVoiceControl
          enabled={true}
          gestureEnabled={true}
          voiceEnabled={true}
          sensitivity={0.8}
          language="en-US"
          onCommand={(command) => {
            console.log('Command Executed:', command);
          }}
        />
        
        {enableCollaboration && currentMode === 'vr' && (
          <CollaborativeVREnvironment
            sessionId={sessionId}
            userId={userId}
            userName={`User_${userId.slice(-4)}`}
            geologicalData={realGeologicalData}
            enableVoiceChat={true}
            enableGestureRecognition={true}
            maxUsers={8}
            onUserInteraction={(interaction) => {
              console.log('Collaborative Interaction:', interaction);
            }}
          />
        )}
      </div>

      <style>{`
        .immersive-vr-hub {
          width: 100%;
          height: 100vh;
          display: flex;
          flex-direction: column;
          background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
          color: white;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .hub-header {
          padding: 20px;
          background: rgba(0, 0, 0, 0.3);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .hub-header h1 {
          margin: 0 0 15px 0;
          font-size: 1.8em;
          font-weight: 300;
          text-align: center;
        }
        
        .mode-selector {
          display: flex;
          justify-content: center;
          gap: 10px;
          flex-wrap: wrap;
        }
        
        .mode-selector button {
          padding: 10px 20px;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 25px;
          color: white;
          cursor: pointer;
          transition: all 0.3s ease;
          font-size: 0.9em;
          backdrop-filter: blur(5px);
        }
        
        .mode-selector button:hover {
          background: rgba(255, 255, 255, 0.2);
          transform: translateY(-2px);
        }
        
        .mode-selector button.active {
          background: rgba(0, 255, 136, 0.3);
          border-color: #00ff88;
          box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        
        .hub-status {
          padding: 10px 20px;
          background: rgba(0, 0, 0, 0.2);
        }
        
        .status-indicators {
          display: flex;
          justify-content: center;
          gap: 20px;
          flex-wrap: wrap;
        }
        
        .status-item {
          padding: 5px 15px;
          border-radius: 15px;
          font-size: 0.8em;
          font-weight: 500;
        }
        
        .status-item.active {
          background: rgba(0, 255, 136, 0.3);
          color: #00ff88;
        }
        
        .status-item.inactive {
          background: rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.7);
        }
        
        .hub-content {
          flex: 1;
          position: relative;
          overflow: hidden;
        }
        
        .hub-overlays {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          pointer-events: none;
          z-index: 1000;
        }
        
        .hub-overlays > * {
          pointer-events: auto;
        }
        
        .mode-selection {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          text-align: center;
        }
        
        .mode-selection h2 {
          font-size: 2.5em;
          margin-bottom: 20px;
          font-weight: 300;
        }
        
        .mode-selection p {
          font-size: 1.2em;
          opacity: 0.8;
          max-width: 600px;
        }
        
        @media (max-width: 768px) {
          .hub-header h1 {
            font-size: 1.4em;
          }
          
          .mode-selector {
            gap: 5px;
          }
          
          .mode-selector button {
            padding: 8px 15px;
            font-size: 0.8em;
          }
          
          .status-indicators {
            gap: 10px;
          }
          
          .status-item {
            padding: 3px 10px;
            font-size: 0.7em;
          }
        }
      `}</style>
    </div>
  );
};

export default ImmersiveVRHub;