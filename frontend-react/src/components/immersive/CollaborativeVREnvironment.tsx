import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { VRButton, XR, Controllers, Hands } from '@react-three/xr';
import { Text, Avatar, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector } from 'react-redux';
import { io, Socket } from 'socket.io-client';
import { 
  CollaborativeSession, 
  VRUser, 
  SharedAnnotation, 
  VoiceChat,
  GestureRecognition 
} from '../../types/immersive';
import { 
  updateCollaborativeSession, 
  addVRUser, 
  removeVRUser,
  addSharedAnnotation,
  updateVoiceChat
} from '../../store/slices/immersiveSlice';

interface CollaborativeVREnvironmentProps {
  sessionId: string;
  userId: string;
  userName: string;
  geologicalData?: any[];
  onUserInteraction?: (interaction: UserInteraction) => void;
  enableVoiceChat?: boolean;
  enableGestureRecognition?: boolean;
  maxUsers?: number;
}

interface UserInteraction {
  userId: string;
  type: 'point' | 'select' | 'annotate' | 'gesture' | 'voice';
  position: THREE.Vector3;
  data?: any;
  timestamp: number;
}

interface VRAvatar {
  userId: string;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  headPosition: THREE.Vector3;
  leftHandPosition: THREE.Vector3;
  rightHandPosition: THREE.Vector3;
  isActive: boolean;
  color: number;
}

const CollaborativeVREnvironment: React.FC<CollaborativeVREnvironmentProps> = ({
  sessionId,
  userId,
  userName,
  geologicalData = [],
  onUserInteraction,
  enableVoiceChat = true,
  enableGestureRecognition = true,
  maxUsers = 8
}) => {
  const dispatch = useDispatch();
  const collaborativeSession = useSelector((state: any) => state.immersive.collaborativeSession);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connectedUsers, setConnectedUsers] = useState<Map<string, VRAvatar>>(new Map());
  const [sharedAnnotations, setSharedAnnotations] = useState<SharedAnnotation[]>([]);
  const [voiceChatActive, setVoiceChatActive] = useState(false);
  const [gestureRecognizer, setGestureRecognizer] = useState<any>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const peerConnectionsRef = useRef<Map<string, RTCPeerConnection>>(new Map());

  useEffect(() => {
    initializeCollaborativeSession();
    
    if (enableVoiceChat) {
      initializeVoiceChat();
    }
    
    if (enableGestureRecognition) {
      initializeGestureRecognition();
    }

    return () => {
      cleanup();
    };
  }, [sessionId, userId, enableVoiceChat, enableGestureRecognition]);

  const initializeCollaborativeSession = useCallback(() => {
    // Initialize WebSocket connection
    const socketConnection = io('/collaborative-vr', {
      query: { sessionId, userId, userName }
    });

    socketConnection.on('connect', () => {
      console.log('Connected to collaborative VR session');
      dispatch(updateCollaborativeSession({
        sessionId,
        connected: true,
        userCount: 1
      }));
    });

    socketConnection.on('user-joined', (user: VRUser) => {
      console.log('User joined:', user);
      const avatar: VRAvatar = {
        userId: user.id,
        position: new THREE.Vector3(0, 0, 0),
        rotation: new THREE.Euler(0, 0, 0),
        headPosition: new THREE.Vector3(0, 1.6, 0),
        leftHandPosition: new THREE.Vector3(-0.3, 1.2, -0.3),
        rightHandPosition: new THREE.Vector3(0.3, 1.2, -0.3),
        isActive: true,
        color: generateUserColor(user.id)
      };
      
      setConnectedUsers(prev => new Map(prev.set(user.id, avatar)));
      dispatch(addVRUser(user));
    });

    socketConnection.on('user-left', (userId: string) => {
      console.log('User left:', userId);
      setConnectedUsers(prev => {
        const newMap = new Map(prev);
        newMap.delete(userId);
        return newMap;
      });
      dispatch(removeVRUser(userId));
    });

    socketConnection.on('user-position-update', (data: {
      userId: string;
      position: THREE.Vector3;
      rotation: THREE.Euler;
      headPosition: THREE.Vector3;
      leftHandPosition: THREE.Vector3;
      rightHandPosition: THREE.Vector3;
    }) => {
      setConnectedUsers(prev => {
        const newMap = new Map(prev);
        const user = newMap.get(data.userId);
        if (user) {
          user.position = data.position;
          user.rotation = data.rotation;
          user.headPosition = data.headPosition;
          user.leftHandPosition = data.leftHandPosition;
          user.rightHandPosition = data.rightHandPosition;
          newMap.set(data.userId, user);
        }
        return newMap;
      });
    });

    socketConnection.on('shared-annotation', (annotation: SharedAnnotation) => {
      setSharedAnnotations(prev => [...prev, annotation]);
      dispatch(addSharedAnnotation(annotation));
    });

    socketConnection.on('voice-chat-data', (data: { userId: string; audioData: ArrayBuffer }) => {
      handleIncomingVoiceData(data);
    });

    setSocket(socketConnection);
  }, [sessionId, userId, userName, dispatch]);

  const initializeVoiceChat = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      audioStreamRef.current = stream;
      
      // Initialize MediaRecorder for voice transmission
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket) {
          socket.emit('voice-chat-data', {
            sessionId,
            userId,
            audioData: event.data
          });
        }
      };
      
      mediaRecorderRef.current = mediaRecorder;
      
    } catch (error) {
      console.error('Voice chat initialization failed:', error);
    }
  }, [socket, sessionId, userId]);

  const initializeGestureRecognition = useCallback(() => {
    // Initialize gesture recognition system
    // This would integrate with a gesture recognition library
    const recognizer = {
      recognize: (handPositions: THREE.Vector3[]) => {
        // Implement gesture recognition logic
        return recognizeGesture(handPositions);
      }
    };
    
    setGestureRecognizer(recognizer);
  }, []);

  const generateUserColor = (userId: string): number => {
    // Generate consistent color for user based on ID
    let hash = 0;
    for (let i = 0; i < userId.length; i++) {
      hash = userId.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    const hue = Math.abs(hash) % 360;
    return new THREE.Color().setHSL(hue / 360, 0.7, 0.5).getHex();
  };

  const recognizeGesture = (handPositions: THREE.Vector3[]): string | null => {
    if (handPositions.length < 2) return null;
    
    const [leftHand, rightHand] = handPositions;
    const distance = leftHand.distanceTo(rightHand);
    
    // Simple gesture recognition
    if (distance < 0.2) {
      return 'clap';
    } else if (leftHand.y > 1.8 && rightHand.y > 1.8) {
      return 'raise_hands';
    } else if (Math.abs(leftHand.y - rightHand.y) > 0.5) {
      return 'wave';
    }
    
    return null;
  };

  const handleIncomingVoiceData = useCallback((data: { userId: string; audioData: ArrayBuffer }) => {
    // Handle incoming voice chat data
    const audioContext = new AudioContext();
    audioContext.decodeAudioData(data.audioData).then(audioBuffer => {
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      
      // Apply 3D spatial audio based on user position
      const panner = audioContext.createPanner();
      panner.panningModel = 'HRTF';
      panner.distanceModel = 'inverse';
      
      const userAvatar = connectedUsers.get(data.userId);
      if (userAvatar) {
        panner.setPosition(
          userAvatar.headPosition.x,
          userAvatar.headPosition.y,
          userAvatar.headPosition.z
        );
      }
      
      source.connect(panner);
      panner.connect(audioContext.destination);
      source.start();
    });
  }, [connectedUsers]);

  const startVoiceChat = useCallback(() => {
    if (mediaRecorderRef.current && !voiceChatActive) {
      mediaRecorderRef.current.start(100); // Send data every 100ms
      setVoiceChatActive(true);
      dispatch(updateVoiceChat({ active: true, userId }));
    }
  }, [voiceChatActive, userId, dispatch]);

  const stopVoiceChat = useCallback(() => {
    if (mediaRecorderRef.current && voiceChatActive) {
      mediaRecorderRef.current.stop();
      setVoiceChatActive(false);
      dispatch(updateVoiceChat({ active: false, userId }));
    }
  }, [voiceChatActive, userId, dispatch]);

  const createSharedAnnotation = useCallback((
    position: THREE.Vector3,
    text: string,
    type: 'note' | 'measurement' | 'highlight' = 'note'
  ) => {
    const annotation: SharedAnnotation = {
      id: `annotation_${Date.now()}`,
      userId,
      userName,
      position: { x: position.x, y: position.y, z: position.z },
      text,
      type,
      timestamp: new Date(),
      color: generateUserColor(userId)
    };
    
    if (socket) {
      socket.emit('create-annotation', { sessionId, annotation });
    }
    
    setSharedAnnotations(prev => [...prev, annotation]);
  }, [socket, sessionId, userId, userName]);

  const broadcastUserPosition = useCallback((
    position: THREE.Vector3,
    rotation: THREE.Euler,
    headPosition: THREE.Vector3,
    leftHandPosition: THREE.Vector3,
    rightHandPosition: THREE.Vector3
  ) => {
    if (socket) {
      socket.emit('user-position-update', {
        sessionId,
        userId,
        position,
        rotation,
        headPosition,
        leftHandPosition,
        rightHandPosition
      });
    }
  }, [socket, sessionId, userId]);

  const cleanup = useCallback(() => {
    if (socket) {
      socket.disconnect();
    }
    
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    
    peerConnectionsRef.current.forEach(pc => pc.close());
    peerConnectionsRef.current.clear();
  }, [socket]);

  return (
    <div className="collaborative-vr-environment">
      <div className="vr-session-info">
        <h3>Collaborative VR Session</h3>
        <p>Session ID: {sessionId}</p>
        <p>Connected Users: {connectedUsers.size + 1}</p>
        <p>Voice Chat: {voiceChatActive ? 'Active' : 'Inactive'}</p>
        
        <div className="voice-controls">
          <button
            onMouseDown={startVoiceChat}
            onMouseUp={stopVoiceChat}
            onTouchStart={startVoiceChat}
            onTouchEnd={stopVoiceChat}
          >
            Push to Talk
          </button>
        </div>
      </div>

      <VRButton />

      <Canvas style={{ height: '100vh' }}>
        <XR>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          
          <Controllers />
          <Hands />
          
          <CollaborativeScene
            geologicalData={geologicalData}
            connectedUsers={connectedUsers}
            sharedAnnotations={sharedAnnotations}
            onUserInteraction={onUserInteraction}
            onCreateAnnotation={createSharedAnnotation}
            onPositionUpdate={broadcastUserPosition}
            gestureRecognizer={gestureRecognizer}
          />
          
          <VRCollaborativeInterface
            connectedUsers={connectedUsers}
            voiceChatActive={voiceChatActive}
            annotationCount={sharedAnnotations.length}
          />
        </XR>
      </Canvas>
    </div>
  );
};

interface CollaborativeSceneProps {
  geologicalData: any[];
  connectedUsers: Map<string, VRAvatar>;
  sharedAnnotations: SharedAnnotation[];
  onUserInteraction?: (interaction: UserInteraction) => void;
  onCreateAnnotation: (position: THREE.Vector3, text: string, type?: 'note' | 'measurement' | 'highlight') => void;
  onPositionUpdate: (
    position: THREE.Vector3,
    rotation: THREE.Euler,
    headPosition: THREE.Vector3,
    leftHandPosition: THREE.Vector3,
    rightHandPosition: THREE.Vector3
  ) => void;
  gestureRecognizer: any;
}

const CollaborativeScene: React.FC<CollaborativeSceneProps> = ({
  geologicalData,
  connectedUsers,
  sharedAnnotations,
  onUserInteraction,
  onCreateAnnotation,
  onPositionUpdate,
  gestureRecognizer
}) => {
  const { camera, gl } = useThree();
  const lastPositionRef = useRef<THREE.Vector3>(new THREE.Vector3());
  const lastRotationRef = useRef<THREE.Euler>(new THREE.Euler());

  useFrame(() => {
    // Track user position and broadcast changes
    const currentPosition = camera.position.clone();
    const currentRotation = camera.rotation.clone();
    
    if (currentPosition.distanceTo(lastPositionRef.current) > 0.01 ||
        Math.abs(currentRotation.x - lastRotationRef.current.x) > 0.01 ||
        Math.abs(currentRotation.y - lastRotationRef.current.y) > 0.01) {
      
      // Get VR controller positions if available
      const session = gl.xr.getSession();
      let leftHandPos = new THREE.Vector3(-0.3, 1.2, -0.3);
      let rightHandPos = new THREE.Vector3(0.3, 1.2, -0.3);
      
      if (session) {
        const leftController = gl.xr.getController(0);
        const rightController = gl.xr.getController(1);
        
        if (leftController) {
          leftHandPos.setFromMatrixPosition(leftController.matrixWorld);
        }
        if (rightController) {
          rightHandPos.setFromMatrixPosition(rightController.matrixWorld);
        }
      }
      
      onPositionUpdate(
        currentPosition,
        currentRotation,
        currentPosition.clone().add(new THREE.Vector3(0, 0.1, 0)), // Head offset
        leftHandPos,
        rightHandPos
      );
      
      lastPositionRef.current.copy(currentPosition);
      lastRotationRef.current.copy(currentRotation);
    }
    
    // Gesture recognition
    if (gestureRecognizer && session) {
      const handPositions = [
        new THREE.Vector3().setFromMatrixPosition(gl.xr.getController(0)?.matrixWorld || new THREE.Matrix4()),
        new THREE.Vector3().setFromMatrixPosition(gl.xr.getController(1)?.matrixWorld || new THREE.Matrix4())
      ];
      
      const gesture = gestureRecognizer.recognize(handPositions);
      if (gesture) {
        onUserInteraction?.({
          userId: 'current_user',
          type: 'gesture',
          position: camera.position.clone(),
          data: { gesture },
          timestamp: Date.now()
        });
      }
    }
  });

  return (
    <>
      {/* Render geological data */}
      {geologicalData.map((data, index) => (
        <GeologicalDataVisualization
          key={index}
          data={data}
          onInteraction={onUserInteraction}
        />
      ))}
      
      {/* Render other users' avatars */}
      {Array.from(connectedUsers.values()).map(user => (
        <UserAvatar
          key={user.userId}
          avatar={user}
        />
      ))}
      
      {/* Render shared annotations */}
      {sharedAnnotations.map(annotation => (
        <SharedAnnotationComponent
          key={annotation.id}
          annotation={annotation}
        />
      ))}
      
      {/* Interaction handlers */}
      <VRInteractionSystem
        onCreateAnnotation={onCreateAnnotation}
        onUserInteraction={onUserInteraction}
      />
    </>
  );
};

const UserAvatar: React.FC<{ avatar: VRAvatar }> = ({ avatar }) => {
  const meshRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.copy(avatar.position);
      meshRef.current.rotation.copy(avatar.rotation);
    }
  });

  return (
    <group ref={meshRef}>
      {/* Head */}
      <Sphere args={[0.1, 8, 8]} position={avatar.headPosition}>
        <meshStandardMaterial color={avatar.color} />
      </Sphere>
      
      {/* Body */}
      <Box args={[0.3, 0.6, 0.15]} position={[0, 1.2, 0]}>
        <meshStandardMaterial color={avatar.color} transparent opacity={0.7} />
      </Box>
      
      {/* Hands */}
      <Sphere args={[0.05, 6, 6]} position={avatar.leftHandPosition}>
        <meshStandardMaterial color={avatar.color} />
      </Sphere>
      <Sphere args={[0.05, 6, 6]} position={avatar.rightHandPosition}>
        <meshStandardMaterial color={avatar.color} />
      </Sphere>
      
      {/* Name tag */}
      <Text
        position={[0, avatar.headPosition.y + 0.2, 0]}
        fontSize={0.05}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        User {avatar.userId.slice(-4)}
      </Text>
    </group>
  );
};

const SharedAnnotationComponent: React.FC<{ annotation: SharedAnnotation }> = ({
  annotation
}) => {
  return (
    <group position={[annotation.position.x, annotation.position.y, annotation.position.z]}>
      <Sphere args={[0.02, 8, 8]}>
        <meshBasicMaterial color={annotation.color} />
      </Sphere>
      
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.03}
        color="white"
        anchorX="center"
        anchorY="middle"
        maxWidth={1}
      >
        {annotation.text}
      </Text>
      
      <Text
        position={[0, 0.05, 0]}
        fontSize={0.02}
        color="lightgray"
        anchorX="center"
        anchorY="middle"
      >
        {annotation.userName}
      </Text>
    </group>
  );
};

const GeologicalDataVisualization: React.FC<{
  data: any;
  onInteraction?: (interaction: UserInteraction) => void;
}> = ({ data, onInteraction }) => {
  // Simplified geological data visualization for collaborative environment
  return (
    <Box args={[1, 0.5, 1]} position={[data.x || 0, data.y || 0, data.z || 0]}>
      <meshStandardMaterial color={data.color || 0x8B4513} transparent opacity={0.8} />
    </Box>
  );
};

const VRInteractionSystem: React.FC<{
  onCreateAnnotation: (position: THREE.Vector3, text: string, type?: 'note' | 'measurement' | 'highlight') => void;
  onUserInteraction?: (interaction: UserInteraction) => void;
}> = ({ onCreateAnnotation, onUserInteraction }) => {
  // Handle VR controller interactions for creating annotations
  return null; // Implementation would handle controller events
};

const VRCollaborativeInterface: React.FC<{
  connectedUsers: Map<string, VRAvatar>;
  voiceChatActive: boolean;
  annotationCount: number;
}> = ({ connectedUsers, voiceChatActive, annotationCount }) => {
  return (
    <group position={[-2, 2, -1]}>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.1}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Collaborative Session
      </Text>
      
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.06}
        color="lightblue"
        anchorX="center"
        anchorY="middle"
      >
        Users: {connectedUsers.size + 1}
      </Text>
      
      <Text
        position={[0, -0.1, 0]}
        fontSize={0.06}
        color={voiceChatActive ? "lightgreen" : "lightgray"}
        anchorX="center"
        anchorY="middle"
      >
        Voice: {voiceChatActive ? "ON" : "OFF"}
      </Text>
      
      <Text
        position={[0, -0.3, 0]}
        fontSize={0.06}
        color="yellow"
        anchorX="center"
        anchorY="middle"
      >
        Annotations: {annotationCount}
      </Text>
    </group>
  );
};

export default CollaborativeVREnvironment;