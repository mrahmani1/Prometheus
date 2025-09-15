import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
// import * as THREE from 'three'; // Used for gesture calculations
import { 
  GestureCommand, 
  VoiceCommand, 
  HandTracking as _HandTracking, 
  SpeechRecognition as _SpeechRecognition 
} from '../../types/immersive';
import { 
  updateGestureRecognition, 
  updateVoiceRecognition,
  executeCommand
} from '../../store/slices/immersiveSlice';

interface GestureVoiceControlProps {
  enabled?: boolean;
  gestureEnabled?: boolean;
  voiceEnabled?: boolean;
  onCommand?: (command: GestureCommand | VoiceCommand) => void;
  sensitivity?: number;
  language?: string;
}

interface HandLandmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

interface GesturePattern {
  name: string;
  description: string;
  landmarks: number[];
  threshold: number;
  action: string;
}

const GestureVoiceControl: React.FC<GestureVoiceControlProps> = ({
  enabled = true,
  gestureEnabled = true,
  voiceEnabled = true,
  onCommand,
  sensitivity = 0.8,
  language = 'en-US'
}) => {
  const dispatch = useDispatch();
  // const _gestureState = useSelector((state: any) => state.immersive.gestureRecognition);
  const voiceState = useSelector((state: any) => state.immersive.voiceRecognition);
  
  const [_isInitialized, setIsInitialized] = useState(false);
  const [_handTracker, _setHandTracker] = useState<any>(null);
  const [speechRecognizer, setSpeechRecognizer] = useState<any>(null);
  const [currentGesture, setCurrentGesture] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [_confidenceThreshold] = useState(0.7);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gestureHistoryRef = useRef<string[]>([]);
  const lastGestureTimeRef = useRef<number>(0);

  // Predefined gesture patterns for geological exploration
  const gesturePatterns: GesturePattern[] = [
    {
      name: 'point',
      description: 'Point at geological feature',
      landmarks: [8], // Index finger tip
      threshold: 0.8,
      action: 'select_feature'
    },
    {
      name: 'grab',
      description: 'Grab and manipulate object',
      landmarks: [4, 8, 12, 16, 20], // All fingertips
      threshold: 0.7,
      action: 'grab_object'
    },
    {
      name: 'pinch',
      description: 'Pinch to zoom or measure',
      landmarks: [4, 8], // Thumb and index finger
      threshold: 0.9,
      action: 'zoom_measure'
    },
    {
      name: 'open_palm',
      description: 'Open palm to show menu',
      landmarks: [0, 4, 8, 12, 16, 20], // Palm and all fingertips
      threshold: 0.6,
      action: 'show_menu'
    },
    {
      name: 'fist',
      description: 'Closed fist for navigation',
      landmarks: [4, 8, 12, 16, 20], // All fingertips (closed)
      threshold: 0.8,
      action: 'navigate'
    },
    {
      name: 'peace_sign',
      description: 'Two fingers for dual selection',
      landmarks: [8, 12], // Index and middle finger
      threshold: 0.85,
      action: 'dual_select'
    }
  ];

  // Voice commands for geological exploration
  const voiceCommands = [
    // Navigation commands
    { phrase: 'zoom in', action: 'zoom_in', confidence: 0.8 },
    { phrase: 'zoom out', action: 'zoom_out', confidence: 0.8 },
    { phrase: 'rotate left', action: 'rotate_left', confidence: 0.8 },
    { phrase: 'rotate right', action: 'rotate_right', confidence: 0.8 },
    { phrase: 'move up', action: 'move_up', confidence: 0.8 },
    { phrase: 'move down', action: 'move_down', confidence: 0.8 },
    { phrase: 'reset view', action: 'reset_view', confidence: 0.9 },
    
    // Geological analysis commands
    { phrase: 'analyze rock', action: 'analyze_rock', confidence: 0.9 },
    { phrase: 'measure distance', action: 'measure_distance', confidence: 0.9 },
    { phrase: 'show layers', action: 'show_layers', confidence: 0.8 },
    { phrase: 'hide layers', action: 'hide_layers', confidence: 0.8 },
    { phrase: 'show faults', action: 'show_faults', confidence: 0.9 },
    { phrase: 'show minerals', action: 'show_minerals', confidence: 0.9 },
    
    // Data collection commands
    { phrase: 'take sample', action: 'take_sample', confidence: 0.9 },
    { phrase: 'record observation', action: 'record_observation', confidence: 0.9 },
    { phrase: 'save location', action: 'save_location', confidence: 0.9 },
    { phrase: 'add annotation', action: 'add_annotation', confidence: 0.9 },
    
    // System commands
    { phrase: 'start recording', action: 'start_recording', confidence: 0.9 },
    { phrase: 'stop recording', action: 'stop_recording', confidence: 0.9 },
    { phrase: 'export data', action: 'export_data', confidence: 0.9 },
    { phrase: 'help', action: 'show_help', confidence: 0.7 },
    { phrase: 'menu', action: 'show_menu', confidence: 0.7 }
  ];

  useEffect(() => {
    if (enabled) {
      initializeControls();
    }

    return () => {
      cleanup();
    };
  }, [enabled, gestureEnabled, voiceEnabled]);

  const initializeControls = useCallback(async () => {
    try {
      if (gestureEnabled) {
        await initializeGestureRecognition();
      }
      
      if (voiceEnabled) {
        await initializeVoiceRecognition();
      }
      
      setIsInitialized(true);
    } catch (error) {
      console.error('Failed to initialize gesture/voice controls:', error);
    }
  }, [gestureEnabled, voiceEnabled]);

  const initializeGestureRecognition = useCallback(async () => {
    try {
      // Initialize camera for hand tracking
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          startHandTracking();
        };
      }
    } catch (error) {
      console.error('Camera access failed:', error);
    }
  }, []);

  const startHandTracking = useCallback(() => {
    // This would integrate with MediaPipe Hands or similar library
    // For now, we'll simulate hand tracking
    const trackHands = () => {
      if (videoRef.current && canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const video = videoRef.current;
        
        if (ctx && video.readyState === 4) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          // Simulate hand landmark detection
          const landmarks = simulateHandLandmarks();
          if (landmarks) {
            const gesture = recognizeGesture(landmarks);
            if (gesture) {
              handleGestureRecognized(gesture);
            }
          }
        }
      }
      
      requestAnimationFrame(trackHands);
    };
    
    trackHands();
  }, []);

  const simulateHandLandmarks = (): HandLandmark[] | null => {
    // This is a simulation - in real implementation, this would come from MediaPipe
    // Return null to indicate no hand detected, or array of landmarks
    if (Math.random() > 0.7) { // Simulate intermittent hand detection
      return Array.from({ length: 21 }, (_, _i) => ({
        x: Math.random(),
        y: Math.random(),
        z: Math.random() * 0.1,
        visibility: Math.random() > 0.2 ? 1 : 0
      }));
    }
    return null;
  };

  const recognizeGesture = (landmarks: HandLandmark[]): string | null => {
    for (const pattern of gesturePatterns) {
      const confidence = calculateGestureConfidence(landmarks, pattern);
      if (confidence > pattern.threshold * sensitivity) {
        return pattern.name;
      }
    }
    return null;
  };

  const calculateGestureConfidence = (landmarks: HandLandmark[], pattern: GesturePattern): number => {
    // Simplified gesture recognition logic
    // In real implementation, this would use proper geometric analysis
    
    let confidence = 0;
    let validLandmarks = 0;
    
    for (const landmarkIndex of pattern.landmarks) {
      if (landmarks[landmarkIndex] && landmarks[landmarkIndex].visibility > 0.5) {
        validLandmarks++;
        
        // Add specific gesture logic here
        switch (pattern.name) {
          case 'point':
            // Check if index finger is extended
            if (landmarkIndex === 8) {
              const fingerTip = landmarks[8];
              const fingerBase = landmarks[5];
              if (fingerTip.y < fingerBase.y) { // Finger pointing up/forward
                confidence += 0.3;
              }
            }
            break;
            
          case 'fist':
            // Check if all fingers are closed
            if (landmarks[landmarkIndex].y > landmarks[0].y) { // Fingertip below palm
              confidence += 0.2;
            }
            break;
            
          case 'open_palm':
            // Check if all fingers are extended
            if (landmarks[landmarkIndex].y < landmarks[0].y) { // Fingertip above palm
              confidence += 0.15;
            }
            break;
            
          case 'pinch':
            // Check distance between thumb and index finger
            if (landmarkIndex === 4 || landmarkIndex === 8) {
              const thumb = landmarks[4];
              const index = landmarks[8];
              const distance = Math.sqrt(
                Math.pow(thumb.x - index.x, 2) + 
                Math.pow(thumb.y - index.y, 2)
              );
              if (distance < 0.05) { // Close together
                confidence += 0.4;
              }
            }
            break;
        }
      }
    }
    
    return validLandmarks > 0 ? confidence : 0;
  };

  const handleGestureRecognized = useCallback((gesture: string) => {
    const now = Date.now();
    
    // Debounce gesture recognition
    if (now - lastGestureTimeRef.current < 500) {
      return;
    }
    
    // Add to gesture history for pattern analysis
    gestureHistoryRef.current.push(gesture);
    if (gestureHistoryRef.current.length > 10) {
      gestureHistoryRef.current.shift();
    }
    
    setCurrentGesture(gesture);
    lastGestureTimeRef.current = now;
    
    const pattern = gesturePatterns.find(p => p.name === gesture);
    if (pattern) {
      const command: GestureCommand = {
        type: 'gesture',
        gesture,
        action: pattern.action,
        confidence: 0.8, // Would be calculated from recognition
        timestamp: now,
        description: pattern.description
      };
      
      dispatch(updateGestureRecognition({
        currentGesture: gesture,
        // confidence: command.confidence, // Property doesn't exist on GestureRecognition
        // timestamp: now // Property doesn't exist on GestureRecognition
      }));
      
      dispatch(executeCommand(command));
      onCommand?.(command);
    }
  }, [dispatch, onCommand]);

  const initializeVoiceRecognition = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported');
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language;
    recognition.maxAlternatives = 3;
    
    recognition.onstart = () => {
      setIsListening(true);
      dispatch(updateVoiceRecognition({ listening: true }));
    };
    
    recognition.onend = () => {
      setIsListening(false);
      dispatch(updateVoiceRecognition({ listening: false }));
      
      // Restart recognition if still enabled
      if (voiceEnabled && enabled) {
        setTimeout(() => recognition.start(), 1000);
      }
    };
    
    recognition.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        const confidence = event.results[i][0].confidence;
        
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
          handleVoiceCommand(transcript, confidence);
        } else {
          interimTranscript += transcript;
        }
      }
      
      dispatch(updateVoiceRecognition({
        transcript: finalTranscript || interimTranscript,
        confidence: event.results[event.resultIndex][0].confidence
      }));
    };
    
    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      dispatch(updateVoiceRecognition({ error: event.error }));
    };
    
    setSpeechRecognizer(recognition);
    recognition.start();
  }, [language, voiceEnabled, enabled, dispatch]);

  const handleVoiceCommand = useCallback((transcript: string, confidence: number) => {
    const normalizedTranscript = transcript.toLowerCase().trim();
    
    for (const voiceCommand of voiceCommands) {
      if (normalizedTranscript.includes(voiceCommand.phrase) && 
          confidence >= voiceCommand.confidence * sensitivity) {
        
        const command: VoiceCommand = {
          type: 'voice',
          phrase: voiceCommand.phrase,
          action: voiceCommand.action,
          confidence,
          timestamp: Date.now(),
          transcript: normalizedTranscript
        };
        
        dispatch(executeCommand(command));
        onCommand?.(command);
        break;
      }
    }
  }, [sensitivity, dispatch, onCommand]);

  const startVoiceRecognition = useCallback(() => {
    if (speechRecognizer && !isListening) {
      speechRecognizer.start();
    }
  }, [speechRecognizer, isListening]);

  const stopVoiceRecognition = useCallback(() => {
    if (speechRecognizer && isListening) {
      speechRecognizer.stop();
    }
  }, [speechRecognizer, isListening]);

  const cleanup = useCallback(() => {
    if (speechRecognizer) {
      speechRecognizer.stop();
    }
    
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  }, [speechRecognizer]);

  return (
    <div className="gesture-voice-control">
      <div className="control-status">
        <h3>Gesture & Voice Control</h3>
        <div className="status-indicators">
          <div className={`status-item ${enabled ? 'enabled' : 'disabled'}`}>
            System: {enabled ? 'Enabled' : 'Disabled'}
          </div>
          <div className={`status-item ${gestureEnabled ? 'enabled' : 'disabled'}`}>
            Gestures: {gestureEnabled ? 'Active' : 'Inactive'}
          </div>
          <div className={`status-item ${voiceEnabled ? 'enabled' : 'disabled'}`}>
            Voice: {voiceEnabled ? 'Active' : 'Inactive'}
          </div>
          <div className={`status-item ${isListening ? 'listening' : 'idle'}`}>
            Listening: {isListening ? 'Yes' : 'No'}
          </div>
        </div>
      </div>

      <div className="current-recognition">
        {currentGesture && (
          <div className="gesture-display">
            <strong>Current Gesture:</strong> {currentGesture}
          </div>
        )}
        
        {voiceState.transcript && (
          <div className="voice-display">
            <strong>Voice Input:</strong> {voiceState.transcript}
            <span className="confidence">
              (Confidence: {(voiceState.confidence * 100).toFixed(0)}%)
            </span>
          </div>
        )}
      </div>

      <div className="control-buttons">
        <button
          onClick={startVoiceRecognition}
          disabled={!voiceEnabled || isListening}
        >
          Start Voice
        </button>
        <button
          onClick={stopVoiceRecognition}
          disabled={!voiceEnabled || !isListening}
        >
          Stop Voice
        </button>
      </div>

      <div className="available-commands">
        <div className="gesture-commands">
          <h4>Available Gestures</h4>
          <ul>
            {gesturePatterns.map(pattern => (
              <li key={pattern.name}>
                <strong>{pattern.name}:</strong> {pattern.description}
              </li>
            ))}
          </ul>
        </div>
        
        <div className="voice-commands">
          <h4>Voice Commands</h4>
          <ul>
            {voiceCommands.slice(0, 10).map((cmd, index) => (
              <li key={index}>
                <strong>"{cmd.phrase}":</strong> {cmd.action.replace('_', ' ')}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Hidden video and canvas for gesture recognition */}
      <video
        ref={videoRef}
        style={{ display: 'none' }}
        autoPlay
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        style={{ display: 'none' }}
      />

      <style>{`
        .gesture-voice-control {
          padding: 20px;
          background: rgba(0, 0, 0, 0.9);
          color: white;
          border-radius: 8px;
          margin: 10px;
          max-width: 600px;
        }
        
        .status-indicators {
          display: flex;
          gap: 15px;
          margin: 10px 0;
          flex-wrap: wrap;
        }
        
        .status-item {
          padding: 5px 10px;
          border-radius: 4px;
          font-size: 0.9em;
        }
        
        .status-item.enabled {
          background: #28a745;
        }
        
        .status-item.disabled {
          background: #6c757d;
        }
        
        .status-item.listening {
          background: #007bff;
          animation: pulse 1s infinite;
        }
        
        .status-item.idle {
          background: #6c757d;
        }
        
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
        
        .current-recognition {
          margin: 15px 0;
          padding: 10px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
        }
        
        .gesture-display, .voice-display {
          margin: 5px 0;
        }
        
        .confidence {
          color: #ffc107;
          font-size: 0.8em;
          margin-left: 10px;
        }
        
        .control-buttons {
          margin: 15px 0;
        }
        
        .control-buttons button {
          margin: 5px;
          padding: 8px 16px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .control-buttons button:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }
        
        .available-commands {
          display: flex;
          gap: 20px;
          margin-top: 20px;
        }
        
        .gesture-commands, .voice-commands {
          flex: 1;
        }
        
        .available-commands h4 {
          margin-bottom: 10px;
          color: #ffc107;
        }
        
        .available-commands ul {
          list-style: none;
          padding: 0;
        }
        
        .available-commands li {
          margin: 5px 0;
          padding: 5px;
          background: rgba(255, 255, 255, 0.05);
          border-radius: 3px;
          font-size: 0.9em;
        }
      `}</style>
    </div>
  );
};

export default GestureVoiceControl;