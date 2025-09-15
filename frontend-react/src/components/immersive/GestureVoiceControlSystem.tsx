/**
 * Gesture and Voice Control System for Immersive Geological Exploration
 * 
 * This system provides natural interaction methods through hand gestures,
 * voice commands, eye tracking, and body pose recognition for intuitive
 * control of geological visualizations and data analysis.
 */

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

// Gesture recognition interfaces
interface GestureRecognizer {
  id: string;
  name: string;
  type: 'hand' | 'body' | 'eye' | 'face';
  isActive: boolean;
  confidence: number;
  device?: any;
}

interface HandGesture {
  type: 'pinch' | 'grab' | 'point' | 'swipe' | 'rotate' | 'scale' | 'tap' | 'fist' | 'open_palm' | 'thumbs_up';
  hand: 'left' | 'right' | 'both';
  confidence: number;
  position: THREE.Vector3;
  direction?: THREE.Vector3;
  velocity?: THREE.Vector3;
  duration: number;
  landmarks?: HandLandmarks;
}

interface HandLandmarks {
  thumb: THREE.Vector3[];
  index: THREE.Vector3[];
  middle: THREE.Vector3[];
  ring: THREE.Vector3[];
  pinky: THREE.Vector3[];
  palm: THREE.Vector3;
  wrist: THREE.Vector3;
}

interface BodyGesture {
  type: 'wave' | 'nod' | 'shake_head' | 'shrug' | 'lean' | 'step' | 'turn' | 'crouch' | 'reach';
  confidence: number;
  bodyParts: BodyPart[];
  duration: number;
  intensity: number;
}

interface BodyPart {
  name: string;
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  velocity: THREE.Vector3;
  confidence: number;
}

interface EyeGaze {
  direction: THREE.Vector3;
  origin: THREE.Vector3;
  confidence: number;
  fixationPoint?: THREE.Vector3;
  blinkState: 'open' | 'closed' | 'blinking';
  pupilDilation: number;
}

interface VoiceCommand {
  command: string;
  intent: string;
  entities: VoiceEntity[];
  confidence: number;
  language: string;
  speaker?: string;
}

interface VoiceEntity {
  type: 'object' | 'action' | 'parameter' | 'value' | 'location';
  value: string;
  confidence: number;
}

interface GestureAction {
  id: string;
  name: string;
  description: string;
  gestureType: string;
  parameters: ActionParameter[];
  callback: (params: any) => void;
  enabled: boolean;
  contextual: boolean;
}

interface ActionParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'vector3' | 'object';
  required: boolean;
  defaultValue?: any;
}

interface ControlContext {
  mode: 'exploration' | 'analysis' | 'annotation' | 'measurement' | 'collaboration';
  selectedObjects: THREE.Object3D[];
  activeTools: string[];
  permissions: string[];
  constraints: ControlConstraints;
}

interface ControlConstraints {
  maxGestureDistance: number;
  minConfidenceThreshold: number;
  gestureTimeout: number;
  voiceTimeout: number;
  multimodalEnabled: boolean;
  accessibilityMode: boolean;
}

export class GestureVoiceControlSystem {
  private gestureRecognizers: Map<string, GestureRecognizer> = new Map();
  private voiceRecognition: any = null;
  // private _handTracker: any = null;
  // private _bodyTracker: any = null;
  // private _eyeTracker: any = null;
  
  private gestureActions: Map<string, GestureAction> = new Map();
  private voiceCommands: Map<string, VoiceCommand> = new Map();
  private context: ControlContext;
  
  private isListening: boolean = false;
  // private _isTrackingGestures: boolean = false;
  // private _lastGesture: HandGesture | null = null;
  private _lastVoiceCommand: VoiceCommand | null = null;
  
  private callbacks: {
    onGestureRecognized?: (gesture: HandGesture | BodyGesture) => void;
    onVoiceCommand?: (command: VoiceCommand) => void;
    onEyeGaze?: (gaze: EyeGaze) => void;
    onActionTriggered?: (action: GestureAction, params: any) => void;
  } = {};

  constructor(initialContext: Partial<ControlContext> = {}) {
    this.context = {
      mode: 'exploration',
      selectedObjects: [],
      activeTools: [],
      permissions: ['view', 'interact', 'annotate'],
      constraints: {
        maxGestureDistance: 2.0,
        minConfidenceThreshold: 0.7,
        gestureTimeout: 5000,
        voiceTimeout: 10000,
        multimodalEnabled: true,
        accessibilityMode: false
      },
      ...initialContext
    };

    this.initializeGestureRecognition();
    this.initializeVoiceRecognition();
    this.setupDefaultActions();
  }

  // Initialize gesture recognition systems
  private async initializeGestureRecognition(): Promise<void> {
    // MediaPipe Hands
    await this.initializeMediaPipeHands();
    
    // MediaPipe Pose
    await this.initializeMediaPipePose();
    
    // WebXR Hand Tracking
    await this.initializeWebXRHandTracking();
    
    // Eye tracking (if available)
    await this.initializeEyeTracking();
  }

  private async initializeMediaPipeHands(): Promise<void> {
    try {
      // This would integrate with MediaPipe Hands
      const handTracker = await this.loadMediaPipeHands();
      
      if (handTracker) {
        const recognizer: GestureRecognizer = {
          id: 'mediapipe-hands',
          name: 'MediaPipe Hand Tracking',
          type: 'hand',
          isActive: true,
          confidence: 0.9,
          device: handTracker
        };
        
        this.gestureRecognizers.set('mediapipe-hands', recognizer);
        // this._handTracker = handTracker;
        
        // Set up hand tracking callbacks
        handTracker.onResults = (results: any) => {
          this.processHandTrackingResults(results);
        };
      }
    } catch (error) {
      console.warn('MediaPipe Hands not available:', error);
    }
  }

  private async initializeMediaPipePose(): Promise<void> {
    try {
      // This would integrate with MediaPipe Pose
      const poseTracker = await this.loadMediaPipePose();
      
      if (poseTracker) {
        const recognizer: GestureRecognizer = {
          id: 'mediapipe-pose',
          name: 'MediaPipe Pose Tracking',
          type: 'body',
          isActive: true,
          confidence: 0.8,
          device: poseTracker
        };
        
        this.gestureRecognizers.set('mediapipe-pose', recognizer);
        // this._bodyTracker = poseTracker;
        
        // Set up pose tracking callbacks
        poseTracker.onResults = (results: any) => {
          this.processBodyTrackingResults(results);
        };
      }
    } catch (error) {
      console.warn('MediaPipe Pose not available:', error);
    }
  }

  private async initializeWebXRHandTracking(): Promise<void> {
    if ('xr' in navigator) {
      try {
        const session = await (navigator as any).xr.requestSession('immersive-vr', {
          optionalFeatures: ['hand-tracking']
        });
        
        if (session) {
          const recognizer: GestureRecognizer = {
            id: 'webxr-hands',
            name: 'WebXR Hand Tracking',
            type: 'hand',
            isActive: true,
            confidence: 0.95,
            device: session
          };
          
          this.gestureRecognizers.set('webxr-hands', recognizer);
        }
      } catch (error) {
        console.warn('WebXR hand tracking not available:', error);
      }
    }
  }

  private async initializeEyeTracking(): Promise<void> {
    try {
      // This would integrate with WebGazer or similar eye tracking library
      const eyeTracker = await this.loadEyeTracker();
      
      if (eyeTracker) {
        const recognizer: GestureRecognizer = {
          id: 'eye-tracker',
          name: 'Eye Tracking',
          type: 'eye',
          isActive: true,
          confidence: 0.7,
          device: eyeTracker
        };
        
        this.gestureRecognizers.set('eye-tracker', recognizer);
        // this._eyeTracker = eyeTracker;
        
        // Set up eye tracking callbacks
        eyeTracker.setGazeListener((data: any) => {
          this.processEyeTrackingResults(data);
        });
      }
    } catch (error) {
      console.warn('Eye tracking not available:', error);
    }
  }

  // Initialize voice recognition
  private initializeVoiceRecognition(): void {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      this.voiceRecognition = new SpeechRecognition();
      
      this.voiceRecognition.continuous = true;
      this.voiceRecognition.interimResults = true;
      this.voiceRecognition.lang = 'en-US';
      this.voiceRecognition.maxAlternatives = 3;

      this.voiceRecognition.onresult = (event: any) => {
        this.processVoiceResults(event);
      };

      this.voiceRecognition.onerror = (event: any) => {
        console.warn('Voice recognition error:', event.error);
      };

      this.voiceRecognition.onend = () => {
        if (this.isListening) {
          // Restart if still supposed to be listening
          this.voiceRecognition?.start();
        }
      };
    }
  }

  // Process tracking results
  private processHandTrackingResults(results: any): void {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) return;

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      const handedness = results.multiHandedness[i].label.toLowerCase();
      
      const gesture = this.recognizeHandGesture(landmarks, handedness);
      if (gesture && gesture.confidence > this.context.constraints.minConfidenceThreshold) {
        this.handleGestureRecognized(gesture);
      }
    }
  }

  private processBodyTrackingResults(results: any): void {
    if (!results.poseLandmarks) return;

    const bodyGesture = this.recognizeBodyGesture(results.poseLandmarks);
    if (bodyGesture && bodyGesture.confidence > this.context.constraints.minConfidenceThreshold) {
      this.handleBodyGestureRecognized(bodyGesture);
    }
  }

  private processEyeTrackingResults(data: any): void {
    const eyeGaze: EyeGaze = {
      direction: new THREE.Vector3(data.x, data.y, -1).normalize(),
      origin: new THREE.Vector3(0, 0, 0),
      confidence: data.confidence || 0.5,
      blinkState: data.blink ? 'blinking' : 'open',
      pupilDilation: data.pupilSize || 0.5
    };

    this.callbacks.onEyeGaze?.(eyeGaze);
  }

  private processVoiceResults(event: any): void {
    const results = event.results;
    const lastResult = results[results.length - 1];
    
    if (lastResult.isFinal) {
      const transcript = lastResult[0].transcript.toLowerCase().trim();
      const confidence = lastResult[0].confidence;
      
      const command = this.parseVoiceCommand(transcript, confidence);
      if (command) {
        this.handleVoiceCommand(command);
      }
    }
  }

  // Gesture recognition algorithms
  private recognizeHandGesture(landmarks: any[], handedness: string): HandGesture | null {
    const handLandmarks = this.convertToHandLandmarks(landmarks);
    
    // Pinch gesture detection
    const thumbTip = handLandmarks.thumb[handLandmarks.thumb.length - 1];
    const indexTip = handLandmarks.index[handLandmarks.index.length - 1];
    const pinchDistance = thumbTip.distanceTo(indexTip);
    
    if (pinchDistance < 0.05) {
      return {
        type: 'pinch',
        hand: handedness as 'left' | 'right',
        confidence: Math.max(0, 1 - pinchDistance * 20),
        position: thumbTip.clone().lerp(indexTip, 0.5),
        duration: 0,
        landmarks: handLandmarks
      };
    }

    // Point gesture detection
    const indexExtended = this.isFingerExtended(handLandmarks.index);
    const othersFolded = !this.isFingerExtended(handLandmarks.middle) &&
                        !this.isFingerExtended(handLandmarks.ring) &&
                        !this.isFingerExtended(handLandmarks.pinky);
    
    if (indexExtended && othersFolded) {
      const direction = indexTip.clone().sub(handLandmarks.index[0]).normalize();
      
      return {
        type: 'point',
        hand: handedness as 'left' | 'right',
        confidence: 0.9,
        position: indexTip,
        direction: direction,
        duration: 0,
        landmarks: handLandmarks
      };
    }

    // Grab gesture detection
    const allFingersFolded = !this.isFingerExtended(handLandmarks.index) &&
                            !this.isFingerExtended(handLandmarks.middle) &&
                            !this.isFingerExtended(handLandmarks.ring) &&
                            !this.isFingerExtended(handLandmarks.pinky);
    
    if (allFingersFolded) {
      return {
        type: 'grab',
        hand: handedness as 'left' | 'right',
        confidence: 0.8,
        position: handLandmarks.palm,
        duration: 0,
        landmarks: handLandmarks
      };
    }

    // Open palm gesture detection
    const allFingersExtended = this.isFingerExtended(handLandmarks.thumb) &&
                              this.isFingerExtended(handLandmarks.index) &&
                              this.isFingerExtended(handLandmarks.middle) &&
                              this.isFingerExtended(handLandmarks.ring) &&
                              this.isFingerExtended(handLandmarks.pinky);
    
    if (allFingersExtended) {
      return {
        type: 'open_palm',
        hand: handedness as 'left' | 'right',
        confidence: 0.9,
        position: handLandmarks.palm,
        duration: 0,
        landmarks: handLandmarks
      };
    }

    return null;
  }

  private recognizeBodyGesture(landmarks: any[]): BodyGesture | null {
    // Convert landmarks to body parts
    const bodyParts = this.convertToBodyParts(landmarks);
    
    // Wave gesture detection
    const rightWrist = bodyParts.find(p => p.name === 'right_wrist');
    const rightElbow = bodyParts.find(p => p.name === 'right_elbow');
    
    if (rightWrist && rightElbow) {
      const armRaised = rightWrist.position.y > rightElbow.position.y;
      const lateralMovement = Math.abs(rightWrist.velocity.x) > 0.5;
      
      if (armRaised && lateralMovement) {
        return {
          type: 'wave',
          confidence: 0.8,
          bodyParts: [rightWrist, rightElbow],
          duration: 0,
          intensity: Math.abs(rightWrist.velocity.x)
        };
      }
    }

    // Nod gesture detection
    const nose = bodyParts.find(p => p.name === 'nose');
    if (nose && Math.abs(nose.velocity.y) > 0.3) {
      return {
        type: 'nod',
        confidence: 0.7,
        bodyParts: [nose],
        duration: 0,
        intensity: Math.abs(nose.velocity.y)
      };
    }

    return null;
  }

  // Voice command parsing
  private parseVoiceCommand(transcript: string, confidence: number): VoiceCommand | null {
    // Define geological voice command patterns
    const commandPatterns = [
      {
        pattern: /show (.*?) (volcano|mountain|fault|layer)/i,
        intent: 'show_geological_feature',
        entities: ['object', 'feature_type']
      },
      {
        pattern: /zoom (in|out)( on (.*))?/i,
        intent: 'zoom',
        entities: ['action', 'target']
      },
      {
        pattern: /rotate (.*?) (left|right|up|down)/i,
        intent: 'rotate',
        entities: ['object', 'direction']
      },
      {
        pattern: /measure (distance|area|volume) (from|between) (.*?) (to|and) (.*)/i,
        intent: 'measure',
        entities: ['measurement_type', 'start_object', 'end_object']
      },
      {
        pattern: /take (sample|core) (at|from) (.*)/i,
        intent: 'sample',
        entities: ['sample_type', 'location']
      },
      {
        pattern: /analyze (.*?) (composition|structure|age|properties)/i,
        intent: 'analyze',
        entities: ['object', 'analysis_type']
      },
      {
        pattern: /hide (all )?layers?/i,
        intent: 'hide_layers',
        entities: ['scope']
      },
      {
        pattern: /show (seismic|thermal|magnetic|gravity) data/i,
        intent: 'show_data_layer',
        entities: ['data_type']
      },
      {
        pattern: /start (collaboration|recording|presentation)/i,
        intent: 'start_session',
        entities: ['session_type']
      },
      {
        pattern: /go to (.*)/i,
        intent: 'navigate',
        entities: ['location']
      }
    ];

    for (const pattern of commandPatterns) {
      const match = transcript.match(pattern.pattern);
      if (match) {
        const entities: VoiceEntity[] = [];
        
        // Extract entities based on capture groups
        for (let i = 1; i < match.length; i++) {
          if (match[i] && pattern.entities[i - 1]) {
            entities.push({
              type: pattern.entities[i - 1] as any,
              value: match[i].trim(),
              confidence: confidence
            });
          }
        }

        return {
          command: transcript,
          intent: pattern.intent,
          entities: entities,
          confidence: confidence,
          language: 'en-US'
        };
      }
    }

    return null;
  }

  // Setup default gesture and voice actions
  private setupDefaultActions(): void {
    // Pinch to select
    this.gestureActions.set('pinch_select', {
      id: 'pinch_select',
      name: 'Pinch to Select',
      description: 'Select objects by pinching',
      gestureType: 'pinch',
      parameters: [
        { name: 'position', type: 'vector3', required: true }
      ],
      callback: (params) => this.handlePinchSelect(params),
      enabled: true,
      contextual: false
    });

    // Point to highlight
    this.gestureActions.set('point_highlight', {
      id: 'point_highlight',
      name: 'Point to Highlight',
      description: 'Highlight objects by pointing',
      gestureType: 'point',
      parameters: [
        { name: 'position', type: 'vector3', required: true },
        { name: 'direction', type: 'vector3', required: true }
      ],
      callback: (params) => this.handlePointHighlight(params),
      enabled: true,
      contextual: false
    });

    // Grab to manipulate
    this.gestureActions.set('grab_manipulate', {
      id: 'grab_manipulate',
      name: 'Grab to Manipulate',
      description: 'Manipulate objects by grabbing',
      gestureType: 'grab',
      parameters: [
        { name: 'position', type: 'vector3', required: true }
      ],
      callback: (params) => this.handleGrabManipulate(params),
      enabled: true,
      contextual: true
    });

    // Voice commands
    this.setupVoiceActions();
  }

  private setupVoiceActions(): void {
    const voiceActions = [
      {
        intent: 'show_geological_feature',
        callback: (command: VoiceCommand) => this.handleShowFeature(command)
      },
      {
        intent: 'zoom',
        callback: (command: VoiceCommand) => this.handleZoomCommand(command)
      },
      {
        intent: 'rotate',
        callback: (command: VoiceCommand) => this.handleRotateCommand(command)
      },
      {
        intent: 'measure',
        callback: (command: VoiceCommand) => this.handleMeasureCommand(command)
      },
      {
        intent: 'sample',
        callback: (command: VoiceCommand) => this.handleSampleCommand(command)
      },
      {
        intent: 'analyze',
        callback: (command: VoiceCommand) => this.handleAnalyzeCommand(command)
      },
      {
        intent: 'hide_layers',
        callback: (command: VoiceCommand) => this.handleHideLayersCommand(command)
      },
      {
        intent: 'show_data_layer',
        callback: (command: VoiceCommand) => this.handleShowDataLayerCommand(command)
      },
      {
        intent: 'start_session',
        callback: (command: VoiceCommand) => this.handleStartSessionCommand(command)
      },
      {
        intent: 'navigate',
        callback: (command: VoiceCommand) => this.handleNavigateCommand(command)
      }
    ];

    voiceActions.forEach(action => {
      this.voiceCommands.set(action.intent, {
        command: '',
        intent: action.intent,
        entities: [],
        confidence: 0,
        language: 'en-US'
      });
    });
  }

  // Gesture handlers
  private handleGestureRecognized(gesture: HandGesture): void {
    // this.lastGesture = gesture;
    this.callbacks.onGestureRecognized?.(gesture);

    // Find matching action
    const action = this.gestureActions.get(`${gesture.type}_${this.getActionSuffix(gesture.type)}`);
    if (action && action.enabled) {
      const params = this.extractGestureParameters(gesture, action);
      action.callback(params);
      this.callbacks.onActionTriggered?.(action, params);
    }
  }

  private handleBodyGestureRecognized(gesture: BodyGesture): void {
    this.callbacks.onGestureRecognized?.(gesture);
    
    // Handle body gestures (wave for greeting, nod for confirmation, etc.)
    switch (gesture.type) {
      case 'wave':
        this.handleWaveGesture(gesture);
        break;
      case 'nod':
        this.handleNodGesture(gesture);
        break;
      // Add more body gesture handlers
    }
  }

  private handleVoiceCommand(command: VoiceCommand): void {
    this._lastVoiceCommand = command;
    this.callbacks.onVoiceCommand?.(command);

    // Find matching voice action
    const voiceAction = this.voiceCommands.get(command.intent);
    if (voiceAction) {
      this.executeVoiceAction(command);
    }
  }

  // Action implementations
  private handlePinchSelect(params: any): void {
    // Implement object selection logic
    console.log('Pinch select at:', params.position);
  }

  private handlePointHighlight(params: any): void {
    // Implement object highlighting logic
    console.log('Point highlight:', params.position, params.direction);
  }

  private handleGrabManipulate(params: any): void {
    // Implement object manipulation logic
    console.log('Grab manipulate at:', params.position);
  }

  private handleWaveGesture(_gesture: BodyGesture): void {
    // Handle wave gesture (e.g., greeting in collaborative mode)
    console.log('Wave gesture detected');
  }

  private handleNodGesture(_gesture: BodyGesture): void {
    // Handle nod gesture (e.g., confirmation)
    console.log('Nod gesture detected');
  }

  // Voice command handlers
  private handleShowFeature(command: VoiceCommand): void {
    const objectEntity = command.entities.find(e => e.type === 'object');
    const featureEntity = command.entities.find(e => e.type === 'parameter');
    
    if (objectEntity && featureEntity) {
      console.log(`Show ${featureEntity.value}: ${objectEntity.value}`);
    }
  }

  private handleZoomCommand(command: VoiceCommand): void {
    const actionEntity = command.entities.find(e => e.type === 'action');
    const targetEntity = command.entities.find(e => e.type === 'object');
    
    console.log(`Zoom ${actionEntity?.value} on ${targetEntity?.value || 'current view'}`);
  }

  private handleRotateCommand(command: VoiceCommand): void {
    const objectEntity = command.entities.find(e => e.type === 'object');
    const directionEntity = command.entities.find(e => e.type === 'parameter');
    
    console.log(`Rotate ${objectEntity?.value} ${directionEntity?.value}`);
  }

  private handleMeasureCommand(command: VoiceCommand): void {
    // const typeEntity = command.entities.find(e => e.type === 'parameter');
    // const startEntity = command.entities.find(e => e.type === 'start_object');
    // const endEntity = command.entities.find(e => e.type === 'end_object');
    
    console.log(`Measure command received: ${command.transcript}`);
  }

  private handleSampleCommand(command: VoiceCommand): void {
    // const typeEntity = command.entities.find(e => e.type === 'sample_type');
    // const locationEntity = command.entities.find(e => e.type === 'location');
    
    console.log(`Sample command received: ${command.transcript}`);
  }

  private handleAnalyzeCommand(command: VoiceCommand): void {
    // const objectEntity = command.entities.find(e => e.type === 'object');
    // const analysisEntity = command.entities.find(e => e.type === 'analysis_type');
    
    console.log(`Analyze command received: ${command.transcript}`);
  }

  private handleHideLayersCommand(command: VoiceCommand): void {
    // const scopeEntity = command.entities.find(e => e.type === 'scope');
    console.log(`Hide layers command received: ${command.transcript}`);
  }

  private handleShowDataLayerCommand(command: VoiceCommand): void {
    // const dataTypeEntity = command.entities.find(e => e.type === 'data_type');
    console.log(`Show data layer command received: ${command.transcript}`);
  }

  private handleStartSessionCommand(command: VoiceCommand): void {
    // const sessionTypeEntity = command.entities.find(e => e.type === 'session_type');
    console.log(`Start session command received: ${command.transcript}`);
  }

  private handleNavigateCommand(command: VoiceCommand): void {
    // const locationEntity = command.entities.find(e => e.type === 'location');
    console.log(`Navigate command received: ${command.transcript}`);
  }

  // Utility methods
  private convertToHandLandmarks(landmarks: any[]): HandLandmarks {
    // Convert MediaPipe landmarks to our format
    return {
      thumb: landmarks.slice(1, 5).map(l => new THREE.Vector3(l.x, l.y, l.z)),
      index: landmarks.slice(5, 9).map(l => new THREE.Vector3(l.x, l.y, l.z)),
      middle: landmarks.slice(9, 13).map(l => new THREE.Vector3(l.x, l.y, l.z)),
      ring: landmarks.slice(13, 17).map(l => new THREE.Vector3(l.x, l.y, l.z)),
      pinky: landmarks.slice(17, 21).map(l => new THREE.Vector3(l.x, l.y, l.z)),
      palm: new THREE.Vector3(landmarks[0].x, landmarks[0].y, landmarks[0].z),
      wrist: new THREE.Vector3(landmarks[0].x, landmarks[0].y, landmarks[0].z)
    };
  }

  private convertToBodyParts(landmarks: any[]): BodyPart[] {
    const bodyParts: BodyPart[] = [];
    
    // Convert MediaPipe pose landmarks to body parts
    const landmarkNames = [
      'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
      'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
      'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ];

    landmarks.forEach((landmark: any, index: number) => {
      if (index < landmarkNames.length) {
        bodyParts.push({
          name: landmarkNames[index],
          position: new THREE.Vector3(landmark.x, landmark.y, landmark.z),
          rotation: new THREE.Quaternion(),
          velocity: new THREE.Vector3(),
          confidence: landmark.visibility || 0.5
        });
      }
    });

    return bodyParts;
  }

  private isFingerExtended(fingerLandmarks: THREE.Vector3[]): boolean {
    if (fingerLandmarks.length < 4) return false;
    
    const tip = fingerLandmarks[3];
    const pip = fingerLandmarks[2];
    const mcp = fingerLandmarks[1];
    
    // Check if tip is further from palm than PIP joint
    const tipDistance = tip.distanceTo(fingerLandmarks[0]);
    const pipDistance = pip.distanceTo(fingerLandmarks[0]);
    
    return tipDistance > pipDistance;
  }

  private getActionSuffix(gestureType: string): string {
    const suffixes: Record<string, string> = {
      'pinch': 'select',
      'point': 'highlight',
      'grab': 'manipulate',
      'swipe': 'navigate',
      'rotate': 'rotate',
      'scale': 'scale'
    };
    
    return suffixes[gestureType] || 'action';
  }

  private extractGestureParameters(gesture: HandGesture, action: GestureAction): any {
    const params: any = {};
    
    action.parameters.forEach(param => {
      switch (param.name) {
        case 'position':
          params.position = gesture.position;
          break;
        case 'direction':
          params.direction = gesture.direction;
          break;
        case 'velocity':
          params.velocity = gesture.velocity;
          break;
        case 'hand':
          params.hand = gesture.hand;
          break;
        case 'confidence':
          params.confidence = gesture.confidence;
          break;
        default:
          params[param.name] = param.defaultValue;
      }
    });
    
    return params;
  }

  private executeVoiceAction(command: VoiceCommand): void {
    switch (command.intent) {
      case 'show_geological_feature':
        this.handleShowFeature(command);
        break;
      case 'zoom':
        this.handleZoomCommand(command);
        break;
      case 'rotate':
        this.handleRotateCommand(command);
        break;
      case 'measure':
        this.handleMeasureCommand(command);
        break;
      case 'sample':
        this.handleSampleCommand(command);
        break;
      case 'analyze':
        this.handleAnalyzeCommand(command);
        break;
      case 'hide_layers':
        this.handleHideLayersCommand(command);
        break;
      case 'show_data_layer':
        this.handleShowDataLayerCommand(command);
        break;
      case 'start_session':
        this.handleStartSessionCommand(command);
        break;
      case 'navigate':
        this.handleNavigateCommand(command);
        break;
    }
  }

  // Public API methods
  public startGestureTracking(): void {
    this.isTrackingGestures = true;
    
    // Start all available gesture recognizers
    for (const recognizer of this.gestureRecognizers.values()) {
      if (recognizer.device && recognizer.device.start) {
        recognizer.device.start();
      }
    }
  }

  public stopGestureTracking(): void {
    this.isTrackingGestures = false;
    
    // Stop all gesture recognizers
    for (const recognizer of this.gestureRecognizers.values()) {
      if (recognizer.device && recognizer.device.stop) {
        recognizer.device.stop();
      }
    }
  }

  public startVoiceListening(): void {
    if (this.voiceRecognition && !this.isListening) {
      this.isListening = true;
      this.voiceRecognition.start();
    }
  }

  public stopVoiceListening(): void {
    if (this.voiceRecognition && this.isListening) {
      this.isListening = false;
      this.voiceRecognition.stop();
    }
  }

  public updateContext(newContext: Partial<ControlContext>): void {
    this.context = { ...this.context, ...newContext };
  }

  public setCallbacks(callbacks: Partial<typeof this.callbacks>): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  public addGestureAction(action: GestureAction): void {
    this.gestureActions.set(action.id, action);
  }

  public removeGestureAction(actionId: string): void {
    this.gestureActions.delete(actionId);
  }

  public getAvailableRecognizers(): GestureRecognizer[] {
    return Array.from(this.gestureRecognizers.values());
  }

  public cleanup(): void {
    this.stopGestureTracking();
    this.stopVoiceListening();
    
    // Cleanup resources
    this.gestureRecognizers.clear();
    this.gestureActions.clear();
    this.voiceCommands.clear();
  }

  // Helper methods for device loading (would integrate with actual SDKs)
  private async loadMediaPipeHands(): Promise<any> {
    // This would load and initialize MediaPipe Hands
    return null;
  }

  private async loadMediaPipePose(): Promise<any> {
    // This would load and initialize MediaPipe Pose
    return null;
  }

  private async loadEyeTracker(): Promise<any> {
    // This would load and initialize eye tracking library
    return null;
  }
}

// React component wrapper
export const GestureVoiceControlProvider: React.FC<{
  children: React.ReactNode;
  initialContext?: Partial<ControlContext>;
  onSystemReady?: (system: GestureVoiceControlSystem) => void;
}> = ({ children, initialContext, onSystemReady }) => {
  const systemRef = useRef<GestureVoiceControlSystem | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const initializeSystem = async () => {
      systemRef.current = new GestureVoiceControlSystem(initialContext);
      setIsReady(true);
      
      if (onSystemReady) {
        onSystemReady(systemRef.current);
      }
    };

    initializeSystem();

    return () => {
      if (systemRef.current) {
        systemRef.current.cleanup();
      }
    };
  }, [initialContext, onSystemReady]);

  return (
    <>
      {children}
    </>
  );
};

export default GestureVoiceControlSystem;