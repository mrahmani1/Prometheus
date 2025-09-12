/**
 * Accessibility VR/AR System for Inclusive Geological Exploration
 * 
 * This system provides comprehensive accessibility features for users with
 * visual, auditory, motor, and cognitive disabilities, ensuring equal access
 * to immersive geological visualization and analysis capabilities.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';

// Accessibility interfaces
interface AccessibilityProfile {
  userId: string;
  visualImpairment: VisualAccessibilityNeeds;
  auditoryImpairment: AuditoryAccessibilityNeeds;
  motorImpairment: MotorAccessibilityNeeds;
  cognitiveNeeds: CognitiveAccessibilityNeeds;
  preferences: AccessibilityPreferences;
}

interface VisualAccessibilityNeeds {
  blindness: boolean;
  lowVision: boolean;
  colorBlindness: ColorBlindnessType;
  contrastSensitivity: number; // 0-1
  motionSensitivity: boolean;
  lightSensitivity: boolean;
  fieldOfViewRestriction: boolean;
}

interface AuditoryAccessibilityNeeds {
  deafness: boolean;
  hardOfHearing: boolean;
  tinnitus: boolean;
  auditoryProcessingDisorder: boolean;
  preferredAudioFrequency: [number, number];
  spatialAudioNeeds: boolean;
}

interface MotorAccessibilityNeeds {
  limitedMobility: boolean;
  tremor: boolean;
  paralysis: string[]; // affected body parts
  reducedDexterity: boolean;
  fatigueIssues: boolean;
  alternativeInputNeeds: boolean;
}

interface CognitiveAccessibilityNeeds {
  memoryImpairment: boolean;
  attentionDeficit: boolean;
  processingSpeed: number; // 0-1, slower to faster
  languageProcessing: boolean;
  executiveFunctionNeeds: boolean;
  sensoryOverloadSensitivity: boolean;
}

interface AccessibilityPreferences {
  audioDescriptions: boolean;
  hapticFeedback: boolean;
  simplifiedInterface: boolean;
  highContrast: boolean;
  largeText: boolean;
  reducedMotion: boolean;
  voiceNavigation: boolean;
  screenReader: boolean;
  captioning: boolean;
  signLanguage: boolean;
}

type ColorBlindnessType = 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia' | 'achromatopsia';

interface AccessibilityFeature {
  id: string;
  name: string;
  description: string;
  category: 'visual' | 'auditory' | 'motor' | 'cognitive';
  enabled: boolean;
  settings: Record<string, any>;
}

interface AudioDescription {
  id: string;
  objectId: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  duration: number;
  spatialPosition?: THREE.Vector3;
}

interface HapticGuidance {
  id: string;
  type: 'navigation' | 'object_outline' | 'texture' | 'warning' | 'confirmation';
  pattern: number[];
  intensity: number;
  duration: number;
  position?: THREE.Vector3;
}

interface VoiceGuidance {
  id: string;
  text: string;
  priority: number;
  interrupt: boolean;
  spatialAudio: boolean;
  position?: THREE.Vector3;
}

interface VisualEnhancement {
  highContrast: boolean;
  colorAdjustment: ColorAdjustment;
  magnification: number;
  edgeEnhancement: boolean;
  motionReduction: boolean;
  flickerReduction: boolean;
}

interface ColorAdjustment {
  type: ColorBlindnessType;
  saturation: number;
  brightness: number;
  contrast: number;
  hueShift: number;
}

interface NavigationAssistance {
  audioBeacons: boolean;
  hapticPathfinding: boolean;
  voiceDirections: boolean;
  simplifiedControls: boolean;
  autoNavigation: boolean;
  obstacleWarnings: boolean;
}

export class AccessibilityVRSystem {
  private profile: AccessibilityProfile | null = null;
  private features: Map<string, AccessibilityFeature> = new Map();
  private audioContext: AudioContext | null = null;
  private speechSynthesis: SpeechSynthesis | null = null;
  private screenReader: ScreenReader | null = null;
  private hapticSystem: any = null;
  
  private audioDescriptions: Map<string, AudioDescription> = new Map();
  private voiceQueue: VoiceGuidance[] = [];
  private currentAudio: HTMLAudioElement | null = null;
  private isDescribing: boolean = false;
  
  private visualEnhancements: VisualEnhancement = {
    highContrast: false,
    colorAdjustment: {
      type: 'none',
      saturation: 1.0,
      brightness: 1.0,
      contrast: 1.0,
      hueShift: 0.0
    },
    magnification: 1.0,
    edgeEnhancement: false,
    motionReduction: false,
    flickerReduction: false
  };

  private navigationAssistance: NavigationAssistance = {
    audioBeacons: false,
    hapticPathfinding: false,
    voiceDirections: false,
    simplifiedControls: false,
    autoNavigation: false,
    obstacleWarnings: false
  };

  private callbacks: {
    onProfileUpdated?: (profile: AccessibilityProfile) => void;
    onFeatureToggled?: (feature: AccessibilityFeature) => void;
    onNavigationAssist?: (assistance: any) => void;
  } = {};

  constructor() {
    this.initializeAccessibilityFeatures();
    this.initializeAudioSystems();
    this.initializeScreenReader();
  }

  // Initialize accessibility features
  private initializeAccessibilityFeatures(): void {
    const features: AccessibilityFeature[] = [
      {
        id: 'audio_descriptions',
        name: 'Audio Descriptions',
        description: 'Detailed audio descriptions of geological features and interactions',
        category: 'visual',
        enabled: false,
        settings: {
          verbosity: 'medium',
          speed: 1.0,
          voice: 'default',
          spatialAudio: true
        }
      },
      {
        id: 'haptic_navigation',
        name: 'Haptic Navigation',
        description: 'Tactile feedback for navigation and object identification',
        category: 'visual',
        enabled: false,
        settings: {
          intensity: 0.7,
          pathfinding: true,
          objectOutlines: true,
          textureMapping: true
        }
      },
      {
        id: 'voice_control',
        name: 'Voice Control',
        description: 'Complete voice-based navigation and interaction',
        category: 'motor',
        enabled: false,
        settings: {
          sensitivity: 0.8,
          language: 'en-US',
          continuousListening: true,
          confirmationRequired: false
        }
      },
      {
        id: 'high_contrast',
        name: 'High Contrast Mode',
        description: 'Enhanced visual contrast for better visibility',
        category: 'visual',
        enabled: false,
        settings: {
          contrastRatio: 4.5,
          edgeEnhancement: true,
          colorInversion: false
        }
      },
      {
        id: 'color_correction',
        name: 'Color Correction',
        description: 'Color adjustments for color blindness and visual impairments',
        category: 'visual',
        enabled: false,
        settings: {
          type: 'deuteranopia',
          intensity: 1.0,
          preserveScientificAccuracy: true
        }
      },
      {
        id: 'motion_reduction',
        name: 'Motion Reduction',
        description: 'Reduced motion and animation for motion sensitivity',
        category: 'visual',
        enabled: false,
        settings: {
          animationSpeed: 0.5,
          autoRotation: false,
          smoothTransitions: true
        }
      },
      {
        id: 'simplified_interface',
        name: 'Simplified Interface',
        description: 'Streamlined interface for cognitive accessibility',
        category: 'cognitive',
        enabled: false,
        settings: {
          reducedOptions: true,
          largerButtons: true,
          clearLabels: true,
          stepByStep: true
        }
      },
      {
        id: 'captions',
        name: 'Captions and Subtitles',
        description: 'Text captions for all audio content',
        category: 'auditory',
        enabled: false,
        settings: {
          fontSize: 'medium',
          background: true,
          position: 'bottom',
          language: 'en-US'
        }
      },
      {
        id: 'sign_language',
        name: 'Sign Language Interpretation',
        description: 'Sign language avatar for audio content',
        category: 'auditory',
        enabled: false,
        settings: {
          avatarStyle: 'realistic',
          position: 'corner',
          size: 'medium',
          language: 'ASL'
        }
      },
      {
        id: 'cognitive_assistance',
        name: 'Cognitive Assistance',
        description: 'Memory aids and processing support',
        category: 'cognitive',
        enabled: false,
        settings: {
          reminders: true,
          progressTracking: true,
          contextualHelp: true,
          pauseOptions: true
        }
      }
    ];

    features.forEach(feature => {
      this.features.set(feature.id, feature);
    });
  }

  private initializeAudioSystems(): void {
    try {
      this.audioContext = new AudioContext();
      this.speechSynthesis = window.speechSynthesis;
    } catch (error) {
      console.warn('Audio systems not available:', error);
    }
  }

  private initializeScreenReader(): void {
    this.screenReader = new ScreenReader();
  }

  // Profile management
  public setAccessibilityProfile(profile: AccessibilityProfile): void {
    this.profile = profile;
    this.applyProfileSettings();
    this.callbacks.onProfileUpdated?.(profile);
  }

  private applyProfileSettings(): void {
    if (!this.profile) return;

    // Apply visual accessibility settings
    if (this.profile.visualImpairment.blindness || this.profile.visualImpairment.lowVision) {
      this.enableFeature('audio_descriptions');
      this.enableFeature('haptic_navigation');
      this.enableFeature('voice_control');
    }

    if (this.profile.visualImpairment.colorBlindness !== 'none') {
      this.enableFeature('color_correction');
      this.updateFeatureSettings('color_correction', {
        type: this.profile.visualImpairment.colorBlindness
      });
    }

    if (this.profile.visualImpairment.contrastSensitivity < 0.5) {
      this.enableFeature('high_contrast');
    }

    if (this.profile.visualImpairment.motionSensitivity) {
      this.enableFeature('motion_reduction');
    }

    // Apply auditory accessibility settings
    if (this.profile.auditoryImpairment.deafness || this.profile.auditoryImpairment.hardOfHearing) {
      this.enableFeature('captions');
      this.enableFeature('haptic_navigation');
      
      if (this.profile.preferences.signLanguage) {
        this.enableFeature('sign_language');
      }
    }

    // Apply motor accessibility settings
    if (this.profile.motorImpairment.limitedMobility || this.profile.motorImpairment.reducedDexterity) {
      this.enableFeature('voice_control');
      this.enableFeature('simplified_interface');
    }

    // Apply cognitive accessibility settings
    if (this.profile.cognitiveNeeds.memoryImpairment || this.profile.cognitiveNeeds.attentionDeficit) {
      this.enableFeature('simplified_interface');
      this.enableFeature('cognitive_assistance');
    }

    if (this.profile.cognitiveNeeds.sensoryOverloadSensitivity) {
      this.enableFeature('motion_reduction');
      this.enableFeature('simplified_interface');
    }
  }

  // Feature management
  public enableFeature(featureId: string): boolean {
    const feature = this.features.get(featureId);
    if (!feature) return false;

    feature.enabled = true;
    this.applyFeature(feature);
    this.callbacks.onFeatureToggled?.(feature);
    return true;
  }

  public disableFeature(featureId: string): boolean {
    const feature = this.features.get(featureId);
    if (!feature) return false;

    feature.enabled = false;
    this.removeFeature(feature);
    this.callbacks.onFeatureToggled?.(feature);
    return true;
  }

  public updateFeatureSettings(featureId: string, settings: Record<string, any>): boolean {
    const feature = this.features.get(featureId);
    if (!feature) return false;

    feature.settings = { ...feature.settings, ...settings };
    
    if (feature.enabled) {
      this.applyFeature(feature);
    }
    
    return true;
  }

  private applyFeature(feature: AccessibilityFeature): void {
    switch (feature.id) {
      case 'audio_descriptions':
        this.setupAudioDescriptions(feature.settings);
        break;
      case 'haptic_navigation':
        this.setupHapticNavigation(feature.settings);
        break;
      case 'voice_control':
        this.setupVoiceControl(feature.settings);
        break;
      case 'high_contrast':
        this.applyHighContrast(feature.settings);
        break;
      case 'color_correction':
        this.applyColorCorrection(feature.settings);
        break;
      case 'motion_reduction':
        this.applyMotionReduction(feature.settings);
        break;
      case 'simplified_interface':
        this.applySimplifiedInterface(feature.settings);
        break;
      case 'captions':
        this.setupCaptions(feature.settings);
        break;
      case 'sign_language':
        this.setupSignLanguage(feature.settings);
        break;
      case 'cognitive_assistance':
        this.setupCognitiveAssistance(feature.settings);
        break;
    }
  }

  private removeFeature(feature: AccessibilityFeature): void {
    switch (feature.id) {
      case 'audio_descriptions':
        this.disableAudioDescriptions();
        break;
      case 'haptic_navigation':
        this.disableHapticNavigation();
        break;
      case 'voice_control':
        this.disableVoiceControl();
        break;
      case 'high_contrast':
        this.removeHighContrast();
        break;
      case 'color_correction':
        this.removeColorCorrection();
        break;
      case 'motion_reduction':
        this.removeMotionReduction();
        break;
      case 'simplified_interface':
        this.removeSimplifiedInterface();
        break;
      case 'captions':
        this.disableCaptions();
        break;
      case 'sign_language':
        this.disableSignLanguage();
        break;
      case 'cognitive_assistance':
        this.disableCognitiveAssistance();
        break;
    }
  }

  // Audio descriptions
  private setupAudioDescriptions(settings: any): void {
    // Initialize text-to-speech for geological descriptions
    if (this.speechSynthesis) {
      this.speechSynthesis.cancel(); // Clear any existing speech
    }
  }

  public addAudioDescription(description: AudioDescription): void {
    this.audioDescriptions.set(description.id, description);
    
    if (this.features.get('audio_descriptions')?.enabled) {
      this.playAudioDescription(description);
    }
  }

  private playAudioDescription(description: AudioDescription): void {
    if (!this.speechSynthesis || this.isDescribing) return;

    const utterance = new SpeechSynthesisUtterance(description.description);
    
    // Configure speech settings
    const settings = this.features.get('audio_descriptions')?.settings;
    if (settings) {
      utterance.rate = settings.speed || 1.0;
      utterance.volume = 0.8;
      utterance.pitch = 1.0;
    }

    // Spatial audio positioning
    if (description.spatialPosition && this.audioContext) {
      this.applySpatialAudio(utterance, description.spatialPosition);
    }

    utterance.onstart = () => {
      this.isDescribing = true;
    };

    utterance.onend = () => {
      this.isDescribing = false;
      this.processVoiceQueue();
    };

    this.speechSynthesis.speak(utterance);
  }

  private applySpatialAudio(utterance: SpeechSynthesisUtterance, position: THREE.Vector3): void {
    // Implement spatial audio positioning for screen reader
    // This would require more advanced audio processing
  }

  private disableAudioDescriptions(): void {
    if (this.speechSynthesis) {
      this.speechSynthesis.cancel();
    }
    this.isDescribing = false;
  }

  // Haptic navigation
  private setupHapticNavigation(settings: any): void {
    // Initialize haptic feedback for navigation
    this.navigationAssistance.hapticPathfinding = settings.pathfinding || false;
  }

  public provideHapticGuidance(guidance: HapticGuidance): void {
    if (!this.features.get('haptic_navigation')?.enabled) return;

    // Trigger haptic feedback based on guidance type
    switch (guidance.type) {
      case 'navigation':
        this.triggerNavigationHaptics(guidance);
        break;
      case 'object_outline':
        this.triggerObjectOutlineHaptics(guidance);
        break;
      case 'texture':
        this.triggerTextureHaptics(guidance);
        break;
      case 'warning':
        this.triggerWarningHaptics(guidance);
        break;
      case 'confirmation':
        this.triggerConfirmationHaptics(guidance);
        break;
    }
  }

  private triggerNavigationHaptics(guidance: HapticGuidance): void {
    // Implement directional haptic feedback for navigation
    if ('vibrate' in navigator) {
      navigator.vibrate(guidance.pattern);
    }
  }

  private triggerObjectOutlineHaptics(guidance: HapticGuidance): void {
    // Implement object outline haptic feedback
    if ('vibrate' in navigator) {
      const outlinePattern = [100, 50, 100, 50, 100]; // Outline pattern
      navigator.vibrate(outlinePattern);
    }
  }

  private triggerTextureHaptics(guidance: HapticGuidance): void {
    // Implement texture-based haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate(guidance.pattern);
    }
  }

  private triggerWarningHaptics(guidance: HapticGuidance): void {
    // Implement warning haptic pattern
    if ('vibrate' in navigator) {
      const warningPattern = [200, 100, 200, 100, 200];
      navigator.vibrate(warningPattern);
    }
  }

  private triggerConfirmationHaptics(guidance: HapticGuidance): void {
    // Implement confirmation haptic pattern
    if ('vibrate' in navigator) {
      const confirmationPattern = [50, 50, 50];
      navigator.vibrate(confirmationPattern);
    }
  }

  private disableHapticNavigation(): void {
    this.navigationAssistance.hapticPathfinding = false;
  }

  // Voice control
  private setupVoiceControl(settings: any): void {
    this.navigationAssistance.voiceDirections = true;
  }

  private disableVoiceControl(): void {
    this.navigationAssistance.voiceDirections = false;
  }

  // Visual enhancements
  private applyHighContrast(settings: any): void {
    this.visualEnhancements.highContrast = true;
    this.visualEnhancements.colorAdjustment.contrast = settings.contrastRatio || 4.5;
    this.visualEnhancements.edgeEnhancement = settings.edgeEnhancement || false;
  }

  private removeHighContrast(): void {
    this.visualEnhancements.highContrast = false;
    this.visualEnhancements.colorAdjustment.contrast = 1.0;
    this.visualEnhancements.edgeEnhancement = false;
  }

  private applyColorCorrection(settings: any): void {
    this.visualEnhancements.colorAdjustment.type = settings.type || 'none';
    
    // Apply color correction matrices based on type
    switch (settings.type) {
      case 'protanopia':
        this.applyProtanopiaCorrection();
        break;
      case 'deuteranopia':
        this.applyDeuteranopiaCorrection();
        break;
      case 'tritanopia':
        this.applyTritanopiaCorrection();
        break;
      case 'achromatopsia':
        this.applyAchromatopsiaCorrection();
        break;
    }
  }

  private applyProtanopiaCorrection(): void {
    // Implement protanopia color correction
    // This would involve shader modifications for the 3D scene
  }

  private applyDeuteranopiaCorrection(): void {
    // Implement deuteranopia color correction
  }

  private applyTritanopiaCorrection(): void {
    // Implement tritanopia color correction
  }

  private applyAchromatopsiaCorrection(): void {
    // Implement achromatopsia color correction
  }

  private removeColorCorrection(): void {
    this.visualEnhancements.colorAdjustment.type = 'none';
  }

  private applyMotionReduction(settings: any): void {
    this.visualEnhancements.motionReduction = true;
    // Reduce animation speeds and disable auto-rotation
  }

  private removeMotionReduction(): void {
    this.visualEnhancements.motionReduction = false;
  }

  // Interface simplification
  private applySimplifiedInterface(settings: any): void {
    // Implement simplified interface logic
  }

  private removeSimplifiedInterface(): void {
    // Remove simplified interface
  }

  // Captions and subtitles
  private setupCaptions(settings: any): void {
    // Initialize caption system
  }

  private disableCaptions(): void {
    // Disable caption system
  }

  // Sign language
  private setupSignLanguage(settings: any): void {
    // Initialize sign language avatar
  }

  private disableSignLanguage(): void {
    // Disable sign language avatar
  }

  // Cognitive assistance
  private setupCognitiveAssistance(settings: any): void {
    // Initialize cognitive assistance features
  }

  private disableCognitiveAssistance(): void {
    // Disable cognitive assistance
  }

  // Voice guidance
  public addVoiceGuidance(guidance: VoiceGuidance): void {
    if (guidance.interrupt) {
      this.voiceQueue.unshift(guidance);
      if (this.speechSynthesis) {
        this.speechSynthesis.cancel();
      }
    } else {
      this.voiceQueue.push(guidance);
    }
    
    this.processVoiceQueue();
  }

  private processVoiceQueue(): void {
    if (this.voiceQueue.length === 0 || this.isDescribing) return;

    const guidance = this.voiceQueue.shift();
    if (guidance && this.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(guidance.text);
      
      utterance.onstart = () => {
        this.isDescribing = true;
      };

      utterance.onend = () => {
        this.isDescribing = false;
        this.processVoiceQueue();
      };

      this.speechSynthesis.speak(utterance);
    }
  }

  // Navigation assistance
  public provideNavigationAssistance(type: string, data: any): void {
    switch (type) {
      case 'obstacle_warning':
        this.handleObstacleWarning(data);
        break;
      case 'path_guidance':
        this.handlePathGuidance(data);
        break;
      case 'destination_reached':
        this.handleDestinationReached(data);
        break;
      case 'orientation_help':
        this.handleOrientationHelp(data);
        break;
    }
  }

  private handleObstacleWarning(data: any): void {
    // Provide obstacle warning through multiple modalities
    this.addVoiceGuidance({
      id: 'obstacle_warning',
      text: `Obstacle detected ${data.direction}`,
      priority: 10,
      interrupt: true,
      spatialAudio: true,
      position: data.position
    });

    this.provideHapticGuidance({
      id: 'obstacle_haptic',
      type: 'warning',
      pattern: [200, 100, 200],
      intensity: 0.8,
      duration: 600
    });
  }

  private handlePathGuidance(data: any): void {
    // Provide path guidance
    this.addVoiceGuidance({
      id: 'path_guidance',
      text: `Turn ${data.direction} in ${data.distance} meters`,
      priority: 5,
      interrupt: false,
      spatialAudio: false
    });
  }

  private handleDestinationReached(data: any): void {
    // Announce destination reached
    this.addVoiceGuidance({
      id: 'destination_reached',
      text: `You have reached ${data.destination}`,
      priority: 8,
      interrupt: false,
      spatialAudio: false
    });

    this.provideHapticGuidance({
      id: 'destination_haptic',
      type: 'confirmation',
      pattern: [100, 50, 100, 50, 100],
      intensity: 0.6,
      duration: 500
    });
  }

  private handleOrientationHelp(data: any): void {
    // Provide orientation assistance
    this.addVoiceGuidance({
      id: 'orientation_help',
      text: `You are facing ${data.direction}. ${data.landmarks}`,
      priority: 3,
      interrupt: false,
      spatialAudio: false
    });
  }

  // Getters
  public getProfile(): AccessibilityProfile | null {
    return this.profile;
  }

  public getEnabledFeatures(): AccessibilityFeature[] {
    return Array.from(this.features.values()).filter(f => f.enabled);
  }

  public getVisualEnhancements(): VisualEnhancement {
    return this.visualEnhancements;
  }

  public getNavigationAssistance(): NavigationAssistance {
    return this.navigationAssistance;
  }

  // Event callbacks
  public setCallbacks(callbacks: Partial<typeof this.callbacks>): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  // Cleanup
  public cleanup(): void {
    if (this.speechSynthesis) {
      this.speechSynthesis.cancel();
    }
    
    if (this.audioContext) {
      this.audioContext.close();
    }
    
    this.voiceQueue = [];
    this.audioDescriptions.clear();
  }
}

// Screen Reader class for VR/AR environments
class ScreenReader {
  private isActive: boolean = false;
  private focusedElement: any = null;
  private navigationMode: 'object' | 'spatial' | 'semantic' = 'object';

  public activate(): void {
    this.isActive = true;
  }

  public deactivate(): void {
    this.isActive = false;
  }

  public focusElement(element: any): void {
    this.focusedElement = element;
    this.announceElement(element);
  }

  private announceElement(element: any): void {
    // Announce the focused element
    const description = this.generateElementDescription(element);
    // This would integrate with the speech synthesis system
  }

  private generateElementDescription(element: any): string {
    // Generate accessible description for 3D objects
    if (element.userData?.geologicalType) {
      return `${element.userData.geologicalType} at position ${element.position.x.toFixed(1)}, ${element.position.y.toFixed(1)}, ${element.position.z.toFixed(1)}`;
    }
    
    return `3D object at position ${element.position.x.toFixed(1)}, ${element.position.y.toFixed(1)}, ${element.position.z.toFixed(1)}`;
  }

  public navigate(direction: 'next' | 'previous' | 'up' | 'down' | 'left' | 'right'): void {
    // Implement spatial navigation for screen reader
  }
}

// React component wrapper
export const AccessibilityVRProvider: React.FC<{
  children: React.ReactNode;
  profile?: AccessibilityProfile;
  onSystemReady?: (system: AccessibilityVRSystem) => void;
}> = ({ children, profile, onSystemReady }) => {
  const systemRef = useRef<AccessibilityVRSystem | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const initializeSystem = async () => {
      systemRef.current = new AccessibilityVRSystem();
      
      if (profile) {
        systemRef.current.setAccessibilityProfile(profile);
      }
      
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
  }, [profile, onSystemReady]);

  return (
    <>
      {children}
    </>
  );
};

export default AccessibilityVRSystem;