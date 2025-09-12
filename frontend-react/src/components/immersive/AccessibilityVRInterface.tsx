import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { VRButton, XR, Controllers, Hands } from '@react-three/xr';
import { Text, Html, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { useDispatch, useSelector } from 'react-redux';
import { 
  AccessibilitySettings, 
  VoiceNavigation, 
  HighContrastMode,
  MotionSettings,
  CognitiveSupport
} from '../../types/immersive';
import { 
  updateAccessibilitySettings,
  enableVoiceNavigation,
  updateMotionSettings,
  setCognitiveSupport
} from '../../store/slices/immersiveSlice';

interface AccessibilityVRInterfaceProps {
  enableVoiceNavigation?: boolean;
  enableHighContrast?: boolean;
  enableMotionReduction?: boolean;
  enableCognitiveSupport?: boolean;
  enableAudioDescriptions?: boolean;
  fontSize?: 'small' | 'medium' | 'large' | 'extra-large';
  colorBlindnessSupport?: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
  onAccessibilityChange?: (settings: AccessibilitySettings) => void;
}

interface AccessibilityFeature {
  id: string;
  name: string;
  description: string;
  category: 'visual' | 'auditory' | 'motor' | 'cognitive';
  enabled: boolean;
  settings: Record<string, any>;
}

interface VoiceCommand {
  phrase: string;
  action: string;
  description: string;
  category: 'navigation' | 'interaction' | 'settings' | 'help';
}

const AccessibilityVRInterface: React.FC<AccessibilityVRInterfaceProps> = ({
  enableVoiceNavigation = true,
  enableHighContrast = false,
  enableMotionReduction = false,
  enableCognitiveSupport = true,
  enableAudioDescriptions = true,
  fontSize = 'medium',
  colorBlindnessSupport = 'none',
  onAccessibilityChange
}) => {
  const dispatch = useDispatch();
  const accessibilityState = useSelector((state: any) => state.immersive.accessibilitySettings);
  const [currentSettings, setCurrentSettings] = useState<AccessibilitySettings>({
    voiceNavigation: enableVoiceNavigation,
    highContrast: enableHighContrast,
    motionReduction: enableMotionReduction,
    cognitiveSupport: enableCognitiveSupport,
    audioDescriptions: enableAudioDescriptions,
    fontSize,
    colorBlindnessSupport,
    customizations: {}
  });
  const [speechRecognizer, setSpeechRecognizer] = useState<any>(null);
  const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [currentFocus, setCurrentFocus] = useState<string | null>(null);
  const [navigationHistory, setNavigationHistory] = useState<string[]>([]);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);

  // Accessibility features configuration
  const accessibilityFeatures: AccessibilityFeature[] = [
    {
      id: 'voice_navigation',
      name: 'Voice Navigation',
      description: 'Navigate using voice commands',
      category: 'auditory',
      enabled: currentSettings.voiceNavigation,
      settings: {
        language: 'en-US',
        sensitivity: 0.8,
        continuous: true
      }
    },
    {
      id: 'high_contrast',
      name: 'High Contrast Mode',
      description: 'Increase visual contrast for better visibility',
      category: 'visual',
      enabled: currentSettings.highContrast,
      settings: {
        contrastRatio: 4.5,
        backgroundColor: '#000000',
        foregroundColor: '#FFFFFF'
      }
    },
    {
      id: 'motion_reduction',
      name: 'Reduced Motion',
      description: 'Minimize animations and motion effects',
      category: 'motor',
      enabled: currentSettings.motionReduction,
      settings: {
        animationSpeed: 0.5,
        disableAutoRotation: true,
        reduceParallax: true
      }
    },
    {
      id: 'cognitive_support',
      name: 'Cognitive Support',
      description: 'Provide additional guidance and simplification',
      category: 'cognitive',
      enabled: currentSettings.cognitiveSupport,
      settings: {
        showInstructions: true,
        simplifyInterface: true,
        provideHints: true
      }
    },
    {
      id: 'audio_descriptions',
      name: 'Audio Descriptions',
      description: 'Provide spoken descriptions of visual elements',
      category: 'auditory',
      enabled: currentSettings.audioDescriptions,
      settings: {
        verbosity: 'detailed',
        speed: 1.0,
        voice: 'default'
      }
    },
    {
      id: 'large_text',
      name: 'Large Text',
      description: 'Increase text size for better readability',
      category: 'visual',
      enabled: currentSettings.fontSize !== 'medium',
      settings: {
        scaleFactor: getFontScaleFactor(currentSettings.fontSize),
        lineHeight: 1.5,
        letterSpacing: 0.05
      }
    }
  ];

  // Voice commands for accessibility navigation
  const voiceCommands: VoiceCommand[] = [
    // Navigation commands
    { phrase: 'go home', action: 'navigate_home', description: 'Navigate to home screen', category: 'navigation' },
    { phrase: 'go back', action: 'navigate_back', description: 'Go to previous screen', category: 'navigation' },
    { phrase: 'next item', action: 'focus_next', description: 'Focus on next interactive element', category: 'navigation' },
    { phrase: 'previous item', action: 'focus_previous', description: 'Focus on previous interactive element', category: 'navigation' },
    
    // Interaction commands
    { phrase: 'select', action: 'activate_focused', description: 'Activate the currently focused element', category: 'interaction' },
    { phrase: 'open menu', action: 'open_menu', description: 'Open the main menu', category: 'interaction' },
    { phrase: 'close menu', action: 'close_menu', description: 'Close the current menu', category: 'interaction' },
    { phrase: 'describe this', action: 'describe_focused', description: 'Provide audio description of focused element', category: 'interaction' },
    
    // Settings commands
    { phrase: 'increase text size', action: 'increase_font_size', description: 'Make text larger', category: 'settings' },
    { phrase: 'decrease text size', action: 'decrease_font_size', description: 'Make text smaller', category: 'settings' },
    { phrase: 'high contrast on', action: 'enable_high_contrast', description: 'Enable high contrast mode', category: 'settings' },
    { phrase: 'high contrast off', action: 'disable_high_contrast', description: 'Disable high contrast mode', category: 'settings' },
    { phrase: 'reduce motion', action: 'enable_motion_reduction', description: 'Enable reduced motion', category: 'settings' },
    
    // Help commands
    { phrase: 'help', action: 'show_help', description: 'Show help information', category: 'help' },
    { phrase: 'what can I say', action: 'list_commands', description: 'List available voice commands', category: 'help' },
    { phrase: 'where am I', action: 'describe_location', description: 'Describe current location and context', category: 'help' }
  ];

  useEffect(() => {
    initializeAccessibilityFeatures();
    
    if (currentSettings.voiceNavigation) {
      initializeVoiceRecognition();
    }
    
    if (currentSettings.audioDescriptions) {
      initializeSpeechSynthesis();
    }
    
    initializeAudioContext();
  }, [currentSettings]);

  function getFontScaleFactor(fontSize: string): number {
    switch (fontSize) {
      case 'small': return 0.8;
      case 'medium': return 1.0;
      case 'large': return 1.3;
      case 'extra-large': return 1.6;
      default: return 1.0;
    }
  }

  const initializeAccessibilityFeatures = useCallback(() => {
    // Apply accessibility settings to the interface
    dispatch(updateAccessibilitySettings(currentSettings));
    onAccessibilityChange?.(currentSettings);
  }, [currentSettings, dispatch, onAccessibilityChange]);

  const initializeVoiceRecognition = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported');
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;
    
    recognition.onstart = () => {
      setIsListening(true);
      announceToUser('Voice navigation activated');
    };
    
    recognition.onend = () => {
      setIsListening(false);
      // Restart recognition if voice navigation is still enabled
      if (currentSettings.voiceNavigation) {
        setTimeout(() => recognition.start(), 1000);
      }
    };
    
    recognition.onresult = (event: any) => {
      const transcript = event.results[event.resultIndex][0].transcript.toLowerCase().trim();
      handleVoiceCommand(transcript);
    };
    
    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      announceToUser('Voice recognition error. Please try again.');
    };
    
    setSpeechRecognizer(recognition);
    recognition.start();
  }, [currentSettings.voiceNavigation]);

  const initializeSpeechSynthesis = useCallback(() => {
    if ('speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis);
    } else {
      console.warn('Speech synthesis not supported');
    }
  }, []);

  const initializeAudioContext = useCallback(() => {
    try {
      const context = new (window.AudioContext || (window as any).webkitAudioContext)();
      setAudioContext(context);
    } catch (error) {
      console.warn('Audio context not available:', error);
    }
  }, []);

  const handleVoiceCommand = useCallback((transcript: string) => {
    const command = voiceCommands.find(cmd => 
      transcript.includes(cmd.phrase) || 
      cmd.phrase.includes(transcript)
    );
    
    if (command) {
      executeVoiceCommand(command);
      announceToUser(`Executing: ${command.description}`);
    } else {
      announceToUser('Command not recognized. Say "help" for available commands.');
    }
  }, []);

  const executeVoiceCommand = useCallback((command: VoiceCommand) => {
    switch (command.action) {
      case 'navigate_home':
        setCurrentFocus('home');
        setNavigationHistory(prev => [...prev, 'home']);
        break;
        
      case 'navigate_back':
        if (navigationHistory.length > 1) {
          const newHistory = [...navigationHistory];
          newHistory.pop();
          const previousLocation = newHistory[newHistory.length - 1];
          setCurrentFocus(previousLocation);
          setNavigationHistory(newHistory);
        }
        break;
        
      case 'focus_next':
        focusNextElement();
        break;
        
      case 'focus_previous':
        focusPreviousElement();
        break;
        
      case 'activate_focused':
        activateFocusedElement();
        break;
        
      case 'describe_focused':
        describeFocusedElement();
        break;
        
      case 'increase_font_size':
        increaseFontSize();
        break;
        
      case 'decrease_font_size':
        decreaseFontSize();
        break;
        
      case 'enable_high_contrast':
        toggleHighContrast(true);
        break;
        
      case 'disable_high_contrast':
        toggleHighContrast(false);
        break;
        
      case 'enable_motion_reduction':
        toggleMotionReduction(true);
        break;
        
      case 'show_help':
        showHelpInformation();
        break;
        
      case 'list_commands':
        listAvailableCommands();
        break;
        
      case 'describe_location':
        describeCurrentLocation();
        break;
    }
  }, [navigationHistory]);

  const announceToUser = useCallback((message: string) => {
    if (speechSynthesis && currentSettings.audioDescriptions) {
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 0.8;
      speechSynthesis.speak(utterance);
    }
  }, [speechSynthesis, currentSettings.audioDescriptions]);

  const focusNextElement = useCallback(() => {
    // Implementation would cycle through focusable elements
    announceToUser('Moving to next element');
  }, []);

  const focusPreviousElement = useCallback(() => {
    // Implementation would cycle through focusable elements in reverse
    announceToUser('Moving to previous element');
  }, []);

  const activateFocusedElement = useCallback(() => {
    if (currentFocus) {
      announceToUser(`Activating ${currentFocus}`);
      // Implementation would trigger the focused element's action
    }
  }, [currentFocus]);

  const describeFocusedElement = useCallback(() => {
    if (currentFocus) {
      // Implementation would provide detailed description of focused element
      announceToUser(`Currently focused on ${currentFocus}. This is an interactive geological visualization element.`);
    } else {
      announceToUser('No element is currently focused');
    }
  }, [currentFocus]);

  const increaseFontSize = useCallback(() => {
    const sizes = ['small', 'medium', 'large', 'extra-large'];
    const currentIndex = sizes.indexOf(currentSettings.fontSize);
    if (currentIndex < sizes.length - 1) {
      const newSize = sizes[currentIndex + 1] as any;
      setCurrentSettings(prev => ({ ...prev, fontSize: newSize }));
      announceToUser(`Text size increased to ${newSize}`);
    } else {
      announceToUser('Text size is already at maximum');
    }
  }, [currentSettings.fontSize]);

  const decreaseFontSize = useCallback(() => {
    const sizes = ['small', 'medium', 'large', 'extra-large'];
    const currentIndex = sizes.indexOf(currentSettings.fontSize);
    if (currentIndex > 0) {
      const newSize = sizes[currentIndex - 1] as any;
      setCurrentSettings(prev => ({ ...prev, fontSize: newSize }));
      announceToUser(`Text size decreased to ${newSize}`);
    } else {
      announceToUser('Text size is already at minimum');
    }
  }, [currentSettings.fontSize]);

  const toggleHighContrast = useCallback((enabled: boolean) => {
    setCurrentSettings(prev => ({ ...prev, highContrast: enabled }));
    announceToUser(`High contrast mode ${enabled ? 'enabled' : 'disabled'}`);
  }, []);

  const toggleMotionReduction = useCallback((enabled: boolean) => {
    setCurrentSettings(prev => ({ ...prev, motionReduction: enabled }));
    announceToUser(`Motion reduction ${enabled ? 'enabled' : 'disabled'}`);
  }, []);

  const showHelpInformation = useCallback(() => {
    const helpText = `
      Welcome to the accessible VR geological interface. 
      You can navigate using voice commands or controllers. 
      Say "what can I say" to hear available commands. 
      Say "describe this" to get information about the current focus.
    `;
    announceToUser(helpText);
  }, []);

  const listAvailableCommands = useCallback(() => {
    const commandList = voiceCommands
      .slice(0, 8) // Limit to first 8 commands to avoid overwhelming
      .map(cmd => `Say "${cmd.phrase}" to ${cmd.description}`)
      .join('. ');
    
    announceToUser(`Available commands: ${commandList}`);
  }, []);

  const describeCurrentLocation = useCallback(() => {
    const locationDescription = `
      You are in the immersive geological visualization environment. 
      There are interactive rock samples and geological structures around you. 
      Use voice commands or controllers to explore and interact with elements.
    `;
    announceToUser(locationDescription);
  }, []);

  const getAccessibleColors = useCallback(() => {
    if (currentSettings.highContrast) {
      return {
        background: '#000000',
        foreground: '#FFFFFF',
        accent: '#FFFF00',
        interactive: '#00FFFF'
      };
    }
    
    // Color blindness support
    switch (currentSettings.colorBlindnessSupport) {
      case 'protanopia':
        return {
          background: '#1a1a1a',
          foreground: '#ffffff',
          accent: '#0099ff',
          interactive: '#ffaa00'
        };
      case 'deuteranopia':
        return {
          background: '#1a1a1a',
          foreground: '#ffffff',
          accent: '#ff6600',
          interactive: '#0066ff'
        };
      case 'tritanopia':
        return {
          background: '#1a1a1a',
          foreground: '#ffffff',
          accent: '#ff0066',
          interactive: '#00ff66'
        };
      default:
        return {
          background: '#2a2a2a',
          foreground: '#ffffff',
          accent: '#00ff88',
          interactive: '#ff8800'
        };
    }
  }, [currentSettings.highContrast, currentSettings.colorBlindnessSupport]);

  return (
    <div className="accessibility-vr-interface">
      <div className="accessibility-controls">
        <h3>Accessibility Settings</h3>
        <div className="settings-grid">
          {accessibilityFeatures.map(feature => (
            <AccessibilityFeatureControl
              key={feature.id}
              feature={feature}
              onToggle={(enabled) => {
                const settingKey = feature.id.replace('_', '') as keyof AccessibilitySettings;
                setCurrentSettings(prev => ({ ...prev, [settingKey]: enabled }));
              }}
            />
          ))}
        </div>
        
        <div className="voice-status">
          <p>Voice Navigation: {isListening ? 'Listening...' : 'Inactive'}</p>
          <p>Current Focus: {currentFocus || 'None'}</p>
        </div>
      </div>

      <VRButton />

      <Canvas style={{ height: '100vh' }}>
        <XR>
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 10, 10]} />
          
          <Controllers />
          <Hands />
          
          <AccessibleVRScene
            settings={currentSettings}
            colors={getAccessibleColors()}
            currentFocus={currentFocus}
            onFocusChange={setCurrentFocus}
            onAnnouncement={announceToUser}
          />
          
          <AccessibilityInterface
            settings={currentSettings}
            colors={getAccessibleColors()}
            isListening={isListening}
            currentFocus={currentFocus}
          />
        </XR>
      </Canvas>
    </div>
  );
};

interface AccessibilityFeatureControlProps {
  feature: AccessibilityFeature;
  onToggle: (enabled: boolean) => void;
}

const AccessibilityFeatureControl: React.FC<AccessibilityFeatureControlProps> = ({
  feature,
  onToggle
}) => {
  return (
    <div className="feature-control">
      <label>
        <input
          type="checkbox"
          checked={feature.enabled}
          onChange={(e) => onToggle(e.target.checked)}
        />
        <span className="feature-name">{feature.name}</span>
      </label>
      <p className="feature-description">{feature.description}</p>
    </div>
  );
};

interface AccessibleVRSceneProps {
  settings: AccessibilitySettings;
  colors: Record<string, string>;
  currentFocus: string | null;
  onFocusChange: (focus: string | null) => void;
  onAnnouncement: (message: string) => void;
}

const AccessibleVRScene: React.FC<AccessibleVRSceneProps> = ({
  settings,
  colors,
  currentFocus,
  onFocusChange,
  onAnnouncement
}) => {
  const interactiveElements = [
    { id: 'rock_sample_1', name: 'Granite Sample', position: new THREE.Vector3(-1, 0, 0) },
    { id: 'rock_sample_2', name: 'Limestone Sample', position: new THREE.Vector3(0, 0, 0) },
    { id: 'rock_sample_3', name: 'Slate Sample', position: new THREE.Vector3(1, 0, 0) },
    { id: 'geological_tool', name: 'Geological Compass', position: new THREE.Vector3(0, 1, -1) }
  ];

  return (
    <>
      {interactiveElements.map(element => (
        <AccessibleElement
          key={element.id}
          element={element}
          settings={settings}
          colors={colors}
          isFocused={currentFocus === element.id}
          onFocus={() => {
            onFocusChange(element.id);
            onAnnouncement(`Focused on ${element.name}`);
          }}
          onActivate={() => {
            onAnnouncement(`Activated ${element.name}`);
          }}
        />
      ))}
      
      {/* Accessible navigation aids */}
      <NavigationAids
        settings={settings}
        colors={colors}
        onAnnouncement={onAnnouncement}
      />
    </>
  );
};

const AccessibleElement: React.FC<{
  element: any;
  settings: AccessibilitySettings;
  colors: Record<string, string>;
  isFocused: boolean;
  onFocus: () => void;
  onActivate: () => void;
}> = ({ element, settings, colors, isFocused, onFocus, onActivate }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (meshRef.current && !settings.motionReduction) {
      // Gentle animation for non-motion-sensitive users
      meshRef.current.rotation.y += 0.01;
    }
    
    if (meshRef.current) {
      const scale = isFocused ? 1.2 : hovered ? 1.1 : 1.0;
      meshRef.current.scale.setScalar(scale);
    }
  });

  const getElementColor = (): number => {
    if (isFocused) return parseInt(colors.accent.replace('#', '0x'));
    if (hovered) return parseInt(colors.interactive.replace('#', '0x'));
    return 0x888888;
  };

  return (
    <group position={element.position}>
      <mesh
        ref={meshRef}
        onPointerEnter={() => {
          setHovered(true);
          onFocus();
        }}
        onPointerLeave={() => setHovered(false)}
        onClick={onActivate}
      >
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial
          color={getElementColor()}
          emissive={isFocused ? 0x222222 : 0x000000}
        />
      </mesh>
      
      <Text
        position={[0, 0.4, 0]}
        fontSize={0.05 * getFontScaleFactor(settings.fontSize)}
        color={colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        {element.name}
      </Text>
      
      {isFocused && (
        <FocusIndicator colors={colors} />
      )}
      
      {settings.cognitiveSupport && (
        <CognitiveAids element={element} colors={colors} />
      )}
    </group>
  );
};

const FocusIndicator: React.FC<{ colors: Record<string, string> }> = ({ colors }) => {
  return (
    <>
      {/* Focus ring */}
      <mesh>
        <ringGeometry args={[0.4, 0.45, 32]} />
        <meshBasicMaterial
          color={parseInt(colors.accent.replace('#', '0x'))}
          transparent
          opacity={0.8}
        />
      </mesh>
      
      {/* Focus corners */}
      {[
        [-0.4, 0.4, 0], [0.4, 0.4, 0], [-0.4, -0.4, 0], [0.4, -0.4, 0]
      ].map((pos, index) => (
        <Box key={index} args={[0.05, 0.05, 0.01]} position={pos as [number, number, number]}>
          <meshBasicMaterial color={parseInt(colors.accent.replace('#', '0x'))} />
        </Box>
      ))}
    </>
  );
};

const CognitiveAids: React.FC<{
  element: any;
  colors: Record<string, string>;
}> = ({ element, colors }) => {
  return (
    <group position={[0, -0.5, 0]}>
      <Text
        position={[0, 0, 0]}
        fontSize={0.03}
        color={colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        Click to examine
      </Text>
      
      <Text
        position={[0, -0.1, 0]}
        fontSize={0.025}
        color={colors.interactive}
        anchorX="center"
        anchorY="middle"
      >
        Say "describe this" for details
      </Text>
    </group>
  );
};

const NavigationAids: React.FC<{
  settings: AccessibilitySettings;
  colors: Record<string, string>;
  onAnnouncement: (message: string) => void;
}> = ({ settings, colors, onAnnouncement }) => {
  return (
    <group position={[0, 2.5, -2]}>
      {/* Spatial audio beacons for navigation */}
      <AudioBeacon
        position={new THREE.Vector3(0, 0, 0)}
        sound="home"
        onActivate={() => onAnnouncement('Home beacon activated')}
      />
      
      {/* Tactile feedback indicators */}
      {settings.cognitiveSupport && (
        <group position={[0, -0.5, 0]}>
          <Text
            fontSize={0.06 * getFontScaleFactor(settings.fontSize)}
            color={colors.foreground}
            anchorX="center"
            anchorY="middle"
          >
            Navigation Help Available
          </Text>
          
          <Text
            position={[0, -0.2, 0]}
            fontSize={0.04 * getFontScaleFactor(settings.fontSize)}
            color={colors.interactive}
            anchorX="center"
            anchorY="middle"
          >
            Say "help" for assistance
          </Text>
        </group>
      )}
    </group>
  );
};

const AudioBeacon: React.FC<{
  position: THREE.Vector3;
  sound: string;
  onActivate: () => void;
}> = ({ position, sound, onActivate }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle pulsing animation
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <mesh ref={meshRef} position={position} onClick={onActivate}>
      <sphereGeometry args={[0.1, 8, 8]} />
      <meshBasicMaterial color={0x00FF88} transparent opacity={0.6} />
    </mesh>
  );
};

const AccessibilityInterface: React.FC<{
  settings: AccessibilitySettings;
  colors: Record<string, string>;
  isListening: boolean;
  currentFocus: string | null;
}> = ({ settings, colors, isListening, currentFocus }) => {
  return (
    <group position={[-3, 2, -1]}>
      <Text
        position={[0, 0.5, 0]}
        fontSize={0.1 * getFontScaleFactor(settings.fontSize)}
        color={colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        Accessibility Active
      </Text>
      
      <Text
        position={[0, 0.2, 0]}
        fontSize={0.06 * getFontScaleFactor(settings.fontSize)}
        color={isListening ? colors.accent : colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        Voice: {isListening ? 'Listening' : 'Inactive'}
      </Text>
      
      <Text
        position={[0, 0, 0]}
        fontSize={0.06 * getFontScaleFactor(settings.fontSize)}
        color={colors.interactive}
        anchorX="center"
        anchorY="middle"
      >
        Focus: {currentFocus || 'None'}
      </Text>
      
      <Text
        position={[0, -0.2, 0]}
        fontSize={0.05 * getFontScaleFactor(settings.fontSize)}
        color={colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        High Contrast: {settings.highContrast ? 'On' : 'Off'}
      </Text>
      
      <Text
        position={[0, -0.4, 0]}
        fontSize={0.05 * getFontScaleFactor(settings.fontSize)}
        color={colors.foreground}
        anchorX="center"
        anchorY="middle"
      >
        Motion: {settings.motionReduction ? 'Reduced' : 'Normal'}
      </Text>
    </group>
  );
};

export default AccessibilityVRInterface;