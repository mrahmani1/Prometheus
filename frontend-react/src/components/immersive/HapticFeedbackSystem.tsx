import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { HapticPattern, HapticDevice, GeologicalMaterial } from '../../types/immersive';
import { setHapticFeedback, updateHapticDevice } from '../../store/slices/immersiveSlice';

interface HapticFeedbackSystemProps {
  enabled?: boolean;
  sensitivity?: number;
  onHapticEvent?: (event: HapticEvent) => void;
}

interface HapticEvent {
  type: 'contact' | 'texture' | 'resistance' | 'vibration';
  intensity: number;
  duration: number;
  pattern?: HapticPattern;
  material?: GeologicalMaterial;
  timestamp: number;
}

interface GamepadHaptic extends Gamepad {
  hapticActuators?: GamepadHapticActuator[];
}

interface GamepadHapticActuator {
  pulse(value: number, duration: number): Promise<boolean>;
  type: string;
}

const HapticFeedbackSystem: React.FC<HapticFeedbackSystemProps> = ({
  enabled = true,
  sensitivity = 1.0,
  onHapticEvent
}) => {
  const dispatch = useDispatch();
  const hapticState = useSelector((state: any) => state.immersive.hapticFeedback);
  const [connectedDevices, setConnectedDevices] = useState<HapticDevice[]>([]);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const gamepadRef = useRef<GamepadHaptic | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const oscillatorRef = useRef<OscillatorNode | null>(null);

  useEffect(() => {
    if (enabled) {
      initializeHapticDevices();
      initializeAudioHaptics();
    }

    return () => {
      cleanup();
    };
  }, [enabled]);

  const initializeHapticDevices = useCallback(() => {
    // Initialize gamepad haptics
    const checkGamepads = () => {
      const gamepads = navigator.getGamepads();
      const hapticDevices: HapticDevice[] = [];

      for (let i = 0; i < gamepads.length; i++) {
        const gamepad = gamepads[i] as GamepadHaptic;
        if (gamepad && gamepad.hapticActuators && gamepad.hapticActuators.length > 0) {
          gamepadRef.current = gamepad;
          
          const device: HapticDevice = {
            id: `gamepad_${i}`,
            type: 'gamepad',
            name: gamepad.id,
            capabilities: {
              supportsVibration: true,
              supportsForce: false,
              supportsTexture: false,
              maxIntensity: 1.0,
              minDuration: 10,
              maxDuration: 5000
            },
            connected: true
          };
          
          hapticDevices.push(device);
          dispatch(updateHapticDevice(device));
        }
      }

      setConnectedDevices(hapticDevices);
    };

    // Check for Web Haptics API (experimental)
    if ('haptics' in navigator) {
      // Future implementation for Web Haptics API
      console.log('Web Haptics API detected (experimental)');
    }

    // Listen for gamepad connections
    window.addEventListener('gamepadconnected', checkGamepads);
    window.addEventListener('gamepaddisconnected', checkGamepads);
    
    // Initial check
    checkGamepads();

    return () => {
      window.removeEventListener('gamepadconnected', checkGamepads);
      window.removeEventListener('gamepaddisconnected', checkGamepads);
    };
  }, [dispatch]);

  const initializeAudioHaptics = useCallback(() => {
    // Initialize audio-based haptic feedback for devices without haptic actuators
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    } catch (error) {
      console.warn('Audio haptics not available:', error);
    }
  }, []);

  const cleanup = useCallback(() => {
    if (oscillatorRef.current) {
      oscillatorRef.current.stop();
      oscillatorRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  const triggerHapticFeedback = useCallback(async (
    intensity: number,
    duration: number,
    pattern?: HapticPattern,
    material?: GeologicalMaterial
  ): Promise<boolean> => {
    if (!enabled) return false;

    const adjustedIntensity = Math.min(1.0, intensity * sensitivity);
    const hapticEvent: HapticEvent = {
      type: pattern?.type || 'vibration',
      intensity: adjustedIntensity,
      duration,
      pattern,
      material,
      timestamp: Date.now()
    };

    // Trigger gamepad haptics
    let success = false;
    if (gamepadRef.current && gamepadRef.current.hapticActuators) {
      try {
        for (const actuator of gamepadRef.current.hapticActuators) {
          await actuator.pulse(adjustedIntensity, duration);
          success = true;
        }
      } catch (error) {
        console.warn('Gamepad haptic feedback failed:', error);
      }
    }

    // Fallback to audio haptics
    if (!success && audioContextRef.current) {
      success = await triggerAudioHaptics(adjustedIntensity, duration, pattern);
    }

    // Dispatch haptic event
    dispatch(setHapticFeedback({
      intensity: adjustedIntensity,
      duration,
      pattern: pattern?.name || 'default',
      timestamp: Date.now()
    }));

    onHapticEvent?.(hapticEvent);
    return success;
  }, [enabled, sensitivity, dispatch, onHapticEvent]);

  const triggerAudioHaptics = useCallback(async (
    intensity: number,
    duration: number,
    pattern?: HapticPattern
  ): Promise<boolean> => {
    if (!audioContextRef.current) return false;

    try {
      const oscillator = audioContextRef.current.createOscillator();
      const gainNode = audioContextRef.current.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContextRef.current.destination);
      
      // Configure based on pattern
      if (pattern) {
        switch (pattern.type) {
          case 'texture':
            oscillator.frequency.setValueAtTime(200 + intensity * 300, audioContextRef.current.currentTime);
            break;
          case 'resistance':
            oscillator.frequency.setValueAtTime(50 + intensity * 100, audioContextRef.current.currentTime);
            break;
          case 'contact':
            oscillator.frequency.setValueAtTime(400 + intensity * 200, audioContextRef.current.currentTime);
            break;
          default:
            oscillator.frequency.setValueAtTime(150 + intensity * 250, audioContextRef.current.currentTime);
        }
      } else {
        oscillator.frequency.setValueAtTime(200, audioContextRef.current.currentTime);
      }
      
      gainNode.gain.setValueAtTime(intensity * 0.1, audioContextRef.current.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.001, audioContextRef.current.currentTime + duration / 1000);
      
      oscillator.start(audioContextRef.current.currentTime);
      oscillator.stop(audioContextRef.current.currentTime + duration / 1000);
      
      oscillatorRef.current = oscillator;
      return true;
    } catch (error) {
      console.warn('Audio haptic feedback failed:', error);
      return false;
    }
  }, []);

  const getGeologicalHapticPattern = useCallback((material: GeologicalMaterial): HapticPattern => {
    const patterns: Record<string, HapticPattern> = {
      granite: {
        name: 'granite',
        type: 'texture',
        intensity: 0.8,
        frequency: 150,
        duration: 200,
        waveform: 'square'
      },
      sandstone: {
        name: 'sandstone',
        type: 'texture',
        intensity: 0.4,
        frequency: 80,
        duration: 300,
        waveform: 'sine'
      },
      limestone: {
        name: 'limestone',
        type: 'texture',
        intensity: 0.6,
        frequency: 120,
        duration: 250,
        waveform: 'triangle'
      },
      shale: {
        name: 'shale',
        type: 'texture',
        intensity: 0.3,
        frequency: 60,
        duration: 400,
        waveform: 'sine'
      },
      basalt: {
        name: 'basalt',
        type: 'texture',
        intensity: 0.9,
        frequency: 180,
        duration: 150,
        waveform: 'square'
      },
      quartz: {
        name: 'quartz',
        type: 'contact',
        intensity: 0.7,
        frequency: 200,
        duration: 100,
        waveform: 'sine'
      }
    };

    return patterns[material.type] || patterns.granite;
  }, []);

  const simulateGeologicalContact = useCallback(async (material: GeologicalMaterial, force: number) => {
    const pattern = getGeologicalHapticPattern(material);
    const intensity = Math.min(1.0, pattern.intensity * force);
    
    await triggerHapticFeedback(intensity, pattern.duration, pattern, material);
  }, [getGeologicalHapticPattern, triggerHapticFeedback]);

  const simulateToolResistance = useCallback(async (hardness: number, toolType: string) => {
    const resistancePattern: HapticPattern = {
      name: 'tool_resistance',
      type: 'resistance',
      intensity: Math.min(1.0, hardness * 0.1),
      frequency: 50 + hardness * 10,
      duration: 500,
      waveform: 'sawtooth'
    };

    await triggerHapticFeedback(resistancePattern.intensity, resistancePattern.duration, resistancePattern);
  }, [triggerHapticFeedback]);

  const calibrateHapticDevices = useCallback(async () => {
    setIsCalibrating(true);
    
    try {
      // Test different intensity levels
      const testIntensities = [0.2, 0.5, 0.8, 1.0];
      
      for (const intensity of testIntensities) {
        await triggerHapticFeedback(intensity, 200);
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      
      // Test different patterns
      const testMaterials: GeologicalMaterial[] = [
        { type: 'granite', hardness: 8, density: 2.7 },
        { type: 'sandstone', hardness: 4, density: 2.2 },
        { type: 'limestone', hardness: 3, density: 2.5 }
      ];
      
      for (const material of testMaterials) {
        await simulateGeologicalContact(material, 0.7);
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      
    } catch (error) {
      console.error('Haptic calibration failed:', error);
    } finally {
      setIsCalibrating(false);
    }
  }, [triggerHapticFeedback, simulateGeologicalContact]);

  return (
    <div className="haptic-feedback-system">
      <div className="haptic-status">
        <h3>Haptic Feedback System</h3>
        <p>Status: {enabled ? 'Enabled' : 'Disabled'}</p>
        <p>Devices: {connectedDevices.length}</p>
        <p>Sensitivity: {(sensitivity * 100).toFixed(0)}%</p>
      </div>

      <div className="haptic-controls">
        <button
          onClick={calibrateHapticDevices}
          disabled={isCalibrating || !enabled}
        >
          {isCalibrating ? 'Calibrating...' : 'Calibrate Devices'}
        </button>
        
        <button
          onClick={() => triggerHapticFeedback(0.5, 200)}
          disabled={!enabled}
        >
          Test Haptics
        </button>
      </div>

      <div className="connected-devices">
        <h4>Connected Devices</h4>
        {connectedDevices.map(device => (
          <div key={device.id} className="device-info">
            <p><strong>{device.name}</strong></p>
            <p>Type: {device.type}</p>
            <p>Vibration: {device.capabilities.supportsVibration ? 'Yes' : 'No'}</p>
            <p>Force: {device.capabilities.supportsForce ? 'Yes' : 'No'}</p>
          </div>
        ))}
        
        {connectedDevices.length === 0 && (
          <p>No haptic devices detected. Audio fallback available.</p>
        )}
      </div>

      <div className="geological-materials">
        <h4>Test Geological Materials</h4>
        <div className="material-buttons">
          <button onClick={() => simulateGeologicalContact({ type: 'granite', hardness: 8, density: 2.7 }, 0.8)}>
            Granite
          </button>
          <button onClick={() => simulateGeologicalContact({ type: 'sandstone', hardness: 4, density: 2.2 }, 0.6)}>
            Sandstone
          </button>
          <button onClick={() => simulateGeologicalContact({ type: 'limestone', hardness: 3, density: 2.5 }, 0.5)}>
            Limestone
          </button>
          <button onClick={() => simulateToolResistance(7, 'hammer')}>
            Tool Resistance
          </button>
        </div>
      </div>

      <style jsx>{`
        .haptic-feedback-system {
          padding: 20px;
          background: rgba(0, 0, 0, 0.8);
          color: white;
          border-radius: 8px;
          margin: 10px;
        }
        
        .haptic-status p {
          margin: 5px 0;
        }
        
        .haptic-controls {
          margin: 15px 0;
        }
        
        .haptic-controls button {
          margin: 5px;
          padding: 8px 16px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .haptic-controls button:disabled {
          background: #666;
          cursor: not-allowed;
        }
        
        .device-info {
          background: rgba(255, 255, 255, 0.1);
          padding: 10px;
          margin: 5px 0;
          border-radius: 4px;
        }
        
        .device-info p {
          margin: 2px 0;
        }
        
        .material-buttons button {
          margin: 5px;
          padding: 6px 12px;
          background: #28a745;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
        }
        
        .material-buttons button:hover {
          background: #218838;
        }
      `}</style>
    </div>
  );
};

// Utility functions for haptic patterns
export const HapticPatterns = {
  // Geological material patterns
  createMaterialPattern: (material: GeologicalMaterial): HapticPattern => {
    const baseIntensity = Math.min(1.0, material.hardness / 10);
    const frequency = 50 + material.hardness * 15;
    
    return {
      name: `material_${material.type}`,
      type: 'texture',
      intensity: baseIntensity,
      frequency,
      duration: 200 + material.hardness * 20,
      waveform: material.hardness > 6 ? 'square' : 'sine'
    };
  },

  // Tool interaction patterns
  createToolPattern: (toolType: string, resistance: number): HapticPattern => {
    const patterns: Record<string, Partial<HapticPattern>> = {
      hammer: { frequency: 100, waveform: 'square' },
      chisel: { frequency: 200, waveform: 'triangle' },
      drill: { frequency: 300, waveform: 'sawtooth' },
      brush: { frequency: 80, waveform: 'sine' }
    };

    const basePattern = patterns[toolType] || patterns.hammer;
    
    return {
      name: `tool_${toolType}`,
      type: 'resistance',
      intensity: Math.min(1.0, resistance),
      frequency: basePattern.frequency || 100,
      duration: 300,
      waveform: basePattern.waveform || 'square'
    };
  },

  // Environmental patterns
  createEnvironmentalPattern: (environment: string): HapticPattern => {
    const patterns: Record<string, HapticPattern> = {
      wind: {
        name: 'wind',
        type: 'vibration',
        intensity: 0.3,
        frequency: 20,
        duration: 1000,
        waveform: 'sine'
      },
      earthquake: {
        name: 'earthquake',
        type: 'vibration',
        intensity: 0.9,
        frequency: 10,
        duration: 2000,
        waveform: 'random'
      },
      water_flow: {
        name: 'water_flow',
        type: 'vibration',
        intensity: 0.4,
        frequency: 60,
        duration: 500,
        waveform: 'sine'
      }
    };

    return patterns[environment] || patterns.wind;
  }
};

export default HapticFeedbackSystem;