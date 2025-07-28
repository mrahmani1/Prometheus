// Geological data types for the Vulcan UI
export interface GeologicalLocation {
  latitude: number;
  longitude: number;
  elevation: number;
  depth?: number;
  coordinateSystem?: string;
}

export interface SeismicEvent {
  id: string;
  timestamp: string;
  location: GeologicalLocation;
  magnitude: number;
  depth: number;
  eventType: 'earthquake' | 'volcanic' | 'explosion' | 'other';
  confidence: number;
}

export interface DeformationMeasurement {
  id: string;
  timestamp: string;
  location: GeologicalLocation;
  displacementVector: [number, number, number]; // [x, y, z] in meters
  measurementType: 'GPS' | 'InSAR' | 'tiltmeter' | 'other';
  uncertainty: [number, number, number];
  referenceFrame: string;
}

export interface GeochemicalSample {
  id: string;
  timestamp: string;
  location: GeologicalLocation;
  sampleType: 'rock' | 'gas' | 'water' | 'other';
  composition: Record<string, number>;
  isotopeRatios?: Record<string, number>;
  temperature?: number;
  pressure?: number;
}

export interface ThermalMeasurement {
  id: string;
  timestamp: string;
  location: GeologicalLocation;
  temperature: number;
  measurementMethod: 'satellite' | 'ground-based' | 'borehole' | 'other';
  spatialResolution: number;
  uncertainty: number;
}

export interface VolcanicRiskAssessment {
  volcanoId: string;
  timestamp: string;
  alertLevel: {
    level: number; // 0-5 scale
    description: string;
    recommendedActions: string[];
    validityPeriod: [string, string];
    confidence: number;
  };
  eruptionProbability: Record<string, number>; // timeframe -> probability
  hazardZones: Record<string, number[][]>; // hazard type -> affected areas
  contributingFactors: string[];
  modelEnsembleAgreement: number;
}

export interface SimulationParameters {
  initialConditions: Record<string, number[]>;
  timeSpan: [string, string];
  spatialDomain: {
    x: [number, number];
    y: [number, number];
    z: [number, number];
  };
  resolution: {
    spatial: number;
    temporal: number;
  };
  physicsModels: string[];
  uncertaintyQuantification: boolean;
}

export interface ModelPrediction {
  id: string;
  timestamp: string;
  predictionType: string;
  values: number[];
  uncertainty: number[];
  confidenceInterval: [number, number];
  modelVersion: string;
  inputDataHash: string;
}

export interface GeologicalWorkflow {
  id: string;
  name: string;
  description: string;
  steps: WorkflowStep[];
  createdAt: string;
  updatedAt: string;
  userId: string;
}

export interface WorkflowStep {
  id: string;
  type: 'data-input' | 'processing' | 'analysis' | 'visualization' | 'export';
  name: string;
  parameters: Record<string, any>;
  position: { x: number; y: number };
  connections: string[]; // IDs of connected steps
}