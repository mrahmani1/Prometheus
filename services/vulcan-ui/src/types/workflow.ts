export interface WorkflowNode {
  id: string;
  type: string;
  name: string;
  description: string;
  position: { x: number; y: number };
  parameters: Record<string, any>;
  inputs: WorkflowPort[];
  outputs: WorkflowPort[];
  category: WorkflowNodeCategory;
  status?: 'idle' | 'running' | 'completed' | 'error';
  error?: string;
  metadata?: Record<string, any>;
}

export interface WorkflowPort {
  id: string;
  name: string;
  type: WorkflowDataType;
  required: boolean;
  description: string;
  defaultValue?: any;
}

export interface WorkflowConnection {
  id: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
  dataType: WorkflowDataType;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  workflow: {
    nodes: WorkflowNode[];
    connections: WorkflowConnection[];
  };
  thumbnail?: string;
  author: string;
  version: string;
  createdAt: string;
  updatedAt: string;
}

export interface WorkflowParameter {
  id: string;
  name: string;
  type: WorkflowParameterType;
  value: any;
  defaultValue: any;
  required: boolean;
  description: string;
  validation?: WorkflowParameterValidation;
  options?: WorkflowParameterOption[];
  dependencies?: string[];
  group?: string;
}

export interface WorkflowParameterValidation {
  min?: number;
  max?: number;
  pattern?: string;
  custom?: (value: any) => string | null;
}

export interface WorkflowParameterOption {
  label: string;
  value: any;
  description?: string;
}

export type WorkflowNodeCategory = 
  | 'data-input'
  | 'data-processing'
  | 'geological-analysis'
  | 'seismic-analysis'
  | 'deformation-analysis'
  | 'geochemical-analysis'
  | 'thermal-analysis'
  | 'ai-modeling'
  | 'visualization'
  | 'data-output'
  | 'control-flow'
  | 'utilities';

export type WorkflowDataType = 
  | 'seismic-data'
  | 'deformation-data'
  | 'geochemical-data'
  | 'thermal-data'
  | 'geological-model'
  | 'time-series'
  | 'spatial-data'
  | 'image'
  | 'mesh'
  | 'point-cloud'
  | 'table'
  | 'text'
  | 'number'
  | 'boolean'
  | 'array'
  | 'object'
  | 'any';

export type WorkflowParameterType = 
  | 'string'
  | 'number'
  | 'boolean'
  | 'select'
  | 'multi-select'
  | 'range'
  | 'date'
  | 'datetime'
  | 'file'
  | 'color'
  | 'coordinate'
  | 'geological-unit'
  | 'time-range'
  | 'spatial-extent'
  | 'custom';

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  startTime: string;
  endTime?: string;
  progress: number;
  currentNode?: string;
  results?: Record<string, any>;
  error?: string;
  logs: WorkflowExecutionLog[];
}

export interface WorkflowExecutionLog {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  nodeId?: string;
  message: string;
  data?: any;
}

export interface WorkflowValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export interface GeologicalWorkflowNodes {
  // Data Input Nodes
  'seismic-data-loader': {
    parameters: {
      dataSource: string;
      timeRange: { start: string; end: string };
      spatialExtent: { bounds: number[] };
      sampleRate: number;
      filterSettings: Record<string, any>;
    };
  };
  
  'deformation-data-loader': {
    parameters: {
      dataType: 'gps' | 'insar' | 'tiltmeter';
      timeRange: { start: string; end: string };
      spatialExtent: { bounds: number[] };
      processingLevel: 'raw' | 'processed';
    };
  };
  
  'geochemical-data-loader': {
    parameters: {
      sampleType: 'rock' | 'gas' | 'water';
      analysisType: string[];
      spatialExtent: { bounds: number[] };
      qualityFilter: string;
    };
  };
  
  // Analysis Nodes
  'seismic-event-detection': {
    parameters: {
      algorithm: 'sta-lta' | 'cnn' | 'template-matching';
      threshold: number;
      windowSize: number;
      minimumMagnitude: number;
    };
  };
  
  'deformation-analysis': {
    parameters: {
      analysisType: 'velocity' | 'acceleration' | 'strain';
      referenceFrame: string;
      smoothingWindow: number;
      outlierThreshold: number;
    };
  };
  
  'magma-chamber-modeling': {
    parameters: {
      modelType: 'mogi' | 'yang' | 'finite-element';
      depthRange: { min: number; max: number };
      volumeRange: { min: number; max: number };
      pressureRange: { min: number; max: number };
    };
  };
  
  'helios-ai-simulation': {
    parameters: {
      simulationType: 'long-term' | 'real-time';
      timeSpan: { start: string; end: string };
      spatialResolution: number;
      physicsConstraints: string[];
      uncertaintyQuantification: boolean;
    };
  };
  
  // Visualization Nodes
  'time-series-plot': {
    parameters: {
      plotType: 'line' | 'scatter' | 'bar';
      xAxis: string;
      yAxis: string[];
      colorBy: string;
      showUncertainty: boolean;
    };
  };
  
  '3d-geological-visualization': {
    parameters: {
      renderMode: 'surface' | 'volume' | 'points';
      colormap: string;
      opacity: number;
      lighting: boolean;
      animation: boolean;
    };
  };
  
  'hazard-map-generator': {
    parameters: {
      hazardType: 'volcanic' | 'seismic' | 'landslide';
      probabilityLevels: number[];
      timeHorizon: string;
      spatialResolution: number;
    };
  };
}

export interface WorkflowNodeDefinition {
  type: string;
  name: string;
  description: string;
  category: WorkflowNodeCategory;
  icon: string;
  color: string;
  inputs: WorkflowPort[];
  outputs: WorkflowPort[];
  parameters: WorkflowParameter[];
  documentation?: string;
  examples?: any[];
}