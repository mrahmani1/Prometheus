import { WorkflowNodeDefinition } from '@/types/workflow';

export const geologicalNodeDefinitions: WorkflowNodeDefinition[] = [
  // Data Input Nodes
  {
    type: 'seismic-data-loader',
    name: 'Seismic Data Loader',
    description: 'Load seismic waveform data from various sources (USGS, local networks)',
    category: 'data-input',
    icon: 'timeline',
    color: '#2196F3',
    inputs: [],
    outputs: [
      {
        id: 'seismic_data',
        name: 'Seismic Data',
        type: 'seismic-data',
        required: true,
        description: 'Raw or processed seismic waveform data'
      }
    ],
    parameters: [
      {
        id: 'dataSource',
        name: 'Data Source',
        type: 'select',
        value: 'usgs',
        defaultValue: 'usgs',
        required: true,
        description: 'Source of seismic data',
        options: [
          { label: 'USGS', value: 'usgs' },
          { label: 'Local Network', value: 'local' },
          { label: 'File Upload', value: 'file' }
        ]
      },
      {
        id: 'timeRange',
        name: 'Time Range',
        type: 'time-range',
        value: { start: '', end: '' },
        defaultValue: { start: '', end: '' },
        required: true,
        description: 'Time range for data retrieval'
      },
      {
        id: 'spatialExtent',
        name: 'Spatial Extent',
        type: 'spatial-extent',
        value: { bounds: [] },
        defaultValue: { bounds: [] },
        required: true,
        description: 'Geographic bounds for data retrieval'
      },
      {
        id: 'sampleRate',
        name: 'Sample Rate (Hz)',
        type: 'number',
        value: 100,
        defaultValue: 100,
        required: false,
        description: 'Desired sample rate for data',
        validation: { min: 1, max: 1000 }
      }
    ]
  },
  
  {
    type: 'deformation-data-loader',
    name: 'Deformation Data Loader',
    description: 'Load ground deformation data from GPS, InSAR, or tiltmeter sources',
    category: 'data-input',
    icon: 'gps_fixed',
    color: '#2196F3',
    inputs: [],
    outputs: [
      {
        id: 'deformation_data',
        name: 'Deformation Data',
        type: 'deformation-data',
        required: true,
        description: 'Ground deformation measurements'
      }
    ],
    parameters: [
      {
        id: 'dataType',
        name: 'Data Type',
        type: 'select',
        value: 'gps',
        defaultValue: 'gps',
        required: true,
        description: 'Type of deformation data',
        options: [
          { label: 'GPS', value: 'gps' },
          { label: 'InSAR', value: 'insar' },
          { label: 'Tiltmeter', value: 'tiltmeter' }
        ]
      },
      {
        id: 'timeRange',
        name: 'Time Range',
        type: 'time-range',
        value: { start: '', end: '' },
        defaultValue: { start: '', end: '' },
        required: true,
        description: 'Time range for data retrieval'
      },
      {
        id: 'processingLevel',
        name: 'Processing Level',
        type: 'select',
        value: 'processed',
        defaultValue: 'processed',
        required: false,
        description: 'Level of data processing',
        options: [
          { label: 'Raw', value: 'raw' },
          { label: 'Processed', value: 'processed' }
        ]
      }
    ]
  },
  
  {
    type: 'geochemical-data-loader',
    name: 'Geochemical Data Loader',
    description: 'Load geochemical analysis data from rock, gas, or water samples',
    category: 'data-input',
    icon: 'science',
    color: '#2196F3',
    inputs: [],
    outputs: [
      {
        id: 'geochemical_data',
        name: 'Geochemical Data',
        type: 'geochemical-data',
        required: true,
        description: 'Geochemical analysis results'
      }
    ],
    parameters: [
      {
        id: 'sampleType',
        name: 'Sample Type',
        type: 'select',
        value: 'rock',
        defaultValue: 'rock',
        required: true,
        description: 'Type of geochemical sample',
        options: [
          { label: 'Rock', value: 'rock' },
          { label: 'Gas', value: 'gas' },
          { label: 'Water', value: 'water' }
        ]
      },
      {
        id: 'analysisType',
        name: 'Analysis Types',
        type: 'multi-select',
        value: ['major_elements'],
        defaultValue: ['major_elements'],
        required: true,
        description: 'Types of geochemical analysis',
        options: [
          { label: 'Major Elements', value: 'major_elements' },
          { label: 'Trace Elements', value: 'trace_elements' },
          { label: 'Isotopes', value: 'isotopes' },
          { label: 'Volatiles', value: 'volatiles' }
        ]
      }
    ]
  },
  
  // Analysis Nodes
  {
    type: 'seismic-event-detection',
    name: 'Seismic Event Detection',
    description: 'Detect and classify seismic events using various algorithms',
    category: 'seismic-analysis',
    icon: 'sensors',
    color: '#E91E63',
    inputs: [
      {
        id: 'seismic_data',
        name: 'Seismic Data',
        type: 'seismic-data',
        required: true,
        description: 'Input seismic waveform data'
      }
    ],
    outputs: [
      {
        id: 'events',
        name: 'Detected Events',
        type: 'table',
        required: true,
        description: 'List of detected seismic events'
      },
      {
        id: 'event_waveforms',
        name: 'Event Waveforms',
        type: 'seismic-data',
        required: false,
        description: 'Waveforms of detected events'
      }
    ],
    parameters: [
      {
        id: 'algorithm',
        name: 'Detection Algorithm',
        type: 'select',
        value: 'sta-lta',
        defaultValue: 'sta-lta',
        required: true,
        description: 'Algorithm for event detection',
        options: [
          { label: 'STA/LTA', value: 'sta-lta' },
          { label: 'CNN', value: 'cnn' },
          { label: 'Template Matching', value: 'template-matching' }
        ]
      },
      {
        id: 'threshold',
        name: 'Detection Threshold',
        type: 'number',
        value: 3.0,
        defaultValue: 3.0,
        required: true,
        description: 'Threshold for event detection',
        validation: { min: 1.0, max: 10.0 }
      },
      {
        id: 'windowSize',
        name: 'Window Size (s)',
        type: 'number',
        value: 10.0,
        defaultValue: 10.0,
        required: false,
        description: 'Analysis window size in seconds',
        validation: { min: 1.0, max: 60.0 }
      }
    ]
  },
  
  {
    type: 'deformation-analysis',
    name: 'Deformation Analysis',
    description: 'Analyze ground deformation patterns and calculate velocities',
    category: 'deformation-analysis',
    icon: 'trending_up',
    color: '#9C27B0',
    inputs: [
      {
        id: 'deformation_data',
        name: 'Deformation Data',
        type: 'deformation-data',
        required: true,
        description: 'Input deformation measurements'
      }
    ],
    outputs: [
      {
        id: 'velocity_field',
        name: 'Velocity Field',
        type: 'spatial-data',
        required: true,
        description: 'Calculated velocity field'
      },
      {
        id: 'strain_field',
        name: 'Strain Field',
        type: 'spatial-data',
        required: false,
        description: 'Calculated strain field'
      }
    ],
    parameters: [
      {
        id: 'analysisType',
        name: 'Analysis Type',
        type: 'select',
        value: 'velocity',
        defaultValue: 'velocity',
        required: true,
        description: 'Type of deformation analysis',
        options: [
          { label: 'Velocity', value: 'velocity' },
          { label: 'Acceleration', value: 'acceleration' },
          { label: 'Strain', value: 'strain' }
        ]
      },
      {
        id: 'referenceFrame',
        name: 'Reference Frame',
        type: 'string',
        value: 'ITRF2014',
        defaultValue: 'ITRF2014',
        required: true,
        description: 'Geodetic reference frame'
      },
      {
        id: 'smoothingWindow',
        name: 'Smoothing Window (days)',
        type: 'number',
        value: 30,
        defaultValue: 30,
        required: false,
        description: 'Temporal smoothing window',
        validation: { min: 1, max: 365 }
      }
    ]
  },
  
  {
    type: 'magma-chamber-modeling',
    name: 'Magma Chamber Modeling',
    description: 'Model magma chamber parameters from deformation data',
    category: 'geological-analysis',
    icon: 'volcano',
    color: '#8BC34A',
    inputs: [
      {
        id: 'deformation_data',
        name: 'Deformation Data',
        type: 'deformation-data',
        required: true,
        description: 'Ground deformation measurements'
      }
    ],
    outputs: [
      {
        id: 'chamber_model',
        name: 'Chamber Model',
        type: 'geological-model',
        required: true,
        description: 'Magma chamber model parameters'
      },
      {
        id: 'model_fit',
        name: 'Model Fit',
        type: 'table',
        required: false,
        description: 'Model fitting statistics'
      }
    ],
    parameters: [
      {
        id: 'modelType',
        name: 'Model Type',
        type: 'select',
        value: 'mogi',
        defaultValue: 'mogi',
        required: true,
        description: 'Type of magma chamber model',
        options: [
          { label: 'Mogi Point Source', value: 'mogi' },
          { label: 'Yang Finite Source', value: 'yang' },
          { label: 'Finite Element', value: 'finite-element' }
        ]
      },
      {
        id: 'depthRange',
        name: 'Depth Range (km)',
        type: 'range',
        value: { min: 1, max: 10 },
        defaultValue: { min: 1, max: 10 },
        required: true,
        description: 'Search range for chamber depth'
      },
      {
        id: 'volumeRange',
        name: 'Volume Range (km³)',
        type: 'range',
        value: { min: 0.001, max: 1.0 },
        defaultValue: { min: 0.001, max: 1.0 },
        required: false,
        description: 'Search range for volume change'
      }
    ]
  },
  
  {
    type: 'helios-ai-simulation',
    name: 'Helios AI Simulation',
    description: 'Run Physics-Informed Neural Network geological simulations',
    category: 'ai-modeling',
    icon: 'psychology',
    color: '#673AB7',
    inputs: [
      {
        id: 'geological_data',
        name: 'Geological Data',
        type: 'any',
        required: true,
        description: 'Input geological datasets'
      }
    ],
    outputs: [
      {
        id: 'simulation_results',
        name: 'Simulation Results',
        type: 'geological-model',
        required: true,
        description: 'PINN simulation results'
      },
      {
        id: 'uncertainty',
        name: 'Uncertainty',
        type: 'spatial-data',
        required: false,
        description: 'Uncertainty quantification'
      }
    ],
    parameters: [
      {
        id: 'simulationType',
        name: 'Simulation Type',
        type: 'select',
        value: 'long-term',
        defaultValue: 'long-term',
        required: true,
        description: 'Type of geological simulation',
        options: [
          { label: 'Long-term Evolution', value: 'long-term' },
          { label: 'Real-time Analysis', value: 'real-time' }
        ]
      },
      {
        id: 'timeSpan',
        name: 'Time Span',
        type: 'time-range',
        value: { start: '', end: '' },
        defaultValue: { start: '', end: '' },
        required: true,
        description: 'Simulation time span'
      },
      {
        id: 'physicsConstraints',
        name: 'Physics Constraints',
        type: 'multi-select',
        value: ['navier-stokes'],
        defaultValue: ['navier-stokes'],
        required: true,
        description: 'Physics laws to enforce',
        options: [
          { label: 'Navier-Stokes', value: 'navier-stokes' },
          { label: 'Heat Transfer', value: 'heat-transfer' },
          { label: 'Elastic Deformation', value: 'elastic-deformation' },
          { label: 'Mass Conservation', value: 'mass-conservation' }
        ]
      },
      {
        id: 'uncertaintyQuantification',
        name: 'Uncertainty Quantification',
        type: 'boolean',
        value: true,
        defaultValue: true,
        required: false,
        description: 'Enable uncertainty quantification using GANs'
      }
    ]
  },
  
  // Visualization Nodes
  {
    type: 'time-series-plot',
    name: 'Time Series Plot',
    description: 'Create interactive time series visualizations',
    category: 'visualization',
    icon: 'show_chart',
    color: '#4CAF50',
    inputs: [
      {
        id: 'time_series_data',
        name: 'Time Series Data',
        type: 'time-series',
        required: true,
        description: 'Time series data to plot'
      }
    ],
    outputs: [
      {
        id: 'plot',
        name: 'Plot',
        type: 'image',
        required: true,
        description: 'Generated time series plot'
      }
    ],
    parameters: [
      {
        id: 'plotType',
        name: 'Plot Type',
        type: 'select',
        value: 'line',
        defaultValue: 'line',
        required: true,
        description: 'Type of plot visualization',
        options: [
          { label: 'Line Plot', value: 'line' },
          { label: 'Scatter Plot', value: 'scatter' },
          { label: 'Bar Chart', value: 'bar' }
        ]
      },
      {
        id: 'xAxis',
        name: 'X-Axis Variable',
        type: 'string',
        value: 'time',
        defaultValue: 'time',
        required: true,
        description: 'Variable for X-axis'
      },
      {
        id: 'yAxis',
        name: 'Y-Axis Variables',
        type: 'multi-select',
        value: [],
        defaultValue: [],
        required: true,
        description: 'Variables for Y-axis'
      },
      {
        id: 'showUncertainty',
        name: 'Show Uncertainty',
        type: 'boolean',
        value: false,
        defaultValue: false,
        required: false,
        description: 'Display uncertainty bands'
      }
    ]
  },
  
  {
    type: '3d-geological-visualization',
    name: '3D Geological Visualization',
    description: 'Create 3D visualizations of geological models and data',
    category: 'visualization',
    icon: 'view_in_ar',
    color: '#4CAF50',
    inputs: [
      {
        id: 'geological_model',
        name: 'Geological Model',
        type: 'geological-model',
        required: true,
        description: '3D geological model data'
      }
    ],
    outputs: [
      {
        id: 'visualization',
        name: '3D Visualization',
        type: 'image',
        required: true,
        description: 'Generated 3D visualization'
      }
    ],
    parameters: [
      {
        id: 'renderMode',
        name: 'Render Mode',
        type: 'select',
        value: 'surface',
        defaultValue: 'surface',
        required: true,
        description: '3D rendering mode',
        options: [
          { label: 'Surface', value: 'surface' },
          { label: 'Volume', value: 'volume' },
          { label: 'Points', value: 'points' }
        ]
      },
      {
        id: 'colormap',
        name: 'Color Map',
        type: 'select',
        value: 'viridis',
        defaultValue: 'viridis',
        required: false,
        description: 'Color mapping scheme',
        options: [
          { label: 'Viridis', value: 'viridis' },
          { label: 'Plasma', value: 'plasma' },
          { label: 'Jet', value: 'jet' },
          { label: 'Terrain', value: 'terrain' }
        ]
      },
      {
        id: 'opacity',
        name: 'Opacity',
        type: 'range',
        value: 0.8,
        defaultValue: 0.8,
        required: false,
        description: 'Visualization opacity',
        validation: { min: 0.0, max: 1.0 }
      }
    ]
  },
  
  {
    type: 'hazard-map-generator',
    name: 'Hazard Map Generator',
    description: 'Generate probabilistic hazard maps for various geological hazards',
    category: 'visualization',
    icon: 'warning',
    color: '#4CAF50',
    inputs: [
      {
        id: 'hazard_data',
        name: 'Hazard Data',
        type: 'any',
        required: true,
        description: 'Hazard assessment data'
      }
    ],
    outputs: [
      {
        id: 'hazard_map',
        name: 'Hazard Map',
        type: 'spatial-data',
        required: true,
        description: 'Generated hazard map'
      }
    ],
    parameters: [
      {
        id: 'hazardType',
        name: 'Hazard Type',
        type: 'select',
        value: 'volcanic',
        defaultValue: 'volcanic',
        required: true,
        description: 'Type of geological hazard',
        options: [
          { label: 'Volcanic', value: 'volcanic' },
          { label: 'Seismic', value: 'seismic' },
          { label: 'Landslide', value: 'landslide' }
        ]
      },
      {
        id: 'probabilityLevels',
        name: 'Probability Levels (%)',
        type: 'multi-select',
        value: [10, 50, 90],
        defaultValue: [10, 50, 90],
        required: true,
        description: 'Probability levels to display'
      },
      {
        id: 'timeHorizon',
        name: 'Time Horizon',
        type: 'select',
        value: '1-year',
        defaultValue: '1-year',
        required: true,
        description: 'Time horizon for hazard assessment',
        options: [
          { label: '1 Month', value: '1-month' },
          { label: '1 Year', value: '1-year' },
          { label: '10 Years', value: '10-years' },
          { label: '100 Years', value: '100-years' }
        ]
      }
    ]
  }
];