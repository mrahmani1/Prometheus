import { WorkflowTemplate } from '@/types/workflow';
import { generateId } from '@/utils/idGenerator';

export const geologicalWorkflowTemplates: WorkflowTemplate[] = [
  {
    id: 'seismic-event-analysis',
    name: 'Seismic Event Analysis',
    description: 'Complete workflow for detecting, analyzing, and visualizing seismic events from continuous waveform data',
    category: 'seismic-analysis',
    tags: ['seismic', 'event-detection', 'analysis', 'visualization'],
    workflow: {
      nodes: [
        {
          id: generateId(),
          type: 'seismic-data-loader',
          name: 'Load Seismic Data',
          description: 'Load seismic waveform data from USGS',
          position: { x: 50, y: 100 },
          parameters: {
            dataSource: 'usgs',
            timeRange: { start: '2024-01-01T00:00:00', end: '2024-01-02T00:00:00' },
            spatialExtent: { bounds: [-120, 35, -119, 36] },
            sampleRate: 100
          },
          inputs: [],
          outputs: [
            {
              id: 'seismic_data',
              name: 'Seismic Data',
              type: 'seismic-data',
              required: true,
              description: 'Raw seismic waveform data'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'seismic-event-detection',
          name: 'Detect Events',
          description: 'Detect seismic events using STA/LTA algorithm',
          position: { x: 350, y: 100 },
          parameters: {
            algorithm: 'sta-lta',
            threshold: 3.0,
            windowSize: 10.0
          },
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
            }
          ],
          category: 'seismic-analysis',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'time-series-plot',
          name: 'Plot Events',
          description: 'Create time series plot of detected events',
          position: { x: 650, y: 100 },
          parameters: {
            plotType: 'scatter',
            xAxis: 'time',
            yAxis: ['magnitude'],
            showUncertainty: false
          },
          inputs: [
            {
              id: 'time_series_data',
              name: 'Time Series Data',
              type: 'table',
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
          category: 'visualization',
          status: 'idle'
        }
      ],
      connections: [
        {
          id: generateId(),
          sourceNodeId: 'seismic-data-loader',
          sourcePortId: 'seismic_data',
          targetNodeId: 'seismic-event-detection',
          targetPortId: 'seismic_data',
          dataType: 'seismic-data'
        },
        {
          id: generateId(),
          sourceNodeId: 'seismic-event-detection',
          sourcePortId: 'events',
          targetNodeId: 'time-series-plot',
          targetPortId: 'time_series_data',
          dataType: 'table'
        }
      ]
    },
    author: 'Prometheus Team',
    version: '1.0.0',
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-15T10:00:00Z'
  },
  
  {
    id: 'volcanic-deformation-monitoring',
    name: 'Volcanic Deformation Monitoring',
    description: 'Real-time monitoring workflow for volcanic ground deformation using GPS and InSAR data',
    category: 'volcanic-monitoring',
    tags: ['volcanic', 'deformation', 'gps', 'insar', 'monitoring'],
    workflow: {
      nodes: [
        {
          id: generateId(),
          type: 'deformation-data-loader',
          name: 'Load GPS Data',
          description: 'Load GPS deformation measurements',
          position: { x: 50, y: 50 },
          parameters: {
            dataType: 'gps',
            timeRange: { start: '2024-01-01T00:00:00', end: '2024-01-31T23:59:59' },
            processingLevel: 'processed'
          },
          inputs: [],
          outputs: [
            {
              id: 'deformation_data',
              name: 'GPS Data',
              type: 'deformation-data',
              required: true,
              description: 'GPS deformation measurements'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'deformation-data-loader',
          name: 'Load InSAR Data',
          description: 'Load InSAR deformation measurements',
          position: { x: 50, y: 200 },
          parameters: {
            dataType: 'insar',
            timeRange: { start: '2024-01-01T00:00:00', end: '2024-01-31T23:59:59' },
            processingLevel: 'processed'
          },
          inputs: [],
          outputs: [
            {
              id: 'deformation_data',
              name: 'InSAR Data',
              type: 'deformation-data',
              required: true,
              description: 'InSAR deformation measurements'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'deformation-analysis',
          name: 'Analyze Deformation',
          description: 'Calculate velocity field from deformation data',
          position: { x: 350, y: 125 },
          parameters: {
            analysisType: 'velocity',
            referenceFrame: 'ITRF2014',
            smoothingWindow: 30
          },
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
            }
          ],
          category: 'deformation-analysis',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'magma-chamber-modeling',
          name: 'Model Magma Chamber',
          description: 'Model magma chamber parameters from deformation',
          position: { x: 650, y: 125 },
          parameters: {
            modelType: 'mogi',
            depthRange: { min: 2, max: 8 },
            volumeRange: { min: 0.001, max: 0.1 }
          },
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
            }
          ],
          category: 'geological-analysis',
          status: 'idle'
        }
      ],
      connections: [
        {
          id: generateId(),
          sourceNodeId: 'deformation-data-loader-gps',
          sourcePortId: 'deformation_data',
          targetNodeId: 'deformation-analysis',
          targetPortId: 'deformation_data',
          dataType: 'deformation-data'
        },
        {
          id: generateId(),
          sourceNodeId: 'deformation-analysis',
          sourcePortId: 'velocity_field',
          targetNodeId: 'magma-chamber-modeling',
          targetPortId: 'deformation_data',
          dataType: 'spatial-data'
        }
      ]
    },
    author: 'Volcano Observatory',
    version: '1.2.0',
    createdAt: '2024-01-10T14:30:00Z',
    updatedAt: '2024-01-20T09:15:00Z'
  },
  
  {
    id: 'helios-ai-simulation',
    name: 'Helios AI Geological Simulation',
    description: 'Complete workflow using Helios AI for long-term geological evolution simulation',
    category: 'ai-modeling',
    tags: ['ai', 'helios', 'simulation', 'pinn', 'geological-evolution'],
    workflow: {
      nodes: [
        {
          id: generateId(),
          type: 'seismic-data-loader',
          name: 'Load Seismic Data',
          description: 'Load historical seismic data',
          position: { x: 50, y: 50 },
          parameters: {
            dataSource: 'usgs',
            timeRange: { start: '2020-01-01T00:00:00', end: '2024-01-01T00:00:00' },
            spatialExtent: { bounds: [-121, 35, -118, 37] },
            sampleRate: 100
          },
          inputs: [],
          outputs: [
            {
              id: 'seismic_data',
              name: 'Seismic Data',
              type: 'seismic-data',
              required: true,
              description: 'Historical seismic data'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'deformation-data-loader',
          name: 'Load Deformation Data',
          description: 'Load historical deformation data',
          position: { x: 50, y: 200 },
          parameters: {
            dataType: 'gps',
            timeRange: { start: '2020-01-01T00:00:00', end: '2024-01-01T00:00:00' },
            processingLevel: 'processed'
          },
          inputs: [],
          outputs: [
            {
              id: 'deformation_data',
              name: 'Deformation Data',
              type: 'deformation-data',
              required: true,
              description: 'Historical deformation data'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'helios-ai-simulation',
          name: 'Helios Simulation',
          description: 'Run Helios AI geological simulation',
          position: { x: 350, y: 125 },
          parameters: {
            simulationType: 'long-term',
            timeSpan: { start: '2024-01-01T00:00:00', end: '2034-01-01T00:00:00' },
            physicsConstraints: ['navier-stokes', 'heat-transfer', 'elastic-deformation'],
            uncertaintyQuantification: true
          },
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
          category: 'ai-modeling',
          status: 'idle'
        },
        {
          id: generateId(),
          type: '3d-geological-visualization',
          name: '3D Visualization',
          description: 'Create 3D visualization of simulation results',
          position: { x: 650, y: 125 },
          parameters: {
            renderMode: 'volume',
            colormap: 'viridis',
            opacity: 0.8
          },
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
          category: 'visualization',
          status: 'idle'
        }
      ],
      connections: [
        {
          id: generateId(),
          sourceNodeId: 'seismic-data-loader',
          sourcePortId: 'seismic_data',
          targetNodeId: 'helios-ai-simulation',
          targetPortId: 'geological_data',
          dataType: 'seismic-data'
        },
        {
          id: generateId(),
          sourceNodeId: 'helios-ai-simulation',
          sourcePortId: 'simulation_results',
          targetNodeId: '3d-geological-visualization',
          targetPortId: 'geological_model',
          dataType: 'geological-model'
        }
      ]
    },
    author: 'AI Research Team',
    version: '2.1.0',
    createdAt: '2024-01-05T16:45:00Z',
    updatedAt: '2024-01-25T11:20:00Z'
  },
  
  {
    id: 'volcanic-hazard-assessment',
    name: 'Volcanic Hazard Assessment',
    description: 'Comprehensive volcanic hazard assessment workflow with probabilistic risk mapping',
    category: 'hazard-assessment',
    tags: ['volcanic', 'hazard', 'risk', 'assessment', 'probabilistic'],
    workflow: {
      nodes: [
        {
          id: generateId(),
          type: 'seismic-data-loader',
          name: 'Load Seismic Data',
          description: 'Load recent seismic activity data',
          position: { x: 50, y: 50 },
          parameters: {
            dataSource: 'usgs',
            timeRange: { start: '2023-01-01T00:00:00', end: '2024-01-01T00:00:00' },
            spatialExtent: { bounds: [-122, 45, -121, 46] },
            sampleRate: 100
          },
          inputs: [],
          outputs: [
            {
              id: 'seismic_data',
              name: 'Seismic Data',
              type: 'seismic-data',
              required: true,
              description: 'Recent seismic activity'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'deformation-data-loader',
          name: 'Load Deformation Data',
          description: 'Load ground deformation measurements',
          position: { x: 50, y: 200 },
          parameters: {
            dataType: 'insar',
            timeRange: { start: '2023-01-01T00:00:00', end: '2024-01-01T00:00:00' },
            processingLevel: 'processed'
          },
          inputs: [],
          outputs: [
            {
              id: 'deformation_data',
              name: 'Deformation Data',
              type: 'deformation-data',
              required: true,
              description: 'Ground deformation data'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'geochemical-data-loader',
          name: 'Load Gas Data',
          description: 'Load volcanic gas emission data',
          position: { x: 50, y: 350 },
          parameters: {
            sampleType: 'gas',
            analysisType: ['volatiles'],
            spatialExtent: { bounds: [-122, 45, -121, 46] }
          },
          inputs: [],
          outputs: [
            {
              id: 'geochemical_data',
              name: 'Gas Data',
              type: 'geochemical-data',
              required: true,
              description: 'Volcanic gas emissions'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'hazard-map-generator',
          name: 'Generate Hazard Map',
          description: 'Create probabilistic volcanic hazard map',
          position: { x: 350, y: 200 },
          parameters: {
            hazardType: 'volcanic',
            probabilityLevels: [10, 50, 90],
            timeHorizon: '1-year'
          },
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
          category: 'visualization',
          status: 'idle'
        }
      ],
      connections: [
        {
          id: generateId(),
          sourceNodeId: 'seismic-data-loader',
          sourcePortId: 'seismic_data',
          targetNodeId: 'hazard-map-generator',
          targetPortId: 'hazard_data',
          dataType: 'seismic-data'
        }
      ]
    },
    author: 'Hazard Assessment Team',
    version: '1.5.0',
    createdAt: '2024-01-12T08:00:00Z',
    updatedAt: '2024-01-22T14:30:00Z'
  },
  
  {
    id: 'geochemical-analysis-workflow',
    name: 'Geochemical Analysis Workflow',
    description: 'Complete geochemical data analysis workflow for rock and gas samples',
    category: 'geochemical-analysis',
    tags: ['geochemical', 'analysis', 'rock', 'gas', 'composition'],
    workflow: {
      nodes: [
        {
          id: generateId(),
          type: 'geochemical-data-loader',
          name: 'Load Rock Samples',
          description: 'Load rock sample geochemical data',
          position: { x: 50, y: 100 },
          parameters: {
            sampleType: 'rock',
            analysisType: ['major_elements', 'trace_elements'],
            spatialExtent: { bounds: [-120, 35, -119, 36] }
          },
          inputs: [],
          outputs: [
            {
              id: 'geochemical_data',
              name: 'Rock Data',
              type: 'geochemical-data',
              required: true,
              description: 'Rock geochemical analysis'
            }
          ],
          category: 'data-input',
          status: 'idle'
        },
        {
          id: generateId(),
          type: 'geochemical-data-loader',
          name: 'Load Gas Samples',
          description: 'Load gas sample geochemical data',
          position: { x: 50, y: 250 },
          parameters: {
            sampleType: 'gas',
            analysisType: ['volatiles'],
            spatialExtent: { bounds: [-120, 35, -119, 36] }
          },
          inputs: [],
          outputs: [
            {
              id: 'geochemical_data',
              name: 'Gas Data',
              type: 'geochemical-data',
              required: true,
              description: 'Gas geochemical analysis'
            }
          ],
          category: 'data-input',
          status: 'idle'
        }
      ],
      connections: []
    },
    author: 'Geochemistry Lab',
    version: '1.0.0',
    createdAt: '2024-01-18T12:00:00Z',
    updatedAt: '2024-01-18T12:00:00Z'
  }
];