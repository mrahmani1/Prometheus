import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  WorkflowNode, 
  WorkflowConnection, 
  WorkflowValidationResult,
  WorkflowNodeDefinition 
} from '@/types/workflow';
import { generateId } from '@/utils/idGenerator';

interface WorkflowState {
  // Current workflow state
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  selectedNode: WorkflowNode | null;
  
  // History for undo/redo
  history: Array<{
    nodes: WorkflowNode[];
    connections: WorkflowConnection[];
  }>;
  historyIndex: number;
  maxHistorySize: number;
  
  // Node definitions
  nodeDefinitions: Record<string, WorkflowNodeDefinition>;
  
  // Actions
  addNode: (nodeType: string, position: { x: number; y: number }) => void;
  updateNode: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  deleteNode: (nodeId: string) => void;
  selectNode: (nodeId: string | null) => void;
  
  addConnection: (connection: Omit<WorkflowConnection, 'id'>) => void;
  deleteConnection: (connectionId: string) => void;
  
  undo: () => void;
  redo: () => void;
  
  clearWorkflow: () => void;
  loadWorkflow: (workflow: { nodes: WorkflowNode[]; connections: WorkflowConnection[] }) => void;
  exportWorkflow: () => { nodes: WorkflowNode[]; connections: WorkflowConnection[] };
  
  validateWorkflow: () => WorkflowValidationResult;
  
  // Node definitions management
  registerNodeDefinition: (definition: WorkflowNodeDefinition) => void;
  getNodeDefinition: (nodeType: string) => WorkflowNodeDefinition | undefined;
}

const createInitialState = () => ({
  nodes: [],
  connections: [],
  selectedNode: null,
  history: [{ nodes: [], connections: [] }],
  historyIndex: 0,
  maxHistorySize: 50,
  nodeDefinitions: {}
});

export const useWorkflowStore = create<WorkflowState>()(
  devtools(
    (set, get) => ({
      ...createInitialState(),
      
      addNode: (nodeType: string, position: { x: number; y: number }) => {
        const definition = get().nodeDefinitions[nodeType];
        if (!definition) {
          console.error(`Node definition not found for type: ${nodeType}`);
          return;
        }
        
        const newNode: WorkflowNode = {
          id: generateId(),
          type: nodeType,
          name: definition.name,
          description: definition.description,
          position,
          parameters: definition.parameters.reduce((acc, param) => {
            acc[param.id] = param.defaultValue;
            return acc;
          }, {} as Record<string, any>),
          inputs: [...definition.inputs],
          outputs: [...definition.outputs],
          category: definition.category,
          status: 'idle'
        };
        
        set((state) => {
          const newNodes = [...state.nodes, newNode];
          const newState = {
            nodes: newNodes,
            connections: state.connections,
            selectedNode: newNode
          };
          
          return {
            ...newState,
            ...addToHistory(state, newState)
          };
        });
      },
      
      updateNode: (nodeId: string, updates: Partial<WorkflowNode>) => {
        set((state) => {
          const nodeIndex = state.nodes.findIndex(n => n.id === nodeId);
          if (nodeIndex === -1) return state;
          
          const updatedNodes = [...state.nodes];
          updatedNodes[nodeIndex] = { ...updatedNodes[nodeIndex], ...updates };
          
          const newState = {
            nodes: updatedNodes,
            connections: state.connections,
            selectedNode: state.selectedNode?.id === nodeId 
              ? updatedNodes[nodeIndex] 
              : state.selectedNode
          };
          
          return {
            ...newState,
            ...addToHistory(state, newState)
          };
        });
      },
      
      deleteNode: (nodeId: string) => {
        set((state) => {
          const newNodes = state.nodes.filter(n => n.id !== nodeId);
          const newConnections = state.connections.filter(
            c => c.sourceNodeId !== nodeId && c.targetNodeId !== nodeId
          );
          
          const newState = {
            nodes: newNodes,
            connections: newConnections,
            selectedNode: state.selectedNode?.id === nodeId ? null : state.selectedNode
          };
          
          return {
            ...newState,
            ...addToHistory(state, newState)
          };
        });
      },
      
      selectNode: (nodeId: string | null) => {
        set((state) => ({
          selectedNode: nodeId ? state.nodes.find(n => n.id === nodeId) || null : null
        }));
      },
      
      addConnection: (connection: Omit<WorkflowConnection, 'id'>) => {
        // Validate connection
        const { nodes } = get();
        const sourceNode = nodes.find(n => n.id === connection.sourceNodeId);
        const targetNode = nodes.find(n => n.id === connection.targetNodeId);
        
        if (!sourceNode || !targetNode) return;
        
        const sourcePort = sourceNode.outputs.find(p => p.id === connection.sourcePortId);
        const targetPort = targetNode.inputs.find(p => p.id === connection.targetPortId);
        
        if (!sourcePort || !targetPort) return;
        
        // Check if connection already exists
        const existingConnection = get().connections.find(
          c => c.sourceNodeId === connection.sourceNodeId &&
               c.sourcePortId === connection.sourcePortId &&
               c.targetNodeId === connection.targetNodeId &&
               c.targetPortId === connection.targetPortId
        );
        
        if (existingConnection) return;
        
        const newConnection: WorkflowConnection = {
          ...connection,
          id: generateId(),
          dataType: sourcePort.type
        };
        
        set((state) => {
          const newConnections = [...state.connections, newConnection];
          const newState = {
            nodes: state.nodes,
            connections: newConnections
          };
          
          return {
            ...state,
            ...newState,
            ...addToHistory(state, newState)
          };
        });
      },
      
      deleteConnection: (connectionId: string) => {
        set((state) => {
          const newConnections = state.connections.filter(c => c.id !== connectionId);
          const newState = {
            nodes: state.nodes,
            connections: newConnections
          };
          
          return {
            ...state,
            ...newState,
            ...addToHistory(state, newState)
          };
        });
      },
      
      undo: () => {
        set((state) => {
          if (state.historyIndex <= 0) return state;
          
          const newIndex = state.historyIndex - 1;
          const historyState = state.history[newIndex];
          
          return {
            ...state,
            nodes: historyState.nodes,
            connections: historyState.connections,
            historyIndex: newIndex,
            selectedNode: null
          };
        });
      },
      
      redo: () => {
        set((state) => {
          if (state.historyIndex >= state.history.length - 1) return state;
          
          const newIndex = state.historyIndex + 1;
          const historyState = state.history[newIndex];
          
          return {
            ...state,
            nodes: historyState.nodes,
            connections: historyState.connections,
            historyIndex: newIndex,
            selectedNode: null
          };
        });
      },
      
      clearWorkflow: () => {
        set((state) => {
          const newState = {
            nodes: [],
            connections: []
          };
          
          return {
            ...state,
            ...newState,
            selectedNode: null,
            ...addToHistory(state, newState)
          };
        });
      },
      
      loadWorkflow: (workflow: { nodes: WorkflowNode[]; connections: WorkflowConnection[] }) => {
        set((state) => ({
          ...state,
          nodes: workflow.nodes,
          connections: workflow.connections,
          selectedNode: null,
          ...addToHistory(state, workflow)
        }));
      },
      
      exportWorkflow: () => {
        const { nodes, connections } = get();
        return { nodes, connections };
      },
      
      validateWorkflow: (): WorkflowValidationResult => {
        const { nodes, connections } = get();
        const errors: string[] = [];
        const warnings: string[] = [];
        
        // Check for disconnected nodes
        const connectedNodeIds = new Set([
          ...connections.map(c => c.sourceNodeId),
          ...connections.map(c => c.targetNodeId)
        ]);
        
        nodes.forEach(node => {
          if (!connectedNodeIds.has(node.id) && nodes.length > 1) {
            warnings.push(`Node "${node.name}" is not connected to any other nodes`);
          }
          
          // Validate required parameters
          const definition = get().nodeDefinitions[node.type];
          if (definition) {
            definition.parameters.forEach(param => {
              if (param.required && (node.parameters[param.id] === undefined || node.parameters[param.id] === null)) {
                errors.push(`Node "${node.name}" is missing required parameter "${param.name}"`);
              }
              
              // Run custom validation if available
              if (param.validation?.custom && node.parameters[param.id] !== undefined) {
                const validationError = param.validation.custom(node.parameters[param.id]);
                if (validationError) {
                  errors.push(`Node "${node.name}" parameter "${param.name}": ${validationError}`);
                }
              }
            });
          }
        });
        
        // Check for circular dependencies
        const visited = new Set<string>();
        const recursionStack = new Set<string>();
        
        const hasCycle = (nodeId: string): boolean => {
          if (recursionStack.has(nodeId)) return true;
          if (visited.has(nodeId)) return false;
          
          visited.add(nodeId);
          recursionStack.add(nodeId);
          
          const outgoingConnections = connections.filter(c => c.sourceNodeId === nodeId);
          for (const connection of outgoingConnections) {
            if (hasCycle(connection.targetNodeId)) return true;
          }
          
          recursionStack.delete(nodeId);
          return false;
        };
        
        for (const node of nodes) {
          if (hasCycle(node.id)) {
            errors.push('Workflow contains circular dependencies');
            break;
          }
        }
        
        return {
          isValid: errors.length === 0,
          errors,
          warnings
        };
      },
      
      registerNodeDefinition: (definition: WorkflowNodeDefinition) => {
        set((state) => ({
          nodeDefinitions: {
            ...state.nodeDefinitions,
            [definition.type]: definition
          }
        }));
      },
      
      getNodeDefinition: (nodeType: string) => {
        return get().nodeDefinitions[nodeType];
      }
    }),
    {
      name: 'workflow-store'
    }
  )
);

// Helper function to add state to history
function addToHistory(
  currentState: WorkflowState, 
  newState: { nodes: WorkflowNode[]; connections: WorkflowConnection[] }
) {
  const newHistory = currentState.history.slice(0, currentState.historyIndex + 1);
  newHistory.push({
    nodes: [...newState.nodes],
    connections: [...newState.connections]
  });
  
  // Limit history size
  if (newHistory.length > currentState.maxHistorySize) {
    newHistory.shift();
  }
  
  return {
    history: newHistory,
    historyIndex: newHistory.length - 1
  };
}