import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { GeologicalWorkflow, WorkflowStep } from '@/types/geological';
import { workflowService } from '@/services/workflowService';

interface WorkflowState {
  workflows: GeologicalWorkflow[];
  activeWorkflow: GeologicalWorkflow | null;
  executionStatus: {
    isRunning: boolean;
    currentStep: string | null;
    progress: number;
    results: Record<string, any>;
  };
  templates: GeologicalWorkflow[];
  isLoading: boolean;
  error: string | null;
}

const initialState: WorkflowState = {
  workflows: [],
  activeWorkflow: null,
  executionStatus: {
    isRunning: false,
    currentStep: null,
    progress: 0,
    results: {},
  },
  templates: [],
  isLoading: false,
  error: null,
};

// Async thunks
export const fetchWorkflows = createAsyncThunk(
  'workflow/fetchWorkflows',
  async (_, { rejectWithValue }) => {
    try {
      return await workflowService.getUserWorkflows();
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch workflows');
    }
  }
);

export const fetchWorkflowTemplates = createAsyncThunk(
  'workflow/fetchWorkflowTemplates',
  async (_, { rejectWithValue }) => {
    try {
      return await workflowService.getWorkflowTemplates();
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch workflow templates');
    }
  }
);

export const saveWorkflow = createAsyncThunk(
  'workflow/saveWorkflow',
  async (workflow: GeologicalWorkflow, { rejectWithValue }) => {
    try {
      if (workflow.id) {
        return await workflowService.updateWorkflow(workflow.id, workflow);
      } else {
        return await workflowService.createWorkflow(workflow);
      }
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to save workflow');
    }
  }
);

export const executeWorkflow = createAsyncThunk(
  'workflow/executeWorkflow',
  async (workflowId: string, { rejectWithValue, dispatch }) => {
    try {
      const executionId = await workflowService.startWorkflowExecution(workflowId);
      
      // Poll for execution status
      const pollStatus = async () => {
        const status = await workflowService.getExecutionStatus(executionId);
        dispatch(updateExecutionStatus({
          currentStep: status.currentStep,
          progress: status.progress,
          results: status.results,
        }));
        
        if (status.isComplete) {
          return status.results;
        } else if (status.hasError) {
          throw new Error(status.error);
        } else {
          // Continue polling
          setTimeout(pollStatus, 1000);
        }
      };
      
      return await pollStatus();
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Workflow execution failed');
    }
  }
);

export const deleteWorkflow = createAsyncThunk(
  'workflow/deleteWorkflow',
  async (workflowId: string, { rejectWithValue }) => {
    try {
      await workflowService.deleteWorkflow(workflowId);
      return workflowId;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to delete workflow');
    }
  }
);

const workflowSlice = createSlice({
  name: 'workflow',
  initialState,
  reducers: {
    // Workflow management
    setActiveWorkflow: (state, action: PayloadAction<GeologicalWorkflow | null>) => {
      state.activeWorkflow = action.payload;
    },
    updateActiveWorkflow: (state, action: PayloadAction<Partial<GeologicalWorkflow>>) => {
      if (state.activeWorkflow) {
        state.activeWorkflow = { ...state.activeWorkflow, ...action.payload };
      }
    },
    
    // Step management
    addWorkflowStep: (state, action: PayloadAction<WorkflowStep>) => {
      if (state.activeWorkflow) {
        state.activeWorkflow.steps.push(action.payload);
      }
    },
    updateWorkflowStep: (state, action: PayloadAction<WorkflowStep>) => {
      if (state.activeWorkflow) {
        const index = state.activeWorkflow.steps.findIndex(step => step.id === action.payload.id);
        if (index !== -1) {
          state.activeWorkflow.steps[index] = action.payload;
        }
      }
    },
    removeWorkflowStep: (state, action: PayloadAction<string>) => {
      if (state.activeWorkflow) {
        state.activeWorkflow.steps = state.activeWorkflow.steps.filter(
          step => step.id !== action.payload
        );
        // Remove connections to this step
        state.activeWorkflow.steps.forEach(step => {
          step.connections = step.connections.filter(conn => conn !== action.payload);
        });
      }
    },
    connectWorkflowSteps: (state, action: PayloadAction<{ fromStepId: string; toStepId: string }>) => {
      if (state.activeWorkflow) {
        const fromStep = state.activeWorkflow.steps.find(step => step.id === action.payload.fromStepId);
        if (fromStep && !fromStep.connections.includes(action.payload.toStepId)) {
          fromStep.connections.push(action.payload.toStepId);
        }
      }
    },
    disconnectWorkflowSteps: (state, action: PayloadAction<{ fromStepId: string; toStepId: string }>) => {
      if (state.activeWorkflow) {
        const fromStep = state.activeWorkflow.steps.find(step => step.id === action.payload.fromStepId);
        if (fromStep) {
          fromStep.connections = fromStep.connections.filter(conn => conn !== action.payload.toStepId);
        }
      }
    },
    
    // Execution management
    updateExecutionStatus: (state, action: PayloadAction<Partial<typeof initialState.executionStatus>>) => {
      state.executionStatus = { ...state.executionStatus, ...action.payload };
    },
    stopWorkflowExecution: (state) => {
      state.executionStatus.isRunning = false;
      state.executionStatus.currentStep = null;
    },
    clearExecutionResults: (state) => {
      state.executionStatus.results = {};
      state.executionStatus.progress = 0;
    },
    
    // Utility actions
    createNewWorkflow: (state, action: PayloadAction<{ name: string; description: string }>) => {
      const newWorkflow: GeologicalWorkflow = {
        id: '',
        name: action.payload.name,
        description: action.payload.description,
        steps: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        userId: '', // Will be set by the service
      };
      state.activeWorkflow = newWorkflow;
    },
    
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch workflows
      .addCase(fetchWorkflows.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchWorkflows.fulfilled, (state, action) => {
        state.isLoading = false;
        state.workflows = action.payload;
      })
      .addCase(fetchWorkflows.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      
      // Fetch templates
      .addCase(fetchWorkflowTemplates.fulfilled, (state, action) => {
        state.templates = action.payload;
      })
      
      // Save workflow
      .addCase(saveWorkflow.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(saveWorkflow.fulfilled, (state, action) => {
        state.isLoading = false;
        const savedWorkflow = action.payload;
        
        // Update workflows list
        const existingIndex = state.workflows.findIndex(w => w.id === savedWorkflow.id);
        if (existingIndex !== -1) {
          state.workflows[existingIndex] = savedWorkflow;
        } else {
          state.workflows.push(savedWorkflow);
        }
        
        // Update active workflow if it's the same one
        if (state.activeWorkflow?.id === savedWorkflow.id || !state.activeWorkflow?.id) {
          state.activeWorkflow = savedWorkflow;
        }
      })
      .addCase(saveWorkflow.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      
      // Execute workflow
      .addCase(executeWorkflow.pending, (state) => {
        state.executionStatus.isRunning = true;
        state.executionStatus.progress = 0;
        state.error = null;
      })
      .addCase(executeWorkflow.fulfilled, (state, action) => {
        state.executionStatus.isRunning = false;
        state.executionStatus.progress = 100;
        state.executionStatus.results = action.payload || {};
      })
      .addCase(executeWorkflow.rejected, (state, action) => {
        state.executionStatus.isRunning = false;
        state.error = action.payload as string;
      })
      
      // Delete workflow
      .addCase(deleteWorkflow.fulfilled, (state, action) => {
        state.workflows = state.workflows.filter(w => w.id !== action.payload);
        if (state.activeWorkflow?.id === action.payload) {
          state.activeWorkflow = null;
        }
      });
  },
});

export const {
  setActiveWorkflow,
  updateActiveWorkflow,
  addWorkflowStep,
  updateWorkflowStep,
  removeWorkflowStep,
  connectWorkflowSteps,
  disconnectWorkflowSteps,
  updateExecutionStatus,
  stopWorkflowExecution,
  clearExecutionResults,
  createNewWorkflow,
  clearError,
} = workflowSlice.actions;

export default workflowSlice.reducer;