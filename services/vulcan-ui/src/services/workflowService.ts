import apiClient from './authService';
import { GeologicalWorkflow, WorkflowStep } from '@/types/geological';

interface WorkflowExecutionStatus {
  id: string;
  isComplete: boolean;
  hasError: boolean;
  error?: string;
  progress: number;
  currentStep?: string;
  results: Record<string, any>;
}

export const workflowService = {
  // Workflow CRUD operations
  async getUserWorkflows(): Promise<GeologicalWorkflow[]> {
    const response = await apiClient.get('/workflows');
    return response.data;
  },

  async getWorkflow(id: string): Promise<GeologicalWorkflow> {
    const response = await apiClient.get(`/workflows/${id}`);
    return response.data;
  },

  async createWorkflow(workflow: Omit<GeologicalWorkflow, 'id' | 'createdAt' | 'updatedAt' | 'userId'>): Promise<GeologicalWorkflow> {
    const response = await apiClient.post('/workflows', workflow);
    return response.data;
  },

  async updateWorkflow(id: string, workflow: Partial<GeologicalWorkflow>): Promise<GeologicalWorkflow> {
    const response = await apiClient.put(`/workflows/${id}`, workflow);
    return response.data;
  },

  async deleteWorkflow(id: string): Promise<void> {
    await apiClient.delete(`/workflows/${id}`);
  },

  async duplicateWorkflow(id: string, newName: string): Promise<GeologicalWorkflow> {
    const response = await apiClient.post(`/workflows/${id}/duplicate`, { name: newName });
    return response.data;
  },

  // Workflow templates
  async getWorkflowTemplates(): Promise<GeologicalWorkflow[]> {
    const response = await apiClient.get('/workflows/templates');
    return response.data;
  },

  async createWorkflowFromTemplate(templateId: string, name: string): Promise<GeologicalWorkflow> {
    const response = await apiClient.post(`/workflows/templates/${templateId}/create`, { name });
    return response.data;
  },

  async saveAsTemplate(workflowId: string, templateName: string, description: string): Promise<GeologicalWorkflow> {
    const response = await apiClient.post(`/workflows/${workflowId}/save-as-template`, {
      name: templateName,
      description,
    });
    return response.data;
  },

  // Workflow execution
  async startWorkflowExecution(workflowId: string, parameters?: Record<string, any>): Promise<string> {
    const response = await apiClient.post(`/workflows/${workflowId}/execute`, { parameters });
    return response.data.executionId;
  },

  async getExecutionStatus(executionId: string): Promise<WorkflowExecutionStatus> {
    const response = await apiClient.get(`/workflows/executions/${executionId}/status`);
    return response.data;
  },

  async getExecutionResults(executionId: string): Promise<Record<string, any>> {
    const response = await apiClient.get(`/workflows/executions/${executionId}/results`);
    return response.data;
  },

  async cancelExecution(executionId: string): Promise<void> {
    await apiClient.post(`/workflows/executions/${executionId}/cancel`);
  },

  async getExecutionHistory(workflowId?: string): Promise<any[]> {
    const params = workflowId ? { workflowId } : {};
    const response = await apiClient.get('/workflows/executions/history', { params });
    return response.data;
  },

  // Step validation and testing
  async validateWorkflowStep(step: WorkflowStep): Promise<{ isValid: boolean; errors: string[] }> {
    const response = await apiClient.post('/workflows/steps/validate', step);
    return response.data;
  },

  async testWorkflowStep(step: WorkflowStep, testData: any): Promise<any> {
    const response = await apiClient.post('/workflows/steps/test', { step, testData });
    return response.data;
  },

  async getAvailableStepTypes(): Promise<any[]> {
    const response = await apiClient.get('/workflows/step-types');
    return response.data;
  },

  async getStepTypeSchema(stepType: string): Promise<any> {
    const response = await apiClient.get(`/workflows/step-types/${stepType}/schema`);
    return response.data;
  },

  // Workflow sharing and collaboration
  async shareWorkflow(workflowId: string, userEmails: string[], permissions: string[]): Promise<void> {
    await apiClient.post(`/workflows/${workflowId}/share`, {
      userEmails,
      permissions,
    });
  },

  async getWorkflowSharing(workflowId: string): Promise<any[]> {
    const response = await apiClient.get(`/workflows/${workflowId}/sharing`);
    return response.data;
  },

  async updateWorkflowSharing(workflowId: string, userId: string, permissions: string[]): Promise<void> {
    await apiClient.put(`/workflows/${workflowId}/sharing/${userId}`, { permissions });
  },

  async removeWorkflowSharing(workflowId: string, userId: string): Promise<void> {
    await apiClient.delete(`/workflows/${workflowId}/sharing/${userId}`);
  },

  // Workflow export/import
  async exportWorkflow(workflowId: string, format: 'json' | 'yaml'): Promise<Blob> {
    const response = await apiClient.get(`/workflows/${workflowId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },

  async importWorkflow(file: File): Promise<GeologicalWorkflow> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/workflows/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Workflow analytics
  async getWorkflowAnalytics(workflowId: string, timeRange?: [string, string]): Promise<any> {
    const params = timeRange ? { startDate: timeRange[0], endDate: timeRange[1] } : {};
    const response = await apiClient.get(`/workflows/${workflowId}/analytics`, { params });
    return response.data;
  },

  async getWorkflowPerformanceMetrics(workflowId: string): Promise<any> {
    const response = await apiClient.get(`/workflows/${workflowId}/performance`);
    return response.data;
  },

  // Workflow scheduling
  async scheduleWorkflow(workflowId: string, schedule: {
    type: 'once' | 'recurring';
    startTime: string;
    interval?: string;
    endTime?: string;
  }): Promise<string> {
    const response = await apiClient.post(`/workflows/${workflowId}/schedule`, schedule);
    return response.data.scheduleId;
  },

  async getWorkflowSchedules(workflowId: string): Promise<any[]> {
    const response = await apiClient.get(`/workflows/${workflowId}/schedules`);
    return response.data;
  },

  async updateWorkflowSchedule(scheduleId: string, schedule: any): Promise<void> {
    await apiClient.put(`/workflows/schedules/${scheduleId}`, schedule);
  },

  async deleteWorkflowSchedule(scheduleId: string): Promise<void> {
    await apiClient.delete(`/workflows/schedules/${scheduleId}`);
  },
};