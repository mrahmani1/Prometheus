import React, { useCallback } from 'react';
import { Box } from '@mui/material';
import { WorkflowBuilder } from '@/components/workflow/WorkflowBuilder';
import { useWorkflowStore } from '@/store/workflow/workflowStore';

export const WorkflowsPage: React.FC = () => {
  const { addNode } = useWorkflowStore();

  const handleExecuteWorkflow = useCallback(async (workflow: any) => {
    // TODO: Implement workflow execution
    console.log('Executing workflow:', workflow);
    
    // This would typically send the workflow to the backend for execution
    // For now, we'll just log it
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log('Workflow execution completed');
        resolve(workflow);
      }, 2000);
    });
  }, []);

  const handleSaveWorkflow = useCallback(async (workflow: any) => {
    // TODO: Implement workflow saving
    console.log('Saving workflow:', workflow);
    
    // This would typically save the workflow to the backend
    // For now, we'll just log it
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log('Workflow saved successfully');
        resolve(workflow);
      }, 1000);
    });
  }, []);

  // Listen for add node events from drag and drop
  React.useEffect(() => {
    const handleAddNode = (event: CustomEvent) => {
      const { nodeType, position } = event.detail;
      addNode(nodeType, position);
    };

    window.addEventListener('addWorkflowNode', handleAddNode as EventListener);
    return () => {
      window.removeEventListener('addWorkflowNode', handleAddNode as EventListener);
    };
  }, [addNode]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <WorkflowBuilder
        onExecute={handleExecuteWorkflow}
        onSave={handleSaveWorkflow}
      />
    </Box>
  );
};