import React, { useState, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Toolbar,
  IconButton,
  Button,
  Divider,
  Alert,
  Snackbar
} from '@mui/material';
import {
  PlayArrow,
  Save,
  FileUpload,
  FileDownload,
  Undo,
  Redo,
  Delete,
  Settings
} from '@mui/icons-material';
import { WorkflowCanvas } from './WorkflowCanvas';
import { ComponentPalette } from './ComponentPalette';
import { ParameterEditor } from './ParameterEditor';
import { TemplateLibrary } from './TemplateLibrary';
import { WorkflowExporter } from './WorkflowExporter';
import { WorkflowImporter } from './WorkflowImporter';
import { useWorkflowStore } from '@/store/workflow/workflowStore';
import { WorkflowTemplate } from '@/types/workflow';

interface WorkflowBuilderProps {
  onExecute?: (workflow: any) => void;
  onSave?: (workflow: any) => void;
}

export const WorkflowBuilder: React.FC<WorkflowBuilderProps> = ({
  onExecute,
  onSave
}) => {
  const {
    nodes,
    connections,
    selectedNode,
    history,
    historyIndex,
    addNode,
    updateNode,
    deleteNode,
    addConnection,
    deleteConnection,
    selectNode,
    undo,
    redo,
    clearWorkflow,
    loadWorkflow,
    exportWorkflow,
    validateWorkflow
  } = useWorkflowStore();

  const [showTemplateLibrary, setShowTemplateLibrary] = useState(false);
  const [showExporter, setShowExporter] = useState(false);
  const [showImporter, setShowImporter] = useState(false);
  const [notification, setNotification] = useState<{
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExecuteWorkflow = useCallback(async () => {
    try {
      const validation = validateWorkflow();
      if (!validation.isValid) {
        setNotification({
          message: `Workflow validation failed: ${validation.errors.join(', ')}`,
          severity: 'error'
        });
        return;
      }

      const workflowData = exportWorkflow();
      await onExecute?.(workflowData);
      
      setNotification({
        message: 'Workflow executed successfully',
        severity: 'success'
      });
    } catch (error) {
      setNotification({
        message: `Execution failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  }, [validateWorkflow, exportWorkflow, onExecute]);

  const handleSaveWorkflow = useCallback(async () => {
    try {
      const workflowData = exportWorkflow();
      await onSave?.(workflowData);
      
      setNotification({
        message: 'Workflow saved successfully',
        severity: 'success'
      });
    } catch (error) {
      setNotification({
        message: `Save failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  }, [exportWorkflow, onSave]);

  const handleLoadTemplate = useCallback((template: WorkflowTemplate) => {
    try {
      loadWorkflow(template.workflow);
      setShowTemplateLibrary(false);
      setNotification({
        message: `Template "${template.name}" loaded successfully`,
        severity: 'success'
      });
    } catch (error) {
      setNotification({
        message: `Failed to load template: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  }, [loadWorkflow]);

  const handleImportWorkflow = useCallback((workflowData: any) => {
    try {
      loadWorkflow(workflowData);
      setShowImporter(false);
      setNotification({
        message: 'Workflow imported successfully',
        severity: 'success'
      });
    } catch (error) {
      setNotification({
        message: `Import failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        severity: 'error'
      });
    }
  }, [loadWorkflow]);

  const handleFileImport = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const workflowData = JSON.parse(e.target?.result as string);
        handleImportWorkflow(workflowData);
      } catch (error) {
        setNotification({
          message: 'Invalid workflow file format',
          severity: 'error'
        });
      }
    };
    reader.readAsText(file);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [handleImportWorkflow]);

  const canUndo = historyIndex > 0;
  const canRedo = historyIndex < history.length - 1;

  return (
    <Box sx={{ display: 'flex', height: '100%', flexDirection: 'column' }}>
      {/* Toolbar */}
      <Paper elevation={1}>
        <Toolbar variant="dense">
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Workflow Builder
          </Typography>
          
          <Button
            startIcon={<PlayArrow />}
            variant="contained"
            color="primary"
            onClick={handleExecuteWorkflow}
            disabled={nodes.length === 0}
            sx={{ mr: 1 }}
          >
            Execute
          </Button>
          
          <Button
            startIcon={<Save />}
            variant="outlined"
            onClick={handleSaveWorkflow}
            disabled={nodes.length === 0}
            sx={{ mr: 1 }}
          >
            Save
          </Button>
          
          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
          
          <IconButton
            onClick={undo}
            disabled={!canUndo}
            title="Undo"
          >
            <Undo />
          </IconButton>
          
          <IconButton
            onClick={redo}
            disabled={!canRedo}
            title="Redo"
          >
            <Redo />
          </IconButton>
          
          <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
          
          <IconButton
            onClick={() => setShowTemplateLibrary(true)}
            title="Template Library"
          >
            <Settings />
          </IconButton>
          
          <IconButton
            onClick={() => setShowExporter(true)}
            disabled={nodes.length === 0}
            title="Export Workflow"
          >
            <FileDownload />
          </IconButton>
          
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleFileImport}
          />
          
          <IconButton
            onClick={() => fileInputRef.current?.click()}
            title="Import Workflow"
          >
            <FileUpload />
          </IconButton>
          
          <IconButton
            onClick={clearWorkflow}
            disabled={nodes.length === 0}
            title="Clear Workflow"
            color="error"
          >
            <Delete />
          </IconButton>
        </Toolbar>
      </Paper>

      {/* Main Content */}
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Component Palette */}
        <Paper 
          elevation={1} 
          sx={{ 
            width: 280, 
            borderRadius: 0, 
            borderRight: 1, 
            borderColor: 'divider',
            overflow: 'auto'
          }}
        >
          <ComponentPalette onAddNode={addNode} />
        </Paper>

        {/* Workflow Canvas */}
        <Box sx={{ flex: 1, position: 'relative' }}>
          <WorkflowCanvas
            nodes={nodes}
            connections={connections}
            selectedNode={selectedNode}
            onNodeSelect={selectNode}
            onNodeUpdate={updateNode}
            onNodeDelete={deleteNode}
            onConnectionAdd={addConnection}
            onConnectionDelete={deleteConnection}
          />
        </Box>

        {/* Parameter Editor */}
        {selectedNode && (
          <Paper 
            elevation={1} 
            sx={{ 
              width: 320, 
              borderRadius: 0, 
              borderLeft: 1, 
              borderColor: 'divider',
              overflow: 'auto'
            }}
          >
            <ParameterEditor
              node={selectedNode}
              onNodeUpdate={updateNode}
            />
          </Paper>
        )}
      </Box>

      {/* Template Library Dialog */}
      <TemplateLibrary
        open={showTemplateLibrary}
        onClose={() => setShowTemplateLibrary(false)}
        onLoadTemplate={handleLoadTemplate}
      />

      {/* Export Dialog */}
      <WorkflowExporter
        open={showExporter}
        onClose={() => setShowExporter(false)}
        workflow={exportWorkflow()}
      />

      {/* Import Dialog */}
      <WorkflowImporter
        open={showImporter}
        onClose={() => setShowImporter(false)}
        onImport={handleImportWorkflow}
      />

      {/* Notifications */}
      <Snackbar
        open={!!notification}
        autoHideDuration={6000}
        onClose={() => setNotification(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setNotification(null)} 
          severity={notification?.severity || 'info'}
          sx={{ width: '100%' }}
        >
          {notification?.message || ''}
        </Alert>
      </Snackbar>
    </Box>
  );
};