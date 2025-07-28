import React, { useState, useCallback } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  Alert,
  Paper,
  LinearProgress,
  Chip,
  Divider
} from '@mui/material';
import {
  FileUpload,
  CheckCircle,
  Error as ErrorIcon,
  Warning
} from '@mui/icons-material';

interface WorkflowImporterProps {
  open: boolean;
  onClose: () => void;
  onImport: (workflowData: any) => void;
}

interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  metadata?: any;
}

export const WorkflowImporter: React.FC<WorkflowImporterProps> = ({
  open,
  onClose,
  onImport
}) => {
  const [importData, setImportData] = useState('');
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const validateWorkflowData = useCallback((data: string): ValidationResult => {
    try {
      const parsed = JSON.parse(data);
      const errors: string[] = [];
      const warnings: string[] = [];

      // Check required top-level structure
      if (!parsed.workflow) {
        errors.push('Missing "workflow" property');
      } else {
        if (!Array.isArray(parsed.workflow.nodes)) {
          errors.push('Workflow must contain a "nodes" array');
        } else if (parsed.workflow.nodes.length === 0) {
          warnings.push('Workflow contains no nodes');
        }

        if (!Array.isArray(parsed.workflow.connections)) {
          errors.push('Workflow must contain a "connections" array');
        }
      }

      // Validate nodes
      if (parsed.workflow?.nodes) {
        parsed.workflow.nodes.forEach((node: any, index: number) => {
          if (!node.id) {
            errors.push(`Node ${index} is missing required "id" property`);
          }
          if (!node.type) {
            errors.push(`Node ${index} is missing required "type" property`);
          }
          if (!node.name) {
            warnings.push(`Node ${index} is missing "name" property`);
          }
          if (!node.position || typeof node.position.x !== 'number' || typeof node.position.y !== 'number') {
            errors.push(`Node ${index} has invalid position`);
          }
          if (!node.parameters || typeof node.parameters !== 'object') {
            warnings.push(`Node ${index} has no parameters`);
          }
        });
      }

      // Validate connections
      if (parsed.workflow?.connections) {
        parsed.workflow.connections.forEach((connection: any, index: number) => {
          if (!connection.id) {
            errors.push(`Connection ${index} is missing required "id" property`);
          }
          if (!connection.sourceNodeId) {
            errors.push(`Connection ${index} is missing "sourceNodeId"`);
          }
          if (!connection.targetNodeId) {
            errors.push(`Connection ${index} is missing "targetNodeId"`);
          }
          if (!connection.sourcePortId) {
            errors.push(`Connection ${index} is missing "sourcePortId"`);
          }
          if (!connection.targetPortId) {
            errors.push(`Connection ${index} is missing "targetPortId"`);
          }

          // Check if referenced nodes exist
          if (parsed.workflow.nodes) {
            const sourceExists = parsed.workflow.nodes.some((n: any) => n.id === connection.sourceNodeId);
            const targetExists = parsed.workflow.nodes.some((n: any) => n.id === connection.targetNodeId);
            
            if (!sourceExists) {
              errors.push(`Connection ${index} references non-existent source node: ${connection.sourceNodeId}`);
            }
            if (!targetExists) {
              errors.push(`Connection ${index} references non-existent target node: ${connection.targetNodeId}`);
            }
          }
        });
      }

      // Check for version compatibility
      if (parsed.version && parsed.version !== '1.0.0') {
        warnings.push(`Workflow version ${parsed.version} may not be fully compatible`);
      }

      // Check for circular dependencies
      if (parsed.workflow?.nodes && parsed.workflow?.connections) {
        const hasCircularDependency = checkCircularDependencies(
          parsed.workflow.nodes,
          parsed.workflow.connections
        );
        if (hasCircularDependency) {
          errors.push('Workflow contains circular dependencies');
        }
      }

      return {
        isValid: errors.length === 0,
        errors,
        warnings,
        metadata: parsed.metadata
      };
    } catch (error) {
      return {
        isValid: false,
        errors: ['Invalid JSON format'],
        warnings: []
      };
    }
  }, []);

  const checkCircularDependencies = (nodes: any[], connections: any[]): boolean => {
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
      if (hasCycle(node.id)) return true;
    }

    return false;
  };

  const handleDataChange = useCallback((data: string) => {
    setImportData(data);
    
    if (data.trim()) {
      setIsValidating(true);
      // Simulate async validation
      setTimeout(() => {
        const result = validateWorkflowData(data);
        setValidationResult(result);
        setIsValidating(false);
      }, 500);
    } else {
      setValidationResult(null);
    }
  }, [validateWorkflowData]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      handleDataChange(content);
    };
    reader.readAsText(file);
  }, [handleDataChange]);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);

    const file = event.dataTransfer.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      handleDataChange(content);
    };
    reader.readAsText(file);
  }, [handleDataChange]);

  const handleImport = () => {
    if (validationResult?.isValid && importData) {
      try {
        const workflowData = JSON.parse(importData);
        onImport(workflowData.workflow);
      } catch (error) {
        console.error('Import failed:', error);
      }
    }
  };

  const handleClose = () => {
    setImportData('');
    setValidationResult(null);
    setIsValidating(false);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FileUpload />
          <Typography variant="h6">Import Workflow</Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {/* File Upload Area */}
        <Paper
          variant="outlined"
          sx={{
            p: 3,
            textAlign: 'center',
            border: dragOver ? '2px dashed #1976d2' : '2px dashed #ccc',
            backgroundColor: dragOver ? '#f3f8ff' : 'transparent',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <FileUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
          <Typography variant="h6" gutterBottom>
            Drop workflow file here or click to browse
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Supports JSON and YAML workflow files
          </Typography>
          <input
            id="file-input"
            type="file"
            accept=".json,.yaml,.yml"
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
        </Paper>

        <Divider>
          <Typography variant="body2" color="text.secondary">
            OR
          </Typography>
        </Divider>

        {/* Manual Input */}
        <TextField
          fullWidth
          multiline
          rows={8}
          label="Paste workflow JSON/YAML"
          value={importData}
          onChange={(e) => handleDataChange(e.target.value)}
          placeholder="Paste your workflow data here..."
          variant="outlined"
        />

        {/* Validation Progress */}
        {isValidating && (
          <Box>
            <Typography variant="body2" gutterBottom>
              Validating workflow...
            </Typography>
            <LinearProgress />
          </Box>
        )}

        {/* Validation Results */}
        {validationResult && !isValidating && (
          <Box>
            {validationResult.isValid ? (
              <Alert severity="success" icon={<CheckCircle />}>
                <Typography variant="body2" fontWeight="bold">
                  Workflow is valid and ready to import
                </Typography>
                {validationResult.metadata && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="caption" display="block">
                      Exported: {new Date(validationResult.metadata.exportedAt).toLocaleString()}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Nodes: {validationResult.metadata.nodeCount}, Connections: {validationResult.metadata.connectionCount}
                    </Typography>
                  </Box>
                )}
              </Alert>
            ) : (
              <Alert severity="error" icon={<ErrorIcon />}>
                <Typography variant="body2" fontWeight="bold">
                  Workflow validation failed
                </Typography>
              </Alert>
            )}

            {/* Errors */}
            {validationResult.errors.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="subtitle2" color="error" gutterBottom>
                  Errors:
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {validationResult.errors.map((error, index) => (
                    <Chip
                      key={index}
                      label={error}
                      size="small"
                      color="error"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            )}

            {/* Warnings */}
            {validationResult.warnings.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="subtitle2" color="warning.main" gutterBottom>
                  Warnings:
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {validationResult.warnings.map((warning, index) => (
                    <Chip
                      key={index}
                      label={warning}
                      size="small"
                      color="warning"
                      variant="outlined"
                      icon={<Warning />}
                    />
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Button onClick={handleClose}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleImport}
          disabled={!validationResult?.isValid || !importData.trim()}
          startIcon={<FileUpload />}
        >
          Import Workflow
        </Button>
      </DialogActions>
    </Dialog>
  );
};