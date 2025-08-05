import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  // TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  Tabs,
  Tab,
  Paper
} from '@mui/material';
import {
  FileDownload,
  Share,
  Code,
  Description
} from '@mui/icons-material';

interface WorkflowExporterProps {
  open: boolean;
  onClose: () => void;
  workflow: { nodes: any[]; connections: any[] };
}

export const WorkflowExporter: React.FC<WorkflowExporterProps> = ({
  open,
  onClose,
  workflow
}) => {
  const [exportFormat, setExportFormat] = useState('json');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  const generateExportData = () => {
    const exportData = {
      version: '1.0.0',
      type: 'prometheus-workflow',
      metadata: includeMetadata ? {
        exportedAt: new Date().toISOString(),
        exportedBy: 'Vulcan UI',
        nodeCount: workflow.nodes.length,
        connectionCount: workflow.connections.length
      } : undefined,
      workflow: {
        nodes: workflow.nodes,
        connections: workflow.connections
      }
    };

    switch (exportFormat) {
      case 'json':
        return JSON.stringify(exportData, null, 2);
      case 'yaml':
        return convertToYAML(exportData);
      case 'python':
        return generatePythonCode(exportData);
      default:
        return JSON.stringify(exportData, null, 2);
    }
  };

  const convertToYAML = (data: any): string => {
    // Simple YAML conversion - in a real implementation, use a proper YAML library
    const yamlLines: string[] = [];
    
    const addToYaml = (obj: any, indent = 0) => {
      const spaces = '  '.repeat(indent);
      
      if (Array.isArray(obj)) {
        obj.forEach(item => {
          yamlLines.push(`${spaces}- `);
          if (typeof item === 'object') {
            addToYaml(item, indent + 1);
          } else {
            yamlLines[yamlLines.length - 1] += String(item);
          }
        });
      } else if (typeof obj === 'object' && obj !== null) {
        Object.entries(obj).forEach(([key, value]) => {
          if (Array.isArray(value)) {
            yamlLines.push(`${spaces}${key}:`);
            addToYaml(value, indent + 1);
          } else if (typeof value === 'object' && value !== null) {
            yamlLines.push(`${spaces}${key}:`);
            addToYaml(value, indent + 1);
          } else {
            yamlLines.push(`${spaces}${key}: ${String(value)}`);
          }
        });
      }
    };
    
    addToYaml(data);
    return yamlLines.join('\n');
  };

  const generatePythonCode = (data: any): string => {
    return `#!/usr/bin/env python3
"""
Prometheus AI Workflow - Generated Python Code
Generated at: ${new Date().toISOString()}
"""

import json
from typing import Dict, List, Any

# Workflow Definition
workflow_data = ${JSON.stringify(data, null, 4)}

class PrometheusWorkflow:
    def __init__(self, workflow_data: Dict[str, Any]):
        self.workflow_data = workflow_data
        self.nodes = workflow_data['workflow']['nodes']
        self.connections = workflow_data['workflow']['connections']
    
    def execute(self):
        """Execute the workflow"""
        print(f"Executing workflow with {len(self.nodes)} nodes and {len(self.connections)} connections")
        
        # TODO: Implement workflow execution logic
        for node in self.nodes:
            print(f"Processing node: {node['name']} ({node['type']})")
            # Add node processing logic here
        
        for connection in self.connections:
            print(f"Processing connection: {connection['sourceNodeId']} -> {connection['targetNodeId']}")
            # Add connection processing logic here
    
    def validate(self) -> bool:
        """Validate the workflow structure"""
        # TODO: Implement validation logic
        return True
    
    def get_node_by_id(self, node_id: str) -> Dict[str, Any]:
        """Get a node by its ID"""
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        raise ValueError(f"Node with ID {node_id} not found")

if __name__ == "__main__":
    # Create and execute workflow
    workflow = PrometheusWorkflow(workflow_data)
    
    if workflow.validate():
        workflow.execute()
    else:
        print("Workflow validation failed")
`;
  };

  const handleDownload = () => {
    const data = generateExportData();
    const filename = `prometheus-workflow.${exportFormat === 'python' ? 'py' : exportFormat}`;
    const mimeType = {
      json: 'application/json',
      yaml: 'application/x-yaml',
      python: 'text/x-python'
    }[exportFormat] || 'text/plain';

    const blob = new Blob([data], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleCopyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generateExportData());
      // You might want to show a success message here
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const exportData = generateExportData();

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FileDownload />
          <Typography variant="h6">Export Workflow</Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(_: any, value: any) => setActiveTab(value)}>
            <Tab label="Export Options" />
            <Tab label="Preview" />
          </Tabs>
        </Box>

        {activeTab === 0 && (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Export Configuration
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <FormControl fullWidth>
                <InputLabel>Export Format</InputLabel>
                <Select
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                  label="Export Format"
                >
                  <MenuItem value="json">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Code />
                      JSON Format
                    </Box>
                  </MenuItem>
                  <MenuItem value="yaml">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Description />
                      YAML Format
                    </Box>
                  </MenuItem>
                  <MenuItem value="python">
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Code />
                      Python Code
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Export Options
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  <Chip
                    label="Include Metadata"
                    variant={includeMetadata ? 'filled' : 'outlined'}
                    onClick={() => setIncludeMetadata(!includeMetadata)}
                    clickable
                  />
                </Box>
              </Box>

              <Alert severity="info">
                <Typography variant="body2">
                  <strong>Export Formats:</strong>
                  <br />
                  • <strong>JSON:</strong> Standard format for importing back into Vulcan UI
                  <br />
                  • <strong>YAML:</strong> Human-readable format for documentation
                  <br />
                  • <strong>Python:</strong> Executable Python code template
                </Typography>
              </Alert>

              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Workflow Summary
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Chip label={`${workflow.nodes.length} Nodes`} variant="outlined" />
                  <Chip label={`${workflow.connections.length} Connections`} variant="outlined" />
                </Box>
              </Box>
            </Box>
          </Box>
        )}

        {activeTab === 1 && (
          <Box sx={{ p: 3, height: 'calc(100% - 48px)' }}>
            <Typography variant="h6" gutterBottom>
              Export Preview
            </Typography>
            
            <Paper
              variant="outlined"
              sx={{
                height: 'calc(100% - 40px)',
                overflow: 'auto',
                p: 2,
                backgroundColor: '#f5f5f5',
                fontFamily: 'monospace',
                fontSize: '0.875rem'
              }}
            >
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {exportData}
              </pre>
            </Paper>
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Button onClick={handleCopyToClipboard} startIcon={<Share />}>
          Copy to Clipboard
        </Button>
        <Button
          variant="contained"
          onClick={handleDownload}
          startIcon={<FileDownload />}
        >
          Download File
        </Button>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};