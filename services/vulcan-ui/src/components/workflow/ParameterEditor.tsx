import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Slider,
  Chip,
  // Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  // Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ExpandMore,
  Info,
  Refresh,
  Check,
  Error as ErrorIcon
} from '@mui/icons-material';
// import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';
import { WorkflowNode, WorkflowParameter } from '@/types/workflow';
import { useWorkflowStore } from '@/store/workflow/workflowStore';

interface ParameterEditorProps {
  node: WorkflowNode;
  onNodeUpdate: (nodeId: string, updates: Partial<WorkflowNode>) => void;
}

export const ParameterEditor: React.FC<ParameterEditorProps> = ({
  node,
  onNodeUpdate
}) => {
  const { getNodeDefinition } = useWorkflowStore();
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['general']));

  const nodeDefinition = getNodeDefinition(node.type);
  
  const parameterGroups = useMemo(() => {
    if (!nodeDefinition) return {};
    
    const groups: Record<string, WorkflowParameter[]> = {};
    
    nodeDefinition.parameters.forEach(param => {
      const group = param.group || 'general';
      if (!groups[group]) {
        groups[group] = [];
      }
      groups[group].push(param);
    });
    
    return groups;
  }, [nodeDefinition]);

  const handleParameterChange = useCallback((parameterId: string, value: any) => {
    const newParameters = { ...node.parameters, [parameterId]: value };
    
    // Validate parameter
    const parameter = nodeDefinition?.parameters.find(p => p.id === parameterId);
    if (parameter?.validation) {
      const error = validateParameter(parameter, value);
      setValidationErrors(prev => ({
        ...prev,
        [parameterId]: error || ''
      }));
    }
    
    onNodeUpdate(node.id, { parameters: newParameters });
  }, [node.id, node.parameters, nodeDefinition, onNodeUpdate]);

  const validateParameter = (parameter: WorkflowParameter, value: any): string | null => {
    if (parameter.required && (value === undefined || value === null || value === '')) {
      return 'This parameter is required';
    }
    
    if (parameter.validation) {
      const { min, max, pattern, custom } = parameter.validation;
      
      if (typeof value === 'number') {
        if (min !== undefined && value < min) {
          return `Value must be at least ${min}`;
        }
        if (max !== undefined && value > max) {
          return `Value must be at most ${max}`;
        }
      }
      
      if (typeof value === 'string' && pattern) {
        const regex = new RegExp(pattern);
        if (!regex.test(value)) {
          return 'Value does not match required pattern';
        }
      }
      
      if (custom) {
        return custom(value);
      }
    }
    
    return null;
  };

  const handleGroupToggle = (group: string) => {
    const newExpanded = new Set(expandedGroups);
    if (newExpanded.has(group)) {
      newExpanded.delete(group);
    } else {
      newExpanded.add(group);
    }
    setExpandedGroups(newExpanded);
  };

  const resetToDefaults = () => {
    if (!nodeDefinition) return;
    
    const defaultParameters = nodeDefinition.parameters.reduce((acc, param) => {
      acc[param.id] = param.defaultValue;
      return acc;
    }, {} as Record<string, any>);
    
    onNodeUpdate(node.id, { parameters: defaultParameters });
    setValidationErrors({});
  };

  const renderParameterInput = (parameter: WorkflowParameter) => {
    const value = node.parameters[parameter.id] ?? parameter.defaultValue;
    const error = validationErrors[parameter.id];
    const hasError = !!error;

    switch (parameter.type) {
      case 'string':
        return (
          <TextField
            fullWidth
            size="small"
            value={value || ''}
            onChange={(e) => handleParameterChange(parameter.id, e.target.value)}
            error={hasError}
            helperText={error || parameter.description}
            placeholder={parameter.defaultValue}
          />
        );

      case 'number':
        return (
          <TextField
            fullWidth
            size="small"
            type="number"
            value={value ?? ''}
            onChange={(e) => handleParameterChange(parameter.id, parseFloat(e.target.value) || 0)}
            error={hasError}
            helperText={error || parameter.description}
            inputProps={{
              min: parameter.validation?.min,
              max: parameter.validation?.max
            }}
          />
        );

      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={value || false}
                onChange={(e) => handleParameterChange(parameter.id, e.target.checked)}
              />
            }
            label={parameter.description}
          />
        );

      case 'select':
        return (
          <FormControl fullWidth size="small" error={hasError}>
            <InputLabel>{parameter.name}</InputLabel>
            <Select
              value={value || ''}
              onChange={(e) => handleParameterChange(parameter.id, e.target.value)}
              label={parameter.name}
            >
              {parameter.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            {(error || parameter.description) && (
              <Typography variant="caption" color={hasError ? 'error' : 'text.secondary'} sx={{ mt: 0.5 }}>
                {error || parameter.description}
              </Typography>
            )}
          </FormControl>
        );

      case 'multi-select':
        return (
          <FormControl fullWidth size="small" error={hasError}>
            <InputLabel>{parameter.name}</InputLabel>
            <Select
              multiple
              value={value || []}
              onChange={(e) => handleParameterChange(parameter.id, e.target.value)}
              label={parameter.name}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(selected as string[]).map((val) => {
                    const option = parameter.options?.find(opt => opt.value === val);
                    return (
                      <Chip key={val} label={option?.label || val} size="small" />
                    );
                  })}
                </Box>
              )}
            >
              {parameter.options?.map(option => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </Select>
            {(error || parameter.description) && (
              <Typography variant="caption" color={hasError ? 'error' : 'text.secondary'} sx={{ mt: 0.5 }}>
                {error || parameter.description}
              </Typography>
            )}
          </FormControl>
        );

      case 'range':
        const rangeValue = value || { min: parameter.validation?.min || 0, max: parameter.validation?.max || 100 };
        return (
          <Box>
            <Typography variant="caption" color="text.secondary">
              {parameter.description}
            </Typography>
            <Box sx={{ px: 1, mt: 1 }}>
              <Slider
                value={[rangeValue.min, rangeValue.max]}
                onChange={(_: any, newValue: any) => {
                  const [min, max] = newValue as number[];
                  handleParameterChange(parameter.id, { min, max });
                }}
                valueLabelDisplay="auto"
                min={parameter.validation?.min || 0}
                max={parameter.validation?.max || 100}
                step={0.1}
              />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
              <TextField
                size="small"
                label="Min"
                type="number"
                value={rangeValue.min}
                onChange={(e) => handleParameterChange(parameter.id, {
                  ...rangeValue,
                  min: parseFloat(e.target.value) || 0
                })}
                sx={{ width: '48%' }}
              />
              <TextField
                size="small"
                label="Max"
                type="number"
                value={rangeValue.max}
                onChange={(e) => handleParameterChange(parameter.id, {
                  ...rangeValue,
                  max: parseFloat(e.target.value) || 0
                })}
                sx={{ width: '48%' }}
              />
            </Box>
          </Box>
        );

      case 'time-range':
        const timeValue = value || { start: '', end: '' };
        return (
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              {parameter.description}
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <TextField
                size="small"
                label="Start Time"
                type="datetime-local"
                value={timeValue.start}
                onChange={(e) => handleParameterChange(parameter.id, {
                  ...timeValue,
                  start: e.target.value
                })}
                InputLabelProps={{ shrink: true }}
              />
              <TextField
                size="small"
                label="End Time"
                type="datetime-local"
                value={timeValue.end}
                onChange={(e) => handleParameterChange(parameter.id, {
                  ...timeValue,
                  end: e.target.value
                })}
                InputLabelProps={{ shrink: true }}
              />
            </Box>
          </Box>
        );

      case 'spatial-extent':
        const spatialValue = value || { bounds: [] };
        return (
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              {parameter.description}
            </Typography>
            <TextField
              fullWidth
              size="small"
              label="Bounds (minLon, minLat, maxLon, maxLat)"
              value={spatialValue.bounds.join(', ')}
              onChange={(e) => {
                const bounds = e.target.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                handleParameterChange(parameter.id, { bounds });
              }}
              placeholder="e.g., -120.5, 35.2, -119.8, 35.9"
              error={hasError}
              helperText={error}
            />
          </Box>
        );

      default:
        return (
          <TextField
            fullWidth
            size="small"
            value={JSON.stringify(value)}
            onChange={(e) => {
              try {
                const parsedValue = JSON.parse(e.target.value);
                handleParameterChange(parameter.id, parsedValue);
              } catch {
                // Invalid JSON, keep as string
                handleParameterChange(parameter.id, e.target.value);
              }
            }}
            error={hasError}
            helperText={error || parameter.description}
            multiline
            rows={2}
          />
        );
    }
  };

  if (!nodeDefinition) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="warning">
          Node definition not found for type: {node.type}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">
            Parameters
          </Typography>
          <Tooltip title="Reset to defaults">
            <IconButton size="small" onClick={resetToDefaults}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          {node.name}
        </Typography>
        
        {Object.keys(validationErrors).some(key => validationErrors[key]) && (
          <Alert severity="error" sx={{ mt: 1 }}>
            Please fix validation errors before executing
          </Alert>
        )}
      </Box>

      {/* Parameter Groups */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {Object.entries(parameterGroups).map(([groupName, parameters]) => (
          <Accordion
            key={groupName}
            expanded={expandedGroups.has(groupName)}
            onChange={() => handleGroupToggle(groupName)}
            disableGutters
            elevation={0}
            sx={{
              '&:before': { display: 'none' },
              borderBottom: 1,
              borderColor: 'divider'
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMore />}
              sx={{ minHeight: 48 }}
            >
              <Typography variant="subtitle2" sx={{ textTransform: 'capitalize' }}>
                {groupName.replace('-', ' ')}
              </Typography>
              <Chip 
                label={parameters.length} 
                size="small" 
                sx={{ ml: 'auto', mr: 1 }}
              />
            </AccordionSummary>
            
            <AccordionDetails sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {parameters.map(parameter => (
                  <Box key={parameter.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Typography variant="body2" fontWeight="medium">
                        {parameter.name}
                        {parameter.required && (
                          <Typography component="span" color="error.main">
                            {' *'}
                          </Typography>
                        )}
                      </Typography>
                      
                      {validationErrors[parameter.id] ? (
                        <ErrorIcon color="error" fontSize="small" />
                      ) : node.parameters[parameter.id] !== undefined ? (
                        <Check color="success" fontSize="small" />
                      ) : null}
                      
                      {parameter.description && (
                        <Tooltip title={parameter.description}>
                          <Info fontSize="small" color="action" />
                        </Tooltip>
                      )}
                    </Box>
                    
                    {renderParameterInput(parameter)}
                  </Box>
                ))}
              </Box>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>

      {/* Footer */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="caption" color="text.secondary" align="center">
          {Object.keys(node.parameters).length} parameters configured
        </Typography>
      </Box>
    </Box>
  );
};