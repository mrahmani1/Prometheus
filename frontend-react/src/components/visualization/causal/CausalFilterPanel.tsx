import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput,
  Checkbox,
  ListItemText,
  Button,
  Divider
} from '@mui/material';
import {
  Close,
  ExpandMore,
  FilterList,
  Refresh
} from '@mui/icons-material';
import {
  CausalFilterConfig,
  GeologicalProcessType,
  SpatialScale,
  TemporalScale,
  CausalEdgeType,
  ValidationStatus
} from './types';

interface CausalFilterPanelProps {
  filterConfig?: CausalFilterConfig;
  onFilterChange: (config: CausalFilterConfig) => void;
  onClose: () => void;
}

const GEOLOGICAL_PROCESS_TYPES: GeologicalProcessType[] = [
  'seismic', 'volcanic', 'deformation', 'geochemical', 'thermal',
  'electromagnetic', 'hydrological', 'atmospheric', 'tectonic',
  'magmatic', 'hydrothermal', 'metamorphic', 'sedimentary'
];

const SPATIAL_SCALES: SpatialScale[] = [
  'mineral', 'crystal', 'rock', 'outcrop', 'local',
  'regional', 'continental', 'global'
];

const TEMPORAL_SCALES: TemporalScale[] = [
  'millisecond', 'second', 'minute', 'hour', 'day',
  'month', 'year', 'decade', 'century', 'millennium', 'geological'
];

const EDGE_TYPES: CausalEdgeType[] = [
  'direct', 'indirect', 'bidirectional', 'conditional',
  'mediating', 'moderating', 'confounding'
];

const VALIDATION_STATUSES: ValidationStatus[] = [
  'validated', 'pending', 'rejected', 'uncertain', 'conflicting'
];

const DISCOVERY_METHODS = [
  'PC Algorithm', 'GES', 'LiNGAM', 'Causal Forests',
  'Granger Causality', 'Transfer Entropy', 'Cross-Correlation',
  'Expert Knowledge', 'Literature Review'
];

export const CausalFilterPanel: React.FC<CausalFilterPanelProps> = ({
  filterConfig,
  onFilterChange,
  onClose
}) => {
  const [localConfig, setLocalConfig] = useState<CausalFilterConfig>(
    filterConfig || {
      strengthRange: [0, 1],
      confidenceThreshold: 0.5,
      processTypes: [],
      spatialScales: [],
      temporalScales: [],
      edgeTypes: [],
      validationStatus: [],
      discoveryMethods: []
    }
  );

  const handleStrengthRangeChange = (_event: Event, newValue: number | number[]) => {
    const range = newValue as [number, number];
    setLocalConfig(prev => ({ ...prev, strengthRange: range }));
  };

  const handleConfidenceThresholdChange = (_event: Event, newValue: number | number[]) => {
    const threshold = newValue as number;
    setLocalConfig(prev => ({ ...prev, confidenceThreshold: threshold }));
  };

  const handleProcessTypesChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      processTypes: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleSpatialScalesChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      spatialScales: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleTemporalScalesChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      temporalScales: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleEdgeTypesChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      edgeTypes: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleValidationStatusChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      validationStatus: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleDiscoveryMethodsChange = (event: any) => {
    const value = event.target.value;
    setLocalConfig(prev => ({
      ...prev,
      discoveryMethods: typeof value === 'string' ? value.split(',') : value
    }));
  };

  const handleApplyFilters = () => {
    onFilterChange(localConfig);
  };

  const handleResetFilters = () => {
    const resetConfig: CausalFilterConfig = {
      strengthRange: [0, 1],
      confidenceThreshold: 0,
      processTypes: [],
      spatialScales: [],
      temporalScales: [],
      edgeTypes: [],
      validationStatus: [],
      discoveryMethods: []
    };
    setLocalConfig(resetConfig);
    onFilterChange(resetConfig);
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'absolute',
        top: 16,
        right: 16,
        width: 350,
        maxHeight: 'calc(100% - 32px)',
        overflow: 'auto',
        zIndex: 1000
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FilterList color="primary" />
            <Typography variant="h6">Causal Network Filters</Typography>
          </Box>
          <IconButton size="small" onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </Box>

      {/* Filter Controls */}
      <Box sx={{ p: 2 }}>
        {/* Strength Range Filter */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Causal Strength Range</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ px: 2 }}>
              <Slider
                value={localConfig.strengthRange}
                onChange={handleStrengthRangeChange}
                valueLabelDisplay="auto"
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' }
                ]}
              />
              <Typography variant="caption" color="textSecondary">
                Filter relationships by causal strength ({localConfig.strengthRange[0].toFixed(2)} - {localConfig.strengthRange[1].toFixed(2)})
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Confidence Threshold Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Confidence Threshold</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ px: 2 }}>
              <Slider
                value={localConfig.confidenceThreshold}
                onChange={handleConfidenceThresholdChange}
                valueLabelDisplay="auto"
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0%' },
                  { value: 0.5, label: '50%' },
                  { value: 0.95, label: '95%' }
                ]}
              />
              <Typography variant="caption" color="textSecondary">
                Minimum confidence level: {(localConfig.confidenceThreshold * 100).toFixed(0)}%
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Process Types Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Geological Process Types</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Process Types</InputLabel>
              <Select
                multiple
                value={localConfig.processTypes}
                onChange={handleProcessTypesChange}
                input={<OutlinedInput label="Process Types" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {GEOLOGICAL_PROCESS_TYPES.map((type) => (
                  <MenuItem key={type} value={type}>
                    <Checkbox checked={localConfig.processTypes.indexOf(type) > -1} />
                    <ListItemText primary={type} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>

        {/* Spatial Scales Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Spatial Scales</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Spatial Scales</InputLabel>
              <Select
                multiple
                value={localConfig.spatialScales}
                onChange={handleSpatialScalesChange}
                input={<OutlinedInput label="Spatial Scales" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {SPATIAL_SCALES.map((scale) => (
                  <MenuItem key={scale} value={scale}>
                    <Checkbox checked={localConfig.spatialScales.indexOf(scale) > -1} />
                    <ListItemText primary={scale} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>

        {/* Temporal Scales Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Temporal Scales</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Temporal Scales</InputLabel>
              <Select
                multiple
                value={localConfig.temporalScales}
                onChange={handleTemporalScalesChange}
                input={<OutlinedInput label="Temporal Scales" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {TEMPORAL_SCALES.map((scale) => (
                  <MenuItem key={scale} value={scale}>
                    <Checkbox checked={localConfig.temporalScales.indexOf(scale) > -1} />
                    <ListItemText primary={scale} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>

        {/* Edge Types Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Causal Edge Types</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Edge Types</InputLabel>
              <Select
                multiple
                value={localConfig.edgeTypes}
                onChange={handleEdgeTypesChange}
                input={<OutlinedInput label="Edge Types" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {EDGE_TYPES.map((type) => (
                  <MenuItem key={type} value={type}>
                    <Checkbox checked={localConfig.edgeTypes.indexOf(type) > -1} />
                    <ListItemText primary={type} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>

        {/* Validation Status Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Validation Status</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Validation Status</InputLabel>
              <Select
                multiple
                value={localConfig.validationStatus}
                onChange={handleValidationStatusChange}
                input={<OutlinedInput label="Validation Status" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {VALIDATION_STATUSES.map((status) => (
                  <MenuItem key={status} value={status}>
                    <Checkbox checked={localConfig.validationStatus.indexOf(status) > -1} />
                    <ListItemText primary={status} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>

        {/* Discovery Methods Filter */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Discovery Methods</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth>
              <InputLabel>Discovery Methods</InputLabel>
              <Select
                multiple
                value={localConfig.discoveryMethods}
                onChange={handleDiscoveryMethodsChange}
                input={<OutlinedInput label="Discovery Methods" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {DISCOVERY_METHODS.map((method) => (
                  <MenuItem key={method} value={method}>
                    <Checkbox checked={localConfig.discoveryMethods.indexOf(method) > -1} />
                    <ListItemText primary={method} />
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </AccordionDetails>
        </Accordion>
      </Box>

      {/* Action Buttons */}
      <Divider />
      <Box sx={{ p: 2, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={handleResetFilters}
        >
          Reset
        </Button>
        <Button
          variant="contained"
          onClick={handleApplyFilters}
        >
          Apply Filters
        </Button>
      </Box>
    </Paper>
  );
};