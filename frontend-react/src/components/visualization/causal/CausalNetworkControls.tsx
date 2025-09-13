import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  IconButton,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Tooltip,
  Chip
} from '@mui/material';
import {
  Close,
  Settings,
  ExpandMore,
  Refresh,
  PlayArrow,
  Pause,
  Stop,
  Speed,
  Visibility,
  VisibilityOff,
  Tune,
  Animation
} from '@mui/icons-material';
import {
  CausalLayoutConfig,
  LayoutAlgorithm,
  ClusteringCriteria,
  InteractionMode,
  RenderMode
} from './types';

interface CausalNetworkControlsProps {
  layoutConfig: CausalLayoutConfig;
  interactionMode: InteractionMode;
  renderMode: RenderMode;
  showLabels: boolean;
  showConfidenceIntervals: boolean;
  showTemporalLags: boolean;
  animationEnabled: boolean;
  isLayoutRunning: boolean;
  onLayoutConfigChange: (config: CausalLayoutConfig) => void;
  onInteractionModeChange: (mode: InteractionMode) => void;
  onRenderModeChange: (mode: RenderMode) => void;
  onShowLabelsChange: (show: boolean) => void;
  onShowConfidenceIntervalsChange: (show: boolean) => void;
  onShowTemporalLagsChange: (show: boolean) => void;
  onAnimationEnabledChange: (enabled: boolean) => void;
  onLayoutStart: () => void;
  onLayoutStop: () => void;
  onLayoutReset: () => void;
  onClose: () => void;
}

const LAYOUT_ALGORITHMS: LayoutAlgorithm[] = [
  'force-directed',
  'hierarchical',
  'circular',
  'grid',
  'dagre',
  'cola',
  'cose-bilkent',
  'breadthfirst',
  'concentric'
];

const CLUSTERING_CRITERIA: ClusteringCriteria[] = [
  'processType',
  'spatialScale',
  'temporalScale',
  'causalStrength',
  'geologicalContext',
  'dataSource'
];

const INTERACTION_MODES: InteractionMode[] = [
  'explore',
  'select',
  'edit',
  'annotate',
  'measure',
  'path-trace'
];

const RENDER_MODES: RenderMode[] = [
  'canvas',
  'webgl',
  'svg',
  'hybrid'
];

export const CausalNetworkControls: React.FC<CausalNetworkControlsProps> = ({
  layoutConfig,
  interactionMode,
  renderMode,
  showLabels,
  showConfidenceIntervals,
  showTemporalLags,
  animationEnabled,
  isLayoutRunning,
  onLayoutConfigChange,
  onInteractionModeChange,
  onRenderModeChange,
  onShowLabelsChange,
  onShowConfidenceIntervalsChange,
  onShowTemporalLagsChange,
  onAnimationEnabledChange,
  onLayoutStart,
  onLayoutStop,
  onLayoutReset,
  onClose
}) => {
  const [localConfig, setLocalConfig] = useState<CausalLayoutConfig>(layoutConfig);

  const handleConfigChange = (updates: Partial<CausalLayoutConfig>) => {
    const newConfig = { ...localConfig, ...updates };
    setLocalConfig(newConfig);
    onLayoutConfigChange(newConfig);
  };

  const handleForceStrengthChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ forceStrength: value as number });
  };

  const handleLinkDistanceChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ linkDistance: value as number });
  };

  const handleNodeRepulsionChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ nodeRepulsion: value as number });
  };

  const handleCenterForceChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ centerForce: value as number });
  };

  const handleCollisionRadiusChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ collisionRadius: value as number });
  };

  const handleAlphaDecayChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ alphaDecay: value as number });
  };

  const handleVelocityDecayChange = (event: Event, value: number | number[]) => {
    handleConfigChange({ velocityDecay: value as number });
  };

  const handleAlgorithmChange = (algorithm: LayoutAlgorithm) => {
    handleConfigChange({ algorithm });
  };

  const handleClusteringEnabledChange = (enabled: boolean) => {
    handleConfigChange({
      clustering: { ...localConfig.clustering, enabled }
    });
  };

  const handleClusteringCriteriaChange = (clusterBy: ClusteringCriteria) => {
    handleConfigChange({
      clustering: { ...localConfig.clustering, clusterBy }
    });
  };

  const handleClusterStrengthChange = (event: Event, value: number | number[]) => {
    handleConfigChange({
      clustering: { ...localConfig.clustering, clusterStrength: value as number }
    });
  };

  const handleHierarchicalEnabledChange = (enabled: boolean) => {
    handleConfigChange({
      hierarchical: { ...localConfig.hierarchical, enabled }
    });
  };

  const handleHierarchicalDirectionChange = (direction: 'TB' | 'BT' | 'LR' | 'RL') => {
    handleConfigChange({
      hierarchical: { ...localConfig.hierarchical, direction }
    });
  };

  const handleLevelSeparationChange = (event: Event, value: number | number[]) => {
    handleConfigChange({
      hierarchical: { ...localConfig.hierarchical, levelSeparation: value as number }
    });
  };

  const handleNodeSeparationChange = (event: Event, value: number | number[]) => {
    handleConfigChange({
      hierarchical: { ...localConfig.hierarchical, nodeSeparation: value as number }
    });
  };

  const resetToDefaults = () => {
    const defaultConfig: CausalLayoutConfig = {
      algorithm: 'force-directed',
      forceStrength: 0.5,
      linkDistance: 100,
      nodeRepulsion: 300,
      centerForce: 0.1,
      collisionRadius: 5,
      alphaDecay: 0.0228,
      velocityDecay: 0.4,
      clustering: {
        enabled: false,
        clusterBy: 'processType',
        clusterStrength: 0.3
      },
      hierarchical: {
        enabled: false,
        direction: 'TB',
        levelSeparation: 100,
        nodeSeparation: 50
      }
    };
    setLocalConfig(defaultConfig);
    onLayoutConfigChange(defaultConfig);
  };

  return (
    <Paper
      elevation={8}
      sx={{
        position: 'absolute',
        bottom: 16,
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
            <Settings color="primary" />
            <Typography variant="h6">Network Controls</Typography>
          </Box>
          <IconButton size="small" onClick={onClose}>
            <Close />
          </IconButton>
        </Box>
      </Box>

      {/* Layout Controls */}
      <Box sx={{ p: 2 }}>
        {/* Layout Algorithm */}
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Layout Algorithm</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <FormControl fullWidth size="small">
              <InputLabel>Algorithm</InputLabel>
              <Select
                value={localConfig.algorithm}
                onChange={(e) => handleAlgorithmChange(e.target.value as LayoutAlgorithm)}
                label="Algorithm"
              >
                {LAYOUT_ALGORITHMS.map((algorithm) => (
                  <MenuItem key={algorithm} value={algorithm}>
                    {algorithm.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Layout Control Buttons */}
            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
              <Tooltip title={isLayoutRunning ? "Pause Layout" : "Start Layout"}>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={isLayoutRunning ? <Pause /> : <PlayArrow />}
                  onClick={isLayoutRunning ? onLayoutStop : onLayoutStart}
                >
                  {isLayoutRunning ? 'Pause' : 'Start'}
                </Button>
              </Tooltip>
              
              <Tooltip title="Reset Layout">
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={onLayoutReset}
                >
                  Reset
                </Button>
              </Tooltip>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Force Parameters */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Force Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* Force Strength */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Force Strength: {localConfig.forceStrength.toFixed(2)}
                </Typography>
                <Slider
                  value={localConfig.forceStrength}
                  onChange={handleForceStrengthChange}
                  min={0.1}
                  max={2.0}
                  step={0.1}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Link Distance */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Link Distance: {localConfig.linkDistance}
                </Typography>
                <Slider
                  value={localConfig.linkDistance}
                  onChange={handleLinkDistanceChange}
                  min={20}
                  max={300}
                  step={10}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Node Repulsion */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Node Repulsion: {localConfig.nodeRepulsion}
                </Typography>
                <Slider
                  value={localConfig.nodeRepulsion}
                  onChange={handleNodeRepulsionChange}
                  min={50}
                  max={1000}
                  step={25}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Center Force */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Center Force: {localConfig.centerForce.toFixed(2)}
                </Typography>
                <Slider
                  value={localConfig.centerForce}
                  onChange={handleCenterForceChange}
                  min={0}
                  max={1}
                  step={0.05}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Collision Radius */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Collision Radius: {localConfig.collisionRadius}
                </Typography>
                <Slider
                  value={localConfig.collisionRadius}
                  onChange={handleCollisionRadiusChange}
                  min={0}
                  max={20}
                  step={1}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Simulation Parameters */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Simulation Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* Alpha Decay */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Alpha Decay: {localConfig.alphaDecay.toFixed(4)}
                </Typography>
                <Slider
                  value={localConfig.alphaDecay}
                  onChange={handleAlphaDecayChange}
                  min={0.001}
                  max={0.1}
                  step={0.001}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Velocity Decay */}
              <Box>
                <Typography variant="caption" gutterBottom>
                  Velocity Decay: {localConfig.velocityDecay.toFixed(2)}
                </Typography>
                <Slider
                  value={localConfig.velocityDecay}
                  onChange={handleVelocityDecayChange}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Clustering Options */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Clustering</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localConfig.clustering.enabled}
                    onChange={(e) => handleClusteringEnabledChange(e.target.checked)}
                  />
                }
                label="Enable Clustering"
              />

              {localConfig.clustering.enabled && (
                <>
                  <FormControl fullWidth size="small">
                    <InputLabel>Cluster By</InputLabel>
                    <Select
                      value={localConfig.clustering.clusterBy}
                      onChange={(e) => handleClusteringCriteriaChange(e.target.value as ClusteringCriteria)}
                      label="Cluster By"
                    >
                      {CLUSTERING_CRITERIA.map((criteria) => (
                        <MenuItem key={criteria} value={criteria}>
                          {criteria.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography variant="caption" gutterBottom>
                      Cluster Strength: {localConfig.clustering.clusterStrength.toFixed(2)}
                    </Typography>
                    <Slider
                      value={localConfig.clustering.clusterStrength}
                      onChange={handleClusterStrengthChange}
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Hierarchical Layout */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Hierarchical Layout</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={localConfig.hierarchical.enabled}
                    onChange={(e) => handleHierarchicalEnabledChange(e.target.checked)}
                  />
                }
                label="Enable Hierarchical Layout"
              />

              {localConfig.hierarchical.enabled && (
                <>
                  <FormControl fullWidth size="small">
                    <InputLabel>Direction</InputLabel>
                    <Select
                      value={localConfig.hierarchical.direction}
                      onChange={(e) => handleHierarchicalDirectionChange(e.target.value as 'TB' | 'BT' | 'LR' | 'RL')}
                      label="Direction"
                    >
                      <MenuItem value="TB">Top to Bottom</MenuItem>
                      <MenuItem value="BT">Bottom to Top</MenuItem>
                      <MenuItem value="LR">Left to Right</MenuItem>
                      <MenuItem value="RL">Right to Left</MenuItem>
                    </Select>
                  </FormControl>

                  <Box>
                    <Typography variant="caption" gutterBottom>
                      Level Separation: {localConfig.hierarchical.levelSeparation}
                    </Typography>
                    <Slider
                      value={localConfig.hierarchical.levelSeparation}
                      onChange={handleLevelSeparationChange}
                      min={50}
                      max={300}
                      step={10}
                      valueLabelDisplay="auto"
                    />
                  </Box>

                  <Box>
                    <Typography variant="caption" gutterBottom>
                      Node Separation: {localConfig.hierarchical.nodeSeparation}
                    </Typography>
                    <Slider
                      value={localConfig.hierarchical.nodeSeparation}
                      onChange={handleNodeSeparationChange}
                      min={20}
                      max={150}
                      step={5}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </>
              )}
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Interaction & Rendering */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="subtitle2">Interaction & Rendering</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* Interaction Mode */}
              <FormControl fullWidth size="small">
                <InputLabel>Interaction Mode</InputLabel>
                <Select
                  value={interactionMode}
                  onChange={(e) => onInteractionModeChange(e.target.value as InteractionMode)}
                  label="Interaction Mode"
                >
                  {INTERACTION_MODES.map((mode) => (
                    <MenuItem key={mode} value={mode}>
                      {mode.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Render Mode */}
              <FormControl fullWidth size="small">
                <InputLabel>Render Mode</InputLabel>
                <Select
                  value={renderMode}
                  onChange={(e) => onRenderModeChange(e.target.value as RenderMode)}
                  label="Render Mode"
                >
                  {RENDER_MODES.map((mode) => (
                    <MenuItem key={mode} value={mode}>
                      {mode.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Display Options */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showLabels}
                      onChange={(e) => onShowLabelsChange(e.target.checked)}
                    />
                  }
                  label="Show Labels"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={showConfidenceIntervals}
                      onChange={(e) => onShowConfidenceIntervalsChange(e.target.checked)}
                    />
                  }
                  label="Show Confidence Intervals"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={showTemporalLags}
                      onChange={(e) => onShowTemporalLagsChange(e.target.checked)}
                    />
                  }
                  label="Show Temporal Lags"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={animationEnabled}
                      onChange={(e) => onAnimationEnabledChange(e.target.checked)}
                    />
                  }
                  label="Enable Animations"
                />
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Reset Button */}
        <Divider sx={{ my: 2 }} />
        <Button
          fullWidth
          variant="outlined"
          startIcon={<Refresh />}
          onClick={resetToDefaults}
        >
          Reset to Defaults
        </Button>
      </Box>
    </Paper>
  );
};