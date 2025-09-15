import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  // ListItemSecondaryAction - unused
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Grid,
  Alert,
  Tooltip,
  Divider,
  // Timeline components - not available in @mui/material
  // Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot, TimelineOppositeContent
} from '@mui/material';
import {
  History as HistoryIcon,
  Restore as RestoreIcon,
  Compare as CompareIcon,
  Save as SaveIcon,
  // Branch as BranchIcon, // Not available in MUI
  AccountTree as BranchIcon, // Alternative for branch
  Merge as MergeIcon,
  Delete as _DeleteIcon,
  Visibility as _VisibilityIcon,
  Edit as _EditIcon,
  Label as _LabelIcon,
  Schedule as _ScheduleIcon,
  Person as PersonIcon,
  Difference as _DifferenceIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  HypothesisVersion,
  VersionBranch,
  VersionDiff,
  MergeConflict,
  // VersionMetadata - unused
} from '../../types/causal-analysis';

interface HypothesisVersionControlProps {
  currentHypothesis: CausalHypothesis | null;
  onHypothesisRestore: (hypothesis: CausalHypothesis) => void;
}

export const HypothesisVersionControl: React.FC<HypothesisVersionControlProps> = ({
  currentHypothesis,
  onHypothesisRestore
}) => {
  const [versions, setVersions] = useState<HypothesisVersion[]>([]);
  const [branches, setBranches] = useState<VersionBranch[]>([]);
  const [currentBranch, setCurrentBranch] = useState<string>('main');
  const [selectedVersions, setSelectedVersions] = useState<Set<string>>(new Set());
  const [showCompareDialog, setShowCompareDialog] = useState(false);
  const [showBranchDialog, setShowBranchDialog] = useState(false);
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [newBranchName, setNewBranchName] = useState('');
  const [commitMessage, setCommitMessage] = useState('');
  const [versionDiff, setVersionDiff] = useState<VersionDiff | null>(null);
  const [mergeConflicts, setMergeConflicts] = useState<MergeConflict[]>([]);

  // Initialize with sample version history
  useEffect(() => {
    if (currentHypothesis && versions.length === 0) {
      initializeVersionHistory(currentHypothesis);
    }
  }, [currentHypothesis, versions.length]);

  const initializeVersionHistory = useCallback((hypothesis: CausalHypothesis) => {
    const initialVersions: HypothesisVersion[] = [
      {
        id: 'v1.0.0',
        hypothesis: { ...hypothesis, name: 'Initial Hypothesis' },
        timestamp: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
        author: 'researcher@example.com',
        message: 'Initial hypothesis creation',
        branch: 'main',
        parentVersion: null,
        tags: ['initial', 'baseline'],
        metadata: {
          changeType: 'creation',
          changesCount: 0,
          testResults: null,
          reviewStatus: 'draft'
        }
      },
      {
        id: 'v1.1.0',
        hypothesis: { 
          ...hypothesis, 
          name: 'Refined Hypothesis',
          edges: hypothesis.edges.map(e => ({ ...e, strength: e.strength * 1.1 }))
        },
        timestamp: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
        author: 'researcher@example.com',
        message: 'Refined causal strengths based on initial testing',
        branch: 'main',
        parentVersion: 'v1.0.0',
        tags: ['refinement'],
        metadata: {
          changeType: 'modification',
          changesCount: hypothesis.edges.length,
          testResults: { 
            id: 'test-1', 
            hypothesisId: hypothesis.id, 
            timestamp: new Date(), 
            pValue: 0.03, 
            effectSize: 0.45, 
            confidenceInterval: { lower: 0.2, upper: 0.7 }, 
            testStatistic: 2.1, 
            degreesOfFreedom: 10, 
            metadata: { testType: 'pearson-correlation', datasetSize: 100, dataQuality: 'high', assumptions: [], warnings: [], computationTime: 1000 } 
          },
          reviewStatus: 'approved'
        }
      },
      {
        id: 'v1.2.0',
        hypothesis: {
          ...hypothesis,
          name: 'Extended Hypothesis',
          variables: [...hypothesis.variables, {
            id: 'temp_var',
            name: 'Temperature Gradient',
            type: 'continuous',
            unit: '°C/km',
            description: 'Thermal gradient in volcanic system',
            geologicalContext: 'volcanic',
            constraints: [],
            metadata: { created: new Date(), source: 'manual', quality: 'high' }
          }]
        },
        timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
        author: 'collaborator@example.com',
        message: 'Added temperature gradient variable',
        branch: 'feature/thermal-analysis',
        parentVersion: 'v1.1.0',
        tags: ['feature', 'thermal'],
        metadata: {
          changeType: 'addition',
          changesCount: 1,
          testResults: { 
            id: 'test-2', 
            hypothesisId: hypothesis.id, 
            timestamp: new Date(), 
            pValue: 0.02, 
            effectSize: 0.52, 
            confidenceInterval: { lower: 0.3, upper: 0.8 }, 
            testStatistic: 2.5, 
            degreesOfFreedom: 12, 
            metadata: { testType: 'pearson-correlation', datasetSize: 120, dataQuality: 'high', assumptions: [], warnings: [], computationTime: 1200 } 
          },
          reviewStatus: 'pending'
        }
      }
    ];

    const initialBranches: VersionBranch[] = [
      {
        id: 'main',
        name: 'main',
        description: 'Main development branch',
        created: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        lastCommit: 'v1.1.0',
        author: 'researcher@example.com',
        isProtected: true
      },
      {
        id: 'feature/thermal-analysis',
        name: 'feature/thermal-analysis',
        description: 'Experimental thermal analysis features',
        created: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
        lastCommit: 'v1.2.0',
        author: 'collaborator@example.com',
        isProtected: false
      }
    ];

    setVersions(initialVersions);
    setBranches(initialBranches);
  }, []);

  const handleSaveVersion = useCallback(async () => {
    if (!currentHypothesis || !commitMessage.trim()) return;

    const newVersion: HypothesisVersion = {
      id: `v${versions.length + 1}.0.0`,
      hypothesis: { ...currentHypothesis },
      timestamp: new Date(),
      author: 'current-user@example.com',
      message: commitMessage,
      branch: currentBranch,
      parentVersion: versions.filter(v => v.branch === currentBranch).slice(-1)[0]?.id || null,
      tags: [],
      metadata: {
        changeType: 'modification',
        changesCount: calculateChanges(currentHypothesis, versions.slice(-1)[0]?.hypothesis),
        testResults: null,
        reviewStatus: 'draft'
      }
    };

    setVersions(prev => [...prev, newVersion]);
    setCommitMessage('');
  }, [currentHypothesis, commitMessage, currentBranch, versions]);

  const handleRestoreVersion = useCallback((version: HypothesisVersion) => {
    onHypothesisRestore(version.hypothesis);
  }, [onHypothesisRestore]);

  const handleCreateBranch = useCallback(() => {
    if (!newBranchName.trim()) return;

    const newBranch: VersionBranch = {
      id: newBranchName,
      name: newBranchName,
      description: `Branch created from ${currentBranch}`,
      created: new Date(),
      lastCommit: versions.filter(v => v.branch === currentBranch).slice(-1)[0]?.id || null,
      author: 'current-user@example.com',
      isProtected: false
    };

    setBranches(prev => [...prev, newBranch]);
    setCurrentBranch(newBranchName);
    setNewBranchName('');
    setShowBranchDialog(false);
  }, [newBranchName, currentBranch, versions]);

  const handleCompareVersions = useCallback(async () => {
    if (selectedVersions.size !== 2) return;

    const [version1Id, version2Id] = Array.from(selectedVersions);
    const version1 = versions.find(v => v.id === version1Id);
    const version2 = versions.find(v => v.id === version2Id);

    if (!version1 || !version2) return;

    const diff = calculateVersionDiff(version1, version2);
    setVersionDiff(diff);
    setShowCompareDialog(true);
  }, [selectedVersions, versions]);

  const handleMergeBranch = useCallback(async (sourceBranch: string, targetBranch: string) => {
    const sourceVersions = versions.filter(v => v.branch === sourceBranch);
    const targetVersions = versions.filter(v => v.branch === targetBranch);
    
    if (sourceVersions.length === 0 || targetVersions.length === 0) return;

    const latestSource = sourceVersions.slice(-1)[0];
    const latestTarget = targetVersions.slice(-1)[0];

    // Detect merge conflicts
    const conflicts = detectMergeConflicts(latestSource.hypothesis, latestTarget.hypothesis);
    
    if (conflicts.length > 0) {
      setMergeConflicts(conflicts);
      setShowMergeDialog(true);
    } else {
      // Auto-merge if no conflicts
      const mergedHypothesis = mergeHypotheses(latestSource.hypothesis, latestTarget.hypothesis);
      const mergeVersion: HypothesisVersion = {
        id: `merge-${Date.now()}`,
        hypothesis: mergedHypothesis,
        timestamp: new Date(),
        author: 'current-user@example.com',
        message: `Merge ${sourceBranch} into ${targetBranch}`,
        branch: targetBranch,
        parentVersion: latestTarget.id,
        tags: ['merge'],
        metadata: {
          changeType: 'merge',
          changesCount: conflicts.length,
          testResults: null,
          reviewStatus: 'draft'
        }
      };

      setVersions(prev => [...prev, mergeVersion]);
    }
  }, [versions]);

  const handleVersionSelect = useCallback((versionId: string, selected: boolean) => {
    setSelectedVersions(prev => {
      const newSet = new Set(prev);
      if (selected) {
        if (newSet.size < 2) {
          newSet.add(versionId);
        }
      } else {
        newSet.delete(versionId);
      }
      return newSet;
    });
  }, []);

  const branchVersions = useMemo(() => {
    return versions.filter(v => v.branch === currentBranch).sort((a, b) => 
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [versions, currentBranch]);

  const canCompare = selectedVersions.size === 2;
  const canSave = currentHypothesis && commitMessage.trim().length > 0;

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HistoryIcon />
          Version Control
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<BranchIcon />}
            onClick={() => setShowBranchDialog(true)}
            size="small"
          >
            New Branch
          </Button>
          <Button
            variant="outlined"
            startIcon={<CompareIcon />}
            onClick={handleCompareVersions}
            disabled={!canCompare}
            size="small"
          >
            Compare ({selectedVersions.size}/2)
          </Button>
        </Box>
      </Box>

      {!currentHypothesis && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Select a hypothesis to view its version history and create new versions.
        </Alert>
      )}

      {/* Branch Selection */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <FormControl fullWidth size="small">
                <InputLabel>Current Branch</InputLabel>
                <Select
                  value={currentBranch}
                  onChange={(e) => setCurrentBranch(e.target.value)}
                >
                  {branches.map(branch => (
                    <MenuItem key={branch.id} value={branch.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <BranchIcon fontSize="small" />
                        {branch.name}
                        {branch.isProtected && <Chip label="Protected" size="small" color="warning" />}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="body2" color="text.secondary">
                {branchVersions.length} version(s) in this branch
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Save New Version */}
      {currentHypothesis && (
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle2" gutterBottom>
              Save New Version
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  size="small"
                  placeholder="Commit message (describe your changes)"
                  value={commitMessage}
                  onChange={(e) => setCommitMessage(e.target.value)}
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSaveVersion}
                  disabled={!canSave}
                >
                  Commit
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Version Timeline */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Version History - {currentBranch}
        </Typography>
        
        {branchVersions.length === 0 ? (
          <Alert severity="info">
            No versions found in this branch. Create your first version by committing changes above.
          </Alert>
        ) : (
          <List>
            {branchVersions.map((version) => (
              <ListItem key={version.id} sx={{ flexDirection: 'column', alignItems: 'stretch' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {version.timestamp.toLocaleDateString()} - {version.timestamp.toLocaleTimeString()}
                </Typography>
                <VersionCard
                  version={version}
                  isSelected={selectedVersions.has(version.id)}
                  onSelect={(selected) => handleVersionSelect(version.id, selected)}
                  onRestore={() => handleRestoreVersion(version)}
                  canSelect={selectedVersions.size < 2 || selectedVersions.has(version.id)}
                />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* Branch Management */}
      {branches.length > 1 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Branch Management
          </Typography>
          <Grid container spacing={2}>
            {branches.filter(b => b.id !== currentBranch).map(branch => (
              <Grid item xs={12} md={6} key={branch.id}>
                <Card variant="outlined">
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="subtitle2">
                        {branch.name}
                      </Typography>
                      {branch.isProtected && <Chip label="Protected" size="small" color="warning" />}
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      {branch.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Last commit: {branch.lastCommit || 'None'}
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      <Button
                        size="small"
                        startIcon={<MergeIcon />}
                        onClick={() => handleMergeBranch(branch.id, currentBranch)}
                        disabled={branch.isProtected}
                      >
                        Merge into {currentBranch}
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* Create Branch Dialog */}
      <Dialog open={showBranchDialog} onClose={() => setShowBranchDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Branch</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Branch Name"
            value={newBranchName}
            onChange={(e) => setNewBranchName(e.target.value)}
            placeholder="feature/new-analysis"
            sx={{ mt: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Branch will be created from current branch: {currentBranch}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowBranchDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateBranch} variant="contained" disabled={!newBranchName.trim()}>
            Create Branch
          </Button>
        </DialogActions>
      </Dialog>

      {/* Compare Versions Dialog */}
      <Dialog open={showCompareDialog} onClose={() => setShowCompareDialog(false)} maxWidth="lg" fullWidth>
        <DialogTitle>Version Comparison</DialogTitle>
        <DialogContent>
          {versionDiff && <VersionDiffView diff={versionDiff} />}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCompareDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Merge Conflicts Dialog */}
      <Dialog open={showMergeDialog} onClose={() => setShowMergeDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Merge Conflicts</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            The following conflicts need to be resolved before merging:
          </Alert>
          {mergeConflicts.map((conflict, index) => (
            <MergeConflictResolver
              key={index}
              conflict={conflict}
              onResolve={(resolution) => {
                // Handle conflict resolution
                console.log('Conflict resolved:', resolution);
              }}
            />
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowMergeDialog(false)}>Cancel</Button>
          <Button variant="contained" disabled={mergeConflicts.some(c => !c.resolved)}>
            Complete Merge
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Version Card Component
const VersionCard: React.FC<{
  version: HypothesisVersion;
  isSelected: boolean;
  onSelect: (selected: boolean) => void;
  onRestore: () => void;
  canSelect: boolean;
}> = ({ version, isSelected, onSelect, onRestore, canSelect }) => {
  return (
    <Card variant="outlined" sx={{ backgroundColor: isSelected ? 'action.selected' : 'background.paper' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle2">
            {version.id} - {version.hypothesis.name}
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <input
              type="checkbox"
              checked={isSelected}
              onChange={(e) => onSelect(e.target.checked)}
              disabled={!canSelect}
            />
            <Tooltip title="Restore this version">
              <IconButton size="small" onClick={onRestore}>
                <RestoreIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
        
        <Typography variant="body2" sx={{ mb: 1 }}>
          {version.message}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
          <Chip label={version.author} size="small" icon={<PersonIcon />} />
          <Chip label={version.metadata.changeType} size="small" />
          {version.metadata.testResults && (
            <Chip 
              label={`p=${version.metadata.testResults.pValue.toFixed(3)}`} 
              size="small" 
              color="primary" 
            />
          )}
          {version.tags.map(tag => (
            <Chip key={tag} label={tag} size="small" variant="outlined" />
          ))}
        </Box>
        
        <Typography variant="caption" color="text.secondary">
          {version.metadata.changesCount} change(s) • {version.metadata.reviewStatus}
        </Typography>
      </CardContent>
    </Card>
  );
};

// Version Diff View Component
const VersionDiffView: React.FC<{ diff: VersionDiff }> = ({ diff }) => {
  return (
    <Box sx={{ pt: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <Typography variant="subtitle2" gutterBottom>
            Version {diff.version1.id}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {diff.version1.hypothesis.name}
          </Typography>
        </Grid>
        <Grid item xs={6}>
          <Typography variant="subtitle2" gutterBottom>
            Version {diff.version2.id}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {diff.version2.hypothesis.name}
          </Typography>
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle2" gutterBottom>
        Changes Summary
      </Typography>
      
      <List>
        {diff.changes.map((change, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={change.description}
              secondary={`${change.type}: ${change.field}`}
            />
            <Chip
              label={change.type}
              size="small"
              color={change.type === 'addition' ? 'success' : 
                     change.type === 'removal' ? 'error' : 'warning'}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

// Merge Conflict Resolver Component
const MergeConflictResolver: React.FC<{
  conflict: MergeConflict;
  onResolve: (resolution: any) => void;
}> = ({ conflict, onResolve }) => {
  const [selectedResolution, setSelectedResolution] = useState<string>('');

  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          Conflict in {conflict.field}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {conflict.description}
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" fontWeight="medium">Current Version:</Typography>
            <Typography variant="body2">{JSON.stringify(conflict.currentValue)}</Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2" fontWeight="medium">Incoming Version:</Typography>
            <Typography variant="body2">{JSON.stringify(conflict.incomingValue)}</Typography>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          <Button
            variant={selectedResolution === 'current' ? 'contained' : 'outlined'}
            onClick={() => {
              setSelectedResolution('current');
              onResolve({ field: conflict.field, resolution: 'current' });
            }}
          >
            Keep Current
          </Button>
          <Button
            variant={selectedResolution === 'incoming' ? 'contained' : 'outlined'}
            onClick={() => {
              setSelectedResolution('incoming');
              onResolve({ field: conflict.field, resolution: 'incoming' });
            }}
          >
            Use Incoming
          </Button>
          <Button
            variant={selectedResolution === 'manual' ? 'contained' : 'outlined'}
            onClick={() => {
              setSelectedResolution('manual');
              onResolve({ field: conflict.field, resolution: 'manual' });
            }}
          >
            Manual Merge
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

// Utility functions
// function _getVersionColor(changeType: string): 'primary' | 'secondary' | 'success' | 'warning' | 'error' {
//   switch (changeType) {
//     case 'creation': return 'success';
//     case 'modification': return 'primary';
//     case 'addition': return 'secondary';
//     case 'removal': return 'warning';
//     case 'merge': return 'error';
//     default: return 'primary';
//   }
// }

// function _getVersionIcon(changeType: string) {
//   switch (changeType) {
//     case 'creation': return <SaveIcon />;
//     case 'modification': return <EditIcon />;
//     case 'addition': return <BranchIcon />;
//     case 'removal': return <DeleteIcon />;
//     case 'merge': return <MergeIcon />;
//     default: return <HistoryIcon />;
//   }
// }

function calculateChanges(current: CausalHypothesis, previous?: CausalHypothesis): number {
  if (!previous) return 0;
  
  let changes = 0;
  
  // Count variable changes
  if (current.variables.length !== previous.variables.length) {
    changes += Math.abs(current.variables.length - previous.variables.length);
  }
  
  // Count edge changes
  if (current.edges.length !== previous.edges.length) {
    changes += Math.abs(current.edges.length - previous.edges.length);
  }
  
  // Count property changes
  if (current.name !== previous.name) changes++;
  if (current.description !== previous.description) changes++;
  
  return changes;
}

function calculateVersionDiff(version1: HypothesisVersion, version2: HypothesisVersion): VersionDiff {
  const changes = [];
  
  // Compare variables
  const v1VarCount = version1.hypothesis.variables.length;
  const v2VarCount = version2.hypothesis.variables.length;
  
  if (v1VarCount !== v2VarCount) {
    changes.push({
      type: v2VarCount > v1VarCount ? ('addition' as const) : ('removal' as const),
      field: 'variables',
      description: `Variables changed from ${v1VarCount} to ${v2VarCount}`,
      oldValue: v1VarCount,
      newValue: v2VarCount
    });
  }
  
  // Compare edges
  const v1EdgeCount = version1.hypothesis.edges.length;
  const v2EdgeCount = version2.hypothesis.edges.length;
  
  if (v1EdgeCount !== v2EdgeCount) {
    changes.push({
      type: v2EdgeCount > v1EdgeCount ? ('addition' as const) : ('removal' as const),
      field: 'edges',
      description: `Edges changed from ${v1EdgeCount} to ${v2EdgeCount}`,
      oldValue: v1EdgeCount,
      newValue: v2EdgeCount
    });
  }
  
  // Compare names
  if (version1.hypothesis.name !== version2.hypothesis.name) {
    changes.push({
      type: 'modification' as const,
      field: 'name',
      description: `Name changed`,
      oldValue: version1.hypothesis.name,
      newValue: version2.hypothesis.name
    });
  }
  
  return {
    version1,
    version2,
    changes,
    summary: `${changes.length} change(s) detected`
  };
}

function detectMergeConflicts(hypothesis1: CausalHypothesis, hypothesis2: CausalHypothesis): MergeConflict[] {
  const conflicts: MergeConflict[] = [];
  
  // Check for conflicting variable definitions
  const commonVariables = hypothesis1.variables.filter(v1 => 
    hypothesis2.variables.some(v2 => v2.id === v1.id)
  );
  
  commonVariables.forEach(v1 => {
    const v2 = hypothesis2.variables.find(v => v.id === v1.id);
    if (v2 && (v1.name !== v2.name || v1.type !== v2.type)) {
      conflicts.push({
        id: `conflict_${v1.id}`,
        field: `variable.${v1.id}`,
        description: `Variable ${v1.id} has conflicting definitions`,
        currentValue: v1,
        incomingValue: v2,
        type: 'variable_conflict',
        resolved: false
      });
    }
  });
  
  return conflicts;
}

function mergeHypotheses(hypothesis1: CausalHypothesis, hypothesis2: CausalHypothesis): CausalHypothesis {
  // Simple merge strategy - combine variables and edges
  const mergedVariables = [...hypothesis1.variables];
  const mergedEdges = [...hypothesis1.edges];
  
  // Add variables from hypothesis2 that don't exist in hypothesis1
  hypothesis2.variables.forEach(v2 => {
    if (!mergedVariables.some(v1 => v1.id === v2.id)) {
      mergedVariables.push(v2);
    }
  });
  
  // Add edges from hypothesis2 that don't exist in hypothesis1
  hypothesis2.edges.forEach(e2 => {
    if (!mergedEdges.some(e1 => e1.id === e2.id)) {
      mergedEdges.push(e2);
    }
  });
  
  return {
    ...hypothesis1,
    name: `Merged: ${hypothesis1.name} + ${hypothesis2.name}`,
    variables: mergedVariables,
    edges: mergedEdges,
    metadata: {
      ...hypothesis1.metadata,
      lastModified: new Date()
    }
  };
}