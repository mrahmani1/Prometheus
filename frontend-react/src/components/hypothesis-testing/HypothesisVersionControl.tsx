import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Divider
} from '@mui/material';
import {
  History as HistoryIcon,
  Restore as RestoreIcon,
  Visibility as ViewIcon,
  Compare as CompareIcon,
  Save as SaveIcon,
  AccountTree as BranchIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  HypothesisVersion,
  VersionBranch,
  ChangeType
} from '../../types/causal-analysis';

interface HypothesisVersionControlProps {
  currentHypothesis: CausalHypothesis | null;
  onHypothesisRestore: (hypothesis: CausalHypothesis) => void;
}

export const HypothesisVersionControl: React.FC<HypothesisVersionControlProps> = ({
  currentHypothesis,
  onHypothesisRestore
}) => {
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showDiffDialog, setShowDiffDialog] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [currentBranch] = useState('main');

  // Mock version history
  const mockVersions: HypothesisVersion[] = [
    {
      id: 'v1',
      hypothesis: currentHypothesis!,
      timestamp: new Date(Date.now() - 86400000), // 1 day ago
      author: 'Dr. Sarah Johnson',
      message: 'Initial hypothesis with basic seismic-volcanic relationships',
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
      id: 'v2',
      hypothesis: currentHypothesis!,
      timestamp: new Date(Date.now() - 43200000), // 12 hours ago
      author: 'Prof. Michael Chen',
      message: 'Added deformation variables and temporal constraints',
      branch: 'main',
      parentVersion: 'v1',
      tags: ['enhancement'],
      metadata: {
        changeType: 'modification',
        changesCount: 3,
        testResults: null,
        reviewStatus: 'pending'
      }
    },
    {
      id: 'v3',
      hypothesis: currentHypothesis!,
      timestamp: new Date(Date.now() - 21600000), // 6 hours ago
      author: 'Dr. Sarah Johnson',
      message: 'Refined causal strengths based on statistical analysis',
      branch: 'main',
      parentVersion: 'v2',
      tags: ['refinement', 'statistical-update'],
      metadata: {
        changeType: 'modification',
        changesCount: 5,
        testResults: null,
        reviewStatus: 'approved'
      }
    }
  ];

  const mockBranches: VersionBranch[] = [
    {
      id: 'main',
      name: 'main',
      description: 'Main development branch',
      created: new Date(Date.now() - 86400000),
      lastCommit: 'v3',
      author: 'Dr. Sarah Johnson',
      isProtected: true
    },
    {
      id: 'experimental',
      name: 'experimental',
      description: 'Experimental features and alternative approaches',
      created: new Date(Date.now() - 43200000),
      lastCommit: null,
      author: 'Prof. Michael Chen',
      isProtected: false
    }
  ];

  const handleSaveVersion = () => {
    if (!currentHypothesis || !saveMessage.trim()) return;

    const newVersion: HypothesisVersion = {
      id: `v${Date.now()}`,
      hypothesis: currentHypothesis,
      timestamp: new Date(),
      author: 'Current User',
      message: saveMessage,
      branch: currentBranch,
      parentVersion: mockVersions[mockVersions.length - 1]?.id || null,
      tags: [],
      metadata: {
        changeType: 'modification',
        changesCount: 1,
        testResults: null,
        reviewStatus: 'draft'
      }
    };

    // In a real implementation, this would save to the backend
    console.log('Saving version:', newVersion);
    setSaveMessage('');
    setShowSaveDialog(false);
  };

  const handleRestoreVersion = (version: HypothesisVersion) => {
    onHypothesisRestore(version.hypothesis);
  };

  const getChangeTypeColor = (changeType: ChangeType) => {
    const colors = {
      creation: 'success',
      modification: 'primary',
      addition: 'info',
      removal: 'warning',
      merge: 'secondary'
    } as const;
    return colors[changeType] || 'default';
  };

  const getReviewStatusColor = (status: string) => {
    const colors: Record<string, 'success' | 'warning' | 'error' | 'default'> = {
      approved: 'success',
      pending: 'warning',
      rejected: 'error',
      draft: 'default'
    };
    return colors[status] || 'default';
  };

  if (!currentHypothesis) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Create a hypothesis to access version control features.
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <HistoryIcon />
        Version Control
      </Typography>

      {/* Current Branch and Actions */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <BranchIcon />
              <Typography variant="subtitle1">
                Branch: {currentBranch}
              </Typography>
              <Chip
                label={mockBranches.find(b => b.name === currentBranch)?.isProtected ? 'Protected' : 'Editable'}
                size="small"
                color={mockBranches.find(b => b.name === currentBranch)?.isProtected ? 'warning' : 'success'}
              />
            </Box>
            
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                startIcon={<SaveIcon />}
                onClick={() => setShowSaveDialog(true)}
                size="small"
                variant="outlined"
              >
                Save Version
              </Button>
              <Button
                startIcon={<CompareIcon />}
                onClick={() => setShowDiffDialog(true)}
                size="small"
                variant="outlined"
                disabled={selectedVersions.length !== 2}
              >
                Compare
              </Button>
            </Box>
          </Box>
          
          <Typography variant="body2" color="text.secondary">
            {mockBranches.find(b => b.name === currentBranch)?.description}
          </Typography>
        </CardContent>
      </Card>

      {/* Version History */}
      <Card>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Version History ({mockVersions.length})
          </Typography>
          
          <List>
            {mockVersions.reverse().map((version, index) => (
              <React.Fragment key={version.id}>
                <ListItem>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Typography variant="subtitle2">{version.id}</Typography>
                        <Chip
                          label={version.metadata.changeType}
                          size="small"
                          color={getChangeTypeColor(version.metadata.changeType)}
                        />
                        <Chip
                          label={version.metadata.reviewStatus}
                          size="small"
                          variant="outlined"
                          color={getReviewStatusColor(version.metadata.reviewStatus)}
                        />
                        {version.tags.map((tag) => (
                          <Chip
                            key={tag}
                            label={tag}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" sx={{ mb: 0.5 }}>
                          {version.message}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {version.author} • {version.timestamp.toLocaleString()} • 
                          {version.metadata.changesCount} changes
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton
                        size="small"
                        onClick={() => {
                          if (selectedVersions.includes(version.id)) {
                            setSelectedVersions(selectedVersions.filter(id => id !== version.id));
                          } else if (selectedVersions.length < 2) {
                            setSelectedVersions([...selectedVersions, version.id]);
                          }
                        }}
                        color={selectedVersions.includes(version.id) ? 'primary' : 'default'}
                        title="Select for comparison"
                      >
                        <CompareIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleRestoreVersion(version)}
                        title="Restore this version"
                      >
                        <RestoreIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        title="View details"
                      >
                        <ViewIcon />
                      </IconButton>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
                {index < mockVersions.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Save Version Dialog */}
      <Dialog open={showSaveDialog} onClose={() => setShowSaveDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Save New Version</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Commit Message"
            value={saveMessage}
            onChange={(e) => setSaveMessage(e.target.value)}
            sx={{ mt: 2 }}
            placeholder="Describe the changes made in this version..."
          />
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Current Branch: {currentBranch}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              This version will be saved to the {currentBranch} branch.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSaveDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveVersion} variant="contained" disabled={!saveMessage.trim()}>
            Save Version
          </Button>
        </DialogActions>
      </Dialog>

      {/* Compare Versions Dialog */}
      <Dialog open={showDiffDialog} onClose={() => setShowDiffDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Compare Versions</DialogTitle>
        <DialogContent>
          {selectedVersions.length === 2 ? (
            <Box>
              <Typography variant="body1" gutterBottom>
                Comparing {selectedVersions[0]} with {selectedVersions[1]}
              </Typography>
              <Alert severity="info">
                Version comparison feature would show detailed differences between the selected versions,
                including changes to variables, relationships, and constraints.
              </Alert>
            </Box>
          ) : (
            <Alert severity="warning">
              Please select exactly 2 versions to compare.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDiffDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};