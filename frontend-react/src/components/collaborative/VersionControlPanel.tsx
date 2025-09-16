/**
 * Version Control Panel Component
 * 
 * Git-like version control interface for collaborative analysis sessions
 * with branching, merging, and snapshot management.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,

  Paper,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Alert,
  // Card, CardContent - unused
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot
} from '@mui/lab';
import {
  Add as AddIcon,
  // History as HistoryIcon - unused
  AccountTree as BranchIcon,
  // Merge as MergeIcon - unused
  Restore as RestoreIcon,
  Compare as CompareIcon,
  Download as DownloadIcon,
  // Upload as UploadIcon - unused
  MoreVert as MoreIcon,
  Person as PersonIcon,
  Schedule as ScheduleIcon,
  Tag as TagIcon,
  Commit as CommitIcon
} from '@mui/icons-material';

interface VersionSnapshot {
  version_id: string;
  author_id: string;
  timestamp: string;
  description: string;
  branch_name: string;
  checksum: string;
  parent_version?: string;
}

interface VersionControlPanelProps {
  sessionId: string;
  currentVersion: string;
  versionHistory: string[];
  onCreateSnapshot: (description: string, branchName?: string) => void;
}

export const VersionControlPanel: React.FC<VersionControlPanelProps> = ({
  sessionId: _sessionId,
  currentVersion,
  versionHistory: _versionHistory,
  onCreateSnapshot
}) => {
  const [createSnapshotOpen, setCreateSnapshotOpen] = useState(false);
  const [compareDialogOpen, setCompareDialogOpen] = useState(false);
  const [branchDialogOpen, setBranchDialogOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedVersion, setSelectedVersion] = useState<string | null>(null);
  const [selectedBranch, setSelectedBranch] = useState('main');
  const [viewMode, setViewMode] = useState<'timeline' | 'tree'>('timeline');

  const [snapshotForm, setSnapshotForm] = useState({
    description: '',
    branchName: 'main'
  });

  const [compareVersions, setCompareVersions] = useState({
    version1: '',
    version2: ''
  });

  // Fallback data structure for when API is unavailable
  const fallbackVersions: VersionSnapshot[] = [
    {
      version_id: 'v1',
      author_id: 'user1',
      timestamp: '2025-01-14T10:00:00Z',
      description: 'Initial causal graph setup',
      branch_name: 'main',
      checksum: 'abc123'
    },
    {
      version_id: 'v2',
      author_id: 'user2',
      timestamp: '2025-01-14T11:30:00Z',
      description: 'Added seismic-volcanic relationships',
      branch_name: 'main',
      checksum: 'def456',
      parent_version: 'v1'
    },
    {
      version_id: 'v3',
      author_id: 'user1',
      timestamp: '2025-01-14T14:15:00Z',
      description: 'Experimental deformation analysis',
      branch_name: 'deformation-branch',
      checksum: 'ghi789',
      parent_version: 'v2'
    }
  ];

  // Real version data from API
  const [versions, setVersions] = useState<VersionSnapshot[]>(fallbackVersions);

  // Fetch real version history
  useEffect(() => {
    const fetchVersions = async () => {
      try {
        const response = await fetch('/api/v1/collaborative/versions');
        if (response.ok) {
          const versionData = await response.json();
          setVersions(versionData);
        }
      } catch (error) {
        console.error('Failed to fetch version history:', error);
      }
    };
    fetchVersions();
  }, []);

  const [branches] = useState(['main', 'deformation-branch', 'geochemical-analysis']);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const getBranchVersions = () => {
    return versions.filter(v => v.branch_name === selectedBranch)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  };

  // const getBranchColor = (_branchName: string) => { // Unused function
  //   const colors = ['primary', 'secondary', 'success', 'warning', 'info', 'error'];
  //   const hash = _branchName.split('').reduce((a: number, b: string) => {
  //     a = ((a << 5) - a) + b.charCodeAt(0);
  //     return a & a;
  //   }, 0);
  //   return colors[Math.abs(hash) % colors.length];
  // };

  const handleCreateSnapshot = () => {
    onCreateSnapshot(snapshotForm.description, snapshotForm.branchName);
    setSnapshotForm({ description: '', branchName: 'main' });
    setCreateSnapshotOpen(false);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, versionId: string) => {
    setMenuAnchor(event.currentTarget);
    setSelectedVersion(versionId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedVersion(null);
  };

  const handleMenuAction = (action: string) => {
    if (!selectedVersion) return;

    switch (action) {
      case 'restore':
        // Restore to this version
        console.log('Restore to version:', selectedVersion);
        break;
      case 'compare':
        setCompareVersions(prev => ({ ...prev, version1: selectedVersion }));
        setCompareDialogOpen(true);
        break;
      case 'branch':
        setBranchDialogOpen(true);
        break;
      case 'download':
        // Download version
        console.log('Download version:', selectedVersion);
        break;
      case 'tag':
        // Create tag
        console.log('Tag version:', selectedVersion);
        break;
    }

    handleMenuClose();
  };

  const renderTimelineView = () => (
    <Timeline>
      {getBranchVersions().map((version, index) => (
        <TimelineItem key={version.version_id}>
          <TimelineSeparator>
            <TimelineDot 
              color={version.version_id === currentVersion ? 'primary' : 'grey'}
              variant={version.version_id === currentVersion ? 'filled' : 'outlined'}
            >
              <CommitIcon />
            </TimelineDot>
            {index < getBranchVersions().length - 1 && <TimelineConnector />}
          </TimelineSeparator>
          <TimelineContent>
            <Paper elevation={1} sx={{ p: 2, mb: 1 }}>
              <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                <Box flex={1}>
                  <Typography variant="subtitle2" gutterBottom>
                    {version.description}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Avatar sx={{ width: 20, height: 20, fontSize: '0.75rem' }}>
                      {version.author_id.charAt(0)}
                    </Avatar>
                    <Typography variant="caption" color="text.secondary">
                      {version.author_id}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(version.timestamp)}
                    </Typography>
                    {version.version_id === currentVersion && (
                      <Chip label="Current" size="small" color="primary" />
                    )}
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {version.checksum.substring(0, 8)}
                  </Typography>
                </Box>
                <IconButton
                  size="small"
                  onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleMenuOpen(e, version.version_id)}
                >
                  <MoreIcon />
                </IconButton>
              </Box>
            </Paper>
          </TimelineContent>
        </TimelineItem>
      ))}
    </Timeline>
  );

  const renderTreeView = () => (
    <List>
      {getBranchVersions().map((version) => (
        <ListItem key={version.version_id}>
          <ListItemIcon>
            <CommitIcon 
              color={version.version_id === currentVersion ? 'primary' : 'action'}
            />
          </ListItemIcon>
          <ListItemText
            primary={
              <Box display="flex" alignItems="center" gap={1}>
                <Typography variant="body2">
                  {version.description}
                </Typography>
                {version.version_id === currentVersion && (
                  <Chip label="Current" size="small" color="primary" />
                )}
              </Box>
            }
            secondary={
              <Box display="flex" alignItems="center" gap={1}>
                <PersonIcon sx={{ fontSize: 14 }} />
                <Typography variant="caption">
                  {version.author_id}
                </Typography>
                <ScheduleIcon sx={{ fontSize: 14 }} />
                <Typography variant="caption">
                  {formatTimestamp(version.timestamp)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {version.checksum.substring(0, 8)}
                </Typography>
              </Box>
            }
          />
          <ListItemSecondaryAction>
            <IconButton
              onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleMenuOpen(e, version.version_id)}
            >
              <MoreIcon />
            </IconButton>
          </ListItemSecondaryAction>
        </ListItem>
      ))}
    </List>
  );

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Version Control
          </Typography>
          <Button
            startIcon={<AddIcon />}
            variant="contained"
            size="small"
            onClick={() => setCreateSnapshotOpen(true)}
          >
            Snapshot
          </Button>
        </Box>

        {/* Branch selector and controls */}
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Branch</InputLabel>
            <Select
              value={selectedBranch}
              onChange={(e) => setSelectedBranch(e.target.value)}
              label="Branch"
            >
              {branches.map(branch => (
                <MenuItem key={branch} value={branch}>
                  <Box display="flex" alignItems="center" gap={1}>
                    <BranchIcon fontSize="small" />
                    {branch}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            size="small"
            startIcon={<BranchIcon />}
            onClick={() => setBranchDialogOpen(true)}
          >
            New Branch
          </Button>

          <Button
            size="small"
            startIcon={<CompareIcon />}
            onClick={() => setCompareDialogOpen(true)}
          >
            Compare
          </Button>
        </Box>

        {/* View mode toggle */}
        <Box display="flex" gap={1}>
          <Button
            size="small"
            variant={viewMode === 'timeline' ? 'contained' : 'outlined'}
            onClick={() => setViewMode('timeline')}
          >
            Timeline
          </Button>
          <Button
            size="small"
            variant={viewMode === 'tree' ? 'contained' : 'outlined'}
            onClick={() => setViewMode('tree')}
          >
            Tree
          </Button>
        </Box>
      </Box>

      {/* Version history */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {viewMode === 'timeline' ? renderTimelineView() : renderTreeView()}
      </Box>

      {/* Create Snapshot Dialog */}
      <Dialog open={createSnapshotOpen} onClose={() => setCreateSnapshotOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create Version Snapshot</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Description"
            value={snapshotForm.description}
            onChange={(e) => setSnapshotForm(prev => ({ ...prev, description: e.target.value }))}
            margin="normal"
            placeholder="Describe the changes in this snapshot..."
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Branch</InputLabel>
            <Select
              value={snapshotForm.branchName}
              onChange={(e) => setSnapshotForm(prev => ({ ...prev, branchName: e.target.value }))}
              label="Branch"
            >
              {branches.map(branch => (
                <MenuItem key={branch} value={branch}>
                  {branch}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateSnapshotOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateSnapshot} variant="contained">Create Snapshot</Button>
        </DialogActions>
      </Dialog>

      {/* Compare Versions Dialog */}
      <Dialog open={compareDialogOpen} onClose={() => setCompareDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Compare Versions</DialogTitle>
        <DialogContent>
          <Box display="flex" gap={2} mb={2}>
            <FormControl fullWidth>
              <InputLabel>Version 1</InputLabel>
              <Select
                value={compareVersions.version1}
                onChange={(e) => setCompareVersions(prev => ({ ...prev, version1: e.target.value }))}
                label="Version 1"
              >
                {versions.map(version => (
                  <MenuItem key={version.version_id} value={version.version_id}>
                    {version.description} ({version.checksum.substring(0, 8)})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Version 2</InputLabel>
              <Select
                value={compareVersions.version2}
                onChange={(e) => setCompareVersions(prev => ({ ...prev, version2: e.target.value }))}
                label="Version 2"
              >
                {versions.map(version => (
                  <MenuItem key={version.version_id} value={version.version_id}>
                    {version.description} ({version.checksum.substring(0, 8)})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          
          {compareVersions.version1 && compareVersions.version2 && (
            <Alert severity="info">
              Comparison view would show differences between the selected versions here.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCompareDialogOpen(false)}>Close</Button>
          <Button variant="contained" disabled={!compareVersions.version1 || !compareVersions.version2}>
            Compare
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Branch Dialog */}
      <Dialog open={branchDialogOpen} onClose={() => setBranchDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Branch</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Branch Name"
            margin="normal"
            placeholder="feature/new-analysis"
          />
          <TextField
            fullWidth
            label="Description"
            margin="normal"
            placeholder="Describe the purpose of this branch..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBranchDialogOpen(false)}>Cancel</Button>
          <Button variant="contained">Create Branch</Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleMenuAction('restore')}>
          <RestoreIcon sx={{ mr: 1 }} />
          Restore to this version
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('compare')}>
          <CompareIcon sx={{ mr: 1 }} />
          Compare with current
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('branch')}>
          <BranchIcon sx={{ mr: 1 }} />
          Create branch from here
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => handleMenuAction('download')}>
          <DownloadIcon sx={{ mr: 1 }} />
          Download
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('tag')}>
          <TagIcon sx={{ mr: 1 }} />
          Create tag
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default VersionControlPanel;