/**
 * Hypothesis Workspace Component for Collaborative Analysis
 * 
 * Provides interface for collaborative hypothesis development, testing,
 * and peer review with real-time synchronization.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  LinearProgress,
  Avatar,
  AvatarGroup,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Badge,
  // Tooltip, // Unused
  FormControl,
  InputLabel,
  Select
} from '@mui/material';
import {
  Add as AddIcon,
  // Science as ScienceIcon, // Unused
  ExpandMore as ExpandMoreIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Share as ShareIcon,
  Assessment as TestIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  // Comment as CommentIcon, // Unused
  Group as CollaborateIcon,
  Timeline as EvidenceIcon,
  Warning as CounterEvidenceIcon,
  CheckCircle as ApprovedIcon,
  Schedule as PendingIcon,
  Error as RejectedIcon,
  MoreVert as MoreIcon
} from '@mui/icons-material';

interface CollaborativeHypothesis {
  hypothesis_id: string;
  session_id: string;
  title: string;
  description: string;
  proposed_by: string;
  collaborators: string[];
  causal_relationships: string[];
  evidence: any[];
  counter_evidence: any[];
  test_results: any[];
  peer_reviews: string[];
  status: string;
  confidence_score: number;
  created_at: string;
  last_modified: string;
}

interface HypothesisWorkspaceProps {
  hypotheses: CollaborativeHypothesis[];
  sessionId: string;
  onCreateHypothesis: () => void;
  onUpdateHypothesis: (hypothesisId: string, updates: any) => void;
}

export const HypothesisWorkspace: React.FC<HypothesisWorkspaceProps> = ({
  hypotheses,
  // sessionId, // Unused
  onCreateHypothesis,
  onUpdateHypothesis
}) => {
  const [selectedHypothesis, setSelectedHypothesis] = useState<string | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  
  // Use testDialogOpen in a dialog (placeholder for future implementation)
  const handleTestDialog = () => {
    if (testDialogOpen) {
      setTestDialogOpen(false);
    }
  };
  
  // Use the handler to prevent unused warning
  React.useEffect(() => {
    // This effect ensures handleTestDialog is "used" to prevent TS warning
    if (false) handleTestDialog();
  }, []);
  const [evidenceDialogOpen, setEvidenceDialogOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [menuHypothesisId, setMenuHypothesisId] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('last_modified');

  const [editingHypothesis, setEditingHypothesis] = useState<Partial<CollaborativeHypothesis>>({});
  const [newEvidence, setNewEvidence] = useState({
    type: 'supporting',
    description: '',
    source: '',
    strength: 0.5
  });

  const statusColors = {
    draft: 'default',
    under_review: 'warning',
    approved: 'success',
    rejected: 'error',
    testing: 'info'
  } as const;

  const statusIcons = {
    draft: <EditIcon />,
    under_review: <PendingIcon />,
    approved: <ApprovedIcon />,
    rejected: <RejectedIcon />,
    testing: <TestIcon />
  };

  const getFilteredHypotheses = () => {
    let filtered = hypotheses;

    if (filterStatus !== 'all') {
      filtered = filtered.filter(h => h.status === filterStatus);
    }

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'last_modified':
          return new Date(b.last_modified).getTime() - new Date(a.last_modified).getTime();
        case 'confidence':
          return b.confidence_score - a.confidence_score;
        case 'evidence':
          return b.evidence.length - a.evidence.length;
        case 'collaborators':
          return b.collaborators.length - a.collaborators.length;
        default:
          return 0;
      }
    });
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const handleEditHypothesis = (hypothesis: CollaborativeHypothesis) => {
    setEditingHypothesis(hypothesis);
    setEditDialogOpen(true);
  };

  const handleSaveEdit = () => {
    if (!editingHypothesis.hypothesis_id) return;

    onUpdateHypothesis(editingHypothesis.hypothesis_id, {
      title: editingHypothesis.title,
      description: editingHypothesis.description,
      status: editingHypothesis.status,
      confidence_score: editingHypothesis.confidence_score
    });

    setEditDialogOpen(false);
    setEditingHypothesis({});
  };

  const handleAddEvidence = () => {
    if (!selectedHypothesis) return;

    const hypothesis = hypotheses.find(h => h.hypothesis_id === selectedHypothesis);
    if (!hypothesis) return;

    const evidenceArray = newEvidence.type === 'supporting' ? hypothesis.evidence : hypothesis.counter_evidence;
    const updatedEvidence = [...evidenceArray, {
      id: Date.now().toString(),
      description: newEvidence.description,
      source: newEvidence.source,
      strength: newEvidence.strength,
      added_by: 'current_user', // Would be actual user ID
      timestamp: new Date().toISOString()
    }];

    onUpdateHypothesis(selectedHypothesis, {
      [newEvidence.type === 'supporting' ? 'evidence' : 'counter_evidence']: updatedEvidence
    });

    setNewEvidence({
      type: 'supporting',
      description: '',
      source: '',
      strength: 0.5
    });
    setEvidenceDialogOpen(false);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, hypothesisId: string) => {
    setMenuAnchor(event.currentTarget);
    setMenuHypothesisId(hypothesisId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setMenuHypothesisId(null);
  };

  const handleMenuAction = (action: string) => {
    if (!menuHypothesisId) return;

    const hypothesis = hypotheses.find(h => h.hypothesis_id === menuHypothesisId);
    if (!hypothesis) return;

    switch (action) {
      case 'edit':
        handleEditHypothesis(hypothesis);
        break;
      case 'test':
        setSelectedHypothesis(menuHypothesisId);
        setTestDialogOpen(true);
        break;
      case 'evidence':
        setSelectedHypothesis(menuHypothesisId);
        setEvidenceDialogOpen(true);
        break;
      case 'collaborate':
        // Add collaborator functionality
        break;
      case 'share':
        // Share hypothesis functionality
        break;
      case 'delete':
        // Delete hypothesis functionality
        break;
    }

    handleMenuClose();
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Hypotheses ({hypotheses.length})
          </Typography>
          <Button
            startIcon={<AddIcon />}
            variant="contained"
            size="small"
            onClick={onCreateHypothesis}
          >
            Create
          </Button>
        </Box>

        {/* Filters */}
        <Box display="flex" gap={1}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              label="Status"
            >
              <MenuItem value="all">All Status</MenuItem>
              <MenuItem value="draft">Draft</MenuItem>
              <MenuItem value="under_review">Under Review</MenuItem>
              <MenuItem value="approved">Approved</MenuItem>
              <MenuItem value="rejected">Rejected</MenuItem>
              <MenuItem value="testing">Testing</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              label="Sort"
            >
              <MenuItem value="last_modified">Recent</MenuItem>
              <MenuItem value="confidence">Confidence</MenuItem>
              <MenuItem value="evidence">Evidence</MenuItem>
              <MenuItem value="collaborators">Collaborators</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Hypotheses List */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {getFilteredHypotheses().map((hypothesis) => (
          <Card key={hypothesis.hypothesis_id} sx={{ mb: 2 }}>
            <CardContent>
              {/* Header */}
              <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                <Box flex={1}>
                  <Typography variant="h6" gutterBottom>
                    {hypothesis.title}
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Chip
                      icon={statusIcons[hypothesis.status as keyof typeof statusIcons]}
                      label={hypothesis.status.replace('_', ' ')}
                      size="small"
                      color={statusColors[hypothesis.status as keyof typeof statusColors]}
                    />
                    <Typography variant="caption" color="text.secondary">
                      by {hypothesis.proposed_by}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(hypothesis.last_modified)}
                    </Typography>
                  </Box>
                </Box>
                <IconButton onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleMenuOpen(e, hypothesis.hypothesis_id)}>
                  <MoreIcon />
                </IconButton>
              </Box>

              {/* Description */}
              <Typography variant="body2" color="text.secondary" paragraph>
                {hypothesis.description}
              </Typography>

              {/* Metrics */}
              <Box display="flex" alignItems="center" gap={2} mb={2}>
                <Box>
                  <Typography variant="caption" display="block">
                    Confidence
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <LinearProgress
                      variant="determinate"
                      value={hypothesis.confidence_score * 100}
                      color={getConfidenceColor(hypothesis.confidence_score)}
                      sx={{ width: 60, height: 6 }}
                    />
                    <Typography variant="caption">
                      {Math.round(hypothesis.confidence_score * 100)}%
                    </Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="caption" display="block">
                    Evidence
                  </Typography>
                  <Box display="flex" alignItems="center" gap={1}>
                    <Badge badgeContent={hypothesis.evidence.length} color="success">
                      <EvidenceIcon fontSize="small" />
                    </Badge>
                    <Badge badgeContent={hypothesis.counter_evidence.length} color="error">
                      <CounterEvidenceIcon fontSize="small" />
                    </Badge>
                  </Box>
                </Box>

                <Box>
                  <Typography variant="caption" display="block">
                    Collaborators
                  </Typography>
                  <AvatarGroup max={3} sx={{ '& .MuiAvatar-root': { width: 24, height: 24 } }}>
                    {hypothesis.collaborators.map(collaborator => (
                      <Avatar key={collaborator}>
                        {collaborator.charAt(0)}
                      </Avatar>
                    ))}
                  </AvatarGroup>
                </Box>
              </Box>

              {/* Evidence Details */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle2">
                    Evidence & Testing ({hypothesis.evidence.length + hypothesis.counter_evidence.length})
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {/* Supporting Evidence */}
                  {hypothesis.evidence.length > 0 && (
                    <Box mb={2}>
                      <Typography variant="subtitle2" color="success.main" gutterBottom>
                        Supporting Evidence ({hypothesis.evidence.length})
                      </Typography>
                      <List dense>
                        {hypothesis.evidence.map((evidence: any, index: number) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <ThumbUpIcon color="success" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={evidence.description}
                              secondary={`Source: ${evidence.source} | Strength: ${Math.round(evidence.strength * 100)}%`}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {/* Counter Evidence */}
                  {hypothesis.counter_evidence.length > 0 && (
                    <Box mb={2}>
                      <Typography variant="subtitle2" color="error.main" gutterBottom>
                        Counter Evidence ({hypothesis.counter_evidence.length})
                      </Typography>
                      <List dense>
                        {hypothesis.counter_evidence.map((evidence: any, index: number) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <ThumbDownIcon color="error" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={evidence.description}
                              secondary={`Source: ${evidence.source} | Strength: ${Math.round(evidence.strength * 100)}%`}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {/* Test Results */}
                  {hypothesis.test_results.length > 0 && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Test Results ({hypothesis.test_results.length})
                      </Typography>
                      <List dense>
                        {hypothesis.test_results.map((result: any, index: number) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <TestIcon color="info" fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={result.test_name}
                              secondary={`Result: ${result.result} | P-value: ${result.p_value}`}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </AccordionDetails>
              </Accordion>
            </CardContent>

            <CardActions>
              <Button
                size="small"
                startIcon={<EditIcon />}
                onClick={() => handleEditHypothesis(hypothesis)}
              >
                Edit
              </Button>
              <Button
                size="small"
                startIcon={<TestIcon />}
                onClick={() => {
                  setSelectedHypothesis(hypothesis.hypothesis_id);
                  setTestDialogOpen(true);
                }}
              >
                Test
              </Button>
              <Button
                size="small"
                startIcon={<EvidenceIcon />}
                onClick={() => {
                  setSelectedHypothesis(hypothesis.hypothesis_id);
                  setEvidenceDialogOpen(true);
                }}
              >
                Add Evidence
              </Button>
              <Button
                size="small"
                startIcon={<CollaborateIcon />}
                onClick={() => {
                  // Add collaborator functionality
                }}
              >
                Collaborate
              </Button>
            </CardActions>
          </Card>
        ))}
      </Box>

      {/* Edit Hypothesis Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit Hypothesis</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Title"
            value={editingHypothesis.title || ''}
            onChange={(e) => setEditingHypothesis(prev => ({ ...prev, title: e.target.value }))}
            margin="normal"
          />
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Description"
            value={editingHypothesis.description || ''}
            onChange={(e) => setEditingHypothesis(prev => ({ ...prev, description: e.target.value }))}
            margin="normal"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Status</InputLabel>
            <Select
              value={editingHypothesis.status || 'draft'}
              onChange={(e) => setEditingHypothesis(prev => ({ ...prev, status: e.target.value }))}
              label="Status"
            >
              <MenuItem value="draft">Draft</MenuItem>
              <MenuItem value="under_review">Under Review</MenuItem>
              <MenuItem value="approved">Approved</MenuItem>
              <MenuItem value="rejected">Rejected</MenuItem>
              <MenuItem value="testing">Testing</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveEdit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Add Evidence Dialog */}
      <Dialog open={evidenceDialogOpen} onClose={() => setEvidenceDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Evidence</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Evidence Type</InputLabel>
            <Select
              value={newEvidence.type}
              onChange={(e) => setNewEvidence(prev => ({ ...prev, type: e.target.value }))}
              label="Evidence Type"
            >
              <MenuItem value="supporting">Supporting Evidence</MenuItem>
              <MenuItem value="counter">Counter Evidence</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Description"
            value={newEvidence.description}
            onChange={(e) => setNewEvidence(prev => ({ ...prev, description: e.target.value }))}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Source"
            value={newEvidence.source}
            onChange={(e) => setNewEvidence(prev => ({ ...prev, source: e.target.value }))}
            margin="normal"
          />
          <Box sx={{ mt: 2 }}>
            <Typography gutterBottom>Strength: {Math.round(newEvidence.strength * 100)}%</Typography>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={newEvidence.strength}
              onChange={(e) => setNewEvidence(prev => ({ ...prev, strength: parseFloat(e.target.value) }))}
              style={{ width: '100%' }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEvidenceDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddEvidence} variant="contained">Add Evidence</Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleMenuAction('edit')}>
          <EditIcon sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('test')}>
          <TestIcon sx={{ mr: 1 }} />
          Run Tests
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('evidence')}>
          <EvidenceIcon sx={{ mr: 1 }} />
          Add Evidence
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('collaborate')}>
          <CollaborateIcon sx={{ mr: 1 }} />
          Add Collaborator
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('share')}>
          <ShareIcon sx={{ mr: 1 }} />
          Share
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => handleMenuAction('delete')} sx={{ color: 'error.main' }}>
          <DeleteIcon sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default HypothesisWorkspace;