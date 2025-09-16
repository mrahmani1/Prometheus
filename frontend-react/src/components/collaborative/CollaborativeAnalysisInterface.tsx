/**
 * Collaborative Analysis Interface Component
 * 
 * Main interface for real-time collaborative geological analysis with multi-user
 * synchronization, annotations, version control, and peer review capabilities.
 * 
 * Features:
 * - Real-time collaborative sessions with WebSocket synchronization
 * - Interactive causal network visualization
 * - Annotation and commenting system
 * - Collaborative hypothesis development
 * - Version control with Git-like branching
 * - Peer review and validation workflows
 * - Role-based access control
 * - Conflict resolution for simultaneous edits
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  IconButton,
  Chip,
  Avatar,
  AvatarGroup,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Badge,
  Alert,
  Snackbar,
  CircularProgress,
  Tabs,
  Tab,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  People as PeopleIcon,
  Comment as CommentIcon,
  Science as ScienceIcon,
  History as HistoryIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  CloudDownload as ExportIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import { useWebSocket } from '../../hooks/useWebSocket';
import { CausalNetworkVisualization } from './CausalNetworkVisualization';
import { AnnotationPanel } from './AnnotationPanel';
import { HypothesisWorkspace } from './HypothesisWorkspace';
import { VersionControlPanel } from './VersionControlPanel';
import { ConflictResolutionDialog } from './ConflictResolutionDialog';
import { ParticipantsList } from './ParticipantsList';
import { RealTimeActivityFeed } from './RealTimeActivityFeed';

// Types and interfaces
interface CollaborativeSession {
  session_id: string;
  name: string;
  description: string;
  owner_id: string;
  participants: Record<string, string>;
  status: string;
  created_at: string;
  last_modified: string;
  causal_graph_version: string;
  shared_hypotheses: string[];
  active_annotations: string[];
  version_history: string[];
}

interface CausalAnnotation {
  annotation_id: string;
  causal_relationship_id: string;
  author_id: string;
  annotation_type: string;
  content: string;
  timestamp: string;
  parent_annotation_id?: string;
  replies: string[];
  votes: Record<string, number>;
  tags: string[];
  resolved: boolean;
  visibility: string;
}

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

interface ConflictInfo {
  conflict_id: string;
  session_id: string;
  conflicting_operations: any[];
  resolution_strategy: string;
  timestamp: string;
}



interface CollaborativeAnalysisInterfaceProps {
  sessionId: string;
  onSessionEnd?: () => void;
}

export const CollaborativeAnalysisInterface: React.FC<CollaborativeAnalysisInterfaceProps> = ({
  sessionId
}) => {
  // State management
  const [session, setSession] = useState<CollaborativeSession | null>(null);
  const [annotations, setAnnotations] = useState<CausalAnnotation[]>([]);
  const [hypotheses, setHypotheses] = useState<CollaborativeHypothesis[]>([]);
  const [conflicts, setConflicts] = useState<ConflictInfo[]>([]);
  const [selectedTab, setSelectedTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'warning' | 'info'>('info');

  // Dialog states
  const [createHypothesisOpen, setCreateHypothesisOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [conflictDialogOpen, setConflictDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [selectedConflict, setSelectedConflict] = useState<ConflictInfo | null>(null);

  // Form states
  const [newHypothesis, setNewHypothesis] = useState({
    title: '',
    description: '',
    causal_relationships: [] as string[]
  });

  // Refs
  const causalGraphRef = useRef<any>(null);
  const activityFeedRef = useRef<any>(null);

  // Hooks
  const user = useSelector((state: RootState) => state.auth.user) || { id: 'anonymous-user' };
  const token = useSelector((state: RootState) => state.auth.token) || '';
  const { 
    isConnected, 
    lastMessage, 
    sendMessage, 
    connect, 
    disconnect 
  } = useWebSocket(`/api/v1/collaborative/sessions/${sessionId}/ws?token=${token}`);

  // Initialize session and WebSocket connection
  useEffect(() => {
    const initializeSession = async () => {
      try {
        setLoading(true);
        
        // Fetch session data
        const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) {
          throw new Error('Failed to fetch session data');
        }
        
        const sessionData = await response.json();
        setSession(sessionData);
        
        // Connect WebSocket
        connect();
        
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to initialize session');
        setLoading(false);
      }
    };

    if (sessionId && token) {
      initializeSession();
    }

    return () => {
      disconnect();
    };
  }, [sessionId, token, connect, disconnect]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const message = JSON.parse(lastMessage.data);
      handleWebSocketMessage(message);
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [lastMessage]);

  // WebSocket message handler
  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'session_state':
        setSession(message.session_info);
        setAnnotations(message.annotations || []);
        setHypotheses(message.hypotheses || []);
        break;

      case 'user_joined':
        showSnackbar(`${message.user_id} joined the session`, 'info');
        updateParticipants();
        break;

      case 'user_left':
        showSnackbar(`${message.user_id} left the session`, 'info');
        updateParticipants();
        break;

      case 'annotation_added':
        setAnnotations(prev => [...prev, message.annotation]);
        showSnackbar('New annotation added', 'info');
        break;

      case 'annotation_updated':
        setAnnotations(prev => 
          prev.map(ann => 
            ann.annotation_id === message.annotation_id 
              ? { ...ann, ...message.updates }
              : ann
          )
        );
        break;

      case 'hypothesis_created':
        setHypotheses(prev => [...prev, message.hypothesis]);
        showSnackbar('New hypothesis created', 'info');
        break;

      case 'hypothesis_updated':
        setHypotheses(prev =>
          prev.map(hyp =>
            hyp.hypothesis_id === message.hypothesis_id
              ? { ...hyp, ...message.updates }
              : hyp
          )
        );
        break;

      case 'conflict_detected':
        setConflicts(prev => [...prev, {
          conflict_id: message.conflict_id,
          session_id: sessionId,
          conflicting_operations: message.operations,
          resolution_strategy: 'manual_resolution',
          timestamp: message.timestamp
        }]);
        showSnackbar('Edit conflict detected', 'warning');
        break;

      case 'conflict_resolved':
        setConflicts(prev => prev.filter(c => c.conflict_id !== message.conflict_id));
        showSnackbar('Conflict resolved', 'success');
        break;

      case 'cursor_update':
        updateUserCursor(message.user_id, message.position);
        break;

      case 'operation_applied':
        // Update UI based on applied operation
        handleOperationApplied(message.operation);
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, [sessionId]);

  // Helper functions
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'warning' | 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  const updateParticipants = async () => {
    // Fetch updated participant list
    try {
      const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const sessionData = await response.json();
        setSession(sessionData);
      }
    } catch (err) {
      console.error('Failed to update participants:', err);
    }
  };

  const updateUserCursor = (userId: string, position: any) => {
    // Update cursor position for collaborative visualization
    if (causalGraphRef.current && causalGraphRef.current.updateCursor) {
      causalGraphRef.current.updateCursor(userId, position);
    }
  };

  const handleOperationApplied = (operation: any) => {
    // Update local state based on the applied operation
    if (causalGraphRef.current) {
      causalGraphRef.current.applyOperation(operation);
    }
  };

  // Event handlers
  const handleCreateHypothesis = async () => {
    try {
      const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}/hypotheses`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(newHypothesis)
      });

      if (!response.ok) {
        throw new Error('Failed to create hypothesis');
      }

      setCreateHypothesisOpen(false);
      setNewHypothesis({ title: '', description: '', causal_relationships: [] });
      showSnackbar('Hypothesis created successfully', 'success');
    } catch (err) {
      showSnackbar('Failed to create hypothesis', 'error');
    }
  };

  const handleCreateAnnotation = async (causalRelationshipId: string, content: string, type: string) => {
    try {
      const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}/annotations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          causal_relationship_id: causalRelationshipId,
          annotation_type: type,
          content: content
        })
      });

      if (!response.ok) {
        throw new Error('Failed to create annotation');
      }

      showSnackbar('Annotation created successfully', 'success');
    } catch (err) {
      showSnackbar('Failed to create annotation', 'error');
    }
  };

  const handleResolveConflict = async (conflictId: string, strategy: string, resolutionData?: any) => {
    try {
      const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}/conflicts/${conflictId}/resolve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          resolution_strategy: strategy,
          resolution_data: resolutionData
        })
      });

      if (!response.ok) {
        throw new Error('Failed to resolve conflict');
      }

      setConflictDialogOpen(false);
      setSelectedConflict(null);
      showSnackbar('Conflict resolved successfully', 'success');
    } catch (err) {
      showSnackbar('Failed to resolve conflict', 'error');
    }
  };

  const handleExportSession = async (format: string, options: any) => {
    try {
      const response = await fetch(`/api/v1/collaborative/sessions/${sessionId}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          export_format: format,
          ...options
        })
      });

      if (!response.ok) {
        throw new Error('Failed to export session');
      }

      const exportData = await response.json();
      
      // Download the exported data
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${sessionId}_export.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setExportDialogOpen(false);
      showSnackbar('Session exported successfully', 'success');
    } catch (err) {
      showSnackbar('Failed to export session', 'error');
    }
  };

  const handleSendMessage = (message: any) => {
    if (isConnected) {
      sendMessage(JSON.stringify(message));
    }
  };

  // Render loading state
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading collaborative session...
        </Typography>
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100vh">
        <ErrorIcon color="error" sx={{ fontSize: 60, mb: 2 }} />
        <Typography variant="h6" color="error" gutterBottom>
          {error}
        </Typography>
        <Button variant="contained" onClick={() => window.location.reload()}>
          Retry
        </Button>
      </Box>
    );
  }

  if (!session) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <Typography variant="h6">Session not found</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper elevation={2} sx={{ p: 2, mb: 1 }}>
        <Grid container alignItems="center" spacing={2}>
          <Grid item xs>
            <Typography variant="h5" component="h1">
              {session.name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {session.description}
            </Typography>
          </Grid>
          
          <Grid item>
            <Box display="flex" alignItems="center" gap={1}>
              {/* Connection status */}
              <Chip
                icon={isConnected ? <CheckCircleIcon /> : <ErrorIcon />}
                label={isConnected ? 'Connected' : 'Disconnected'}
                color={isConnected ? 'success' : 'error'}
                size="small"
              />
              
              {/* Participants */}
              <AvatarGroup max={4}>
                {Object.entries(session.participants).map(([userId, role]) => (
                  <Tooltip key={userId} title={`${userId} (${role})`}>
                    <Avatar sx={{ width: 32, height: 32 }}>
                      {userId.charAt(0).toUpperCase()}
                    </Avatar>
                  </Tooltip>
                ))}
              </AvatarGroup>
              
              {/* Conflicts indicator */}
              {conflicts.length > 0 && (
                <Badge badgeContent={conflicts.length} color="warning">
                  <IconButton
                    onClick={() => {
                      setSelectedConflict(conflicts[0]);
                      setConflictDialogOpen(true);
                    }}
                  >
                    <WarningIcon />
                  </IconButton>
                </Badge>
              )}
              
              {/* Action buttons */}
              <IconButton onClick={() => setCreateHypothesisOpen(true)}>
                <AddIcon />
              </IconButton>
              
              <IconButton onClick={() => setExportDialogOpen(true)}>
                <ExportIcon />
              </IconButton>
              
              <IconButton onClick={() => setSettingsDialogOpen(true)}>
                <SettingsIcon />
              </IconButton>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left panel - Causal network visualization */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <CausalNetworkVisualization
            ref={causalGraphRef}
            sessionId={sessionId}
            onNodeSelect={(nodeId) => console.log('Node selected:', nodeId)}
            onEdgeSelect={(edgeId) => console.log('Edge selected:', edgeId)}
            onCreateAnnotation={handleCreateAnnotation}
            onSendMessage={handleSendMessage}
            readOnly={false}
          />
        </Box>

        {/* Right panel - Tabs for different tools */}
        <Paper sx={{ width: 400, display: 'flex', flexDirection: 'column' }}>
          <Tabs
            value={selectedTab}
            onChange={(_: React.SyntheticEvent, newValue: number) => setSelectedTab(newValue)}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab icon={<CommentIcon />} label="Annotations" />
            <Tab icon={<ScienceIcon />} label="Hypotheses" />
            <Tab icon={<HistoryIcon />} label="Versions" />
            <Tab icon={<PeopleIcon />} label="Participants" />
            <Tab icon={<TimelineIcon />} label="Activity" />
          </Tabs>

          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {selectedTab === 0 && (
              <AnnotationPanel
                annotations={annotations}
                sessionId={sessionId}
                onCreateAnnotation={handleCreateAnnotation}
                onUpdateAnnotation={(_id: string, _updates: any) => {
                  // Handle annotation update
                }}
              />
            )}
            
            {selectedTab === 1 && (
              <HypothesisWorkspace
                hypotheses={hypotheses}
                sessionId={sessionId}
                onCreateHypothesis={() => setCreateHypothesisOpen(true)}
                onUpdateHypothesis={(_id: string, _updates: any) => {
                  // Handle hypothesis update
                }}
              />
            )}
            
            {selectedTab === 2 && (
              <VersionControlPanel
                sessionId={sessionId}
                currentVersion={session.causal_graph_version}
                versionHistory={session.version_history}
                onCreateSnapshot={(_description: string) => {
                  // Handle version snapshot creation
                }}
              />
            )}
            
            {selectedTab === 3 && (
              <ParticipantsList
                participants={Object.entries(session.participants)}
                currentUser={user?.id || ''}
                onInviteUser={(_email: string, _role: string) => {
                  // Handle user invitation
                }}
              />
            )}
            
            {selectedTab === 4 && (
              <RealTimeActivityFeed
                ref={activityFeedRef}
                sessionId={sessionId}
                activities={[]} // Would be populated from WebSocket messages
              />
            )}
          </Box>
        </Paper>
      </Box>

      {/* Dialogs */}
      
      {/* Create Hypothesis Dialog */}
      <Dialog open={createHypothesisOpen} onClose={() => setCreateHypothesisOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Hypothesis</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Title"
            value={newHypothesis.title}
            onChange={(e) => setNewHypothesis(prev => ({ ...prev, title: e.target.value }))}
            margin="normal"
          />
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Description"
            value={newHypothesis.description}
            onChange={(e) => setNewHypothesis(prev => ({ ...prev, description: e.target.value }))}
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateHypothesisOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateHypothesis} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export Session Data</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Export Format</InputLabel>
            <Select defaultValue="json">
              <MenuItem value="json">JSON</MenuItem>
              <MenuItem value="csv">CSV</MenuItem>
              <MenuItem value="pdf">PDF Report</MenuItem>
            </Select>
          </FormControl>
          
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Include Annotations"
          />
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Include Hypotheses"
          />
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Include Peer Reviews"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => handleExportSession('json', {})} variant="contained">Export</Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={settingsDialogOpen} onClose={() => setSettingsDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Session Settings</DialogTitle>
        <DialogContent>
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Real-time Synchronization"
          />
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Show User Cursors"
          />
          <FormControlLabel
            control={<Switch defaultChecked />}
            label="Auto-save Changes"
          />
          <FormControlLabel
            control={<Switch />}
            label="Require Approval for Changes"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => setSettingsDialogOpen(false)} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Conflict Resolution Dialog */}
      {selectedConflict && (
        <ConflictResolutionDialog
          open={conflictDialogOpen}
          conflict={selectedConflict}
          onResolve={handleResolveConflict}
          onClose={() => {
            setConflictDialogOpen(false);
            setSelectedConflict(null);
          }}
        />
      )}

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
      >
        <Alert
          onClose={() => setSnackbarOpen(false)}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default CollaborativeAnalysisInterface;