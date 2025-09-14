/**
 * Participants List Component
 * 
 * Displays and manages participants in collaborative analysis sessions
 * with role management and real-time presence indicators.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
  Avatar,
  IconButton,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Menu,
  Badge,
  Tooltip,
  Divider
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreIcon,
  PersonAdd as InviteIcon,
  Edit as EditIcon,
  Remove as RemoveIcon,
  AdminPanelSettings as AdminIcon,
  Visibility as ViewerIcon,
  Science as AnalystIcon,
  RateReview as ReviewerIcon,
  Star as OwnerIcon,
  Circle as OnlineIcon,
  RadioButtonUnchecked as OfflineIcon,
  VideoCall as CallIcon,
  Chat as ChatIcon,
  Share as ShareIcon
} from '@mui/icons-material';

interface Participant {
  user_id: string;
  username: string;
  role: string;
  online: boolean;
  last_activity: string;
  cursor_position?: { x: number; y: number };
}

interface ParticipantsListProps {
  participants: [string, string][];
  currentUser: string;
  onInviteUser: (email: string, role: string) => void;
}

export const ParticipantsList: React.FC<ParticipantsListProps> = ({
  participants,
  currentUser,
  onInviteUser
}) => {
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [selectedParticipant, setSelectedParticipant] = useState<string | null>(null);
  
  const [inviteForm, setInviteForm] = useState({
    email: '',
    role: 'analyst'
  });

  // Mock participant data with presence info
  const [participantDetails] = useState<Record<string, Participant>>({
    'user1': {
      user_id: 'user1',
      username: 'Dr. Sarah Chen',
      role: 'owner',
      online: true,
      last_activity: new Date().toISOString(),
      cursor_position: { x: 100, y: 200 }
    },
    'user2': {
      user_id: 'user2',
      username: 'Prof. Michael Rodriguez',
      role: 'reviewer',
      online: true,
      last_activity: new Date(Date.now() - 5 * 60000).toISOString()
    },
    'user3': {
      user_id: 'user3',
      username: 'Dr. Emily Johnson',
      role: 'analyst',
      online: false,
      last_activity: new Date(Date.now() - 2 * 3600000).toISOString()
    },
    'user4': {
      user_id: 'user4',
      username: 'Alex Thompson',
      role: 'viewer',
      online: true,
      last_activity: new Date(Date.now() - 30000).toISOString()
    }
  });

  const roleInfo = {
    owner: { 
      label: 'Owner', 
      icon: <OwnerIcon />, 
      color: 'primary',
      description: 'Full control over the session'
    },
    admin: { 
      label: 'Admin', 
      icon: <AdminIcon />, 
      color: 'secondary',
      description: 'Can manage participants and settings'
    },
    reviewer: { 
      label: 'Reviewer', 
      icon: <ReviewerIcon />, 
      color: 'success',
      description: 'Can review and approve hypotheses'
    },
    analyst: { 
      label: 'Analyst', 
      icon: <AnalystIcon />, 
      color: 'info',
      description: 'Can create and edit analyses'
    },
    viewer: { 
      label: 'Viewer', 
      icon: <ViewerIcon />, 
      color: 'default',
      description: 'Read-only access'
    }
  };

  const formatLastActivity = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'Active now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  const getParticipantInfo = (userId: string, role: string): Participant => {
    return participantDetails[userId] || {
      user_id: userId,
      username: userId,
      role: role,
      online: false,
      last_activity: new Date().toISOString()
    };
  };

  const handleInviteUser = () => {
    if (!inviteForm.email.trim()) return;
    
    onInviteUser(inviteForm.email, inviteForm.role);
    setInviteForm({ email: '', role: 'analyst' });
    setInviteDialogOpen(false);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, userId: string) => {
    setMenuAnchor(event.currentTarget);
    setSelectedParticipant(userId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedParticipant(null);
  };

  const handleMenuAction = (action: string) => {
    if (!selectedParticipant) return;

    switch (action) {
      case 'change_role':
        // Open role change dialog
        break;
      case 'remove':
        // Remove participant
        break;
      case 'message':
        // Send private message
        break;
      case 'call':
        // Start video call
        break;
      case 'share_screen':
        // Share screen
        break;
    }

    handleMenuClose();
  };

  const canManageParticipants = (userRole: string) => {
    return ['owner', 'admin'].includes(userRole);
  };

  const getCurrentUserRole = () => {
    const participant = participants.find(([userId]) => userId === currentUser);
    return participant ? participant[1] : 'viewer';
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6">
            Participants ({participants.length})
          </Typography>
          {canManageParticipants(getCurrentUserRole()) && (
            <Button
              startIcon={<AddIcon />}
              variant="contained"
              size="small"
              onClick={() => setInviteDialogOpen(true)}
            >
              Invite
            </Button>
          )}
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          {participants.filter(([userId]) => getParticipantInfo(userId, '').online).length} online
        </Typography>
      </Box>

      {/* Participants List */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {participants.map(([userId, role]) => {
            const participant = getParticipantInfo(userId, role);
            const roleConfig = roleInfo[role as keyof typeof roleInfo] || roleInfo.viewer;
            
            return (
              <ListItem key={userId}>
                <ListItemAvatar>
                  <Badge
                    overlap="circular"
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    badgeContent={
                      <Tooltip title={participant.online ? 'Online' : 'Offline'}>
                        {participant.online ? (
                          <OnlineIcon sx={{ color: 'success.main', fontSize: 12 }} />
                        ) : (
                          <OfflineIcon sx={{ color: 'grey.400', fontSize: 12 }} />
                        )}
                      </Tooltip>
                    }
                  >
                    <Avatar sx={{ bgcolor: `${roleConfig.color}.main` }}>
                      {participant.username.charAt(0).toUpperCase()}
                    </Avatar>
                  </Badge>
                </ListItemAvatar>

                <ListItemText
                  primary={
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2" fontWeight={userId === currentUser ? 'bold' : 'normal'}>
                        {participant.username}
                        {userId === currentUser && ' (You)'}
                      </Typography>
                      <Chip
                        icon={roleConfig.icon}
                        label={roleConfig.label}
                        size="small"
                        color={roleConfig.color as any}
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        {formatLastActivity(participant.last_activity)}
                      </Typography>
                      {participant.cursor_position && participant.online && (
                        <Typography variant="caption" color="success.main" display="block">
                          Active in workspace
                        </Typography>
                      )}
                    </Box>
                  }
                />

                <ListItemSecondaryAction>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    {participant.online && (
                      <>
                        <Tooltip title="Start video call">
                          <IconButton size="small">
                            <CallIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Send message">
                          <IconButton size="small">
                            <ChatIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </>
                    )}
                    
                    {(canManageParticipants(getCurrentUserRole()) || userId === currentUser) && (
                      <IconButton
                        size="small"
                        onClick={(e: React.MouseEvent<HTMLButtonElement>) => handleMenuOpen(e, userId)}
                      >
                        <MoreIcon />
                      </IconButton>
                    )}
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            );
          })}
        </List>

        {/* Role Legend */}
        <Box sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Roles
          </Typography>
          {Object.entries(roleInfo).map(([key, info]) => (
            <Box key={key} display="flex" alignItems="center" gap={1} mb={0.5}>
              <Chip
                icon={info.icon}
                label={info.label}
                size="small"
                color={info.color as any}
                variant="outlined"
              />
              <Typography variant="caption" color="text.secondary">
                {info.description}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      {/* Invite User Dialog */}
      <Dialog open={inviteDialogOpen} onClose={() => setInviteDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Invite Participant</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Email Address"
            type="email"
            value={inviteForm.email}
            onChange={(e) => setInviteForm(prev => ({ ...prev, email: e.target.value }))}
            margin="normal"
            placeholder="colleague@university.edu"
          />
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Role</InputLabel>
            <Select
              value={inviteForm.role}
              onChange={(e) => setInviteForm(prev => ({ ...prev, role: e.target.value }))}
              label="Role"
            >
              {Object.entries(roleInfo).map(([key, info]) => (
                <MenuItem key={key} value={key}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {info.icon}
                    <Box>
                      <Typography variant="body2">{info.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {info.description}
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInviteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleInviteUser} variant="contained" startIcon={<InviteIcon />}>
            Send Invitation
          </Button>
        </DialogActions>
      </Dialog>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        {selectedParticipant !== currentUser && (
          <>
            <MenuItem onClick={() => handleMenuAction('message')}>
              <ChatIcon sx={{ mr: 1 }} />
              Send Message
            </MenuItem>
            <MenuItem onClick={() => handleMenuAction('call')}>
              <CallIcon sx={{ mr: 1 }} />
              Video Call
            </MenuItem>
            <MenuItem onClick={() => handleMenuAction('share_screen')}>
              <ShareIcon sx={{ mr: 1 }} />
              Share Screen
            </MenuItem>
            <Divider />
          </>
        )}
        
        {canManageParticipants(getCurrentUserRole()) && selectedParticipant !== currentUser && (
          <>
            <MenuItem onClick={() => handleMenuAction('change_role')}>
              <EditIcon sx={{ mr: 1 }} />
              Change Role
            </MenuItem>
            <MenuItem onClick={() => handleMenuAction('remove')} sx={{ color: 'error.main' }}>
              <RemoveIcon sx={{ mr: 1 }} />
              Remove from Session
            </MenuItem>
          </>
        )}
      </Menu>
    </Box>
  );
};

export default ParticipantsList;