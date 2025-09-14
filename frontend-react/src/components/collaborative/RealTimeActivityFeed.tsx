/**
 * Real-Time Activity Feed Component
 * 
 * Displays live activity stream for collaborative analysis sessions
 * with filtering, search, and activity type categorization.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  // Badge - unused
  Divider,
  Card,
  CardContent,
  InputAdornment,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Search as SearchIcon,
  // FilterList as FilterIcon - unused
  Clear as ClearIcon,
  Person as PersonIcon,
  Comment as CommentIcon,
  Science as HypothesisIcon,
  Edit as EditIcon,
  // Add as AddIcon, Delete as DeleteIcon, Merge as MergeIcon - unused
  AccountTree as BranchIcon,
  ThumbUp as VoteIcon,
  Share as ShareIcon,
  Visibility as ViewIcon,
  Warning as ConflictIcon,
  CheckCircle as ResolveIcon,
  // Schedule as TimeIcon - unused
  TrendingUp as TrendingIcon
} from '@mui/icons-material';

interface Activity {
  id: string;
  type: string;
  user_id: string;
  username: string;
  timestamp: string;
  description: string;
  target_id?: string;
  target_type?: string;
  metadata?: any;
}

interface RealTimeActivityFeedProps {
  sessionId: string;
  activities: Activity[];
}

export interface RealTimeActivityFeedRef {
  addActivity: (activity: Activity) => void;
  clearActivities: () => void;
}

export const RealTimeActivityFeed = forwardRef<RealTimeActivityFeedRef, RealTimeActivityFeedProps>(
  ({ sessionId: _sessionId, activities: initialActivities }, ref) => {
    const [activities, setActivities] = useState<Activity[]>(initialActivities);
    const [filteredActivities, setFilteredActivities] = useState<Activity[]>([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState('all');
    const [autoScroll, setAutoScroll] = useState(true);
    const [showTimestamps, setShowTimestamps] = useState(true);

    // Mock activities for demonstration
    useEffect(() => {
      const mockActivities: Activity[] = [
        {
          id: '1',
          type: 'user_joined',
          user_id: 'user1',
          username: 'Dr. Sarah Chen',
          timestamp: new Date().toISOString(),
          description: 'joined the session'
        },
        {
          id: '2',
          type: 'annotation_added',
          user_id: 'user2',
          username: 'Prof. Rodriguez',
          timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
          description: 'added annotation to seismic-volcanic relationship',
          target_id: 'rel_123',
          target_type: 'causal_relationship'
        },
        {
          id: '3',
          type: 'hypothesis_created',
          user_id: 'user1',
          username: 'Dr. Sarah Chen',
          timestamp: new Date(Date.now() - 10 * 60000).toISOString(),
          description: 'created hypothesis "Magma Chamber Pressure Correlation"',
          target_id: 'hyp_456',
          target_type: 'hypothesis'
        },
        {
          id: '4',
          type: 'conflict_detected',
          user_id: 'system',
          username: 'System',
          timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
          description: 'detected edit conflict between user1 and user2',
          metadata: { conflict_id: 'conf_789' }
        },
        {
          id: '5',
          type: 'version_created',
          user_id: 'user3',
          username: 'Dr. Johnson',
          timestamp: new Date(Date.now() - 20 * 60000).toISOString(),
          description: 'created version snapshot "Added deformation analysis"',
          target_id: 'v_101',
          target_type: 'version'
        }
      ];

      if (initialActivities.length === 0) {
        setActivities(mockActivities);
      }
    }, [initialActivities]);

    // Filter activities based on search and type
    useEffect(() => {
      let filtered = activities;

      if (searchTerm) {
        filtered = filtered.filter(activity =>
          activity.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          activity.username.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }

      if (filterType !== 'all') {
        filtered = filtered.filter(activity => activity.type === filterType);
      }

      // Sort by timestamp (newest first)
      filtered.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      setFilteredActivities(filtered);
    }, [activities, searchTerm, filterType]);

    const activityTypes = [
      { value: 'all', label: 'All Activities', icon: <TrendingIcon /> },
      { value: 'user_joined', label: 'User Actions', icon: <PersonIcon /> },
      { value: 'annotation_added', label: 'Annotations', icon: <CommentIcon /> },
      { value: 'hypothesis_created', label: 'Hypotheses', icon: <HypothesisIcon /> },
      { value: 'edit_operation', label: 'Edits', icon: <EditIcon /> },
      { value: 'conflict_detected', label: 'Conflicts', icon: <ConflictIcon /> },
      { value: 'version_created', label: 'Versions', icon: <BranchIcon /> }
    ];

    const getActivityIcon = (type: string) => {
      const iconMap: Record<string, JSX.Element> = {
        user_joined: <PersonIcon />,
        user_left: <PersonIcon />,
        annotation_added: <CommentIcon />,
        annotation_updated: <EditIcon />,
        hypothesis_created: <HypothesisIcon />,
        hypothesis_updated: <EditIcon />,
        edit_operation: <EditIcon />,
        conflict_detected: <ConflictIcon />,
        conflict_resolved: <ResolveIcon />,
        version_created: <BranchIcon />,
        vote_cast: <VoteIcon />,
        share_session: <ShareIcon />,
        view_change: <ViewIcon />
      };
      return iconMap[type] || <TrendingIcon />;
    };

    const getActivityColor = (type: string) => {
      const colorMap: Record<string, string> = {
        user_joined: 'success',
        user_left: 'warning',
        annotation_added: 'info',
        annotation_updated: 'info',
        hypothesis_created: 'primary',
        hypothesis_updated: 'primary',
        edit_operation: 'default',
        conflict_detected: 'error',
        conflict_resolved: 'success',
        version_created: 'secondary',
        vote_cast: 'info',
        share_session: 'primary',
        view_change: 'default'
      };
      return colorMap[type] || 'default';
    };

    const formatTimestamp = (timestamp: string) => {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString();
    };

    const addActivity = (activity: Activity) => {
      setActivities(prev => [activity, ...prev]);
    };

    const clearActivities = () => {
      setActivities([]);
    };

    // Expose methods through ref
    useImperativeHandle(ref, () => ({
      addActivity,
      clearActivities
    }));

    const handleClearSearch = () => {
      setSearchTerm('');
    };

    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6" gutterBottom>
            Activity Feed
          </Typography>

          {/* Search */}
          <TextField
            fullWidth
            size="small"
            placeholder="Search activities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
              endAdornment: searchTerm && (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={handleClearSearch}>
                    <ClearIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
            sx={{ mb: 1 }}
          />

          {/* Filter */}
          <FormControl fullWidth size="small" sx={{ mb: 1 }}>
            <InputLabel>Filter by Type</InputLabel>
            <Select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              label="Filter by Type"
            >
              {activityTypes.map(type => (
                <MenuItem key={type.value} value={type.value}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {type.icon}
                    {type.label}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Options */}
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <FormControlLabel
              control={
                <Switch
                  checked={showTimestamps}
                  onChange={(e) => setShowTimestamps(e.target.checked)}
                  size="small"
                />
              }
              label="Show timestamps"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                  size="small"
                />
              }
              label="Auto-scroll"
            />
          </Box>
        </Box>

        {/* Activity List */}
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          {filteredActivities.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                {searchTerm || filterType !== 'all' 
                  ? 'No activities match your filters'
                  : 'No activities yet'
                }
              </Typography>
            </Box>
          ) : (
            <List>
              {filteredActivities.map((activity, index) => (
                <React.Fragment key={activity.id}>
                  <ListItem alignItems="flex-start">
                    <ListItemAvatar>
                      <Avatar 
                        sx={{ 
                          bgcolor: `${getActivityColor(activity.type)}.main`,
                          width: 32,
                          height: 32
                        }}
                      >
                        {getActivityIcon(activity.type)}
                      </Avatar>
                    </ListItemAvatar>

                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                          <Typography variant="body2" fontWeight="medium">
                            {activity.username}
                          </Typography>
                          <Chip
                            label={activity.type.replace('_', ' ')}
                            size="small"
                            color={getActivityColor(activity.type) as any}
                            variant="outlined"
                          />
                          {showTimestamps && (
                            <Typography variant="caption" color="text.secondary">
                              {formatTimestamp(activity.timestamp)}
                            </Typography>
                          )}
                        </Box>
                      }
                      secondary={
                        <Typography variant="body2" color="text.secondary">
                          {activity.description}
                        </Typography>
                      }
                    />
                  </ListItem>
                  
                  {index < filteredActivities.length - 1 && <Divider variant="inset" component="li" />}
                </React.Fragment>
              ))}
            </List>
          )}
        </Box>

        {/* Activity Summary */}
        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <Card variant="outlined">
            <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Session Activity Summary
              </Typography>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box display="flex" gap={2}>
                  <Box textAlign="center">
                    <Typography variant="h6" color="primary">
                      {activities.length}
                    </Typography>
                    <Typography variant="caption">Total</Typography>
                  </Box>
                  <Box textAlign="center">
                    <Typography variant="h6" color="success.main">
                      {activities.filter(a => a.type.includes('created') || a.type.includes('added')).length}
                    </Typography>
                    <Typography variant="caption">Created</Typography>
                  </Box>
                  <Box textAlign="center">
                    <Typography variant="h6" color="warning.main">
                      {activities.filter(a => a.type.includes('conflict')).length}
                    </Typography>
                    <Typography variant="caption">Conflicts</Typography>
                  </Box>
                </Box>
                <Tooltip title="Activity trend">
                  <TrendingIcon color="action" />
                </Tooltip>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    );
  }
);

RealTimeActivityFeed.displayName = 'RealTimeActivityFeed';

export default RealTimeActivityFeed;