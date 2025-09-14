/**
 * Conflict Resolution Dialog Component
 * 
 * Provides interface for resolving conflicts between simultaneous edits
 * in collaborative analysis sessions with multiple resolution strategies.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  RadioGroup,
  FormControlLabel,
  Radio,
  Chip,
  Alert,
  Paper,
  Avatar,
  Stack
} from '@mui/material';
import {
  Merge as MergeIcon,
  LastPage as LastWriteIcon,
  Build as ManualIcon,
  AutoMode as AutoIcon,
  Person as PersonIcon,
  Schedule as ScheduleIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

interface ConflictInfo {
  conflict_id: string;
  session_id: string;
  conflicting_operations: any[];
  resolution_strategy: string;
  timestamp: string;
}

interface ConflictResolutionDialogProps {
  open: boolean;
  conflict: ConflictInfo;
  onResolve: (conflictId: string, strategy: string, resolutionData?: any) => void;
  onClose: () => void;
}

export const ConflictResolutionDialog: React.FC<ConflictResolutionDialogProps> = ({
  open,
  conflict,
  onResolve,
  onClose
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState('operational_transform');

  const resolutionStrategies = [
    {
      value: 'operational_transform',
      label: 'Operational Transform',
      description: 'Automatically merge changes using operational transformation algorithms',
      icon: <AutoIcon />,
      recommended: true
    },
    {
      value: 'merge_changes',
      label: 'Merge Changes',
      description: 'Attempt to merge non-conflicting changes automatically',
      icon: <MergeIcon />,
      recommended: false
    },
    {
      value: 'last_write_wins',
      label: 'Last Write Wins',
      description: 'Keep the most recent change and discard others',
      icon: <LastWriteIcon />,
      recommended: false
    },
    {
      value: 'manual_resolution',
      label: 'Manual Resolution',
      description: 'Manually review and resolve conflicts',
      icon: <ManualIcon />,
      recommended: false
    }
  ];

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getUserColor = (userId: string) => {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'];
    const hash = userId.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  };

  const getOperationDescription = (operation: any) => {
    const { operation_type, target_type, target_id } = operation;
    return `${operation_type} ${target_type} ${target_id}`;
  };

  const handleResolve = () => {
    onResolve(conflict.conflict_id, selectedStrategy);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <WarningIcon color="warning" />
          <Typography variant="h6">Resolve Edit Conflict</Typography>
          <Chip
            label={`${conflict.conflicting_operations.length} operations`}
            size="small"
            color="warning"
          />
        </Box>
      </DialogTitle>

      <DialogContent>
        <Alert severity="warning" sx={{ mb: 3 }}>
          Multiple users have made conflicting changes to the same element.
          Please choose a resolution strategy to continue.
        </Alert>

        {/* Conflict Details */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Conflict Details
            </Typography>

            <Box display="flex" alignItems="center" gap={2} mb={2}>
              <ScheduleIcon color="action" />
              <Typography variant="body2">
                Detected at: {formatTimestamp(conflict.timestamp)}
              </Typography>
            </Box>

            {/* Timeline of conflicting operations */}
            <Typography variant="subtitle2" gutterBottom>
              Conflicting Operations:
            </Typography>

            <Stack spacing={2} sx={{ mt: 2 }}>
              {conflict.conflicting_operations.map((operation, index) => (
                <Box key={index} display="flex" alignItems="flex-start" gap={2}>
                  <Avatar
                    sx={{
                      bgcolor: getUserColor(operation.user_id),
                      width: 32,
                      height: 32
                    }}
                  >
                    <PersonIcon sx={{ fontSize: 16 }} />
                  </Avatar>
                  <Paper elevation={1} sx={{ p: 2, flex: 1 }}>
                    <Typography variant="body2" fontWeight="bold">
                      {operation.user_id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {getOperationDescription(operation)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTimestamp(operation.timestamp)}
                    </Typography>
                  </Paper>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>

        {/* Resolution Strategy Selection */}
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Choose Resolution Strategy
            </Typography>

            <RadioGroup
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
            >
              {resolutionStrategies.map((strategy) => (
                <Box key={strategy.value} sx={{ mb: 2 }}>
                  <FormControlLabel
                    value={strategy.value}
                    control={<Radio />}
                    label={
                      <Box display="flex" alignItems="center" gap={1}>
                        {strategy.icon}
                        <Typography variant="body1">
                          {strategy.label}
                        </Typography>
                        {strategy.recommended && (
                          <Chip label="Recommended" size="small" color="primary" />
                        )}
                      </Box>
                    }
                  />
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ ml: 4, mt: 0.5 }}
                  >
                    {strategy.description}
                  </Typography>
                </Box>
              ))}
            </RadioGroup>

            {/* Strategy-specific options */}
            {selectedStrategy === 'manual_resolution' && (
              <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Manual Resolution Options
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  You will be able to manually review and edit the conflicting changes
                  in the next step. This provides the most control but requires more time.
                </Typography>
              </Box>
            )}

            {selectedStrategy === 'operational_transform' && (
              <Box sx={{ mt: 3, p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Operational Transform Details
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This strategy will automatically transform the operations to be compatible
                  with each other, preserving the intent of all users' changes where possible.
                </Typography>
              </Box>
            )}

            {selectedStrategy === 'last_write_wins' && (
              <Box sx={{ mt: 3, p: 2, bgcolor: 'warning.50', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Last Write Wins Warning
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This strategy will keep only the most recent change and discard all others.
                  Some work may be lost. Consider using operational transform instead.
                </Typography>

                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" fontWeight="bold">
                    Operation to keep:
                  </Typography>
                  {conflict.conflicting_operations
                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
                    .slice(0, 1)
                    .map((operation, index) => (
                      <Box key={index} sx={{ mt: 1, p: 1, bgcolor: 'white', borderRadius: 1 }}>
                        <Typography variant="body2">
                          <strong>{operation.user_id}:</strong> {getOperationDescription(operation)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatTimestamp(operation.timestamp)}
                        </Typography>
                      </Box>
                    ))
                  }
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>
          Cancel
        </Button>
        <Button
          onClick={handleResolve}
          variant="contained"
          color="primary"
          startIcon={resolutionStrategies.find(s => s.value === selectedStrategy)?.icon}
        >
          Resolve Conflict
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConflictResolutionDialog;