/**
 * Annotation Panel Component for Collaborative Analysis
 * 
 * Provides interface for viewing, creating, and managing annotations
 * on causal relationships with real-time collaboration features.
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
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Menu,
  MenuItem,
  Divider,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  Collapse
} from '@mui/material';
import {
  Add as AddIcon,
  Comment as CommentIcon,
  Science as ScienceIcon,
  Verified as VerifiedIcon,
  Report as CritiqueIcon,
  Lightbulb as SuggestionIcon,
  Link as ReferenceIcon,
  MoreVert as MoreIcon,
  Reply as ReplyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Flag as FlagIcon
} from '@mui/icons-material';

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

interface AnnotationPanelProps {
  annotations: CausalAnnotation[];
  sessionId: string;
  onCreateAnnotation: (relationshipId: string, content: string, type: string) => void;
  onUpdateAnnotation: (annotationId: string, updates: any) => void;
}

export const AnnotationPanel: React.FC<AnnotationPanelProps> = ({
  annotations,
  onCreateAnnotation,
  onUpdateAnnotation
}) => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [replyDialogOpen, setReplyDialogOpen] = useState(false);
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [expandedAnnotations, setExpandedAnnotations] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState('all');
  const [sortBy, setSortBy] = useState('timestamp');
  const [menuAnchor, setMenuAnchor] = useState<null | HTMLElement>(null);
  const [menuAnnotationId, setMenuAnnotationId] = useState<string | null>(null);

  const [newAnnotation, setNewAnnotation] = useState({
    causal_relationship_id: '',
    annotation_type: 'comment',
    content: '',
    tags: [] as string[],
    visibility: 'public'
  });

  const [replyContent, setReplyContent] = useState('');

  const annotationTypes = [
    { value: 'comment', label: 'Comment', icon: <CommentIcon />, color: 'default' },
    { value: 'hypothesis', label: 'Hypothesis', icon: <ScienceIcon />, color: 'primary' },
    { value: 'validation', label: 'Validation', icon: <VerifiedIcon />, color: 'success' },
    { value: 'critique', label: 'Critique', icon: <CritiqueIcon />, color: 'warning' },
    { value: 'suggestion', label: 'Suggestion', icon: <SuggestionIcon />, color: 'info' },
    { value: 'reference', label: 'Reference', icon: <ReferenceIcon />, color: 'secondary' }
  ];

  const getAnnotationTypeInfo = (type: string) => {
    return annotationTypes.find(t => t.value === type) || annotationTypes[0];
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

  const getFilteredAnnotations = () => {
    let filtered = annotations.filter(ann => !ann.parent_annotation_id); // Top-level only

    if (filterType !== 'all') {
      filtered = filtered.filter(ann => ann.annotation_type === filterType);
    }

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'timestamp':
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        case 'votes':
          const aVotes = Object.values(a.votes).reduce((sum, vote) => sum + vote, 0);
          const bVotes = Object.values(b.votes).reduce((sum, vote) => sum + vote, 0);
          return bVotes - aVotes;
        case 'replies':
          return b.replies.length - a.replies.length;
        default:
          return 0;
      }
    });
  };

  const getReplies = (annotationId: string) => {
    return annotations.filter(ann => ann.parent_annotation_id === annotationId);
  };

  const getVoteScore = (annotation: CausalAnnotation) => {
    return Object.values(annotation.votes).reduce((sum, vote) => sum + vote, 0);
  };

  const handleCreateAnnotation = () => {
    if (!newAnnotation.content.trim() || !newAnnotation.causal_relationship_id) return;

    onCreateAnnotation(
      newAnnotation.causal_relationship_id,
      newAnnotation.content,
      newAnnotation.annotation_type
    );

    setNewAnnotation({
      causal_relationship_id: '',
      annotation_type: 'comment',
      content: '',
      tags: [],
      visibility: 'public'
    });
    setCreateDialogOpen(false);
  };

  const handleReply = () => {
    if (!replyContent.trim() || !selectedAnnotation) return;

    onCreateAnnotation(
      selectedAnnotation,
      replyContent,
      'comment'
    );

    setReplyContent('');
    setReplyDialogOpen(false);
    setSelectedAnnotation(null);
  };

  const handleVote = (annotationId: string, vote: number) => {
    // Implementation would send vote to server
    console.log('Vote:', annotationId, vote);
  };

  const handleToggleExpanded = (annotationId: string) => {
    const newExpanded = new Set(expandedAnnotations);
    if (newExpanded.has(annotationId)) {
      newExpanded.delete(annotationId);
    } else {
      newExpanded.add(annotationId);
    }
    setExpandedAnnotations(newExpanded);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, annotationId: string) => {
    setMenuAnchor(event.currentTarget);
    setMenuAnnotationId(annotationId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setMenuAnnotationId(null);
  };

  const handleMenuAction = (action: string) => {
    if (!menuAnnotationId) return;

    switch (action) {
      case 'edit':
        // Open edit dialog
        break;
      case 'delete':
        // Delete annotation
        break;
      case 'flag':
        // Flag annotation
        break;
      case 'resolve':
        onUpdateAnnotation(menuAnnotationId, { resolved: true });
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
            Annotations ({annotations.length})
          </Typography>
          <Button
            startIcon={<AddIcon />}
            variant="contained"
            size="small"
            onClick={() => setCreateDialogOpen(true)}
          >
            Add
          </Button>
        </Box>

        {/* Filters */}
        <Box display="flex" gap={1} mb={1}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Type</InputLabel>
            <Select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              label="Type"
            >
              <MenuItem value="all">All Types</MenuItem>
              {annotationTypes.map(type => (
                <MenuItem key={type.value} value={type.value}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {type.icon}
                    {type.label}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort</InputLabel>
            <Select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              label="Sort"
            >
              <MenuItem value="timestamp">Recent</MenuItem>
              <MenuItem value="votes">Most Voted</MenuItem>
              <MenuItem value="replies">Most Replies</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Annotations List */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {getFilteredAnnotations().map((annotation) => {
            const typeInfo = getAnnotationTypeInfo(annotation.annotation_type);
            const replies = getReplies(annotation.annotation_id);
            const voteScore = getVoteScore(annotation);
            const isExpanded = expandedAnnotations.has(annotation.annotation_id);

            return (
              <React.Fragment key={annotation.annotation_id}>
                <ListItem alignItems="flex-start">
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: `${typeInfo.color}.main` }}>
                      {typeInfo.icon}
                    </Avatar>
                  </ListItemAvatar>

                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1} mb={1}>
                        <Typography variant="subtitle2">
                          {annotation.author_id}
                        </Typography>
                        <Chip
                          label={typeInfo.label}
                          size="small"
                          color={typeInfo.color as any}
                          variant="outlined"
                        />
                        {annotation.resolved && (
                          <Chip
                            label="Resolved"
                            size="small"
                            color="success"
                            variant="filled"
                          />
                        )}
                        <Typography variant="caption" color="text.secondary">
                          {formatTimestamp(annotation.timestamp)}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          {annotation.content}
                        </Typography>

                        {annotation.tags.length > 0 && (
                          <Box display="flex" gap={0.5} mb={1}>
                            {annotation.tags.map(tag => (
                              <Chip key={tag} label={tag} size="small" variant="outlined" />
                            ))}
                          </Box>
                        )}

                        {/* Actions */}
                        <Box display="flex" alignItems="center" gap={1}>
                          <Box display="flex" alignItems="center">
                            <IconButton
                              size="small"
                              onClick={() => handleVote(annotation.annotation_id, 1)}
                            >
                              <ThumbUpIcon fontSize="small" />
                            </IconButton>
                            <Typography variant="caption" sx={{ mx: 0.5 }}>
                              {voteScore}
                            </Typography>
                            <IconButton
                              size="small"
                              onClick={() => handleVote(annotation.annotation_id, -1)}
                            >
                              <ThumbDownIcon fontSize="small" />
                            </IconButton>
                          </Box>

                          <Button
                            size="small"
                            startIcon={<ReplyIcon />}
                            onClick={() => {
                              setSelectedAnnotation(annotation.annotation_id);
                              setReplyDialogOpen(true);
                            }}
                          >
                            Reply
                          </Button>

                          {replies.length > 0 && (
                            <Button
                              size="small"
                              startIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                              onClick={() => handleToggleExpanded(annotation.annotation_id)}
                            >
                              {replies.length} {replies.length === 1 ? 'Reply' : 'Replies'}
                            </Button>
                          )}
                        </Box>
                      </Box>
                    }
                  />

                  <ListItemSecondaryAction>
                    <IconButton
                      onClick={(e: React.MouseEvent<HTMLElement>) => handleMenuOpen(e, annotation.annotation_id)}
                    >
                      <MoreIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>

                {/* Replies */}
                <Collapse in={isExpanded}>
                  <Box sx={{ ml: 8, mr: 2 }}>
                    {replies.map(reply => (
                      <Card key={reply.annotation_id} sx={{ mb: 1 }}>
                        <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                          <Box display="flex" alignItems="center" gap={1} mb={1}>
                            <Avatar sx={{ width: 24, height: 24 }}>
                              {reply.author_id.charAt(0)}
                            </Avatar>
                            <Typography variant="caption" fontWeight="bold">
                              {reply.author_id}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {formatTimestamp(reply.timestamp)}
                            </Typography>
                          </Box>
                          <Typography variant="body2">
                            {reply.content}
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                </Collapse>

                <Divider />
              </React.Fragment>
            );
          })}
        </List>
      </Box>

      {/* Create Annotation Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create Annotation</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Causal Relationship ID"
            value={newAnnotation.causal_relationship_id}
            onChange={(e) => setNewAnnotation(prev => ({ ...prev, causal_relationship_id: e.target.value }))}
            margin="normal"
            placeholder="Enter relationship ID or select from graph"
          />

          <FormControl fullWidth margin="normal">
            <InputLabel>Annotation Type</InputLabel>
            <Select
              value={newAnnotation.annotation_type}
              onChange={(e) => setNewAnnotation(prev => ({ ...prev, annotation_type: e.target.value }))}
              label="Annotation Type"
            >
              {annotationTypes.map(type => (
                <MenuItem key={type.value} value={type.value}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {type.icon}
                    {type.label}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            fullWidth
            multiline
            rows={4}
            label="Content"
            value={newAnnotation.content}
            onChange={(e) => setNewAnnotation(prev => ({ ...prev, content: e.target.value }))}
            margin="normal"
            placeholder="Enter your annotation..."
          />

          <FormControl fullWidth margin="normal">
            <InputLabel>Visibility</InputLabel>
            <Select
              value={newAnnotation.visibility}
              onChange={(e) => setNewAnnotation(prev => ({ ...prev, visibility: e.target.value }))}
              label="Visibility"
            >
              <MenuItem value="public">Public</MenuItem>
              <MenuItem value="team">Team Only</MenuItem>
              <MenuItem value="private">Private</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateAnnotation} variant="contained">Create</Button>
        </DialogActions>
      </Dialog>

      {/* Reply Dialog */}
      <Dialog open={replyDialogOpen} onClose={() => setReplyDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Reply to Annotation</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Reply"
            value={replyContent}
            onChange={(e) => setReplyContent(e.target.value)}
            margin="normal"
            placeholder="Enter your reply..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReplyDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleReply} variant="contained">Reply</Button>
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
        <MenuItem onClick={() => handleMenuAction('resolve')}>
          <VerifiedIcon sx={{ mr: 1 }} />
          Mark Resolved
        </MenuItem>
        <MenuItem onClick={() => handleMenuAction('flag')}>
          <FlagIcon sx={{ mr: 1 }} />
          Flag
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

export default AnnotationPanel;