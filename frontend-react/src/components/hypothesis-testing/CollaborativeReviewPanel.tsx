import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Avatar,
  Chip,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Badge,
  Tooltip,
  Card,
  CardContent,
  Grid,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider
} from '@mui/material';
import {
  Comment as CommentIcon,
  Reply as ReplyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Person as PersonIcon,
  Group as GroupIcon,
  Notifications as NotificationsIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  CollaborativeSession,
  Comment,
  Review,
  CollaborativeUser,
  ReviewStatus,
  CommentType,
  ChangeRequest,
  ConflictResolution
} from '../../types/causal-analysis';

interface CollaborativeReviewPanelProps {
  hypothesis: CausalHypothesis | null;
  collaborativeSession?: CollaborativeSession;
  onHypothesisChange: (hypothesis: CausalHypothesis) => void;
}

export const CollaborativeReviewPanel: React.FC<CollaborativeReviewPanelProps> = ({
  hypothesis,
  collaborativeSession,
  onHypothesisChange
}) => {
  const [comments, setComments] = useState<Comment[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [newComment, setNewComment] = useState('');
  const [selectedCommentType, setSelectedCommentType] = useState<CommentType>('general');
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [showResolvedComments, setShowResolvedComments] = useState(false);
  const [activeUsers, setActiveUsers] = useState<CollaborativeUser[]>([]);
  const [changeRequests, setChangeRequests] = useState<ChangeRequest[]>([]);
  const [conflicts, setConflicts] = useState<ConflictResolution[]>([]);
  const [showConflictDialog, setShowConflictDialog] = useState(false);
  const [selectedConflict, setSelectedConflict] = useState<ConflictResolution | null>(null);

  const currentUser = collaborativeSession?.currentUser || 'anonymous';
  const isOwner = hypothesis?.metadata.author === currentUser;
  const canEdit = isOwner || collaborativeSession?.permissions?.canEdit;

  // Simulate real-time updates
  useEffect(() => {
    if (!collaborativeSession) return;

    const interval = setInterval(() => {
      // Simulate receiving updates from other users
      fetchCollaborativeUpdates();
    }, 5000);

    return () => clearInterval(interval);
  }, [collaborativeSession]);

  const fetchCollaborativeUpdates = useCallback(async () => {
    // In a real implementation, this would fetch from a WebSocket or polling API
    // For now, we'll simulate some collaborative activity
    
    if (Math.random() > 0.7) { // 30% chance of new activity
      const simulatedComment: Comment = {
        id: `comment_${Date.now()}`,
        author: 'colleague@example.com',
        content: 'This causal relationship looks interesting. Have you considered the temporal lag?',
        timestamp: new Date(),
        type: 'suggestion',
        targetElement: hypothesis?.edges[0]?.id || '',
        resolved: false,
        likes: 0,
        replies: []
      };
      
      setComments(prev => [simulatedComment, ...prev]);
    }
  }, [hypothesis]);

  const handleAddComment = useCallback(async () => {
    if (!newComment.trim() || !hypothesis) return;

    const comment: Comment = {
      id: `comment_${Date.now()}`,
      author: currentUser,
      content: newComment,
      timestamp: new Date(),
      type: selectedCommentType,
      targetElement: replyingTo || hypothesis.id,
      resolved: false,
      likes: 0,
      replies: []
    };

    if (replyingTo) {
      // Add as reply
      setComments(prev => prev.map(c => 
        c.id === replyingTo 
          ? { ...c, replies: [...c.replies, comment] }
          : c
      ));
    } else {
      // Add as new comment
      setComments(prev => [comment, ...prev]);
    }

    setNewComment('');
    setReplyingTo(null);

    // Notify other collaborators
    if (collaborativeSession) {
      await notifyCollaborators(comment);
    }
  }, [newComment, selectedCommentType, replyingTo, hypothesis, currentUser, collaborativeSession]);

  const handleResolveComment = useCallback((commentId: string) => {
    setComments(prev => prev.map(c => 
      c.id === commentId 
        ? { ...c, resolved: true, resolvedBy: currentUser, resolvedAt: new Date() }
        : c
    ));
  }, [currentUser]);

  const handleLikeComment = useCallback((commentId: string) => {
    setComments(prev => prev.map(c => 
      c.id === commentId 
        ? { ...c, likes: c.likes + 1 }
        : c
    ));
  }, []);

  const handleSubmitReview = useCallback(async (status: ReviewStatus, feedback: string) => {
    if (!hypothesis) return;

    const review: Review = {
      id: `review_${Date.now()}`,
      reviewer: currentUser,
      hypothesis: hypothesis.id,
      status,
      feedback,
      timestamp: new Date(),
      suggestions: []
    };

    setReviews(prev => [review, ...prev]);

    // Update hypothesis metadata
    const updatedHypothesis = {
      ...hypothesis,
      metadata: {
        ...hypothesis.metadata,
        reviewStatus: status,
        lastReviewed: new Date(),
        reviewers: [...(hypothesis.metadata.reviewers || []), currentUser]
      }
    };

    onHypothesisChange(updatedHypothesis);
  }, [hypothesis, currentUser, onHypothesisChange]);

  const handleCreateChangeRequest = useCallback((description: string, targetElement: string) => {
    const changeRequest: ChangeRequest = {
      id: `change_${Date.now()}`,
      requester: currentUser,
      description,
      targetElement,
      status: 'pending',
      timestamp: new Date(),
      votes: { approve: 0, reject: 0 }
    };

    setChangeRequests(prev => [changeRequest, ...prev]);
  }, [currentUser]);

  const handleVoteOnChange = useCallback((changeId: string, vote: 'approve' | 'reject') => {
    setChangeRequests(prev => prev.map(cr => 
      cr.id === changeId 
        ? { 
            ...cr, 
            votes: { 
              ...cr.votes, 
              [vote]: cr.votes[vote] + 1 
            }
          }
        : cr
    ));
  }, []);

  const handleResolveConflict = useCallback((conflictId: string, resolution: 'accept' | 'reject' | 'merge') => {
    setConflicts(prev => prev.map(c => 
      c.id === conflictId 
        ? { ...c, status: 'resolved', resolution, resolvedBy: currentUser, resolvedAt: new Date() }
        : c
    ));
    setShowConflictDialog(false);
  }, [currentUser]);

  const notifyCollaborators = useCallback(async (comment: Comment) => {
    // In a real implementation, this would send notifications via WebSocket or push notifications
    console.log('Notifying collaborators about new comment:', comment);
  }, []);

  const filteredComments = useMemo(() => {
    return comments.filter(comment => showResolvedComments || !comment.resolved);
  }, [comments, showResolvedComments]);

  const pendingReviews = useMemo(() => {
    return reviews.filter(review => review.status === 'pending');
  }, [reviews]);

  const unresolvedConflicts = useMemo(() => {
    return conflicts.filter(conflict => conflict.status === 'pending');
  }, [conflicts]);

  if (!collaborativeSession) {
    return (
      <Alert severity="info">
        Collaborative features are not available. Enable collaborative mode to review and comment on hypotheses with your team.
      </Alert>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <GroupIcon />
        Collaborative Review
        {unresolvedConflicts.length > 0 && (
          <Badge badgeContent={unresolvedConflicts.length} color="error">
            <WarningIcon />
          </Badge>
        )}
      </Typography>

      {!hypothesis && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Select a hypothesis to start collaborative review.
        </Alert>
      )}

      {/* Active Collaborators */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom>
            Active Collaborators ({activeUsers.length + 1})
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              avatar={<Avatar sx={{ width: 24, height: 24 }}>{currentUser[0]?.toUpperCase()}</Avatar>}
              label={`${currentUser} (You)`}
              color="primary"
              size="small"
            />
            {activeUsers.map(user => (
              <Chip
                key={user.id}
                avatar={<Avatar sx={{ width: 24, height: 24 }}>{user.name[0]?.toUpperCase()}</Avatar>}
                label={user.name}
                color={user.isOnline ? 'success' : 'default'}
                size="small"
              />
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* Conflicts Alert */}
      {unresolvedConflicts.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="subtitle2">
            {unresolvedConflicts.length} unresolved conflict(s) require attention
          </Typography>
          <Button
            size="small"
            onClick={() => setShowConflictDialog(true)}
            sx={{ mt: 1 }}
          >
            Resolve Conflicts
          </Button>
        </Alert>
      )}

      {/* Review Status */}
      {hypothesis && (
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">Review Status</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="text.secondary">
                  Current Status: 
                  <Chip
                    label={hypothesis.metadata.reviewStatus || 'Not Reviewed'}
                    color={getStatusColor(hypothesis.metadata.reviewStatus)}
                    size="small"
                    sx={{ ml: 1 }}
                  />
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" color="text.secondary">
                  Reviewers: {hypothesis.metadata.reviewers?.length || 0}
                </Typography>
              </Grid>
            </Grid>

            {canEdit && (
              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  color="success"
                  startIcon={<CheckCircleIcon />}
                  onClick={() => handleSubmitReview('approved', 'Hypothesis approved')}
                  size="small"
                >
                  Approve
                </Button>
                <Button
                  variant="outlined"
                  color="warning"
                  startIcon={<EditIcon />}
                  onClick={() => handleSubmitReview('needs-revision', 'Needs revision')}
                  size="small"
                >
                  Request Changes
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<CancelIcon />}
                  onClick={() => handleSubmitReview('rejected', 'Hypothesis rejected')}
                  size="small"
                >
                  Reject
                </Button>
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
      )}

      {/* Comments Section */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">
            Comments & Discussions ({filteredComments.length})
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          {/* Add Comment */}
          <Box sx={{ mb: 2 }}>
            <Grid container spacing={1}>
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  placeholder={replyingTo ? 'Write a reply...' : 'Add a comment or suggestion...'}
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  size="small"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Type</InputLabel>
                  <Select
                    value={selectedCommentType}
                    onChange={(e) => setSelectedCommentType(e.target.value as CommentType)}
                  >
                    <MenuItem value="general">General</MenuItem>
                    <MenuItem value="suggestion">Suggestion</MenuItem>
                    <MenuItem value="question">Question</MenuItem>
                    <MenuItem value="concern">Concern</MenuItem>
                    <MenuItem value="praise">Praise</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Button
                variant="contained"
                startIcon={<CommentIcon />}
                onClick={handleAddComment}
                disabled={!newComment.trim()}
                size="small"
              >
                {replyingTo ? 'Reply' : 'Comment'}
              </Button>
              
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Button
                  size="small"
                  startIcon={showResolvedComments ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  onClick={() => setShowResolvedComments(!showResolvedComments)}
                >
                  {showResolvedComments ? 'Hide' : 'Show'} Resolved
                </Button>
              </Box>
            </Box>
          </Box>

          {/* Comments List */}
          <List>
            {filteredComments.map((comment) => (
              <CommentItem
                key={comment.id}
                comment={comment}
                currentUser={currentUser}
                onResolve={handleResolveComment}
                onLike={handleLikeComment}
                onReply={setReplyingTo}
                canModerate={canEdit}
              />
            ))}
          </List>

          {filteredComments.length === 0 && (
            <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ py: 4 }}>
              No comments yet. Start the discussion by adding a comment above.
            </Typography>
          )}
        </AccordionDetails>
      </Accordion>

      {/* Change Requests */}
      {changeRequests.length > 0 && (
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">
              Change Requests ({changeRequests.filter(cr => cr.status === 'pending').length} pending)
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            {changeRequests.map((request) => (
              <ChangeRequestItem
                key={request.id}
                request={request}
                currentUser={currentUser}
                onVote={handleVoteOnChange}
                canVote={request.requester !== currentUser}
              />
            ))}
          </AccordionDetails>
        </Accordion>
      )}

      {/* Conflict Resolution Dialog */}
      <Dialog open={showConflictDialog} onClose={() => setShowConflictDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Resolve Conflicts</DialogTitle>
        <DialogContent>
          {unresolvedConflicts.map((conflict) => (
            <ConflictResolutionItem
              key={conflict.id}
              conflict={conflict}
              onResolve={handleResolveConflict}
            />
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowConflictDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Comment Item Component
const CommentItem: React.FC<{
  comment: Comment;
  currentUser: string;
  onResolve: (id: string) => void;
  onLike: (id: string) => void;
  onReply: (id: string) => void;
  canModerate: boolean;
}> = ({ comment, currentUser, onResolve, onLike, onReply, canModerate }) => {
  const isAuthor = comment.author === currentUser;

  return (
    <ListItem
      sx={{
        flexDirection: 'column',
        alignItems: 'stretch',
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        mb: 1,
        backgroundColor: comment.resolved ? 'action.hover' : 'background.paper'
      }}
    >
      <Box sx={{ display: 'flex', width: '100%' }}>
        <ListItemAvatar>
          <Avatar sx={{ width: 32, height: 32 }}>
            {comment.author[0]?.toUpperCase()}
          </Avatar>
        </ListItemAvatar>
        
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="subtitle2">{comment.author}</Typography>
              <Chip label={comment.type} size="small" variant="outlined" />
              {comment.resolved && <Chip label="Resolved" size="small" color="success" />}
            </Box>
            <Typography variant="caption" color="text.secondary">
              {comment.timestamp.toLocaleString()}
            </Typography>
          </Box>
          
          <Typography variant="body2" sx={{ mb: 1 }}>
            {comment.content}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              startIcon={<ThumbUpIcon />}
              onClick={() => onLike(comment.id)}
            >
              {comment.likes}
            </Button>
            <Button
              size="small"
              startIcon={<ReplyIcon />}
              onClick={() => onReply(comment.id)}
            >
              Reply
            </Button>
            {(canModerate || isAuthor) && !comment.resolved && (
              <Button
                size="small"
                startIcon={<CheckCircleIcon />}
                onClick={() => onResolve(comment.id)}
              >
                Resolve
              </Button>
            )}
          </Box>
        </Box>
      </Box>

      {/* Replies */}
      {comment.replies.length > 0 && (
        <Box sx={{ ml: 6, mt: 1, pl: 2, borderLeft: 2, borderColor: 'divider' }}>
          {comment.replies.map((reply) => (
            <Box key={reply.id} sx={{ mb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Avatar sx={{ width: 24, height: 24 }}>
                  {reply.author[0]?.toUpperCase()}
                </Avatar>
                <Typography variant="body2" fontWeight="medium">
                  {reply.author}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {reply.timestamp.toLocaleString()}
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ ml: 4 }}>
                {reply.content}
              </Typography>
            </Box>
          ))}
        </Box>
      )}
    </ListItem>
  );
};

// Change Request Item Component
const ChangeRequestItem: React.FC<{
  request: ChangeRequest;
  currentUser: string;
  onVote: (id: string, vote: 'approve' | 'reject') => void;
  canVote: boolean;
}> = ({ request, currentUser, onVote, canVote }) => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle2">
            Change Request by {request.requester}
          </Typography>
          <Chip
            label={request.status}
            color={request.status === 'approved' ? 'success' : 
                   request.status === 'rejected' ? 'error' : 'default'}
            size="small"
          />
        </Box>
        
        <Typography variant="body2" sx={{ mb: 2 }}>
          {request.description}
        </Typography>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`👍 ${request.votes.approve}`}
              size="small"
              color="success"
              variant="outlined"
            />
            <Chip
              label={`👎 ${request.votes.reject}`}
              size="small"
              color="error"
              variant="outlined"
            />
          </Box>
          
          {canVote && request.status === 'pending' && (
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant="outlined"
                color="success"
                onClick={() => onVote(request.id, 'approve')}
              >
                Approve
              </Button>
              <Button
                size="small"
                variant="outlined"
                color="error"
                onClick={() => onVote(request.id, 'reject')}
              >
                Reject
              </Button>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

// Conflict Resolution Item Component
const ConflictResolutionItem: React.FC<{
  conflict: ConflictResolution;
  onResolve: (id: string, resolution: 'accept' | 'reject' | 'merge') => void;
}> = ({ conflict, onResolve }) => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle2" gutterBottom>
          Conflict: {conflict.description}
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Between: {conflict.conflictingUsers.join(', ')}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            color="success"
            onClick={() => onResolve(conflict.id, 'accept')}
          >
            Accept Changes
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={() => onResolve(conflict.id, 'reject')}
          >
            Reject Changes
          </Button>
          <Button
            variant="outlined"
            color="primary"
            onClick={() => onResolve(conflict.id, 'merge')}
          >
            Merge Changes
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

// Utility functions
function getStatusColor(status?: string): 'success' | 'warning' | 'error' | 'default' {
  switch (status) {
    case 'approved': return 'success';
    case 'needs-revision': return 'warning';
    case 'rejected': return 'error';
    default: return 'default';
  }
}