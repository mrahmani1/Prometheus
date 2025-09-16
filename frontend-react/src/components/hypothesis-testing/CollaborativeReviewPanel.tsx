import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert
} from '@mui/material';
import {
  Comment as CommentIcon,
  ThumbUp as ThumbUpIcon,
  Reply as ReplyIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import {
  CausalHypothesis,
  CollaborativeSession,
  Comment,
  Review,
  ReviewStatus
} from '../../types/causal-analysis';

interface CollaborativeReviewPanelProps {
  hypothesis: CausalHypothesis | null;
  collaborativeSession?: CollaborativeSession;
  onHypothesisChange: (hypothesis: CausalHypothesis) => void;
}

export const CollaborativeReviewPanel: React.FC<CollaborativeReviewPanelProps> = ({
  hypothesis,
  collaborativeSession,
  onHypothesisChange: _onHypothesisChange
}) => {
  const [newComment, setNewComment] = useState('');
  const [showReviewDialog, setShowReviewDialog] = useState(false);
  const [reviewFeedback, setReviewFeedback] = useState('');
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus>('pending');

  // Mock comments for demonstration
  const mockComments: Comment[] = [
    {
      id: 'comment1',
      author: 'Dr. Sarah Johnson',
      content: 'The causal relationship between seismic activity and volcanic unrest looks well-supported by the data.',
      timestamp: new Date(Date.now() - 3600000), // 1 hour ago
      type: 'general',
      targetElement: 'hypothesis',
      resolved: false,
      likes: 3,
      replies: []
    },
    {
      id: 'comment2',
      author: 'Prof. Michael Chen',
      content: 'Consider adding temporal lag constraints for the deformation-eruption relationship.',
      timestamp: new Date(Date.now() - 7200000), // 2 hours ago
      type: 'suggestion',
      targetElement: 'edge_deformation_eruption',
      resolved: false,
      likes: 1,
      replies: []
    }
  ];

  const mockReviews: Review[] = [
    {
      id: 'review1',
      reviewer: 'Dr. Sarah Johnson',
      hypothesis: hypothesis?.id || '',
      status: 'approved',
      feedback: 'Well-structured hypothesis with good statistical support.',
      timestamp: new Date(Date.now() - 86400000), // 1 day ago
      suggestions: []
    }
  ];

  if (!hypothesis) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        Create a hypothesis to enable collaborative review features.
      </Alert>
    );
  }

  if (!collaborativeSession) {
    return (
      <Alert severity="warning" sx={{ m: 2 }}>
        Join a collaborative session to participate in reviews and discussions.
      </Alert>
    );
  }

  const handleAddComment = () => {
    if (!newComment.trim()) return;

    const comment: Comment = {
      id: `comment_${Date.now()}`,
      author: collaborativeSession.currentUser,
      content: newComment,
      timestamp: new Date(),
      type: 'general',
      targetElement: 'hypothesis',
      resolved: false,
      likes: 0,
      replies: []
    };

    // In a real implementation, this would be sent to the server
    console.log('Adding comment:', comment);
    setNewComment('');
  };

  const handleSubmitReview = () => {
    if (!reviewFeedback.trim()) return;

    const review: Review = {
      id: `review_${Date.now()}`,
      reviewer: collaborativeSession.currentUser,
      hypothesis: hypothesis.id,
      status: reviewStatus,
      feedback: reviewFeedback,
      timestamp: new Date(),
      suggestions: []
    };

    // In a real implementation, this would be sent to the server
    console.log('Submitting review:', review);
    setShowReviewDialog(false);
    setReviewFeedback('');
    setReviewStatus('pending');
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <GroupIcon />
        Collaborative Review
      </Typography>

      {/* Active Participants */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            Active Participants ({collaborativeSession.participants.length})
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {collaborativeSession.participants.map((participant) => (
              <Chip
                key={participant.id}
                avatar={<Avatar sx={{ width: 24, height: 24 }}>{participant.name.charAt(0)}</Avatar>}
                label={participant.name}
                color={participant.isOnline ? 'success' : 'default'}
                variant={participant.isOnline ? 'filled' : 'outlined'}
                size="small"
              />
            ))}
          </Box>
        </CardContent>
      </Card>

      {/* Comments Section */}
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CommentIcon />
            Comments & Discussion
          </Typography>
          
          {/* Add Comment */}
          <Box sx={{ mb: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              placeholder="Add a comment or suggestion..."
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              sx={{ mb: 1 }}
            />
            <Button
              variant="contained"
              onClick={handleAddComment}
              disabled={!newComment.trim()}
              size="small"
            >
              Add Comment
            </Button>
          </Box>
          
          {/* Comments List */}
          <List>
            {mockComments.map((comment) => (
              <ListItem key={comment.id} alignItems="flex-start">
                <ListItemAvatar>
                  <Avatar>{comment.author.charAt(0)}</Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography variant="subtitle2">{comment.author}</Typography>
                      <Chip
                        label={comment.type}
                        size="small"
                        variant="outlined"
                        color={comment.type === 'suggestion' ? 'primary' : 'default'}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {comment.timestamp.toLocaleString()}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {comment.content}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <IconButton size="small">
                          <ThumbUpIcon fontSize="small" />
                        </IconButton>
                        <Typography variant="caption">{comment.likes}</Typography>
                        <IconButton size="small">
                          <ReplyIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Reviews Section */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">
              Peer Reviews ({mockReviews.length})
            </Typography>
            <Button
              variant="outlined"
              onClick={() => setShowReviewDialog(true)}
              size="small"
            >
              Submit Review
            </Button>
          </Box>
          
          <List>
            {mockReviews.map((review) => (
              <ListItem key={review.id} alignItems="flex-start">
                <ListItemAvatar>
                  <Avatar>{review.reviewer.charAt(0)}</Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Typography variant="subtitle2">{review.reviewer}</Typography>
                      <Chip
                        label={review.status}
                        size="small"
                        color={
                          review.status === 'approved' ? 'success' :
                          review.status === 'needs-revision' ? 'warning' :
                          review.status === 'rejected' ? 'error' : 'default'
                        }
                      />
                      <Typography variant="caption" color="text.secondary">
                        {review.timestamp.toLocaleString()}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Typography variant="body2">
                      {review.feedback}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Review Dialog */}
      <Dialog open={showReviewDialog} onClose={() => setShowReviewDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Submit Peer Review</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Review Feedback"
            value={reviewFeedback}
            onChange={(e) => setReviewFeedback(e.target.value)}
            sx={{ mt: 2, mb: 2 }}
            placeholder="Provide detailed feedback on the hypothesis structure, statistical validity, and scientific reasoning..."
          />
          
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label="Approved"
              clickable
              color={reviewStatus === 'approved' ? 'success' : 'default'}
              onClick={() => setReviewStatus('approved')}
            />
            <Chip
              label="Needs Revision"
              clickable
              color={reviewStatus === 'needs-revision' ? 'warning' : 'default'}
              onClick={() => setReviewStatus('needs-revision')}
            />
            <Chip
              label="Rejected"
              clickable
              color={reviewStatus === 'rejected' ? 'error' : 'default'}
              onClick={() => setReviewStatus('rejected')}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowReviewDialog(false)}>Cancel</Button>
          <Button onClick={handleSubmitReview} variant="contained" disabled={!reviewFeedback.trim()}>
            Submit Review
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};