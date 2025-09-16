/**
 * Peer Review Interface Component
 * 
 * Provides comprehensive peer review functionality for collaborative
 * analysis with structured review criteria and approval workflows.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Rating,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemText,

  Avatar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,

  LinearProgress,

  Accordion,
  AccordionSummary,
  AccordionDetails,

  // IconButton, Tooltip - unused
} from '@mui/material';
import {
  RateReview as ReviewIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Schedule as PendingIcon,
  // Star as StarIcon, Comment as CommentIcon, Assignment as CriteriaIcon, Person as ReviewerIcon, Timeline as ProgressIcon - unused
  Visibility as ViewIcon,
  Edit as EditIcon,
  Send as SubmitIcon
} from '@mui/icons-material';

interface PeerReview {
  review_id: string;
  target_id: string;
  target_type: string;
  reviewer_id: string;
  reviewer_name: string;
  review_criteria: string[];
  scores: Record<string, number>;
  comments: string;
  recommendations: string[];
  approval_status: string;
  created_at: string;
  completed_at?: string;
}

interface ReviewTarget {
  id: string;
  type: string;
  title: string;
  description: string;
  author: string;
  created_at: string;
}

interface PeerReviewInterfaceProps {
  sessionId: string;
  reviews: PeerReview[];
  onCreateReview: (targetId: string, targetType: string, reviewData: any) => void;
  onUpdateReview: (reviewId: string, updates: any) => void;
}

export const PeerReviewInterface: React.FC<PeerReviewInterfaceProps> = ({
  sessionId: _sessionId,
  reviews,
  onCreateReview,
  onUpdateReview
}) => {
  const [createReviewOpen, setCreateReviewOpen] = useState(false);
  const [viewReviewOpen, setViewReviewOpen] = useState(false);
  const [selectedReview, setSelectedReview] = useState<PeerReview | null>(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterType, setFilterType] = useState('all');

  const [newReview, setNewReview] = useState({
    target_id: '',
    target_type: 'hypothesis',
    review_criteria: [] as string[],
    scores: {} as Record<string, number>,
    comments: '',
    recommendations: [] as string[]
  });

  // Fallback data structure for when API is unavailable
  const fallbackTargets: ReviewTarget[] = [
    {
      id: 'hyp_1',
      type: 'hypothesis',
      title: 'Magma Chamber Pressure Correlation',
      description: 'Seismic activity increases correlate with rising magma chamber pressure',
      author: 'Dr. Sarah Chen',
      created_at: '2025-01-14T10:00:00Z'
    },
    {
      id: 'rel_1',
      type: 'causal_relationship',
      title: 'Seismic-Deformation Coupling',
      description: 'Strong causal relationship between seismic events and ground deformation',
      author: 'Prof. Rodriguez',
      created_at: '2025-01-14T11:30:00Z'
    }
  ];

  // Real review targets from API
  const [reviewTargets, setReviewTargets] = useState<ReviewTarget[]>(fallbackTargets);

  // Fetch real review targets
  useEffect(() => {
    const fetchReviewTargets = async () => {
      try {
        const response = await fetch('/api/v1/collaborative/review-targets');
        if (response.ok) {
          const targets = await response.json();
          setReviewTargets(targets);
        }
      } catch (error) {
        console.error('Failed to fetch review targets:', error);
      }
    };
    fetchReviewTargets();
  }, []);

  const reviewCriteria = {
    hypothesis: [
      'Scientific Validity',
      'Methodological Rigor',
      'Evidence Quality',
      'Statistical Significance',
      'Reproducibility',
      'Novelty',
      'Clarity of Presentation'
    ],
    causal_relationship: [
      'Causal Inference Validity',
      'Statistical Evidence',
      'Temporal Precedence',
      'Confounding Control',
      'Mechanism Plausibility',
      'Data Quality',
      'Interpretation Accuracy'
    ]
  };

  const statusColors = {
    pending: 'warning',
    approved: 'success',
    rejected: 'error',
    revision_requested: 'info'
  } as const;

  const statusIcons = {
    pending: <PendingIcon />,
    approved: <ApproveIcon />,
    rejected: <RejectIcon />,
    revision_requested: <EditIcon />
  };

  const getFilteredReviews = () => {
    let filtered = reviews;

    if (filterStatus !== 'all') {
      filtered = filtered.filter(r => r.approval_status === filterStatus);
    }

    if (filterType !== 'all') {
      filtered = filtered.filter(r => r.target_type === filterType);
    }

    return filtered.sort((a, b) => 
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  };

  const getAverageScore = (scores: Record<string, number>) => {
    const values = Object.values(scores);
    return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const handleCreateReview = () => {
    if (!newReview.target_id || !newReview.comments.trim()) return;

    const criteria = reviewCriteria[newReview.target_type as keyof typeof reviewCriteria] || [];
    const reviewData = {
      review_criteria: criteria,
      scores: newReview.scores,
      comments: newReview.comments,
      recommendations: newReview.recommendations
    };

    onCreateReview(newReview.target_id, newReview.target_type, reviewData);
    
    setNewReview({
      target_id: '',
      target_type: 'hypothesis',
      review_criteria: [],
      scores: {},
      comments: '',
      recommendations: []
    });
    setCreateReviewOpen(false);
  };

  const handleScoreChange = (criterion: string, score: number) => {
    setNewReview(prev => ({
      ...prev,
      scores: { ...prev.scores, [criterion]: score }
    }));
  };

  const handleViewReview = (review: PeerReview) => {
    setSelectedReview(review);
    setViewReviewOpen(true);
  };

  const handleUpdateReviewStatus = (reviewId: string, status: string) => {
    onUpdateReview(reviewId, { approval_status: status, completed_at: new Date().toISOString() });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            Peer Reviews ({reviews.length})
          </Typography>
          <Button
            startIcon={<ReviewIcon />}
            variant="contained"
            size="small"
            onClick={() => setCreateReviewOpen(true)}
          >
            Create Review
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
              <MenuItem value="pending">Pending</MenuItem>
              <MenuItem value="approved">Approved</MenuItem>
              <MenuItem value="rejected">Rejected</MenuItem>
              <MenuItem value="revision_requested">Revision Requested</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Type</InputLabel>
            <Select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              label="Type"
            >
              <MenuItem value="all">All Types</MenuItem>
              <MenuItem value="hypothesis">Hypotheses</MenuItem>
              <MenuItem value="causal_relationship">Relationships</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Reviews List */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 1 }}>
        {getFilteredReviews().map((review) => {
          const averageScore = getAverageScore(review.scores);
          const target = reviewTargets.find(t => t.id === review.target_id);

          return (
            <Card key={review.review_id} sx={{ mb: 2 }}>
              <CardContent>
                {/* Header */}
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                  <Box flex={1}>
                    <Typography variant="h6" gutterBottom>
                      Review of: {target?.title || review.target_id}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Chip
                        icon={statusIcons[review.approval_status as keyof typeof statusIcons]}
                        label={review.approval_status.replace('_', ' ')}
                        size="small"
                        color={statusColors[review.approval_status as keyof typeof statusColors]}
                      />
                      <Chip
                        label={review.target_type.replace('_', ' ')}
                        size="small"
                        variant="outlined"
                      />
                      <Typography variant="caption" color="text.secondary">
                        by {review.reviewer_name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatTimestamp(review.created_at)}
                      </Typography>
                    </Box>
                  </Box>
                </Box>

                {/* Score Summary */}
                <Box display="flex" alignItems="center" gap={2} mb={2}>
                  <Box>
                    <Typography variant="caption" display="block">
                      Overall Score
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Rating value={averageScore} readOnly precision={0.1} size="small" />
                      <Typography variant="body2">
                        {averageScore.toFixed(1)}/5
                      </Typography>
                    </Box>
                  </Box>
                  <Box flex={1}>
                    <Typography variant="caption" display="block">
                      Progress
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={review.completed_at ? 100 : 50}
                      color={review.completed_at ? 'success' : 'warning'}
                      sx={{ height: 6 }}
                    />
                  </Box>
                </Box>

                {/* Comments Preview */}
                <Typography variant="body2" color="text.secondary" paragraph>
                  {review.comments.length > 150 
                    ? `${review.comments.substring(0, 150)}...`
                    : review.comments
                  }
                </Typography>

                {/* Detailed Scores */}
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2">
                      Detailed Scores ({Object.keys(review.scores).length} criteria)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {Object.entries(review.scores).map(([criterion, score]) => (
                        <ListItem key={criterion}>
                          <ListItemText
                            primary={criterion}
                            secondary={
                              <Box display="flex" alignItems="center" gap={1}>
                                <Rating value={score} readOnly size="small" />
                                <Typography variant="caption">
                                  {score}/5
                                </Typography>
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>

                {/* Recommendations */}
                {review.recommendations.length > 0 && (
                  <Box mt={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      Recommendations
                    </Typography>
                    <List dense>
                      {review.recommendations.map((rec, index) => (
                        <ListItem key={index}>
                          <ListItemText primary={rec} />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </CardContent>

              <CardActions>
                <Button
                  size="small"
                  startIcon={<ViewIcon />}
                  onClick={() => handleViewReview(review)}
                >
                  View Details
                </Button>
                
                {review.approval_status === 'pending' && (
                  <>
                    <Button
                      size="small"
                      startIcon={<ApproveIcon />}
                      color="success"
                      onClick={() => handleUpdateReviewStatus(review.review_id, 'approved')}
                    >
                      Approve
                    </Button>
                    <Button
                      size="small"
                      startIcon={<RejectIcon />}
                      color="error"
                      onClick={() => handleUpdateReviewStatus(review.review_id, 'rejected')}
                    >
                      Reject
                    </Button>
                  </>
                )}
              </CardActions>
            </Card>
          );
        })}
      </Box>

      {/* Create Review Dialog */}
      <Dialog open={createReviewOpen} onClose={() => setCreateReviewOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Peer Review</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Target Type</InputLabel>
            <Select
              value={newReview.target_type}
              onChange={(e) => setNewReview(prev => ({ ...prev, target_type: e.target.value }))}
              label="Target Type"
            >
              <MenuItem value="hypothesis">Hypothesis</MenuItem>
              <MenuItem value="causal_relationship">Causal Relationship</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel>Target Item</InputLabel>
            <Select
              value={newReview.target_id}
              onChange={(e) => setNewReview(prev => ({ ...prev, target_id: e.target.value }))}
              label="Target Item"
            >
              {reviewTargets
                .filter(t => t.type === newReview.target_type)
                .map(target => (
                  <MenuItem key={target.id} value={target.id}>
                    {target.title}
                  </MenuItem>
                ))
              }
            </Select>
          </FormControl>

          {/* Review Criteria Scores */}
          {newReview.target_type && (
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>
                Review Criteria
              </Typography>
              {reviewCriteria[newReview.target_type as keyof typeof reviewCriteria]?.map(criterion => (
                <Box key={criterion} display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                  <Typography variant="body2">{criterion}</Typography>
                  <Rating
                    value={newReview.scores[criterion] || 0}
                    onChange={(_, value) => handleScoreChange(criterion, value || 0)}
                  />
                </Box>
              ))}
            </Box>
          )}

          <TextField
            fullWidth
            multiline
            rows={4}
            label="Comments"
            value={newReview.comments}
            onChange={(e) => setNewReview(prev => ({ ...prev, comments: e.target.value }))}
            margin="normal"
            placeholder="Provide detailed feedback and comments..."
          />

          <TextField
            fullWidth
            multiline
            rows={2}
            label="Recommendations"
            margin="normal"
            placeholder="Enter recommendations (one per line)"
            onChange={(e) => {
              const recommendations = e.target.value.split('\n').filter(r => r.trim());
              setNewReview(prev => ({ ...prev, recommendations }));
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateReviewOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateReview} variant="contained" startIcon={<SubmitIcon />}>
            Submit Review
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Review Dialog */}
      <Dialog open={viewReviewOpen} onClose={() => setViewReviewOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Review Details
          {selectedReview && (
            <Chip
              icon={statusIcons[selectedReview.approval_status as keyof typeof statusIcons]}
              label={selectedReview.approval_status.replace('_', ' ')}
              size="small"
              color={statusColors[selectedReview.approval_status as keyof typeof statusColors]}
              sx={{ ml: 1 }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {selectedReview && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {reviewTargets.find(t => t.id === selectedReview.target_id)?.title}
              </Typography>
              
              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Reviewer Information
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Avatar sx={{ width: 32, height: 32 }}>
                    {selectedReview.reviewer_name.charAt(0)}
                  </Avatar>
                  <Box>
                    <Typography variant="body2">{selectedReview.reviewer_name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Submitted: {formatTimestamp(selectedReview.created_at)}
                    </Typography>
                  </Box>
                </Box>
              </Box>

              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Detailed Scores
                </Typography>
                {Object.entries(selectedReview.scores).map(([criterion, score]) => (
                  <Box key={criterion} display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                    <Typography variant="body2">{criterion}</Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Rating value={score} readOnly size="small" />
                      <Typography variant="caption">{score}/5</Typography>
                    </Box>
                  </Box>
                ))}
              </Box>

              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Comments
                </Typography>
                <Typography variant="body2" paragraph>
                  {selectedReview.comments}
                </Typography>
              </Box>

              {selectedReview.recommendations.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Recommendations
                  </Typography>
                  <List>
                    {selectedReview.recommendations.map((rec, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={rec} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewReviewOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default PeerReviewInterface;