import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  TextField,
  InputAdornment,
  Card,
  CardContent,
  CardActions,
  Chip,
  Grid,
  Tabs,
  Tab,
  Avatar,
  IconButton,
  // Tooltip
} from '@mui/material';
import {
  Search,
  Category,
  // Person,
  // Schedule,
  // Visibility,
  GetApp,
  Star,
  StarBorder
} from '@mui/icons-material';
import { WorkflowTemplate } from '@/types/workflow';
import { geologicalWorkflowTemplates } from '@/data/workflow/workflowTemplates';

interface TemplateLibraryProps {
  open: boolean;
  onClose: () => void;
  onLoadTemplate: (template: WorkflowTemplate) => void;
}

export const TemplateLibrary: React.FC<TemplateLibraryProps> = ({
  open,
  onClose,
  onLoadTemplate
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const categories = useMemo(() => {
    const cats = new Set<string>();
    geologicalWorkflowTemplates.forEach(template => {
      cats.add(template.category);
    });
    return ['all', ...Array.from(cats).sort()];
  }, []);

  const filteredTemplates = useMemo(() => {
    return geologicalWorkflowTemplates.filter(template => {
      const matchesSearch = !searchTerm || 
        template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
      
      return matchesSearch && matchesCategory;
    });
  }, [searchTerm, selectedCategory]);

  const handleLoadTemplate = (template: WorkflowTemplate) => {
    onLoadTemplate(template);
  };

  const toggleFavorite = (templateId: string) => {
    const newFavorites = new Set(favorites);
    if (newFavorites.has(templateId)) {
      newFavorites.delete(templateId);
    } else {
      newFavorites.add(templateId);
    }
    setFavorites(newFavorites);
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'seismic-analysis': '#E91E63',
      'volcanic-monitoring': '#FF5722',
      'deformation-analysis': '#9C27B0',
      'geochemical-analysis': '#00BCD4',
      'ai-modeling': '#673AB7',
      'hazard-assessment': '#FF9800',
      'resource-exploration': '#8BC34A',
      'visualization': '#4CAF50'
    };
    return colors[category] || '#757575';
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Category />
          <Typography variant="h6">Workflow Template Library</Typography>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0 }}>
        {/* Search and Filter Bar */}
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Search templates..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                )
              }}
            />
          </Box>

          <Tabs
            value={selectedCategory}
            onChange={(_: any, value: any) => setSelectedCategory(value)}
            variant="scrollable"
            scrollButtons="auto"
          >
            {categories.map(category => (
              <Tab
                key={category}
                label={category === 'all' ? 'All Categories' : category.split('-').map(word => 
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ')}
                value={category}
              />
            ))}
          </Tabs>
        </Box>

        {/* Template Grid */}
        <Box sx={{ p: 2, height: 'calc(100% - 120px)', overflow: 'auto' }}>
          <Grid container spacing={2}>
            {filteredTemplates.map(template => (
              <Grid item xs={12} sm={6} md={4} key={template.id}>
                <Card
                  variant="outlined"
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    transition: 'all 0.2s',
                    '&:hover': {
                      boxShadow: 4,
                      transform: 'translateY(-2px)'
                    }
                  }}
                >
                  {/* Template Header */}
                  <Box
                    sx={{
                      p: 1.5,
                      backgroundColor: getCategoryColor(template.category),
                      color: 'white',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}
                  >
                    <Typography variant="subtitle2" fontWeight="bold" noWrap>
                      {template.name}
                    </Typography>
                    
                    <IconButton
                      size="small"
                      onClick={() => toggleFavorite(template.id)}
                      sx={{ color: 'white' }}
                    >
                      {favorites.has(template.id) ? <Star /> : <StarBorder />}
                    </IconButton>
                  </Box>

                  <CardContent sx={{ flex: 1, p: 1.5 }}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        display: '-webkit-box',
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        mb: 1
                      }}
                    >
                      {template.description}
                    </Typography>

                    {/* Tags */}
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                      {template.tags.slice(0, 3).map(tag => (
                        <Chip
                          key={tag}
                          label={tag}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.65rem', height: 20 }}
                        />
                      ))}
                      {template.tags.length > 3 && (
                        <Chip
                          label={`+${template.tags.length - 3}`}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.65rem', height: 20 }}
                        />
                      )}
                    </Box>

                    {/* Template Stats */}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 'auto' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Avatar sx={{ width: 20, height: 20, fontSize: '0.7rem' }}>
                          {template.author.charAt(0).toUpperCase()}
                        </Avatar>
                        <Typography variant="caption" color="text.secondary">
                          {template.author}
                        </Typography>
                      </Box>
                      
                      <Typography variant="caption" color="text.secondary">
                        v{template.version}
                      </Typography>
                    </Box>

                    {/* Node/Connection Count */}
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        Nodes: {template.workflow.nodes.length}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Connections: {template.workflow.connections.length}
                      </Typography>
                    </Box>
                  </CardContent>

                  <CardActions sx={{ p: 1.5, pt: 0 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      size="small"
                      startIcon={<GetApp />}
                      onClick={() => handleLoadTemplate(template)}
                      sx={{
                        backgroundColor: getCategoryColor(template.category),
                        '&:hover': {
                          backgroundColor: getCategoryColor(template.category),
                          filter: 'brightness(0.9)'
                        }
                      }}
                    >
                      Load Template
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>

          {filteredTemplates.length === 0 && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '50%',
                textAlign: 'center'
              }}
            >
              <Category sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No templates found
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Try adjusting your search terms or category filter
              </Typography>
            </Box>
          )}
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>
          {filteredTemplates.length} template{filteredTemplates.length !== 1 ? 's' : ''} available
        </Typography>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};