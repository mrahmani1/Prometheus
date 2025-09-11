/**
 * Resource Assessment page placeholder
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';

const ResourceAssessment: React.FC = () => {
  const { t } = useTranslation();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('resources.title')}
      </Typography>
      <Typography variant="body1">
        Resource assessment interface will be implemented in task 4.4
      </Typography>
    </Box>
  );
};

export default ResourceAssessment;