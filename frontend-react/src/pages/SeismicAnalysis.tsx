/**
 * Seismic Analysis page placeholder
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';

const SeismicAnalysis: React.FC = () => {
  const { t } = useTranslation();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        {t('seismic.title')}
      </Typography>
      <Typography variant="body1">
        Seismic analysis interface will be implemented in task 4.2
      </Typography>
    </Box>
  );
};

export default SeismicAnalysis;