/**
 * Global notification system component
 */

import React from 'react';
import { Snackbar, Alert, AlertTitle, IconButton, Stack } from '@mui/material';
import { Close } from '@mui/icons-material';
import { useAppSelector, useAppDispatch } from '../../hooks/redux';
import { removeNotification } from '../../store/slices/notificationSlice';

const NotificationSystem: React.FC = () => {
  const { notifications, settings } = useAppSelector(state => state.notifications);
  const dispatch = useAppDispatch();

  const handleClose = (notificationId: string) => {
    dispatch(removeNotification(notificationId));
  };

  return (
    <Stack
      spacing={1}
      sx={{
        position: 'fixed',
        top: settings.position.includes('top') ? 24 : 'auto',
        bottom: settings.position.includes('bottom') ? 24 : 'auto',
        left: settings.position.includes('left') ? 24 : 'auto',
        right: settings.position.includes('right') ? 24 : 'auto',
        zIndex: 1400,
        maxWidth: 400,
      }}
    >
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.persistent ? null : (notification.duration || settings.defaultDuration)}
          onClose={() => handleClose(notification.id)}
        >
          <Alert
            severity={notification.type}
            variant="filled"
            action={
              <IconButton
                size="small"
                aria-label="close"
                color="inherit"
                onClick={() => handleClose(notification.id)}
              >
                <Close fontSize="small" />
              </IconButton>
            }
          >
            {notification.title && <AlertTitle>{notification.title}</AlertTitle>}
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </Stack>
  );
};

export default NotificationSystem;