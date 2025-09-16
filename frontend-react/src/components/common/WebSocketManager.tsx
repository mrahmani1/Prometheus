/**
 * WebSocket connection manager for real-time data
 */

import React, { useEffect, useRef } from 'react';
import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { setWebSocketStatus } from '../../store/slices/monitoringSlice';
import { addNotification } from '../../store/slices/notificationSlice';
import { WEBSOCKET_CONFIG } from '../../utils/constants';

const WebSocketManager: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isAuthenticated } = useAppSelector(state => state.auth);
  const { connected, messageCount } = useAppSelector(state => state.monitoring.websocket);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = () => {
    if (!isAuthenticated || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      wsRef.current = new WebSocket(WEBSOCKET_CONFIG.BASE_URL);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        dispatch(setWebSocketStatus({
          connected: true,
          reconnecting: false,
          lastConnected: new Date(),
        }));
        reconnectAttemptsRef.current = 0;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Handle different message types
          switch (data.type) {
            case 'seismic_event':
              // Dispatch seismic event update
              break;
            case 'volcanic_alert':
              // Dispatch volcanic alert
              dispatch(addNotification({
                type: 'warning',
                title: 'Volcanic Alert',
                message: data.message,
                persistent: true,
              }));
              break;
            case 'system_status':
              // Update system health
              break;
            default:
              console.log('Unknown message type:', data.type);
          }
          
          dispatch(setWebSocketStatus({
            messageCount: messageCount + 1,
          }));
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        dispatch(setWebSocketStatus({
          connected: false,
          reconnecting: false,
        }));

        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && reconnectAttemptsRef.current < WEBSOCKET_CONFIG.MAX_RECONNECT_ATTEMPTS) {
          scheduleReconnect();
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        dispatch(addNotification({
          type: 'error',
          title: 'Connection Error',
          message: 'Lost connection to real-time data stream',
          duration: 5000,
          persistent: false,
        }));
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
    }
  };

  const scheduleReconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    dispatch(setWebSocketStatus({ reconnecting: true }));
    reconnectAttemptsRef.current += 1;

    const delay = Math.min(
      WEBSOCKET_CONFIG.RECONNECT_INTERVAL * Math.pow(2, reconnectAttemptsRef.current - 1),
      30000 // Max 30 seconds
    );

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, delay);
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User logout');
      wsRef.current = null;
    }

    dispatch(setWebSocketStatus({
      connected: false,
      reconnecting: false,
    }));
  };

  // Connect when authenticated, disconnect when not
  useEffect(() => {
    if (isAuthenticated) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [isAuthenticated]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    if (!connected) return;

    const heartbeatInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, WEBSOCKET_CONFIG.HEARTBEAT_INTERVAL);

    return () => {
      clearInterval(heartbeatInterval);
    };
  }, [connected]);

  // This component doesn't render anything
  return null;
};

export default WebSocketManager;