/**
 * WebSocket Hook for Real-Time Collaboration
 * 
 * Custom React hook for managing WebSocket connections with automatic
 * reconnection, message queuing, and connection state management.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketMessage {
  data: string;
  timestamp: number;
}

interface UseWebSocketOptions {
  reconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionState: string;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: string) => void;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
}

export const useWebSocket = (
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn => {
  const {
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    heartbeatInterval = 30000,
    onOpen,
    onClose,
    onError,
    onMessage
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState<string>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const messageQueueRef = useRef<string[]>([]);
  const shouldReconnectRef = useRef(true);

  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearTimeouts();
    heartbeatTimeoutRef.current = setTimeout(() => {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.send(JSON.stringify({ type: 'ping' }));
        startHeartbeat();
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, clearTimeouts]);

  const processMessageQueue = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN && messageQueueRef.current.length > 0) {
      const messages = [...messageQueueRef.current];
      messageQueueRef.current = [];
      
      messages.forEach(message => {
        websocketRef.current?.send(message);
      });
    }
  }, []);

  const handleOpen = useCallback((event: Event) => {
    console.log('WebSocket connected');
    setIsConnected(true);
    setConnectionState('connected');
    reconnectAttemptsRef.current = 0;
    
    // Process queued messages
    processMessageQueue();
    
    // Start heartbeat
    startHeartbeat();
    
    onOpen?.(event);
  }, [onOpen, processMessageQueue, startHeartbeat]);

  const handleClose = useCallback((event: CloseEvent) => {
    console.log('WebSocket disconnected:', event.code, event.reason);
    setIsConnected(false);
    setConnectionState('disconnected');
    clearTimeouts();
    
    onClose?.(event);
    
    // Attempt reconnection if not manually closed
    if (shouldReconnectRef.current && reconnectAttemptsRef.current < reconnectAttempts) {
      setConnectionState('reconnecting');
      reconnectAttemptsRef.current++;
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Reconnection attempt ${reconnectAttemptsRef.current}/${reconnectAttempts}`);
        connect();
      }, reconnectInterval * reconnectAttemptsRef.current);
    } else if (reconnectAttemptsRef.current >= reconnectAttempts) {
      setConnectionState('failed');
      console.error('Max reconnection attempts reached');
    }
  }, [onClose, reconnectAttempts, reconnectInterval, clearTimeouts]);

  const handleError = useCallback((event: Event) => {
    console.error('WebSocket error:', event);
    setConnectionState('error');
    onError?.(event);
  }, [onError]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = {
        data: event.data,
        timestamp: Date.now()
      };
      
      // Handle ping/pong for heartbeat
      const parsedData = JSON.parse(event.data);
      if (parsedData.type === 'pong') {
        return; // Don't process pong messages
      }
      
      setLastMessage(message);
      onMessage?.(message);
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  }, [onMessage]);

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      shouldReconnectRef.current = true;
      setConnectionState('connecting');
      
      websocketRef.current = new WebSocket(url);
      
      websocketRef.current.addEventListener('open', handleOpen);
      websocketRef.current.addEventListener('close', handleClose);
      websocketRef.current.addEventListener('error', handleError);
      websocketRef.current.addEventListener('message', handleMessage);
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionState('error');
    }
  }, [url, handleOpen, handleClose, handleError, handleMessage]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearTimeouts();
    
    if (websocketRef.current) {
      websocketRef.current.removeEventListener('open', handleOpen);
      websocketRef.current.removeEventListener('close', handleClose);
      websocketRef.current.removeEventListener('error', handleError);
      websocketRef.current.removeEventListener('message', handleMessage);
      
      if (websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.close(1000, 'Manual disconnect');
      }
      
      websocketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionState('disconnected');
  }, [handleOpen, handleClose, handleError, handleMessage, clearTimeouts]);

  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(connect, 100);
  }, [disconnect, connect]);

  const sendMessage = useCallback((message: string) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(message);
    } else {
      // Queue message for when connection is restored
      messageQueueRef.current.push(message);
      console.warn('WebSocket not connected, message queued');
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Handle visibility change to manage connection
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, reduce heartbeat frequency or pause
        clearTimeouts();
      } else {
        // Page is visible, resume normal operation
        if (isConnected) {
          startHeartbeat();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isConnected, startHeartbeat, clearTimeouts]);

  // Handle network status changes
  useEffect(() => {
    const handleOnline = () => {
      console.log('Network back online, attempting to reconnect...');
      if (!isConnected && shouldReconnectRef.current) {
        reconnect();
      }
    };

    const handleOffline = () => {
      console.log('Network offline');
      setConnectionState('offline');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [isConnected, reconnect]);

  return {
    isConnected,
    connectionState,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnect
  };
};