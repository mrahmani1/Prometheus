/**
 * Redux store configuration with RTK Query for the Prometheus AI Platform
 * Implements advanced state management with real-time capabilities
 */

import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
// import { persistStore, persistReducer } from 'redux-persist';
// import storage from 'redux-persist/lib/storage';

import { apiSlice } from './api/apiSlice';
import authReducer from './slices/authSlice';
import uiReducer from './slices/uiSlice';
import geologicalReducer from './slices/geologicalSlice';
import visualizationReducer from './slices/visualizationSlice';
import monitoringReducer from './slices/monitoringSlice';
import dashboardReducer from './slices/dashboardSlice';
import notificationReducer from './slices/notificationSlice';

// Persist configuration (temporarily disabled)
// const persistConfig = {
//   key: 'prometheus-ai',
//   storage,
//   whitelist: ['auth', 'ui', 'dashboard'], // Only persist these slices
//   blacklist: ['api', 'geological', 'monitoring'], // Don't persist real-time data
// };

// Root reducer
const rootReducer = {
  api: apiSlice.reducer,
  auth: authReducer,
  ui: uiReducer,
  geological: geologicalReducer,
  visualization: visualizationReducer,
  monitoring: monitoringReducer,
  dashboard: dashboardReducer,
  notifications: notificationReducer,
};

// Configure store with advanced middleware
export const store = configureStore({
  reducer: rootReducer,
  middleware: getDefaultMiddleware =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
        ignoredPaths: ['register'],
      },
      immutableCheck: {
        warnAfter: 128,
      },
    }).concat(apiSlice.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

// Setup RTK Query listeners for automatic refetching
setupListeners(store.dispatch);

// Create persistor (temporarily disabled to fix type issues)
// export const persistor = persistStore(store);

// Export types
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Export store instance
export default store;