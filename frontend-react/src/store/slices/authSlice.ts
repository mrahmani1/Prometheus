/**
 * Authentication slice for user management and security
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { UserProfile } from '../../types/api';

interface AuthState {
  isAuthenticated: boolean;
  token: string | null;
  refreshToken: string | null;
  user: UserProfile | null;
  loading: boolean;
  error: string | null;
  sessionExpiry: number | null;
  mfaRequired: boolean;
  lastActivity: number;
}

const initialState: AuthState = {
  isAuthenticated: false,
  token: null,
  refreshToken: null,
  user: null,
  loading: false,
  error: null,
  sessionExpiry: null,
  mfaRequired: false,
  lastActivity: Date.now(),
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart: state => {
      state.loading = true;
      state.error = null;
    },
    loginSuccess: (state, action: PayloadAction<{
      token: string;
      refreshToken: string;
      user: UserProfile;
      expiresIn: number;
    }>) => {
      state.isAuthenticated = true;
      state.token = action.payload.token;
      state.refreshToken = action.payload.refreshToken;
      state.user = action.payload.user;
      state.sessionExpiry = Date.now() + action.payload.expiresIn * 1000;
      state.loading = false;
      state.error = null;
      state.mfaRequired = false;
      state.lastActivity = Date.now();
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
      state.isAuthenticated = false;
      state.token = null;
      state.refreshToken = null;
      state.user = null;
    },
    mfaRequired: state => {
      state.mfaRequired = true;
      state.loading = false;
    },
    logout: state => {
      state.isAuthenticated = false;
      state.token = null;
      state.refreshToken = null;
      state.user = null;
      state.sessionExpiry = null;
      state.mfaRequired = false;
      state.error = null;
      state.loading = false;
    },
    refreshTokenSuccess: (state, action: PayloadAction<{
      token: string;
      refreshToken: string;
      expiresIn: number;
    }>) => {
      state.token = action.payload.token;
      state.refreshToken = action.payload.refreshToken;
      state.sessionExpiry = Date.now() + action.payload.expiresIn * 1000;
      state.lastActivity = Date.now();
    },
    updateLastActivity: state => {
      state.lastActivity = Date.now();
    },
    updateUserProfile: (state, action: PayloadAction<Partial<UserProfile>>) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
      }
    },
    clearError: state => {
      state.error = null;
    },
  },
});

export const {
  loginStart,
  loginSuccess,
  loginFailure,
  mfaRequired,
  logout,
  refreshTokenSuccess,
  updateLastActivity,
  updateUserProfile,
  clearError,
} = authSlice.actions;

export default authSlice.reducer;