import React, { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAppSelector } from '@/store';
import { LoginForm } from '@/components/auth/LoginForm';

export const LoginPage: React.FC = () => {
  const { isAuthenticated } = useAppSelector((state) => state.auth);
  const [_showRegister, setShowRegister] = useState(false);
  const [_showForgotPassword, setShowForgotPassword] = useState(false);

  // Redirect if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const handleSwitchToRegister = () => {
    setShowRegister(true);
  };

  const handleForgotPassword = () => {
    setShowForgotPassword(true);
  };

  // For now, just show the login form
  // In a complete implementation, you would have separate components for register and forgot password
  return (
    <LoginForm
      onSwitchToRegister={handleSwitchToRegister}
      onForgotPassword={handleForgotPassword}
    />
  );
};