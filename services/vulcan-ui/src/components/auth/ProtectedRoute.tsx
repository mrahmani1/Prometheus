import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAppSelector, useAppDispatch } from '@/store';
import { getCurrentUser } from '@/store/slices/authSlice';
import { LoadingOverlay } from '../common/LoadingOverlay';
import { Permission } from '@/types/auth';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: string[];
  fallbackPath?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermissions = [],
  fallbackPath = '/login',
}) => {
  const dispatch = useAppDispatch();
  const location = useLocation();
  const { isAuthenticated, user, token, isLoading } = useAppSelector((state) => state.auth);

  // Try to get user data if we have a token but no user
  useEffect(() => {
    if (token && !user && !isLoading) {
      dispatch(getCurrentUser());
    }
  }, [dispatch, token, user, isLoading]);

  // Show loading while checking authentication
  if (isLoading || (token && !user)) {
    return <LoadingOverlay message="Verifying authentication..." />;
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated || !user) {
    return (
      <Navigate
        to={fallbackPath}
        state={{ from: location }}
        replace
      />
    );
  }

  // Check permissions if required
  if (requiredPermissions.length > 0) {
    const hasRequiredPermissions = requiredPermissions.every(requiredPermission => {
      return user.permissions.some((permission: Permission) => {
        // Check if user has the required resource permission
        const [resource, action] = requiredPermission.split(':');
        return permission.resource === resource && 
               (permission.actions.includes(action as any) || permission.actions.includes('*' as any));
      });
    });

    if (!hasRequiredPermissions) {
      return (
        <Navigate
          to="/unauthorized"
          state={{ from: location, requiredPermissions }}
          replace
        />
      );
    }
  }

  return <>{children}</>;
};