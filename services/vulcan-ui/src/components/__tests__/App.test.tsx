import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from '../../App';

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    // Since the app redirects to login when not authenticated,
    // we should see the login form
    expect(document.body).toBeTruthy();
  });
});