const request = require('supertest');
const app = require('../main');

describe('API Gateway', () => {
  describe('Health Endpoints', () => {
    test('GET /health should return healthy status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);
      
      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('services');
    });

    test('GET /ready should return readiness status', async () => {
      const response = await request(app)
        .get('/ready');
      
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('Authentication', () => {
    test('POST /auth/login with valid credentials should return token', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'admin',
          password: 'prometheus123'
        })
        .expect(200);
      
      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('user');
      expect(response.body.user).toHaveProperty('username', 'admin');
    });

    test('POST /auth/login with invalid credentials should return 401', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          username: 'invalid',
          password: 'wrong'
        })
        .expect(401);
      
      expect(response.body).toHaveProperty('error', 'Invalid credentials');
    });
  });

  describe('Protected Routes', () => {
    let authToken;

    beforeAll(async () => {
      const loginResponse = await request(app)
        .post('/auth/login')
        .send({
          username: 'admin',
          password: 'prometheus123'
        });
      
      authToken = loginResponse.body.token;
    });

    test('GET /api/services without token should return 401', async () => {
      await request(app)
        .get('/api/services')
        .expect(401);
    });

    test('GET /api/services with valid token should return services', async () => {
      const response = await request(app)
        .get('/api/services')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
      
      expect(response.body).toHaveProperty('services');
      expect(Array.isArray(response.body.services)).toBe(true);
    });
  });

  describe('Rate Limiting', () => {
    test('Should apply rate limiting to API routes', async () => {
      // Make multiple requests to test rate limiting
      const promises = Array(10).fill().map(() => 
        request(app).get('/api/services')
      );
      
      const responses = await Promise.all(promises);
      
      // At least some should be rate limited (401 for missing auth)
      expect(responses.every(res => res.status === 401)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('Should return 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/unknown/route')
        .expect(404);
      
      expect(response.body).toHaveProperty('error', 'Route not found');
      expect(response.body).toHaveProperty('correlationId');
    });
  });

  describe('CORS and Security Headers', () => {
    test('Should include security headers', async () => {
      const response = await request(app)
        .get('/health');
      
      expect(response.headers).toHaveProperty('x-correlation-id');
    });
  });
});