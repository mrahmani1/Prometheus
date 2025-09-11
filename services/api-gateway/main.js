const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const consul = require('consul');
const redis = require('redis');
const winston = require('winston');
const jwt = require('jsonwebtoken');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const CONSUL_HOST = process.env.CONSUL_HOST || 'localhost';
const CONSUL_PORT = process.env.CONSUL_PORT || 8500;
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const JWT_SECRET = process.env.JWT_SECRET || 'prometheus-secret-key';

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Initialize Consul client
const consulClient = consul({
  host: CONSUL_HOST,
  port: CONSUL_PORT,
  promisify: true
});

// Initialize Redis client
const redisClient = redis.createClient({ url: REDIS_URL });
redisClient.connect().catch(console.error);

// Service registry to track available services
const serviceRegistry = new Map();

// Middleware setup
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Request correlation ID middleware
app.use((req, res, next) => {
  req.correlationId = req.headers['x-correlation-id'] || uuidv4();
  res.setHeader('x-correlation-id', req.correlationId);
  logger.info('Request received', {
    correlationId: req.correlationId,
    method: req.method,
    url: req.url,
    userAgent: req.get('User-Agent')
  });
  next();
});

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false
});
app.use('/api/', limiter);

// JWT Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      logger.warn('Invalid token attempt', {
        correlationId: req.correlationId,
        error: err.message
      });
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Role-based access control middleware
const requireRole = (roles) => {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }
    next();
  };
};

// Service discovery functions
async function discoverServices() {
  try {
    const services = await consulClient.catalog.service.list();
    const prometheusServices = {};
    
    for (const [serviceName] of Object.entries(services)) {
      if (serviceName.startsWith('prometheus-')) {
        const serviceInstances = await consulClient.catalog.service.nodes(serviceName);
        if (serviceInstances.length > 0) {
          const healthyInstances = serviceInstances.filter(instance => 
            instance.Checks && instance.Checks.every(check => check.Status === 'passing')
          );
          
          if (healthyInstances.length > 0) {
            prometheusServices[serviceName] = healthyInstances.map(instance => ({
              host: instance.ServiceAddress || instance.Address,
              port: instance.ServicePort,
              id: instance.ServiceID
            }));
          }
        }
      }
    }
    
    // Update service registry
    serviceRegistry.clear();
    Object.entries(prometheusServices).forEach(([name, instances]) => {
      serviceRegistry.set(name, instances);
    });
    
    logger.info('Service discovery updated', {
      services: Object.keys(prometheusServices),
      totalInstances: Object.values(prometheusServices).reduce((sum, instances) => sum + instances.length, 0)
    });
  } catch (error) {
    logger.error('Service discovery failed', { error: error.message });
  }
}

// Load balancer - simple round-robin
const serviceCounters = new Map();

function getServiceInstance(serviceName) {
  const instances = serviceRegistry.get(serviceName);
  if (!instances || instances.length === 0) {
    return null;
  }
  
  const counter = serviceCounters.get(serviceName) || 0;
  const instance = instances[counter % instances.length];
  serviceCounters.set(serviceName, counter + 1);
  
  return instance;
}

// Dynamic proxy middleware
const createServiceProxy = (serviceName) => {
  return createProxyMiddleware({
    target: 'http://placeholder', // Will be dynamically set
    changeOrigin: true,
    pathRewrite: {
      [`^/api/v1/${serviceName}`]: ''
    },
    router: (req) => {
      const instance = getServiceInstance(`prometheus-${serviceName}`);
      if (!instance) {
        logger.error('No healthy service instance found', { 
          service: serviceName,
          correlationId: req.correlationId 
        });
        return null;
      }
      return `http://${instance.host}:${instance.port}`;
    },
    onError: (err, req, res) => {
      logger.error('Proxy error', {
        service: serviceName,
        correlationId: req.correlationId,
        error: err.message
      });
      res.status(502).json({ error: 'Service temporarily unavailable' });
    },
    onProxyReq: (proxyReq, req) => {
      proxyReq.setHeader('x-correlation-id', req.correlationId);
      if (req.user) {
        proxyReq.setHeader('x-user-id', req.user.id);
        proxyReq.setHeader('x-user-role', req.user.role);
      }
    }
  });
};

// API Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: Array.from(serviceRegistry.keys())
  });
});

app.get('/ready', async (req, res) => {
  try {
    await redisClient.ping();
    await consulClient.agent.self();
    res.json({
      status: 'ready',
      timestamp: new Date().toISOString(),
      dependencies: {
        redis: 'connected',
        consul: 'connected'
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'not ready',
      error: error.message
    });
  }
});

// Authentication endpoint
app.post('/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    // TODO: Implement proper authentication against user database
    // For now, using simple demo credentials
    if (username === 'admin' && password === 'prometheus123') {
      const token = jwt.sign(
        { id: 1, username: 'admin', role: 'admin' },
        JWT_SECRET,
        { expiresIn: '24h' }
      );
      
      res.json({
        token,
        user: { id: 1, username: 'admin', role: 'admin' }
      });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (error) {
    logger.error('Authentication error', {
      correlationId: req.correlationId,
      error: error.message
    });
    res.status(500).json({ error: 'Authentication failed' });
  }
});

// Service status endpoint
app.get('/api/services', authenticateToken, (req, res) => {
  const services = Array.from(serviceRegistry.entries()).map(([name, instances]) => ({
    name,
    instances: instances.length,
    healthy: instances.length > 0
  }));
  
  res.json({ services });
});

// Protected API routes with dynamic routing
app.use('/api/v1/helios', authenticateToken, createServiceProxy('helios-core'));
app.use('/api/v1/data', authenticateToken, createServiceProxy('data-ingestion'));
app.use('/api/v1/volcano', authenticateToken, createServiceProxy('volcano-watchtower'));

// Admin-only routes
app.use('/api/v1/admin', authenticateToken, requireRole(['admin']), createServiceProxy('admin'));

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error', {
    correlationId: req.correlationId,
    error: error.message,
    stack: error.stack
  });
  
  res.status(500).json({
    error: 'Internal server error',
    correlationId: req.correlationId
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found',
    correlationId: req.correlationId
  });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  
  try {
    await redisClient.quit();
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown', { error: error.message });
    process.exit(1);
  }
});

// Start server
app.listen(PORT, async () => {
  logger.info(`API Gateway started on port ${PORT}`);
  
  // Initial service discovery
  await discoverServices();
  
  // Periodic service discovery refresh
  setInterval(discoverServices, 30000); // Every 30 seconds
});

module.exports = app;