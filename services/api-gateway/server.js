const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { createProxyMiddleware } = require('http-proxy-middleware');
const consul = require('consul');
const redis = require('redis');
const winston = require('winston');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 8080;

// Logger configuration
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Initialize Consul client for service discovery
const consulClient = consul({
  host: process.env.CONSUL_HOST || 'localhost',
  port: process.env.CONSUL_PORT || 8500,
  promisify: true
});

// Initialize Redis client for caching and rate limiting
const redisClient = redis.createClient({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379
});

redisClient.on('error', (err) => {
  logger.error('Redis Client Error', err);
});

redisClient.connect();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
});
app.use(limiter);

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Service registry for dynamic routing
const serviceRegistry = new Map();

// Function to discover services from Consul
async function discoverServices() {
  try {
    const services = await consulClient.catalog.service.list();
    
    for (const [serviceName] of Object.entries(services)) {
      if (serviceName.startsWith('prometheus-')) {
        const serviceInstances = await consulClient.catalog.service.nodes(serviceName);
        if (serviceInstances.length > 0) {
          const instance = serviceInstances[0];
          serviceRegistry.set(serviceName, {
            host: instance.ServiceAddress || instance.Address,
            port: instance.ServicePort,
            health: 'healthy'
          });
          logger.info(`Discovered service: ${serviceName} at ${instance.ServiceAddress}:${instance.ServicePort}`);
        }
      }
    }
  } catch (error) {
    logger.error('Error discovering services:', error);
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: Object.fromEntries(serviceRegistry)
  });
});

// Ready check endpoint
app.get('/ready', (req, res) => {
  const isReady = serviceRegistry.size > 0;
  res.status(isReady ? 200 : 503).json({
    status: isReady ? 'ready' : 'not ready',
    timestamp: new Date().toISOString(),
    discoveredServices: serviceRegistry.size
  });
});

// Dynamic proxy middleware
const createDynamicProxy = (serviceName) => {
  return createProxyMiddleware({
    target: `http://localhost:3000`, // Default fallback
    changeOrigin: true,
    pathRewrite: {
      [`^/api/${serviceName}`]: ''
    },
    router: (req) => {
      const service = serviceRegistry.get(`prometheus-${serviceName}`);
      if (service) {
        return `http://${service.host}:${service.port}`;
      }
      logger.warn(`Service not found: prometheus-${serviceName}`);
      return null;
    },
    onError: (err, req, res) => {
      logger.error(`Proxy error for ${serviceName}:`, err);
      res.status(503).json({
        error: 'Service temporarily unavailable',
        service: serviceName,
        timestamp: new Date().toISOString()
      });
    },
    onProxyReq: (proxyReq, req, res) => {
      logger.info(`Proxying ${req.method} ${req.url} to ${serviceName}`);
    }
  });
};

// API routes with service discovery
app.use('/api/helios', createDynamicProxy('helios-core'));
app.use('/api/data-ingestion', createDynamicProxy('data-ingestion'));
app.use('/api/simulation', createDynamicProxy('simulation'));
app.use('/api/realtime', createDynamicProxy('realtime-processor'));
app.use('/api/volcano-watchtower', createDynamicProxy('volcano-watchtower'));
app.use('/api/workflow', createDynamicProxy('workflow-engine'));
app.use('/api/nlp', createDynamicProxy('nlp-service'));
app.use('/api/visualization', createDynamicProxy('visualization'));

// API documentation endpoint
app.get('/api/docs', (req, res) => {
  res.json({
    title: 'Prometheus AI Geodynamic Platform API',
    version: '1.0.0',
    description: 'API Gateway for the Prometheus AI-powered geodynamic simulation platform',
    endpoints: {
      '/api/helios': 'Helios AI Model Core Engine',
      '/api/data-ingestion': 'Multi-source geological data ingestion',
      '/api/simulation': 'Long-term geological simulations',
      '/api/realtime': 'Real-time data processing',
      '/api/volcano-watchtower': 'Volcanic monitoring and alerting',
      '/api/workflow': 'Workflow orchestration engine',
      '/api/nlp': 'Natural language processing interface',
      '/api/visualization': '2D/3D/4D geological visualizations'
    },
    health: '/health',
    ready: '/ready'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
    requestId: req.id
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl,
    timestamp: new Date().toISOString()
  });
});

// Start server
async function startServer() {
  try {
    // Discover services on startup
    await discoverServices();
    
    // Set up periodic service discovery
    setInterval(discoverServices, 30000); // Every 30 seconds
    
    app.listen(PORT, () => {
      logger.info(`API Gateway started on port ${PORT}`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`Discovered ${serviceRegistry.size} services`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await redisClient.quit();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await redisClient.quit();
  process.exit(0);
});

startServer();